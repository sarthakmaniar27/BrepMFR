"""Scan an MFCAD++ (or CADSynth) bin dataset and report:

- per-file tensor shapes for every ndata/edata/extras key
- global unique values + ranges for every integer field the encoder consumes
- face_label histogram across the dataset
- explicit OUT-OF-RANGE flags vs the model's encoder embeddings
- consistency checks: edges_path indices vs num_edges, spatial_pos NxN, etc.

Encoder expectations (from models/modules/layers/brep_encoder_layer.py):
    face_type_encoder  : nn.Embedding(8,   padding_idx=0)  -> values in [0, 7]
    face_loop_encoder  : nn.Embedding(256, padding_idx=0)  -> values in [0, 255]
    edge_type_encoder  : nn.Embedding(6,   padding_idx=0)  -> values in [0, 5]
    edge_conv_encoder  : nn.Embedding(3,   padding_idx=0)  -> values in [0, 2]
    degree_encoder     : nn.Embedding(128, padding_idx=0)  -> degree+1 in [0, 127]
                                                              -> raw degree in [0, 126]
    classifier head    : num_classes=25                     -> face_label in [0, 24]

Usage:
    python scripts\\scan_mfcadpp.py --root Z:\\Experiment6\\target_dataset\\output\\bin --tag mfcadpp --workers 16
    python scripts\\scan_mfcadpp.py --root Z:\\Experiment6\\source_dataset\\output\\bin --tag cadsynth --workers 16
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from dgl.data.utils import load_graphs


ENCODER_EXPECT = {
    "face_type":  ("ndata['z']", 0, 7),    # Embedding(8)
    "face_loop":  ("ndata['l']", 0, 255),  # Embedding(256)
    "edge_type":  ("edata['t']", 0, 5),    # Embedding(6)
    "edge_conv":  ("edata['c']", 0, 2),    # Embedding(3)
    "face_label": ("ndata['f']", 0, 24),   # CrossEntropy num_classes=25
    "node_degree": ("derived (in_degrees)", 0, 126),  # Embedding(128) after +1
}

# ------------------------- per-file scan -------------------------

def _safe_minmax(t: torch.Tensor) -> Tuple[float, float]:
    if t is None or t.numel() == 0:
        return (math.nan, math.nan)
    return (float(t.min().item()), float(t.max().item()))


def _uniques_int(t: torch.Tensor, cap: int = 1024) -> List[int]:
    if t is None or t.numel() == 0:
        return []
    if t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        return []
    u = torch.unique(t).tolist()
    return u[:cap]


def scan_one(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": path}
    try:
        glist, ext = load_graphs(path)
        g = glist[0]
        n_nodes = int(g.num_nodes())
        n_edges = int(g.num_edges())
        out["num_nodes"] = n_nodes
        out["num_edges"] = n_edges

        # ---------- ndata ----------
        nd = g.ndata
        out["ndata_keys"] = sorted(nd.keys())
        for k, v in nd.items():
            out[f"shape:ndata.{k}"] = tuple(v.shape)
            out[f"dtype:ndata.{k}"] = str(v.dtype)

        ed = g.edata
        out["edata_keys"] = sorted(ed.keys())
        for k, v in ed.items():
            out[f"shape:edata.{k}"] = tuple(v.shape)
            out[f"dtype:edata.{k}"] = str(v.dtype)

        # ---------- integer fields the encoder/loss consume ----------
        # face_type
        if "z" in nd:
            mn, mx = _safe_minmax(nd["z"])
            out["face_type_min"], out["face_type_max"] = mn, mx
            out["face_type_uniques"] = _uniques_int(nd["z"])
        # face_loop
        if "l" in nd:
            mn, mx = _safe_minmax(nd["l"])
            out["face_loop_min"], out["face_loop_max"] = mn, mx
            out["face_loop_uniques"] = _uniques_int(nd["l"])
        # face_adj (used internally only for stats here; not consumed by model)
        if "a" in nd:
            mn, mx = _safe_minmax(nd["a"])
            out["face_adj_min"], out["face_adj_max"] = mn, mx
        # face_label
        if "f" in nd:
            mn, mx = _safe_minmax(nd["f"])
            out["face_label_min"], out["face_label_max"] = mn, mx
            out["face_label_uniques"] = _uniques_int(nd["f"])
            # per-class counts contributed by this file
            try:
                vals, cnts = torch.unique(nd["f"], return_counts=True)
                out["face_label_hist"] = dict(zip(vals.tolist(), cnts.tolist()))
            except Exception:
                out["face_label_hist"] = {}
        # face_area
        if "y" in nd:
            mn, mx = _safe_minmax(nd["y"])
            out["face_area_min"], out["face_area_max"] = mn, mx
            out["face_area_has_nan"] = bool(torch.isnan(nd["y"]).any().item()) if nd["y"].is_floating_point() else False
            out["face_area_has_neg"] = bool((nd["y"] < 0).any().item())

        # edge_type
        if "t" in ed:
            mn, mx = _safe_minmax(ed["t"])
            out["edge_type_min"], out["edge_type_max"] = mn, mx
            out["edge_type_uniques"] = _uniques_int(ed["t"])
        # edge_conv
        if "c" in ed:
            mn, mx = _safe_minmax(ed["c"])
            out["edge_conv_min"], out["edge_conv_max"] = mn, mx
            out["edge_conv_uniques"] = _uniques_int(ed["c"])
        # edge_len
        if "l" in ed:
            mn, mx = _safe_minmax(ed["l"])
            out["edge_len_min"], out["edge_len_max"] = mn, mx
            out["edge_len_has_nan"] = bool(torch.isnan(ed["l"]).any().item()) if ed["l"].is_floating_point() else False
            out["edge_len_has_neg"] = bool((ed["l"] < 0).any().item())
        # edge_ang (dihedral)
        if "a" in ed:
            mn, mx = _safe_minmax(ed["a"])
            out["edge_ang_min"], out["edge_ang_max"] = mn, mx
            out["edge_ang_outside_pi"] = bool(((ed["a"] < -math.pi - 1e-3) | (ed["a"] > math.pi + 1e-3)).any().item())

        # ---------- UV grids ----------
        if "x" in nd:
            x = nd["x"]
            out["face_uv_shape"] = tuple(x.shape)  # [num_nodes, U, V, 7]
            out["face_uv_has_nan"] = bool(torch.isnan(x).any().item()) if x.is_floating_point() else False
        if "x" in ed:
            xe = ed["x"]
            out["edge_uv_shape"] = tuple(xe.shape)  # [num_edges, U, ?]
            out["edge_uv_has_nan"] = bool(torch.isnan(xe).any().item()) if xe.is_floating_point() else False

        # ---------- derived: in-degree (degree_encoder input) ----------
        try:
            deg = g.in_degrees()
            mn, mx = _safe_minmax(deg)
            out["node_degree_min"], out["node_degree_max"] = mn, mx
        except Exception:
            pass

        # ---------- proximity extras (graphfile[1]) ----------
        out["extras_keys"] = sorted(ext.keys())
        for k, v in ext.items():
            if torch.is_tensor(v):
                out[f"shape:extras.{k}"] = tuple(v.shape)
                out[f"dtype:extras.{k}"] = str(v.dtype)
        # spatial_pos should be [N, N]
        if "spatial_pos" in ext:
            sp = ext["spatial_pos"]
            out["spatial_pos_square_eq_N"] = bool(sp.shape[0] == n_nodes and sp.shape[1] == n_nodes)
            mn, mx = _safe_minmax(sp)
            out["spatial_pos_min"], out["spatial_pos_max"] = mn, mx
        if "edges_path" in ext:
            ep = ext["edges_path"]
            out["edges_path_shape"] = tuple(ep.shape)
            # values should be in [-1, n_edges-1]; -1 is the "no edge" sentinel
            mn, mx = _safe_minmax(ep)
            out["edges_path_min"], out["edges_path_max"] = mn, mx
            out["edges_path_oor"] = bool(((ep < -1) | (ep >= n_edges)).any().item())
        if "d2_distance" in ext:
            d2 = ext["d2_distance"]
            out["d2_distance_shape"] = tuple(d2.shape)
        if "angle_distance" in ext:
            ad = ext["angle_distance"]
            out["angle_distance_shape"] = tuple(ad.shape)

        # ---------- explicit out-of-range flags vs encoder ----------
        def _oor(field, lo, hi, mn_key, mx_key):
            mn = out.get(mn_key); mx = out.get(mx_key)
            if mn is None or mx is None or (isinstance(mn, float) and math.isnan(mn)):
                return None
            return bool(mn < lo or mx > hi)
        out["oor_face_type"]  = _oor("face_type",  0, 7,   "face_type_min",  "face_type_max")
        out["oor_face_loop"]  = _oor("face_loop",  0, 255, "face_loop_min",  "face_loop_max")
        out["oor_edge_type"]  = _oor("edge_type",  0, 5,   "edge_type_min",  "edge_type_max")
        out["oor_edge_conv"]  = _oor("edge_conv",  0, 2,   "edge_conv_min",  "edge_conv_max")
        out["oor_face_label"] = _oor("face_label", 0, 24,  "face_label_min", "face_label_max")
        out["oor_degree"]     = _oor("node_degree",0, 126, "node_degree_min","node_degree_max")
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = repr(e)
        out["tb"] = traceback.format_exc(limit=2)
    return out


# ------------------------- aggregation -------------------------

class Aggregator:
    def __init__(self):
        self.n_files = 0
        self.n_ok = 0
        self.n_err = 0
        self.errors: List[Tuple[str, str]] = []
        # global uniques per integer field
        self.uniques = {k: set() for k in
                        ("face_type", "face_loop", "edge_type", "edge_conv", "face_label")}
        # global mins/maxes
        self.gmin: Dict[str, float] = {}
        self.gmax: Dict[str, float] = {}
        self.gn: Dict[str, int] = {}
        # face_label histogram across dataset
        self.label_hist: Counter = Counter()
        # tensor-shape signatures (e.g. (32,32,7) for face UV grid) -> count
        self.face_uv_shapes: Counter = Counter()
        self.edge_uv_shapes: Counter = Counter()
        # OOR file lists per field
        self.oor: Dict[str, List[str]] = defaultdict(list)
        self.oor_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # ndata / edata / extras key sets observed
        self.key_signatures: Counter = Counter()
        # spatial_pos shape mismatches
        self.spatial_pos_bad: List[str] = []
        self.edges_path_bad: List[str] = []
        # NaN / negative occurrences
        self.nan_files: List[str] = []
        self.neg_area_files: List[str] = []
        self.dihedral_oor_files: List[str] = []
        # global degree max
        self.deg_max = 0
        self.deg_max_file = ""

    def update_minmax(self, key: str, mn, mx, n=None):
        if mn is None or (isinstance(mn, float) and math.isnan(mn)):
            return
        if key not in self.gmin or mn < self.gmin[key]:
            self.gmin[key] = mn
        if key not in self.gmax or mx > self.gmax[key]:
            self.gmax[key] = mx
        if n is not None:
            self.gn[key] = self.gn.get(key, 0) + n

    def add(self, r: Dict[str, Any]):
        self.n_files += 1
        if not r.get("ok"):
            self.n_err += 1
            self.errors.append((r.get("path", "?"), r.get("error", "?")))
            return
        self.n_ok += 1
        path = r["path"]
        # uniques
        for fld in ("face_type", "face_loop", "edge_type", "edge_conv", "face_label"):
            for v in r.get(f"{fld}_uniques", []) or []:
                self.uniques[fld].add(int(v))
        # mins/maxes
        for fld in ("face_type", "face_loop", "edge_type", "edge_conv", "face_label",
                    "face_adj", "face_area", "edge_len", "edge_ang", "node_degree"):
            mn = r.get(f"{fld}_min"); mx = r.get(f"{fld}_max")
            self.update_minmax(fld, mn, mx, n=r.get("num_nodes" if "face" in fld else "num_edges"))
        # face label hist
        h = r.get("face_label_hist") or {}
        for k, v in h.items():
            self.label_hist[int(k)] += int(v)
        # UV shapes
        if "face_uv_shape" in r:
            self.face_uv_shapes[tuple(r["face_uv_shape"])] += 1
        if "edge_uv_shape" in r:
            self.edge_uv_shapes[tuple(r["edge_uv_shape"])] += 1
        # key signature
        sig = (
            "ndata=" + ",".join(r.get("ndata_keys") or []) +
            "|edata=" + ",".join(r.get("edata_keys") or []) +
            "|extras=" + ",".join(r.get("extras_keys") or [])
        )
        self.key_signatures[sig] += 1
        # OOR flags
        for fld in ("face_type", "face_loop", "edge_type", "edge_conv", "face_label", "degree"):
            if r.get(f"oor_{fld}"):
                self.oor[fld].append(path)
                if len(self.oor_examples[fld]) < 5:
                    ex = {"path": path,
                          "min": r.get(f"{fld}_min" if fld != "degree" else "node_degree_min"),
                          "max": r.get(f"{fld}_max" if fld != "degree" else "node_degree_max"),
                          "uniques": r.get(f"{fld}_uniques") if fld != "degree" else None}
                    self.oor_examples[fld].append(ex)
        # extras integrity
        if r.get("spatial_pos_square_eq_N") is False:
            self.spatial_pos_bad.append(path)
        if r.get("edges_path_oor"):
            self.edges_path_bad.append(path)
        # NaNs / negatives
        if r.get("face_uv_has_nan") or r.get("edge_uv_has_nan") or r.get("face_area_has_nan") or r.get("edge_len_has_nan"):
            self.nan_files.append(path)
        if r.get("face_area_has_neg") or r.get("edge_len_has_neg"):
            self.neg_area_files.append(path)
        if r.get("edge_ang_outside_pi"):
            self.dihedral_oor_files.append(path)
        # degree
        nd_max = r.get("node_degree_max")
        if nd_max is not None and not (isinstance(nd_max, float) and math.isnan(nd_max)) and nd_max > self.deg_max:
            self.deg_max = nd_max
            self.deg_max_file = path

    def report(self, tag: str, out_dir: Path):
        lines: List[str] = []
        p = lines.append
        p("=" * 88)
        p(f"DATASET SCAN REPORT  -  tag={tag}")
        p("=" * 88)
        p(f"files scanned       : {self.n_files}")
        p(f"  ok                : {self.n_ok}")
        p(f"  errors            : {self.n_err}")
        if self.errors:
            p("  first 5 errors    :")
            for path, err in self.errors[:5]:
                p(f"    - {Path(path).name}: {err}")

        p("")
        p("-" * 88)
        p("KEY SIGNATURES (top 5; each is ndata|edata|extras keys present)")
        p("-" * 88)
        for sig, cnt in self.key_signatures.most_common(5):
            p(f"  {cnt:>7}  {sig}")

        p("")
        p("-" * 88)
        p("TENSOR SHAPE SIGNATURES")
        p("-" * 88)
        p("  face UV grid (ndata['x']):")
        for shp, cnt in self.face_uv_shapes.most_common(5):
            p(f"    {cnt:>7}  {shp}")
        p("  edge UV grid (edata['x']):")
        for shp, cnt in self.edge_uv_shapes.most_common(5):
            p(f"    {cnt:>7}  {shp}")

        p("")
        p("-" * 88)
        p("INTEGER FIELDS  vs  ENCODER EXPECTATIONS")
        p("-" * 88)
        p(f"{'field':<14}{'expect':<14}{'observed_min':>14}{'observed_max':>14}  uniques")
        for fld, (src, lo, hi) in ENCODER_EXPECT.items():
            mn = self.gmin.get(fld, math.nan)
            mx = self.gmax.get(fld, math.nan)
            uniq_str = ""
            if fld in self.uniques:
                uvals = sorted(self.uniques[fld])
                uniq_str = "{" + ",".join(str(v) for v in uvals) + "}"
            ok = "OK"
            if not (isinstance(mn, float) and math.isnan(mn)):
                if mn < lo or mx > hi:
                    ok = "**OOR**"
            p(f"{fld:<14}[{lo:>3},{hi:>4}]  {mn:>14}  {mx:>14}  {uniq_str}   {ok}    src={src}")

        p("")
        p("-" * 88)
        p("FACE LABEL HISTOGRAM (cumulative across dataset)")
        p("-" * 88)
        total = sum(self.label_hist.values())
        p(f"  total labeled faces : {total}")
        p(f"  {'label':>5}  {'count':>12}  {'%':>7}")
        for k in sorted(self.label_hist.keys()):
            c = self.label_hist[k]
            pct = 100.0 * c / total if total else 0.0
            p(f"  {k:>5}  {c:>12}  {pct:>6.2f}%")

        p("")
        p("-" * 88)
        p("FLOAT FIELD RANGES")
        p("-" * 88)
        for fld in ("face_area", "edge_len", "edge_ang"):
            mn = self.gmin.get(fld, math.nan); mx = self.gmax.get(fld, math.nan)
            p(f"  {fld:<10}: min={mn}  max={mx}")
        p(f"  files with NaN UV/area/len  : {len(self.nan_files)}")
        p(f"  files with negative area/len: {len(self.neg_area_files)}")
        p(f"  files with edge_ang outside [-pi, pi]: {len(self.dihedral_oor_files)}")

        p("")
        p("-" * 88)
        p("OUT-OF-RANGE COUNTS  (files where any value violates encoder embedding range)")
        p("-" * 88)
        for fld in ("face_type", "face_loop", "edge_type", "edge_conv", "face_label", "degree"):
            files = self.oor.get(fld, [])
            p(f"  {fld:<12}: {len(files):>6} files")
            for ex in self.oor_examples.get(fld, []):
                p(f"     ex {Path(ex['path']).name}: min={ex['min']} max={ex['max']} uniques={ex.get('uniques')}")

        p("")
        p("-" * 88)
        p("PROXIMITY EXTRAS INTEGRITY")
        p("-" * 88)
        p(f"  spatial_pos shape != [N,N] : {len(self.spatial_pos_bad)} files")
        p(f"  edges_path index out of range: {len(self.edges_path_bad)} files")
        p(f"  max in_degree observed     : {self.deg_max}  (file {Path(self.deg_max_file).name if self.deg_max_file else '?'})")

        report_text = "\n".join(lines) + "\n"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"scan_{tag}_summary.txt"
        report_path.write_text(report_text, encoding="utf-8")
        print(report_text)
        print(f"\n[wrote {report_path}]")

        # Also drop OOR file lists
        for fld, files in self.oor.items():
            if files:
                p_ = out_dir / f"scan_{tag}_oor_{fld}.txt"
                p_.write_text("\n".join(files), encoding="utf-8")
                print(f"[wrote {p_} with {len(files)} entries]")
        if self.nan_files:
            (out_dir / f"scan_{tag}_nan_files.txt").write_text("\n".join(self.nan_files), encoding="utf-8")
        if self.dihedral_oor_files:
            (out_dir / f"scan_{tag}_dihedral_oor_files.txt").write_text("\n".join(self.dihedral_oor_files), encoding="utf-8")


# ------------------------- driver -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder containing .bin files (recursively scanned)")
    ap.add_argument("--tag", required=True, help="Output tag, e.g. mfcadpp or cadsynth")
    ap.add_argument("--out_dir", default="scripts/scan_reports")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0, help="If >0, scan only the first N files (debug)")
    args = ap.parse_args()

    root = Path(args.root)
    print(f"[scan_mfcadpp] root={root} (enumerating)", flush=True)
    t_enum = time.time()
    if root.is_dir():
        # Flat directory listing -- much faster than rglob on network drives.
        try:
            entries = os.listdir(root)
        except OSError as e:
            print(f"listdir failed: {e}"); sys.exit(1)
        files = sorted(str(root / e) for e in entries if e.endswith(".bin"))
    else:
        print(f"root not a directory: {root}"); sys.exit(1)
    print(f"[scan_mfcadpp] enumerated {len(files)} files in {time.time()-t_enum:.1f}s", flush=True)
    if args.limit:
        files = files[:args.limit]
    print(f"[scan_mfcadpp] scanning {len(files)} files with workers={args.workers}", flush=True)
    if not files:
        print("no files found"); sys.exit(1)

    agg = Aggregator()
    t0 = time.time()
    progress_every = max(50, len(files) // 200)

    if args.workers <= 1:
        # Sequential for debugging / very low-resource situations.
        for i, f in enumerate(files, 1):
            agg.add(scan_one(f))
            if i % progress_every == 0 or i == len(files):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(files) - i) / rate if rate > 0 else 0
                print(f"  {i:>7}/{len(files)}  ({100.0*i/len(files):5.1f}%)  "
                      f"elapsed={elapsed:6.0f}s  eta={eta:6.0f}s  ok={agg.n_ok}  err={agg.n_err}",
                      flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            # Submit lazily in chunks of 4*workers so memory doesn't spike with
            # 60k+ pending futures, and so the first results stream in quickly.
            chunk_size = max(args.workers * 4, 64)
            done = 0
            it = iter(files)
            in_flight = {}
            # prime
            for _ in range(min(chunk_size, len(files))):
                try:
                    f = next(it)
                except StopIteration:
                    break
                in_flight[ex.submit(scan_one, f)] = f
            while in_flight:
                # wait for any one to finish
                for fut in as_completed(list(in_flight.keys()), timeout=None):
                    in_flight.pop(fut)
                    r = fut.result()
                    agg.add(r)
                    done += 1
                    # refill
                    try:
                        nf = next(it)
                        in_flight[ex.submit(scan_one, nf)] = nf
                    except StopIteration:
                        pass
                    if done % progress_every == 0 or done == len(files):
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (len(files) - done) / rate if rate > 0 else 0
                        print(f"  {done:>7}/{len(files)}  ({100.0*done/len(files):5.1f}%)  "
                              f"elapsed={elapsed:6.0f}s  eta={eta:6.0f}s  ok={agg.n_ok}  err={agg.n_err}",
                              flush=True)
                    break  # restart the as_completed loop with fresh in_flight set

    out_dir = Path(args.out_dir)
    agg.report(args.tag, out_dir)


if __name__ == "__main__":
    main()
