# scan_postmortem_to_csv.py
# Combined dataset scanner + postmortem exporter for BrepMFR DGL .bin graphs.
# Outputs 5 CSVs into OUT_DIR:
#   1) __dataset_summary.csv
#   2) __global_key_stats.csv
#   3) __per_file_summary.csv
#   4) __extremes_topk.csv
#   5) __tensor_shapes.csv   <-- NEW: per-key tensor shape schema / channels / dim ranges

import os
import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from dgl.data.utils import load_graphs
from tqdm import tqdm


DEFAULT_FILE_SUFFIX = ".bin"
DEFAULT_UNIQUE_CAP = 200_000
DEFAULT_TOPK = 50
DEFAULT_SHAPE_SAMPLES = 25  # cap of distinct shape strings stored per key


def walk_files(root: Path, suffix: str) -> List[Path]:
    out: List[Path] = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(suffix):
                out.append(Path(r) / fn)
    return sorted(out)


def is_int_dtype(dt: torch.dtype) -> bool:
    return dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def safe_min_max(t: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
        return None, None
    return float(t.min().item()), float(t.max().item())


def count_nonfinite(t: torch.Tensor) -> Tuple[int, int]:
    if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
        return 0, 0
    if not t.is_floating_point():
        return 0, 0
    nan = int(torch.isnan(t).sum().item())
    inf = int(torch.isinf(t).sum().item())
    return nan, inf


def approx_add_uniques(u: Set[int], t: torch.Tensor, cap: int) -> None:
    if len(u) >= cap:
        return
    if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
        return
    if not is_int_dtype(t.dtype):
        return
    flat = t.reshape(-1)
    room = cap - len(u)
    if room <= 0:
        return

    if flat.numel() > 5_000_000:
        steps = min(room, 50_000)
        idx = torch.linspace(0, flat.numel() - 1, steps=steps).long()
        vals = flat[idx].tolist()
    else:
        vals = flat.tolist()

    for v in vals[:room]:
        u.add(int(v))


def safe_shape_str(t: Any) -> str:
    if t is None or (not torch.is_tensor(t)):
        return ""
    return "(" + ",".join(str(x) for x in list(t.shape)) + ")"


def unravel_flat_index(flat_idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) == 0:
        return ()
    idxs = []
    rem = int(flat_idx)
    for dim in reversed(shape):
        idxs.append(rem % dim)
        rem //= dim
    return tuple(reversed(idxs))


@dataclass
class GlobalStat:
    dtype: Optional[str] = None
    min_v: Optional[float] = None
    max_v: Optional[float] = None
    count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    uniques: Set[int] = field(default_factory=set)

    def update(self, t: torch.Tensor, unique_cap: int) -> None:
        if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
            return
        if self.dtype is None:
            self.dtype = str(t.dtype)

        mn, mx = safe_min_max(t)
        if mn is not None:
            if self.min_v is None or mn < self.min_v:
                self.min_v = mn
            if self.max_v is None or mx > self.max_v:
                self.max_v = mx

        self.count += int(t.numel())
        nan, inf = count_nonfinite(t)
        self.nan_count += nan
        self.inf_count += inf

        approx_add_uniques(self.uniques, t, unique_cap)


@dataclass
class ShapeStat:
    dtype: Optional[str] = None
    # rank
    rank_min: Optional[int] = None
    rank_max: Optional[int] = None
    # per-dimension min/max (tracked for up to 6 dims; extend if needed)
    dim_min: List[Optional[int]] = field(default_factory=lambda: [None] * 6)
    dim_max: List[Optional[int]] = field(default_factory=lambda: [None] * 6)
    # inferred "channels" (last dim) min/max
    ch_min: Optional[int] = None
    ch_max: Optional[int] = None
    # some example shapes observed
    shape_samples: Set[str] = field(default_factory=set)
    # count of tensors observed (files where key existed)
    tensors_seen: int = 0

    def update(self, t: torch.Tensor, sample_cap: int) -> None:
        if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
            return

        self.tensors_seen += 1

        if self.dtype is None:
            self.dtype = str(t.dtype)

        shape = tuple(int(s) for s in t.shape)
        r = len(shape)

        if self.rank_min is None or r < self.rank_min:
            self.rank_min = r
        if self.rank_max is None or r > self.rank_max:
            self.rank_max = r

        # Track dims up to 6
        for i in range(min(r, 6)):
            d = shape[i]
            if self.dim_min[i] is None or d < self.dim_min[i]:
                self.dim_min[i] = d
            if self.dim_max[i] is None or d > self.dim_max[i]:
                self.dim_max[i] = d

        # "channels" = last dim (best-effort)
        if r >= 1:
            ch = shape[-1]
            if self.ch_min is None or ch < self.ch_min:
                self.ch_min = ch
            if self.ch_max is None or ch > self.ch_max:
                self.ch_max = ch

        # Store sample shape strings
        if len(self.shape_samples) < sample_cap:
            self.shape_samples.add(str(shape))


@dataclass
class ExtremeHit:
    dataset_tag: str
    file: str
    key: str
    abs_value: float
    value: float
    dtype: str
    shape: str
    idx0: Optional[int] = None
    idx1: Optional[int] = None
    idx2: Optional[int] = None
    idx3: Optional[int] = None
    note: str = ""


class TopK:
    def __init__(self, k: int):
        self.k = k
        self.items: List[ExtremeHit] = []

    def push(self, hit: ExtremeHit) -> None:
        self.items.append(hit)
        self.items.sort(key=lambda h: h.abs_value, reverse=True)
        if len(self.items) > self.k:
            self.items = self.items[: self.k]


def validate_edges_path_vs_num_edges(aux: Dict[str, Any], num_edges: int) -> Tuple[int, Optional[int]]:
    if not isinstance(aux, dict) or "edges_path" not in aux:
        return 0, None
    ep = aux["edges_path"]
    if not torch.is_tensor(ep) or ep.numel() == 0:
        return 0, None
    ep_valid = ep[ep >= 0]
    if ep_valid.numel() == 0:
        return 0, None
    mx = int(ep_valid.max().item())
    oob = int((ep_valid >= num_edges).sum().item())
    return oob, mx


def tensor_minmax_row(prefix: str, t: torch.Tensor) -> Dict[str, Any]:
    if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
        return {
            f"{prefix}_dtype": "",
            f"{prefix}_shape": "",
            f"{prefix}_min": "",
            f"{prefix}_max": "",
            f"{prefix}_nan": 0,
            f"{prefix}_inf": 0,
        }
    mn, mx = safe_min_max(t)
    nan, inf = count_nonfinite(t)
    return {
        f"{prefix}_dtype": str(t.dtype),
        f"{prefix}_shape": safe_shape_str(t),
        f"{prefix}_min": mn if mn is not None else "",
        f"{prefix}_max": mx if mx is not None else "",
        f"{prefix}_nan": nan,
        f"{prefix}_inf": inf,
    }


def scan_extremes_for_key(dataset_tag: str, fp: Path, key: str, t: torch.Tensor, topk: TopK) -> None:
    if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
        return
    tc = t.detach().cpu()
    if tc.is_floating_point():
        abs_tc = torch.abs(tc)
        abs_tc = torch.nan_to_num(abs_tc, nan=float("-inf"), posinf=float("inf"), neginf=float("inf"))
        flat = abs_tc.reshape(-1)
        flat_idx = int(flat.argmax().item())
        abs_val = float(flat[flat_idx].item())
        val = float(tc.reshape(-1)[flat_idx].item())
    else:
        abs_tc = torch.abs(tc.to(torch.int64))
        flat = abs_tc.reshape(-1)
        flat_idx = int(flat.argmax().item())
        abs_val = float(flat[flat_idx].item())
        val = float(int(tc.reshape(-1)[flat_idx].item()))

    idxs = unravel_flat_index(flat_idx, tuple(tc.shape))
    idxs4 = list(idxs[:4]) + [None] * (4 - len(idxs[:4]))

    topk.push(
        ExtremeHit(
            dataset_tag=dataset_tag,
            file=fp.name,
            key=key,
            abs_value=abs_val,
            value=val,
            dtype=str(tc.dtype),
            shape=safe_shape_str(tc),
            idx0=idxs4[0],
            idx1=idxs4[1],
            idx2=idxs4[2],
            idx3=idxs4[3],
            note="argmax(|tensor|)",
        )
    )


def scan_dataset(
    dataset_tag: str,
    root_dir: Path,
    out_dir: Path,
    suffix: str,
    unique_cap: int,
    topk_k: int,
    shape_sample_cap: int,
    quiet: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = walk_files(root_dir, suffix)
    t0 = time.time()

    global_stats: Dict[str, GlobalStat] = {}
    shape_stats: Dict[str, ShapeStat] = {}

    present_counts = {
        "aux:spatial_pos": 0,
        "aux:edges_path": 0,
        "aux:d2_distance": 0,
        "aux:angle_distance": 0,
        "ndata:x": 0,
        "edata:x": 0,
        "ndata:f": 0,
    }

    n_files = len(files)
    n_loaded = 0
    n_failed = 0
    edge_path_oob_files = 0
    edge_path_oob_total = 0

    topk = TopK(topk_k)
    per_file_rows: List[Dict[str, Any]] = []

    per_file_csv = out_dir / f"{dataset_tag}__per_file_summary.csv"
    global_csv = out_dir / f"{dataset_tag}__global_key_stats.csv"
    extremes_csv = out_dir / f"{dataset_tag}__extremes_topk.csv"
    dataset_csv = out_dir / f"{dataset_tag}__dataset_summary.csv"
    shapes_csv = out_dir / f"{dataset_tag}__tensor_shapes.csv"  # NEW

    pbar = tqdm(files, desc=f"[{dataset_tag}] scanning", unit="file", disable=quiet)

    for fp in pbar:
        row: Dict[str, Any] = {
            "dataset": dataset_tag,
            "file": fp.name,
            "path": str(fp),
            "loaded": 0,
            "error": "",
            "num_nodes": "",
            "num_edges": "",
            "mirrored_bidirectional": "",
            "edges_path_oob_count": "",
            "edges_path_max_valid": "",
            "has_aux": 0,
        }

        try:
            graphs, aux = load_graphs(str(fp))
            g = graphs[0]
            row["loaded"] = 1
            row["has_aux"] = 1 if isinstance(aux, dict) else 0
            n_loaded += 1

            row["num_nodes"] = int(g.num_nodes())
            row["num_edges"] = int(g.num_edges())

            u, v = g.edges()
            edge_pairs = set(zip(u.tolist(), v.tolist()))
            mirrored = all((b, a) in edge_pairs for (a, b) in edge_pairs) if len(edge_pairs) else True
            row["mirrored_bidirectional"] = int(bool(mirrored))

            # ndata
            for k, t in g.ndata.items():
                key = f"ndata:{k}"
                global_stats.setdefault(key, GlobalStat()).update(t, unique_cap)
                shape_stats.setdefault(key, ShapeStat()).update(t, shape_sample_cap)

            # edata
            for k, t in g.edata.items():
                key = f"edata:{k}"
                global_stats.setdefault(key, GlobalStat()).update(t, unique_cap)
                shape_stats.setdefault(key, ShapeStat()).update(t, shape_sample_cap)

            # aux
            if isinstance(aux, dict):
                for k, t in aux.items():
                    key = f"aux:{k}"
                    global_stats.setdefault(key, GlobalStat()).update(t, unique_cap)
                    shape_stats.setdefault(key, ShapeStat()).update(t, shape_sample_cap)

            # derived
            deg = g.in_degrees()
            global_stats.setdefault("derived:node_degree", GlobalStat()).update(deg, unique_cap)
            shape_stats.setdefault("derived:node_degree", ShapeStat()).update(deg, shape_sample_cap)

            # Presence counts
            if isinstance(aux, dict):
                for k in ("spatial_pos", "edges_path", "d2_distance", "angle_distance"):
                    kk = f"aux:{k}"
                    if k in aux and torch.is_tensor(aux[k]) and aux[k].numel() > 0:
                        present_counts[kk] += 1
            if "x" in g.ndata and torch.is_tensor(g.ndata["x"]) and g.ndata["x"].numel() > 0:
                present_counts["ndata:x"] += 1
            if "x" in g.edata and torch.is_tensor(g.edata["x"]) and g.edata["x"].numel() > 0:
                present_counts["edata:x"] += 1
            if "f" in g.ndata and torch.is_tensor(g.ndata["f"]) and g.ndata["f"].numel() > 0:
                present_counts["ndata:f"] += 1

            # edges_path validation
            if isinstance(aux, dict):
                oob, mx_valid = validate_edges_path_vs_num_edges(aux, int(g.num_edges()))
                row["edges_path_oob_count"] = oob
                row["edges_path_max_valid"] = "" if mx_valid is None else mx_valid
                if oob > 0:
                    edge_path_oob_files += 1
                    edge_path_oob_total += int(oob)

            # Core per-file columns
            core_cols = {}
            core_cols.update(tensor_minmax_row("ndata_z", g.ndata.get("z")))
            core_cols.update(tensor_minmax_row("ndata_y", g.ndata.get("y")))
            core_cols.update(tensor_minmax_row("ndata_l", g.ndata.get("l")))
            core_cols.update(tensor_minmax_row("ndata_a", g.ndata.get("a")))
            core_cols.update(tensor_minmax_row("ndata_f", g.ndata.get("f")))
            core_cols.update(tensor_minmax_row("ndata_x", g.ndata.get("x")))

            core_cols.update(tensor_minmax_row("edata_t", g.edata.get("t")))
            core_cols.update(tensor_minmax_row("edata_l", g.edata.get("l")))
            core_cols.update(tensor_minmax_row("edata_c", g.edata.get("c")))
            core_cols.update(tensor_minmax_row("edata_a", g.edata.get("a")))
            core_cols.update(tensor_minmax_row("edata_x", g.edata.get("x")))

            if isinstance(aux, dict):
                core_cols.update(tensor_minmax_row("aux_spatial_pos", aux.get("spatial_pos")))
                core_cols.update(tensor_minmax_row("aux_edges_path", aux.get("edges_path")))
                core_cols.update(tensor_minmax_row("aux_d2_distance", aux.get("d2_distance")))
                core_cols.update(tensor_minmax_row("aux_angle_distance", aux.get("angle_distance")))
            else:
                core_cols.update(tensor_minmax_row("aux_spatial_pos", None))
                core_cols.update(tensor_minmax_row("aux_edges_path", None))
                core_cols.update(tensor_minmax_row("aux_d2_distance", None))
                core_cols.update(tensor_minmax_row("aux_angle_distance", None))

            row.update(core_cols)

            # Extremes (topK)
            scan_extremes_for_key(dataset_tag, fp, "ndata:x", g.ndata.get("x"), topk)
            scan_extremes_for_key(dataset_tag, fp, "edata:x", g.edata.get("x"), topk)
            if isinstance(aux, dict):
                scan_extremes_for_key(dataset_tag, fp, "aux:d2_distance", aux.get("d2_distance"), topk)
                scan_extremes_for_key(dataset_tag, fp, "aux:angle_distance", aux.get("angle_distance"), topk)

        except Exception as e:
            n_failed += 1
            row["error"] = repr(e)

        per_file_rows.append(row)

    dt = time.time() - t0

    # ----------------------------
    # Write per-file CSV
    # ----------------------------
    all_cols: List[str] = []
    col_set = set()
    for r in per_file_rows:
        for k in r.keys():
            if k not in col_set:
                col_set.add(k)
                all_cols.append(k)

    with per_file_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        w.writerows(per_file_rows)

    # ----------------------------
    # Write global key stats CSV
    # ----------------------------
    global_rows: List[Dict[str, Any]] = []
    for key in sorted(global_stats.keys()):
        s = global_stats[key]
        global_rows.append(
            {
                "dataset": dataset_tag,
                "key": key,
                "dtype": s.dtype or "",
                "min": "" if s.min_v is None else s.min_v,
                "max": "" if s.max_v is None else s.max_v,
                "count": s.count,
                "nan_count": s.nan_count,
                "inf_count": s.inf_count,
                "unique_approx": len(s.uniques),
            }
        )
    with global_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "key", "dtype", "min", "max", "count", "nan_count", "inf_count", "unique_approx"],
        )
        w.writeheader()
        w.writerows(global_rows)

    # ----------------------------
    # Write tensor shapes CSV (NEW)
    # ----------------------------
    shape_rows: List[Dict[str, Any]] = []
    for key in sorted(shape_stats.keys()):
        s = shape_stats[key]
        shape_rows.append(
            {
                "dataset": dataset_tag,
                "key": key,
                "dtype": s.dtype or "",
                "tensors_seen": s.tensors_seen,
                "rank_min": "" if s.rank_min is None else s.rank_min,
                "rank_max": "" if s.rank_max is None else s.rank_max,
                "dim0_min": "" if s.dim_min[0] is None else s.dim_min[0],
                "dim0_max": "" if s.dim_max[0] is None else s.dim_max[0],
                "dim1_min": "" if s.dim_min[1] is None else s.dim_min[1],
                "dim1_max": "" if s.dim_max[1] is None else s.dim_max[1],
                "dim2_min": "" if s.dim_min[2] is None else s.dim_min[2],
                "dim2_max": "" if s.dim_max[2] is None else s.dim_max[2],
                "dim3_min": "" if s.dim_min[3] is None else s.dim_min[3],
                "dim3_max": "" if s.dim_max[3] is None else s.dim_max[3],
                "dim4_min": "" if s.dim_min[4] is None else s.dim_min[4],
                "dim4_max": "" if s.dim_max[4] is None else s.dim_max[4],
                "dim5_min": "" if s.dim_min[5] is None else s.dim_min[5],
                "dim5_max": "" if s.dim_max[5] is None else s.dim_max[5],
                "channels_lastdim_min": "" if s.ch_min is None else s.ch_min,
                "channels_lastdim_max": "" if s.ch_max is None else s.ch_max,
                "shape_samples": "; ".join(sorted(s.shape_samples)),
            }
        )

    with shapes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(shape_rows[0].keys()) if shape_rows else ["dataset", "key"],
        )
        w.writeheader()
        w.writerows(shape_rows)

    # ----------------------------
    # Write extremes topK CSV
    # ----------------------------
    with extremes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "file",
                "key",
                "abs_value",
                "value",
                "dtype",
                "shape",
                "idx0",
                "idx1",
                "idx2",
                "idx3",
                "note",
            ],
        )
        w.writeheader()
        for h in topk.items:
            w.writerow(
                {
                    "dataset": h.dataset_tag,
                    "file": h.file,
                    "key": h.key,
                    "abs_value": h.abs_value,
                    "value": h.value,
                    "dtype": h.dtype,
                    "shape": h.shape,
                    "idx0": "" if h.idx0 is None else h.idx0,
                    "idx1": "" if h.idx1 is None else h.idx1,
                    "idx2": "" if h.idx2 is None else h.idx2,
                    "idx3": "" if h.idx3 is None else h.idx3,
                    "note": h.note,
                }
            )

    # ----------------------------
    # Write dataset summary CSV (single row)
    # ----------------------------
    summary = {
        "dataset": dataset_tag,
        "root_dir": str(root_dir),
        "suffix": suffix,
        "files_found": n_files,
        "files_loaded_ok": n_loaded,
        "files_failed": n_failed,
        "scan_seconds": round(dt, 3),
        "edges_path_oob_files": edge_path_oob_files,
        "edges_path_oob_total": edge_path_oob_total,
    }
    for k, v in present_counts.items():
        summary[f"present_{k}"] = v
        summary[f"present_{k}_pct"] = (100.0 * v / n_loaded) if n_loaded else 0.0

    with dataset_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    # Minimal terminal output
    print(f"[DONE] dataset_tag={dataset_tag}")
    print(f"  per_file_summary   -> {per_file_csv}")
    print(f"  global_key_stats   -> {global_csv}")
    print(f"  tensor_shapes      -> {shapes_csv}")
    print(f"  extremes_topk      -> {extremes_csv}")
    print(f"  dataset_summary    -> {dataset_csv}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_tag", type=str, required=True)
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--suffix", type=str, default=DEFAULT_FILE_SUFFIX)
    ap.add_argument("--unique_cap", type=int, default=DEFAULT_UNIQUE_CAP)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--shape_samples", type=int, default=DEFAULT_SHAPE_SAMPLES)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    scan_dataset(
        dataset_tag=args.dataset_tag,
        root_dir=Path(args.root_dir),
        out_dir=Path(args.out_dir),
        suffix=args.suffix,
        unique_cap=args.unique_cap,
        topk_k=args.topk,
        shape_sample_cap=args.shape_samples,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()