"""
Dump sorted, aligned per-file feature CSVs from DGL .bin files for row-by-row
comparison between our regenerated CADSynth bins and the authors' originals.

For each input dataset ("ours" and/or "authors") the script emits three CSVs:

    <prefix>_faces.csv
        one row per face, sorted deterministically within each file by
        (face_area, uv_avg, face_adjacency, face_loop). Columns:
            file_stem, sort_idx, orig_idx,
            face_area, face_type, face_loop, face_adj, face_label,
            uv_mean_ch0..ch6, uv_std_ch0..ch6, uv_min_ch0..ch6, uv_max_ch0..ch6

    <prefix>_edges.csv
        one row per edge, sorted by (edge_length, uv_avg, edge_conv, edge_type).
        Columns:
            file_stem, sort_idx, orig_idx,
            src_orig, dst_orig, src_sort, dst_sort,
            edge_type, edge_length, edge_conv, edge_dihedral,
            uv_mean_ch0..ch6, uv_std_ch0..ch6, uv_min_ch0..ch6, uv_max_ch0..ch6

    <prefix>_pairs.csv   (A1 + A2 + A3 proximity)
        one row per face pair (i, j) in the FACE-SORTED order, so row i,j in
        the two datasets describes the same pair of sorted faces. Columns:
            file_stem, i_sort, j_sort, i_orig, j_orig,
            spatial_pos,                                  # A1
            d2_sum, d2_mean, d2_std, d2_argmax, d2_nz,    # A2 (d2_distance)
            ang_sum, ang_mean, ang_std, ang_argmax, ang_nz,   # A3 (angle_distance)
            edges_path_nonzero                            # count of non-pad entries in edges_path[i,j]

Why this layout makes comparison easy:
  - The sort keys use continuous-valued invariants first (face_area, edge_len)
    so ties are rare. When they occur, integer attributes break them.
  - Because the geometry underlying our bins and authors' bins is the same
    (we verified 98.8 % identical face-area multisets in scripts/
     cadsynth_validation_report_1k.txt), the sort yields the same face and
    edge order in both files, making row-by-row comparison meaningful.
  - Running on the same set of file stems twice (once per dataset) produces
    row-aligned CSVs you can diff, join in pandas, or open side-by-side in
    Excel.

Usage:
    # default: process first 100 file stems that exist in both datasets
    python scripts/dataset_utils/dump_sorted_bins_to_csv.py

    # process a specific number, with stratified sampling for good coverage
    python scripts/dataset_utils/dump_sorted_bins_to_csv.py --n 500 --strategy stratified

    # process everything (will produce very large CSVs)
    python scripts/dataset_utils/dump_sorted_bins_to_csv.py --all

    # only ours (no authors)
    python scripts/dataset_utils/dump_sorted_bins_to_csv.py --sides ours

    # custom output folder
    python scripts/dataset_utils/dump_sorted_bins_to_csv.py --out-dir scripts/sorted_dumps
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dgl.data.utils import load_graphs


DATASETS = {
    "ours": Path(r"Z:\Experiment6\source_dataset\output\bin"),
    "auth": Path(r"Z:\authors_data\bin"),
}


# --------------------------------------------------------------------------- #
# IO helpers                                                                  #
# --------------------------------------------------------------------------- #

def stem_of(p: Path, side: str) -> str:
    name = p.stem
    if side == "ours" and name.endswith("_101"):
        return name[:-4]
    return name


def build_index(side: str) -> Dict[str, Path]:
    folder = DATASETS[side]
    out: Dict[str, Path] = {}
    for p in folder.glob("*.bin"):
        out[stem_of(p, side)] = p
    return out


# --------------------------------------------------------------------------- #
# Per-channel UV statistics                                                   #
# --------------------------------------------------------------------------- #

def uv_stats(x: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    x shape: either [N, U, V, C] (faces) or [N, U, C] (edges).
    Returns per-node, per-channel mean/std/min/max.
    """
    arr = x.detach().cpu().float().numpy()
    # reduce over every dim except the first (node dim) and the last (channel dim)
    if arr.ndim == 4:
        reduce_axes = (1, 2)
    elif arr.ndim == 3:
        reduce_axes = (1,)
    else:
        # degenerate (no samples)
        n = arr.shape[0] if arr.ndim >= 1 else 0
        c = arr.shape[-1] if arr.ndim >= 1 else 0
        return {
            "mean": np.zeros((n, c), dtype=np.float32),
            "std":  np.zeros((n, c), dtype=np.float32),
            "min":  np.zeros((n, c), dtype=np.float32),
            "max":  np.zeros((n, c), dtype=np.float32),
        }
    return {
        "mean": arr.mean(axis=reduce_axes),
        "std":  arr.std(axis=reduce_axes),
        "min":  arr.min(axis=reduce_axes),
        "max":  arr.max(axis=reduce_axes),
    }


# --------------------------------------------------------------------------- #
# Deterministic sort orders                                                   #
# --------------------------------------------------------------------------- #

def face_sort_order(
    num_nodes: int,
    face_area: np.ndarray,
    uv_mean: np.ndarray,
    face_adj: np.ndarray,
    face_loop: np.ndarray,
) -> np.ndarray:
    """
    Sort faces by (face_area, avg of uv-mean-channels, face_adj, face_loop).
    Uses np.lexsort, which sorts by the LAST key primary.
    """
    uv_avg = uv_mean.mean(axis=1) if uv_mean.size else np.zeros(num_nodes, dtype=np.float32)
    # Tertiary keys first, primary key last in lexsort's keys list.
    keys = (
        face_loop.astype(np.float64),
        face_adj.astype(np.float64),
        uv_avg.astype(np.float64),
        face_area.astype(np.float64),
    )
    return np.lexsort(keys)


def edge_sort_order(
    num_edges: int,
    edge_len: np.ndarray,
    uv_mean: np.ndarray,
    edge_conv: np.ndarray,
    edge_type: np.ndarray,
) -> np.ndarray:
    uv_avg = uv_mean.mean(axis=1) if uv_mean.size else np.zeros(num_edges, dtype=np.float32)
    keys = (
        edge_type.astype(np.float64),
        edge_conv.astype(np.float64),
        uv_avg.astype(np.float64),
        edge_len.astype(np.float64),
    )
    return np.lexsort(keys)


# --------------------------------------------------------------------------- #
# Histogram summaries for A2 / A3                                             #
# --------------------------------------------------------------------------- #

def hist_summary(hist_2d: np.ndarray) -> Tuple[float, float, float, int, int]:
    """
    hist_2d: [64] float histogram for one face pair.
    Returns (sum, mean, std, argmax_bin, nonzero_count).
    """
    s = float(hist_2d.sum())
    m = float(hist_2d.mean()) if hist_2d.size else 0.0
    sd = float(hist_2d.std()) if hist_2d.size else 0.0
    arg = int(hist_2d.argmax()) if hist_2d.size else -1
    nz = int(np.count_nonzero(hist_2d))
    return s, m, sd, arg, nz


# --------------------------------------------------------------------------- #
# Core: process one file into three row iterables                             #
# --------------------------------------------------------------------------- #

def process_file(stem: str, path: Path) -> Optional[Dict[str, list]]:
    try:
        graphs, label_dict = load_graphs(str(path))
    except Exception as e:
        return {"error": f"load_error: {e}"}
    if not graphs:
        return {"error": "no_graph"}
    g = graphs[0]
    n = int(g.num_nodes())
    m = int(g.num_edges())

    # -------- face tensors --------
    y = g.ndata["y"].detach().cpu().float().numpy() if "y" in g.ndata else np.zeros(n, np.float32)
    z = g.ndata["z"].detach().cpu().long().numpy() if "z" in g.ndata else np.zeros(n, np.int64)
    fl = g.ndata["l"].detach().cpu().long().numpy() if "l" in g.ndata else np.zeros(n, np.int64)
    fa = g.ndata["a"].detach().cpu().long().numpy() if "a" in g.ndata else np.zeros(n, np.int64)
    ff = g.ndata["f"].detach().cpu().long().numpy() if "f" in g.ndata else np.zeros(n, np.int64)

    face_uv_stats = uv_stats(g.ndata["x"]) if "x" in g.ndata else {
        "mean": np.zeros((n, 7), np.float32),
        "std":  np.zeros((n, 7), np.float32),
        "min":  np.zeros((n, 7), np.float32),
        "max":  np.zeros((n, 7), np.float32),
    }

    face_pi = face_sort_order(n, y, face_uv_stats["mean"], fa, fl)
    # inverse permutation: orig_idx -> sort_idx
    face_inv = np.empty_like(face_pi)
    face_inv[face_pi] = np.arange(n)

    # -------- edge tensors --------
    src, dst = g.edges()
    src = src.detach().cpu().long().numpy()
    dst = dst.detach().cpu().long().numpy()
    et = g.edata["t"].detach().cpu().long().numpy() if "t" in g.edata else np.zeros(m, np.int64)
    el = g.edata["l"].detach().cpu().float().numpy() if "l" in g.edata else np.zeros(m, np.float32)
    ec = g.edata["c"].detach().cpu().long().numpy() if "c" in g.edata else np.zeros(m, np.int64)
    ea = g.edata["a"].detach().cpu().float().numpy() if "a" in g.edata else np.zeros(m, np.float32)

    edge_uv_stats = uv_stats(g.edata["x"]) if "x" in g.edata else {
        "mean": np.zeros((m, 7), np.float32),
        "std":  np.zeros((m, 7), np.float32),
        "min":  np.zeros((m, 7), np.float32),
        "max":  np.zeros((m, 7), np.float32),
    }

    edge_pi = edge_sort_order(m, el, edge_uv_stats["mean"], ec, et)

    # -------- proximity tensors --------
    spatial_pos = label_dict.get("spatial_pos") if label_dict is not None else None
    d2_distance = label_dict.get("d2_distance") if label_dict is not None else None
    angle_distance = label_dict.get("angle_distance") if label_dict is not None else None
    edges_path = label_dict.get("edges_path") if label_dict is not None else None

    sp = spatial_pos.detach().cpu().long().numpy() if spatial_pos is not None else None
    d2 = d2_distance.detach().cpu().float().numpy() if d2_distance is not None else None
    ad = angle_distance.detach().cpu().float().numpy() if angle_distance is not None else None
    ep = edges_path.detach().cpu().long().numpy() if edges_path is not None else None

    # -------- build face rows (sorted) --------
    face_rows: List[list] = []
    for sort_idx, orig_idx in enumerate(face_pi.tolist()):
        row: List = [
            stem,
            sort_idx,
            orig_idx,
            float(y[orig_idx]),
            int(z[orig_idx]),
            int(fl[orig_idx]),
            int(fa[orig_idx]),
            int(ff[orig_idx]),
        ]
        row.extend(face_uv_stats["mean"][orig_idx].tolist())
        row.extend(face_uv_stats["std"][orig_idx].tolist())
        row.extend(face_uv_stats["min"][orig_idx].tolist())
        row.extend(face_uv_stats["max"][orig_idx].tolist())
        face_rows.append(row)

    # -------- build edge rows (sorted) --------
    edge_rows: List[list] = []
    for sort_idx, orig_idx in enumerate(edge_pi.tolist()):
        s_o = int(src[orig_idx])
        d_o = int(dst[orig_idx])
        row = [
            stem,
            sort_idx,
            orig_idx,
            s_o,
            d_o,
            int(face_inv[s_o]),
            int(face_inv[d_o]),
            int(et[orig_idx]),
            float(el[orig_idx]),
            int(ec[orig_idx]),
            float(ea[orig_idx]),
        ]
        row.extend(edge_uv_stats["mean"][orig_idx].tolist())
        row.extend(edge_uv_stats["std"][orig_idx].tolist())
        row.extend(edge_uv_stats["min"][orig_idx].tolist())
        row.extend(edge_uv_stats["max"][orig_idx].tolist())
        edge_rows.append(row)

    # -------- build pair rows (in sorted face order) --------
    pair_rows: List[list] = []
    if n > 0 and (sp is not None or d2 is not None or ad is not None):
        for i_sort in range(n):
            i_orig = int(face_pi[i_sort])
            for j_sort in range(n):
                j_orig = int(face_pi[j_sort])
                row = [stem, i_sort, j_sort, i_orig, j_orig]
                row.append(int(sp[i_orig, j_orig]) if sp is not None else -1)
                if d2 is not None:
                    s, mean, std, arg, nz = hist_summary(d2[i_orig, j_orig])
                else:
                    s, mean, std, arg, nz = 0.0, 0.0, 0.0, -1, 0
                row.extend([s, mean, std, arg, nz])
                if ad is not None:
                    s2, mean2, std2, arg2, nz2 = hist_summary(ad[i_orig, j_orig])
                else:
                    s2, mean2, std2, arg2, nz2 = 0.0, 0.0, 0.0, -1, 0
                row.extend([s2, mean2, std2, arg2, nz2])
                ep_nz = int(np.count_nonzero(ep[i_orig, j_orig])) if ep is not None else 0
                row.append(ep_nz)
                pair_rows.append(row)

    return {
        "faces": face_rows,
        "edges": edge_rows,
        "pairs": pair_rows,
    }


# --------------------------------------------------------------------------- #
# CSV headers                                                                 #
# --------------------------------------------------------------------------- #

def uv_channel_headers(prefix: str, num_channels: int = 7) -> List[str]:
    out = []
    for stat in ("mean", "std", "min", "max"):
        out.extend([f"{prefix}_{stat}_ch{i}" for i in range(num_channels)])
    return out


FACE_HEADER = (
    [
        "file_stem", "sort_idx", "orig_idx",
        "face_area", "face_type", "face_loop", "face_adj", "face_label",
    ]
    + uv_channel_headers("face_uv")
)

EDGE_HEADER = (
    [
        "file_stem", "sort_idx", "orig_idx",
        "src_orig", "dst_orig", "src_sort", "dst_sort",
        "edge_type", "edge_length", "edge_conv", "edge_dihedral",
    ]
    + uv_channel_headers("edge_uv")
)

PAIR_HEADER = [
    "file_stem", "i_sort", "j_sort", "i_orig", "j_orig",
    "spatial_pos",
    "d2_sum", "d2_mean", "d2_std", "d2_argmax", "d2_nz",
    "ang_sum", "ang_mean", "ang_std", "ang_argmax", "ang_nz",
    "edges_path_nonzero",
]


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def pick_stems(common: List[str], n: int, strategy: str, seed: int) -> List[str]:
    rng = random.Random(seed)
    if strategy == "all" or n >= len(common):
        return common
    if strategy == "head":
        return common[:n]
    if strategy == "random":
        return rng.sample(common, n)
    # stratified: head + mid + tail + random fill
    head = common[: n // 4]
    tail = common[-(n // 4):]
    mid_start = len(common) // 2 - n // 8
    mid = common[mid_start : mid_start + n // 4]
    rest = [x for x in common if x not in set(head) | set(tail) | set(mid)]
    fill = rng.sample(rest, max(0, n - len(head) - len(mid) - len(tail)))
    picked = list(dict.fromkeys(head + mid + tail + fill))[:n]
    return sorted(picked)


def _read_done(done_path: Path) -> Tuple[set, Optional[Tuple[int, int, int]]]:
    """
    Returns (done_stems_set, last_offsets or None).

    done.tsv format: one line per CHECKPOINT (not per file). Each line lists the
    stems that were committed in this checkpoint (tab-separated after the offsets),
    plus the 3 CSV byte offsets valid at that point:

        face_off<TAB>edge_off<TAB>pair_off<TAB>stem1<TAB>stem2<TAB>...

    Only the LAST such line's stems are the true resume set (because after a crash
    we truncate CSVs to the last committed offsets; anything after is discarded).
    """
    if not done_path.exists():
        return set(), None
    done: set = set()
    last_off: Optional[Tuple[int, int, int]] = None
    with open(done_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                fo = int(parts[0]); eo = int(parts[1]); po = int(parts[2])
            except ValueError:
                continue
            stems = parts[3:]
            for s in stems:
                if s:
                    done.add(s)
            last_off = (fo, eo, po)
    return done, last_off


def write_dataset(
    side: str,
    stems: List[str],
    out_dir: Path,
    write_pairs: bool,
    progress_every: int,
    resume: bool = False,
) -> Dict[str, int]:
    idx = build_index(side)
    missing = [s for s in stems if s not in idx]
    if missing:
        print(f"[{side}] {len(missing)} stems not in this dataset; skipping them.")

    face_path = out_dir / f"{side}_faces.csv"
    edge_path = out_dir / f"{side}_edges.csv"
    pair_path = out_dir / f"{side}_pairs.csv"
    done_path = out_dir / f"{side}_done.tsv"

    done_stems: set = set()
    resume_ok = False
    if resume:
        done_stems, last_off = _read_done(done_path)
        if done_stems and face_path.exists() and edge_path.exists() and (not write_pairs or pair_path.exists()) and last_off is not None:
            # truncate partial tails that weren't committed to done.tsv
            try:
                os.truncate(face_path, last_off[0])
                os.truncate(edge_path, last_off[1])
                if write_pairs:
                    os.truncate(pair_path, last_off[2])
                resume_ok = True
                print(f"[{side}] RESUME: {len(done_stems)} stems already done; CSVs truncated to offsets face={last_off[0]:,} edge={last_off[1]:,} pair={last_off[2]:,}", flush=True)
            except OSError as e:
                print(f"[{side}] resume failed ({e}); starting fresh")
                resume_ok = False
                done_stems = set()

    print(f"[{side}] -> {face_path.name}, {edge_path.name}" + (f", {pair_path.name}" if write_pairs else "") + (f"  [resume {len(done_stems)} done]" if resume_ok else "  [fresh]"))

    counts = {"files": 0, "faces": 0, "edges": 0, "pairs": 0, "errors": 0}

    if resume_ok:
        face_fh = open(face_path, "a", newline="", encoding="utf-8")
        edge_fh = open(edge_path, "a", newline="", encoding="utf-8")
        pair_fh = open(pair_path, "a", newline="", encoding="utf-8") if write_pairs else None
        done_fh = open(done_path, "a", encoding="utf-8")
    else:
        face_fh = open(face_path, "w", newline="", encoding="utf-8")
        edge_fh = open(edge_path, "w", newline="", encoding="utf-8")
        pair_fh = open(pair_path, "w", newline="", encoding="utf-8") if write_pairs else None
        done_fh = open(done_path, "w", encoding="utf-8")

    try:
        face_w = csv.writer(face_fh)
        edge_w = csv.writer(edge_fh)
        pair_w = csv.writer(pair_fh) if pair_fh else None

        if not resume_ok:
            face_w.writerow(FACE_HEADER)
            edge_w.writerow(EDGE_HEADER)
            if pair_w is not None:
                pair_w.writerow(PAIR_HEADER)
            done_fh.write("# stem\tface_off\tedge_off\tpair_off\n")
            face_fh.flush(); edge_fh.flush();
            if pair_fh:
                pair_fh.flush()
            done_fh.flush()

        t0 = time.time()
        stems_to_do = [s for s in stems if s in idx and s not in done_stems]
        n_total = len(stems_to_do)
        n_skipped = len([s for s in stems if s in idx and s in done_stems])
        if n_skipped:
            print(f"[{side}] skipping {n_skipped} already-done stems")

        uncommitted: List[str] = []  # stems written to CSV but not yet in done.tsv

        for i, s in enumerate(stems_to_do, 1):
            res = process_file(s, idx[s])
            if res is None or "error" in res:
                counts["errors"] += 1
                continue
            for row in res["faces"]:
                face_w.writerow(row)
            for row in res["edges"]:
                edge_w.writerow(row)
            if pair_w is not None:
                for row in res["pairs"]:
                    pair_w.writerow(row)
            counts["files"] += 1
            counts["faces"] += len(res["faces"])
            counts["edges"] += len(res["edges"])
            counts["pairs"] += len(res["pairs"])
            uncommitted.append(s)

            # checkpoint every progress_every files (or at the end)
            if i % progress_every == 0 or i == n_total:
                face_fh.flush(); edge_fh.flush()
                face_off = face_fh.tell()
                edge_off = edge_fh.tell()
                if pair_fh:
                    pair_fh.flush()
                    pair_off = pair_fh.tell()
                else:
                    pair_off = 0
                try:
                    os.fsync(face_fh.fileno())
                    os.fsync(edge_fh.fileno())
                    if pair_fh:
                        os.fsync(pair_fh.fileno())
                except OSError:
                    pass
                # write one done.tsv line listing all stems committed in this checkpoint
                done_fh.write(f"{face_off}\t{edge_off}\t{pair_off}\t" + "\t".join(uncommitted) + "\n")
                done_fh.flush()
                try:
                    os.fsync(done_fh.fileno())
                except OSError:
                    pass
                uncommitted.clear()
                rate = i / max(1e-6, time.time() - t0)
                print(f"  [{i}/{n_total}] files={counts['files']} faces={counts['faces']:,} edges={counts['edges']:,} pairs={counts['pairs']:,} ({rate:.1f} f/s)", flush=True)
    finally:
        face_fh.close()
        edge_fh.close()
        if pair_fh:
            pair_fh.close()
        done_fh.close()
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="Number of files to process (default 100)")
    ap.add_argument("--all", action="store_true", help="Process ALL common stems (overrides --n)")
    ap.add_argument("--strategy", choices=["head", "random", "stratified"], default="stratified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "sorted_dumps"))
    ap.add_argument("--sides", nargs="+", choices=["ours", "auth"], default=["ours", "auth"])
    ap.add_argument("--no-pairs", action="store_true", help="Skip A1/A2/A3 pair CSV (faster, smaller)")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint in {side}_done.tsv (append mode)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build indices and find common stems
    print("[index] scanning datasets ...", flush=True)
    indices = {s: build_index(s) for s in DATASETS}
    common = sorted(set(indices["ours"]) & set(indices["auth"]))
    print(f"[index] ours={len(indices['ours']):,} auth={len(indices['auth']):,} common={len(common):,}")

    stems = pick_stems(common, args.n if not args.all else len(common), "all" if args.all else args.strategy, args.seed)
    print(f"[pick] selected {len(stems)} stems (strategy={'all' if args.all else args.strategy})")

    write_pairs = not args.no_pairs
    summary = {}
    for side in args.sides:
        print()
        counts = write_dataset(side, stems, out_dir, write_pairs, args.progress_every, resume=args.resume)
        summary[side] = counts

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  out_dir          : {out_dir}")
    print(f"  selected stems   : {len(stems)}")
    for side, c in summary.items():
        print(f"  [{side}] files written={c['files']}  faces={c['faces']:,}  edges={c['edges']:,}  pairs={c['pairs']:,}  errors={c['errors']}")
    print()
    print("  Row-by-row diff example (PowerShell):")
    print(f"    fc /w {out_dir}\\ours_faces.csv {out_dir}\\auth_faces.csv > {out_dir}\\faces_diff.txt")
    print("  Or in pandas:")
    print("    import pandas as pd")
    print(f"    ours = pd.read_csv(r'{out_dir}\\\\ours_faces.csv')")
    print(f"    auth = pd.read_csv(r'{out_dir}\\\\auth_faces.csv')")
    print("    joined = ours.merge(auth, on=['file_stem','sort_idx'], suffixes=('_ours','_auth'))")
    print("    joined['delta_area'] = (joined['face_area_ours'] - joined['face_area_auth']).abs()")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
