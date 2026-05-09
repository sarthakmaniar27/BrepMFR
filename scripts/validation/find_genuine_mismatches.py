"""
Walk both datasets and find small files whose MULTISETS (histograms)
of face_label, face_type, edge_type, edge_conv, or edge_dihedral
genuinely differ between OURS and AUTHORS - i.e. not sort-artifacts.

Usage:
    python scripts/find_genuine_mismatches.py --max 30 --n-faces-max 15
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from dgl.data.utils import load_graphs


DATASETS = {
    "ours": Path(r"Z:\Experiment6\source_dataset\output\bin"),
    "auth": Path(r"Z:\authors_data\bin"),
}


def stem_of(p: Path, side: str) -> str:
    name = p.stem
    if side == "ours" and name.endswith("_101"):
        return name[:-4]
    return name


def load_attrs(p: Path) -> Dict:
    graphs, _ = load_graphs(str(p))
    g = graphs[0]
    n = int(g.num_nodes()); m = int(g.num_edges())
    return {
        "n": n, "m": m,
        "z": g.ndata["z"].cpu().long().numpy() if "z" in g.ndata else np.zeros(n, np.int64),
        "ff": g.ndata["f"].cpu().long().numpy() if "f" in g.ndata else np.zeros(n, np.int64),
        "y": g.ndata["y"].cpu().float().numpy() if "y" in g.ndata else np.zeros(n, np.float32),
        "et": g.edata["t"].cpu().long().numpy() if "t" in g.edata else np.zeros(m, np.int64),
        "ec": g.edata["c"].cpu().long().numpy() if "c" in g.edata else np.zeros(m, np.int64),
        "el": g.edata["l"].cpu().float().numpy() if "l" in g.edata else np.zeros(m, np.float32),
        "ea": g.edata["a"].cpu().float().numpy() if "a" in g.edata else np.zeros(m, np.float32),
    }


def multiset(a: np.ndarray) -> Dict[int, int]:
    u, c = np.unique(a.astype(np.int64), return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=30, help="max mismatching files to print")
    ap.add_argument("--n-faces-max", type=int, default=15)
    ap.add_argument("--sample", type=int, default=5000, help="number of common stems to scan")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    print("[index] scanning datasets ...")
    idx_o = {stem_of(p, "ours"): p for p in DATASETS["ours"].glob("*.bin")}
    idx_a = {stem_of(p, "auth"): p for p in DATASETS["auth"].glob("*.bin")}
    common = sorted(set(idx_o) & set(idx_a))
    print(f"[index] common={len(common):,}")

    rng = random.Random(args.seed)
    sample = rng.sample(common, min(args.sample, len(common)))

    hits: List[tuple] = []
    for i, stem in enumerate(sample, 1):
        try:
            o = load_attrs(idx_o[stem])
            a = load_attrs(idx_a[stem])
        except Exception:
            continue
        if o["n"] > args.n_faces_max:
            continue
        # compare multisets
        diffs = []
        for key, label in (("z", "face_type"), ("ff", "face_label"), ("et", "edge_type"), ("ec", "edge_conv")):
            mo = multiset(o[key]); ma = multiset(a[key])
            if mo != ma:
                diffs.append(label)
        # for dihedral: compare sorted values with tolerance
        o_sorted = np.sort(o["ea"]); a_sorted = np.sort(a["ea"])
        if len(o_sorted) == len(a_sorted):
            bad = int(np.sum(np.abs(o_sorted - a_sorted) > 1e-3))
            if bad > 0:
                diffs.append(f"edge_dihedral({bad}/{len(o_sorted)})")
        else:
            diffs.append(f"edge_dihedral(count:{len(o_sorted)}vs{len(a_sorted)})")
        if diffs:
            hits.append((o["n"], o["m"], stem, diffs))
        if len(hits) >= args.max * 3:
            break
        if i % 500 == 0:
            print(f"  scanned {i}/{len(sample)}  hits={len(hits)}")

    hits.sort()
    print("\nn_faces  n_edges  stem       mismatched_attrs")
    for n, m, s, diffs in hits[:args.max]:
        print(f"{n:>7}  {m:>7}  {s}  {', '.join(diffs)}")
    print(f"\n[total hits from scan] {len(hits)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
