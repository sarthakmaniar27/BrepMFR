"""
Scan common .bin files and find small ones whose edge-type MULTISETS
(Counter of edata['t']) differ between ours and authors. Prints the
first few candidates sorted by smallest total edges.

Usage:
    python scripts/find_edge_type_mismatch.py --max-edges 60 --sample 3000
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from dgl.data.utils import load_graphs

OURS = Path(r"Z:\Experiment6\source_dataset\output\bin")
AUTH = Path(r"Z:\authors_data\bin")


def ours_stem(p: Path) -> str:
    s = p.stem
    return s[:-4] if s.endswith("_101") else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-edges", type=int, default=60, help="Only consider files with <= this many edges on either side")
    ap.add_argument("--sample", type=int, default=3000, help="Number of random common stems to scan")
    ap.add_argument("--top", type=int, default=10, help="Report this many candidates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    ours_stems = {ours_stem(p): p for p in OURS.glob("*.bin")}
    auth_stems = {p.stem: p for p in AUTH.glob("*.bin")}
    common = sorted(set(ours_stems) & set(auth_stems))
    print(f"common stems: {len(common)}")

    if args.sample < len(common):
        stems = random.sample(common, args.sample)
    else:
        stems = common

    hits = []
    for i, stem in enumerate(stems):
        try:
            go, _ = load_graphs(str(ours_stems[stem]))
            ga, _ = load_graphs(str(auth_stems[stem]))
        except Exception:
            continue
        go, ga = go[0], ga[0]
        mo, ma = int(go.num_edges()), int(ga.num_edges())
        if mo > args.max_edges and ma > args.max_edges:
            continue
        t_ours = go.edata.get("t")
        t_auth = ga.edata.get("t")
        if t_ours is None or t_auth is None:
            continue
        co = Counter(t_ours.tolist())
        ca = Counter(t_auth.tolist())
        if co != ca:
            diff = {k: (co.get(k, 0), ca.get(k, 0)) for k in sorted(set(co) | set(ca)) if co.get(k, 0) != ca.get(k, 0)}
            hits.append((mo + ma, stem, mo, ma, dict(co), dict(ca), diff))
        if (i + 1) % 500 == 0:
            print(f"  scanned {i+1}/{len(stems)}  hits={len(hits)}")

    hits.sort(key=lambda r: (r[0], r[1]))
    print()
    print(f"Found {len(hits)} files with edge-type multiset mismatch")
    print("-" * 100)
    for total, stem, mo, ma, co, ca, diff in hits[: args.top]:
        print(f"{stem}  edges ours={mo} auth={ma}  ours={co}  auth={ca}  diff={diff}")


if __name__ == "__main__":
    main()
