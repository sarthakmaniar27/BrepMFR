"""
Scan every .bin in each dataset and report the full set of unique values
of graph.ndata['z'] (face_type) and graph.ndata['t']/edata['t'] etc.

Also reports the per-class count, min/max, and files that contain each
value -- enough to confirm whether 0 is a real class or reserved for padding.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from dgl.data.utils import load_graphs

DATASETS = {
    "ours_cadsynth":   Path(r"Z:\Experiment6\source_dataset\output\bin"),
    "auth_cadsynth":   Path(r"Z:\authors_data\bin"),
    "ours_mfcad++":    Path(r"Z:\Experiment6\target_dataset\output\bin"),
}


def scan_folder(folder: Path, limit: int = 0, progress_every: int = 5000) -> Dict[str, object]:
    files = list(folder.glob("*.bin"))
    if limit and limit < len(files):
        files = files[:limit]
    print(f"[scan] {folder}  files={len(files):,}", flush=True)
    t0 = time.time()

    z_counter = Counter()        # face_type  (ndata z)
    t_counter = Counter()        # edge_type  (edata t)
    c_counter = Counter()        # edge_conv  (edata c)
    f_counter = Counter()        # label_feat (ndata f)
    l_counter = Counter()        # face_loop  (ndata l)
    a_counter = Counter()        # face_adj   (ndata a)

    # track one example file per unique face_type value we encounter
    z_example: Dict[int, str] = {}

    failed = 0
    for i, p in enumerate(files, 1):
        try:
            gs, _ = load_graphs(str(p))
            g = gs[0]
            if "z" in g.ndata:
                vals = g.ndata["z"].long().cpu().numpy().tolist()
                z_counter.update(vals)
                for v in vals:
                    if v not in z_example:
                        z_example[v] = p.name
            if "t" in g.edata:
                t_counter.update(g.edata["t"].long().cpu().numpy().tolist())
            if "c" in g.edata:
                c_counter.update(g.edata["c"].long().cpu().numpy().tolist())
            if "f" in g.ndata:
                f_counter.update(g.ndata["f"].long().cpu().numpy().tolist())
            if "l" in g.ndata:
                l_counter.update(g.ndata["l"].long().cpu().numpy().tolist())
            if "a" in g.ndata:
                a_counter.update(g.ndata["a"].long().cpu().numpy().tolist())
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  [warn] failed to load {p.name}: {e}")
        if i % progress_every == 0:
            print(f"  [{i:>6}/{len(files):,}]  elapsed={time.time()-t0:.1f}s", flush=True)

    print(f"[scan] done: {len(files):,} files, failed={failed}, elapsed={time.time()-t0:.1f}s")
    return {
        "folder": str(folder),
        "n_files": len(files),
        "failed": failed,
        "z": dict(z_counter),
        "t": dict(t_counter),
        "c": dict(c_counter),
        "f": dict(f_counter),
        "l": dict(l_counter),
        "a": dict(a_counter),
        "z_example_file_per_value": z_example,
    }


def format_counter_row(name: str, counter: Dict[int, int]) -> str:
    if not counter:
        return f"  {name:<20}  -- empty --"
    total = sum(counter.values())
    parts = []
    for k in sorted(counter):
        v = counter[k]
        parts.append(f"{k}: {v:,} ({100*v/total:.2f}%)")
    return f"  {name:<20}  unique={len(counter)}  total={total:,}\n    " + "\n    ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Cap files per dataset (0 = all)")
    args = ap.parse_args()

    all_results = {}
    for name, folder in DATASETS.items():
        if not folder.exists():
            print(f"[skip] {name} folder missing: {folder}")
            continue
        print("\n" + "=" * 96)
        print(f"  {name}  ({folder})")
        print("=" * 96)
        res = scan_folder(folder, limit=args.limit)
        all_results[name] = res

        print()
        print(format_counter_row("ndata['z']  face_type", res["z"]))
        print(format_counter_row("edata['t']  edge_type", res["t"]))
        print(format_counter_row("edata['c']  edge_conv", res["c"]))
        print(format_counter_row("ndata['f']  feature_label", res["f"]))
        print(f"  ndata['l']  face_loop   unique={len(res['l'])}  min={min(res['l']) if res['l'] else None}  max={max(res['l']) if res['l'] else None}")
        print(f"  ndata['a']  face_adj    unique={len(res['a'])}  min={min(res['a']) if res['a'] else None}  max={max(res['a']) if res['a'] else None}")
        # Show example file for each z value
        print("  z example files:")
        for v in sorted(res["z_example_file_per_value"]):
            print(f"    z={v}  first seen in  {res['z_example_file_per_value'][v]}")

    # cross-dataset comparison
    print("\n" + "=" * 96)
    print("  CROSS-DATASET UNIQUE VALUES FOR face_type ('z'):")
    print("=" * 96)
    for name, res in all_results.items():
        print(f"  {name:<20}  unique_values={sorted(res['z'].keys())}")

    print("\n" + "=" * 96)
    print("  PADDING CHECK:")
    print("=" * 96)
    print("  DGL .bin files store per-node values -- there is NO padding inside the .bin itself.")
    print("  Padding (to a fixed sequence length) is added LATER inside the BrepEncoder during batching,")
    print("  which uses padding_idx=0 for the embedding lookup. Therefore, if z=0 also appears as a real")
    print("  surface-type value in the dataset, the embedding for z=0 is BOTH 'real-plane' and 'padding',")
    print("  which collapses those two concepts to the same vector.")
    for name, res in all_results.items():
        n_zero = res["z"].get(0, 0)
        print(f"    {name:<20}  count(z==0) = {n_zero:,}  (as fraction of all faces = {100*n_zero/max(1,sum(res['z'].values())):.2f}%)")

    # save to json file
    import json
    out_path = Path(__file__).parent / "face_type_range_scan.json"
    with open(out_path, "w") as fh:
        json.dump({k: {kk: (dict(vv) if hasattr(vv, "items") else vv) for kk, vv in v.items()} for k, v in all_results.items()}, fh, indent=2, default=str)
    print(f"\n[out] {out_path}")


if __name__ == "__main__":
    main()
