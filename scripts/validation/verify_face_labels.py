"""
Re-verify face-label parity across our regenerated bins vs authors' bins.

For every common stem, compare ONLY the multiset of ndata['f'] (face_label).
All other channels (face_type, edge_type, dihedral, etc.) are intentionally
ignored.

Outputs (under scripts/sorted_dumps_full/):
    label_verify_summary.txt
        Aggregate report: total scanned, identical, mismatched, missing-on-side,
        topology-size-differs, errors. Plus per-label deltas across the
        mismatched subset.
    label_verify_mismatches.txt
        One stem per line for files whose face_label multiset still differs.
    label_verify_mismatches.csv
        Per-file label deltas (ours vs authors) for each mismatched stem.

Usage:
    python scripts/validation/verify_face_labels.py
    python scripts/validation/verify_face_labels.py --workers 10
    python scripts/validation/verify_face_labels.py --limit 1000   # debug
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dgl.data.utils import load_graphs

OURS_DIR = Path(r"Z:\Experiment6\source_dataset\output\bin")
AUTH_DIR = Path(r"Z:\authors_data\bin")
OUT_DIR = Path(r"C:\Users\D58\Desktop\BrepMFR\scripts\sorted_dumps_full")

FACE_LABEL_NAME = {
    0: "Stock",
    1: "Rectangular through slot",
    2: "Triangular through slot",
    3: "Rectangular passage",
    4: "Triangular passage",
    5: "6-sided passage",
    6: "Rectangular through step",
    7: "2-sided through step",
    8: "Slanted through step",
    9: "Rectangular blind step",
    10: "Triangular blind step",
    11: "Rectangular blind slot",
    12: "Rectangular pocket",
    13: "Triangular pocket",
    14: "6-sided pocket",
    15: "Chamfer",
    16: "Circular through slot",
    17: "Through hole",
    18: "Circular blind step",
    19: "Horizontal circular end blind slot",
    20: "Vertical circular end blind slot",
    21: "Circular end pocket",
    22: "O-ring",
    23: "Blind hole",
    24: "Round",
}


def ours_stem(p: Path) -> str:
    s = p.stem
    return s[:-4] if s.endswith("_101") else s


def _worker(task):
    stem, p_ours, p_auth = task
    try:
        go, _ = load_graphs(p_ours); go = go[0]
        ga, _ = load_graphs(p_auth); ga = ga[0]
    except Exception as e:
        return (stem, "error", None, None, str(e)[:120])

    no, na = int(go.num_nodes()), int(ga.num_nodes())
    if no != na:
        return (stem, "topo_diff", {"num_faces": no}, {"num_faces": na}, "")

    fo = go.ndata.get("f"); fa = ga.ndata.get("f")
    if fo is None or fa is None:
        return (stem, "missing_label", None, None,
                f"ours={'yes' if fo is not None else 'no'} auth={'yes' if fa is not None else 'no'}")

    co = dict(Counter(fo.tolist()))
    ca = dict(Counter(fa.tolist()))
    if co == ca:
        return (stem, "match", co, ca, "")
    return (stem, "mismatch", co, ca, "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0, help="Max stems (0 = all)")
    ap.add_argument("--stem-list", type=str, default="",
                    help="Optional path to a .txt file with one stem per line. "
                         "If given, only those stems are verified.")
    ap.add_argument("--out-tag", type=str, default="",
                    help="Optional tag inserted into output filenames "
                         "(e.g. 'fixed' -> label_verify_summary_fixed.txt)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ours_map = {ours_stem(p): str(p) for p in OURS_DIR.glob("*.bin")}
    auth_map = {p.stem: str(p) for p in AUTH_DIR.glob("*.bin")}
    common_set = set(ours_map) & set(auth_map)
    only_ours = sorted(set(ours_map) - set(auth_map))
    only_auth = sorted(set(auth_map) - set(ours_map))

    requested_missing_ours = []
    requested_missing_auth = []

    if args.stem_list:
        wanted = []
        with open(args.stem_list, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    wanted.append(s)
        wanted_set = set(wanted)
        common = sorted(wanted_set & common_set)
        requested_missing_ours = sorted(wanted_set - set(ours_map))
        requested_missing_auth = sorted(wanted_set - set(auth_map))
        print(f"[scan] using stem list     : {args.stem_list}")
        print(f"[scan] requested stems     : {len(wanted)}")
        print(f"[scan] missing in ours dir : {len(requested_missing_ours)}")
        print(f"[scan] missing in auth dir : {len(requested_missing_auth)}")
    else:
        common = sorted(common_set)

    if args.limit:
        common = common[: args.limit]
    total = len(common)
    print(f"[scan] common stems used   : {total}")
    print(f"[scan] only in ours        : {len(only_ours)}")
    print(f"[scan] only in authors     : {len(only_auth)}")
    print(f"[scan] workers             : {args.workers}")

    tasks = [(s, ours_map[s], auth_map[s]) for s in common]

    n_match = 0
    n_topo = 0
    n_missing = 0
    n_error = 0
    mismatches = []
    per_label_files = defaultdict(int)
    per_label_ours_total = defaultdict(int)
    per_label_auth_total = defaultdict(int)

    processed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futs):
            stem, status, co, ca, info = fut.result()
            processed += 1
            if status == "match":
                n_match += 1
            elif status == "topo_diff":
                n_topo += 1
            elif status == "missing_label":
                n_missing += 1
            elif status == "error":
                n_error += 1
            elif status == "mismatch":
                mismatches.append((stem, co, ca))
                keys = set(co) | set(ca)
                for k in keys:
                    if co.get(k, 0) != ca.get(k, 0):
                        per_label_files[k] += 1
                    per_label_ours_total[k] += co.get(k, 0)
                    per_label_auth_total[k] += ca.get(k, 0)

            if processed % 5000 == 0:
                print(f"  {processed}/{total}  match={n_match}  mismatch={len(mismatches)}  "
                      f"topo={n_topo}  missing={n_missing}  err={n_error}")

    mismatches.sort(key=lambda r: r[0])

    tag = f"_{args.out_tag}" if args.out_tag else ""
    txt_path = OUT_DIR / f"label_verify_mismatches{tag}.txt"
    csv_path = OUT_DIR / f"label_verify_mismatches{tag}.csv"
    sum_path = OUT_DIR / f"label_verify_summary{tag}.txt"

    with txt_path.open("w", encoding="utf-8") as f:
        for stem, _, _ in mismatches:
            f.write(stem + "\n")

    all_labels = sorted({k for _, co, ca in mismatches for k in (set(co) | set(ca))})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["stem"] + [f"ours_{k}" for k in all_labels] + [f"auth_{k}" for k in all_labels] \
             + [f"delta_{k}" for k in all_labels]
        w.writerow(cols)
        for stem, co, ca in mismatches:
            row = [stem]
            row += [co.get(k, 0) for k in all_labels]
            row += [ca.get(k, 0) for k in all_labels]
            row += [co.get(k, 0) - ca.get(k, 0) for k in all_labels]
            w.writerow(row)

    pct = lambda n: f"{100.0 * n / max(total, 1):6.2f}%"
    lines = []
    lines.append("=== face-label verification (multiset comparison) ===")
    lines.append(f"  ours dir            : {OURS_DIR}")
    lines.append(f"  authors dir         : {AUTH_DIR}")
    lines.append(f"  common stems        : {total}")
    lines.append(f"  match               : {n_match:>7}  {pct(n_match)}")
    lines.append(f"  mismatch            : {len(mismatches):>7}  {pct(len(mismatches))}")
    lines.append(f"  topology size diff  : {n_topo:>7}  {pct(n_topo)}")
    lines.append(f"  missing 'f' tensor  : {n_missing:>7}  {pct(n_missing)}")
    lines.append(f"  errors              : {n_error:>7}  {pct(n_error)}")
    lines.append(f"  only-in-ours        : {len(only_ours)}")
    lines.append(f"  only-in-authors     : {len(only_auth)}")
    lines.append("")
    lines.append("=== per-label deltas across mismatched files ===")
    lines.append(f"{'lbl':>3}  {'name':38s}  {'ours_total':>10}  {'auth_total':>10}  {'delta(O-A)':>11}  {'files_shifted':>13}")
    lines.append("-" * 100)
    for k in sorted(set(per_label_ours_total) | set(per_label_auth_total)):
        ou = per_label_ours_total[k]
        au = per_label_auth_total[k]
        d = ou - au
        files = per_label_files[k]
        name = FACE_LABEL_NAME.get(k, f"<{k}>")[:38]
        lines.append(f"{k:>3}  {name:38s}  {ou:>10}  {au:>10}  {d:>+11}  {files:>13}")
    lines.append("")
    lines.append(f"file list  : {txt_path}")
    lines.append(f"per-file   : {csv_path}")

    summary_text = "\n".join(lines)
    with sum_path.open("w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print()
    print(summary_text)


if __name__ == "__main__":
    main()
