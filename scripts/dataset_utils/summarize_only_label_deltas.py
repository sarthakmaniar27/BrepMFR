"""Summarize per-label deltas from only_face_label_mismatch.csv.

For each label class k, report:
    total_ours_count - total_auth_count   (over all 15,766 hit files)
and the number of files where that label specifically shifted.
"""
import csv
from collections import defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
CSV_PATH = _SCRIPTS / "sorted_dumps_full" / "only_face_label_mismatch.csv"

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


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]

    ours_cols = [(i, int(c.replace("ours_lbl_", "")))
                 for i, c in enumerate(header) if c.startswith("ours_lbl_")]
    auth_cols = [(i, int(c.replace("auth_lbl_", "")))
                 for i, c in enumerate(header) if c.startswith("auth_lbl_")]
    o_map = {k: i for i, k in ours_cols}
    a_map = {k: i for i, k in auth_cols}
    labels = sorted(set(o_map) | set(a_map))

    total_ours = defaultdict(int)
    total_auth = defaultdict(int)
    files_label_shifts = defaultdict(int)
    ours_zero_auth_k = defaultdict(int)
    auth_zero_ours_k = defaultdict(int)
    n_files = len(rows)

    for row in rows:
        for k in labels:
            oc = int(row[o_map[k]]) if k in o_map else 0
            ac = int(row[a_map[k]]) if k in a_map else 0
            total_ours[k] += oc
            total_auth[k] += ac
            if oc != ac:
                files_label_shifts[k] += 1
                if oc == 0 and ac > 0:
                    ours_zero_auth_k[k] += 1
                elif ac == 0 and oc > 0:
                    auth_zero_ours_k[k] += 1

    print(f"{'lbl':>3}  {'name':38s}  {'ours_total':>10}  {'auth_total':>10}  {'delta(O-A)':>11}  {'files_shifted':>13}  {'only_auth_has':>13}  {'only_ours_has':>13}")
    print("-" * 130)
    rows_out = []
    for k in labels:
        rows_out.append((
            k,
            FACE_LABEL_NAME.get(k, f"<{k}>")[:38],
            total_ours[k],
            total_auth[k],
            total_ours[k] - total_auth[k],
            files_label_shifts[k],
            ours_zero_auth_k[k],
            auth_zero_ours_k[k],
        ))
    rows_out.sort(key=lambda r: -r[5])
    for k, name, ou, au, d, ff, oza, azo in rows_out:
        print(f"{k:>3}  {name:38s}  {ou:>10}  {au:>10}  {d:>+11}  {ff:>13}  {oza:>13}  {azo:>13}")

    print()
    print(f"files scanned : {n_files}")


if __name__ == "__main__":
    main()
