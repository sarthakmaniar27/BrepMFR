"""
Scan every common .bin and report stems where the ONLY disagreement
between ours and authors is the face_label (ndata['f']) multiset.

"Only" here means all of these must be equal:
    - num_faces, num_edges
    - multiset of face_type  (ndata['z'])
    - multiset of face_loop  (ndata['l'])
    - multiset of face_adj   (ndata['a'])
    - multiset of edge_type  (edata['t'])
    - multiset of edge_conv  (edata['c'])
while multiset of face_label (ndata['f']) must differ.

Continuous-valued channels (face_area, edge_length, edge_dihedral,
UV grids) are intentionally ignored because sampling-noise alone
causes harmless byte-level drift.

Outputs:
    scripts/sorted_dumps_full/only_face_label_mismatch.txt   (one stem per line)
    scripts/sorted_dumps_full/only_face_label_mismatch.csv   (stem + per-label deltas)

Usage:
    python scripts/find_only_face_label_mismatch.py
    python scripts/find_only_face_label_mismatch.py --limit 5000   # debug
    python scripts/find_only_face_label_mismatch.py --workers 8    # parallel
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dgl.data.utils import load_graphs

OURS_DIR = Path(r"Z:\Experiment6\source_dataset\output\bin")
AUTH_DIR = Path(r"Z:\authors_data\bin")
OUT_DIR = Path(r"C:\Users\D58\Desktop\BrepMFR\scripts\sorted_dumps_full")


def ours_stem(p: Path) -> str:
    s = p.stem
    return s[:-4] if s.endswith("_101") else s


def _counter(t) -> dict:
    if t is None:
        return {}
    return dict(Counter(t.tolist()))


def check_one(stem_path_ours: str, stem_path_auth: str) -> tuple | None:
    """Return (stem, only_label, ours_lbl, auth_lbl, info) or None on error."""
    try:
        go, _ = load_graphs(stem_path_ours); go = go[0]
        ga, _ = load_graphs(stem_path_auth); ga = ga[0]
    except Exception:
        return None

    if int(go.num_nodes()) != int(ga.num_nodes()):
        return ("", False, {}, {}, "num_faces_differ")
    if int(go.num_edges()) != int(ga.num_edges()):
        return ("", False, {}, {}, "num_edges_differ")

    o_ft = _counter(go.ndata.get("z"))
    a_ft = _counter(ga.ndata.get("z"))
    o_fl = _counter(go.ndata.get("l"))
    a_fl = _counter(ga.ndata.get("l"))
    o_fa = _counter(go.ndata.get("a"))
    a_fa = _counter(ga.ndata.get("a"))
    o_lb = _counter(go.ndata.get("f"))
    a_lb = _counter(ga.ndata.get("f"))
    o_et = _counter(go.edata.get("t"))
    a_et = _counter(ga.edata.get("t"))
    o_ec = _counter(go.edata.get("c"))
    a_ec = _counter(ga.edata.get("c"))

    label_differs = o_lb != a_lb
    others_equal = (
        o_ft == a_ft and o_fl == a_fl and o_fa == a_fa
        and o_et == a_et and o_ec == a_ec
    )

    return (label_differs and others_equal, o_lb, a_lb)


def _worker(task):
    stem, p_ours, p_auth = task
    try:
        go, _ = load_graphs(p_ours); go = go[0]
        ga, _ = load_graphs(p_auth); ga = ga[0]
    except Exception as e:
        return (stem, None, None, None, f"error:{e!s}")

    if int(go.num_nodes()) != int(ga.num_nodes()):
        return (stem, False, None, None, "num_faces_differ")
    if int(go.num_edges()) != int(ga.num_edges()):
        return (stem, False, None, None, "num_edges_differ")

    o_ft = _counter(go.ndata.get("z")); a_ft = _counter(ga.ndata.get("z"))
    o_fl = _counter(go.ndata.get("l")); a_fl = _counter(ga.ndata.get("l"))
    o_fa = _counter(go.ndata.get("a")); a_fa = _counter(ga.ndata.get("a"))
    o_lb = _counter(go.ndata.get("f")); a_lb = _counter(ga.ndata.get("f"))
    o_et = _counter(go.edata.get("t")); a_et = _counter(ga.edata.get("t"))
    o_ec = _counter(go.edata.get("c")); a_ec = _counter(ga.edata.get("c"))

    label_differs = o_lb != a_lb
    others_equal = (
        o_ft == a_ft and o_fl == a_fl and o_fa == a_fa
        and o_et == a_et and o_ec == a_ec
    )
    hit = bool(label_differs and others_equal)
    return (stem, hit, o_lb, a_lb, "ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max stems to scan (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ours_map = {ours_stem(p): str(p) for p in OURS_DIR.glob("*.bin")}
    auth_map = {p.stem: str(p) for p in AUTH_DIR.glob("*.bin")}
    common = sorted(set(ours_map) & set(auth_map))
    if args.limit:
        common = common[: args.limit]
    total = len(common)
    print(f"[scan] common stems: {total}  workers: {args.workers}")

    tasks = [(stem, ours_map[stem], auth_map[stem]) for stem in common]

    hits: list[tuple] = []
    errors = 0
    both_equal = 0
    topology_diff = 0
    label_plus_others = 0
    processed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed(ex.submit(_worker, t) for t in tasks):
            stem, hit, o_lb, a_lb, info = fut.result()
            processed += 1
            if info != "ok":
                if info.startswith("error"):
                    errors += 1
                else:
                    topology_diff += 1
            elif hit:
                hits.append((stem, o_lb, a_lb))
            elif o_lb == a_lb:
                both_equal += 1
            else:
                label_plus_others += 1

            if processed % 2000 == 0:
                print(f"  {processed}/{total}  hits={len(hits)}  label+others={label_plus_others}  "
                      f"topology_diff={topology_diff}  errors={errors}")

    hits.sort(key=lambda r: r[0])
    txt_path = OUT_DIR / "only_face_label_mismatch.txt"
    csv_path = OUT_DIR / "only_face_label_mismatch.csv"

    with txt_path.open("w", encoding="utf-8") as f:
        for stem, _, _ in hits:
            f.write(stem + "\n")

    all_labels = set()
    for _, o, a in hits:
        all_labels.update(o); all_labels.update(a)
    cols = ["stem"] + [f"ours_lbl_{k}" for k in sorted(all_labels)] + [f"auth_lbl_{k}" for k in sorted(all_labels)]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for stem, o, a in hits:
            row = [stem]
            row += [o.get(k, 0) for k in sorted(all_labels)]
            row += [a.get(k, 0) for k in sorted(all_labels)]
            w.writerow(row)

    print()
    print(f"=== summary (scanned {processed} stems) ===")
    print(f"  only_face_label_mismatch : {len(hits)}")
    print(f"  label_identical          : {both_equal}")
    print(f"  label_plus_other_channel : {label_plus_others}")
    print(f"  topology_size_differs    : {topology_diff}")
    print(f"  errors                   : {errors}")
    print(f"  --> {txt_path}")
    print(f"  --> {csv_path}")


if __name__ == "__main__":
    main()
