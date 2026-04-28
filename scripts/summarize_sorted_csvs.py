"""
Read the six row-aligned CSVs written by dump_sorted_bins_to_csv.py and
produce a single, readable comparison summary of OUR regenerated CADSynth
bins vs the authors' originals.

Input files expected in --dir (default: scripts/sorted_dumps_full):
    ours_faces.csv / auth_faces.csv
    ours_edges.csv / auth_edges.csv
    ours_pairs.csv / auth_pairs.csv   (optional)

Zero-dep (pandas-free) streaming:
    Both CSVs for a table are opened and iterated in lockstep, row by row.
    This keeps memory tiny regardless of input size.

Outputs (to --out-dir, default same as --dir):
    comparison_summary.txt     human-readable report
    face_confusion_f.csv       confusion matrix for face labels
    face_confusion_z.csv       confusion matrix for face surface-types
    edge_confusion_t.csv       confusion matrix for edge types
    edge_confusion_c.csv       confusion matrix for edge convexity
    worst_files.csv            top-K files with largest per-file disagreement
    worst_files_full.csv       per-file stats for every file (for your own drill-down)

Usage:
    python scripts/summarize_sorted_csvs.py
    python scripts/summarize_sorted_csvs.py --dir scripts/sorted_dumps
    python scripts/summarize_sorted_csvs.py --topk 50
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Column groups                                                                #
# --------------------------------------------------------------------------- #

FACE_DISCRETE = ["face_type", "face_loop", "face_adj", "face_label"]
FACE_CONTINUOUS = ["face_area"]
FACE_UV_PREFIXES = ["face_uv_mean", "face_uv_std", "face_uv_min", "face_uv_max"]

EDGE_DISCRETE = ["edge_type", "edge_conv"]
EDGE_CONTINUOUS = ["edge_length", "edge_dihedral"]
EDGE_UV_PREFIXES = ["edge_uv_mean", "edge_uv_std", "edge_uv_min", "edge_uv_max"]

PAIR_DISCRETE = ["spatial_pos", "d2_argmax", "d2_nz", "ang_argmax", "ang_nz", "edges_path_nonzero"]
PAIR_CONTINUOUS = ["d2_sum", "d2_mean", "d2_std", "ang_sum", "ang_mean", "ang_std"]


def uv_channel_cols(prefix: str, num_channels: int = 7) -> List[str]:
    return [f"{prefix}_ch{i}" for i in range(num_channels)]


# --------------------------------------------------------------------------- #
# Streaming accumulator                                                       #
# --------------------------------------------------------------------------- #

class TableAccumulator:
    def __init__(
        self,
        name: str,
        discrete_cols: List[str],
        continuous_cols: List[str],
        uv_prefixes: List[str],
    ):
        self.name = name
        self.discrete_cols = discrete_cols
        self.continuous_cols = continuous_cols
        self.uv_prefixes = uv_prefixes
        self.uv_cols: List[str] = []
        for p in uv_prefixes:
            self.uv_cols.extend(uv_channel_cols(p))

        self.n_rows = 0

        # discrete
        self.discrete_equal: Dict[str, int] = {c: 0 for c in discrete_cols}
        self.discrete_confusion: Dict[str, Dict[Tuple[int, int], int]] = {c: defaultdict(int) for c in discrete_cols}
        self.discrete_hist_ours: Dict[str, Dict[int, int]] = {c: defaultdict(int) for c in discrete_cols}
        self.discrete_hist_auth: Dict[str, Dict[int, int]] = {c: defaultdict(int) for c in discrete_cols}

        # continuous
        self.cont_equal_tol: Dict[str, int] = {c: 0 for c in continuous_cols}
        self.cont_sum_abs_err: Dict[str, float] = {c: 0.0 for c in continuous_cols}
        self.cont_max_abs_err: Dict[str, float] = {c: 0.0 for c in continuous_cols}
        self.cont_sum_sq_err: Dict[str, float] = {c: 0.0 for c in continuous_cols}
        self.cont_range_ours: Dict[str, Tuple[float, float]] = {c: (math.inf, -math.inf) for c in continuous_cols}
        self.cont_range_auth: Dict[str, Tuple[float, float]] = {c: (math.inf, -math.inf) for c in continuous_cols}

        # UV per-channel
        self.uv_max_abs_err: Dict[str, float] = {c: 0.0 for c in self.uv_cols}
        self.uv_sum_abs_err: Dict[str, float] = {c: 0.0 for c in self.uv_cols}

        # per-file aggregation
        self.per_file: Dict[str, Dict[str, int]] = {}

        # cached col indices after we see the header
        self._idx_o: Dict[str, int] = {}
        self._idx_a: Dict[str, int] = {}
        self._file_stem_idx_o: int = -1
        self._file_stem_idx_a: int = -1

    def bind_headers(self, hdr_o: List[str], hdr_a: List[str]) -> None:
        self._idx_o = {c: i for i, c in enumerate(hdr_o)}
        self._idx_a = {c: i for i, c in enumerate(hdr_a)}
        self._file_stem_idx_o = self._idx_o.get("file_stem", -1)
        self._file_stem_idx_a = self._idx_a.get("file_stem", -1)

    def update_row(self, row_o: List[str], row_a: List[str]) -> None:
        self.n_rows += 1

        # file-stem per-file bucket
        stem = row_o[self._file_stem_idx_o] if self._file_stem_idx_o >= 0 else ""
        entry = self.per_file.setdefault(stem, {"n_rows": 0, "any_disagree": 0, "label_disagree": 0})
        entry["n_rows"] += 1

        any_disagree = False
        label_disagree = False

        # DISCRETE
        for c in self.discrete_cols:
            i_o = self._idx_o.get(c, -1)
            i_a = self._idx_a.get(c, -1)
            if i_o < 0 or i_a < 0:
                continue
            try:
                o = int(float(row_o[i_o]))
                a = int(float(row_a[i_a]))
            except ValueError:
                continue
            eq = (o == a)
            if eq:
                self.discrete_equal[c] += 1
            else:
                any_disagree = True
                if c == "face_label":
                    label_disagree = True
            self.discrete_hist_ours[c][o] += 1
            self.discrete_hist_auth[c][a] += 1
            self.discrete_confusion[c][(o, a)] += 1

        if any_disagree:
            entry["any_disagree"] += 1
        if label_disagree:
            entry["label_disagree"] += 1

        # CONTINUOUS
        for c in self.continuous_cols:
            i_o = self._idx_o.get(c, -1)
            i_a = self._idx_a.get(c, -1)
            if i_o < 0 or i_a < 0:
                continue
            try:
                o = float(row_o[i_o])
                a = float(row_a[i_a])
            except ValueError:
                continue
            d = o - a
            ad = abs(d)
            if ad <= 1e-5:
                self.cont_equal_tol[c] += 1
            self.cont_sum_abs_err[c] += ad
            self.cont_sum_sq_err[c] += d * d
            if ad > self.cont_max_abs_err[c]:
                self.cont_max_abs_err[c] = ad
            if o < self.cont_range_ours[c][0]:
                self.cont_range_ours[c] = (o, self.cont_range_ours[c][1])
            if o > self.cont_range_ours[c][1]:
                self.cont_range_ours[c] = (self.cont_range_ours[c][0], o)
            if a < self.cont_range_auth[c][0]:
                self.cont_range_auth[c] = (a, self.cont_range_auth[c][1])
            if a > self.cont_range_auth[c][1]:
                self.cont_range_auth[c] = (self.cont_range_auth[c][0], a)

        # UV per-channel
        for c in self.uv_cols:
            i_o = self._idx_o.get(c, -1)
            i_a = self._idx_a.get(c, -1)
            if i_o < 0 or i_a < 0:
                continue
            try:
                o = float(row_o[i_o])
                a = float(row_a[i_a])
            except ValueError:
                continue
            ad = abs(o - a)
            self.uv_sum_abs_err[c] += ad
            if ad > self.uv_max_abs_err[c]:
                self.uv_max_abs_err[c] = ad


# --------------------------------------------------------------------------- #
# Streaming reader                                                             #
# --------------------------------------------------------------------------- #

def process_table(
    name: str,
    path_o: Path,
    path_a: Path,
    discrete_cols: List[str],
    continuous_cols: List[str],
    uv_prefixes: List[str],
    progress_every: int,
) -> Optional[TableAccumulator]:
    if not path_o.exists() or not path_a.exists():
        print(f"[skip] {name}: missing {'ours' if not path_o.exists() else 'auth'} CSV")
        return None
    size_o = path_o.stat().st_size / 1e6
    size_a = path_a.stat().st_size / 1e6
    print(f"[{name}] streaming ours={size_o:,.1f} MB, auth={size_a:,.1f} MB", flush=True)

    acc = TableAccumulator(name, discrete_cols, continuous_cols, uv_prefixes)
    t0 = time.time()
    with open(path_o, newline="", encoding="utf-8") as fo, open(path_a, newline="", encoding="utf-8") as fa:
        r_o = csv.reader(fo)
        r_a = csv.reader(fa)
        hdr_o = next(r_o)
        hdr_a = next(r_a)
        if hdr_o != hdr_a:
            print(f"[{name}] WARNING: headers differ")
        acc.bind_headers(hdr_o, hdr_a)
        n = 0
        for row_o, row_a in zip(r_o, r_a):
            acc.update_row(row_o, row_a)
            n += 1
            if n % progress_every == 0:
                print(f"  [{name}] rows={n:,}  elapsed={time.time()-t0:.1f}s", flush=True)
        # detect dangling rows
        tail_o = sum(1 for _ in r_o)
        tail_a = sum(1 for _ in r_a)
        if tail_o or tail_a:
            print(f"  [{name}] WARNING: mismatched row counts — extra rows ours={tail_o} auth={tail_a}")
    print(f"[{name}] done: rows={acc.n_rows:,} in {time.time()-t0:.1f}s", flush=True)
    return acc


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #

def fmt_pct(x: float) -> str:
    return f"{x:6.2f}%"


def write_confusion_csv(path: Path, conf: Dict[Tuple[int, int], int]) -> None:
    classes = sorted({c for pair in conf for c in pair})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ours\\auth"] + [str(c) for c in classes] + ["row_total"])
        for r in classes:
            row: List = [r]
            total = 0
            for c in classes:
                v = conf.get((r, c), 0)
                row.append(v)
                total += v
            row.append(total)
            w.writerow(row)


def write_worst_files_csv(
    worst_path: Path,
    full_path: Path,
    faces: Optional[TableAccumulator],
    edges: Optional[TableAccumulator],
    topk: int,
) -> None:
    stems = set()
    if faces:
        stems |= set(faces.per_file.keys())
    if edges:
        stems |= set(edges.per_file.keys())
    rows = []
    for stem in sorted(stems):
        fr = faces.per_file.get(stem, {}) if faces else {}
        er = edges.per_file.get(stem, {}) if edges else {}
        n_faces = fr.get("n_rows", 0)
        n_edges = er.get("n_rows", 0)
        face_label_dis = fr.get("label_disagree", 0)
        face_any_dis = fr.get("any_disagree", 0)
        edge_any_dis = er.get("any_disagree", 0)
        rows.append({
            "file_stem": stem,
            "n_faces": n_faces,
            "face_any_disagree": face_any_dis,
            "face_label_disagree": face_label_dis,
            "face_label_err_rate": face_label_dis / max(1, n_faces),
            "face_any_err_rate": face_any_dis / max(1, n_faces),
            "n_edges": n_edges,
            "edge_any_disagree": edge_any_dis,
            "edge_any_err_rate": edge_any_dis / max(1, n_edges),
        })
    # full
    cols = list(rows[0].keys()) if rows else []
    with open(full_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # worst
    rows.sort(key=lambda r: r["face_label_disagree"], reverse=True)
    with open(worst_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows[:topk])


def pretty_report(
    faces: Optional[TableAccumulator],
    edges: Optional[TableAccumulator],
    pairs: Optional[TableAccumulator],
    out_path: Path,
) -> None:
    lines: List[str] = []

    def P(s: str = "") -> None:
        lines.append(s)

    def hr(char: str = "=", n: int = 120) -> None:
        P(char * n)

    hr("=")
    P("  SORTED CSV COMPARISON — OURS vs AUTHORS  (all row-aligned via deterministic sort)")
    P("  Sort keys: faces = (face_area -> uv_avg -> face_adj -> face_loop)")
    P("             edges = (edge_length -> uv_avg -> edge_conv -> edge_type)")
    hr("=")

    for name, acc in (("FACES", faces), ("EDGES", edges), ("PAIRS (A1 spatial_pos / A2 d2 / A3 angle)", pairs)):
        if acc is None:
            continue
        P()
        hr("-")
        P(f"  {name}  —  total rows: {acc.n_rows:,}  files: {len(acc.per_file):,}")
        hr("-")

        # discrete
        if acc.discrete_cols:
            P(f"  {'Discrete column':<20}{'agree':>10}{'agree_pct':>12}{'uniq_ours':>11}{'uniq_auth':>11}   ours_top5                          auth_top5")
            for c in acc.discrete_cols:
                agree = acc.discrete_equal[c]
                pct = 100.0 * agree / max(1, acc.n_rows)
                uo = len(acc.discrete_hist_ours[c])
                ua = len(acc.discrete_hist_auth[c])
                t5o = ", ".join(f"{k}:{v}" for k, v in Counter(acc.discrete_hist_ours[c]).most_common(5))
                t5a = ", ".join(f"{k}:{v}" for k, v in Counter(acc.discrete_hist_auth[c]).most_common(5))
                P(f"  {c:<20}{agree:>10,}{fmt_pct(pct):>12}{uo:>11}{ua:>11}   {t5o:<36}{t5a}")

        # continuous
        if acc.continuous_cols:
            P()
            P(f"  {'Continuous column':<20}{'within_1e-5':>14}{'MAE':>14}{'RMSE':>14}{'max_abs_err':>16}   range_ours              range_auth")
            for c in acc.continuous_cols:
                pct = 100.0 * acc.cont_equal_tol[c] / max(1, acc.n_rows)
                mae = acc.cont_sum_abs_err[c] / max(1, acc.n_rows)
                rmse = math.sqrt(acc.cont_sum_sq_err[c] / max(1, acc.n_rows))
                ro = acc.cont_range_ours[c]
                ra = acc.cont_range_auth[c]
                P(f"  {c:<20}{fmt_pct(pct):>14}{mae:>14.4g}{rmse:>14.4g}{acc.cont_max_abs_err[c]:>16.4g}   [{ro[0]:>9.3f},{ro[1]:>9.3f}]  [{ra[0]:>9.3f},{ra[1]:>9.3f}]")

        # UV
        if acc.uv_cols:
            P()
            P(f"  UV per-channel MAE and max_abs_err (aggregated across sorted rows)")
            P(f"    {'channel':<25}{'MAE':>14}{'max_abs_err':>16}")
            for c in acc.uv_cols:
                mae = acc.uv_sum_abs_err[c] / max(1, acc.n_rows)
                P(f"    {c:<25}{mae:>14.4g}{acc.uv_max_abs_err[c]:>16.4g}")

    hr("=")
    P("  NOTES")
    hr("=")
    P("  - 'agree' / 'agree_pct' compares row-i of OURS to row-i of AUTHORS AFTER the deterministic sort.")
    P("  - Large UV MAE reflects that SolidWorks and OpenCascade sample UV points at different parametric")
    P("    locations; this is expected. What matters is whether face_label, face_type, edge_type and edge_conv")
    P("    agree — those are pipeline decisions unrelated to sampling.")
    P("  - For face_area and edge_length, MAE should be close to zero if the same geometry is being read;")
    P("    a non-zero MAE means the sort keys (area / length) are NOT aligning identical geometric entities,")
    P("    which could indicate that our macro is emitting a different set of edges / faces.")
    hr("=")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"\n[out] wrote report to {out_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="scripts/sorted_dumps_full")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--progress-every", type=int, default=500_000)
    ap.add_argument("--topk", type=int, default=50, help="How many worst-offending files to keep in worst_files.csv")
    ap.add_argument("--skip-pairs", action="store_true", help="Skip pair-level stats (A1/A2/A3)")
    args = ap.parse_args()

    in_dir = Path(args.dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    faces = process_table(
        "faces", in_dir / "ours_faces.csv", in_dir / "auth_faces.csv",
        FACE_DISCRETE, FACE_CONTINUOUS, FACE_UV_PREFIXES, args.progress_every,
    )
    edges = process_table(
        "edges", in_dir / "ours_edges.csv", in_dir / "auth_edges.csv",
        EDGE_DISCRETE, EDGE_CONTINUOUS, EDGE_UV_PREFIXES, args.progress_every,
    )
    pairs = None
    if not args.skip_pairs:
        pairs = process_table(
            "pairs", in_dir / "ours_pairs.csv", in_dir / "auth_pairs.csv",
            PAIR_DISCRETE, PAIR_CONTINUOUS, [], args.progress_every,
        )

    if faces is not None:
        write_confusion_csv(out_dir / "face_confusion_label.csv", faces.discrete_confusion["face_label"])
        write_confusion_csv(out_dir / "face_confusion_type.csv", faces.discrete_confusion["face_type"])
        write_confusion_csv(out_dir / "face_confusion_loop.csv", faces.discrete_confusion["face_loop"])
        write_confusion_csv(out_dir / "face_confusion_adj.csv", faces.discrete_confusion["face_adj"])
    if edges is not None:
        write_confusion_csv(out_dir / "edge_confusion_type.csv", edges.discrete_confusion["edge_type"])
        write_confusion_csv(out_dir / "edge_confusion_conv.csv", edges.discrete_confusion["edge_conv"])

    write_worst_files_csv(out_dir / "worst_files.csv", out_dir / "worst_files_full.csv", faces, edges, args.topk)

    pretty_report(faces, edges, pairs, out_dir / "comparison_summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
