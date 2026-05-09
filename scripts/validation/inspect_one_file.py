"""
Pick one .bin file (or random small one) and display all key attributes
side-by-side for OURS vs AUTHORS: every face row and every edge row.

Usage:
    python scripts/inspect_one_file.py 00057053
    python scripts/inspect_one_file.py --random --max-faces 15

Output: a formatted text table printed to stdout AND saved to
    scripts/sorted_dumps_full/inspect_{stem}.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from dgl.data.utils import load_graphs


DATASETS = {
   
    # "ours": Path(r"C:\Users\D58\Desktop\BrepMFR\test\bin"), 
    "ours": Path(r"Z:\Experiment6\source_dataset\output\bin"),
    "auth": Path(r"Z:\authors_data\bin"),
}


def stem_of(p: Path, side: str) -> str:
    name = p.stem
    if side == "ours" and name.endswith("_101"):
        return name[:-4]
    return name


def find_bin(stem: str, side: str) -> Path:
    folder = DATASETS[side]
    for p in folder.glob(f"{stem}*.bin"):
        if stem_of(p, side) == stem:
            return p
    raise FileNotFoundError(f"{stem} not found in {folder}")


def face_sort_order(y: np.ndarray, uv_mean: np.ndarray, fa: np.ndarray, fl: np.ndarray) -> np.ndarray:
    uv_avg = uv_mean.mean(axis=1) if uv_mean.size else np.zeros_like(y)
    return np.lexsort((fl.astype(np.float64), fa.astype(np.float64), uv_avg.astype(np.float64), y.astype(np.float64)))


def edge_sort_order(el: np.ndarray, uv_mean: np.ndarray, ec: np.ndarray, et: np.ndarray) -> np.ndarray:
    uv_avg = uv_mean.mean(axis=1) if uv_mean.size else np.zeros_like(el)
    return np.lexsort((et.astype(np.float64), ec.astype(np.float64), uv_avg.astype(np.float64), el.astype(np.float64)))


def load_side(stem: str, side: str) -> Dict:
    p = find_bin(stem, side)
    graphs, label_dict = load_graphs(str(p))
    g = graphs[0]
    n = int(g.num_nodes()); m = int(g.num_edges())

    y  = g.ndata["y"].cpu().float().numpy() if "y" in g.ndata else np.zeros(n, np.float32)
    z  = g.ndata["z"].cpu().long().numpy()  if "z" in g.ndata else np.zeros(n, np.int64)
    fl = g.ndata["l"].cpu().long().numpy()  if "l" in g.ndata else np.zeros(n, np.int64)
    fa = g.ndata["a"].cpu().long().numpy()  if "a" in g.ndata else np.zeros(n, np.int64)
    ff = g.ndata["f"].cpu().long().numpy()  if "f" in g.ndata else np.zeros(n, np.int64)
    fx = g.ndata["x"].cpu().float().numpy() if "x" in g.ndata else np.zeros((n, 10, 10, 7), np.float32)
    face_uv_mean = fx.mean(axis=(1, 2)) if fx.ndim == 4 else np.zeros((n, 7), np.float32)

    src, dst = g.edges()
    src = src.cpu().long().numpy(); dst = dst.cpu().long().numpy()
    et = g.edata["t"].cpu().long().numpy()  if "t" in g.edata else np.zeros(m, np.int64)
    el = g.edata["l"].cpu().float().numpy() if "l" in g.edata else np.zeros(m, np.float32)
    ec = g.edata["c"].cpu().long().numpy()  if "c" in g.edata else np.zeros(m, np.int64)
    ea = g.edata["a"].cpu().float().numpy() if "a" in g.edata else np.zeros(m, np.float32)
    ex = g.edata["x"].cpu().float().numpy() if "x" in g.edata else np.zeros((m, 10, 7), np.float32)
    edge_uv_mean = ex.mean(axis=1) if ex.ndim == 3 else np.zeros((m, 7), np.float32)

    face_pi = face_sort_order(y, face_uv_mean, fa, fl)
    edge_pi = edge_sort_order(el, edge_uv_mean, ec, et)

    return {
        "path": p, "n": n, "m": m,
        "y": y, "z": z, "fl": fl, "fa": fa, "ff": ff,
        "src": src, "dst": dst, "et": et, "el": el, "ec": ec, "ea": ea,
        "face_pi": face_pi, "edge_pi": edge_pi,
    }


FACE_TYPE_NAME = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 6: "other"}
EDGE_TYPE_NAME = {0: "line", 1: "circle", 2: "ellipse", 5: "bspline/other"}
EDGE_CONV_NAME = {0: "smooth", 1: "convex", 2: "concave"}
# CADSynth 25-class machining feature map
# (from remap_mfcadpp_labels_to_cadsynth.py, the authoritative project mapping)
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


def name_or_q(mapping: Dict[int, str], k: int) -> str:
    return mapping.get(k, f"?{k}")


def inspect(stem: str, out_dir: Path) -> None:
    ours = load_side(stem, "ours")
    auth = load_side(stem, "auth")

    lines: List[str] = []
    def P(s: str = ""):
        lines.append(s)

    P(f"==================== FILE: {stem} ====================")
    P(f"  ours path : {ours['path']}")
    P(f"  auth path : {auth['path']}")
    P(f"  ours: n_faces={ours['n']}  n_edges={ours['m']}")
    P(f"  auth: n_faces={auth['n']}  n_edges={auth['m']}")

    # ----------------------------- FACES --------------------------------------
    P()
    P("=== FACES (sorted by face_area -> uv_avg -> face_adj -> face_loop) ===")
    hdr = (f"  {'i':>2} | "
           f"{'OURS    area':>14} {'typ':>4} {'lop':>3} {'adj':>3} {'lbl':>3} {'label_name':<36} | "
           f"{'AUTH    area':>14} {'typ':>4} {'lop':>3} {'adj':>3} {'lbl':>3} {'label_name':<36} | "
           f"DIFF")
    P(hdr)
    P("  " + "-" * (len(hdr) - 2))
    mismatch_faces = 0
    N = max(ours["n"], auth["n"])
    for i in range(N):
        o_str = "(pad)"; a_str = "(pad)"
        diff_parts = []
        if i < ours["n"]:
            oi = int(ours["face_pi"][i])
            o_area = float(ours["y"][oi]); o_z = int(ours["z"][oi]); o_l = int(ours["fl"][oi])
            o_a = int(ours["fa"][oi]); o_f = int(ours["ff"][oi])
            o_str = (f"{o_area:14.5f} {o_z:4d} {o_l:3d} {o_a:3d} {o_f:3d} "
                     f"{name_or_q(FACE_LABEL_NAME, o_f):<36}")
        if i < auth["n"]:
            ai = int(auth["face_pi"][i])
            a_area = float(auth["y"][ai]); a_z = int(auth["z"][ai]); a_l = int(auth["fl"][ai])
            a_a = int(auth["fa"][ai]); a_f = int(auth["ff"][ai])
            a_str = (f"{a_area:14.5f} {a_z:4d} {a_l:3d} {a_a:3d} {a_f:3d} "
                     f"{name_or_q(FACE_LABEL_NAME, a_f):<36}")
        # diff
        if i < ours["n"] and i < auth["n"]:
            oi = int(ours["face_pi"][i]); ai = int(auth["face_pi"][i])
            if abs(float(ours["y"][oi]) - float(auth["y"][ai])) > 1e-4:
                diff_parts.append(f"d_area={float(ours['y'][oi]) - float(auth['y'][ai]):+.4f}")
            if int(ours["z"][oi]) != int(auth["z"][ai]):
                diff_parts.append(f"type {int(ours['z'][oi])}->{int(auth['z'][ai])} ({name_or_q(FACE_TYPE_NAME, int(ours['z'][oi]))}/{name_or_q(FACE_TYPE_NAME, int(auth['z'][ai]))})")
            if int(ours["ff"][oi]) != int(auth["ff"][ai]):
                diff_parts.append(f"label {int(ours['ff'][oi])}->{int(auth['ff'][ai])} ({name_or_q(FACE_LABEL_NAME, int(ours['ff'][oi]))}/{name_or_q(FACE_LABEL_NAME, int(auth['ff'][ai]))})")
            if int(ours["fa"][oi]) != int(auth["fa"][ai]):
                diff_parts.append(f"adj {int(ours['fa'][oi])}->{int(auth['fa'][ai])}")
            if int(ours["fl"][oi]) != int(auth["fl"][ai]):
                diff_parts.append(f"loop {int(ours['fl'][oi])}->{int(auth['fl'][ai])}")
        diff_str = "; ".join(diff_parts) if diff_parts else "MATCH"
        if diff_parts:
            mismatch_faces += 1
        P(f"  {i:>2} | {o_str} | {a_str} | {diff_str}")

    P(f"\n  face mismatches on sort-aligned rows: {mismatch_faces} / {N}")

    # ----------------------------- EDGES --------------------------------------
    P()
    P("=== EDGES (sorted by edge_length -> uv_avg -> edge_conv -> edge_type) ===")
    hdr = (f"  {'i':>3} | "
           f"{'OURS     length':>16} {'typ':>4} {'cnv':>4} {'dihed':>9} | "
           f"{'AUTH     length':>16} {'typ':>4} {'cnv':>4} {'dihed':>9} | "
           f"DIFF")
    P(hdr)
    P("  " + "-" * (len(hdr) - 2))
    mismatch_edges = 0
    M = max(ours["m"], auth["m"])
    for i in range(M):
        o_str = "(pad)"; a_str = "(pad)"
        diff_parts = []
        if i < ours["m"]:
            oi = int(ours["edge_pi"][i])
            o_str = f"{float(ours['el'][oi]):16.5f} {int(ours['et'][oi]):4d} {int(ours['ec'][oi]):4d} {float(ours['ea'][oi]):+9.4f}"
        if i < auth["m"]:
            ai = int(auth["edge_pi"][i])
            a_str = f"{float(auth['el'][ai]):16.5f} {int(auth['et'][ai]):4d} {int(auth['ec'][ai]):4d} {float(auth['ea'][ai]):+9.4f}"
        if i < ours["m"] and i < auth["m"]:
            oi = int(ours["edge_pi"][i]); ai = int(auth["edge_pi"][i])
            if abs(float(ours["el"][oi]) - float(auth["el"][ai])) > 1e-4:
                diff_parts.append(f"d_len={float(ours['el'][oi]) - float(auth['el'][ai]):+.4f}")
            if int(ours["et"][oi]) != int(auth["et"][ai]):
                diff_parts.append(f"type {int(ours['et'][oi])}->{int(auth['et'][ai])} ({name_or_q(EDGE_TYPE_NAME, int(ours['et'][oi]))}/{name_or_q(EDGE_TYPE_NAME, int(auth['et'][ai]))})")
            if int(ours["ec"][oi]) != int(auth["ec"][ai]):
                diff_parts.append(f"conv {int(ours['ec'][oi])}->{int(auth['ec'][ai])} ({name_or_q(EDGE_CONV_NAME, int(ours['ec'][oi]))}/{name_or_q(EDGE_CONV_NAME, int(auth['ec'][ai]))})")
            if abs(float(ours["ea"][oi]) - float(auth["ea"][ai])) > 1e-3:
                diff_parts.append(f"d_dihed={float(ours['ea'][oi]) - float(auth['ea'][ai]):+.4f}")
        diff_str = "; ".join(diff_parts) if diff_parts else "MATCH"
        if diff_parts:
            mismatch_edges += 1
        P(f"  {i:>3} | {o_str} | {a_str} | {diff_str}")

    P(f"\n  edge mismatches on sort-aligned rows: {mismatch_edges} / {M}")

    # Summary
    P()
    P("=== AGGREGATE (multisets, order-independent) ===")
    # face_type histogram
    def hist(a: np.ndarray) -> str:
        u, c = np.unique(a, return_counts=True)
        return ", ".join(f"{int(k)}:{int(v)}" for k, v in zip(u, c))
    P(f"  face_type   ours: {{{hist(ours['z'])}}}  auth: {{{hist(auth['z'])}}}")
    P(f"  face_loop   ours: {{{hist(ours['fl'])}}}  auth: {{{hist(auth['fl'])}}}")
    P(f"  face_adj    ours: {{{hist(ours['fa'])}}}  auth: {{{hist(auth['fa'])}}}")
    P(f"  face_label  ours: {{{hist(ours['ff'])}}}  auth: {{{hist(auth['ff'])}}}")
    P(f"  edge_type   ours: {{{hist(ours['et'])}}}  auth: {{{hist(auth['et'])}}}")
    P(f"  edge_conv   ours: {{{hist(ours['ec'])}}}  auth: {{{hist(auth['ec'])}}}")

    text = "\n".join(lines)
    print(text)
    out_path = out_dir / f"inspect_{stem}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"\n[out] {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?", default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--out-dir", default="scripts/sorted_dumps_full")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.random:
        import csv
        cands: List[str] = []
        with open(Path(args.out_dir) / "worst_files_full.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if 10 <= int(r["n_faces"]) <= 15 and 2 <= int(r["face_label_disagree"]) <= 5:
                    cands.append(r["file_stem"])
        stem = random.choice(cands) if cands else "00057053"
    else:
        stem = args.stem or "00057053"
    inspect(stem, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
