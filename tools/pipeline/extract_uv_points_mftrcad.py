#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract UV-point JSON for MFTRCAD graphs (MFTReNet JSON layout).

Reads MFTRCAD ``graph_face_grid`` (per-face ``7 × 10 × 10`` channel-major) and
BrepMFR-style ``labels`` (or raw MFTR ``cls``). Uses NumPy only (no Torch) for grids,
writes the same top-level schema as
``extract_uv_points.py``: ``file``, ``bin_path``, ``label_path``, face counts, and
``faces[]`` with ``uv_grid`` shaped as ``[10][10][C]`` after inference.

**Smoke test**
  ``--limit 20`` processes the first N graph files (sorted). Default ``--root`` layout
  matches post-``mftrcad_sync_rename`` folders: ``mftrnet_graphs``, ``mftrnet_labels``, ``uv_json``.

Usage::

  python tools/pipeline/extract_uv_points_mftrcad.py --root Y:\\mftrcad_dataset --limit 5
  python tools/pipeline/extract_uv_points_mftrcad.py --root Y:\\mftrcad_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PIPE = Path(__file__).resolve().parent
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

import numpy as np

from mftrcad_sync_rename import mftrcad_cls_to_brepmfr_labels


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {path} | {e}")
        return None


def tensor_to_nested_list(arr: np.ndarray) -> Any:
    return arr.tolist()


def infer_uv_grid_mftrcad(uv: np.ndarray) -> Tuple[List, Dict[str, Any]]:
    """
    Interpret per-face MFTRCAD UV array as a 10×10 grid (100 samples × C channels).
    Mirrors ``extract_uv_points._infer_uv_grid`` with 5→10 and 25→100.
    """
    meta: Dict[str, Any] = {"original_shape": list(uv.shape)}
    t = uv
    if t.ndim == 3 and t.shape[0] == 10 and t.shape[1] == 10:
        meta["interpreted_as"] = "[10,10,C]"
        return tensor_to_nested_list(t), meta

    if t.ndim == 2:
        if t.shape[0] == 100:
            c = t.shape[1]
            meta["interpreted_as"] = "[100,C] -> [10,10,C]"
            return tensor_to_nested_list(t.reshape(10, 10, c)), meta
        if t.shape[1] == 100:
            c = t.shape[0]
            meta["interpreted_as"] = "[C,100] -> [100,C] -> [10,10,C]"
            t2 = np.ascontiguousarray(t.T)
            return tensor_to_nested_list(t2.reshape(10, 10, c)), meta

    if t.ndim == 3:
        if t.shape[1] == 10 and t.shape[2] == 10:
            meta["interpreted_as"] = "[C,10,10] -> [10,10,C]"
            t2 = np.transpose(t, (1, 2, 0))
            return tensor_to_nested_list(np.ascontiguousarray(t2)), meta
        if 100 in t.shape:
            meta["interpreted_as"] = "squeezed_fallback"
            t2 = np.squeeze(t)
            if isinstance(t2, np.ndarray) and t2.ndim >= 2:
                return infer_uv_grid_mftrcad(t2)

    meta["interpreted_as"] = "raw_fallback"
    return t.tolist() if isinstance(t, np.ndarray) else t, meta


def load_labels_list(
    label_path: Path,
    file_stem: str,
    num_faces: int,
    label_obj: Optional[Dict[str, Any]] = None,
) -> Optional[List[int]]:
    data = label_obj if label_obj is not None else _safe_read_json(label_path)
    if not data:
        return None
    try:
        conv = mftrcad_cls_to_brepmfr_labels(data, file_stem, num_faces)
        return [int(x) for x in conv["labels"]]
    except ValueError as e:
        print(f"[ERROR] {label_path}: {e}")
        return None


def extract_uv_points_mftrcad_one(
    graph_json_path: Path,
    label_json_path: Path,
) -> Optional[Dict[str, Any]]:
    graph_raw = _safe_read_json(graph_json_path)
    if not graph_raw:
        return None
    if not isinstance(graph_raw, list) or len(graph_raw) < 2:
        print(f"[ERROR] Expected [stem, obj] in {graph_json_path}")
        return None

    # Use filesystem stem so outputs match renamed files (body [0] may still hold an old name).
    stem = graph_json_path.stem

    obj = graph_raw[1]
    grids = obj.get("graph_face_grid")
    if not isinstance(grids, list):
        print(f"[ERROR] Missing graph_face_grid in {graph_json_path}")
        return None

    n_graph = int(obj.get("graph", {}).get("num_nodes", -1))
    if n_graph < 0:
        print(f"[ERROR] Missing graph.num_nodes in {graph_json_path}")
        return None
    if len(grids) != n_graph:
        print(
            f"[WARN] len(graph_face_grid)={len(grids)} != num_nodes={n_graph} in {graph_json_path.name}; "
            f"using min length."
        )

    num_faces = min(n_graph, len(grids))
    labels = load_labels_list(label_json_path, stem, num_faces)
    if labels is None:
        return None
    if len(labels) < num_faces:
        num_faces = len(labels)

    faces_out: List[Dict[str, Any]] = []
    for face_hi in range(num_faces):
        lab = labels[face_hi]
        if isinstance(lab, bool):
            lab = int(lab)
        if not isinstance(lab, (int, float)):
            continue
        if int(lab) == 0:
            continue

        face_grid = grids[face_hi]
        uv_arr = np.asarray(face_grid, dtype=np.float32)
        if uv_arr.ndim == 3 and uv_arr.shape[0] == 7 and uv_arr.shape[1] == 10 and uv_arr.shape[2] == 10:
            uv_arr = np.ascontiguousarray(np.transpose(uv_arr, (1, 2, 0)))

        uv_grid, meta = infer_uv_grid_mftrcad(uv_arr)

        faces_out.append(
            {
                "face_index": face_hi,
                "label": int(lab),
                "uv_grid": uv_grid,
                "uv_meta": meta,
            }
        )

    return {
        "file": stem,
        "bin_path": str(graph_json_path.resolve()),
        "label_path": str(label_json_path.resolve()),
        "num_faces_in_graph": int(n_graph),
        "num_labels_in_json": int(len(labels)),
        "num_labeled_faces": int(len(faces_out)),
        "faces": faces_out,
    }


def iter_graph_jobs(
    graphs_dir: Path,
    labels_dir: Path,
    limit: Optional[int],
) -> List[Tuple[Path, Path]]:
    graphs = sorted(graphs_dir.glob("*.json"))
    jobs: List[Tuple[Path, Path]] = []
    for g in graphs:
        if g.stem.endswith("_rel"):
            continue
        lab = labels_dir / f"{g.stem}.json"
        if not lab.is_file():
            print(f"[SKIP] missing label for graph {g.name}")
            continue
        jobs.append((g, lab))
        if limit is not None and len(jobs) >= limit:
            break
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "MFTRCAD → BrepMFR-compatible uv_json (10×10×7 UV lattice per face). "
            "Graph JSON + label JSON → uv_json/*.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/pipeline/extract_uv_points_mftrcad.py --root Y:\\mftrcad_dataset --limit 10\n"
            "  python tools/pipeline/extract_uv_points_mftrcad.py "
            "--graphs-dir Y:\\mftrcad_dataset\\mftrnet_graphs "
            "--labels-dir Y:\\mftrcad_dataset\\mftrnet_labels "
            "--out-dir Y:\\mftrcad_dataset\\uv_json\n"
        ),
    )
    ap.add_argument("--root", type=Path, default=Path(r"Y:\mftrcad_dataset"))
    ap.add_argument("--graphs-dir", type=Path, default=None)
    ap.add_argument("--labels-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--uv-subdir", type=str, default="uv_json")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N valid graph/label pairs (sorted by graph name).",
    )
    ap.add_argument(
        "--stats-labels",
        action="store_true",
        help="After run, print min/max label over all processed label JSON (expects BrepMFR or MFTR cls).",
    )
    args = ap.parse_args()

    graphs_dir = args.graphs_dir or (args.root / "mftrnet_graphs")
    labels_dir = args.labels_dir or (args.root / "mftrnet_labels")
    out_dir = args.out_dir or (args.root / args.uv_subdir)

    if not graphs_dir.is_dir():
        print(f"[ERROR] graphs directory missing: {graphs_dir}", file=sys.stderr)
        return 1
    if not labels_dir.is_dir():
        print(f"[ERROR] labels directory missing: {labels_dir}", file=sys.stderr)
        return 1

    jobs = iter_graph_jobs(graphs_dir, labels_dir, args.limit)
    if not jobs:
        print("[ERROR] No graph/label pairs found.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    gmin, gmax = None, None
    ok, skipped = 0, 0

    for g_path, l_path in jobs:
        if args.stats_labels:
            lbl_data = _safe_read_json(l_path)
            if lbl_data and isinstance(lbl_data, dict):
                vals: List[int] = []
                if "labels" in lbl_data and isinstance(lbl_data["labels"], list):
                    vals = [int(x) for x in lbl_data["labels"]]
                elif "cls" in lbl_data and isinstance(lbl_data["cls"], dict):
                    vals = [int(v) for v in lbl_data["cls"].values()]
                if vals:
                    lo, hi = min(vals), max(vals)
                    gmin = lo if gmin is None else min(gmin, lo)
                    gmax = hi if gmax is None else max(gmax, hi)

        result = extract_uv_points_mftrcad_one(g_path, l_path)
        if result is None:
            skipped += 1
            continue
        out_path = out_dir / f"{g_path.stem}.json"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"[OK] {out_path.name} | faces={result['num_labeled_faces']}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] write {out_path}: {e}")
            skipped += 1

    print(f"\nDONE. ok={ok}, skipped={skipped}, pairs={len(jobs)}, out_dir={out_dir}")
    if args.stats_labels and gmin is not None and gmax is not None:
        print(f"[stats] label min={gmin} max={gmax} (over label files touched)")
    return 0 if skipped == 0 or ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
