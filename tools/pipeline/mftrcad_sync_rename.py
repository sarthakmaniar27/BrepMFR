#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFTRCAD dataset sync: graph-first triplet alignment, remove *_rel.json and orphans,
two-phase sequential rename (0000001.step / .json / .json), and BrepMFR label JSON
conversion from MFTRCAD ``cls`` (same schema as ``json_to_brepmfr_pyg``).

**Smoke / safety**
  Default is dry-run (no file moves or deletes). Pass ``--apply`` to mutate disk.

**Backup**
  ``--apply`` overwrites label JSON content with ``{\"file_name\", \"labels\"}``.
  Archive ``mftrnet_labels`` (or the whole ``--root``) before running if you need raw MFTR keys.

Usage::

  python tools/pipeline/mftrcad_sync_rename.py --root Y:\\mftrcad_dataset
  python tools/pipeline/mftrcad_sync_rename.py --root Y:\\mftrcad_dataset --apply
  python tools/pipeline/mftrcad_sync_rename.py --root Y:\\mftrcad_dataset --apply --delete-rel-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _num_nodes_from_graph_json(graph_path: Path) -> int:
    data = _read_json(graph_path)
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Unexpected graph JSON layout (expected [stem, obj]): {graph_path}")
    obj = data[1]
    n = obj.get("graph", {}).get("num_nodes")
    if n is None:
        raise ValueError(f"Missing graph.num_nodes in {graph_path}")
    return int(n)


def _is_rel_label_path(path: Path) -> bool:
    return path.suffix.lower() == ".json" and path.stem.endswith("_rel")


def _graph_stems(graphs_dir: Path) -> Set[str]:
    stems: Set[str] = set()
    for p in graphs_dir.glob("*.json"):
        if _is_rel_label_path(p):
            continue
        stems.add(p.stem)
    return stems


def _step_stems(steps_dir: Path) -> Set[str]:
    stems: Set[str] = set()
    for suf in (".step", ".STEP", ".stp", ".STP"):
        for p in steps_dir.glob(f"*{suf}"):
            stems.add(p.stem)
    return stems


def _main_label_stems(labels_dir: Path) -> Set[str]:
    stems: Set[str] = set()
    for p in labels_dir.glob("*.json"):
        if _is_rel_label_path(p):
            continue
        stems.add(p.stem)
    return stems


def _rel_label_paths(labels_dir: Path) -> List[Path]:
    return [p for p in labels_dir.glob("*.json") if _is_rel_label_path(p)]


def mftrcad_cls_to_brepmfr_labels(
    label_obj: Dict[str, Any],
    file_stem: str,
    num_faces: int,
) -> Dict[str, Any]:
    """Build BrepMFR label payload from MFTRCAD ``cls`` or pass through existing ``labels``."""
    if "labels" in label_obj and isinstance(label_obj["labels"], list):
        labels = [int(x) for x in label_obj["labels"]]
        if len(labels) != num_faces:
            raise ValueError(
                f"labels length {len(labels)} != num_faces {num_faces} (file_name={file_stem})"
            )
        return {"file_name": file_stem, "labels": labels}

    cls = label_obj.get("cls")
    if not isinstance(cls, dict):
        raise ValueError(f"Need 'cls' dict or 'labels' list for {file_stem}")

    labels: List[int] = []
    missing: List[int] = []
    for i in range(num_faces):
        key = str(i)
        if key not in cls:
            missing.append(i)
        else:
            v = cls[key]
            labels.append(int(v))
    if missing:
        raise ValueError(
            f"cls missing keys for faces {missing[:16]}{'...' if len(missing) > 16 else ''} ({file_stem})"
        )
    return {"file_name": file_stem, "labels": labels}


def _write_json(path: Path, obj: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
        else:
            json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_mapping_csv(csv_path: Path, entries: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_stem", "new_id", "new_stem"])
        for e in entries:
            w.writerow([e["old_stem"], e["new_id"], e["new_stem"]])


def collect_plan(
    root: Path,
    steps_sub: str,
    labels_sub: str,
    graphs_sub: str,
) -> Tuple[List[str], List[Path], List[Path], List[Path], List[Path]]:
    """
    Returns:
      valid_stems_sorted,
      rel_paths_to_delete,
      orphan_graph_paths,
      orphan_step_paths,
      orphan_label_paths,
    """
    steps_dir = root / steps_sub
    labels_dir = root / labels_sub
    graphs_dir = root / graphs_sub

    print(f"[scan] listing *_rel.json in {labels_dir} ...", flush=True)
    rel_paths = _rel_label_paths(labels_dir)
    print(f"[scan] {len(rel_paths)} rel label files", flush=True)

    print(f"[scan] listing graph json in {graphs_dir} ...", flush=True)
    g_stems = _graph_stems(graphs_dir)
    print(f"[scan] {len(g_stems)} graph stems", flush=True)

    valid: List[str] = []
    invalid_graphs: List[Path] = []

    for stem in sorted(g_stems):
        gpath = graphs_dir / f"{stem}.json"
        lpath = labels_dir / f"{stem}.json"
        step_path = None
        for suf in (".step", ".STEP", ".stp", ".STP"):
            cand = steps_dir / f"{stem}{suf}"
            if cand.is_file():
                step_path = cand
                break
        if step_path is None or not lpath.is_file():
            invalid_graphs.append(gpath)
            continue
        valid.append(stem)

    print(f"[scan] valid triplets (pre-orphan)={len(valid)} invalid_graphs={len(invalid_graphs)}", flush=True)

    valid_set = set(valid)

    print(f"[scan] orphan steps in {steps_dir} ...", flush=True)
    orphan_steps: List[Path] = []
    for p in steps_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".step", ".stp"):
            continue
        if p.stem not in valid_set:
            orphan_steps.append(p)

    orphan_labels: List[Path] = []
    for p in labels_dir.glob("*.json"):
        if _is_rel_label_path(p):
            continue
        if p.stem not in valid_set:
            orphan_labels.append(p)

    return valid, rel_paths, invalid_graphs, orphan_steps, orphan_labels


def run_sync(
    root: Path,
    steps_sub: str,
    labels_sub: str,
    graphs_sub: str,
    *,
    id_width: int,
    apply: bool,
    delete_rel_only: bool,
    manifest_path: Path,
) -> int:
    steps_dir = root / steps_sub
    labels_dir = root / labels_sub
    graphs_dir = root / graphs_sub

    valid, rel_paths, orphan_graphs, orphan_steps, orphan_labels = collect_plan(
        root, steps_sub, labels_sub, graphs_sub
    )

    if delete_rel_only:
        rel_csv = manifest_path.with_name(f"{manifest_path.stem}_rel_delete.csv")
        rel_csv.parent.mkdir(parents=True, exist_ok=True)
        with rel_csv.open("w", encoding="utf-8", newline="") as rf:
            for p in rel_paths:
                rf.write(f"{p.resolve()}\n")
        manifest = {
            "root": str(root.resolve()),
            "mode": "delete_rel_only",
            "apply": apply,
            "count_rel": len(rel_paths),
            "rel_paths_csv": str(rel_csv.resolve()),
        }
        _write_json(manifest_path, manifest)
        print(f"[manifest] wrote {manifest_path} (+ {rel_csv.name})")
        print(f"[*rel.json] count={len(rel_paths)}")
        if not apply:
            print("[dry-run] no deletes performed. Pass --apply with --delete-rel-only to delete.")
            return 0
        for p in rel_paths:
            p.unlink(missing_ok=True)
        print(f"[OK] deleted {len(rel_paths)} *_rel.json files.")
        return 0

    new_entries = []
    for rank, old_stem in enumerate(valid):
        new_id = rank + 1
        new_stem = f"{new_id:0{id_width}d}"
        new_entries.append({"old_stem": old_stem, "new_id": new_id, "new_stem": new_stem})

    manifest: Dict[str, Any] = {
        "root": str(root.resolve()),
        "apply": apply,
        "counts": {
            "valid_triplets": len(valid),
            "rel_json_to_delete": len(rel_paths),
            "orphan_invalid_graphs": len(orphan_graphs),
            "orphan_steps": len(orphan_steps),
            "orphan_labels": len(orphan_labels),
        },
        "mapping_csv": str((manifest_path.parent / f"{manifest_path.stem}.csv").resolve()),
        "paths": {
            "steps": str(steps_dir),
            "labels": str(labels_dir),
            "graphs": str(graphs_dir),
        },
    }
    _write_json(manifest_path, manifest)
    _write_mapping_csv(manifest_path.parent / f"{manifest_path.stem}.csv", new_entries)
    print(f"[manifest] wrote {manifest_path} (+ {manifest_path.stem}.csv)")
    print(
        f"valid_triplets={len(valid)} | "
        f"del_rel={len(rel_paths)} | "
        f"orphan_graphs={len(orphan_graphs)} | "
        f"orphan_steps={len(orphan_steps)} | "
        f"orphan_labels={len(orphan_labels)}"
    )

    if not apply:
        print("\n[dry-run] No deletes or renames. Pass --apply to execute.")
        print("Smoke: inspect manifest counts; rerun with --apply when satisfied.")
        return 0

    # --- deletes ---
    for p in rel_paths:
        p.unlink(missing_ok=True)
    for p in orphan_graphs + orphan_steps + orphan_labels:
        p.unlink(missing_ok=True)

    # --- phase A: move valid triplets to temp-ranked names ---
    for rank, old_stem in enumerate(valid):
        tmp = f"__mftr_ph1_{rank:06d}"
        g_old = graphs_dir / f"{old_stem}.json"
        l_old = labels_dir / f"{old_stem}.json"
        step_old = None
        for suf in (".step", ".STEP", ".stp", ".STP"):
            cand = steps_dir / f"{old_stem}{suf}"
            if cand.is_file():
                step_old = cand
                step_suf = suf
                break
        if step_old is None or not g_old.is_file() or not l_old.is_file():
            print(f"[FATAL] missing triplet during phase A at stem {old_stem}", file=sys.stderr)
            return 1

        g_old.rename(graphs_dir / f"{tmp}.json")
        l_old.rename(labels_dir / f"{tmp}.json")
        step_old.rename(steps_dir / f"{tmp}{step_suf}")

    # --- phase B: final names + BrepMFR label rewrite ---
    for rank, old_stem in enumerate(valid):
        new_id = rank + 1
        new_stem = f"{new_id:0{id_width}d}"
        tmp = f"__mftr_ph1_{rank:06d}"
        g_tmp = graphs_dir / f"{tmp}.json"
        l_tmp = labels_dir / f"{tmp}.json"
        step_tmp = None
        step_suf_final = ".step"
        for suf in (".step", ".STEP", ".stp", ".STP"):
            cand = steps_dir / f"{tmp}{suf}"
            if cand.is_file():
                step_tmp = cand
                step_suf_final = suf
                break
        if step_tmp is None or not g_tmp.is_file() or not l_tmp.is_file():
            print(f"[FATAL] missing triplet during phase B at rank {rank}", file=sys.stderr)
            return 1

        num_faces = _num_nodes_from_graph_json(g_tmp)
        raw_lbl = _read_json(l_tmp)
        if not isinstance(raw_lbl, dict):
            raise ValueError(f"Label JSON must be an object: {l_tmp}")
        brep_lbl = mftrcad_cls_to_brepmfr_labels(raw_lbl, new_stem, num_faces)

        g_dst = graphs_dir / f"{new_stem}.json"
        l_dst = labels_dir / f"{new_stem}.json"
        s_dst = steps_dir / f"{new_stem}{step_suf_final.lower()}"

        g_tmp.rename(g_dst)
        step_tmp.rename(s_dst)
        _write_json(l_dst, brep_lbl)
        l_tmp.unlink(missing_ok=True)

    print(f"\n[OK] Renamed {len(valid)} triplets to {id_width}-digit IDs under {root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "MFTRCAD: align steps/labels/graphs, remove *_rel.json and orphans, "
            "sequential rename, and BrepMFR label conversion. Default: dry-run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Smoke workflow:\n"
            "  1) Run without --apply → inspect manifest counts.\n"
            "  2) Optional: --apply --delete-rel-only to drop *_rel.json first.\n"
            "  3) Run with --apply for full sync + rename.\n"
            "Then run extract_uv_points_mftrcad.py to fill uv_json/."
        ),
    )
    ap.add_argument("--root", type=Path, default=Path(r"Y:\mftrcad_dataset"))
    ap.add_argument("--steps-subdir", type=str, default="steps")
    ap.add_argument("--labels-subdir", type=str, default="mftrnet_labels")
    ap.add_argument("--graphs-subdir", type=str, default="mftrnet_graphs")
    ap.add_argument("--id-width", type=int, default=7, help="Zero-pad width for new IDs (default 7).")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletes and renames (default off = dry-run only).",
    )
    ap.add_argument(
        "--delete-rel-only",
        action="store_true",
        help="Only delete *_rel.json under labels (still requires --apply to delete).",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Summary JSON path; stem mapping is written beside it as <stem>.csv (default: <root>/mftrcad_sync_manifest.json).",
    )
    args = ap.parse_args()
    manifest = args.manifest or (args.root / "mftrcad_sync_manifest.json")

    if not args.root.is_dir():
        print(f"[ERROR] --root is not a directory: {args.root}", file=sys.stderr)
        return 1

    return run_sync(
        args.root,
        args.steps_subdir,
        args.labels_subdir,
        args.graphs_subdir,
        id_width=args.id_width,
        apply=bool(args.apply),
        delete_rel_only=bool(args.delete_rel_only),
        manifest_path=manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
