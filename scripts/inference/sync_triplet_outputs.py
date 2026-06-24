#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align ``output/bin``, ``output/bin_skip_a2``, and ``output/label`` to the exact same stems.

**Authoritative inventory:** stems that appear under ``output/bin/*.pt``.

Steps (with ``--apply``):

1. Delete ``bin_skip_a2/*.pt`` and ``label/*.json`` whose stems are missing from ``bin/``.
2. For **every** ``bin/{stem}.pt``:
   - Write ``label/{stem}.json`` from tensor ``label_feature`` (canonical with full graphs).
   - Write ``bin_skip_a2/{stem}.pt``: load the bin graph in memory, **delete** dense A2 tensors
     (``d2_distance``, ``angle_distance``), set ``has_a2=False`` (and related flags), then save
     (never modifies files under ``bin/``).

``output/bin/*.pt`` are never altered by this script.

Dry-run prints counts only (no reads of every ``.``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm


def _stems(folder: Path, suffix: str) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.glob(f"*.{suffix}")}


def plan_sync(output_dir: Path) -> dict[str, object]:
    d_bin = output_dir / "bin"
    d_skip = output_dir / "bin_skip_a2"
    d_lbl = output_dir / "label"

    stems_bin = _stems(d_bin, "pt")
    stems_skip = _stems(d_skip, "pt")
    stems_lbl = _stems(d_lbl, "json")

    orphans_skip = sorted(stems_skip - stems_bin)
    orphans_lbl = sorted(stems_lbl - stems_bin)
    return {
        "stems_master": stems_bin,
        "stems_skip_before": stems_skip,
        "stems_lbl_before": stems_lbl,
        "n_orphans_skip": len(orphans_skip),
        "n_orphans_lbl": len(orphans_lbl),
        "orphans_skip_sample": orphans_skip[:5],
        "orphans_lbl_sample": orphans_lbl[:5],
    }


def execute_sync(output_dir: Path) -> None:
    d_bin = output_dir / "bin"
    d_skip = output_dir / "bin_skip_a2"
    d_lbl = output_dir / "label"

    if not d_bin.is_dir():
        raise FileNotFoundError(f"Missing output/bin: {d_bin}")

    plan = plan_sync(output_dir)
    stems_bin: set[str] = plan["stems_master"]
    orphans_skip = set(plan["stems_skip_before"]) - stems_bin
    orphans_lbl = set(plan["stems_lbl_before"]) - stems_bin

    d_skip.mkdir(parents=True, exist_ok=True)
    d_lbl.mkdir(parents=True, exist_ok=True)

    tag = output_dir.parent.name + "/output"
    for stem in tqdm(sorted(orphans_skip), desc=f"{tag} rm skip orphans", disable=len(orphans_skip) == 0):
        (d_skip / f"{stem}.pt").unlink(missing_ok=True)
    for stem in tqdm(sorted(orphans_lbl), desc=f"{tag} rm label orphans", disable=len(orphans_lbl) == 0):
        (d_lbl / f"{stem}.json").unlink(missing_ok=True)

    for stem in tqdm(sorted(stems_bin), desc=f"{tag} write skip+label"):
        bin_pt = d_bin / f"{stem}.pt"
        skip_pt = d_skip / f"{stem}.pt"
        lbl_pt = d_lbl / f"{stem}.json"

        g = torch.load(bin_pt, map_location="cpu", weights_only=False)
        lf = getattr(g, "label_feature", None)
        if lf is None:
            raise RuntimeError(f"{stem}: bin graph missing label_feature")

        labels_list = lf.view(-1).long().tolist()
        lbl_pt.write_text(
            json.dumps({"file_name": stem, "labels": labels_list}, separators=(",", ":")),
            encoding="utf-8",
        )

        if hasattr(g, "d2_distance"):
            delattr(g, "d2_distance")
        if hasattr(g, "angle_distance"):
            delattr(g, "angle_distance")
        g.has_a2 = False
        g.has_a1 = getattr(g, "has_a1", True)
        g.has_a3 = getattr(g, "has_a3", True)

        torch.save(g, skip_pt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "dataset_roots",
        nargs="+",
        type=Path,
        help="Roots like .../Experiment6_PyG/source_dataset (uses child output/)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Delete orphans and regenerate skip_a2 + label from bin (default dry-run)",
    )
    args = ap.parse_args()

    for root in args.dataset_roots:
        out_dir = root / "output"
        print(f"\n=== {root} ===")
        if not out_dir.is_dir():
            print(f"  [skip] no directory: {out_dir}")
            continue
        info = plan_sync(out_dir)
        print(f"  master |bin|: {len(info['stems_master'])}")
        print(f"  before |skip|: {len(info['stems_skip_before'])}  |label|: {len(info['stems_lbl_before'])}")
        print(f"  orphans skip: {info['n_orphans_skip']}  label: {info['n_orphans_lbl']}")
        print(f"  sample orphan skips: {info['orphans_skip_sample']}")
        print(f"  sample orphan labels: {info['orphans_lbl_sample']}")
        print(f"  after sync (expected): triplets = {len(info['stems_master'])} files each folder")

        if not args.apply:
            print("  (dry-run; pass --apply to execute)")
            continue

        execute_sync(out_dir)

        aft = plan_sync(out_dir)
        print(
            "  VERIFY |skip|:",
            len(aft["stems_skip_before"]),
            "|label|:",
            len(aft["stems_lbl_before"]),
            "|bin|:",
            len(aft["stems_master"]),
        )
        sym = (
            aft["stems_skip_before"]
            == aft["stems_lbl_before"]
            == aft["stems_master"]
        )
        if not sym:
            raise RuntimeError("Post-sync stem sets still differ")


if __name__ == "__main__":
    main()
