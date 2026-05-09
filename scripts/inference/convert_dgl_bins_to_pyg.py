#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DGL .bin graphs under Z:\\Experiment6 (read-only) to torch.save'd PyG Data (.pt)
under Z:\\Experiment6_PyG. Copies label JSON files next to outputs.

Delegates tensor layout to ``data.dgl_bin_to_pyg.bin_to_pyg`` (single source of truth with
parity checks vs ``json_to_brepmfr_pyg``).

Requires: conda env with dgl + torch matching your .bin producer (e.g. brep_mfr with torch 1.13).

Example:
  conda activate brep_mfr_pyg
  python scripts/inference/convert_dgl_bins_to_pyg.py \\
    --src-root Z:/Experiment6 \\
    --dst-root Z:/Experiment6_PyG
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.dgl_bin_to_pyg import bin_to_pyg  # noqa: E402


def mirror_json(src: Path, dst: Path):
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path(r"Z:\Experiment6"),
        help="Root containing source_dataset/ and target_dataset/",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path(r"Z:\Experiment6_PyG"),
        help="Parallel root; only this tree is written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan only; do not write files.",
    )
    args = parser.parse_args()

    bins = list(args.src_root.rglob("*.bin"))
    print(f"Found {len(bins)} .bin files under {args.src_root}")

    for bin_path in tqdm(bins):
        rel = bin_path.relative_to(args.src_root)
        out_pt = args.dst_root / rel.with_suffix(".pt")

        if args.dry_run:
            continue

        out_pt.parent.mkdir(parents=True, exist_ok=True)
        pyg = bin_to_pyg(bin_path)
        torch.save(pyg, out_pt)

    # Copy JSON sidecars under * / output / (matches Experiment6 layout; skips macro input/*.json)
    for json_path in tqdm(list(args.src_root.rglob("*.json")), desc="json copy"):
        if "output" not in json_path.parts:
            continue
        rel = json_path.relative_to(args.src_root)
        dst = args.dst_root / rel
        if args.dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, dst)

    for txt_path in tqdm(list(args.src_root.rglob("*.txt")), desc="txt copy"):
        rel = txt_path.relative_to(args.src_root)
        dst = args.dst_root / rel
        if args.dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(txt_path, dst)

    print("Done.")


if __name__ == "__main__":
    main()
