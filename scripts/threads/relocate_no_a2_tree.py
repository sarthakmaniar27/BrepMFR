#!/usr/bin/env python3
"""Copy an existing no_a2 dataset tree to another drive, then optionally delete the source.

Use this when D: cannot hold the remaining A1+A3 graphs. Lite graphs stay where they
are; only the no_a2 training tree moves. After a verified copy you can delete the
source tree (and leftover torch .tmp files) to free D:.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def _folder_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def _count_pt(root: Path) -> int:
    pyg = root / "pyg"
    if not pyg.is_dir():
        return 0
    return sum(1 for _ in pyg.glob("*.pt"))


def _cleanup_tmp(root: Path) -> int:
    pyg = root / "pyg"
    if not pyg.is_dir():
        return 0
    n = 0
    for tmp in pyg.glob("*.pt.*.tmp"):
        try:
            tmp.unlink()
            n += 1
        except OSError:
            pass
    return n


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    robocopy = shutil.which("robocopy")
    if robocopy:
        # 0-7 = success/partial; 8+ = failure.
        completed = subprocess.run(
            [
                robocopy,
                str(src),
                str(dst),
                "/E",
                "/MT:8",
                "/R:2",
                "/W:2",
                "/NFL",
                "/NDL",
                "/NP",
            ],
            check=False,
        )
        if completed.returncode >= 8:
            raise SystemExit(f"robocopy failed with exit code {completed.returncode}")
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Existing no_a2 root on D:")
    parser.add_argument("--dst", required=True, type=Path, help="New no_a2 root on C:")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=24.0,
        help="Require this much free space on the destination drive AFTER the copy "
        "(default 24 GB for the remaining ~16k A1+A3 graphs).",
    )
    parser.add_argument(
        "--delete-source-after",
        action="store_true",
        help="Delete the source tree only after the destination pyg count matches.",
    )
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    if src == dst:
        raise SystemExit("src and dst must differ")
    if not src.is_dir():
        raise SystemExit(f"source not found: {src}")

    src_pt = _count_pt(src)
    src_bytes = _folder_bytes(src)
    dst.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(dst).free
    n_tmp = _cleanup_tmp(src)
    print(f"src={src}")
    print(f"dst={dst}")
    print(f"src_pt={src_pt:,}  src_gb={_gb(src_bytes):.2f}")
    print(f"dst_free_before_gb={_gb(free_before):.2f}  removed_src_tmp={n_tmp}")
    needed = src_bytes + int(args.min_free_gb * 1024**3)
    if free_before < needed:
        raise SystemExit(
            f"C: does not have enough free space. Need about {_gb(needed):.1f} GB "
            f"({_gb(src_bytes):.1f} GB to copy the existing tree + {args.min_free_gb:.1f} GB "
            f"for the remaining upgrade). Free {_gb(free_before):.1f} GB now."
        )

    print("Copying no_a2 tree...", flush=True)
    _copy_tree(src, dst)
    dst_pt = _count_pt(dst)
    dst_bytes = _folder_bytes(dst)
    free_after = shutil.disk_usage(dst).free
    print(f"dst_pt={dst_pt:,}  dst_gb={_gb(dst_bytes):.2f}  dst_free_after_gb={_gb(free_after):.2f}")
    if dst_pt < src_pt:
        raise SystemExit(f"copy incomplete: dst has {dst_pt:,} .pt files, src has {src_pt:,}")
    for name in ("train.txt", "val.txt", "test.txt"):
        if not (dst / name).is_file():
            raise SystemExit(f"missing split list after copy: {dst / name}")

    if args.delete_source_after:
        print(f"Deleting source tree {src} ...", flush=True)
        shutil.rmtree(src)
        print("source deleted")
    else:
        print("Source kept. Re-run with --delete-source-after after you confirm dst_pt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
