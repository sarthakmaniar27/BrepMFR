#!/usr/bin/env python3
"""
Stage a hash-sharded subset of .step / .stp files from a source folder into a
target folder. Used by the Jenkins pipeline so each VM works on a disjoint
slice of the master STEP collection.

The shard for a file is determined by:
    md5(lowercased_basename) % total_shards == shard_index

So given the same source folder and the same total_shards, each shard_index
gets a stable, non-overlapping subset, regardless of file order or which VM
runs which shard.

Hard-links are used by default (instant, no disk space cost) and falls back
to copy if hard-link fails (e.g. cross-volume).

Usage:
  python stage_shard.py ^
    --source \\\\fileserver\\steps_master ^
    --target C:\\ThreadRecognition\\STEPS ^
    --shard-index 0 ^
    --total-shards 4
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path


def shard_of(name: str, total: int) -> int:
    h = hashlib.md5(name.lower().encode("utf-8")).hexdigest()
    return int(h, 16) % total


def stage(source: Path, target: Path, shard_index: int, total_shards: int) -> int:
    target.mkdir(parents=True, exist_ok=True)
    n_staged = 0
    n_seen = 0
    for f in source.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".step", ".stp"):
            continue
        n_seen += 1
        if shard_of(f.name, total_shards) != shard_index:
            continue
        dest = target / f.name
        if dest.exists():
            continue
        try:
            os.link(f, dest)
        except OSError:
            shutil.copy2(f, dest)
        n_staged += 1

    print(f"Source files seen: {n_seen}")
    print(f"Staged for shard {shard_index}/{total_shards}: {n_staged}")
    return n_staged


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path,
                   help="Master folder containing all .step/.stp files (read-only).")
    p.add_argument("--target", required=True, type=Path,
                   help="Local STEPS folder this VM will actually point SW at.")
    p.add_argument("--shard-index", required=True, type=int)
    p.add_argument("--total-shards", required=True, type=int)
    args = p.parse_args()

    if args.total_shards < 1:
        print("total-shards must be >= 1", file=sys.stderr)
        return 2
    if not (0 <= args.shard_index < args.total_shards):
        print(f"shard-index must be in [0, {args.total_shards})", file=sys.stderr)
        return 2
    if not args.source.is_dir():
        print(f"Source not a directory: {args.source}", file=sys.stderr)
        return 2

    stage(args.source, args.target, args.shard_index, args.total_shards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
