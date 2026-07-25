#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Commit only keys that agents actually placed in abc_steps_filtered.

Reads success_*.txt files (one key per line) from --success-dir, then:
  - appends them to stage2_distributed_keys.txt
  - removes them from pending_keys.txt

Keys that were MISSING / timed out / offline stay in pending for the next wave.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from key_utils import (
    DEFAULT_STATE_DIR,
    append_keys,
    load_keys,
    remove_keys,
    state_paths,
    write_keys,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--success-dir",
        type=Path,
        required=True,
        help="Folder with success_*.txt from agents.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = state_paths(args.state_dir)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    for p in paths.values():
        if not p.exists():
            write_keys(p, set())

    shipped: set[str] = set()
    if args.success_dir.is_dir():
        for path in sorted(args.success_dir.glob("success_*.txt")):
            keys = load_keys(path)
            shipped |= keys
            print(f"[INFO] {path.name}: {len(keys)} keys")

    print(f"[INFO] Total successful unique keys: {len(shipped)}")
    if args.dry_run:
        print(f"[DRY-RUN] Would commit {len(shipped)} keys")
        return 0

    if not shipped:
        print("[WARN] No success_*.txt keys found — pending left unchanged.")
        return 0

    added, total_d = append_keys(paths["stage2_distributed"], shipped)
    removed, total_p = remove_keys(paths["pending"], shipped)
    print(f"[OK] Distributed ledger += {added} (total={total_d})")
    print(f"[OK] Pending removed {removed} (total pending={total_p})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
