#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge harvested Stage-2 keys from agents into the central done ledger.

Also optionally subtract done keys from pending + distributed so they never
get re-shipped.
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
    parser.add_argument(
        "--harvest-dir",
        type=Path,
        required=True,
        help="Folder containing harvested_*.txt files from agents.",
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--also-prune-ledgers",
        action="store_true",
        help="Remove done keys from pending + distributed ledgers.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = state_paths(args.state_dir)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    for p in paths.values():
        if not p.exists():
            write_keys(p, set())

    harvested: set[str] = set()
    if args.harvest_dir.is_dir():
        for path in sorted(args.harvest_dir.glob("harvested_*.txt")):
            harvested |= load_keys(path)
            print(f"[INFO] Loaded {path.name}: {len(load_keys(path))} keys")

    print(f"[INFO] Total harvested unique keys: {len(harvested)}")
    if args.dry_run:
        existing = load_keys(paths["stage2_done"])
        print(f"[DRY-RUN] Would add {len(harvested - existing)} new done keys")
        return 0

    added, total = append_keys(paths["stage2_done"], harvested)
    print(f"[OK] stage2_done += {added} (total={total})")

    if args.also_prune_ledgers and harvested:
        r1, t1 = remove_keys(paths["pending"], harvested)
        r2, t2 = remove_keys(paths["stage2_distributed"], harvested)
        print(f"[OK] pending removed {r1} (total={t1})")
        print(f"[OK] distributed removed {r2} (total={t2})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
