#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed stage2_done_keys.txt from already-generated synthetic JSONs.

Default source (already processed ~10k JSONs / ~3k STEP keys):
  D:\\thread_and_text\\abc_json   on GR-SW66464

Run once on the state machine (or any node that can see that folder) before
starting continuous distribute/cleanup jobs:

  python standalone_scripts/pipeline_dedup/seed_stage2_done_keys.py
  python standalone_scripts/pipeline_dedup/seed_stage2_done_keys.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from key_utils import (
    DEFAULT_DONE_JSON_DIR,
    DEFAULT_STATE_DIR,
    append_keys,
    keys_from_json_dir,
    load_keys,
    state_paths,
    write_keys,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_DONE_JSON_DIR,
        help="Folder of Stage-2 synthetic JSONs (default: D:\\thread_and_text\\abc_json).",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="Pipeline ledger folder (default: D:\\thread_and_text\\pipeline_state).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite stage2_done_keys.txt instead of merging.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.json_dir.is_dir():
        print(f"ERROR: JSON dir not found: {args.json_dir}", file=sys.stderr)
        return 1

    found = keys_from_json_dir(args.json_dir)
    paths = state_paths(args.state_dir)
    done_path = paths["stage2_done"]
    existing = load_keys(done_path)

    print(f"[INFO] JSON dir          : {args.json_dir}")
    print(f"[INFO] Unique STEP keys  : {len(found)}")
    print(f"[INFO] Existing done keys: {len(existing)}")
    print(f"[INFO] State file        : {done_path}")

    if args.dry_run:
        new = found if args.replace else (found - existing)
        print(f"[DRY-RUN] Would write {len(found if args.replace else existing | found)} keys "
              f"(new={len(new)})")
        return 0

    args.state_dir.mkdir(parents=True, exist_ok=True)
    if args.replace:
        n = write_keys(done_path, found)
        print(f"[OK] Replaced done ledger with {n} keys")
    else:
        added, total = append_keys(done_path, found)
        print(f"[OK] Merged: added={added} total={total}")

    # Ensure sibling ledgers exist (empty ok).
    for name, path in paths.items():
        if name == "stage2_done":
            continue
        if not path.exists():
            write_keys(path, set())
            print(f"[OK] Created empty ledger: {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
