#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append-copy allowlisted STEPs into C:\\abc_steps_filtered (never wipe).

Unlike filter_abc_steps_by_allowlist.py --clear-dest, this only ADDS missing
files so Stage-2 CLI work already in the folder is not disturbed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from key_utils import extract_key, list_step_files, load_keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(r"C:\abc_steps"))
    parser.add_argument("--dest", type=Path, default=Path(r"C:\abc_steps_filtered"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.allowlist.is_file():
        print(f"ERROR: Allowlist not found: {args.allowlist}", file=sys.stderr)
        return 1
    if not args.source.is_dir():
        print(f"ERROR: Source not found: {args.source}", file=sys.stderr)
        return 1

    allow = load_keys(args.allowlist)
    if not allow:
        print("[INFO] Empty allowlist chunk — nothing to copy.")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    existing = {extract_key(p.name) for p in list_step_files(args.dest)}
    existing.discard(None)

    copied = skipped_exists = missing = 0
    by_key: dict[str, Path] = {}
    for step in list_step_files(args.source):
        key = extract_key(step.name)
        if key and key in allow and key not in by_key:
            by_key[key] = step

    for key in sorted(allow):
        if key in existing:
            skipped_exists += 1
            continue
        src = by_key.get(key)
        if src is None:
            missing += 1
            print(f"  MISSING in source: {key}")
            continue
        dest_path = args.dest / src.name
        if args.dry_run:
            print(f"  would copy: {src.name}")
            copied += 1
            continue
        try:
            shutil.copy2(src, dest_path)
            copied += 1
        except OSError as exc:
            print(f"  FAILED {src.name}: {exc}", file=sys.stderr)
            return 2

    host = os.environ.get("COMPUTERNAME", "local")
    print(
        f"[OK] {host}: allow={len(allow)} copied={copied} "
        f"already_in_dest={skipped_exists} missing_source={missing} dest={args.dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
