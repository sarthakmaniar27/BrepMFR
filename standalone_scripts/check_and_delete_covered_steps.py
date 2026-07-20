#!/usr/bin/env python3
"""Check that C:\\jsons covers STEPs in C:\\abc_steps_not_in_allowlist, then delete covered STEPs.

Matching key (shared):
  00031969_..._step_000.step  <->  00031969_..._step_000_101.json
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

JSON_FOLDER = Path(r"C:\jsons")
STEP_FOLDER = Path(r"C:\abc_steps_not_in_allowlist")

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)


def extract_key(filename: str) -> str | None:
    match = KEY_PATTERN.match(Path(filename).stem)
    return match.group("key").lower() if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report coverage; do not delete STEPs.",
    )
    parser.add_argument(
        "--json-dir", type=Path, default=JSON_FOLDER
    )
    parser.add_argument(
        "--step-dir", type=Path, default=STEP_FOLDER
    )
    args = parser.parse_args()

    if not args.json_dir.is_dir():
        print(f"ERROR: JSON folder not found: {args.json_dir}", file=sys.stderr)
        return 1
    if not args.step_dir.is_dir():
        print(f"ERROR: STEP folder not found: {args.step_dir}", file=sys.stderr)
        return 1

    json_keys: set[str] = set()
    for name in os.listdir(args.json_dir):
        if name.lower().endswith(".json") and (args.json_dir / name).is_file():
            key = extract_key(name)
            if key:
                json_keys.add(key)

    step_files: list[Path] = []
    for name in os.listdir(args.step_dir):
        lower = name.lower()
        if lower.endswith(".step") or lower.endswith(".stp"):
            step_files.append(args.step_dir / name)

    covered: list[Path] = []
    missing: list[Path] = []
    invalid: list[Path] = []

    for step in sorted(step_files, key=lambda p: p.name.lower()):
        key = extract_key(step.name)
        if not key:
            invalid.append(step)
            continue
        if key in json_keys:
            covered.append(step)
        else:
            missing.append(step)

    total = len(step_files)
    print(f"JSON folder:   {args.json_dir}")
    print(f"STEP folder:   {args.step_dir}")
    print(f"Unique JSON keys: {len(json_keys)}")
    print()
    print(f"STEP files total:     {total}")
    print(f"Covered by JSON:      {len(covered)}")
    print(f"Missing JSON:         {len(missing)}")
    print(f"Invalid STEP names:   {len(invalid)}")

    if total > 0:
        pct = 100.0 * len(covered) / total
        print(f"Coverage:             {len(covered)} / {total} ({pct:.1f}%)")

    if missing:
        print("\nSTEPs still missing a JSON (sample):")
        for path in missing[:20]:
            print(f"  {path.name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if args.dry_run:
        print("\nDRY RUN: no files deleted.")
        print("Re-run without --dry-run to delete covered STEPs.")
        return 0 if not missing else 1

    deleted = 0
    failed = 0
    for path in covered:
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            failed += 1
            print(f"FAILED: {path.name} -- {exc}", file=sys.stderr)

    print(f"\nDeleted covered STEPs: {deleted}")
    print(f"Failed deletions:      {failed}")
    print(f"STEPs left in folder:  {len(missing) + len(invalid) + failed}")

    if missing:
        print("Folder NOT fully covered — left the missing STEPs in place.")
        return 1

    print("Fully covered — all matching STEPs deleted.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
