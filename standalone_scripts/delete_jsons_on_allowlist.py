#!/usr/bin/env python3
"""Delete JSONs that are NOT on the no-Thread/Text allowlist.

KEEP:  files whose STEP key is in allowed_step_keys_p1/p2/p3.txt
       (no confident Thread/Text — the ~9464 files / matched keys)
DELETE: files whose STEP key is NOT in the allowlist
       (the other ~6785 — may have Thread/Text)

Matching uses ..._step_NNN only (ignores both_v8 / engrave / body suffixes).

Default folder: E:\\jsons_from_all_machines
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
ALLOWLIST_PARTS = [
    REPO / "allowed_step_keys_p1.txt",
    REPO / "allowed_step_keys_p2.txt",
    REPO / "allowed_step_keys_p3.txt",
]
ALLOWLIST_COMBINED = REPO / "allowed_step_keys.txt"
DEFAULT_JSON_FOLDER = Path(r"E:\jsons_from_all_machines")

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)


def extract_key(filename: str) -> str | None:
    match = KEY_PATTERN.match(Path(filename).stem)
    return match.group("key").lower() if match else None


def load_allowlist() -> set[str]:
    paths = [p for p in ALLOWLIST_PARTS if p.is_file()]
    if not paths and ALLOWLIST_COMBINED.is_file():
        paths = [ALLOWLIST_COMBINED]
    keys: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key = extract_key(line) if "_step_" in line.lower() else line.lower()
            if key:
                keys.add(key)
        print(f"[INFO] Loaded {path.name}")
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_FOLDER)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts only; do not delete.",
    )
    args = parser.parse_args()

    if not args.json_dir.is_dir():
        print(f"ERROR: JSON folder not found: {args.json_dir}", file=sys.stderr)
        return 1

    allowlist = load_allowlist()
    if not allowlist:
        print("ERROR: Empty allowlist.", file=sys.stderr)
        return 1

    to_delete: list[Path] = []  # NOT in allowlist
    to_keep: list[Path] = []  # IN allowlist (no Thread/Text)
    invalid = 0

    for name in os.listdir(args.json_dir):
        path = args.json_dir / name
        if not path.is_file() or not name.lower().endswith(".json"):
            continue
        key = extract_key(name)
        if not key:
            invalid += 1
            continue
        if key in allowlist:
            to_keep.append(path)
        else:
            to_delete.append(path)

    print(f"[INFO] JSON folder:                         {args.json_dir}")
    print(f"[INFO] Allowlist keys (no Thread/Text):     {len(allowlist)}")
    print(f"[INFO] KEEP  (key IN allowlist):            {len(to_keep)}")
    print(f"[INFO] DELETE (key NOT in allowlist):       {len(to_delete)}")
    print(f"[INFO] Invalid names:                       {invalid}")

    if args.dry_run:
        print("\nDRY RUN — sample files that would be DELETED (not in allowlist):")
        for path in sorted(to_delete, key=lambda p: p.name.lower())[:15]:
            print(f"  DELETE {path.name}")
        if len(to_delete) > 15:
            print(f"  ... and {len(to_delete) - 15} more")
        print("\nSample KEEP (in allowlist / no Thread/Text):")
        for path in sorted(to_keep, key=lambda p: p.name.lower())[:5]:
            print(f"  KEEP   {path.name}")
        return 0

    deleted = 0
    failed = 0
    for path in to_delete:
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            failed += 1
            print(f"FAILED: {path.name} -- {exc}", file=sys.stderr)

    print(f"\nDeleted (not in allowlist): {deleted}")
    print(f"Failed:                     {failed}")
    print(f"Kept (no Thread/Text):      {len(to_keep)}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
