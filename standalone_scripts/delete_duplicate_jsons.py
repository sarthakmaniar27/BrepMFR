"""
Clean C:\\jsons:
  1) Delete leftover SolidWorks temp *.SLDPRT files
  2) Keep one JSON per STEP key (usually *_101); delete extra multi-body JSONs

Example:
  00000001_..._step_000_101.json  <- KEEP (lexicographically first)
  00000001_..._step_000_102.json  <- delete
  00000001_..._step_000.SLDPRT    <- delete

Key used for grouping:
  00000001_..._step_000
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

JSON_FOLDER = Path(r"C:\jsons")

# Keep True for the first run. Set False only after checking the counts.
DRY_RUN = False

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)


def extract_key(filename: str) -> str | None:
    match = KEY_PATTERN.match(Path(filename).stem)
    if not match:
        return None
    return match.group("key").lower()


def list_files(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    results: list[Path] = []
    for name in os.listdir(folder):
        lower = name.lower()
        if any(lower.endswith(sfx) for sfx in suffixes):
            results.append(folder / name)
    return results


def delete_paths(paths: list[Path], label: str) -> tuple[int, int]:
    deleted = 0
    failed = 0
    for path in paths:
        try:
            path.unlink()
            deleted += 1
        except OSError as error:
            failed += 1
            print(f"FAILED ({label}): {path} -- {error}")
    return deleted, failed


def main() -> int:
    if not JSON_FOLDER.exists():
        print(f"ERROR: JSON folder is not accessible: {JSON_FOLDER}")
        return 1

    json_files = list_files(JSON_FOLDER, (".json",))
    sldprt_files = list_files(JSON_FOLDER, (".sldprt",))

    by_key: dict[str, list[Path]] = defaultdict(list)
    invalid_names = 0

    for json_file in json_files:
        key = extract_key(json_file.name)
        if key:
            by_key[key].append(json_file)
        else:
            invalid_names += 1

    keep_files: list[Path] = []
    delete_json_files: list[Path] = []

    for key in sorted(by_key):
        group = sorted(by_key[key], key=lambda p: p.name.lower())
        keep_files.append(group[0])
        delete_json_files.extend(group[1:])

    print(f"JSON folder:                        {JSON_FOLDER}")
    print(f"SLDPRT temp files found:            {len(sldprt_files)}")
    print(f"JSON files found:                   {len(json_files)}")
    print(f"Unique STEP keys:                   {len(by_key)}")
    print(f"JSON files to KEEP:                 {len(keep_files)}")
    print(f"Extra body JSONs to DELETE:         {len(delete_json_files)}")
    print(f"JSON files remaining afterward:     {len(keep_files) + invalid_names}")

    if invalid_names:
        print(f"JSON files with unexpected names:   {invalid_names} (left untouched)")

    print("\nKeep rule: for each STEP key, keep the lexicographically first JSON")
    print("(usually *_101.json). All *.SLDPRT temps are deleted.")

    if DRY_RUN:
        print("\nDRY RUN: No files were deleted.")
        if sldprt_files:
            print(f"Would delete {len(sldprt_files)} SLDPRT file(s). Sample:")
            for path in sldprt_files[:10]:
                print(f"  DELETE {path.name}")
            if len(sldprt_files) > 10:
                print(f"  ... and {len(sldprt_files) - 10} more")
        print("Sample KEEP / DELETE JSON pairs (first 10 keys with extras):")
        shown = 0
        for key in sorted(by_key):
            group = sorted(by_key[key], key=lambda p: p.name.lower())
            if len(group) < 2:
                continue
            print(f"  KEEP   {group[0].name}")
            for dup in group[1:]:
                print(f"  DELETE {dup.name}")
            shown += 1
            if shown >= 10:
                break
        print("\nSet DRY_RUN = False to perform the deletion.")
        return 0

    sld_deleted, sld_failed = delete_paths(sldprt_files, "SLDPRT")
    json_deleted, json_failed = delete_paths(delete_json_files, "JSON")

    print(f"\nDeleted SLDPRT files:       {sld_deleted}")
    print(f"Failed SLDPRT deletions:    {sld_failed}")
    print(f"Deleted extra body JSONs:   {json_deleted}")
    print(f"Failed JSON deletions:      {json_failed}")
    print(f"JSON files left (approx):   {len(json_files) - json_deleted}")

    return 0 if (sld_failed + json_failed) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
