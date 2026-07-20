"""
Delete STEP files in abc_steps when a corresponding JSON exists in abc.

JSON folder (many variants per STEP):
  \\\\GR-SW26859\\abc
  e.g. 00000001_..._step_000_both_v8_102.json
       00000001_..._step_000_engrave_101.json
       00000001_..._step_000_thread_v5_104.json

STEP folder:
  \\\\GR-SW65551\\abc_steps
  e.g. 00000001_..._step_000.step

Matching key (ignores both/engrave/thread/version/body suffixes):
  00000001_..._step_000

Many JSONs can share one key; that still means only ONE STEP should be deleted.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

JSON_FOLDER = Path("//GR-SW26859/abc")
STEP_FOLDER = Path("//GR-SW65551/abc_steps")

# Keep True for the first run. Set False only after checking the counts.
DRY_RUN = False

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)


def extract_key(filename: str) -> str | None:
    """Extract shared STEP identity; strips both/engrave/thread/vN/body suffixes."""
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


def main() -> int:
    if not JSON_FOLDER.exists():
        print(f"ERROR: JSON folder is not accessible: {JSON_FOLDER}")
        return 1

    if not STEP_FOLDER.exists():
        print(f"ERROR: STEP folder is not accessible: {STEP_FOLDER}")
        return 1

    json_files = list_files(JSON_FOLDER, (".json",))
    step_files = list_files(STEP_FOLDER, (".step", ".stp"))

    json_keys: set[str] = set()
    json_files_per_key: dict[str, int] = {}
    invalid_json_names = 0

    for json_file in json_files:
        key = extract_key(json_file.name)
        if key:
            json_keys.add(key)
            json_files_per_key[key] = json_files_per_key.get(key, 0) + 1
        else:
            invalid_json_names += 1

    matching_step_files: list[Path] = []
    invalid_step_names = 0

    for step_file in step_files:
        key = extract_key(step_file.name)
        if not key:
            invalid_step_names += 1
            continue
        if key in json_keys:
            matching_step_files.append(step_file)

    multi_json_keys = sum(1 for count in json_files_per_key.values() if count > 1)
    avg_json_per_key = (len(json_files) / len(json_keys)) if json_keys else 0.0

    print(f"JSON folder:                        {JSON_FOLDER}")
    print(f"STEP folder:                        {STEP_FOLDER}")
    print(f"JSON files found:                   {len(json_files)}")
    print(f"Unique JSON->STEP keys:             {len(json_keys)}")
    print(f"JSON keys with multiple variants:   {multi_json_keys}")
    print(f"Avg JSON files per STEP key:        {avg_json_per_key:.2f}")
    print(f"Total STEP files found:             {len(step_files)}")
    print(f"Matching STEP files (to delete):    {len(matching_step_files)}")
    print(f"STEP files remaining afterward:     {len(step_files) - len(matching_step_files)}")
    print(
        "\nNote: variants like *_both_v8_102, *_engrave_101, *_thread_v5_104 "
        "all collapse to the same ..._step_NNN key."
    )

    if invalid_json_names:
        print(f"JSON files with unexpected names:   {invalid_json_names}")
    if invalid_step_names:
        print(f"STEP files with unexpected names:   {invalid_step_names}")

    if DRY_RUN:
        print("\nDRY RUN: No files were deleted.")
        print("Sample matching STEP files (first 30):")
        for step_file in matching_step_files[:30]:
            print(f"  {step_file.name}")
        if len(matching_step_files) > 30:
            print(f"  ... and {len(matching_step_files) - 30} more")
        print("\nSet DRY_RUN = False to perform the deletion.")
        return 0

    deleted_count = 0
    failed_count = 0

    for step_file in matching_step_files:
        try:
            step_file.unlink()
            deleted_count += 1
        except OSError as error:
            failed_count += 1
            print(f"FAILED: {step_file} -- {error}")

    print(f"\nDeleted STEP files: {deleted_count}")
    print(f"Failed deletions:   {failed_count}")
    print(f"STEP files left:    {len(step_files) - deleted_count}")

    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
