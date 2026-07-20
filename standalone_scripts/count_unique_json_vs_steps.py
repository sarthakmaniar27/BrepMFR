"""
Report unique STEP keys covered by JSONs in C:\\jsons vs STEPs in abc_steps.

JSON example:
  00000001_1ffb81a71e5b402e966b9341_step_000_101.json

STEP example:
  00000001_1ffb81a71e5b402e966b9341_step_000.step

Shared key:
  00000001_1ffb81a71e5b402e966b9341_step_000
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

JSON_FOLDER = Path(r"C:\jsons")
STEP_FOLDER = Path("//GR-SW65551/abc_steps")

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)


def extract_key(filename: str) -> str | None:
    match = KEY_PATTERN.match(Path(filename).stem)
    if not match:
        return None
    return match.group("key").lower()


def list_files(folder: Path, suffixes: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for name in os.listdir(folder):
        lower = name.lower()
        if any(lower.endswith(sfx) for sfx in suffixes):
            names.append(name)
    return names


def main() -> int:
    if not JSON_FOLDER.exists():
        print(f"ERROR: JSON folder is not accessible: {JSON_FOLDER}")
        return 1

    if not STEP_FOLDER.exists():
        print(f"ERROR: STEP folder is not accessible: {STEP_FOLDER}")
        return 1

    json_names = list_files(JSON_FOLDER, (".json",))
    step_names = list_files(STEP_FOLDER, (".step", ".stp"))

    json_keys: set[str] = set()
    json_files_per_key: dict[str, int] = {}
    invalid_json = 0

    for name in json_names:
        key = extract_key(name)
        if key:
            json_keys.add(key)
            json_files_per_key[key] = json_files_per_key.get(key, 0) + 1
        else:
            invalid_json += 1

    step_keys: set[str] = set()
    invalid_step = 0

    for name in step_names:
        key = extract_key(name)
        if key:
            step_keys.add(key)
        else:
            invalid_step += 1

    matched = json_keys & step_keys
    json_only = json_keys - step_keys
    step_only = step_keys - json_keys

    print(f"JSON folder:  {JSON_FOLDER}")
    print(f"STEP folder:  {STEP_FOLDER}")
    print()
    print(f"JSON files (total):                 {len(json_names)}")
    print(f"Unique STEP keys from JSONs:        {len(json_keys)}")
    print(f"STEP files (total):                 {len(step_names)}")
    print(f"Unique STEP keys from STEPs:        {len(step_keys)}")
    print()
    print(f"Keys present in BOTH (match):       {len(matched)}")
    print(f"JSON keys with NO matching STEP:    {len(json_only)}")
    print(f"STEP keys with NO matching JSON:    {len(step_only)}")
    print()
    print(f"Unique JSONs vs STEPs difference:   {len(json_keys) - len(step_keys):+d}")
    print(
        f"  (positive => more unique JSON keys than STEPs; "
        f"negative => fewer unique JSON keys than STEPs)"
    )
    print()
    print(f"STEPs already covered by JSONs:     {len(matched)}")
    print(f"STEPs still without a JSON:         {len(step_only)}")
    print(f"JSON keys whose STEP is missing:    {len(json_only)}")

    if json_keys:
        avg = len(json_names) / len(json_keys)
        multi = sum(1 for c in json_files_per_key.values() if c > 1)
        print()
        print(f"JSON keys with multiple files:      {multi}")
        print(f"Avg JSON files per unique key:      {avg:.2f}")

    if invalid_json:
        print(f"JSON files with unexpected names:   {invalid_json}")
    if invalid_step:
        print(f"STEP files with unexpected names:   {invalid_step}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
