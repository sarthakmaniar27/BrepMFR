#!/usr/bin/env python3
"""Match no-Thread/Text allowlist keys against JSONs in E:\\jsons_from_all_machines.

Allowlist (from Stage-2 inference): standalone_scripts/allowed_step_keys_p{1,2,3}.txt
JSON example with variants:
  00000001_..._step_000_both_v8_102.json
Matching key:
  00000001_..._step_000
"""

from __future__ import annotations

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
# Also accept the combined file if present
ALLOWLIST_COMBINED = REPO / "allowed_step_keys.txt"

JSON_FOLDER = Path(r"E:\jsons_from_all_machines")

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)


def extract_key(name_or_path: str) -> str | None:
    stem = Path(name_or_path).stem
    match = KEY_PATTERN.match(stem)
    return match.group("key").lower() if match else None


def load_allowlist(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.is_file():
            print(f"[WARN] Missing allowlist part: {path}")
            continue
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
    parts = [p for p in ALLOWLIST_PARTS if p.is_file()]
    if not parts and ALLOWLIST_COMBINED.is_file():
        parts = [ALLOWLIST_COMBINED]
    if not parts:
        print("ERROR: No allowlist txt files found in standalone_scripts/", file=sys.stderr)
        return 1

    if not JSON_FOLDER.is_dir():
        print(f"ERROR: JSON folder not found: {JSON_FOLDER}", file=sys.stderr)
        return 1

    allowlist = load_allowlist(parts)
    print(f"[INFO] Unique allowlist STEP keys: {len(allowlist)}")
    print(f"[INFO] JSON folder: {JSON_FOLDER}")

    json_names = [
        name
        for name in os.listdir(JSON_FOLDER)
        if name.lower().endswith(".json") and (JSON_FOLDER / name).is_file()
    ]

    json_keys: set[str] = set()
    json_files_per_key: dict[str, int] = {}
    invalid = 0

    for name in json_names:
        key = extract_key(name)
        if key:
            json_keys.add(key)
            json_files_per_key[key] = json_files_per_key.get(key, 0) + 1
        else:
            invalid += 1

    matched_keys = allowlist & json_keys
    allowlist_only = allowlist - json_keys
    json_only = json_keys - allowlist

    matched_json_files = sum(json_files_per_key[k] for k in matched_keys)

    print()
    print("=== STATS ===")
    print(f"JSON files in folder:                    {len(json_names)}")
    print(f"Unique STEP keys in JSON folder:         {len(json_keys)}")
    print(f"Allowlist STEP keys (no Thread/Text):    {len(allowlist)}")
    print()
    print(f"Matching STEP keys (in both):            {len(matched_keys)}")
    print(f"JSON files belonging to matched keys:    {matched_json_files}")
    print(f"Allowlist keys NOT found in JSON folder: {len(allowlist_only)}")
    print(f"JSON keys NOT in allowlist:              {len(json_only)}")
    if invalid:
        print(f"JSON files with unexpected names:        {invalid}")

    if allowlist:
        pct_keys = 100.0 * len(matched_keys) / len(allowlist)
        print()
        print(
            f"Coverage: {len(matched_keys)} / {len(allowlist)} allowlist keys "
            f"have >=1 JSON ({pct_keys:.1f}%)"
        )
    if json_names:
        pct_files = 100.0 * matched_json_files / len(json_names)
        print(
            f"Of folder: {matched_json_files} / {len(json_names)} JSON files "
            f"match an allowlist key ({pct_files:.1f}%)"
        )

    # Optional detail dumps next to the folder
    out_dir = JSON_FOLDER / "_match_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matched_step_keys.txt").write_text(
        "\n".join(sorted(matched_keys)) + ("\n" if matched_keys else ""),
        encoding="utf-8",
    )
    (out_dir / "allowlist_missing_from_json_folder.txt").write_text(
        "\n".join(sorted(allowlist_only)) + ("\n" if allowlist_only else ""),
        encoding="utf-8",
    )
    print(f"\n[INFO] Wrote reports under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
