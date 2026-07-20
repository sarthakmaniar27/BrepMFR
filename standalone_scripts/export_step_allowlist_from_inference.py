#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a STEP allowlist from Stage-2 inference flags (no confident Thread/Text).

Reads:
  C:\\jsons\\inference\\no_confident_thread_or_text.txt
  (or the CSV with a ``json`` column)

Writes a clean allowlist of STEP stems (one per line), e.g.:
  00000014_5b1c2f8a8c6f40fdaae1e69d_step_000

Use that file in Jenkins / filter_abc_steps_by_allowlist.py — do NOT paste
thousands of names into the Jenkinsfile.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)

DEFAULT_TXT = Path(r"C:\jsons\inference\no_confident_thread_or_text.txt")
DEFAULT_CSV = Path(r"C:\jsons\inference\no_confident_thread_or_text.csv")
DEFAULT_OUT = Path(r"C:\jsons\inference\allowed_step_keys.txt")


def extract_key(name_or_path: str) -> str | None:
    stem = Path(name_or_path.strip().strip('"')).stem
    match = KEY_PATTERN.match(stem)
    return match.group("key").lower() if match else None


def keys_from_txt(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = extract_key(line)
        if key:
            keys.add(key)
    return keys


def keys_from_csv(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "json" not in reader.fieldnames:
            raise ValueError(f"CSV must have a 'json' column: {path}")
        for row in reader:
            key = extract_key(row["json"])
            if key:
                keys.add(key)
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt", type=Path, default=DEFAULT_TXT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    keys: set[str] = set()
    if args.txt.is_file():
        keys |= keys_from_txt(args.txt)
        print(f"[INFO] Loaded keys from TXT: {args.txt} ({args.txt.stat().st_size} bytes)")
    if args.csv.is_file():
        keys |= keys_from_csv(args.csv)
        print(f"[INFO] Loaded keys from CSV: {args.csv}")

    if not keys:
        print("ERROR: No keys found. Run Stage-2 inference first.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(keys)
    args.out.write_text("\n".join(sorted_keys) + "\n", encoding="utf-8")

    # Also refresh the 3-way split used by Jenkins / match scripts
    repo = Path(__file__).resolve().parent
    n = len(sorted_keys)
    n1 = (n + 2) // 3
    n2 = (n + 1) // 3
    chunks = [
        sorted_keys[:n1],
        sorted_keys[n1 : n1 + n2],
        sorted_keys[n1 + n2 :],
    ]
    for i, chunk in enumerate(chunks, start=1):
        part_path = repo / f"allowed_step_keys_p{i}.txt"
        part_path.write_text("\n".join(chunk) + ("\n" if chunk else ""), encoding="utf-8")
        print(f"[INFO] Wrote {part_path.name}: {len(chunk)} keys")

    # Keep combined copy in repo too
    (repo / "allowed_step_keys.txt").write_text(
        "\n".join(sorted_keys) + "\n", encoding="utf-8"
    )

    print(f"[INFO] Unique STEP keys (allowlist): {len(sorted_keys)}")
    print(f"[INFO] Wrote: {args.out}")
    print("Sample:")
    for key in sorted_keys[:5]:
        print(f"  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
