#!/usr/bin/env python3
"""Copy STEPs for JSON keys that are NOT in the no-Thread/Text allowlist.

Workflow:
  1. Load allowlist from allowed_step_keys_p1/p2/p3.txt (~2688 keys)
  2. Scan JSON folder for unique ..._step_NNN keys
  3. Take keys present in JSONs but NOT in allowlist (the \"JSON keys NOT in allowlist\")
  4. Copy matching .step/.stp from abc_steps into a local folder

Example:
  JSON:  00000001_..._step_000_both_v8_102.json
  Key:   00000001_..._step_000
  STEP:  \\\\GR-SW65551\\abc_steps\\00000001_..._step_000.step
  Dest:  C:\\abc_steps_not_in_allowlist\\00000001_..._step_000.step
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
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
DEFAULT_STEP_FOLDER = Path("//GR-SW65551/abc_steps")
DEFAULT_DEST_FOLDER = Path(r"C:\abc_steps_not_in_allowlist")

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)


def extract_key(name_or_path: str) -> str | None:
    stem = Path(name_or_path).stem
    match = KEY_PATTERN.match(stem)
    return match.group("key").lower() if match else None


def load_allowlist(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.is_file():
            print(f"[WARN] Missing: {path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key = extract_key(line) if "_step_" in line.lower() else line.lower()
            if key:
                keys.add(key)
        print(f"[INFO] Loaded allowlist: {path.name}")
    return keys


def collect_json_keys(json_folder: Path) -> set[str]:
    keys: set[str] = set()
    for name in os.listdir(json_folder):
        if not name.lower().endswith(".json"):
            continue
        if not (json_folder / name).is_file():
            continue
        key = extract_key(name)
        if key:
            keys.add(key)
    return keys


def index_step_files(step_folder: Path) -> dict[str, Path]:
    """Map STEP key -> first matching .step/.stp path."""
    by_key: dict[str, Path] = {}
    for name in os.listdir(step_folder):
        lower = name.lower()
        if not (lower.endswith(".step") or lower.endswith(".stp")):
            continue
        key = extract_key(name)
        if key and key not in by_key:
            by_key[key] = step_folder / name
    return by_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_FOLDER)
    parser.add_argument("--step-dir", type=Path, default=DEFAULT_STEP_FOLDER)
    parser.add_argument("--dest-dir", type=Path, default=DEFAULT_DEST_FOLDER)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts / samples; do not copy.",
    )
    parser.add_argument(
        "--clear-dest",
        action="store_true",
        help="Delete existing .step/.stp in dest before copying.",
    )
    args = parser.parse_args()

    parts = [p for p in ALLOWLIST_PARTS if p.is_file()]
    if not parts and ALLOWLIST_COMBINED.is_file():
        parts = [ALLOWLIST_COMBINED]
    if not parts:
        print("ERROR: No allowlist txt files found.", file=sys.stderr)
        return 1
    if not args.json_dir.is_dir():
        print(f"ERROR: JSON folder not found: {args.json_dir}", file=sys.stderr)
        return 1
    if not args.step_dir.is_dir():
        print(f"ERROR: STEP folder not found: {args.step_dir}", file=sys.stderr)
        return 1

    allowlist = load_allowlist(parts)
    json_keys = collect_json_keys(args.json_dir)
    not_in_allowlist = json_keys - allowlist

    print(f"[INFO] Allowlist keys:                 {len(allowlist)}")
    print(f"[INFO] Unique JSON STEP keys:          {len(json_keys)}")
    print(f"[INFO] JSON keys NOT in allowlist:     {len(not_in_allowlist)}")
    print(f"[INFO] STEP source:                    {args.step_dir}")
    print(f"[INFO] Local dest:                     {args.dest_dir}")

    step_index = index_step_files(args.step_dir)
    print(f"[INFO] STEP files indexed:             {len(step_index)}")

    to_copy: list[tuple[str, Path]] = []
    missing: list[str] = []
    for key in sorted(not_in_allowlist):
        path = step_index.get(key)
        if path is None:
            missing.append(key)
        else:
            to_copy.append((key, path))

    print()
    print("=== STATS ===")
    print(f"Keys to copy (not in allowlist):       {len(not_in_allowlist)}")
    print(f"Found in abc_steps (will copy):        {len(to_copy)}")
    print(f"Missing from abc_steps:                {len(missing)}")

    if args.dry_run:
        print("\nDRY RUN — no files copied. Sample:")
        for key, path in to_copy[:15]:
            print(f"  COPY {path.name}")
        if len(to_copy) > 15:
            print(f"  ... and {len(to_copy) - 15} more")
        if missing[:10]:
            print("Sample missing keys:")
            for key in missing[:10]:
                print(f"  MISS {key}")
        print("\nRe-run without --dry-run to copy.")
        return 0

    args.dest_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_dest:
        for name in os.listdir(args.dest_dir):
            lower = name.lower()
            if lower.endswith(".step") or lower.endswith(".stp"):
                (args.dest_dir / name).unlink(missing_ok=True)

    copied = 0
    failed = 0
    for key, src in to_copy:
        dest = args.dest_dir / src.name
        try:
            shutil.copy2(src, dest)
            copied += 1
        except OSError as exc:
            failed += 1
            print(f"FAILED: {src.name} -- {exc}", file=sys.stderr)

    report_dir = args.dest_dir / "_copy_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "copied_step_keys.txt").write_text(
        "\n".join(k for k, _ in to_copy) + ("\n" if to_copy else ""),
        encoding="utf-8",
    )
    (report_dir / "missing_step_keys.txt").write_text(
        "\n".join(missing) + ("\n" if missing else ""),
        encoding="utf-8",
    )

    print()
    print(f"Copied:  {copied}")
    print(f"Failed:  {failed}")
    print(f"Dest:    {args.dest_dir}")
    print(f"Reports: {report_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
