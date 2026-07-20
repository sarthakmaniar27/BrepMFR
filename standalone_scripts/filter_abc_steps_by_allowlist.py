#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy STEPs matching the no-Thread/Text allowlist into abc_steps_filtered.

On each of the 10 old machines, STEPs live in ``C:\\abc_steps``. This script
copies allowlisted files into ``C:\\abc_steps_filtered`` (same machine).

Modes:
  local   — run ON a machine: read C:\\abc_steps, write C:\\abc_steps_filtered
  remote  — run from a controller: use \\\\HOST\\c$\\abc_steps for each host
            (needs admin share access; Jenkins agents usually use ``local``)

Allowlist file (one STEP key per line, no extension), e.g.:
  00000014_5b1c2f8a8c6f40fdaae1e69d_step_000

Generate it with:
  python standalone_scripts/export_step_allowlist_from_inference.py
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)

DEFAULT_MACHINES = [
    "walswkqa19383",
    "walswkqa19382",
    "walswkqa19381",
    "walswkqa19380",
    "walswkqa19374",
    "walswkqa19437",
    "walswkqa19438",
    "walswkqa19439",
    "walswkqa19440",
    "walswkqa19441",
]

DEFAULT_ALLOWLIST = Path(r"C:\jsons\inference\allowed_step_keys.txt")


def extract_key(filename: str) -> str | None:
    match = KEY_PATTERN.match(Path(filename).stem)
    return match.group("key").lower() if match else None


def load_allowlist(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept bare key, or a filename/path
        key = extract_key(line) if ("_step_" in line.lower() or line.lower().endswith((".step", ".stp", ".json"))) else line.lower()
        if key:
            keys.add(key)
    return keys


def list_step_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for name in os.listdir(folder):
        lower = name.lower()
        if lower.endswith(".step") or lower.endswith(".stp"):
            out.append(folder / name)
    return out


def filter_one_folder(
    source: Path,
    dest: Path,
    allowlist: set[str],
    *,
    dry_run: bool,
    clear_dest: bool,
) -> dict[str, int]:
    steps = list_step_files(source)
    matched: list[Path] = []
    for step in steps:
        key = extract_key(step.name)
        if key and key in allowlist:
            matched.append(step)

    stats = {
        "source_steps": len(steps),
        "matched": len(matched),
        "copied": 0,
        "failed": 0,
    }

    print(f"  Source:  {source}  ({stats['source_steps']} STEP/STP)")
    print(f"  Dest:    {dest}")
    print(f"  Matched: {stats['matched']}")

    if dry_run:
        for step in matched[:15]:
            print(f"    would copy: {step.name}")
        if len(matched) > 15:
            print(f"    ... and {len(matched) - 15} more")
        return stats

    if clear_dest and dest.exists():
        for old in list_step_files(dest):
            old.unlink(missing_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    for step in matched:
        try:
            shutil.copy2(step, dest / step.name)
            stats["copied"] += 1
        except OSError as exc:
            stats["failed"] += 1
            print(f"    FAILED {step.name}: {exc}", file=sys.stderr)

    print(f"  Copied:  {stats['copied']}  failed={stats['failed']}")
    return stats


def unc_path(host: str, relative: str) -> Path:
    # //host/c$/abc_steps
    rel = relative.replace("\\", "/").lstrip("/")
    return Path(f"//{host}/c$/{rel}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("local", "remote"),
        default="local",
        help="local=this machine C:\\ paths; remote=\\\\host\\c$\\ for --machines",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="Text file of STEP keys (from export_step_allowlist_from_inference.py).",
    )
    parser.add_argument(
        "--source-rel",
        default="abc_steps",
        help="Folder name under C:\\ (default: abc_steps).",
    )
    parser.add_argument(
        "--dest-rel",
        default="abc_steps_filtered",
        help="Destination folder name under C:\\ (default: abc_steps_filtered).",
    )
    parser.add_argument(
        "--machines",
        nargs="*",
        default=DEFAULT_MACHINES,
        help="Hostnames for --mode remote.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matches only; do not copy.",
    )
    parser.add_argument(
        "--clear-dest",
        action="store_true",
        help="Delete existing STEP/STP in dest before copying.",
    )
    args = parser.parse_args()

    if not args.allowlist.is_file():
        print(f"ERROR: Allowlist not found: {args.allowlist}", file=sys.stderr)
        print(
            "Generate it with: python standalone_scripts/export_step_allowlist_from_inference.py",
            file=sys.stderr,
        )
        return 1

    allowlist = load_allowlist(args.allowlist)
    print(f"[INFO] Allowlist keys: {len(allowlist)}  ({args.allowlist})")
    print(f"[INFO] Mode: {args.mode}  dry_run={args.dry_run}")

    grand = {"source_steps": 0, "matched": 0, "copied": 0, "failed": 0}

    if args.mode == "local":
        source = Path(r"C:\\") / args.source_rel
        dest = Path(r"C:\\") / args.dest_rel
        if not source.is_dir():
            print(f"ERROR: Source not found: {source}", file=sys.stderr)
            return 1
        print(f"\n=== LOCAL {os.environ.get('COMPUTERNAME', 'this-pc')} ===")
        stats = filter_one_folder(
            source, dest, allowlist, dry_run=args.dry_run, clear_dest=args.clear_dest
        )
        for k, v in stats.items():
            grand[k] = grand.get(k, 0) + v
    else:
        for host in args.machines:
            source = unc_path(host, args.source_rel)
            dest = unc_path(host, args.dest_rel)
            print(f"\n=== REMOTE {host} ===")
            if not source.is_dir():
                print(f"  SKIP: cannot access {source}")
                continue
            stats = filter_one_folder(
                source, dest, allowlist, dry_run=args.dry_run, clear_dest=args.clear_dest
            )
            for k, v in stats.items():
                grand[k] = grand.get(k, 0) + v

    print("\n=== TOTAL ===")
    print(f"Source STEPs scanned : {grand['source_steps']}")
    print(f"Matched allowlist    : {grand['matched']}")
    print(f"Copied               : {grand['copied']}")
    print(f"Failed               : {grand['failed']}")
    return 0 if grand["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
