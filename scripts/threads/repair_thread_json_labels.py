#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair SolidWorks thread-dataset JSON face labels.

- Replaces ``label: -1`` with ``0`` (stock), with confirmation or ``--yes-minus-one``.
- Optionally remaps thread id ``70`` → ``1`` for BrepMFR Stage 1 with ``--num_classes 2``
  (``--yes-remap-70`` or interactive prompt).

Expects top-level ``"faces"`` list; each face may have integer ``"label"``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_kw):
        return it


def _iter_json_files(json_dir: Path) -> list[Path]:
    direct = sorted(json_dir.glob("*.json"))
    return direct if direct else sorted(json_dir.rglob("*.json"))


def scan_labels(paths: list[Path]) -> tuple[Counter, int, int, int]:
    """Returns (counter, files_with_minus_one, total_minus_one_faces, files_with_label_70)."""
    counter: Counter = Counter()
    files_m1 = 0
    total_m1 = 0
    files_70 = 0
    for jp in tqdm(paths, desc="Scan JSON", unit="file"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skip {jp}: {e}", file=sys.stderr)
            continue
        fh_m1 = False
        fh_70 = False
        for face in data.get("faces") or []:
            if not isinstance(face, dict) or "label" not in face:
                continue
            try:
                v = int(face["label"])
            except (TypeError, ValueError):
                continue
            counter[v] += 1
            if v == -1:
                fh_m1 = True
                total_m1 += 1
            if v == 70:
                fh_70 = True
        if fh_m1:
            files_m1 += 1
        if fh_70:
            files_70 += 1
    return counter, files_m1, total_m1, files_70


def _apply_minus_one_to_zero(paths: list[Path]) -> int:
    changed = 0
    for jp in tqdm(paths, desc="Replace -1 to 0", unit="file"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skip {jp}: {e}", file=sys.stderr)
            continue
        dirty = False
        for face in data.get("faces") or []:
            if not isinstance(face, dict) or "label" not in face:
                continue
            try:
                if int(face["label"]) == -1:
                    face["label"] = 0
                    dirty = True
            except (TypeError, ValueError):
                pass
        if dirty:
            jp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            changed += 1
    return changed


def _apply_70_to_1(paths: list[Path]) -> int:
    changed = 0
    for jp in tqdm(paths, desc="Replace 70 to 1", unit="file"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skip {jp}: {e}", file=sys.stderr)
            continue
        dirty = False
        for face in data.get("faces") or []:
            if not isinstance(face, dict) or "label" not in face:
                continue
            try:
                if int(face["label"]) == 70:
                    face["label"] = 1
                    dirty = True
            except (TypeError, ValueError):
                pass
        if dirty:
            jp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            changed += 1
    return changed


def _prompt_yes_no(msg: str) -> bool:
    if not sys.stdin.isatty():
        print(
            f"{msg}\n(Non-interactive: pass --yes-minus-one / --yes-remap-70.)",
            file=sys.stderr,
        )
        return False
    ans = input(f"{msg} [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Repair thread JSON labels: -1 to 0, optional 70 to 1 for 2-class training."
    )
    ap.add_argument("--json-dir", type=Path, required=True, help="Folder of *.json")
    ap.add_argument("--dry-run", action="store_true", help="Print stats only; do not write.")
    ap.add_argument("--yes-minus-one", action="store_true", help="Apply -1 to 0 without prompting.")
    ap.add_argument("--yes-remap-70", action="store_true", help="Apply 70 to 1 without prompting.")
    ap.add_argument(
        "--only-remap-70",
        action="store_true",
        help="Skip -1 branch; only scan and optionally remap 70 to 1.",
    )
    args = ap.parse_args()
    root = args.json_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    paths = _iter_json_files(root)
    print(f"JSON directory: {root}")
    print(f"JSON files: {len(paths):,}")

    counter, files_m1, total_m1, files_70 = scan_labels(paths)

    print("\n--- Per-label face counts ---")
    for k in sorted(counter.keys()):
        print(f"  label {k:>4}: {counter[k]:>12,} faces")
    n_faces = sum(counter.values())
    print(f"  TOTAL faces (with integer label): {n_faces:,}")
    print(f"\nFiles with any -1: {files_m1:,}  |  total -1 faces: {total_m1:,}")
    print(f"Files with any 70: {files_70:,}  |  total 70 faces: {counter.get(70, 0):,}")

    if args.dry_run:
        print("\n[--dry-run] No files modified.")
        return

    if not args.only_remap_70 and files_m1 > 0:
        do = args.yes_minus_one or _prompt_yes_no(
            f"Replace all {total_m1:,} face label(s) -1 to 0 in {files_m1:,} file(s)?"
        )
        if do:
            n_mod = _apply_minus_one_to_zero(paths)
            print(f"\nModified {n_mod:,} file(s) (-1 to 0).")
            paths = _iter_json_files(root)
            counter, files_m1, total_m1, files_70 = scan_labels(paths)
            print(f"Re-scan: files with -1 remaining = {files_m1}")
        else:
            print("\nSkipped -1 to 0.")
    elif not args.only_remap_70:
        print("\nNo -1 labels; skipping -1 to 0.")

    n70 = counter.get(70, 0)
    if n70 > 0:
        do70 = args.yes_remap_70 or _prompt_yes_no(
            f"Remap label 70 to 1 ({n70:,} faces in {files_70:,} files) for --num_classes 2?"
        )
        if do70:
            n_mod = _apply_70_to_1(paths)
            print(f"\nModified {n_mod:,} file(s) (70 to 1).")
        else:
            print("\nSkipped 70 to 1. Stage 1 needs labels in {{0,1}} for num_classes=2.")
    else:
        print("\nNo label 70; 70 to 1 not needed.")


if __name__ == "__main__":
    main()
