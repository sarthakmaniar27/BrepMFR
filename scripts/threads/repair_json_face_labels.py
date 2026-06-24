#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remap SolidWorks-style face ``label`` integers using a JSON map (data-driven).

Typical use: thread + text (3-class) via
``remap_maps/thread_text_sw_to_brep.json`` (-1/0/70/101 -> 0/1/2).

- ``--dry-run``: print per-label face counts and unknown labels; no writes.
- ``--yes-write``: apply map and write JSON with indent=2.
- ``--fail-on-unknown``: exit non-zero if any face label is not in the map.
- ``--allow-unmapped``: with ``--yes-write``, leave labels not in the map unchanged
  (default is to refuse writes when unmapped labels exist).
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


def load_remap(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Map JSON must be an object of old_label -> new_label")
    out: dict[int, int] = {}
    for k, v in data.items():
        out[int(k)] = int(v)
    return out


def scan_faces(
    paths: list[Path],
    remap: dict[int, int],
) -> tuple[Counter, Counter, int]:
    """
    Returns (raw_label_counter, unmapped_label_counter, total_faces_with_label).
    unmapped counts only faces whose label is not in remap.
    """
    raw: Counter = Counter()
    unmapped: Counter = Counter()
    total = 0
    for jp in tqdm(paths, desc="Scan JSON", unit="file"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skip {jp}: {e}", file=sys.stderr)
            continue
        for face in data.get("faces") or []:
            if not isinstance(face, dict) or "label" not in face:
                continue
            try:
                v = int(face["label"])
            except (TypeError, ValueError):
                continue
            total += 1
            raw[v] += 1
            if v not in remap:
                unmapped[v] += 1
    return raw, unmapped, total


def apply_remap(
    paths: list[Path],
    remap: dict[int, int],
    *,
    allow_unmapped: bool,
) -> tuple[int, int]:
    """Returns (files_modified, faces_rewritten)."""
    files_mod = 0
    faces_rw = 0
    for jp in tqdm(paths, desc="Remap + write", unit="file"):
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
                v = int(face["label"])
            except (TypeError, ValueError):
                continue
            if v in remap:
                nv = remap[v]
                if nv != v:
                    face["label"] = nv
                    dirty = True
                    faces_rw += 1
            elif not allow_unmapped:
                raise RuntimeError(
                    f"Internal error: unmapped label {v} in {jp} (should have been rejected earlier)"
                )
        if dirty:
            jp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            files_mod += 1
    return files_mod, faces_rw


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remap face label integers in B-rep JSON using a JSON map file."
    )
    ap.add_argument("--json-dir", type=Path, required=True, help="Folder of *.json")
    ap.add_argument(
        "--map-json",
        type=Path,
        required=True,
        help="JSON object: {\"old\": new, ...} (keys may be strings).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print label histogram and unmapped counts; do not write.",
    )
    ap.add_argument(
        "--yes-write",
        action="store_true",
        help="Apply remap and write files (indent=2).",
    )
    ap.add_argument(
        "--fail-on-unknown",
        action="store_true",
        help="Exit with code 1 if any face label is missing from the map.",
    )
    ap.add_argument(
        "--allow-unmapped",
        action="store_true",
        help="When writing, leave labels not in the map unchanged (use with care).",
    )
    args = ap.parse_args()

    if args.dry_run == args.yes_write:
        raise SystemExit("Specify exactly one of: --dry-run OR --yes-write")

    root = args.json_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    map_path = args.map_json.resolve()
    if not map_path.is_file():
        raise SystemExit(f"Map file not found: {map_path}")

    remap = load_remap(map_path)
    paths = _iter_json_files(root)
    print(f"JSON directory: {root}")
    print(f"Map file:       {map_path}")
    print(f"Remap entries:  {len(remap)}  ->  {dict(sorted(remap.items()))}")
    print(f"JSON files:     {len(paths):,}")

    raw, unmapped, n_faces = scan_faces(paths, remap)
    print(f"\nFaces with integer label: {n_faces:,}")
    print("\n--- Raw label counts ---")
    for k in sorted(raw.keys()):
        pct = 100.0 * raw[k] / n_faces if n_faces else 0.0
        print(f"  label {k:>6}: {raw[k]:>12,}  ({pct:5.2f}%)")

    n_unmapped_faces = sum(unmapped.values())
    if unmapped:
        print("\n--- Labels NOT in map (faces) ---")
        for k in sorted(unmapped.keys()):
            print(f"  label {k:>6}: {unmapped[k]:>12,} faces")

    if args.fail_on_unknown and n_unmapped_faces:
        print(
            f"\n[fail-on-unknown] {n_unmapped_faces:,} face(s) have labels outside the map.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.dry_run:
        print("\n[--dry-run] No files modified.")
        return

    if not args.allow_unmapped and n_unmapped_faces:
        print(
            f"\nRefusing write: {n_unmapped_faces:,} face(s) have labels not in the map.\n"
            "Fix the map JSON, clean labels, or pass --allow-unmapped (not recommended).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    n_mod, n_rw = apply_remap(paths, remap, allow_unmapped=bool(args.allow_unmapped))
    print(f"\nModified {n_mod:,} file(s); remapped {n_rw:,} face label value(s).")
    raw2, _, n2 = scan_faces(paths, remap)
    print(f"Re-scan faces: {n2:,}")
    print("\n--- Post-write label counts ---")
    for k in sorted(raw2.keys()):
        pct = 100.0 * raw2[k] / n2 if n2 else 0.0
        print(f"  label {k:>6}: {raw2[k]:>12,}  ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
