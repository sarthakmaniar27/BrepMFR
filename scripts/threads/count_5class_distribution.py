#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count per-class face distribution for the 5-class subset from JSON files.

Classes:
    0  : stock
    15 : chamfer
    24 : fillet
    70 : thread
    101: text
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_kw):
        return it


# ── class mapping ────────────────────────────────────────────────────
CLASS_MAP: dict[int, str] = {
    0: "stock",
    15: "chamfer",
    24: "fillet",
    70: "thread",
    101: "text",
}

JSON_DIR = Path(r"\\gr-sw36912\C\Threads\conversion\jsons")


def _iter_json(json_dir: Path) -> list[Path]:
    """Return sorted list of JSON files (flat first, then recursive)."""
    g = sorted(json_dir.glob("*.json"))
    return g if g else sorted(json_dir.rglob("*.json"))


def count_classes(json_dir: Path) -> tuple[Counter, int, int]:
    """Count faces per class label across all JSON files.

    Returns
    -------
    class_counter : Counter
        Counts for each of the 5 target classes.
    other_count : int
        Faces with labels outside the 5 target classes.
    total_files : int
        Number of JSON files processed.
    """
    class_counter: Counter = Counter({lid: 0 for lid in CLASS_MAP})
    other_count = 0
    files = _iter_json(json_dir)
    total_files = 0

    for jp in tqdm(files, desc="Scanning JSONs", unit="file"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skip {jp.name}: {e}", file=sys.stderr)
            continue
        total_files += 1
        for face in data.get("faces") or []:
            if not isinstance(face, dict) or "label" not in face:
                continue
            try:
                label = int(face["label"])
            except (TypeError, ValueError):
                other_count += 1
                continue
            if label in CLASS_MAP:
                class_counter[label] += 1
            else:
                other_count += 1

    return class_counter, other_count, total_files


def main() -> None:
    json_dir = JSON_DIR
    if not json_dir.is_dir():
        raise SystemExit(f"Not a directory: {json_dir}")

    print(f"Source : {json_dir}")
    print("Classes: {0: stock, 15: chamfer, 24: fillet, 70: thread, 101: text}\n")

    class_counter, other_count, total_files = count_classes(json_dir)

    total_faces = sum(class_counter.values()) + other_count
    target_faces = sum(class_counter.values())

    print(f"Files scanned      : {total_files:,}")
    print(f"Total faces        : {total_faces:,}")
    print(f"Target-class faces : {target_faces:,}")
    print(f"Other-class faces  : {other_count:,}")
    print()

    # ── per-class breakdown ──────────────────────────────────────────
    header = f"{'Class':>6}  {'Name':<10}  {'Count':>12}  {'% of Target':>12}  {'% of Total':>12}"
    print(header)
    print("-" * len(header))

    for lid in sorted(CLASS_MAP.keys()):
        n = class_counter[lid]
        pct_target = 100.0 * n / target_faces if target_faces else 0.0
        pct_total = 100.0 * n / total_faces if total_faces else 0.0
        print(
            f"{lid:>6}  {CLASS_MAP[lid]:<10}  {n:>12,}  {pct_target:>11.2f}%  {pct_total:>11.2f}%"
        )

    if other_count:
        pct_total = 100.0 * other_count / total_faces if total_faces else 0.0
        print(
            f"{'--':>6}  {'other':<10}  {other_count:>12,}  {'--':>12}  {pct_total:>11.2f}%"
        )


if __name__ == "__main__":
    main()
