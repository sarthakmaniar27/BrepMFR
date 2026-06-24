#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-write JSON files with stable pretty-printing (indent=2).

Use this if labels were fixed earlier with a minified writer and editors now
show one long line. Does not change JSON data, only whitespace.

Example:

  conda run -n brep_mfr_pyg python scripts/threads/prettify_thread_json.py --json-dir D:/threads/json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_kw):
        return it


def _iter_json(json_dir: Path) -> list[Path]:
    g = sorted(json_dir.glob("*.json"))
    return g if g else sorted(json_dir.rglob("*.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pretty-print thread JSON (indent=2).")
    ap.add_argument("--json-dir", type=Path, required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files only; do not write.",
    )
    args = ap.parse_args()
    root = args.json_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    paths = _iter_json(root)
    if args.dry_run:
        print(f"Would reformat {len(paths):,} JSON file(s) under {root}")
        return

    n_ok = 0
    for jp in tqdm(paths, desc="Prettify JSON", unit="file"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skip {jp}: {e}", file=sys.stderr)
            continue
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        jp.write_text(text, encoding="utf-8", newline="\n")
        n_ok += 1
    print(f"Reformatted {n_ok:,} file(s).")


if __name__ == "__main__":
    main()
