#!/usr/bin/env python3
"""Intersect prepared Stock-only JSON stems with a dataset split list."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def _split_stems(path: Path) -> list[str]:
    return [
        line.strip().removesuffix(".pt")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stock-manifest",
        required=True,
        type=Path,
        help="stock_label_manifest.csv from prepare_approved_abc_stock_jsons.py.",
    )
    parser.add_argument(
        "--split-file",
        required=True,
        type=Path,
        help="Combined dataset val.txt or test.txt.",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest = args.stock_manifest.resolve()
    split_file = args.split_file.resolve()
    out = args.out.resolve()
    if not manifest.is_file():
        raise SystemExit(f"Stock manifest not found: {manifest}")
    if not split_file.is_file():
        raise SystemExit(f"Dataset split not found: {split_file}")

    stock_stems: set[str] = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "output_json" not in (reader.fieldnames or []):
            raise SystemExit(f"Manifest has no output_json column: {manifest}")
        for row in reader:
            value = (row.get("output_json") or "").strip()
            if value:
                stock_stems.add(Path(value).stem)

    ordered_split = _split_stems(split_file)
    selected = [stem for stem in ordered_split if stem in stock_stems]
    if not selected:
        raise SystemExit(
            "No Stock-only manifest stems occur in the requested split. "
            "Check filenames and STEP-family split allocation."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + f".{os.getpid()}.tmp")
    temporary.write_text("".join(f"{stem}\n" for stem in selected), encoding="utf-8")
    os.replace(temporary, out)
    print(f"Stock manifest stems: {len(stock_stems):,}")
    print(f"Input split stems:    {len(ordered_split):,}")
    print(f"Stock-only selected:  {len(selected):,}")
    print(f"Wrote:                {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
