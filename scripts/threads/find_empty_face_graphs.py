#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List .pt graph files whose label_feature is empty (zero faces)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scan-root",
        type=Path,
        required=True,
        help="Directory to rglob for *[0-9].pt (e.g. D:/threads/lite/pyg)",
    )
    args = ap.parse_args()
    root = args.scan_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    paths = sorted(root.rglob("*[0-9].pt"))
    bad: list[Path] = []
    for p in tqdm(paths, desc="Scan .pt", unit="file"):
        try:
            g = torch.load(p, map_location="cpu", weights_only=False)
            lf = getattr(g, "label_feature", None)
            if lf is None or lf.numel() == 0:
                bad.append(p)
        except Exception as e:
            print(f"[WARN] {p}: {e}", file=sys.stderr)
    print(f"Scanned {len(paths):,} files; empty or missing label_feature: {len(bad):,}")
    for p in bad[:200]:
        print(p)
    if len(bad) > 200:
        print(f"... and {len(bad) - 200} more", file=sys.stderr)


if __name__ == "__main__":
    main()
