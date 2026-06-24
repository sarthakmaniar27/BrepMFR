#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write ``train.txt`` / ``val.txt`` / ``test.txt`` (one graph stem per line) for Stage 1.

``segmentation.py`` / ``CADSynth`` resolve split lists under ``--dataset_path``,
``--dataset_path/output/``, or (if ``--dataset_path`` ends with ``pyg``) the parent folder.
Place ``*.pt`` graphs under ``--dataset_path`` or use ``--pt_subdir`` (see segmentation CLI).

Example (splits next to ``pyg`` folder):

  python scripts/threads/make_random_splits.py --pyg-dir D:/threads/lite/pyg --out-dir D:/threads/lite
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def _stems(root: Path, kind: str) -> list[str]:
    if kind == "pt":
        paths = sorted(root.rglob("*.pt"))
    else:
        g = sorted(root.glob("*.json"))
        paths = g if g else sorted(root.rglob("*.json"))
    return sorted({p.stem for p in paths})


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pyg-dir", type=Path, help="Directory containing .pt graphs")
    g.add_argument("--json-dir", type=Path, help="Directory containing .json (stems only)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to write train/val/test.txt")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.pyg_dir is not None:
        root = args.pyg_dir.resolve()
        kind = "pt"
    else:
        root = args.json_dir.resolve()
        kind = "json"
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    stems = _stems(root, kind)
    if not stems:
        raise SystemExit(f"No *.{kind} files under {root}")

    rng = random.Random(args.seed)
    rng.shuffle(stems)
    n = len(stems)
    n_train = max(1, min(int(round(n * args.train_frac)), n - 2))
    n_val = max(1, min(int(round(n * args.val_frac)), n - n_train - 1))
    train = stems[:n_train]
    val = stems[n_train : n_train + n_val]
    test = stems[n_train + n_val :]

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, subset in ("train", train), ("val", val), ("test", test):
        p = out / f"{name}.txt"
        p.write_text("\n".join(subset) + ("\n" if subset else ""), encoding="utf-8")
        print(f"Wrote {p}  ({len(subset):,} stems)")

    print(f"Total stems: {n:,}  (train={len(train)}, val={len(val)}, test={len(test)})")


if __name__ == "__main__":
    main()
