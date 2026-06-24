#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print per-label face counts from thread JSON or from PyG ``.pt`` ``label_feature``."""
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


def _iter_json(json_dir: Path) -> list[Path]:
    g = sorted(json_dir.glob("*.json"))
    return g if g else sorted(json_dir.rglob("*.json"))


def count_json(json_dir: Path) -> Counter:
    c: Counter = Counter()
    for jp in tqdm(_iter_json(json_dir), desc="JSON", unit="file"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skip {jp}: {e}", file=sys.stderr)
            continue
        for face in data.get("faces") or []:
            if not isinstance(face, dict) or "label" not in face:
                continue
            try:
                c[int(face["label"])] += 1
            except (TypeError, ValueError):
                c["<non-int>"] += 1
    return c


def count_pyg(pyg_dir: Path) -> Counter:
    import torch

    c: Counter = Counter()
    paths = sorted(pyg_dir.rglob("*.pt"))
    for pp in tqdm(paths, desc=".pt", unit="file"):
        try:
            g = torch.load(pp, map_location="cpu", weights_only=False)
            lf = getattr(g, "label_feature", None)
            if lf is None:
                continue
            for v in lf.detach().cpu().numpy().ravel().tolist():
                c[int(v)] += 1
        except Exception as e:
            print(f"[WARN] skip {pp}: {e}", file=sys.stderr)
    return c


def _parse_group_spec(spec: str) -> dict[int, str]:
    """Parse ``0:stock,1:thread,2:text`` -> {0: 'stock', ...}."""
    out: dict[int, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --group segment (need id:name): {part!r}")
        lid_s, name = part.split(":", 1)
        out[int(lid_s.strip())] = name.strip()
    if not out:
        raise ValueError("Empty --group")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--json-dir", type=Path, help="Count face[].label in JSON")
    g.add_argument("--pyg-dir", type=Path, help="Count label_feature in .pt graphs")
    ap.add_argument(
        "--group",
        type=str,
        default=None,
        metavar="SPEC",
        help='Optional named buckets, e.g. "0:stock,1:thread,2:text" (sums from counter).',
    )
    args = ap.parse_args()

    if args.json_dir is not None:
        root = args.json_dir.resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        counter = count_json(root)
        label = str(root)
    else:
        root = args.pyg_dir.resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        counter = count_pyg(root)
        label = str(root)

    total = sum(counter.values())
    print(f"Source: {label}")
    print(f"Total labeled faces: {total:,}\n")

    def sort_key(k):
        if isinstance(k, str):
            return (1, k)
        return (0, k)

    for k in sorted(counter.keys(), key=sort_key):
        pct = 100.0 * counter[k] / total if total else 0.0
        print(f"  label {k!s:>8}: {counter[k]:>12,}  ({pct:5.2f}%)")

    if args.group:
        try:
            buckets = _parse_group_spec(args.group)
        except ValueError as e:
            raise SystemExit(f"--group: {e}") from e
        print("\n--- Grouped (--group) ---")
        for lid in sorted(buckets.keys()):
            n = int(counter.get(lid, 0))
            pct = 100.0 * n / total if total else 0.0
            print(f"  {buckets[lid]} ({lid}): {n:>12,}  ({pct:5.2f}%)")
        grouped_ids = set(buckets.keys())
        other = sum(
            int(counter[k]) for k in counter.keys() if isinstance(k, int) and k not in grouped_ids
        )
        if other:
            pct = 100.0 * other / total if total else 0.0
            print(f"  (other int labels): {other:>12,}  ({pct:5.2f}%)")

    stock = int(counter.get(0, 0))
    thread1 = int(counter.get(1, 0))
    thread70 = int(counter.get(70, 0))
    unk = int(counter.get(-1, 0))
    if stock or thread1 or thread70 or unk:
        print("\n--- Grouped ---")
        print(f"  stock (0):        {stock:>12,}")
        print(f"  thread (1):       {thread1:>12,}")
        print(f"  thread (70):      {thread70:>12,}  (remap to 1 before num_classes=2)")
        print(f"  unknown (-1):     {unk:>12,}  (remap to 0 before training)")


if __name__ == "__main__":
    main()
