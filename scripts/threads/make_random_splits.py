#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write ``train.txt`` / ``val.txt`` / ``test.txt`` (one graph stem per line) for Stage 1.

Splits are **STEP-key aware**: all variants of the same part
(``..._step_000_101``, ``..._step_000_both_v8_102``, …) share key
``..._step_000`` and are assigned to the **same** split (no train/test leakage).

Optional ``--abc-json-dir``: stems that appear in that folder are tracked so at
least ``--abc-min-train-frac`` (default 0.8) of those stems land in **train**.
Remaining ABC groups go only to val/test (not back into train).

Example:

  python scripts/threads/make_random_splits.py ^
    --pyg-dir D:/thread_and_text/lite/pyg ^
    --out-dir D:/thread_and_text/lite ^
    --abc-json-dir D:/thread_and_text/abc_jsons ^
    --abc-min-train-frac 0.8
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

STEP_KEY_PATTERN = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)


def step_key(stem: str) -> str:
    """Group id for leakage control; falls back to full stem if no _step_NNN."""
    match = STEP_KEY_PATTERN.match(stem)
    return match.group("key").lower() if match else stem.lower()


def _stems(root: Path, kind: str) -> list[str]:
    if kind == "pt":
        paths = sorted(root.rglob("*.pt"))
    else:
        g = sorted(root.glob("*.json"))
        paths = g if g else sorted(root.rglob("*.json"))
    return sorted({p.stem for p in paths})


def _stems_from_json_dir(json_dir: Path) -> set[str]:
    paths = sorted(json_dir.glob("*.json"))
    if not paths:
        paths = sorted(json_dir.rglob("*.json"))
    return {p.stem for p in paths}


def _split_groups(
    groups: list[tuple[str, list[str]]],
    *,
    train_frac: float,
    val_frac: float,
    rng: random.Random,
    allow_train: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Split atomic groups into train/val/test by target stem fractions."""
    if not groups:
        return [], [], []

    order = list(groups)
    rng.shuffle(order)
    total_stems = sum(len(stems) for _, stems in order)
    if total_stems == 0:
        return [], [], []

    if allow_train:
        n_train_target = int(round(total_stems * train_frac))
        n_val_target = int(round(total_stems * val_frac))
        if total_stems >= 3:
            n_train_target = max(1, min(n_train_target, total_stems - 2))
            n_val_target = max(1, min(n_val_target, total_stems - n_train_target - 1))
        elif total_stems == 2:
            n_train_target = 1
            n_val_target = 0
        else:
            n_train_target = 1
            n_val_target = 0
    else:
        # val / test only (e.g. leftover ABC after train quota filled)
        n_train_target = 0
        rem_val_share = val_frac / max(val_frac + (1.0 - train_frac - val_frac), 1e-9)
        n_val_target = int(round(total_stems * rem_val_share))
        if total_stems >= 2:
            n_val_target = max(1, min(n_val_target, total_stems - 1))
        else:
            n_val_target = total_stems

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    n_train = n_val = 0

    for _, stems in order:
        n = len(stems)
        if allow_train and n_train < n_train_target:
            train.extend(stems)
            n_train += n
        elif n_val < n_val_target:
            val.extend(stems)
            n_val += n
        else:
            test.extend(stems)

    return train, val, test


def _uniq(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pyg-dir", type=Path, help="Directory containing .pt graphs")
    g.add_argument("--json-dir", type=Path, help="Directory containing .json (stems only)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to write train/val/test.txt")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--abc-json-dir",
        type=Path,
        default=None,
        help="Optional JSON folder whose stems must have >= --abc-min-train-frac in train.",
    )
    ap.add_argument(
        "--abc-min-train-frac",
        type=float,
        default=0.8,
        help="Minimum fraction of --abc-json-dir stems assigned to train (default 0.8).",
    )
    ap.add_argument(
        "--no-group-by-step",
        action="store_true",
        help="Disable STEP-key grouping (old leaky behavior).",
    )
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
    group_by_step = not args.no_group_by_step

    if group_by_step:
        groups_map: dict[str, list[str]] = defaultdict(list)
        for stem in stems:
            groups_map[step_key(stem)].append(stem)
        for key in groups_map:
            groups_map[key].sort()
    else:
        groups_map = {stem.lower(): [stem] for stem in stems}

    abc_stems: set[str] = set()
    if args.abc_json_dir is not None:
        abc_dir = args.abc_json_dir.resolve()
        if not abc_dir.is_dir():
            raise SystemExit(f"--abc-json-dir is not a directory: {abc_dir}")
        abc_stems = _stems_from_json_dir(abc_dir) & set(stems)
        print(f"ABC stems present in pool: {len(abc_stems):,}  (from {abc_dir})")

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    if abc_stems and group_by_step:
        abc_groups: list[tuple[str, list[str]]] = []
        other_groups: list[tuple[str, list[str]]] = []
        for key, gstems in groups_map.items():
            if any(s in abc_stems for s in gstems):
                abc_groups.append((key, gstems))
            else:
                other_groups.append((key, gstems))

        abc_only_count = len(abc_stems)
        min_abc_train = int(round(abc_only_count * args.abc_min_train_frac))
        if abc_only_count:
            min_abc_train = max(1, min(min_abc_train, abc_only_count))
        else:
            min_abc_train = 0

        rng.shuffle(abc_groups)
        abc_train_groups: list[tuple[str, list[str]]] = []
        abc_rest_groups: list[tuple[str, list[str]]] = []
        abc_train_stem_hits = 0
        for key, gstems in abc_groups:
            hits = sum(1 for s in gstems if s in abc_stems)
            if abc_train_stem_hits < min_abc_train:
                abc_train_groups.append((key, gstems))
                abc_train_stem_hits += hits
            else:
                abc_rest_groups.append((key, gstems))

        for _, gstems in abc_train_groups:
            train.extend(gstems)

        # Leftover ABC → val/test only (keeps train quota from ballooning past intent)
        t2, v2, te2 = _split_groups(
            abc_rest_groups,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            rng=rng,
            allow_train=False,
        )
        train.extend(t2)
        val.extend(v2)
        test.extend(te2)

        # Non-ABC groups: normal 80/10/10 among themselves
        t3, v3, te3 = _split_groups(
            other_groups,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            rng=rng,
            allow_train=True,
        )
        train.extend(t3)
        val.extend(v3)
        test.extend(te3)

        abc_in_train = sum(1 for s in train if s in abc_stems)
        print(
            f"ABC constraint: {abc_in_train}/{abc_only_count} abc stems in train "
            f"({(100.0 * abc_in_train / abc_only_count) if abc_only_count else 0:.1f}%; "
            f"target >= {100.0 * args.abc_min_train_frac:.0f}%)"
        )
        print(
            f"ABC groups -> train: {len(abc_train_groups)}, "
            f"ABC leftover -> val/test: {len(abc_rest_groups)}, "
            f"other groups: {len(other_groups)}"
        )
        if abc_only_count and abc_in_train < min_abc_train:
            print(
                f"[WARN] Could not reach ABC train quota (got {abc_in_train}, need {min_abc_train}). "
                "Large multi-variant groups may overshoot/undershoot."
            )
    else:
        train, val, test = _split_groups(
            list(groups_map.items()),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            rng=rng,
            allow_train=True,
        )
        if abc_stems and not group_by_step:
            print("[WARN] --abc-json-dir quota requires STEP grouping; use without --no-group-by-step")

    train, val, test = _uniq(train), _uniq(val), _uniq(test)

    if group_by_step:

        def keys_of(seq: list[str]) -> set[str]:
            return {step_key(s) for s in seq}

        leak_tv = keys_of(train) & keys_of(val)
        leak_tt = keys_of(train) & keys_of(test)
        leak_vt = keys_of(val) & keys_of(test)
        if leak_tv or leak_tt or leak_vt:
            raise SystemExit(
                f"STEP-key leakage detected: train∩val={len(leak_tv)} "
                f"train∩test={len(leak_tt)} val∩test={len(leak_vt)}"
            )

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, subset in ("train", train), ("val", val), ("test", test):
        p = out / f"{name}.txt"
        p.write_text("\n".join(subset) + ("\n" if subset else ""), encoding="utf-8")
        print(f"Wrote {p}  ({len(subset):,} stems)")

    n = len(stems)
    print(
        f"Total stems: {n:,}  (train={len(train)}, val={len(val)}, test={len(test)})  "
        f"groups={len(groups_map):,}  group_by_step={group_by_step}"
    )


if __name__ == "__main__":
    main()
