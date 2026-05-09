#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random JSON↔``.bin`` parity over **Experiment6** layout (reference ``.bin`` post-``append_angle``).

For each sampled stem, compares ``data.dgl_bin_to_pyg.bin_to_pyg`` vs
``json_to_brepmfr_pyg.build_pyg_from_json_path`` (same tensors as full migration path).

**Source:** ``<root>/source_dataset/input/*.json`` + ``.../output/bin/*.bin``.

**Target:** ``.../target_dataset/input/json_new_labels_cadsynth_label_indices/*.json`` +
``.../output/bin/*.bin`` (label indices must match ``ndata['f']`` in bins).

Requires ``dgl`` (e.g. ``conda activate brep_mfr``).

Example:

  conda activate brep_mfr
  python scripts/diagnostics/migration_random_parity_samples.py \\
    --experiment6 Z:/Experiment6 --n-each 10 --seed 42
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path


def _bootstrap_repo(script_path: Path) -> Path:
    for ancestor in script_path.parents:
        bst = ancestor / "bootstrap_path.py"
        if bst.is_file():
            spec = importlib.util.spec_from_file_location("__brepmfr_bootstrap", bst)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup(str(script_path))
    raise RuntimeError("bootstrap_path.py not found")


def _load_compare_two_pyg():
    here = Path(__file__).resolve().parent
    pt = here / "compare_pt_vs_pt.py"
    spec = importlib.util.spec_from_file_location("_cpt", pt)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.compare_two_pyg


def _paired_stems(json_dir: Path, bin_dir: Path) -> list[str]:
    if not json_dir.is_dir():
        raise FileNotFoundError(f"Missing json_dir: {json_dir}")
    if not bin_dir.is_dir():
        raise FileNotFoundError(f"Missing bin_dir: {bin_dir}")
    jstems = {p.stem for p in json_dir.glob("*.json")}
    bstems = {p.stem for p in bin_dir.glob("*.bin")}
    return sorted(jstems & bstems)


def _run_split(
    name: str,
    json_dir: Path,
    bin_dir: Path,
    rng: random.Random,
    n_each: int,
    rtol: float,
    atol: float,
    bin_to_pyg,
    build_pyg_from_json_path,
    compare_two_pyg,
) -> tuple[int, int]:
    stems = _paired_stems(json_dir, bin_dir)
    k = min(n_each, len(stems))
    if k == 0:
        print(f"[{name}] No paired JSON+.bin stems under\n  {json_dir}\n  {bin_dir}")
        return 0, 0
    pick = rng.sample(stems, k) if k < len(stems) else list(stems)

    ok_count = 0
    fail_count = 0
    print(f"\n=== {name}: {k} sample(s) from {len(stems)} paired stems")
    for stem in pick:
        jp = json_dir / f"{stem}.json"
        bp = bin_dir / f"{stem}.bin"
        try:
            ref = bin_to_pyg(bp)
            cand = build_pyg_from_json_path(jp)
            errs = compare_two_pyg(ref, cand, rtol=rtol, atol=atol)
            if errs:
                fail_count += 1
                print(f"[FAIL] {name} {stem}")
                for e in errs[:25]:
                    print(f"       {e}")
                if len(errs) > 25:
                    print(f"       ... +{len(errs) - 25}")
            else:
                ok_count += 1
                print(f"[OK]   {name} {stem}")
        except Exception as exc:  # noqa: BLE001 — surface conversion bugs
            fail_count += 1
            print(f"[FAIL] {name} {stem} (exception) {exc}")

    return ok_count, fail_count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiment6",
        type=Path,
        default=Path(r"Z:\Experiment6"),
        help="Experiment6 root containing source_dataset/ and target_dataset/",
    )
    ap.add_argument("--n-each", type=int, default=10, help="Samples per split (source and target)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    script = Path(__file__).resolve()
    _repo = _bootstrap_repo(script)

    _infer = _repo / "scripts" / "inference"
    if str(_infer) not in sys.path:
        sys.path.insert(0, str(_infer))

    from data.dgl_bin_to_pyg import bin_to_pyg  # noqa: E402

    import json_to_brepmfr_pyg as j2p  # noqa: E402

    compare_two_pyg = _load_compare_two_pyg()

    root = args.experiment6
    src_json = root / "source_dataset" / "input"
    src_bin = root / "source_dataset" / "output" / "bin"
    tgt_json = root / "target_dataset" / "input" / "json_new_labels_cadsynth_label_indices"
    tgt_bin = root / "target_dataset" / "output" / "bin"

    rng = random.Random(args.seed)
    print(f"Experiment6 root: {root}")
    print(f"n_each={args.n_each} seed={args.seed}")

    o1, f1 = _run_split(
        "source",
        src_json,
        src_bin,
        rng,
        args.n_each,
        args.rtol,
        args.atol,
        bin_to_pyg,
        j2p.build_pyg_from_json_path,
        compare_two_pyg,
    )
    o2, f2 = _run_split(
        "target",
        tgt_json,
        tgt_bin,
        rng,
        args.n_each,
        args.rtol,
        args.atol,
        bin_to_pyg,
        j2p.build_pyg_from_json_path,
        compare_two_pyg,
    )

    total_fail = f1 + f2
    total_ok = o1 + o2
    print(f"\nTotals: OK={total_ok}  FAIL={total_fail}")
    if total_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
