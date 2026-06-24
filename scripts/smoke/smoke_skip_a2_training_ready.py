#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-sync checks before training the no-A2 (A2 tensors omitted) ablation."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stems(folder: Path, suffix: str) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.glob(f"*.{suffix}")}


def verify_triplets(output_dir: Path, tag: str) -> int:
    d_bin = output_dir / "bin"
    d_skip = output_dir / "bin_skip_a2"
    d_lbl = output_dir / "label"
    sb, sk, lj = (
        _stems(d_bin, "pt"),
        _stems(d_skip, "pt"),
        _stems(d_lbl, "json"),
    )
    print(f"[{tag}] |bin|={len(sb)} |bin_skip_a2|={len(sk)} |label|={len(lj)}")
    if sb != sk or sb != lj:
        raise RuntimeError(
            f"{tag}: stem mismatch — only_in_skip {sorted(sk - sb)[:5]} "
            f"only_label {sorted(lj - sb)[:5]} missing_skip {sorted(sb - sk)[:5]}"
        )
    return len(sb)


def sample_absent_a2(skip_pt_dir: Path, n: int, seed: int) -> None:
    import torch

    pts = sorted(p for p in skip_pt_dir.rglob("*[0-9].pt"))
    if len(pts) < n:
        raise RuntimeError(f"Not enough .pt under {skip_pt_dir}")
    rng = random.Random(seed)
    pick = rng.sample(pts, n)
    for p in pick:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if getattr(d, "d2_distance", None) is not None or getattr(d, "angle_distance", None) is not None:
            raise AssertionError(f"{p.name}: expected omitted d2_distance / angle_distance")
        if hasattr(d, "has_a2") and d.has_a2:
            raise AssertionError(f"{p.name}: has_a2 should be False when A2 omitted")
    print(f"   sampled {n} graphs: OK (no dense A2 tensors)")


def cad_synth_one_sample(dataset_root: Path, pt_subdir: str) -> None:
    from data.collator import collator
    from data.dataset import CADSynth

    ds = CADSynth(
        root_dir=dataset_root,
        split="train",
        random_rotate=False,
        num_class=25,
        pt_subdir=pt_subdir,
    )
    if len(ds) < 1:
        raise RuntimeError("CADSynth train split resolved 0 graphs")
    g = ds[0]
    if getattr(g, "d2_distance", None) is not None or getattr(g, "angle_distance", None) is not None:
        raise AssertionError("Graph should omit d2_distance / angle_distance for bin_skip_a2")
    b = collator([g], multi_hop_max_dist=16, spatial_pos_max=32)
    if b["d2_distance"] is not None or b["angle_distance"] is not None:
        raise AssertionError("Collated batch must pass d2_distance=None when graphs omit A2")
    print(f"   CADSynth(train): {len(ds)} files; idx 0 loads + collates OK.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "dataset_roots",
        nargs="+",
        type=Path,
        help="e.g. Z:/Experiment6_PyG/source_dataset Z:/Experiment6_PyG/target_dataset",
    )
    ap.add_argument(
        "--pt_subdir",
        default="output/bin_skip_a2",
        help="Subdir under dataset root scanned for CADSynth graphs (default: Experiment6 layout)",
    )
    ap.add_argument("--sample_zero", type=int, default=6, help="Renamed conceptually: sample graphs verifying A2 absent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--cad_synth_smoke",
        action="store_true",
        help="CADSynth(train) __getitem__(0); first call rglob-scan is slow on full corpora",
    )
    args = ap.parse_args()

    for root in args.dataset_roots:
        out_dir = root / "output"
        verify_triplets(out_dir, root.name)
        skip_pt = Path(root) / args.pt_subdir
        sample_absent_a2(skip_pt, args.sample_zero, args.seed)
        if args.cad_synth_smoke:
            cad_synth_one_sample(root, args.pt_subdir)

    print("All readiness checks passed.")


if __name__ == "__main__":
    main()
