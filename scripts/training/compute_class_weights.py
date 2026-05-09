# -*- coding: utf-8 -*-
"""
Compute per-class loss weights from a labeled split for class-balanced training.

Why:
    The Stage 1 diagnostic showed source training data is dominated by class 0
    (57.65% stock). The resulting CE-trained classifier learns a P_source(0)
    prior that wrecks performance under label shift on the target domain.
    Re-training Stage 1 with class-balanced loss weights addresses this at the
    source: rare classes get larger gradients, dominant classes get smaller
    ones, and the resulting encoder is less over-confident.

How:
    For each sample in the chosen split, read .label_feature, count per-class
    occurrences, and compute weights:

        freq_c   = count_c / sum(counts)
        w_c      = (1 / freq_c) ** alpha       # alpha=0.5 → sqrt-inverse, 1.0 → full inverse
        w_c      = w_c / mean(w_c)             # normalise so mean = 1
        w_c      = clip(w_c, weight_min, weight_max)

    The resulting JSON is consumed by `models/brepseg_model.py` via the
    --class_weights_path CLI flag on `segmentation.py`.

Usage (PowerShell, single line):

  python scripts/training/compute_class_weights.py `
    --dataset_path "Z:/Experiment6_PyG/source_dataset" `
    --split train `
    --num_classes 25 `
    --alpha 0.5 `
    --out "artifacts/class_weights/stage1/source_train_alpha05.json"

Choose alpha based on how aggressive you want the rebalancing:
    0.0  uniform         (no weighting; sanity check)
    0.5  sqrt-inverse    (recommended starting point — robust, well-behaved)
    1.0  full inverse    (aggressive; can destabilise training on extreme imbalance)
"""

import argparse
import importlib.util
import json
import pathlib
import sys
from datetime import datetime
from pathlib import Path

_bf = Path(__file__).resolve()
for _ancestor in _bf.parents:
    _bst = _ancestor / "bootstrap_path.py"
    if _bst.is_file():
        _spec = importlib.util.spec_from_file_location("__brepmfr_bootstrap", _bst)
        _bm = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_bm)
        _bm.setup(str(_bf))
        break
else:
    raise RuntimeError(
        "bootstrap_path.py not found; keep scripts inside the BrepMFR_PyG repository."
    )

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.dataset import _resolve_dataset_split_list


class _LabelOnlyDataset(Dataset):
    """Loads only .label_feature from each .pt — no graph attention overhead."""

    def __init__(self, root: str, filelist: str):
        path = pathlib.Path(root)
        list_path = _resolve_dataset_split_list(path, filelist)
        with open(list_path, "r", encoding="utf-8") as f:
            wanted = set(line.strip() for line in f if line.strip())
        self.paths = [p for p in path.rglob("*[0-9].pt") if p.stem in wanted]
        if not self.paths:
            raise RuntimeError(f"No samples matched '{filelist}' under {path}")
        print(f"[{filelist}] resolved {list_path} -> {len(self.paths):,} files")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # We still pay the cost of torch.load() for the whole PyGGraph because
        # the .pt files are not partial-loadable. We just discard everything
        # except the labels in the collator.
        g = torch.load(self.paths[idx], map_location="cpu", weights_only=False)
        return g.label_feature.long()


def _collate_labels(batch):
    return torch.cat([t.flatten() for t in batch], dim=0)


def main():
    parser = argparse.ArgumentParser("Compute class weights for BrepSeg")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--split", default="train",
                        help="Filelist base name (will append .txt)")
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight exponent. 0=uniform, 0.5=sqrt-inv, 1.0=full inv freq.")
    parser.add_argument("--weight_min", type=float, default=0.1)
    parser.add_argument("--weight_max", type=float, default=20.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="If >0, sample only this many files (smoke-test or quick estimate).",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = _LabelOnlyDataset(args.dataset_path, args.split + ".txt")
    if args.max_files > 0 and args.max_files < len(ds.paths):
        # Deterministic subsample (first N) for reproducibility — fine for
        # smoke tests where we only need a non-zero count vector to flow through.
        ds.paths = ds.paths[: args.max_files]
        print(f"  --max_files: sampling first {len(ds.paths):,} files")

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_labels,
        pin_memory=False,
        drop_last=False,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 1
    loader = DataLoader(ds, **dl_kwargs)

    counts = np.zeros(args.num_classes, dtype=np.int64)
    for labels in tqdm(loader, desc="counting", dynamic_ncols=True):
        labels_np = labels.numpy()
        labels_np = labels_np[(labels_np >= 0) & (labels_np < args.num_classes)]
        counts += np.bincount(labels_np, minlength=args.num_classes)

    total = int(counts.sum())
    print(f"\nTotal labelled faces: {total:,}")
    print(f"  {'class':>5} {'count':>12} {'pct':>8}")
    for c in range(args.num_classes):
        pct = 100.0 * counts[c] / max(1, total)
        print(f"  {c:5d} {counts[c]:12,d} {pct:7.3f}%")

    # Compute weights
    freqs = counts.astype(np.float64) / max(1, total)
    freqs = np.maximum(freqs, 1e-8)
    weights = (1.0 / freqs) ** args.alpha
    weights = weights / weights.mean()
    weights = np.clip(weights, args.weight_min, args.weight_max)

    print(f"\nClass weights (method=inv_freq_pow, alpha={args.alpha}, "
          f"clip=[{args.weight_min}, {args.weight_max}]):")
    print(f"  {'class':>5} {'weight':>10}")
    for c in range(args.num_classes):
        print(f"  {c:5d} {weights[c]:10.4f}")
    print(f"  mean = {weights.mean():.4f}, "
          f"min = {weights.min():.4f}, max = {weights.max():.4f}")

    output = {
        "method": "inv_freq_pow",
        "alpha": float(args.alpha),
        "num_classes": int(args.num_classes),
        "num_samples": len(ds),
        "total_faces": total,
        "weight_min": float(args.weight_min),
        "weight_max": float(args.weight_max),
        "counts": counts.tolist(),
        "weights": weights.tolist(),
        "computed_from": str(pathlib.Path(args.dataset_path).resolve()),
        "split_file": args.split + ".txt",
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote class weights to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
