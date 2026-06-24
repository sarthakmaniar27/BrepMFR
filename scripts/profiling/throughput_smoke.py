# -*- coding: utf-8 -*-
"""
Wall-clock smoke: measure collated batch delivery rate (CPU + disk + collator).
Does not run the model — isolates DataLoader / input pipeline vs GPU.

Examples (PowerShell):

  python scripts/profiling/throughput_smoke.py --dataset_path Z:\\path\\CADSynth \\
    --batch_size 8 --timed_batches 40 --warmup_batches 4

  python scripts/profiling/throughput_smoke.py ... --pin_memory \\
    --dataloader_prefetch_factor 2 --num_workers 4

Requires the same CADSynth layout as ``segmentation.py`` (train.txt under root or output/).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data.dataset import CADSynth, TransferDataset


def _consume_batches(loader, n: int) -> int:
    """Pull up to n batches from ``loader``."""
    n_done = 0
    for _ in loader:
        n_done += 1
        if n_done >= int(n):
            break
    return n_done


def bench_once(
    *,
    transfer: bool,
    dataset_path: str | None,
    source_path: str | None,
    target_path: str | None,
    stage: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    warmup_batches: int,
    timed_batches: int,
    pt_subdir: str | None,
    num_classes: int,
):
    if transfer:
        if not source_path or not target_path:
            raise ValueError("--source_path and --target_path required with --transfer")
        ds = TransferDataset(
            root_dir_source=source_path,
            root_dir_target=target_path,
            split=stage,
            random_rotate=False,
            num_class=num_classes,
            pt_subdir=pt_subdir,
        )
    else:
        if not dataset_path:
            raise ValueError("--dataset_path required for CADSynth")
        ds = CADSynth(
            dataset_path,
            split=stage,
            random_rotate=False,
            num_class=num_classes,
            pt_subdir=pt_subdir,
        )

    dl = ds.get_dataloader(
        batch_size=batch_size,
        shuffle=(stage == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    # Warmup passes (worker pool, file cache, mmap, etc.)
    for _ in range(2):
        _consume_batches(dl, warmup_batches)

    t0 = time.perf_counter()
    n_done = _consume_batches(dl, timed_batches)
    elapsed = max(time.perf_counter() - t0, 1e-9)
    return n_done, elapsed, len(ds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="CADSynth root (Stage 1). Not used when --transfer is set.",
    )
    ap.add_argument(
        "--transfer",
        action="store_true",
        help="Use TransferDataset (Stage 2 paths). Pass --source_path and --target_path.",
    )
    ap.add_argument("--source_path", type=str, default=None)
    ap.add_argument("--target_path", type=str, default=None)
    ap.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Which split to load.",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--pin_memory",
        action="store_true",
        help="DataLoader pin_memory=True (GPU training usually benefits downstream).",
    )
    ap.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=None,
        metavar="N",
        help="prefetch_factor when num_workers > 0 (default in code: 1).",
    )
    ap.add_argument("--warmup_batches", type=int, default=3)
    ap.add_argument("--timed_batches", type=int, default=30)
    ap.add_argument("--pt_subdir", type=str, default=None)
    ap.add_argument("--num_classes", type=int, default=25)
    ap.add_argument(
        "--cuda_launch_blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING for process (normally unused in this smoke).",
    )
    args = ap.parse_args()

    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    if args.transfer:
        if not args.source_path or not args.target_path:
            ap.error("--transfer requires --source_path and --target_path")
        dp = None
    else:
        if not args.dataset_path:
            ap.error("--dataset_path is required unless --transfer is used")
        dp = args.dataset_path

    n_done, elapsed, n_ds = bench_once(
        transfer=args.transfer,
        dataset_path=dp,
        source_path=args.source_path,
        target_path=args.target_path,
        stage=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.dataloader_prefetch_factor,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
        pt_subdir=args.pt_subdir,
        num_classes=args.num_classes,
    )

    batches_per_s = n_done / elapsed
    graphs_per_s = batches_per_s * args.batch_size
    print(f"Consumed batches: {n_done} / requested {args.timed_batches}")
    print(f"Wall-clock:       {elapsed:.3f}s")
    print(f"Batches/sec:      {batches_per_s:.2f}")
    print(f"Graphs/sec:       ~{graphs_per_s:.1f}  (batch_size={args.batch_size})")
    print(f"dataset len:      {n_ds}")
    print(
        "Settings:",
        {
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "prefetch_factor": args.dataloader_prefetch_factor,
            "split": args.split,
        },
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
