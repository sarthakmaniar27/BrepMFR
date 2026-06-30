# -*- coding: utf-8 -*-
"""Length-bucketed batch sampler for BrepMFR graph segmentation.

Groups graphs by face-count bucket so that large graphs (complex 3D parts
with many faces) always land in small (size-1 or size-2) batches, preventing
O(N^2) attention OOM spikes while keeping all training data.

Bucket thresholds and per-bucket batch sizes (derived from ``base_batch_size``):
  - small  (<= bucket_small_max,  default 150): batch_size = base_batch_size
  - medium (<= bucket_medium_max, default 300): batch_size = max(1, base_batch_size // 2)
  - large  (>  bucket_medium_max):              batch_size = 1

**Preferred**: pass ``node_counts`` (a list of actual node counts, one per
file in ``file_paths``) obtained from ``CADSynth._actual_node_counts`` after
the ``--drop_invalid_graphs`` scan.  This is the most reliable source.

**Fallback**: when ``node_counts`` is ``None``, face counts are parsed from
the trailing ``_N`` integer in each ``.pt`` filename (e.g.
``00000000_both_v3_104.pt`` -> 104).  Files whose stem does not match
``_(\\d+)$`` receive face count = 0 and land in the *small* bucket.
**This is unsafe if those files are actually large** — always prefer passing
real ``node_counts`` from the dataset scan.
"""
from __future__ import annotations

import pathlib
import re
from typing import List, Optional, Sequence

import torch


_FACE_COUNT_RE = re.compile(r"_(\d+)$")


def parse_face_count(path) -> int:
    """Extract the trailing integer (face count) from a .pt filename stem.

    Returns 0 for filenames that do not end with ``_<digits>``.  A return
    value of 0 means the face count is *unknown*, not that the graph is
    empty.  Callers should treat 0 as a sentinel and fall back to the actual
    node count loaded from the .pt file.
    """
    stem = pathlib.Path(path).stem
    m = _FACE_COUNT_RE.search(stem)
    return int(m.group(1)) if m else 0


class LengthBucketBatchSampler:
    """Batch sampler that buckets graphs by face count to prevent OOM spikes.

    Yields lists of dataset indices. When ``shuffle=True`` the indices within
    each bucket and the order of batches are re-shuffled on every ``__iter__``
    call (i.e. every epoch) using an internal epoch counter, so each epoch
    sees a different batch composition while large graphs never share a batch
    with another large graph.
    """

    def __init__(
        self,
        file_paths: Sequence,
        base_batch_size: int,
        *,
        node_counts: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        seed: int = 0,
        bucket_small_max: int = 150,
        bucket_medium_max: int = 300,
        drop_last: bool = False,
    ):
        if base_batch_size < 1:
            raise ValueError("base_batch_size must be >= 1")
        if bucket_small_max >= bucket_medium_max:
            raise ValueError("bucket_small_max must be < bucket_medium_max")
        if node_counts is not None and len(node_counts) != len(file_paths):
            raise ValueError(
                f"node_counts length ({len(node_counts)}) must match "
                f"file_paths length ({len(file_paths)})"
            )
        self.base_batch_size = int(base_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_small_max = int(bucket_small_max)
        self.bucket_medium_max = int(bucket_medium_max)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        small_bs = self.base_batch_size
        medium_bs = max(1, self.base_batch_size // 2)
        large_bs = 1

        unknown_count = 0
        small_idx: List[int] = []
        medium_idx: List[int] = []
        large_idx: List[int] = []
        for i, p in enumerate(file_paths):
            if node_counts is not None:
                fc = int(node_counts[i])
            else:
                fc = parse_face_count(p)
                if fc == 0:
                    # filename did not contain a face count -> unknown size.
                    # Treat as large (bs=1) to be safe; never pair unknowns.
                    unknown_count += 1
                    large_idx.append(i)
                    continue
            if fc <= self.bucket_small_max:
                small_idx.append(i)
            elif fc <= self.bucket_medium_max:
                medium_idx.append(i)
            else:
                large_idx.append(i)

        self._buckets = [
            (small_idx, small_bs),
            (medium_idx, medium_bs),
            (large_idx, large_bs),
        ]
        self._n_samples = len(file_paths)
        source = "actual node counts" if node_counts is not None else "filename parsing"
        if unknown_count:
            print(
                f"LengthBucketBatchSampler: {unknown_count} files had no face count in "
                f"their filename -> treated as large (bs=1). "
                f"Pass node_counts= for accurate bucketing.",
                flush=True,
            )
        print(
            f"LengthBucketBatchSampler source={source}: "
            f"small(<={bucket_small_max}) n={len(small_idx)} bs={small_bs} | "
            f"medium(<={bucket_medium_max}) n={len(medium_idx)} bs={medium_bs} | "
            f"large(>{bucket_medium_max}) n={len(large_idx)} bs=1",
            flush=True,
        )

    def _make_batches(self) -> List[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batches: List[List[int]] = []
        for indices, bs in self._buckets:
            if not indices:
                continue
            order = indices
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                order = [indices[i] for i in perm]
            for start in range(0, len(order), bs):
                batch = order[start:start + bs]
                if self.drop_last and len(batch) < bs:
                    continue
                batches.append(batch)
        if self.shuffle and batches:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self):
        if self.shuffle:
            self.epoch += 1
        return iter(self._make_batches())

    def __len__(self) -> int:
        total = 0
        for indices, bs in self._buckets:
            if not indices:
                continue
            n = len(indices)
            total += n // bs
            if n % bs and not self.drop_last:
                total += 1
        return total
