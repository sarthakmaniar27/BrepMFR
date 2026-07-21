# -*- coding: utf-8 -*-
"""Cost-aware batch sampling for dense B-rep graph attention.

The legacy three-bucket policy is retained. When ``node_sq_budget`` is set,
similarly sized graphs are greedily packed while enforcing

    batch_size * padded_max_nodes**2 <= node_sq_budget

This approximates the dominant padded self-attention cost. Graphs above and
below ``a3_node_cap`` are packed separately so an oversized graph never
disables A3 for eligible graphs in the same batch.
"""
from __future__ import annotations

import pathlib
import re
from typing import List, Optional, Sequence

import torch


_FACE_COUNT_RE = re.compile(r"_(\d+)$")


def parse_face_count(path) -> int:
    """Extract a trailing ``_<digits>`` face count, or zero when unknown."""
    match = _FACE_COUNT_RE.search(pathlib.Path(path).stem)
    return int(match.group(1)) if match else 0


class LengthBucketBatchSampler:
    """Yield length-local batches using a legacy or quadratic-budget policy."""

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
        node_sq_budget: Optional[int] = None,
        a3_node_cap: Optional[int] = None,
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
        self.node_sq_budget = (
            int(node_sq_budget)
            if node_sq_budget is not None and int(node_sq_budget) > 0
            else None
        )
        self.a3_node_cap = (
            int(a3_node_cap)
            if a3_node_cap is not None and int(a3_node_cap) > 0
            else None
        )
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._n_samples = len(file_paths)

        counts: List[int] = []
        unknown_count = 0
        for index, path in enumerate(file_paths):
            count = (
                int(node_counts[index])
                if node_counts is not None
                else parse_face_count(path)
            )
            if count <= 0:
                unknown_count += 1
                # A count above sqrt(budget) forces a safe singleton.
                count = max(self.node_sq_budget or 1, 1)
            counts.append(count)
        self._node_counts = counts

        source = "actual node counts" if node_counts is not None else "filename parsing"
        if unknown_count:
            print(
                f"LengthBucketBatchSampler: {unknown_count} unknown face counts; "
                "keeping them singleton-safe.",
                flush=True,
            )

        if self.node_sq_budget is not None:
            self._buckets = []
            if self.a3_node_cap is None:
                self._adaptive_groups = [list(range(self._n_samples))]
            else:
                self._adaptive_groups = [
                    [
                        index
                        for index, count in enumerate(counts)
                        if count <= self.a3_node_cap
                    ],
                    [
                        index
                        for index, count in enumerate(counts)
                        if count > self.a3_node_cap
                    ],
                ]
            preview_batches = self._make_adaptive_batches(shuffle_batches=False)
            self._cached_len = len(preview_batches)
            padded_cost = sum(
                len(batch) * max(counts[index] for index in batch) ** 2
                for batch in preview_batches
            )
            raw_cost = sum(count * count for count in counts)
            packing_efficiency = raw_cost / padded_cost if padded_cost else 1.0
            mean_graphs = self._n_samples / max(1, self._cached_len)
            max_graphs = max((len(batch) for batch in preview_batches), default=0)
            print(
                f"QuadraticBudgetBatchSampler source={source}: "
                f"samples={self._n_samples:,} | max_graphs={self.base_batch_size} | "
                f"node_sq_budget={self.node_sq_budget:,} | "
                f"A3 split cap={self.a3_node_cap or 'disabled'} | "
                f"batches={self._cached_len:,} | mean_graphs/batch={mean_graphs:.1f} | "
                f"largest_batch={max_graphs} | padding_efficiency={packing_efficiency:.1%}",
                flush=True,
            )
            return

        small_bs = self.base_batch_size
        medium_bs = max(1, self.base_batch_size // 2)
        large_bs = 1
        small_idx: List[int] = []
        medium_idx: List[int] = []
        large_idx: List[int] = []
        for index, count in enumerate(counts):
            if count <= self.bucket_small_max:
                small_idx.append(index)
            elif count <= self.bucket_medium_max:
                medium_idx.append(index)
            else:
                large_idx.append(index)
        self._buckets = [
            (small_idx, small_bs),
            (medium_idx, medium_bs),
            (large_idx, large_bs),
        ]
        print(
            f"LengthBucketBatchSampler source={source}: "
            f"small(<={bucket_small_max}) n={len(small_idx)} bs={small_bs} | "
            f"medium(<={bucket_medium_max}) n={len(medium_idx)} bs={medium_bs} | "
            f"large(>{bucket_medium_max}) n={len(large_idx)} bs=1",
            flush=True,
        )

    def _make_adaptive_batches(self, *, shuffle_batches: bool = True) -> List[List[int]]:
        """Pack similarly sized graphs under the padded quadratic-cost budget."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        batches: List[List[int]] = []

        for group in self._adaptive_groups:
            tie_breakers = (
                torch.rand(len(group), generator=generator).tolist()
                if self.shuffle
                else [0.0] * len(group)
            )
            order = sorted(
                zip(group, tie_breakers),
                key=lambda pair: (self._node_counts[pair[0]], pair[1]),
            )
            current: List[int] = []
            current_max = 0
            for index, _ in order:
                node_count = self._node_counts[index]
                projected_max = max(current_max, node_count)
                projected_size = len(current) + 1
                projected_cost = projected_size * projected_max * projected_max
                if current and (
                    projected_size > self.base_batch_size
                    or projected_cost > self.node_sq_budget
                ):
                    batches.append(current)
                    current = []
                    current_max = 0
                current.append(index)
                current_max = max(current_max, node_count)
            if current and (not self.drop_last or len(current) == self.base_batch_size):
                batches.append(current)

        if self.shuffle and shuffle_batches and batches:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in permutation]
        return batches

    def _make_legacy_batches(self) -> List[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        batches: List[List[int]] = []
        for indices, batch_size in self._buckets:
            if not indices:
                continue
            order = indices
            if self.shuffle:
                permutation = torch.randperm(
                    len(indices), generator=generator
                ).tolist()
                order = [indices[index] for index in permutation]
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                if self.drop_last and len(batch) < batch_size:
                    continue
                batches.append(batch)
        if self.shuffle and batches:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in permutation]
        return batches

    def __iter__(self):
        if self.shuffle:
            self.epoch += 1
        if self.node_sq_budget is not None:
            return iter(self._make_adaptive_batches())
        return iter(self._make_legacy_batches())

    def __len__(self) -> int:
        if self.node_sq_budget is not None:
            return self._cached_len
        total = 0
        for indices, batch_size in self._buckets:
            if not indices:
                continue
            count = len(indices)
            total += count // batch_size
            if count % batch_size and not self.drop_last:
                total += 1
        return total
