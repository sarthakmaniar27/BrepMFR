from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from data.collator import collator
from data.dataset import CADSynth, _make_dataloader
from data.length_bucket_batch_sampler import LengthBucketBatchSampler

from .constants import IGNORE_LABEL, MULTI_HOP_MAX_DIST, SPATIAL_POS_MAX


def _torch_load(path: pathlib.Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_split(path: pathlib.Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required split file not found: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(values) != len({value.casefold() for value in values}):
        raise ValueError(f"Duplicate stems in split: {path}")
    return values


class UnlabeledGraphDataset(Dataset):
    """Strict dataset for graphs whose labels must all be the ignore sentinel."""

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        split: str = "train",
        *,
        pt_subdir: str = "pyg",
        max_nodes_for_a3: int | None = 768,
        scan_graphs: bool = True,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("UnlabeledGraphDataset supports only train/val splits")
        self.root = pathlib.Path(root_dir).expanduser().resolve()
        self.split = split
        self.graph_root = self.root / pt_subdir
        self.max_nodes_for_a3 = max_nodes_for_a3
        if not self.graph_root.is_dir():
            raise FileNotFoundError(f"Unlabeled graph directory not found: {self.graph_root}")

        stems = _read_split(self.root / f"{split}.txt")
        index = {path.stem.casefold(): path for path in self.graph_root.rglob("*.pt")}
        missing = [stem for stem in stems if stem.casefold() not in index]
        if missing:
            raise ValueError(
                f"{len(missing)} unlabeled split stem(s) have no .pt; first: {missing[:10]}"
            )
        self.file_paths = [index[stem.casefold()] for stem in stems]
        self._actual_node_counts: list[int] = []
        if scan_graphs:
            self._strict_scan()

    def _strict_scan(self) -> None:
        counts: list[int] = []
        for path in self.file_paths:
            graph = _torch_load(path)
            labels = getattr(graph, "label_feature", None)
            if labels is None or labels.numel() == 0:
                raise ValueError(f"Missing/empty ignore labels in unlabeled graph: {path}")
            unique = torch.unique(labels.detach().cpu()).tolist()
            if unique != [IGNORE_LABEL]:
                raise ValueError(
                    f"Unlabeled graph contains non-ignore labels {unique}: {path}. "
                    "Never use model predictions or default class 0 as unlabeled targets."
                )
            n = int(graph.node_data.shape[0])
            if n <= 0:
                raise ValueError(f"Zero-face unlabeled graph: {path}")
            has_a1 = bool(getattr(graph, "has_a1", getattr(graph, "spatial_pos", None) is not None))
            has_a2 = bool(
                getattr(
                    graph,
                    "has_a2",
                    getattr(graph, "d2_distance", None) is not None
                    and getattr(graph, "angle_distance", None) is not None,
                )
            )
            has_a3 = bool(getattr(graph, "has_a3", getattr(graph, "edge_path", None) is not None))
            if not has_a1 or has_a2 or not has_a3:
                raise ValueError(
                    f"Expected no_a2 graph (A1=True, A2=False, A3=True), got "
                    f"A1={has_a1}, A2={has_a2}, A3={has_a3}: {path}"
                )
            counts.append(n)
        self._actual_node_counts = counts

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int):
        graph = _torch_load(self.file_paths[index])
        labels = graph.label_feature.flatten()
        if not torch.all(labels == IGNORE_LABEL):
            raise RuntimeError(f"Unlabeled sentinel invariant failed: {self.file_paths[index]}")
        return graph

    def _collate(self, items: Sequence[Any]) -> dict[str, torch.Tensor]:
        return collator(
            items,
            multi_hop_max_dist=MULTI_HOP_MAX_DIST,
            spatial_pos_max=SPATIAL_POS_MAX,
            max_nodes_for_a3=self.max_nodes_for_a3,
        )

    def get_dataloader(
        self,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        prefetch_factor: int | None,
        pin_memory: bool,
        persistent_workers: bool,
        length_bucket_batching: bool,
        batch_node_sq_budget: int,
    ):
        batch_sampler = None
        if length_bucket_batching:
            batch_sampler = LengthBucketBatchSampler(
                self.file_paths,
                base_batch_size=batch_size,
                shuffle=shuffle,
                node_counts=self._actual_node_counts or None,
                node_sq_budget=batch_node_sq_budget,
                a3_node_cap=self.max_nodes_for_a3,
            )
        return _make_dataloader(
            self,
            self._collate,
            batch_size,
            shuffle,
            num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            batch_sampler=batch_sampler,
        )


def build_labeled_dataset(
    root: str | pathlib.Path,
    split: str,
    *,
    max_nodes_for_a3: int | None,
) -> CADSynth:
    return CADSynth(
        root_dir=root,
        split=split,
        random_rotate=False,
        num_class=3,
        pt_subdir="pyg",
        max_nodes_for_a3=max_nodes_for_a3,
        drop_invalid_graphs=True,
        require_no_a2_a1_a3=True,
    )


def build_labeled_loader(
    dataset: CADSynth,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int | None,
    pin_memory: bool,
    persistent_workers: bool,
    length_bucket_batching: bool,
    batch_node_sq_budget: int,
):
    return dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        length_bucket_batching=length_bucket_batching,
        batch_node_sq_budget=batch_node_sq_budget,
    )

