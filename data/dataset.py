# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import pathlib
from typing import Optional, Sequence, Union
from tqdm import tqdm
import random
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PYGGraph
from .collator import collator, collator_st
from .utils import get_random_rotation, rotate_uvgrid
from .subgraph_sampler import (
    make_rng_for_index,
    parse_seeds_per_class,
    sample_balanced_subgraph,
)
from .length_bucket_batch_sampler import LengthBucketBatchSampler


def _load_pyg_sample(path: pathlib.Path) -> PYGGraph:
    return torch.load(path, map_location="cpu", weights_only=False)


def _labels_from_sample(path: pathlib.Path, num_class: int) -> torch.Tensor:
    obj = _load_pyg_sample(path)
    return obj.label_feature


def _resolve_dataset_split_list(root_dir: pathlib.Path, filename: str) -> pathlib.Path:
    """Resolve ``train.txt`` / ``val.txt`` / ``test.txt``.

    Search order:

    1. ``root_dir / name`` (default: splits next to graphs).
    2. ``root_dir / output / name`` (Experiment6 layout after conversion).
    3. If ``root_dir`` is a folder named ``pyg``, ``root_dir.parent / name`` (splits one
       level up, e.g. ``.../lite/train.txt`` with graphs under ``.../lite/pyg``).
    """
    direct = root_dir / filename
    if direct.is_file():
        return direct
    under_output = root_dir / "output" / filename
    if under_output.is_file():
        return under_output
    if root_dir.name.lower() == "pyg":
        parent_list = root_dir.parent / filename
        if parent_list.is_file():
            return parent_list
    raise FileNotFoundError(
        f"Split list missing: tried '{direct}', '{under_output}'"
        + (
            f", '{root_dir.parent / filename}'"
            if root_dir.name.lower() == "pyg"
            else ""
        )
        + "."
    )


def _resolve_graph_pt_scan_root(root_dir: pathlib.Path, pt_subdir: str | None) -> pathlib.Path:
    """
    Directory under ``root_dir`` to ``rglob("*[0-9].pt")`` for graph files.

    When ``pt_subdir`` is None, scan the entire ``root_dir`` tree (legacy).
    When set (e.g. ``output/bin_skip_a2``), only graphs under that subfolder load—avoids mixing
    full-A2 vs zero-A2 duplicates for the same split stem list.
    """
    root_dir = pathlib.Path(root_dir).resolve()
    if not pt_subdir:
        return root_dir
    sub = pathlib.Path(pt_subdir)
    scan = root_dir / sub if not sub.is_absolute() else sub
    if not scan.is_dir():
        raise FileNotFoundError(
            f"--pt_subdir resolved to missing directory: {scan}\n"
            f"dataset root was: {root_dir}"
        )
    return scan


def _dataloader_kw(
    num_workers: int,
    *,
    prefetch_factor: int | None = None,
    pin_memory: bool = False,
):
    # prefetch_factor capped low by default — raise via CLI (--dataloader_prefetch_factor)
    # if profiling shows idle GPU waiting on workers (watch RAM / page-file on Windows).
    kw = dict(num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
    if num_workers > 0:
        kw["prefetch_factor"] = 1 if prefetch_factor is None else int(prefetch_factor)
        # On Windows, persistent workers keep file mappings across epochs and often
        # contribute to ERROR_COMMITMENT_LIMIT (1455) with huge collated batches.
        kw["persistent_workers"] = os.name != "nt"
    return kw


def _make_dataloader(
    dataset,
    collate_fn,
    batch_size,
    shuffle,
    num_workers,
    *,
    prefetch_factor: int | None = None,
    pin_memory: bool = False,
    batch_sampler=None,
):
    """Workers use torch ``DataLoader`` only (no stacked prefetch on Windows).

    ``num_workers==0`` uses a vanilla ``DataLoader``—BackgroundGenerator-style wrappers
    were removed after stuck-after-sanity reports on Windows + Lightning + CUDA.

    When ``batch_sampler`` is provided (a :class:`LengthBucketBatchSampler`),
    ``batch_size`` / ``shuffle`` / ``drop_last`` are NOT forwarded to ``DataLoader``
    because PyTorch forbids specifying them alongside a batch_sampler.
    """
    dl_kw = _dataloader_kw(
        num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory
    )
    if batch_sampler is not None:
        # batch_sampler is mutually exclusive with batch_size, shuffle, sampler,
        # AND drop_last. _dataloader_kw includes drop_last=True, so strip it.
        sampler_kw = {k: v for k, v in dl_kw.items() if k != "drop_last"}
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            **sampler_kw,
        )
    if num_workers > 0:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **dl_kw,
        )
    # num_workers==0: use the stock DataLoader. DataLoaderX+BackgroundGenerator
    # prefetch has caused stuck training (no epochs logged) after sanity check on
    # Windows + Lightning + CUDA in this project.
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **dl_kw,
    )


class CADSynth(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        random_rotate=False,
        num_class=33,
        pt_subdir=None,
        max_graph_nodes: Optional[int] = None,
        drop_invalid_graphs: bool = False,
        *,
        # ------------------------------------------------------------------
        # Subgraph training (opt-in; default = False preserves old behavior)
        # ------------------------------------------------------------------
        subgraph_training: bool = False,
        subgraph_k_hop: int = 2,
        subgraph_seeds_per_class: Optional[Union[str, Sequence[int]]] = None,
        subgraph_on_nontrain: bool = False,
        subgraph_global_seed: int = 42,
    ):
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)
        self.split = split
        self.num_class = num_class
        self.random_rotate = random_rotate
        self.pt_subdir = pt_subdir
        self.max_graph_nodes = max_graph_nodes
        self.drop_invalid_graphs = bool(drop_invalid_graphs)

        # Subgraph sampling configuration (ignored when subgraph_training=False)
        self.subgraph_training = bool(subgraph_training)
        self.subgraph_k_hop = int(subgraph_k_hop)
        # Store a concrete list for fast path; support string specs from CLI
        if subgraph_seeds_per_class is None:
            self.subgraph_seeds_per_class = None
        else:
            parsed = parse_seeds_per_class(subgraph_seeds_per_class, num_class)
            self.subgraph_seeds_per_class = [int(x) for x in parsed]
        self.subgraph_on_nontrain = bool(subgraph_on_nontrain)
        self.subgraph_global_seed = int(subgraph_global_seed)
        # Can be bumped by training code each epoch for more variety on the same file
        self.subgraph_epoch = 0

        self.file_paths = []
        # Actual node counts per kept file — populated by _filter_graphs_by_constraints
        # when that scan runs (i.e. when --drop_invalid_graphs or --max_graph_nodes is set).
        # Used by get_dataloader(length_bucket_batching=True) for accurate bucketing.
        self._actual_node_counts: list[int] = []
        self._get_filenames(path, filelist=split + ".txt")
        if self.drop_invalid_graphs or self.max_graph_nodes is not None:
            self._filter_graphs_by_constraints()

    def _filter_graphs_by_constraints(self) -> None:
        """Drop empty graphs and optionally graphs larger than ``max_graph_nodes``.

        As a side-effect, populates ``self._actual_node_counts`` with the true
        node count for every kept graph.  This is consumed by
        ``get_dataloader(length_bucket_batching=True)`` so the batch sampler
        uses verified counts instead of parsing the filename.
        """
        cap = self.max_graph_nodes
        kept: list[pathlib.Path] = []
        kept_counts: list[int] = []
        dropped_bad = 0
        dropped_large = 0
        parts = []
        if self.drop_invalid_graphs:
            parts.append("drop_invalid")
        if cap is not None:
            parts.append(f"max_nodes<={int(cap)}")
        desc = "filter: " + ("+".join(parts) if parts else "constraints")
        for p in tqdm(self.file_paths, desc=desc):
            try:
                g = _load_pyg_sample(pathlib.Path(p))
            except Exception:
                dropped_bad += 1
                continue
            n = int(g.node_data.size(0))
            lf = getattr(g, "label_feature", None)
            if lf is None or lf.numel() == 0 or n == 0:
                dropped_bad += 1
                continue
            if cap is not None and n > int(cap):
                dropped_large += 1
                continue
            kept.append(pathlib.Path(p))
            kept_counts.append(n)
        print(
            f"Graph filter ({desc}): kept {len(kept):,} | "
            f"dropped_bad={dropped_bad:,} dropped_large={dropped_large:,} "
            f"(from {len(self.file_paths):,} split-listed paths)",
            flush=True,
        )
        self.file_paths = kept
        self._actual_node_counts = kept_counts

    def _get_filenames(self, root_dir, filelist):
        print(f"Loading data...")
        list_path = _resolve_dataset_split_list(root_dir, filelist)
        with open(list_path, "r", encoding="utf-8") as f:
            file_list = [x.strip() for x in f.readlines()]
        scan_root = _resolve_graph_pt_scan_root(root_dir, self.pt_subdir)
        if self.pt_subdir:
            print(f"  (--pt_subdir) scanning graphs under: {scan_root}")
        for x in tqdm(scan_root.rglob(f"*[0-9].pt")):
            if x.stem in file_list:
                self.file_paths.append(x)
        print("Done loading {} files".format(len(self.file_paths)))

    def load_one_graph(self, file_path):
        pyg_graph = _load_pyg_sample(pathlib.Path(file_path))
        if self.random_rotate:
            rotation = get_random_rotation()
            pyg_graph.node_data = rotate_uvgrid(pyg_graph.node_data, rotation)
            pyg_graph.edge_data = rotate_uvgrid(pyg_graph.edge_data, rotation)

        lf = pyg_graph.label_feature
        if lf is None:
            raise ValueError(f"Missing label_feature on graph: {file_path}")
        lf = lf.flatten()
        if lf.numel() == 0:
            raise ValueError(
                f"Empty label_feature (zero faces) in {file_path}. "
                "Remove this stem from train/val/test.txt or fix the source JSON."
            )
        if (lf.max() >= self.num_class) or (lf.min() < 0):
            print(
                f"Invalid label in graph id: {getattr(pyg_graph, 'data_id', '?')}, "
                f"min={lf.min().item()}, max={lf.max().item()}, "
                f"expected range=[0, {self.num_class - 1}]"
            )

        return pyg_graph

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fn = self.file_paths[idx]
        sample = self.load_one_graph(fn)

        # ------------------------------------------------------------------
        # Optional subgraph training path (disabled by default)
        # When enabled, we replace the full graph with a small k-hop union
        # around a class-balanced set of seed faces. All tensors are sliced
        # so the collator / encoder / loss see a perfectly normal mini-graph.
        # ------------------------------------------------------------------
        if self.subgraph_training and (self.split == "train" or self.subgraph_on_nontrain):
            rng = make_rng_for_index(
                self.subgraph_global_seed,
                getattr(self, "subgraph_epoch", 0),
                int(idx),
                self.split,
            )
            seeds_spec = self.subgraph_seeds_per_class or (2, 3, 3)
            sample = sample_balanced_subgraph(
                sample,
                k_hop=self.subgraph_k_hop,
                seeds_per_class=seeds_spec,
                num_classes=self.num_class,
                rng=rng,
            )
        return sample

    def _collate(self, batch):
        return collator(
            batch,
            multi_hop_max_dist=16,
            spatial_pos_max=32,
        )

    def get_dataloader(
        self,
        batch_size,
        shuffle=True,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        length_bucket_batching: bool = False,
    ):
        batch_sampler = None
        if length_bucket_batching:
            # Prefer actual node counts collected during _filter_graphs_by_constraints
            # (populated when --drop_invalid_graphs or --max_graph_nodes is used).
            # Falls back to filename parsing when the scan wasn't run, but files whose
            # names don't contain a face count are then conservatively treated as large
            # (bs=1) to prevent O(N²) OOM from unexpectedly large graphs.
            node_counts = self._actual_node_counts if self._actual_node_counts else None
            batch_sampler = LengthBucketBatchSampler(
                self.file_paths,
                base_batch_size=batch_size,
                shuffle=shuffle,
                node_counts=node_counts,
            )
            print(
                f"  total batches={len(batch_sampler)}",
                flush=True,
            )
        return _make_dataloader(
            self,
            self._collate,
            batch_size,
            shuffle,
            num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            batch_sampler=batch_sampler,
        )


class TransferDataset(Dataset):
    def __init__(
        self,
        root_dir_source,
        root_dir_target,
        split="train",
        random_rotate=False,
        num_class=25,
        open_set=0,
        pt_subdir=None,
    ):
        assert split in ("train", "val", "test")
        source_path = pathlib.Path(root_dir_source)
        target_path = pathlib.Path(root_dir_target)
        self.split = split
        self.random_rotate = random_rotate
        self.num_class = num_class
        self.open_set = bool(open_set)
        self.pt_subdir = pt_subdir

        self.source_file_paths = []
        self.target_file_paths = []
        self._get_filenames(source_path, target_path)

    def _get_filenames(self, source_dir, target_dir):
        if self.split == "train":
            filelist_s = "s_train.txt"
            filelist_t = "t_train.txt"
        elif self.split == "val":
            filelist_s = "s_val.txt"
            filelist_t = "t_val.txt"
        elif self.split == "test":
            filelist_s = "s_test.txt"
            filelist_t = "t_test.txt"

        print(f"Loading source data...")
        s_list_path = _resolve_dataset_split_list(source_dir, filelist_s)
        with open(s_list_path, "r", encoding="utf-8") as f:
            s_file_list = [x.strip() for x in f.readlines()]
        scan_s = _resolve_graph_pt_scan_root(source_dir, self.pt_subdir)
        if self.pt_subdir:
            print(f"  (--pt_subdir) source graph scan root: {scan_s}")
        for x in tqdm(scan_s.rglob(f"*[0-9].pt")):
            if x.stem in s_file_list:
                if self.open_set:
                    face_labels = _labels_from_sample(x, self.num_class).flatten()
                    if face_labels.numel() == 0:
                        continue
                    if torch.max(face_labels) > self.num_class:
                        continue
                self.source_file_paths.append(x)
        print("Done loading {} files".format(len(self.source_file_paths)))

        print(f"Loading target data...")
        t_list_path = _resolve_dataset_split_list(target_dir, filelist_t)
        with open(t_list_path, "r", encoding="utf-8") as f:
            t_file_list = [x.strip() for x in f.readlines()]
        scan_t = _resolve_graph_pt_scan_root(target_dir, self.pt_subdir)
        if self.pt_subdir:
            print(f"  (--pt_subdir) target graph scan root: {scan_t}")
        for x in tqdm(scan_t.rglob(f"*[0-9].pt")):
            if x.stem in t_file_list:
                if self.open_set:
                    face_labels = _labels_from_sample(x, self.num_class).flatten()
                    if face_labels.numel() == 0:
                        continue
                    if torch.max(face_labels) > self.num_class:
                        continue
                self.target_file_paths.append(x)
        print("Done loading {} files".format(len(self.target_file_paths)))

        if self.split != "test":
            random.shuffle(self.source_file_paths)
            random.shuffle(self.target_file_paths)

    def load_one_graph(self, file_path):
        pyg_graph = _load_pyg_sample(pathlib.Path(file_path))
        if self.random_rotate:
            rotation = get_random_rotation()
            pyg_graph.node_data = rotate_uvgrid(pyg_graph.node_data, rotation)
            pyg_graph.edge_data = rotate_uvgrid(pyg_graph.edge_data, rotation)

        _, file_extension = os.path.splitext(file_path)
        basename = os.path.basename(file_path).replace(file_extension, "")
        pyg_graph.data_id = int(basename.split("_")[-1])

        return pyg_graph

    def __len__(self):
        if self.split == "train":
            return max(len(self.source_file_paths), len(self.target_file_paths))
        else:
            return len(self.target_file_paths)

    def __getitem__(self, idx):
        idx_s = idx
        idx_t = idx
        if idx_s >= len(self.source_file_paths):
            idx_s = random.randint(0, len(self.source_file_paths) - 1)
        if idx_t >= len(self.target_file_paths):
            idx_t = random.randint(0, len(self.target_file_paths) - 1)

        fn_s = self.source_file_paths[idx_s]
        fn_t = self.target_file_paths[idx_t]

        sample_s = self.load_one_graph(fn_s)
        sample_t = self.load_one_graph(fn_t)
        sample = {"source_data": sample_s, "target_data": sample_t}
        return sample

    def _collate(self, batch):
        return collator_st(
            batch,
            multi_hop_max_dist=16,
            spatial_pos_max=32,
        )

    def get_dataloader(
        self,
        batch_size,
        shuffle=True,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
    ):
        return _make_dataloader(
            self,
            self._collate,
            batch_size,
            shuffle,
            num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
