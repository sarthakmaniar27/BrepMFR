# -*- coding: utf-8 -*-
import os
import pathlib
from tqdm import tqdm
import random
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PYGGraph
from prefetch_generator import BackgroundGenerator

from .collator import collator, collator_st
from .utils import get_random_rotation, rotate_uvgrid


def _load_pyg_sample(path: pathlib.Path) -> PYGGraph:
    return torch.load(path, map_location="cpu", weights_only=False)


def _labels_from_sample(path: pathlib.Path, num_class: int) -> torch.Tensor:
    obj = _load_pyg_sample(path)
    return obj.label_feature


def _resolve_dataset_split_list(root_dir: pathlib.Path, filename: str) -> pathlib.Path:
    """Prefer root_dir/name; fall back to root_dir/output/name (Experiment6 layout after conversion)."""
    direct = root_dir / filename
    if direct.is_file():
        return direct
    under_output = root_dir / "output" / filename
    if under_output.is_file():
        return under_output
    raise FileNotFoundError(
        f"Split list missing: '{direct}' and '{under_output}' not found."
    )


def _dataloader_kw(num_workers: int):
    # pin_memory=False: avoids extra copies; IPC still uses file-backed sharing on Windows.
    kw = dict(num_workers=num_workers, drop_last=True, pin_memory=False)
    if num_workers > 0:
        kw["prefetch_factor"] = 1
        # On Windows, persistent workers keep file mappings across epochs and often
        # contribute to ERROR_COMMITMENT_LIMIT (1455) with huge collated batches.
        kw["persistent_workers"] = os.name != "nt"
    return kw


class DataLoaderX(DataLoader):
    """Prefetch on the main thread only; use for num_workers=0."""

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def _make_dataloader(dataset, collate_fn, batch_size, shuffle, num_workers):
    """Workers + BackgroundGenerator double-prefetch and inflate Windows page-file use."""
    dl_kw = _dataloader_kw(num_workers)
    if num_workers > 0:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **dl_kw,
        )
    return DataLoaderX(
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
    ):
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)
        self.split = split
        self.num_class = num_class
        self.random_rotate = random_rotate
        self.file_paths = []
        self._get_filenames(path, filelist=split + ".txt")

    def _get_filenames(self, root_dir, filelist):
        print(f"Loading data...")
        list_path = _resolve_dataset_split_list(root_dir, filelist)
        with open(list_path, "r", encoding="utf-8") as f:
            file_list = [x.strip() for x in f.readlines()]
        for x in tqdm(root_dir.rglob(f"*[0-9].pt")):
            if x.stem in file_list:
                self.file_paths.append(x)
        print("Done loading {} files".format(len(self.file_paths)))

    def load_one_graph(self, file_path):
        pyg_graph = _load_pyg_sample(pathlib.Path(file_path))
        if self.random_rotate:
            rotation = get_random_rotation()
            pyg_graph.node_data = rotate_uvgrid(pyg_graph.node_data, rotation)
            pyg_graph.edge_data = rotate_uvgrid(pyg_graph.edge_data, rotation)

        if torch.max(pyg_graph.label_feature) >= self.num_class or torch.min(
            pyg_graph.label_feature
        ) < 0:
            print(
                f"Invalid label in graph id: {pyg_graph.data_id}, "
                f"min={torch.min(pyg_graph.label_feature).item()}, "
                f"max={torch.max(pyg_graph.label_feature).item()}, "
                f"expected range=[0, {self.num_class - 1}]"
            )

        return pyg_graph

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fn = self.file_paths[idx]
        sample = self.load_one_graph(fn)
        return sample

    def _collate(self, batch):
        return collator(
            batch,
            multi_hop_max_dist=16,
            spatial_pos_max=32,
        )

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return _make_dataloader(
            self, self._collate, batch_size, shuffle, num_workers,
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
    ):
        assert split in ("train", "val", "test")
        source_path = pathlib.Path(root_dir_source)
        target_path = pathlib.Path(root_dir_target)
        self.split = split
        self.random_rotate = random_rotate
        self.num_class = num_class
        self.open_set = bool(open_set)

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
        for x in tqdm(source_dir.rglob(f"*[0-9].pt")):
            if x.stem in s_file_list:
                if self.open_set:
                    face_labels = _labels_from_sample(x, self.num_class)
                    if torch.max(face_labels) > self.num_class:
                        continue
                self.source_file_paths.append(x)
        print("Done loading {} files".format(len(self.source_file_paths)))

        print(f"Loading target data...")
        t_list_path = _resolve_dataset_split_list(target_dir, filelist_t)
        with open(t_list_path, "r", encoding="utf-8") as f:
            t_file_list = [x.strip() for x in f.readlines()]
        for x in tqdm(target_dir.rglob(f"*[0-9].pt")):
            if x.stem in t_file_list:
                if self.open_set:
                    face_labels = _labels_from_sample(x, self.num_class)
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

    def get_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return _make_dataloader(
            self, self._collate, batch_size, shuffle, num_workers,
        )
