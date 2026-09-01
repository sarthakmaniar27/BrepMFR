#!/usr/bin/env python3
"""Developer integration smoke using prepared unlabeled graphs as synthetic class-0 views.

This does not create a candidate and is not an experiment recipe. It exists to
exercise Lightning/CombinedLoader/model hooks without requiring the network
training dataset on the current machine.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from data.collator import collator  # noqa: E402
from unsupervised_training.config import ExperimentConfig  # noqa: E402
from unsupervised_training.constants import MULTI_HOP_MAX_DIST, SPATIAL_POS_MAX  # noqa: E402
from unsupervised_training.data import UnlabeledGraphDataset  # noqa: E402
from unsupervised_training.semi_model import SemiSupervisedBrepSeg  # noqa: E402

try:
    from pytorch_lightning.utilities.combined_loader import CombinedLoader
except ImportError:  # pragma: no cover
    from lightning.pytorch.utilities.combined_loader import CombinedLoader  # type: ignore[no-redef]


class SyntheticLabeledView(Dataset):
    def __init__(self, source: UnlabeledGraphDataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        graph = copy.deepcopy(self.source[index])
        graph.label_feature = torch.zeros_like(graph.label_feature)
        return graph


def _collate(items):
    return collator(
        items,
        multi_hop_max_dist=MULTI_HOP_MAX_DIST,
        spatial_pos_max=SPATIAL_POS_MAX,
        max_nodes_for_a3=768,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--unlabeled-dataset-root", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()

    train_source = UnlabeledGraphDataset(args.unlabeled_dataset_root, "train", scan_graphs=True)
    val_source = UnlabeledGraphDataset(args.unlabeled_dataset_root, "val", scan_graphs=True)
    labeled_train = DataLoader(SyntheticLabeledView(train_source), batch_size=1, collate_fn=_collate)
    unlabeled_train = train_source.get_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        persistent_workers=False,
        length_bucket_batching=False,
        batch_node_sq_budget=0,
    )
    labeled_val = DataLoader(SyntheticLabeledView(val_source), batch_size=1, collate_fn=_collate)
    combined = CombinedLoader(
        {"labeled": labeled_train, "unlabeled": unlabeled_train}, mode="min_size"
    )
    config = ExperimentConfig(
        experiment_name="lightning_smoke",
        champion_checkpoint=str(Path(args.champion_checkpoint).resolve()),
        labeled_dataset_root=".",
        unlabeled_dataset_root=str(Path(args.unlabeled_dataset_root).resolve()),
        accelerator=args.device,
        devices=1,
        precision="32",
        max_epochs=1,
        num_workers=0,
        warmup_steps=0,
    )
    model = SemiSupervisedBrepSeg(config)
    trainer = pl.Trainer(
        accelerator=args.device,
        devices=1,
        max_epochs=1,
        precision="32",
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=combined, val_dataloaders=labeled_val)
    print("Lightning semi-supervised integration smoke: PASSED")


if __name__ == "__main__":
    main()

