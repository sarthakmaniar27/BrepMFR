#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger  # noqa: E402

try:  # Lightning 2.x
    from pytorch_lightning.utilities.combined_loader import CombinedLoader  # noqa: E402
except ImportError:  # pragma: no cover - compatibility fallback
    from lightning.pytorch.utilities.combined_loader import CombinedLoader  # type: ignore[no-redef]

from unsupervised_training.config import ExperimentConfig  # noqa: E402
from unsupervised_training.constants import REPO_ROOT  # noqa: E402
from unsupervised_training.data import (  # noqa: E402
    UnlabeledGraphDataset,
    build_labeled_dataset,
    build_labeled_loader,
)
from unsupervised_training.semi_model import SemiSupervisedBrepSeg  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_manifest(config: ExperimentConfig, run_dir: Path, smoke: bool) -> None:
    labeled = Path(config.labeled_dataset_root).expanduser().resolve()
    unlabeled = Path(config.unlabeled_dataset_root).expanduser().resolve()
    champion = Path(config.champion_checkpoint).expanduser().resolve()
    files = [
        champion,
        labeled / "train.txt",
        labeled / "val.txt",
        labeled / "test.txt",
        unlabeled / "train.txt",
        unlabeled / "val.txt",
        unlabeled / "preparation_summary.json",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required experiment input(s) missing: {missing}")
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config.to_dict(),
        "smoke": smoke,
        "git_revision": _git_revision(),
        "input_sha256": {str(path): _sha256(path) for path in files},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / "run_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _loader_kwargs(config: ExperimentConfig) -> dict:
    workers = int(config.num_workers)
    return {
        "num_workers": workers,
        "prefetch_factor": config.prefetch_factor if workers > 0 else None,
        "pin_memory": bool(config.pin_memory),
        "persistent_workers": bool(config.persistent_workers and workers > 0),
        "length_bucket_batching": bool(config.length_bucket_batching),
        "batch_node_sq_budget": int(config.batch_node_sq_budget),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train masked-geometry ABC adaptation")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "abc_masked_geometry_v1.json"),
    )
    parser.add_argument("--champion-checkpoint")
    parser.add_argument("--labeled-dataset-root")
    parser.add_argument("--unlabeled-dataset-root")
    parser.add_argument("--run-name")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Two train/val batches for integration testing")
    parser.add_argument(
        "--skip-unlabeled-scan",
        action="store_true",
        help="Skip the startup sentinel/profile scan (only after a successful strict run)",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config).with_overrides(
        champion_checkpoint=args.champion_checkpoint,
        labeled_dataset_root=args.labeled_dataset_root,
        unlabeled_dataset_root=args.unlabeled_dataset_root,
        experiment_name=args.run_name,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
    )
    if args.smoke:
        config = config.with_overrides(
            experiment_name=f"{config.experiment_name}_smoke",
            max_epochs=1,
            num_workers=min(config.num_workers, 2),
        )

    pl.seed_everything(config.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = REPO_ROOT / "results" / "unsupervised" / config.experiment_name
    log_dir = REPO_ROOT / "results" / "logs" / "unsupervised" / config.experiment_name
    _write_manifest(config, run_dir, args.smoke)

    labeled_train = build_labeled_dataset(
        config.labeled_dataset_root,
        "train",
        max_nodes_for_a3=config.max_nodes_for_a3,
    )
    labeled_val = build_labeled_dataset(
        config.labeled_dataset_root,
        "val",
        max_nodes_for_a3=config.max_nodes_for_a3,
    )
    unlabeled_train = UnlabeledGraphDataset(
        config.unlabeled_dataset_root,
        "train",
        max_nodes_for_a3=config.max_nodes_for_a3,
        scan_graphs=not args.skip_unlabeled_scan,
    )

    common = _loader_kwargs(config)
    labeled_loader = build_labeled_loader(
        labeled_train,
        batch_size=config.labeled_batch_size,
        shuffle=True,
        **common,
    )
    unlabeled_loader = unlabeled_train.get_dataloader(
        batch_size=config.unlabeled_batch_size,
        shuffle=True,
        **common,
    )
    validation_loader = build_labeled_loader(
        labeled_val,
        batch_size=config.labeled_batch_size,
        shuffle=False,
        **common,
    )
    combined = CombinedLoader(
        {"labeled": labeled_loader, "unlabeled": unlabeled_loader},
        mode=config.combined_loader_mode,
    )

    checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="candidate-epoch{epoch:02d}-step{step}",
        monitor="val/guarded_score",
        mode="max",
        save_top_k=config.save_top_k,
        save_last=True,
        every_n_epochs=config.check_val_every_n_epoch,
        auto_insert_metric_name=False,
    )
    loggers = [
        TensorBoardLogger(save_dir=str(log_dir), name="tensorboard", version="version_0"),
        CSVLogger(save_dir=str(log_dir), name="csv", version="version_0"),
    ]
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        max_epochs=config.max_epochs,
        precision=config.precision,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=loggers,
        gradient_clip_val=config.gradient_clip_val,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        log_every_n_steps=config.log_every_n_steps,
        num_sanity_val_steps=0,
        limit_train_batches=2 if args.smoke else 1.0,
        limit_val_batches=2 if args.smoke else 1.0,
    )
    model = SemiSupervisedBrepSeg(config)
    resume = None
    if args.resume_from_checkpoint:
        resume = str(Path(args.resume_from_checkpoint).expanduser().resolve())
        if not Path(resume).is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")

    print("\nSemi-supervised ABC experiment")
    print(f"  champion:       {Path(config.champion_checkpoint).expanduser().resolve()}")
    print(f"  labeled root:   {Path(config.labeled_dataset_root).expanduser().resolve()}")
    print(f"  unlabeled root: {Path(config.unlabeled_dataset_root).expanduser().resolve()}")
    print(f"  run directory:  {run_dir}")
    print(f"  method:         masked geometry + fixed champion distillation")
    print(f"  unlabeled CE:   disabled (sentinel asserted at runtime)")
    trainer.fit(
        model,
        train_dataloaders=combined,
        val_dataloaders=validation_loader,
        ckpt_path=resume,
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()

