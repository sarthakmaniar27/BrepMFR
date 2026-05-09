# -*- coding: utf-8 -*-
import argparse
import pathlib
import re
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.transfer_model import DomainAdapt
from data.dataset import TransferDataset
from models.modules.utils.macro import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def _sanitize_run_name(name: str) -> str:
    n = re.sub(r'[<>:"/\\|?*]+', "_", name.strip())
    return n or "unnamed_run"


def main():
    parser = argparse.ArgumentParser("BrepSeg Network model")
    parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
    parser.add_argument("--num_classes", type=int, default=25, help="Number of features")
    parser.add_argument("--open_set", type=int, default=0)
    parser.add_argument("--dataset", choices=("cadsynth", "transfer"), default="transfer", help="Dataset to train on")
    parser.add_argument("--source_path", type=str, help="Path to source_dataset")
    parser.add_argument("--target_path", type=str, help="Path to target_dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per domain (effective batch = 2× this)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help=(
            "DataLoader worker processes. Each worker loads .pt files and runs the "
            "collator in parallel while the GPU processes the previous batch. "
            "On Windows, huge batches can hit error 1455 (page file); try 2 or 0 if that happens."
        ),
    )
    parser.add_argument(
        "--pre_train",
        type=str,
        default=None,
        help="Checkpoint file to load weights from for pre-trained model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file to load weights from for testing",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "One folder under results/stage2/ for this training run. "
            "Default prefix: transfer_iwdan_weighted__ if --iwdan, else transfer_dann__. "
            "Suffix: YYYY-MM-DD_HHMMSS_mmm. Override for ablations, e.g. transfer_iwdan_weighted__slow_grl."
        ),
    )
    # Transformer module default parameters
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--act-dropout", type=float, default=0.3)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dim_node", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_layers_encode", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    # GRL schedule (slow ramp tied to actual training length)
    parser.add_argument(
        "--grl_max_iters",
        type=int,
        default=0,
        help=(
            "Override GRL max_iters explicitly. If 0 (default), uses "
            "estimated_steps_per_epoch * max_epochs * grl_ramp_frac, which "
            "ramps lambda smoothly over a real fraction of training instead "
            "of saturating in <1 epoch (the dalib default of 1000)."
        ),
    )
    parser.add_argument(
        "--grl_ramp_frac",
        type=float,
        default=0.5,
        help="Fraction of total steps over which lambda ramps from 0 to ~1.",
    )
    parser.add_argument(
        "--estimated_steps_per_epoch",
        type=int,
        default=2444,
        help=(
            "Estimated training steps per epoch (used to size the GRL ramp). "
            "Default 2444 matches batch_size=32 on the full source train split."
        ),
    )

    # IWDAN (Importance-Weighted DANN, Tachet des Combes et al. NeurIPS 2020)
    parser.add_argument(
        "--iwdan",
        action="store_true",
        help=(
            "Enable IWDAN: per-class importance weight w[c]=P_T(c)/P_S(c) "
            "on the source side of the discriminator loss. Required for "
            "label-shifted DA (Zhao et al. ICML 2019)."
        ),
    )
    parser.add_argument(
        "--iwdan_source_priors",
        type=str,
        default=None,
        help=(
            "Path to source priors JSON (scripts/training/compute_class_weights.py output; "
            "store canonical IWDAN inputs under artifacts/class_weights/stage2_iwdan/)."
        ),
    )
    parser.add_argument(
        "--iwdan_target_priors",
        type=str,
        default=None,
        help=(
            "Path to target priors JSON (same format as scripts/training/compute_class_weights.py; "
            "store canonical files under artifacts/class_weights/stage2_iwdan/)."
        ),
    )
    parser.add_argument(
        "--iwdan_clip",
        type=float,
        default=10.0,
        help="Clip per-class IW ratio to [1/clip, clip] for stability.",
    )

    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).parent

    if args.dataset == "transfer":
        Dataset = TransferDataset
    else:
        raise ValueError("Unsupported dataset")

    if args.traintest == "train":
        if args.run_name is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
            prefix = "transfer_iwdan_weighted" if args.iwdan else "transfer_dann"
            args.run_name = f"{prefix}__{ts}"
        args.run_name = _sanitize_run_name(args.run_name)
        args.experiment_name = args.run_name

        results_path = repo_root.joinpath("results", "stage2", args.run_name)
        results_path.mkdir(parents=True, exist_ok=True)

        # ModelCheckpoint: monitor="eval_loss" with mode="min" because
        # eval_loss = 1 / target_accuracy → lowest eval_loss = highest accuracy.
        checkpoint_callback = ModelCheckpoint(
            monitor="eval_loss",
            mode="min",
            dirpath=str(results_path),
            filename="best",
            save_top_k=10,
            save_last=True,
        )

        trainer = Trainer(
            max_epochs=args.max_epochs,
            log_every_n_steps=args.log_every_n_steps,
            callbacks=[checkpoint_callback],
            logger=TensorBoardLogger(
                save_dir=str(results_path),
                name="tensorboard",
            ),
            accelerator="gpu",
            devices=1,
            gradient_clip_val=1.0,
        )
        print(
            f"""
-----------------------------------------------------------------------------------
Transfer learning / domain adaptation (Stage 2)
-----------------------------------------------------------------------------------
Run folder:
  results/stage2/{args.run_name}/

TensorBoard logs:
  results/stage2/{args.run_name}/tensorboard/

To monitor training:
  tensorboard --logdir results/stage2/{args.run_name}/tensorboard

Best checkpoint:
  results/stage2/{args.run_name}/best.ckpt
-----------------------------------------------------------------------------------
    """
        )
        model = DomainAdapt(args)
        train_data = Dataset(
            root_dir_source=args.source_path,
            root_dir_target=args.target_path,
            split="train",
            random_rotate=True,
            num_class=args.num_classes,
            open_set=args.open_set,
        )
        val_data = Dataset(
            root_dir_source=args.source_path,
            root_dir_target=args.target_path,
            split="val",
            random_rotate=False,
            num_class=args.num_classes,
            open_set=args.open_set,
        )
        train_loader = train_data.get_dataloader(
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        val_loader = val_data.get_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        trainer.fit(model, train_loader, val_loader)

    else:
        assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        test_data = Dataset(
            root_dir_source=args.source_path,
            root_dir_target=args.target_path,
            split="test",
            num_class=args.num_classes,
            open_set=args.open_set,
        )
        test_loader = test_data.get_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        model = DomainAdapt.load_from_checkpoint(args.checkpoint)
        trainer.test(model, dataloaders=test_loader, ckpt_path=args.checkpoint)


# Guard required for Windows multiprocessing (spawn context):
# Without this, each DataLoader worker re-runs the module top-level, which
# tries to parse args and crashes. With this guard only the main process runs main().
if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
