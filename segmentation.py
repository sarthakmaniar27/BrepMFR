# -*- coding: utf-8 -*-
import argparse
import pathlib
import re
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import CADSynth
from models.brepseg_model import BrepSeg
from models.modules.utils.macro import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use all available GPUs. Set to "0" to use only the first GPU, "1" for the second, etc.

parser = argparse.ArgumentParser("BrepMFR Network model")
parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
parser.add_argument("--num_classes", type=int, default=25, help="Number of features")
parser.add_argument("--dataset", choices=("cadsynth", "transfer"), default="cadsynth", help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=12,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
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
        "One folder under results/stage1/ for this training run. "
        "Use a stable slug + date, e.g. ce_unweighted_baseline__2026-05-10. "
        "Default: ce_weighted_balanced__YYYY-MM-DD_HHMMSS_mmm (time includes ms)."
    ),
)

# Optional argument to load weights from a checkpoint for fine-tuning. 
# If provided, the model will be initialized with the weights from the checkpoint and training will continue from there. If not provided, training will start from scratch.
parser.add_argument(
    "--pre_train",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for fine-tuning",
)

# parser.add_argument(
#     "--device",
#     type=str,
#     default="gpu",
#     help="Device to run on (default: gpu)",
# )

#设置transformer模块的默认参数
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--attention_dropout", type=float, default=0.3)
parser.add_argument("--act-dropout", type=float, default=0.3)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dim_node", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=32)
parser.add_argument("--n_layers_encode", type=int, default=8)
parser.add_argument("--warmup_freeze_epochs", type=int, default=3)
parser.add_argument("--max_epochs", type=int, default=1000)
parser.add_argument("--log_every_n_steps", type=int, default=50)
parser.add_argument(
    "--class_weights_path",
    type=str,
    default=None,
    help=(
        "Path to a JSON file produced by scripts/training/compute_class_weights.py "
        "(canonical copies under artifacts/class_weights/stage1/). "
        "When set, the source CE loss is multiplied per-class by the provided "
        "weights. This counteracts the class-0 dominance (~58%% stock) and "
        "produces a less over-confident encoder, which closes the label-shift "
        "gap on target evaluation."
    ),
)

def _sanitize_run_name(name: str) -> str:
    """Filesystem-safe single path segment (no separators or odd chars)."""
    n = re.sub(r'[<>:"/\\|?*]+', "_", name.strip())
    return n or "unnamed_run"


def main():
    # On Windows the DataLoader uses spawn(); without an `if __name__ == "__main__"`
    # guard around training setup, every worker re-imports this module and tries to
    # restart training, eventually crashing in `_check_not_importing_main`.
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    repo_root = pathlib.Path(__file__).parent

    if args.dataset == "cadsynth":
        Dataset = CADSynth
    else:
        raise ValueError("Unsupported dataset")

    if args.traintest == "train":
        if args.run_name is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
            args.run_name = f"ce_weighted_balanced__{ts}"
        args.run_name = _sanitize_run_name(args.run_name)
        # Hyperparameters / checkpoints still expose experiment_name for Lightning parity.
        args.experiment_name = args.run_name

        results_path = repo_root.joinpath("results", "stage1", args.run_name)
        results_path.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            monitor="eval_loss",
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
B-rep model feature recognition (Stage 1)
-----------------------------------------------------------------------------------
Run folder:
  results/stage1/{args.run_name}/

TensorBoard logs:
  results/stage1/{args.run_name}/tensorboard/

To monitor training:
  tensorboard --logdir results/stage1/{args.run_name}/tensorboard

Best checkpoint (Lightning top-k also in same folder):
  results/stage1/{args.run_name}/best.ckpt
-----------------------------------------------------------------------------------
        """
        )
        # if args.pre_train is not None:
        #     model = BrepSeg.load_from_checkpoint(args.pre_train, args=args, strict=False)
        # else:
        #     model = BrepSeg(args)

        model = BrepSeg(args)

        train_data = Dataset(root_dir=args.dataset_path, split="train", random_rotate=True, num_class=args.num_classes)
        val_data = Dataset(root_dir=args.dataset_path, split="val", random_rotate=False, num_class=args.num_classes)
        train_loader = train_data.get_dataloader(
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        val_loader = val_data.get_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        trainer.fit(model, train_loader, val_loader)

    else:
        assert (
            args.checkpoint is not None
        ), "Expected the --checkpoint argument to be provided"
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        test_data = Dataset(root_dir=args.dataset_path, split="test", random_rotate=False, num_class=args.num_classes)
        test_loader = test_data.get_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        model = BrepSeg.load_from_checkpoint(args.checkpoint)
        trainer.test(model, dataloaders=test_loader, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    # Required on Windows so that DataLoader workers spawned with `spawn()` can
    # safely import this module without re-running training.
    from multiprocessing import freeze_support
    freeze_support()
    main()

