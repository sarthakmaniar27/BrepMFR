# -*- coding: utf-8 -*-
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings


def _silence_known_third_party_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"google\.api_core",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"google\.cloud",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*infer the `batch_size` from an ambiguous collection.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*persistent_workers=True.*",
        module=r"pytorch_lightning\.trainer\.connectors\.data_connector",
    )


_silence_known_third_party_warnings()

import argparse
import pathlib
import re
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.training_logging import (
    build_loggers,
    build_pytorch_profiler,
    build_train_callbacks,
)
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
        "--pin_memory",
        action="store_true",
        help=(
            "DataLoader pin_memory=True (faster staging to GPU when CUDA; slightly more host RAM)."
        ),
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=None,
        metavar="N",
        help="DataLoader prefetch_factor when num_workers>0 (default: 1). Try 2 or 4 if GPU idles.",
    )
    parser.add_argument(
        "--cuda_launch_blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1 (CUDA debugging only — large slowdown).",
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
    parser.add_argument(
        "--pt_subdir",
        type=str,
        default=None,
        help=(
            "Load graphs only under `<source_path>/<pt_subdir>` and `<target_path>/<pt_subdir>`. "
            "Example: `output/bin_skip_a2`."
        ),
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Also log to Weights & Biases (requires pip install wandb).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (default: brepmfr-pyg).",
    )
    parser.add_argument(
        "--csv_log",
        action="store_true",
        help=(
            "Also write Lightning CSVLogger metrics under "
            "results/logs/stage2/<run_name>/csv_metrics/."
        ),
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="Caps training batches each epoch (Lightning). Use ~2–5 with --max_epochs 1 for smoke runs.",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="Caps validation batches each epoch (Lightning).",
    )
    parser.add_argument(
        "--tb_surrogate_trace",
        action="store_true",
        help=(
            "TorchInfo + TB add_graph tiny surrogate at train start. Off by default (can hang on Windows+CUDA)."
        ),
    )
    parser.add_argument(
        "--tb_full_graph",
        action="store_true",
        help=(
            "Log extra TensorBoard GRAPHs: BrepEncoder+head from one small source-domain graph, "
            "plus domain discriminator / GRL+d stack when applicable."
        ),
    )
    parser.add_argument(
        "--tb_profile",
        action="store_true",
        help="Emit PyTorch profiler traces under the run logs dir for TensorBoard PROFILE.",
    )
    parser.add_argument("--tb_profile_wait", type=int, default=1, help="Profiler schedule: wait steps.")
    parser.add_argument("--tb_profile_warmup", type=int, default=1, help="Profiler schedule: warmup steps.")
    parser.add_argument("--tb_profile_active", type=int, default=3, help="Profiler schedule: active steps per repeat.")
    parser.add_argument("--tb_profile_repeat", type=int, default=1, help="Profiler schedule: repeat count.")
    parser.add_argument(
        "--tb_profile_cuda_only",
        action="store_true",
        help="Profiler: CUDA activities only (lower overhead; less CPU detail). Requires GPU.",
    )

    args = parser.parse_args()

    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

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
        logs_path = repo_root.joinpath(
            "results", "logs", "stage2", args.run_name
        )
        logs_path.mkdir(parents=True, exist_ok=True)

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

        callbacks = build_train_callbacks(
            checkpoint=checkpoint_callback,
            stage="stage2",
            dim_node=args.dim_node,
            hyperparam_extras={
                "source_path": args.source_path,
                "target_path": args.target_path,
                "pt_subdir": args.pt_subdir,
                "iwdan": args.iwdan,
                "iwdan_source_priors": args.iwdan_source_priors,
                "iwdan_target_priors": args.iwdan_target_priors,
                "pre_train": args.pre_train,
                "checkpoint_dir": str(results_path),
                "logs_dir": str(logs_path),
                "multi_hop_max_dist": 16,
                "spatial_pos_max": 32,
                "pin_memory": bool(args.pin_memory),
                "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
            },
            repo_root=repo_root,
            tb_full_graph=args.tb_full_graph,
            tb_surrogate_trace=args.tb_surrogate_trace or None,
        )
        loggers = build_loggers(
            logs_save_dir=logs_path,
            experiment_name=args.run_name,
            csv_log=args.csv_log,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
        )

        profiler = build_pytorch_profiler(
            logs_path,
            enabled=args.tb_profile,
            wait=args.tb_profile_wait,
            warmup=args.tb_profile_warmup,
            active=args.tb_profile_active,
            repeat=args.tb_profile_repeat,
            cuda_only=args.tb_profile_cuda_only,
        )

        tk = dict(
            max_epochs=args.max_epochs,
            log_every_n_steps=args.log_every_n_steps,
            callbacks=callbacks,
            logger=loggers,
            accelerator="gpu",
            devices=1,
            gradient_clip_val=1.0,
        )
        if profiler is not None:
            tk["profiler"] = profiler
        if args.limit_train_batches is not None:
            tk["limit_train_batches"] = args.limit_train_batches
        if args.limit_val_batches is not None:
            tk["limit_val_batches"] = args.limit_val_batches
        trainer = Trainer(**tk)
        print(
            f"""
-----------------------------------------------------------------------------------
Transfer learning / domain adaptation (Stage 2)
-----------------------------------------------------------------------------------
Run folder:
  results/stage2/{args.run_name}/     (checkpoints only)

TensorBoard (+ optional CSV / W&B) logs:
  results/logs/stage2/{args.run_name}/tensorboard/
  tensorboard --logdir results/logs/stage2/{args.run_name}/
  (use run folder root to include PROFILE traces when --tb_profile is set)

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
            pt_subdir=args.pt_subdir,
        )
        val_data = Dataset(
            root_dir_source=args.source_path,
            root_dir_target=args.target_path,
            split="val",
            random_rotate=False,
            num_class=args.num_classes,
            open_set=args.open_set,
            pt_subdir=args.pt_subdir,
        )
        train_loader = train_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor,
        )
        val_loader = val_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor,
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
            pt_subdir=args.pt_subdir,
        )
        test_loader = test_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor,
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
