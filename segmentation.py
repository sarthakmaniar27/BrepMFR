# -*- coding: utf-8 -*-
# Windows / conda: LLVM OpenMP (libomp) + Intel OpenMP (libiomp5) often load together — must set before torch/numpy.
from __future__ import annotations

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
    # Suppress Lightning's generic persistent-worker suggestion; this is controlled explicitly.
    warnings.filterwarnings(
        "ignore",
        message=r".*persistent_workers=True.*",
        module=r"pytorch_lightning\.trainer\.connectors\.data_connector",
    )


_silence_known_third_party_warnings()

import argparse
import atexit
import pathlib
import re
import shutil
import sys
from datetime import datetime

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.training_logging import (
    build_loggers,
    build_pytorch_profiler,
    build_train_callbacks,
)
from data.dataset import CADSynth
from models.brepseg_model import BrepSeg
from models.modules.utils.macro import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# CUDA_LAUNCH_BLOCKING forces each kernel to finish before CUDA APIs return —
# invaluable while debugging illegal memory access; disastrous for throughput.
# Opt in with `--cuda_launch_blocking` only for CUDA error triage.

# ---------------------------------------------------------------------------
# Training profiles: pre-baked argument presets that replace the old .ps1
# launcher scripts.  Pick a profile with --training_profile and optionally
# override individual settings on the command line.
#
# "custom" (default) applies NO overrides — fully backward-compatible.
# See docs/training_recipes.md for usage examples.
# ---------------------------------------------------------------------------
TRAINING_PROFILES: dict[str, dict] = {
    "a1_a3_finetune_from_lite": {
        "num_classes": 3,
        "pt_subdir": "pyg",
        "drop_invalid_graphs": True,
        "batch_size": 64,
        "batch_node_sq_budget": 4000000,
        "precision": "16-mixed",
        "max_epochs": 30,
        "num_workers": 4,
        "pin_memory": True,
        "allow_tf32": True,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 2,
        "warmup_freeze_epochs": 0,
        "learning_rate": 0.0001,
        "a1_a3_learning_rate": 0.001,
        "optimizer_warmup_steps": 1000,
        "a1_a3_ramp_epochs": 5,
        "a1_a3_start_scale": 0.1,
        "max_nodes_for_a3": 768,
        "loss_type": "ce",
        "length_bucket_batching": True,
        "dataloader_prefetch_factor": 2,
        "persistent_workers": True,
    },
    "no_a2_from_scratch": {
        "num_classes": 3,
        "pt_subdir": "pyg",
        "drop_invalid_graphs": True,
        "batch_size": 64,
        "batch_node_sq_budget": 4000000,
        "precision": "16-mixed",
        "max_epochs": 100,
        "num_workers": 4,
        "pin_memory": True,
        "allow_tf32": True,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 2,
        "warmup_freeze_epochs": 0,
        "learning_rate": 0.002,
        "a1_a3_learning_rate": 0.002,
        "optimizer_warmup_steps": 1000,
        "a1_a3_ramp_epochs": 0,
        "max_nodes_for_a3": 768,
        "loss_type": "ce",
        "length_bucket_batching": True,
        "dataloader_prefetch_factor": 2,
        "persistent_workers": True,
    },
    "new_abc_finetune": {
        "num_classes": 3,
        "pt_subdir": "pyg",
        "drop_invalid_graphs": True,
        "batch_size": 64,
        "batch_node_sq_budget": 4000000,
        "precision": "16-mixed",
        "max_epochs": 15,
        "num_workers": 4,
        "pin_memory": True,
        "allow_tf32": True,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 1,
        "warmup_freeze_epochs": 0,
        "learning_rate": 0.0001,
        "a1_a3_learning_rate": 0.0001,
        "optimizer_warmup_steps": 500,
        "a1_a3_ramp_epochs": 0,
        "max_nodes_for_a3": 768,
        "loss_type": "ce",
        "length_bucket_batching": True,
        "csv_log": True,
        "dataloader_prefetch_factor": 2,
        "persistent_workers": True,
    },
    "model_a_unique_abc_finetune": {
        "num_classes": 3,
        "pt_subdir": "pyg",
        "batch_size": 64,
        "batch_node_sq_budget": 4000000,
        "precision": "16-mixed",
        "max_epochs": 8,
        "num_workers": 4,
        "pin_memory": True,
        "allow_tf32": True,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 1,
        "warmup_freeze_epochs": 0,
        "learning_rate": 0.00002,
        "a1_a3_learning_rate": 0.00002,
        "optimizer_warmup_steps": 500,
        "a1_a3_ramp_epochs": 0,
        "max_nodes_for_a3": 768,
        "loss_type": "ce",
        "length_bucket_batching": True,
        "csv_log": True,
        "dataloader_prefetch_factor": 2,
        "persistent_workers": True,
    },
}


def _collect_explicit_cli_args(the_parser: argparse.ArgumentParser) -> set[str]:
    """Identify which argument *dest* names were explicitly provided on the CLI.

    This works by checking sys.argv for any option string that maps to a known
    dest.  The result lets _apply_training_profile skip profile defaults for
    anything the user explicitly typed.
    """
    argv_set = set(sys.argv[1:])
    explicit: set[str] = set()
    for action in the_parser._actions:
        if any(opt in argv_set for opt in action.option_strings):
            explicit.add(action.dest)
    return explicit


def _apply_training_profile(
    args: argparse.Namespace, the_parser: argparse.ArgumentParser
) -> None:
    """Apply a training profile's defaults for any arg not explicitly set on the CLI."""
    if args.training_profile == "custom":
        return
    profile = TRAINING_PROFILES[args.training_profile]
    explicit = _collect_explicit_cli_args(the_parser)
    applied = []
    for key, value in profile.items():
        if key not in explicit:
            setattr(args, key, value)
            applied.append(key)
    if applied:
        print(
            f"Training profile '{args.training_profile}' applied "
            f"{len(applied)} default(s): {', '.join(sorted(applied))}",
            flush=True,
        )


parser = argparse.ArgumentParser("BrepMFR Network model")
parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
parser.add_argument(
    "--training_profile",
    type=str,
    default="custom",
    choices=["custom"] + list(TRAINING_PROFILES.keys()),
    help=(
        "Pre-baked argument preset that replaces the old .ps1 launcher scripts. "
        "Pick a profile and optionally override individual settings on the CLI. "
        "'custom' (default) applies no overrides. "
        "See docs/training_recipes.md for details."
    ),
)
parser.add_argument("--num_classes", type=int, default=25, help="Number of features")
parser.add_argument("--dataset", choices=("cadsynth", "transfer"), default="cadsynth", help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument(
    "--accumulate_grad_batches",
    type=int,
    default=1,
    metavar="N",
    help=(
        "Lightning accumulate_grad_batches: optimizer step every N micro-batches. "
        "Use e.g. --batch_size 1 --accumulate_grad_batches 32 to cap attention memory "
        "(max faces = largest graph in the micro-batch) while keeping effective batch 32."
    ),
)
parser.add_argument(
    "--precision",
    type=str,
    default="32",
    choices=("32", "16-mixed", "bf16-mixed"),
    help="Lightning precision. 16-mixed / bf16-mixed reduce activation memory on CUDA.",
)
parser.add_argument(
    "--max_graph_nodes",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Drop graphs with more than N faces after split resolution (loads each .pt once at startup). "
        "Self-attention memory scales with max N in the batch; use with smaller --batch_size if OOM."
    ),
)
parser.add_argument(
    "--drop_invalid_graphs",
    action="store_true",
    help=(
        "At dataset init, drop .pt with zero faces or empty label_feature (one load per listed graph). "
        "Use with SolidWorks/thread exports that occasionally emit empty step graphs. "
        "Also runs when --max_graph_nodes is set (same pass drops oversize graphs)."
    ),
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=12,
    help=(
        "DataLoader worker processes. On Windows start with 2 and prefetch_factor=2; "
        "fall back to 0 if host commit memory is exhausted."
    ),
)
parser.add_argument(
    "--pin_memory",
    action="store_true",
    help=(
        "DataLoader pin_memory=True (recommended on CUDA for faster async H→D staging; "
        "increases host RAM slightly). Default off for parity with older Windows-stable runs."
    ),
)
parser.add_argument(
    "--dataloader_prefetch_factor",
    type=int,
    default=None,
    metavar="N",
    help=(
        "DataLoader prefetch_factor when num_workers>0 (default: 1 — conservative RAM). "
        "Try 2 (PyTorch default) or 4 if workers keep the GPU idle after enabling --pin_memory."
    ),
)
parser.add_argument(
    "--persistent_workers",
    action="store_true",
    help=(
        "Keep DataLoader worker processes alive across epochs. Recommended with "
        "the bounded adaptive batch budget; disable if Windows commit memory is tight."
    ),
)
parser.add_argument(
    "--length_bucket_batching",
    action="store_true",
    help=(
        "Use size-aware batching. Without --batch_node_sq_budget this uses legacy "
        "<=150/<=300/>300 buckets; with a budget it adaptively packs by padded N^2 cost."
    ),
)
parser.add_argument(
    "--batch_node_sq_budget",
    type=int,
    default=0,
    metavar="N2",
    help=(
        "With --length_bucket_batching, greedily pack similar-size graphs while "
        "batch_size * padded_max_nodes^2 <= N2. A value such as 4000000 replaces "
        "the coarse >300-faces batch-size-1 rule and substantially reduces steps."
    ),
)
parser.add_argument(
    "--cuda_launch_blocking",
    action="store_true",
    help="Set CUDA_LAUNCH_BLOCKING=1 before training (CUDA debug only — large slowdown).",
)
parser.add_argument(
    "--allow_tf32",
    action="store_true",
    help="Enable TF32 for remaining float32 CUDA matmuls (faster on Ampere/Hopper GPUs).",
)
parser.add_argument(
    "--cudnn_benchmark",
    action="store_true",
    help="Let cuDNN autotune the fixed-size face/edge convolution kernels.",
)
parser.add_argument(
    "--fused_adamw",
    action="store_true",
    help="Use PyTorch's fused CUDA AdamW implementation when available.",
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
        "Use an explicit date+time(+ms) slug to avoid overwriting runs "
        "(e.g. ce_weighted_balanced_skip_a2__2026-05-10_143022041). "
        "Default auto name: ce_weighted_balanced__YYYY-MM-DD_HHMMSS_mmm "
        "(local clock, millisecond field)."
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
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    metavar="PATH",
    help=(
        "Resume Lightning training from last.ckpt (optimizer + epoch + step). "
        "Keep the same --run_name so checkpoints and logs stay in one folder."
    ),
)
parser.add_argument(
    "--full_a1_a3_from_scratch",
    action="store_true",
    help=(
<<<<<<< HEAD
        "On train: randomly initialize and require no-A2 graphs with A1/A3. "
        "On test: only require that same graph contract (A1+A3 present, no A2). "
        "Train mode rejects --pre_train/--resume_from_checkpoint, forces A1/A3 fully "
        "active from epoch 0, and disables encoder freezing."
=======
        "Start a new randomly initialized model and require no-A2 graphs with A1/A3. "
        "This mode rejects --pre_train/--resume_from_checkpoint, forces A1/A3 fully "
        "active from epoch 0, and disables encoder freezing. Omit this flag to keep "
        "the legacy initialization behavior."
>>>>>>> ba9ceddb4df8f2e01d2036bdbff47f1eab4afd2d
    ),
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help=(
        "Optional reproducibility seed for model initialization, data workers, and "
        "samplers. --full_a1_a3_from_scratch defaults this to 42 when omitted. "
        "Legacy runs remain unseeded unless this argument is supplied."
    ),
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
parser.add_argument(
    "--batchnorm_finetune_mode",
    choices=("update", "freeze_stats", "freeze_all"),
    default="update",
    help=(
        "BatchNorm behavior during training. 'update' preserves the legacy "
        "behavior. 'freeze_stats' keeps pretrained running_mean/running_var "
        "fixed while allowing affine weight/bias updates. 'freeze_all' also "
        "freezes BatchNorm affine parameters; recommended for controlled "
        "fine-tuning when the target dataset previously caused BN domain drift."
    ),
)
parser.add_argument(
    "--learning_rate",
    "--learning-rate",
    type=float,
    default=0.002,
    help="AdamW learning rate for the pretrained backbone/classifier (default: 0.002).",
)
parser.add_argument(
    "--a1_a3_learning_rate",
    "--a1-a3-learning-rate",
    type=float,
    default=None,
    help=(
        "Optional separate AdamW learning rate for brep_encoder.graph_attn_bias. "
        "Use a higher value than --learning_rate when introducing previously unused A1/A3 branches."
    ),
)
parser.add_argument(
    "--optimizer_warmup_steps",
    "--optimizer-warmup-steps",
    type=int,
    default=5000,
    help="Linearly warm each optimizer parameter group to its configured learning rate.",
)
parser.add_argument(
    "--a1_a3_ramp_epochs",
    "--a1-a3-ramp-epochs",
    type=int,
    default=0,
    help=(
        "Gradually increase A1/A3 attention-bias contribution to 1.0 over this many epochs. "
        "Use 5 when fine-tuning a lite checkpoint; 0 applies full A1/A3 immediately."
    ),
)
parser.add_argument(
    "--a1_a3_start_scale",
    "--a1-a3-start-scale",
    type=float,
    default=0.1,
    help="Initial A1/A3 multiplier when --a1_a3_ramp_epochs is enabled (default: 0.1).",
)
parser.add_argument(
    "--max_nodes_for_a3",
    "--max-nodes-for-a3",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Skip dense A3 collation/encoding for batches padded above N faces while retaining A1. "
        "Recommended: 768 for mixed-size training. Use 0 for no cap."
    ),
)
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
parser.add_argument(
    "--reuse_checkpoint_class_weights",
    action="store_true",
    help=(
        "For --pre_train/--resume_from_checkpoint, enable weighted loss using "
        "the class_weights buffer embedded in that checkpoint. This is safer "
        "for controlled reproduction than a JSON path whose contents may have "
        "changed. Cannot be combined with --class_weights_path."
    ),
)
parser.add_argument(
    "--loss_type",
    type=str,
    default="ce",
    choices=("ce", "focal"),
    help=(
        "Loss function for training. "
        "'ce' = standard weighted cross-entropy (default). "
        "'focal' = Focal Loss (Lin et al., ICCV 2017) which dynamically "
        "down-weights easy examples via a (1-p_t)^gamma factor — effective "
        "for severe class imbalance (e.g., 85%% text vs 0.8%% thread). "
        "Combines with --class_weights_path for both easy-example suppression "
        "and class-frequency correction."
    ),
)
parser.add_argument(
    "--focal_gamma",
    type=float,
    default=2.0,
    help=(
        "Focal Loss focusing parameter (only used when --loss_type focal). "
        "gamma=0 reduces to CE; gamma=2.0 is the paper default. "
        "Higher gamma more aggressively suppresses easy examples."
    ),
)
parser.add_argument(
    "--pt_subdir",
    type=str,
    default=None,
    help=(
        "Load only `<dataset_path>/<pt_subdir>/**/*.pt` (split lists still resolved from dataset_path). "
        "Example: `output/bin_skip_a2` for zero-A2 ablation graphs."
    ),
)

# --------------------------------------------------------------------------------------
# Subgraph training (k-hop neighborhoods). Completely opt-in.
# When disabled (default), CADSynth returns full graphs exactly as before.
# --------------------------------------------------------------------------------------
parser.add_argument(
    "--subgraph_training",
    action="store_true",
    help=(
        "Enable subgraph training: for each loaded graph, sample a small number of seed faces "
        "(balanced across classes) and train only on their k-hop union neighborhood. "
        "This is one of the most effective ways to combat extreme face-level class imbalance "
        "(e.g. huge text regions vs small thread regions) because you control *how many seeds* "
        "of each class the model sees per step instead of how many faces a feature happens to have."
    ),
)
parser.add_argument(
    "--subgraph_k_hop",
    type=int,
    default=2,
    help="Hop radius for subgraph extraction (2 or 3 recommended). Larger = more context, smaller = faster + more focused.",
)
parser.add_argument(
    "--subgraph_seeds_per_class",
    type=str,
    default="2,3,3",
    help=(
        "Comma-separated max seeds to draw per class (positional). "
        "Example for a 3-class thread/text problem: '2,3,3' means up to 2 stock + 3 thread + 3 text seeds per original graph. "
        "Fewer seeds are taken if a class is absent in that part. "
        "Use '0:2,1:3,2:3' syntax for explicit class ids if you have >3 classes."
    ),
)
parser.add_argument(
    "--subgraph_on_val",
    action="store_true",
    help="Also apply subgraph sampling to the validation set (default: val uses full graphs for comparable metrics).",
)
parser.add_argument(
    "--subgraph_on_test",
    action="store_true",
    help="Also apply subgraph sampling at test time (rarely useful; default keeps full graphs).",
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
        "results/logs/stage1/<run_name>/csv_metrics/."
    ),
)
parser.add_argument(
    "--train_log_file",
    type=str,
    default=None,
    metavar="PATH",
    help=(
        "Append Unicode console stdout/stderr to PATH (training only). "
        "Use with long runs so PowerShell tail has a stable file."
    ),
)
parser.add_argument(
    "--archive_lightning_logs",
    action="store_true",
    help=(
        "Before training: move tensorboard/version_0 and csv_metrics/version_0 to "
        "results/logs/_lightning_log_archive/<run_name>/<timestamp>/ (outside this run's logdir). "
        "TensorBoard merges every events file under --logdir recursively; archiving inside the "
        "run folder does not remove bad curves. Use once after polluted smoke resumes."
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
    help="Caps validation batches each epoch (Lightning). Omit for full val.",
)
parser.add_argument(
    "--num_sanity_val_steps",
    type=int,
    default=2,
    metavar="N",
    help="Lightning sanity validation batches before fit (2 default). Use 0 to skip if stuck after sanity.",
)
parser.add_argument(
    "--check_val_every_n_epoch",
    type=int,
    default=1,
    metavar="N",
    help="Run full validation every N epochs (default: 1).",
)
parser.add_argument(
    "--tb_surrogate_trace",
    action="store_true",
    help=(
        "Run TorchInfo + TensorBoard add_graph on a tiny attention/classifier surrogate at train start. "
        "Off by default: can stall before the first epoch on Windows+CUDA with live modules."
    ),
)
parser.add_argument(
    "--tb_full_graph",
    action="store_true",
    help=(
        "Log an extra TensorBoard GRAPH for BrepEncoder+attention+classifier using one "
        "small real graph from the dataset (bounded node/edge counts)."
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

def _sanitize_run_name(name: str) -> str:
    """Filesystem-safe single path segment (no separators or odd chars)."""
    n = re.sub(r'[<>:"/\\|?*]+', "_", name.strip())
    return n or "unnamed_run"


def _archive_lightning_version_dirs(logs_save_dir: pathlib.Path) -> pathlib.Path | None:
    """Move Lightning TB/CSV version_0 trees outside logs_save_dir so TB stops merging stale scalars."""
    run_name = logs_save_dir.name
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
    stage_dir = logs_save_dir.parent
    logs_root = stage_dir.parent
    archive_root = logs_root.joinpath("_lightning_log_archive", run_name, ts)
    moved_any = False
    for sub in ("tensorboard", "csv_metrics"):
        v0 = logs_save_dir / sub / "version_0"
        if not v0.is_dir():
            continue
        try:
            has_files = any(v0.iterdir())
        except OSError:
            continue
        if not has_files:
            continue
        dest = archive_root / sub / "version_0"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(v0), str(dest))
        v0.mkdir(parents=True, exist_ok=True)
        moved_any = True
    return archive_root if moved_any else None


def _maybe_tee_train_log(path_str: str) -> None:
    path = pathlib.Path(path_str).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = path.open("a", encoding="utf-8")

    class _StdoutTee:
        __slots__ = ()

        def write(self, data) -> None:
            if data:
                sys.__stdout__.write(data)
                log_fp.write(data)
                log_fp.flush()
                sys.__stdout__.flush()

        def flush(self) -> None:
            sys.__stdout__.flush()
            log_fp.flush()

        def __getattr__(self, item):
            return getattr(sys.__stdout__, item)

    class _StderrTee:
        __slots__ = ()

        def write(self, data) -> None:
            if data:
                sys.__stderr__.write(data)
                log_fp.write(data)
                log_fp.flush()
                sys.__stderr__.flush()

        def flush(self) -> None:
            sys.__stderr__.flush()
            log_fp.flush()

        def __getattr__(self, item):
            return getattr(sys.__stderr__, item)

    atexit.register(log_fp.close)
    sys.stdout = _StdoutTee()
    sys.stderr = _StderrTee()


def main():
    # On Windows the DataLoader uses spawn(); without an `if __name__ == "__main__"`
    # guard around training setup, every worker re-imports this module and tries to
    # restart training, eventually crashing in `_check_not_importing_main`.
    args = parser.parse_args()
<<<<<<< HEAD
    _apply_training_profile(args, parser)
    if args.full_a1_a3_from_scratch:
        if args.traintest == "test":
            args.initialization_mode = "full_a1_a3_from_scratch"
            print(
                "Test: requiring no-A2 graphs with A1+A3 (--full_a1_a3_from_scratch).",
                flush=True,
            )
        else:
            if args.pre_train or args.resume_from_checkpoint:
                parser.error(
                    "--full_a1_a3_from_scratch cannot be combined with "
                    "--pre_train or --resume_from_checkpoint"
                )
            if args.warmup_freeze_epochs != 0:
                print(
                    "Scratch A1/A3 mode: overriding --warmup_freeze_epochs to 0.",
                    flush=True,
                )
            if args.a1_a3_ramp_epochs != 0:
                print(
                    "Scratch A1/A3 mode: overriding --a1_a3_ramp_epochs to 0.",
                    flush=True,
                )
            args.warmup_freeze_epochs = 0
            args.a1_a3_ramp_epochs = 0
            if args.a1_a3_learning_rate is None:
                args.a1_a3_learning_rate = float(args.learning_rate)
            if args.seed is None:
                args.seed = 42
            args.initialization_mode = "full_a1_a3_from_scratch"
            print(
                "Initialization mode: random weights, no checkpoint, "
                "A1/A3 fully active from epoch 0.",
                flush=True,
            )
=======
    if args.full_a1_a3_from_scratch:
        if args.traintest != "train":
            parser.error("--full_a1_a3_from_scratch is valid only with the train command")
        if args.pre_train or args.resume_from_checkpoint:
            parser.error(
                "--full_a1_a3_from_scratch cannot be combined with "
                "--pre_train or --resume_from_checkpoint"
            )
        if args.warmup_freeze_epochs != 0:
            print(
                "Scratch A1/A3 mode: overriding --warmup_freeze_epochs to 0.",
                flush=True,
            )
        if args.a1_a3_ramp_epochs != 0:
            print(
                "Scratch A1/A3 mode: overriding --a1_a3_ramp_epochs to 0.",
                flush=True,
            )
        args.warmup_freeze_epochs = 0
        args.a1_a3_ramp_epochs = 0
        if args.a1_a3_learning_rate is None:
            args.a1_a3_learning_rate = float(args.learning_rate)
        if args.seed is None:
            args.seed = 42
        args.initialization_mode = "full_a1_a3_from_scratch"
        print(
            "Initialization mode: random weights, no checkpoint, "
            "A1/A3 fully active from epoch 0.",
            flush=True,
        )
>>>>>>> ba9ceddb4df8f2e01d2036bdbff47f1eab4afd2d
    elif args.resume_from_checkpoint:
        args.initialization_mode = "exact_resume"
    elif args.pre_train:
        args.initialization_mode = "pretrained_finetune"
    else:
        args.initialization_mode = "legacy_scratch"

    if args.seed is not None:
        seed_everything(int(args.seed), workers=True)
        print(f"Reproducibility seed: {int(args.seed)}", flush=True)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("CUDA TF32 enabled for float32 matmuls.", flush=True)
    if args.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN convolution autotuning enabled.", flush=True)
    if args.fused_adamw:
        print(
            "WARNING: fused AdamW requested; disabling Lightning gradient clipping "
            "because AMP fused optimizers unscale gradients internally.",
            flush=True,
        )
    if args.pre_train and args.resume_from_checkpoint:
        parser.error("Use only one of --pre_train (fresh fine-tune state) or --resume_from_checkpoint (exact resume).")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be > 0")
    if args.a1_a3_learning_rate is not None and args.a1_a3_learning_rate <= 0:
        parser.error("--a1_a3_learning_rate must be > 0")
    if args.optimizer_warmup_steps < 0:
        parser.error("--optimizer_warmup_steps must be >= 0")
    if (
        args.traintest == "train"
        and args.batchnorm_finetune_mode != "update"
        and not (args.pre_train or args.resume_from_checkpoint)
    ):
        parser.error(
            "--batchnorm_finetune_mode freeze_stats/freeze_all requires "
            "--pre_train or --resume_from_checkpoint; freezing randomly "
            "initialized BatchNorm statistics is not a valid fine-tune."
        )
    if args.reuse_checkpoint_class_weights:
        if not (args.pre_train or args.resume_from_checkpoint):
            parser.error(
                "--reuse_checkpoint_class_weights requires --pre_train or "
                "--resume_from_checkpoint"
            )
        if args.class_weights_path:
            parser.error(
                "Use either --reuse_checkpoint_class_weights or "
                "--class_weights_path, not both"
            )
    if args.a1_a3_ramp_epochs < 0:
        parser.error("--a1_a3_ramp_epochs must be >= 0")
    if not 0.0 <= args.a1_a3_start_scale <= 1.0:
        parser.error("--a1_a3_start_scale must be in [0, 1]")
    if args.max_nodes_for_a3 is not None and args.max_nodes_for_a3 <= 0:
        args.max_nodes_for_a3 = None
    if args.traintest == "train" and args.train_log_file:
        _maybe_tee_train_log(args.train_log_file)
    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

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
        logs_path = repo_root.joinpath(
            "results", "logs", "stage1", args.run_name
        )
        logs_path.mkdir(parents=True, exist_ok=True)
        if args.archive_lightning_logs:
            arch = _archive_lightning_version_dirs(logs_path)
            if arch is not None:
                print(f"Archived Lightning TB/CSV version_0 under: {arch}", flush=True)

        checkpoint_callback = ModelCheckpoint(
            monitor="eval_loss",
            dirpath=str(results_path),
            filename="best",
            save_top_k=10,
            save_last=True,
            every_n_epochs=max(1, int(args.check_val_every_n_epoch)),
        )

        callbacks = build_train_callbacks(
            checkpoint=checkpoint_callback,
            stage="stage1",
            dim_node=args.dim_node,
            hyperparam_extras={
                "dataset_path": args.dataset_path,
                "class_weights_path": args.class_weights_path,
                "reuse_checkpoint_class_weights": bool(
                    args.reuse_checkpoint_class_weights
                ),
                "pt_subdir": args.pt_subdir,
                "checkpoint_dir": str(results_path),
                "logs_dir": str(logs_path),
                "multi_hop_max_dist": 16,
                "spatial_pos_max": 32,
                "pin_memory": bool(args.pin_memory),
                "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
                "persistent_workers": bool(args.persistent_workers),
                "batch_node_sq_budget": int(args.batch_node_sq_budget),
                "fused_adamw": bool(args.fused_adamw),
                "cudnn_benchmark": bool(args.cudnn_benchmark),
                "accumulate_grad_batches": int(args.accumulate_grad_batches),
                "precision": args.precision,
                "max_graph_nodes": args.max_graph_nodes,
                "drop_invalid_graphs": bool(args.drop_invalid_graphs),
                "learning_rate": float(args.learning_rate),
                "a1_a3_learning_rate": args.a1_a3_learning_rate,
                "optimizer_warmup_steps": int(args.optimizer_warmup_steps),
                "a1_a3_ramp_epochs": int(args.a1_a3_ramp_epochs),
                "a1_a3_start_scale": float(args.a1_a3_start_scale),
                "max_nodes_for_a3": args.max_nodes_for_a3,
                "batchnorm_finetune_mode": args.batchnorm_finetune_mode,
                "initialization_mode": args.initialization_mode,
                "seed": args.seed,
                "require_no_a2_a1_a3": bool(args.full_a1_a3_from_scratch),
                "subgraph_training": bool(args.subgraph_training),
                "subgraph_k_hop": int(args.subgraph_k_hop),
                "subgraph_seeds_per_class": args.subgraph_seeds_per_class,
                "subgraph_on_val": bool(args.subgraph_on_val),
                "subgraph_on_test": bool(args.subgraph_on_test),
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

        if args.limit_train_batches is not None or args.limit_val_batches is not None:
            print(
                "WARNING: --limit_* batches shorten each epoch but epochs still increment; "
                "ModelCheckpoint can advance last.ckpt to max_epochs after quick smoke runs.",
                flush=True,
            )

        tk = dict(
            max_epochs=args.max_epochs,
            log_every_n_steps=args.log_every_n_steps,
            callbacks=callbacks,
            logger=loggers,
            accelerator="gpu",
            devices=1,
            # Lightning AMP cannot externally unscale/clip gradients for fused
            # AdamW because that optimizer owns unscaling internally.
            gradient_clip_val=0.0 if args.fused_adamw else 1.0,
            num_sanity_val_steps=int(args.num_sanity_val_steps),
            accumulate_grad_batches=int(args.accumulate_grad_batches),
            check_val_every_n_epoch=max(1, int(args.check_val_every_n_epoch)),
            benchmark=bool(args.cudnn_benchmark),
        )
        if args.precision != "32":
            tk["precision"] = args.precision
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
B-rep model feature recognition (Stage 1)
-----------------------------------------------------------------------------------
Run folder:
  results/stage1/{args.run_name}/     (best.ckpt / last.ckpt only)

TensorBoard (+ optional CSV / W&B) logs:
  results/logs/stage1/{args.run_name}/tensorboard/
  tensorboard --logdir results/logs/stage1/{args.run_name}/
  (use run folder root to include PROFILE traces when --tb_profile is set)

Best checkpoint:
  results/stage1/{args.run_name}/best.ckpt
-----------------------------------------------------------------------------------
        """
        )
        if args.pre_train is not None:
            pre_path = pathlib.Path(args.pre_train).expanduser().resolve()
            if not pre_path.is_file():
                raise FileNotFoundError(f"--pre_train checkpoint not found: {pre_path}")
            print(f"Loading pre-trained weights from: {pre_path}")
            model = BrepSeg.load_from_checkpoint(str(pre_path), args=args, strict=False)
        else:
            model = BrepSeg(args)

        # Stash a back-reference so the model can advance subgraph_epoch each epoch
        # (gives different random crops of the same part across epochs when using
        # --subgraph_training). No effect when subgraph_training is off.
        model._train_dataset_for_subgraph = None
        dataset_profile_kwargs = {}
        if args.full_a1_a3_from_scratch:
            # Omit this newer validation hook for legacy/fine-tune modes so an
            # older compatible CADSynth constructor continues to work.
            dataset_profile_kwargs["require_no_a2_a1_a3"] = True

        train_data = Dataset(
            root_dir=args.dataset_path,
            split="train",
            random_rotate=True,
            num_class=args.num_classes,
            pt_subdir=args.pt_subdir,
            max_graph_nodes=args.max_graph_nodes,
            max_nodes_for_a3=args.max_nodes_for_a3,
            drop_invalid_graphs=args.drop_invalid_graphs,
            # Subgraph training (defaults keep old full-graph behavior)
            subgraph_training=args.subgraph_training,
            subgraph_k_hop=args.subgraph_k_hop,
            subgraph_seeds_per_class=args.subgraph_seeds_per_class,
            subgraph_on_nontrain=args.subgraph_on_val,  # val controlled separately
            subgraph_global_seed=42,
            **dataset_profile_kwargs,
        )
        model._train_dataset_for_subgraph = train_data
        val_data = Dataset(
            root_dir=args.dataset_path,
            split="val",
            random_rotate=False,
            num_class=args.num_classes,
            pt_subdir=args.pt_subdir,
            max_graph_nodes=args.max_graph_nodes,
            max_nodes_for_a3=args.max_nodes_for_a3,
            drop_invalid_graphs=args.drop_invalid_graphs,
            subgraph_training=args.subgraph_training and args.subgraph_on_val,
            subgraph_k_hop=args.subgraph_k_hop,
            subgraph_seeds_per_class=args.subgraph_seeds_per_class,
            subgraph_on_nontrain=args.subgraph_on_val,
            subgraph_global_seed=42,
            **dataset_profile_kwargs,
        )
        train_loader = train_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor,
            persistent_workers=args.persistent_workers,
            length_bucket_batching=args.length_bucket_batching,
            batch_node_sq_budget=args.batch_node_sq_budget,
        )
        val_loader = val_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor,
            persistent_workers=args.persistent_workers,
            length_bucket_batching=args.length_bucket_batching,
            batch_node_sq_budget=args.batch_node_sq_budget,
        )
        if len(train_data) == 0:
            raise RuntimeError(
                "Train dataset is empty after filters (--drop_invalid_graphs / --max_graph_nodes / split lists / .pt scan). "
                "Relax filters or fix dataset_path and train.txt."
            )
        fit_kw = {}
        if args.resume_from_checkpoint:
            ckpt_path = pathlib.Path(args.resume_from_checkpoint).expanduser().resolve()
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"--resume_from_checkpoint not found: {ckpt_path}")
            fit_kw["ckpt_path"] = str(ckpt_path)
            print(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.fit(model, train_loader, val_loader, **fit_kw)

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
        dataset_profile_kwargs = {}
        if args.full_a1_a3_from_scratch:
            dataset_profile_kwargs["require_no_a2_a1_a3"] = True
        test_data = Dataset(
            root_dir=args.dataset_path,
            split="test",
            random_rotate=False,
            num_class=args.num_classes,
            pt_subdir=args.pt_subdir,
            max_graph_nodes=args.max_graph_nodes,
            max_nodes_for_a3=args.max_nodes_for_a3,
            drop_invalid_graphs=args.drop_invalid_graphs,
            subgraph_training=args.subgraph_training and args.subgraph_on_test,
            subgraph_k_hop=args.subgraph_k_hop,
            subgraph_seeds_per_class=args.subgraph_seeds_per_class,
            subgraph_on_nontrain=args.subgraph_on_test,
            subgraph_global_seed=42,
            **dataset_profile_kwargs,
        )
        test_loader = test_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor,
        )
        model = BrepSeg.load_from_checkpoint(args.checkpoint)
        trainer.test(model, dataloaders=test_loader, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    # Required on Windows so that DataLoader workers spawned with `spawn()` can
    # safely import this module without re-running training.
    from multiprocessing import freeze_support
    freeze_support()
    main()

