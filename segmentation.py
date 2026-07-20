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
    # We intentionally omit persistent_workers on Windows (see data/dataset.py).
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
from pytorch_lightning import Trainer
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

parser = argparse.ArgumentParser("BrepMFR Network model")
parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
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
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
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
    "--length_bucket_batching",
    action="store_true",
    help=(
        "Use a length-bucketed batch sampler: graphs with <=150 faces use --batch_size, "
        "151-300 faces use batch_size//2, >300 faces use batch_size=1. Prevents OOM spikes "
        "from O(N^2) attention on large graphs while keeping ALL training data (including "
        "the 0.09%% of graphs with >500 faces). Recommended when training on mixed-size "
        "BrepMFR data and hitting intermittent CUDA OOM at random epochs."
    ),
)
parser.add_argument(
    "--cuda_launch_blocking",
    action="store_true",
    help="Set CUDA_LAUNCH_BLOCKING=1 before training (CUDA debug only — large slowdown).",
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
    if args.pre_train and args.resume_from_checkpoint:
        parser.error("Use only one of --pre_train (fresh fine-tune state) or --resume_from_checkpoint (exact resume).")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be > 0")
    if args.a1_a3_learning_rate is not None and args.a1_a3_learning_rate <= 0:
        parser.error("--a1_a3_learning_rate must be > 0")
    if args.optimizer_warmup_steps < 0:
        parser.error("--optimizer_warmup_steps must be >= 0")
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
        )

        callbacks = build_train_callbacks(
            checkpoint=checkpoint_callback,
            stage="stage1",
            dim_node=args.dim_node,
            hyperparam_extras={
                "dataset_path": args.dataset_path,
                "class_weights_path": args.class_weights_path,
                "pt_subdir": args.pt_subdir,
                "checkpoint_dir": str(results_path),
                "logs_dir": str(logs_path),
                "multi_hop_max_dist": 16,
                "spatial_pos_max": 32,
                "pin_memory": bool(args.pin_memory),
                "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
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
            gradient_clip_val=1.0,
            num_sanity_val_steps=int(args.num_sanity_val_steps),
            accumulate_grad_batches=int(args.accumulate_grad_batches),
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
        # if args.pre_train is not None:
        #     model = BrepSeg.load_from_checkpoint(args.pre_train, args=args, strict=False)
        # else:
        #     model = BrepSeg(args)

        model = BrepSeg(args)

        # Stash a back-reference so the model can advance subgraph_epoch each epoch
        # (gives different random crops of the same part across epochs when using
        # --subgraph_training). No effect when subgraph_training is off.
        model._train_dataset_for_subgraph = None

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
        )
        train_loader = train_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor,
            length_bucket_batching=args.length_bucket_batching,
        )
        val_loader = val_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor,
            length_bucket_batching=args.length_bucket_batching,
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

