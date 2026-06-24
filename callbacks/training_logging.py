# -*- coding: utf-8 -*-
"""Lightning callbacks and logger factories for Stage 1 / Stage 2 training."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn as nn
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger


def git_short_sha(cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


class TrainingMetaLoggerCallback(Callback):
    """Log dataset sizes and environment metadata to TensorBoard (text + scalars)."""

    def __init__(
        self,
        *,
        stage: str,
        hyperparam_extras: Optional[dict[str, Any]] = None,
        repo_root: Optional[Path] = None,
    ):
        self.stage = stage
        self.hyperparam_extras = dict(hyperparam_extras or {})
        self.repo_root = repo_root

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from models.tensorboard_media import tb_add_scalar, tb_add_text

        meta_lines = [f"stage={self.stage}", f"global_rank={trainer.global_rank}"]
        meta_lines.append(f"git_sha={git_short_sha(self.repo_root)}")

        hp_raw = getattr(pl_module, "hparams", None)
        hp_clean: dict[str, Any]
        if hp_raw is None:
            hp_clean = {}
        elif isinstance(hp_raw, dict):
            hp_clean = {str(k): v for k, v in hp_raw.items()}
        else:
            try:
                hp_clean = {str(k): v for k, v in dict(hp_raw).items()}
            except Exception:
                try:
                    hp_clean = {str(k): v for k, v in vars(hp_raw).items()}
                except Exception:
                    hp_clean = {"repr": str(hp_raw)}

        cw_path = hp_clean.get("class_weights_path") or self.hyperparam_extras.get(
            "class_weights_path"
        )
        if cw_path:
            meta_lines.append(f"class_weights_path={cw_path}")

        iwdan_src = hp_clean.get("iwdan_source_priors") or self.hyperparam_extras.get(
            "iwdan_source_priors"
        )
        iwdan_tgt = hp_clean.get("iwdan_target_priors") or self.hyperparam_extras.get(
            "iwdan_target_priors"
        )
        if iwdan_src:
            meta_lines.append(f"iwdan_source_priors={iwdan_src}")
        if iwdan_tgt:
            meta_lines.append(f"iwdan_target_priors={iwdan_tgt}")

        tb_add_text(trainer, "meta/hparams_json", json.dumps(hp_clean, indent=2, default=str))
        tb_add_text(trainer, "meta/run_notes", "\n".join(meta_lines))

        for lg in getattr(trainer, "loggers", []) or []:
            if hasattr(lg, "log_hyperparams"):
                try:
                    lg.log_hyperparams(hp_clean)
                except Exception:
                    pass

        # Dataset lengths (may be IterableDataset in edge cases — guard)
        step0 = int(trainer.global_step)
        dl_tr = trainer.train_dataloader
        dl_va = trainer.val_dataloaders
        if dl_tr is not None and hasattr(dl_tr, "dataset"):
            try:
                n = len(dl_tr.dataset)
                tb_add_scalar(trainer, "meta/train_dataset_len", float(n), step0)
            except TypeError:
                pass
        if dl_va is not None:
            dl = dl_va[0] if isinstance(dl_va, list) else dl_va
            if hasattr(dl, "dataset"):
                try:
                    n = len(dl.dataset)
                    tb_add_scalar(trainer, "meta/val_dataset_len", float(n), step0)
                except TypeError:
                    pass

        # Alpha from class-weights JSON if present
        if cw_path:
            p = Path(str(cw_path))
            if p.is_file():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    if "alpha" in obj:
                        tb_add_scalar(
                            trainer,
                            "meta/class_weights_alpha",
                            float(obj["alpha"]),
                            step0,
                        )
                except (json.JSONDecodeError, OSError, TypeError, ValueError):
                    pass


class ModelArchitectureLoggerCallback(Callback):
    """One-shot TB GRAPH for Attention+Classifier surrogate; TEXT summary of parameters."""

    def __init__(
        self,
        dim_node: int = 256,
        *,
        tb_full_graph: bool = False,
        tb_surrogate_trace: Optional[bool] = None,
        stage: str = "stage1",
        hyperparam_extras: Optional[dict[str, Any]] = None,
    ):
        self.dim_node = dim_node
        self.tb_full_graph = bool(tb_full_graph)
        if tb_surrogate_trace is None:
            env = os.environ.get("BREP_TB_SURROGATE_TRACE", "").strip().lower()
            tb_surrogate_trace = env in ("1", "true", "yes")
        self.tb_surrogate_trace = bool(tb_surrogate_trace)
        self.stage = stage
        self.hyperparam_extras = dict(hyperparam_extras or {})
        self._done = False

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._done:
            return
        self._done = True

        from models.tensorboard_media import tb_add_text

        # Parameter breakdown (always useful in TB TEXT tab)
        rows = []
        n_train = 0
        for name, p in pl_module.named_parameters():
            if p.requires_grad:
                rows.append(f"{name}\t{p.numel()}")
                n_train += int(p.numel())
        head = f"trainable_parameter_tensors\t{len(rows)}\ntrainable_elements\t{n_train}\n\n"
        tb_add_text(trainer, "model/trainable_params", head + "\n".join(rows[:400]))

        if not self.tb_surrogate_trace:
            tb_add_text(
                trainer,
                "model/torchinfo_attn_classifier",
                "(skipped) TorchInfo + TB add_graph for attention/classifier surrogate are OFF by "
                "default (they trace live CUDA modules and can hang before epoch 1 on some setups). "
                "Enable via --tb_surrogate_trace or env BREP_TB_SURROGATE_TRACE=1.",
            )
            if self.tb_full_graph and hasattr(pl_module, "brep_encoder"):
                self._maybe_log_full_encoder_graph(trainer, pl_module, tb_add_text)
            if hasattr(pl_module, "domain_adv"):
                self._maybe_log_domain_graphs(trainer, pl_module, tb_add_text)
            return

        if not hasattr(pl_module, "attention") or not hasattr(pl_module, "classifier"):
            tb_add_text(trainer, "model/add_graph", "No attention/classifier on module; skip attention/classifier surrogate graph.")
        else:
            class _Surrogate(nn.Module):
                def __init__(self, attention: nn.Module, classifier: nn.Module):
                    super().__init__()
                    self.attention = attention
                    self.classifier = classifier

                def forward(self, z: torch.Tensor) -> torch.Tensor:
                    z2 = self.attention([z, z])
                    return self.classifier(z2)

            device = pl_module.device
            surrogate = _Surrogate(pl_module.attention, pl_module.classifier).to(device)
            surrogate.eval()
            dummy = torch.randn(4, self.dim_node, device=device)

            try:
                from torchinfo import summary

                info_str = str(
                    summary(
                        surrogate,
                        input_data=dummy,
                        depth=6,
                        col_names=("num_params",),
                        row_settings=("var_names",),
                    )
                )
                tb_add_text(trainer, "model/torchinfo_attn_classifier", info_str[:48000])
            except Exception as exc:  # noqa: BLE001
                tb_add_text(
                    trainer,
                    "model/torchinfo_attn_classifier",
                    f"(skipped) {type(exc).__name__}: {exc}",
                )

            for lg in getattr(trainer, "loggers", []) or []:
                if isinstance(lg, TensorBoardLogger):
                    try:
                        lg.experiment.add_graph(surrogate, dummy)
                    except Exception as exc:  # noqa: BLE001
                        tb_add_text(
                            trainer,
                            "model/add_graph_error",
                            f"{type(exc).__name__}: {exc}",
                        )
                    break

        # ---- Full encoder + head trace (optional; bounded real batch from dataset) ----
        if self.tb_full_graph and hasattr(pl_module, "brep_encoder"):
            self._maybe_log_full_encoder_graph(trainer, pl_module, tb_add_text)

        # ---- Stage 2 domain discriminator (+ guarded GRL path) ----
        if hasattr(pl_module, "domain_adv"):
            self._maybe_log_domain_graphs(trainer, pl_module, tb_add_text)

    def _maybe_log_full_encoder_graph(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        tb_add_text,
    ) -> None:
        from pathlib import Path

        from models.tb_graph_utils import (
            EncoderSegTraceWrapper,
            batch_to_flat,
            move_batch_to_device,
            summarize_trace_batch,
            try_build_trace_batch_from_dataset,
        )

        extras = self.hyperparam_extras
        if self.stage == "stage2":
            ds_root = extras.get("source_path")
        else:
            ds_root = extras.get("dataset_path")
        pt_subdir = extras.get("pt_subdir")
        mhd = int(extras.get("multi_hop_max_dist", 16))
        spm = int(extras.get("spatial_pos_max", 32))

        if not ds_root:
            tb_add_text(
                trainer,
                "model/graph_brep_encoder_pipeline_note",
                "tb_full_graph: no dataset_path/source_path in hyperparam_extras; skip encoder graph.",
            )
            return

        if not hasattr(pl_module, "attention") or not hasattr(pl_module, "classifier"):
            tb_add_text(
                trainer,
                "model/graph_brep_encoder_pipeline_note",
                "tb_full_graph: module missing attention/classifier; skip encoder pipeline graph.",
            )
            return

        batch_cpu, note = try_build_trace_batch_from_dataset(
            Path(ds_root),
            pt_subdir,
            multi_hop_max_dist=mhd,
            spatial_pos_max=spm,
        )
        if batch_cpu is None:
            batch_cpu, note = try_build_trace_batch_from_dataset(
                Path(ds_root),
                pt_subdir,
                multi_hop_max_dist=mhd,
                spatial_pos_max=spm,
                max_nodes=512,
                max_edges=1024,
                max_files_to_scan=400,
            )
            note = f"(relaxed caps) {note}"
        if batch_cpu is None:
            tb_add_text(
                trainer,
                "model/graph_brep_encoder_pipeline_note",
                f"tb_full_graph: could not build trace batch: {note}",
            )
            return

        device = pl_module.device
        batch = move_batch_to_device(batch_cpu, device)

        wrap = EncoderSegTraceWrapper(
            pl_module.brep_encoder,
            pl_module.attention,
            pl_module.classifier,
        ).eval()
        flat_in = tuple(t.detach() for t in batch_to_flat(batch))
        tb_add_text(
            trainer,
            "model/graph_brep_encoder_pipeline_note",
            f"{note}\n{summarize_trace_batch(batch)}\ndevice={device}",
        )

        for lg in getattr(trainer, "loggers", []) or []:
            if isinstance(lg, TensorBoardLogger):
                try:
                    lg.experiment.add_graph(wrap, flat_in)
                except Exception as exc:  # noqa: BLE001
                    tb_add_text(
                        trainer,
                        "model/graph_brep_encoder_pipeline_error",
                        f"{type(exc).__name__}: {exc}",
                    )
                break

    def _maybe_log_domain_graphs(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        tb_add_text,
    ) -> None:
        da = getattr(pl_module, "domain_adv", None)
        if da is None:
            return
        dim = int(getattr(pl_module.hparams, "dim_node", self.dim_node))
        device = pl_module.device

        disc = getattr(da, "domain_discriminator", None)
        if disc is not None:
            dummy_n = torch.randn(8, dim, device=device)
            for lg in getattr(trainer, "loggers", []) or []:
                if isinstance(lg, TensorBoardLogger):
                    try:
                        lg.experiment.add_graph(disc, dummy_n)
                    except Exception as exc:  # noqa: BLE001
                        tb_add_text(
                            trainer,
                            "model/graph_domain_discriminator_error",
                            f"{type(exc).__name__}: {exc}",
                        )
                    break

        from models.tb_graph_utils import DomainGrlDiscTraceWrapper

        grl_wrap = DomainGrlDiscTraceWrapper(da).eval()
        f_cat = torch.randn(16, dim, device=device)
        for lg in getattr(trainer, "loggers", []) or []:
            if isinstance(lg, TensorBoardLogger):
                try:
                    lg.experiment.add_graph(grl_wrap, f_cat)
                except Exception as exc:  # noqa: BLE001
                    tb_add_text(
                        trainer,
                        "model/graph_domain_grl_disc_error",
                        f"(fallback ok) {type(exc).__name__}: {exc}",
                    )
                break


def build_train_callbacks(
    *,
    checkpoint: ModelCheckpoint,
    stage: str,
    dim_node: int,
    hyperparam_extras: Optional[dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
    tb_full_graph: bool = False,
    tb_surrogate_trace: Optional[bool] = None,
) -> List[Callback]:
    return [
        LearningRateMonitor(logging_interval="epoch"),
        TrainingMetaLoggerCallback(
            stage=stage,
            hyperparam_extras=hyperparam_extras,
            repo_root=repo_root,
        ),
        ModelArchitectureLoggerCallback(
            dim_node=dim_node,
            tb_full_graph=tb_full_graph,
            tb_surrogate_trace=tb_surrogate_trace,
            stage=stage,
            hyperparam_extras=hyperparam_extras,
        ),
        checkpoint,
    ]


def build_pytorch_profiler(
    logs_save_dir: Path | str,
    *,
    enabled: bool,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
    cuda_only: bool = False,
):
    """Lightning PyTorchProfiler writing traces for TensorBoard PROFILE tab, or None."""
    if not enabled:
        return None
    logdir = str(Path(logs_save_dir).resolve())
    try:
        import torch.profiler as torch_profiler
        try:
            from pytorch_lightning.profilers import PyTorchProfiler
        except ImportError:
            from lightning.pytorch.profilers import PyTorchProfiler
    except ImportError as exc:
        print(
            "WARNING: --tb_profile requested but pytorch_lightning/torch.profiler unavailable:",
            exc,
            flush=True,
        )
        return None

    cuda_avail = False
    try:
        cuda_avail = torch.cuda.is_available()
    except Exception:
        pass

    if cuda_only and cuda_avail:
        activities = [torch_profiler.ProfilerActivity.CUDA]
    elif cuda_only and not cuda_avail:
        print(
            "WARNING: --tb_profile_cuda_only requested but CUDA unavailable; using CPU-only activities.",
            flush=True,
        )
        activities = [torch_profiler.ProfilerActivity.CPU]
    else:
        activities = [torch_profiler.ProfilerActivity.CPU]
        if cuda_avail:
            activities.append(torch_profiler.ProfilerActivity.CUDA)

    schedule = torch_profiler.schedule(
        wait=max(0, int(wait)),
        warmup=max(0, int(warmup)),
        active=max(1, int(active)),
        repeat=max(1, int(repeat)),
    )
    trace_handler = torch_profiler.tensorboard_trace_handler(logdir)

    ctor_kw = dict(
        dirpath=logdir,
        filename="pytorch_profiler",
        schedule=schedule,
        activities=activities,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
    )
    try:
        return PyTorchProfiler(**ctor_kw)
    except TypeError:
        try:
            return PyTorchProfiler(
                dirpath=logdir,
                filename="pytorch_profiler",
                profiler_kwargs={
                    "schedule": schedule,
                    "activities": activities,
                    "on_trace_ready": trace_handler,
                    "record_shapes": True,
                    "profile_memory": True,
                },
            )
        except TypeError as exc:
            print(f"WARNING: could not construct PyTorchProfiler: {exc}", flush=True)
            return None


def build_loggers(
    *,
    logs_save_dir: Path,
    experiment_name: str,
    csv_log: bool,
    use_wandb: bool,
    wandb_project: Optional[str],
) -> List[Logger]:
    logs_save_dir = Path(logs_save_dir)
    logs_save_dir.mkdir(parents=True, exist_ok=True)
    # Pin version subdirs so resumed `Trainer.fit` keeps writing to tensorboard/version_0
    # and csv_metrics/version_0 (Lightning auto-increment otherwise → duplicate TB runs).
    loggers: List[Logger] = [
        TensorBoardLogger(
            save_dir=str(logs_save_dir),
            name="tensorboard",
            version=0,
        ),
    ]
    if csv_log:
        loggers.append(
            CSVLogger(
                save_dir=str(logs_save_dir),
                name="csv_metrics",
                version=0,
            ),
        )
    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger

            project = wandb_project or "brepmfr-pyg"
            loggers.append(
                WandbLogger(
                    project=project,
                    name=experiment_name,
                    save_dir=str(logs_save_dir),
                    log_model=False,
                )
            )
        except ImportError:
            print(
                "WARNING: --use_wandb set but wandb / WandbLogger not installed. "
                "Install with: pip install wandb",
                flush=True,
            )
    return loggers
