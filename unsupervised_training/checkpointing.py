from __future__ import annotations

import copy
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from models.brepseg_model import BrepSeg


def _namespace_from_checkpoint(checkpoint: dict[str, Any]) -> Namespace:
    hyperparameters = checkpoint.get("hyper_parameters")
    if not hyperparameters:
        raise ValueError("Stage-1 checkpoint is missing hyper_parameters")
    source = hyperparameters.get("args", hyperparameters)
    if isinstance(source, Namespace):
        values = vars(source)
    elif isinstance(source, dict):
        values = source
    else:
        values = vars(source)
    return Namespace(**copy.deepcopy(values))


def load_stage1_model(
    checkpoint_path: str | Path,
    *,
    max_nodes_for_a3: int | None,
    map_location: str | torch.device = "cpu",
) -> tuple[BrepSeg, dict[str, Any], Namespace]:
    """Load only deployable Stage-1 weights from a Lightning checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0 compatibility
        checkpoint = torch.load(path, map_location=map_location)
    if "state_dict" not in checkpoint:
        raise ValueError(f"Not a Lightning Stage-1 checkpoint: {path}")

    args = _namespace_from_checkpoint(checkpoint)
    # Disable initialization-time mutations and external training artifacts.
    args.pre_train = None
    args.resume_from_checkpoint = None
    args.warmup_freeze_epochs = 0
    args.a1_a3_ramp_epochs = 0
    args.max_nodes_for_a3 = max_nodes_for_a3
    args.class_weights_path = None
    args.reuse_checkpoint_class_weights = False
    args.batchnorm_finetune_mode = "update"
    args.fused_adamw = False
    args.check_val_every_n_epoch = 1

    model = BrepSeg(args)
    raw_state = checkpoint["state_dict"]
    deployable_prefixes = ("brep_encoder.", "attention.", "classifier.")
    state = {key: value for key, value in raw_state.items() if key.startswith(deployable_prefixes)}
    if "class_weights" in raw_state:
        state["class_weights"] = raw_state["class_weights"]
    if not state:
        raise ValueError(f"Checkpoint has no Stage-1 model weights: {path}")

    result = model.load_state_dict(state, strict=False)
    non_ignorable_missing = [key for key in result.missing_keys if key != "class_weights"]
    if non_ignorable_missing:
        raise ValueError(
            "Checkpoint is incompatible with the current Stage-1 architecture; "
            f"first missing keys: {non_ignorable_missing[:10]}"
        )
    model.eval()
    return model, checkpoint, args


def extract_student_checkpoint(
    semi_supervised_checkpoint: str | Path,
    champion_checkpoint: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a standard Stage-1 checkpoint consumable by existing ONNX tools."""

    semi_path = Path(semi_supervised_checkpoint).expanduser().resolve()
    champion_path = Path(champion_checkpoint).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not semi_path.is_file():
        raise FileNotFoundError(f"Semi-supervised checkpoint not found: {semi_path}")
    if not champion_path.is_file():
        raise FileNotFoundError(f"Champion checkpoint not found: {champion_path}")

    try:
        semi = torch.load(semi_path, map_location="cpu", weights_only=False)
        champion = torch.load(champion_path, map_location="cpu", weights_only=False)
    except TypeError:
        semi = torch.load(semi_path, map_location="cpu")
        champion = torch.load(champion_path, map_location="cpu")

    student_state = {
        key.removeprefix("student."): value
        for key, value in semi.get("state_dict", {}).items()
        if key.startswith("student.")
    }
    deployable = {
        key: value
        for key, value in student_state.items()
        if key.startswith(("brep_encoder.", "attention.", "classifier."))
        or key == "class_weights"
    }
    if not deployable:
        raise ValueError(f"No student Stage-1 weights found in: {semi_path}")

    exported = {
        "epoch": int(semi.get("epoch", -1)),
        "global_step": int(semi.get("global_step", 0)),
        "pytorch-lightning_version": champion.get("pytorch-lightning_version"),
        "state_dict": deployable,
        "hyper_parameters": copy.deepcopy(champion.get("hyper_parameters", {})),
        "unsupervised_training_metadata": {
            "source_joint_checkpoint": str(semi_path),
            "champion_checkpoint": str(champion_path),
            "export_kind": "student_stage1",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(exported, temporary)
    temporary.replace(destination)
    return destination

