from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    champion_checkpoint: str
    labeled_dataset_root: str
    unlabeled_dataset_root: str

    num_classes: int = 3
    seed: int = 42
    max_epochs: int = 12
    precision: str = "16-mixed"
    accelerator: str = "gpu"
    devices: int = 1

    labeled_batch_size: int = 32
    unlabeled_batch_size: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    length_bucket_batching: bool = True
    batch_node_sq_budget: int = 2_000_000
    max_nodes_for_a3: int = 768

    student_learning_rate: float = 1.0e-5
    head_learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_clip_val: float = 1.0

    mask_ratio: float = 0.15
    supervised_weight: float = 1.0
    source_distillation_weight: float = 0.50
    unlabeled_distillation_weight: float = 0.20
    unlabeled_feature_anchor_weight: float = 0.05
    masked_continuous_weight: float = 0.50
    masked_categorical_weight: float = 0.50
    distillation_temperature: float = 2.0
    use_checkpoint_class_weights: bool = True
    freeze_batchnorm_stats: bool = True

    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 25
    save_top_k: int = 5
    combined_loader_mode: str = "max_size_cycle"

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"Configuration must be a JSON object: {config_path}")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
        config = cls(**payload)
        config.validate()
        return config

    def with_overrides(self, **overrides: Any) -> "ExperimentConfig":
        payload = asdict(self)
        payload.update({key: value for key, value in overrides.items() if value is not None})
        config = type(self)(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.num_classes != 3:
            raise ValueError("This experiment is deliberately fixed to Stock/Thread/Text (3 classes).")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.labeled_batch_size <= 0 or self.unlabeled_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if self.student_learning_rate <= 0 or self.head_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive")
        if self.combined_loader_mode not in {"min_size", "max_size_cycle"}:
            raise ValueError("combined_loader_mode must be min_size or max_size_cycle")
        for name in (
            "supervised_weight",
            "source_distillation_weight",
            "unlabeled_distillation_weight",
            "unlabeled_feature_anchor_weight",
            "masked_continuous_weight",
            "masked_categorical_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

