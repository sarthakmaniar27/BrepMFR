from __future__ import annotations

import copy
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .checkpointing import load_stage1_model
from .config import ExperimentConfig
from .constants import IGNORE_LABEL
from .graph_ops import (
    continuous_geometry_targets,
    cosine_anchor_loss,
    forward_stage1,
    masked_geometry_batch,
    sample_face_mask,
    soft_distillation_loss,
)
from .losses import MaskedGeometryHead


def _confusion(labels: torch.Tensor, predictions: torch.Tensor, classes: int) -> torch.Tensor:
    encoded = labels.long() * classes + predictions.long()
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def _metrics_from_confusion(confusion: torch.Tensor) -> dict[str, torch.Tensor]:
    matrix = confusion.float()
    tp = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    union = support + predicted - tp
    recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    iou = torch.where(union > 0, tp / union, torch.zeros_like(tp))
    valid = support > 0
    macro_iou = iou[valid].mean() if valid.any() else matrix.sum() * 0.0
    accuracy = tp.sum() / matrix.sum().clamp_min(1.0)
    return {
        "accuracy": accuracy,
        "macro_iou": macro_iou,
        "recall": recall,
        "precision": precision,
        "iou": iou,
    }


class SemiSupervisedBrepSeg(pl.LightningModule):
    """Model-A-preserving masked geometry adaptation.

    ``student`` is the only model exported after training. ``teacher`` is a
    frozen copy of the champion and supplies soft behavioral anchors; it is
    never updated or used as a hard pseudo-label generator.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters({"config": config.to_dict()})

        student, _, source_args = load_stage1_model(
            config.champion_checkpoint,
            max_nodes_for_a3=config.max_nodes_for_a3,
        )
        self.student = student
        self.teacher = copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.student.batchnorm_finetune_mode = (
            "freeze_stats" if config.freeze_batchnorm_stats else "update"
        )
        self.student._configure_batchnorm_finetune()
        self.teacher.batchnorm_finetune_mode = "freeze_all"
        self.teacher._configure_batchnorm_finetune()

        embedding_dim = int(getattr(source_args, "dim_node", 256))
        self.reconstruction_head = MaskedGeometryHead(embedding_dim)

        checkpoint_weights = self.student.class_weights.detach().float().clone()
        if not config.use_checkpoint_class_weights:
            checkpoint_weights.fill_(1.0)
        self.register_buffer("supervised_class_weights", checkpoint_weights)

        classes = config.num_classes
        self.register_buffer(
            "_student_val_confusion",
            torch.zeros(classes, classes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_teacher_val_confusion",
            torch.zeros(classes, classes, dtype=torch.long),
            persistent=False,
        )
        for name in ("_val_regressions", "_val_improvements", "_val_disagreements", "_val_faces"):
            self.register_buffer(name, torch.zeros((), dtype=torch.long), persistent=False)

    def train(self, mode: bool = True):
        result = super().train(mode)
        self.teacher.eval()
        if mode:
            self.student._enforce_batchnorm_finetune()
        return result

    def _unsupervised_scale(self) -> float:
        # Three epochs avoids a sudden target-domain gradient at the start.
        return min(1.0, float(self.current_epoch + 1) / 3.0)

    def _validate_batches(self, labeled: dict[str, torch.Tensor], unlabeled: dict[str, torch.Tensor]) -> None:
        labels = labeled["label_feature"]
        if labels.numel() == 0 or labels.min() < 0 or labels.max() >= self.config.num_classes:
            raise RuntimeError("Labeled CE batch must contain only class IDs 0/1/2")
        unlabels = unlabeled["label_feature"]
        if unlabels.numel() == 0 or not torch.all(unlabels == IGNORE_LABEL):
            values = torch.unique(unlabels.detach()).tolist()
            raise RuntimeError(
                f"Unlabeled batch must contain only sentinel {IGNORE_LABEL}; got {values}"
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # The strict checkpoint loader constructs both nested Stage-1 models in
        # eval mode. Reassert modes at the step boundary; student.train() keeps
        # only its configured BatchNorm layers in eval.
        self.student.train(True)
        self.teacher.eval()
        if not isinstance(batch, dict) or set(batch) != {"labeled", "unlabeled"}:
            raise RuntimeError("Expected CombinedLoader keys: labeled and unlabeled")
        labeled = batch["labeled"]
        unlabeled = batch["unlabeled"]
        self._validate_batches(labeled, unlabeled)

        student_source = forward_stage1(self.student, labeled)
        with torch.no_grad():
            teacher_source = forward_stage1(self.teacher, labeled)
        source_labels = labeled["label_feature"].long()
        supervised = F.cross_entropy(
            student_source.logits,
            source_labels,
            weight=self.supervised_class_weights,
        )
        source_distillation = soft_distillation_loss(
            student_source.logits,
            teacher_source.logits,
            self.config.distillation_temperature,
        )

        continuous_targets = continuous_geometry_targets(unlabeled)
        face_mask = sample_face_mask(unlabeled["padding_mask"], self.config.mask_ratio)
        context_batch = masked_geometry_batch(unlabeled, face_mask)
        student_target = forward_stage1(self.student, context_batch)
        with torch.no_grad():
            teacher_target = forward_stage1(self.teacher, unlabeled)

        keep_mask = ~face_mask
        target_distillation = soft_distillation_loss(
            student_target.logits,
            teacher_target.logits,
            self.config.distillation_temperature,
            mask=keep_mask,
        )
        feature_anchor = cosine_anchor_loss(
            student_target.fused_embeddings,
            teacher_target.fused_embeddings,
            mask=keep_mask,
        )
        reconstruction_predictions = self.reconstruction_head(
            student_target.face_embeddings[face_mask]
        )
        reconstruction = self.reconstruction_head.loss(
            reconstruction_predictions,
            continuous_target=continuous_targets[face_mask],
            face_type=unlabeled["face_type"][face_mask],
            face_loop=unlabeled["face_loop"][face_mask],
            degree=unlabeled["in_degree"][face_mask],
        )

        scale = self._unsupervised_scale()
        total = (
            self.config.supervised_weight * supervised
            + self.config.source_distillation_weight * source_distillation
            + scale
            * (
                self.config.unlabeled_distillation_weight * target_distillation
                + self.config.unlabeled_feature_anchor_weight * feature_anchor
                + self.config.masked_continuous_weight * reconstruction.continuous
                + self.config.masked_categorical_weight * reconstruction.categorical
            )
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite semi-supervised loss")

        face_count = int(source_labels.numel())
        self.log_dict(
            {
                "train/supervised_ce": supervised,
                "train/source_distillation": source_distillation,
                "train/unlabeled_distillation": target_distillation,
                "train/unlabeled_feature_anchor": feature_anchor,
                "train/masked_continuous": reconstruction.continuous,
                "train/masked_categorical": reconstruction.categorical,
                "train/unsupervised_scale": torch.tensor(scale, device=self.device),
                "train/masked_faces": face_mask.sum().float(),
            },
            on_step=True,
            on_epoch=True,
            batch_size=face_count,
        )
        self.log("train_loss", total, on_step=True, on_epoch=True, prog_bar=True, batch_size=face_count)
        return total

    def on_validation_epoch_start(self) -> None:
        self._student_val_confusion.zero_()
        self._teacher_val_confusion.zero_()
        self._val_regressions.zero_()
        self._val_improvements.zero_()
        self._val_disagreements.zero_()
        self._val_faces.zero_()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        labels = batch["label_feature"].long()
        student = forward_stage1(self.student, batch)
        with torch.no_grad():
            teacher = forward_stage1(self.teacher, batch)
        student_predictions = student.logits.argmax(dim=-1)
        teacher_predictions = teacher.logits.argmax(dim=-1)
        student_correct = student_predictions == labels
        teacher_correct = teacher_predictions == labels
        self._student_val_confusion.add_(
            _confusion(labels, student_predictions, self.config.num_classes)
        )
        self._teacher_val_confusion.add_(
            _confusion(labels, teacher_predictions, self.config.num_classes)
        )
        self._val_regressions.add_((teacher_correct & ~student_correct).sum())
        self._val_improvements.add_((~teacher_correct & student_correct).sum())
        self._val_disagreements.add_((student_predictions != teacher_predictions).sum())
        self._val_faces.add_(labels.numel())
        loss = F.cross_entropy(student.logits, labels, weight=self.supervised_class_weights)
        self.log("val/supervised_ce", loss, on_step=False, on_epoch=True, batch_size=int(labels.numel()))

    def on_validation_epoch_end(self) -> None:
        student = _metrics_from_confusion(self._student_val_confusion)
        teacher = _metrics_from_confusion(self._teacher_val_confusion)
        faces = self._val_faces.float().clamp_min(1.0)
        regression_rate = self._val_regressions.float() / faces
        improvement_rate = self._val_improvements.float() / faces
        disagreement_rate = self._val_disagreements.float() / faces
        guarded_score = student["macro_iou"] - 2.0 * regression_rate
        self.log("val/face_accuracy", student["accuracy"], prog_bar=True)
        self.log("val/macro_iou", student["macro_iou"], prog_bar=True)
        self.log("val/champion_macro_iou", teacher["macro_iou"])
        self.log("val/regression_rate", regression_rate, prog_bar=True)
        self.log("val/improvement_rate", improvement_rate)
        self.log("val/disagreement_rate", disagreement_rate)
        self.log("val/guarded_score", guarded_score, prog_bar=True)
        for class_id in range(self.config.num_classes):
            self.log(f"val/class_{class_id}_recall", student["recall"][class_id])
            self.log(f"val/class_{class_id}_precision", student["precision"][class_id])
            self.log(f"val/class_{class_id}_iou", student["iou"][class_id])

    def configure_optimizers(self):
        student_parameters = [parameter for parameter in self.student.parameters() if parameter.requires_grad]
        head_parameters = [parameter for parameter in self.reconstruction_head.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": student_parameters, "lr": self.config.student_learning_rate},
                {"params": head_parameters, "lr": self.config.head_learning_rate},
            ],
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=self.config.weight_decay,
        )
        estimated = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = min(int(self.config.warmup_steps), max(0, estimated - 1))

        def schedule(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            progress = float(step - warmup) / float(max(1, estimated - warmup))
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["unsupervised_training"] = {
            "schema_version": 1,
            "method": "masked_geometry_with_fixed_champion_distillation",
            "config": self.config.to_dict(),
        }
