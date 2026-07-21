# -*- coding: utf-8 -*-
"""TensorBoard images / histograms / text helpers (Lightning + TB-first)."""

from __future__ import annotations

import io
from typing import Iterable, List, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pytorch_lightning.loggers import TensorBoardLogger


def iter_tensorboard_loggers(trainer) -> Iterable["TensorBoardLogger"]:
    from pytorch_lightning.loggers import TensorBoardLogger

    loggers = getattr(trainer, "loggers", None)
    if loggers:
        for lg in loggers:
            if isinstance(lg, TensorBoardLogger):
                yield lg
        return
    lg = getattr(trainer, "logger", None)
    if isinstance(lg, TensorBoardLogger):
        yield lg


def tb_add_text(trainer, tag: str, text: str, global_step: int = 0) -> None:
    for lg in iter_tensorboard_loggers(trainer):
        lg.experiment.add_text(tag, text, global_step)


def tb_add_image(trainer, tag: str, img_chw: torch.Tensor, global_step: int) -> None:
    """img_chw: float [C,H,W] in [0,1]."""
    for lg in iter_tensorboard_loggers(trainer):
        lg.experiment.add_image(tag, img_chw, global_step)


def tb_add_histogram(
    trainer, tag: str, values: torch.Tensor, global_step: int, bins: str = "tensorflow"
) -> None:
    v = values.detach().cpu().float().reshape(-1)
    if v.numel() == 0:
        return
    for lg in iter_tensorboard_loggers(trainer):
        lg.experiment.add_histogram(tag, v, global_step, bins=bins)


def tb_add_scalar(trainer, tag: str, value: float, global_step: int) -> None:
    for lg in iter_tensorboard_loggers(trainer):
        lg.experiment.add_scalar(tag, value, global_step)


def confusion_matrix_counts_figure_tensor(
    counts: np.ndarray,
    normalize_rows: bool = True,
) -> torch.Tensor:
    """Render a pre-aggregated [true, predicted] confusion matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(counts, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got {cm.shape}")
    if normalize_rows:
        row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
        cm = cm / row_sum

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    im = ax.imshow(cm, vmin=0.0, vmax=1.0 if normalize_rows else None, cmap="magma")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    img = Image.open(buf).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def confusion_matrix_figure_tensor(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    normalize_rows: bool = True,
) -> torch.Tensor:
    """Build normalized confusion visualization as CHW float tensor [0,1]."""
    preds = preds.astype(np.int64).ravel()
    labels = labels.astype(np.int64).ravel()
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    valid = (
        (labels >= 0)
        & (labels < num_classes)
        & (preds >= 0)
        & (preds < num_classes)
    )
    np.add.at(cm, (labels[valid], preds[valid]), 1.0)
    return confusion_matrix_counts_figure_tensor(cm, normalize_rows=normalize_rows)

def per_class_recall(
    preds: np.ndarray, labels: np.ndarray, num_classes: int
) -> List[float]:
    preds = preds.astype(np.int64).ravel()
    labels = labels.astype(np.int64).ravel()
    out: List[float] = []
    for i in range(num_classes):
        pos = np.where(labels == i)[0]
        if pos.size == 0:
            out.append(float("nan"))
        else:
            out.append(float((preds[pos] == labels[pos]).mean()))
    return out


def log_per_class_iou_scalars(
    trainer,
    preds_np: np.ndarray,
    labels_np: np.ndarray,
    num_classes: int,
    epoch: int,
    prefix: str = "val",
) -> None:
    """Match ``BrepSeg.on_test_epoch_end`` IoU branch; log one scalar per class when defined."""
    preds = preds_np.astype(np.int64).ravel()
    labels = labels_np.astype(np.int64).ravel()
    for i in range(num_classes):
        label_pos = np.where(labels == i)[0]
        pred_pos = np.where(preds == i)[0]
        if pred_pos.size > 0 and label_pos.size > 0:
            class_i_preds = preds[label_pos]
            class_i_label = labels[label_pos]
            inter = (class_i_preds == class_i_label).astype(np.float64)
            union = (class_i_preds != class_i_label).astype(np.float64)
            class_i_preds_ = preds[pred_pos]
            class_i_label_ = labels[pred_pos]
            union_ = (class_i_preds_ != class_i_label_).astype(np.float64)
            iou = float(
                np.sum(inter)
                / (np.sum(union) + np.sum(inter) + np.sum(union_) + 1e-9)
            )
            tb_add_scalar(trainer, f"{prefix}/per_class_iou/c{i:02d}", iou, epoch)


def log_segmentation_val_media(
    trainer,
    preds_np: np.ndarray,
    labels_np: np.ndarray,
    num_classes: int,
    epoch: int,
    prefix: str = "val",
) -> None:
    """Confusion figure + per-class recall/IoU scalars."""
    img = confusion_matrix_figure_tensor(preds_np, labels_np, num_classes)
    tb_add_image(trainer, f"{prefix}/confusion_matrix", img, epoch)

    recalls = per_class_recall(preds_np, labels_np, num_classes)
    for i, r in enumerate(recalls):
        if not np.isnan(r):
            tb_add_scalar(trainer, f"{prefix}/per_class_recall/c{i:02d}", r, epoch)

    log_per_class_iou_scalars(trainer, preds_np, labels_np, num_classes, epoch, prefix)


def log_segmentation_val_confusion(
    trainer,
    counts: np.ndarray,
    epoch: int,
    prefix: str = "val",
) -> None:
    """Log validation media/scalars without retaining every face prediction."""
    cm = np.asarray(counts, dtype=np.float64)
    img = confusion_matrix_counts_figure_tensor(cm)
    tb_add_image(trainer, f"{prefix}/confusion_matrix", img, epoch)
    rows = cm.sum(axis=1)
    cols = cm.sum(axis=0)
    diag = np.diag(cm)
    for i in range(cm.shape[0]):
        if rows[i] > 0:
            tb_add_scalar(trainer, f"{prefix}/per_class_recall/c{i:02d}", float(diag[i] / rows[i]), epoch)
        union = rows[i] + cols[i] - diag[i]
        if rows[i] > 0 and cols[i] > 0 and union > 0:
            tb_add_scalar(trainer, f"{prefix}/per_class_iou/c{i:02d}", float(diag[i] / union), epoch)
