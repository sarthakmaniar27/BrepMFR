from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ForwardOutput:
    face_embeddings: torch.Tensor
    fused_embeddings: torch.Tensor
    logits: torch.Tensor


def forward_stage1(model, batch: dict[str, torch.Tensor]) -> ForwardOutput:
    node_states, graph_embeddings = model.brep_encoder(batch, last_state_only=True)
    padded = node_states[0].permute(1, 0, 2)[:, 1:, :]
    node_positions = torch.where(~batch["padding_mask"])
    face_embeddings = padded[node_positions]
    per_face_graph = graph_embeddings[node_positions[0]]
    fused = model.attention([face_embeddings, per_face_graph])
    logits = model.classifier.forward_logits(fused)
    return ForwardOutput(face_embeddings, fused, logits)


def sample_face_mask(padding_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    """Return a flat mask aligned with collator-concatenated face tensors."""

    graph_ids = torch.where(~padding_mask)[0]
    result = torch.zeros(graph_ids.numel(), dtype=torch.bool, device=padding_mask.device)
    for graph_id in range(int(padding_mask.shape[0])):
        indices = torch.where(graph_ids == graph_id)[0]
        if indices.numel() == 0:
            continue
        count = max(1, int(round(indices.numel() * ratio)))
        count = min(count, int(indices.numel()))
        chosen = indices[torch.randperm(indices.numel(), device=indices.device)[:count]]
        result[chosen] = True
    return result


def masked_geometry_batch(
    batch: dict[str, torch.Tensor],
    face_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mask local face evidence while retaining graph topology and edge context."""

    masked = dict(batch)
    for key in ("node_data", "face_area", "face_type", "face_loop", "in_degree", "out_degree"):
        value = batch.get(key)
        if value is None:
            continue
        clone = value.clone()
        clone[face_mask] = 0
        masked[key] = clone
    return masked


def continuous_geometry_targets(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build bounded, rotation-sensitive descriptors for masked-face reconstruction.

    SolidWorks export already normalizes parts. Smooth-L1 is used downstream so
    outlying face areas cannot dominate the self-supervised objective.
    """

    uv = batch["node_data"].float()
    valid = (uv[..., 6:7] > 0).float()
    count = valid.sum(dim=(1, 2)).clamp_min(1.0)
    xyz = uv[..., 0:3]
    normal = uv[..., 3:6]
    xyz_mean = (xyz * valid).sum(dim=(1, 2)) / count
    xyz_second = (xyz.square() * valid).sum(dim=(1, 2)) / count
    xyz_std = (xyz_second - xyz_mean.square()).clamp_min(0.0).sqrt()
    normal_mean = (normal * valid).sum(dim=(1, 2)) / count
    log_area = torch.log1p(batch["face_area"].float().clamp_min(0.0)).unsqueeze(1)
    return torch.cat((xyz_mean, xyz_std, normal_mean, log_area), dim=1)


def soft_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
    if student_logits.numel() == 0:
        return student_logits.sum() * 0.0
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=-1),
        F.softmax(teacher_logits.detach() / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)


def cosine_anchor_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        student_features = student_features[mask]
        teacher_features = teacher_features[mask]
    if student_features.numel() == 0:
        return student_features.sum() * 0.0
    return (1.0 - F.cosine_similarity(student_features, teacher_features.detach(), dim=-1)).mean()

