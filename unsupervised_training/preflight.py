#!/usr/bin/env python3
"""One-graph, no-training integration check for the experiment stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from data.collator import collator  # noqa: E402
from scripts.inference.json_to_brepmfr_pyg_optimized import build_pyg_from_json_path  # noqa: E402
from unsupervised_training.config import ExperimentConfig  # noqa: E402
from unsupervised_training.constants import IGNORE_LABEL, MULTI_HOP_MAX_DIST, SPATIAL_POS_MAX  # noqa: E402
from unsupervised_training.graph_ops import (  # noqa: E402
    continuous_geometry_targets,
    forward_stage1,
    masked_geometry_batch,
    sample_face_mask,
)
from unsupervised_training.semi_model import SemiSupervisedBrepSeg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--unlabeled-json", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    checkpoint = str(Path(args.champion_checkpoint).expanduser().resolve())
    json_path = Path(args.unlabeled_json).expanduser().resolve()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    graph = build_pyg_from_json_path(json_path, inference_profile="no_a2")
    graph.label_feature = torch.full(
        (int(graph.node_data.shape[0]),), IGNORE_LABEL, dtype=torch.int32
    )
    batch = collator(
        [graph],
        multi_hop_max_dist=MULTI_HOP_MAX_DIST,
        spatial_pos_max=SPATIAL_POS_MAX,
        max_nodes_for_a3=768,
    )
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    config = ExperimentConfig(
        experiment_name="preflight",
        champion_checkpoint=checkpoint,
        labeled_dataset_root=".",
        unlabeled_dataset_root=".",
        accelerator=args.device,
        precision="32",
    )
    model = SemiSupervisedBrepSeg(config).to(device).eval()
    with torch.inference_mode():
        teacher = forward_stage1(model.teacher, batch)
        mask = sample_face_mask(batch["padding_mask"], config.mask_ratio)
        masked = masked_geometry_batch(batch, mask)
        student = forward_stage1(model.student, masked)
        targets = continuous_geometry_targets(batch)
        predictions = model.reconstruction_head(student.face_embeddings[mask])
        losses = model.reconstruction_head.loss(
            predictions,
            continuous_target=targets[mask],
            face_type=batch["face_type"][mask],
            face_loop=batch["face_loop"][mask],
            degree=batch["in_degree"][mask],
        )

    checks = {
        "faces": int(graph.node_data.shape[0]),
        "edges": int(graph.edge_data.shape[0]),
        "masked_faces": int(mask.sum()),
        "teacher_logits_shape": list(teacher.logits.shape),
        "student_logits_shape": list(student.logits.shape),
        "continuous_loss": float(losses.continuous),
        "categorical_loss": float(losses.categorical),
        "finite": bool(
            torch.isfinite(teacher.logits).all()
            and torch.isfinite(student.logits).all()
            and torch.isfinite(losses.continuous)
            and torch.isfinite(losses.categorical)
        ),
        "unlabeled_sentinel_only": bool(torch.all(batch["label_feature"] == IGNORE_LABEL)),
        "profile": getattr(graph, "inference_profile", None),
    }
    print(json.dumps(checks, indent=2))
    if not checks["finite"] or not checks["unlabeled_sentinel_only"] or checks["profile"] != "no_a2":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

