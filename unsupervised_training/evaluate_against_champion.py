#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from data.collator import collator  # noqa: E402
from unsupervised_training.checkpointing import load_stage1_model  # noqa: E402
from unsupervised_training.constants import CLASS_NAMES, MULTI_HOP_MAX_DIST, SPATIAL_POS_MAX  # noqa: E402
from unsupervised_training.graph_ops import forward_stage1  # noqa: E402


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _split_stems(dataset_root: Path, split: str | None, split_file: str | None) -> list[str]:
    path = Path(split_file).expanduser().resolve() if split_file else dataset_root / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _confusion_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    matrix = confusion.double()
    tp = matrix.diag()
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    union = support + predicted - tp
    recall = torch.where(support > 0, tp / support, torch.full_like(tp, float("nan")))
    precision = torch.where(predicted > 0, tp / predicted, torch.full_like(tp, float("nan")))
    iou = torch.where(union > 0, tp / union, torch.full_like(tp, float("nan")))
    return {
        "confusion": confusion.tolist(),
        "accuracy": float(tp.sum() / matrix.sum().clamp_min(1)),
        "macro_iou": float(iou[~torch.isnan(iou)].mean()),
        "classes": {
            CLASS_NAMES[index]: {
                "support": int(support[index]),
                "predicted": int(predicted[index]),
                "precision": None if torch.isnan(precision[index]) else float(precision[index]),
                "recall": None if torch.isnan(recall[index]) else float(recall[index]),
                "iou": None if torch.isnan(iou[index]) else float(iou[index]),
            }
            for index in range(3)
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a face/part regression report against the frozen champion"
    )
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True, help="Extracted standard Stage-1 checkpoint")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pt-subdir", default="pyg")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-nodes-for-a3", type=int, default=768)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ui-review-count", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    graph_root = dataset_root / args.pt_subdir
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = _split_stems(dataset_root, args.split, args.split_file)
    graph_index = {path.stem.casefold(): path for path in graph_root.rglob("*.pt")}
    missing = [stem for stem in stems if stem.casefold() not in graph_index]
    if missing:
        raise SystemExit(f"{len(missing)} split graphs missing; first: {missing[:10]}")

    champion, _, _ = load_stage1_model(
        args.champion_checkpoint,
        max_nodes_for_a3=args.max_nodes_for_a3,
    )
    candidate, _, _ = load_stage1_model(
        args.candidate_checkpoint,
        max_nodes_for_a3=args.max_nodes_for_a3,
    )
    champion.to(device).eval()
    candidate.to(device).eval()

    champion_confusion = torch.zeros(3, 3, dtype=torch.long)
    candidate_confusion = torch.zeros(3, 3, dtype=torch.long)
    parts: list[dict[str, Any]] = []
    face_rows: list[dict[str, Any]] = []
    labelled_faces = 0

    with torch.inference_mode():
        for number, stem in enumerate(stems, start=1):
            graph = _torch_load(graph_index[stem.casefold()])
            batch = collator(
                [graph],
                multi_hop_max_dist=MULTI_HOP_MAX_DIST,
                spatial_pos_max=SPATIAL_POS_MAX,
                max_nodes_for_a3=args.max_nodes_for_a3,
            )
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            champion_output = forward_stage1(champion, batch)
            candidate_output = forward_stage1(candidate, batch)
            champion_probability = champion_output.logits.softmax(-1)
            candidate_probability = candidate_output.logits.softmax(-1)
            champion_prediction = champion_probability.argmax(-1)
            candidate_prediction = candidate_probability.argmax(-1)
            labels = batch["label_feature"].long()
            valid = (labels >= 0) & (labels < 3)
            if valid.any():
                labelled_faces += int(valid.sum())
                champion_confusion += torch.bincount(
                    labels[valid] * 3 + champion_prediction[valid], minlength=9
                ).reshape(3, 3).cpu()
                candidate_confusion += torch.bincount(
                    labels[valid] * 3 + candidate_prediction[valid], minlength=9
                ).reshape(3, 3).cpu()

            disagreements = champion_prediction != candidate_prediction
            counts_champion = Counter(champion_prediction.cpu().tolist())
            counts_candidate = Counter(candidate_prediction.cpu().tolist())
            champion_correct = champion_prediction == labels
            candidate_correct = candidate_prediction == labels
            regressions = valid & champion_correct & ~candidate_correct
            improvements = valid & ~champion_correct & candidate_correct
            part_row = {
                "stem": stem,
                "faces": int(labels.numel()),
                "disagreements": int(disagreements.sum()),
                "regressions": int(regressions.sum()),
                "improvements": int(improvements.sum()),
                "champion_stock": counts_champion[0],
                "champion_thread": counts_champion[1],
                "champion_text": counts_champion[2],
                "candidate_stock": counts_candidate[0],
                "candidate_thread": counts_candidate[1],
                "candidate_text": counts_candidate[2],
                "thread_delta": counts_candidate[1] - counts_champion[1],
                "text_delta": counts_candidate[2] - counts_champion[2],
                "mean_champion_confidence": float(champion_probability.max(-1).values.mean()),
                "mean_candidate_confidence": float(candidate_probability.max(-1).values.mean()),
            }
            parts.append(part_row)

            for face_index in torch.where(disagreements)[0].cpu().tolist():
                champion_class = int(champion_prediction[face_index])
                candidate_class = int(candidate_prediction[face_index])
                label = int(labels[face_index])
                face_rows.append(
                    {
                        "stem": stem,
                        "face_index": face_index,
                        "ground_truth": label if 0 <= label < 3 else "",
                        "champion_class": champion_class,
                        "champion_name": CLASS_NAMES[champion_class],
                        "champion_probability": float(champion_probability[face_index, champion_class]),
                        "candidate_class": candidate_class,
                        "candidate_name": CLASS_NAMES[candidate_class],
                        "candidate_probability": float(candidate_probability[face_index, candidate_class]),
                    }
                )
            if number % 250 == 0 or number == len(stems):
                print(f"Evaluated {number:,}/{len(stems):,}", flush=True)

    part_fields = list(parts[0]) if parts else []
    face_fields = list(face_rows[0]) if face_rows else [
        "stem", "face_index", "ground_truth", "champion_class", "champion_name",
        "champion_probability", "candidate_class", "candidate_name", "candidate_probability",
    ]
    _write_csv(output_dir / "parts.csv", parts, part_fields)
    _write_csv(output_dir / "face_disagreements.csv", face_rows, face_fields)

    queue = sorted(
        parts,
        key=lambda row: (
            row["regressions"],
            row["disagreements"],
            abs(row["thread_delta"]) + abs(row["text_delta"]),
        ),
        reverse=True,
    )[: args.ui_review_count]
    _write_csv(output_dir / "solidworks_ui_review_queue.csv", queue, part_fields)

    summary: dict[str, Any] = {
        "champion_checkpoint": str(Path(args.champion_checkpoint).expanduser().resolve()),
        "candidate_checkpoint": str(Path(args.candidate_checkpoint).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "split": str(Path(args.split_file).resolve()) if args.split_file else args.split,
        "parts": len(parts),
        "faces": sum(row["faces"] for row in parts),
        "labelled_faces": labelled_faces,
        "disagreement_faces": sum(row["disagreements"] for row in parts),
        "parts_with_disagreement": sum(row["disagreements"] > 0 for row in parts),
        "regression_faces": sum(row["regressions"] for row in parts),
        "improvement_faces": sum(row["improvements"] for row in parts),
    }
    if labelled_faces:
        summary["champion_metrics"] = _confusion_metrics(champion_confusion)
        summary["candidate_metrics"] = _confusion_metrics(candidate_confusion)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

