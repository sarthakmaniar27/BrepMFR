#!/usr/bin/env python3
"""Extend the canonical Klavuz ONNX comparison with another model.

The base comparison CSV already contains the corrected ground truth:
the original Model A predictions, with the known Model A mistakes changed
back to Stock. This script keeps that reference fixed, joins a new ONNX
prediction CSV by face_index, and writes an expanded all-face CSV, JSON
summary, and concise Markdown report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


CLASSES = ("Stock", "Thread", "Text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-comparison", type=Path, required=True)
    parser.add_argument("--new-predictions", type=Path, required=True)
    parser.add_argument("--model-id", default="D")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def model_summary(rows: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    class_key = f"model_{model_id}_class"
    confidence_key = f"model_{model_id}_confidence"
    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    class_to_id = {name: idx for idx, name in enumerate(CLASSES)}
    correct_confidences: list[float] = []
    incorrect_confidences: list[float] = []
    errors: list[int] = []

    for row in rows:
        truth = row["corrected_ground_truth"]
        pred = row[class_key]
        if truth not in class_to_id or pred not in class_to_id:
            raise ValueError(
                f"Unknown class at face {row['face_index']}: truth={truth!r}, pred={pred!r}"
            )
        confusion[class_to_id[truth]][class_to_id[pred]] += 1
        confidence = float(row[confidence_key])
        if truth == pred:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)
            errors.append(int(row["face_index"]))

    per_class: dict[str, Any] = {}
    ious: list[float] = []
    for class_id, class_name in enumerate(CLASSES):
        tp = confusion[class_id][class_id]
        support = sum(confusion[class_id])
        predicted = sum(confusion[r][class_id] for r in range(len(CLASSES)))
        fp = predicted - tp
        fn = support - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        iou = safe_div(tp, tp + fp + fn)
        if iou is not None:
            ious.append(iou)
        per_class[class_name] = {
            "support": support,
            "pred": predicted,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "iou": iou,
        }

    known_errors_remaining = [
        int(row["face_index"])
        for row in rows
        if str(row["known_original_error"]) in {"1", "True", "true"}
        and row[class_key] != row["corrected_ground_truth"]
    ]
    stock_as_thread = [
        int(row["face_index"])
        for row in rows
        if row["corrected_ground_truth"] == "Stock" and row[class_key] == "Thread"
    ]
    stock_as_text = [
        int(row["face_index"])
        for row in rows
        if row["corrected_ground_truth"] == "Stock" and row[class_key] == "Text"
    ]
    thread_misses = [
        int(row["face_index"])
        for row in rows
        if row["corrected_ground_truth"] == "Thread" and row[class_key] != "Thread"
    ]
    text_misses = [
        int(row["face_index"])
        for row in rows
        if row["corrected_ground_truth"] == "Text" and row[class_key] != "Text"
    ]
    return {
        "accuracy": (len(rows) - len(errors)) / len(rows),
        "error_count": len(errors),
        "miou": sum(ious) / len(ious),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "prediction_counts": dict(
            Counter(str(row[class_key]) for row in rows)
        ),
        "known_errors_remaining": known_errors_remaining,
        "stock_as_thread": stock_as_thread,
        "stock_as_text": stock_as_text,
        "thread_misses": thread_misses,
        "text_misses": text_misses,
        "mean_confidence_correct": (
            sum(correct_confidences) / len(correct_confidences)
            if correct_confidences
            else None
        ),
        "mean_confidence_incorrect": (
            sum(incorrect_confidences) / len(incorrect_confidences)
            if incorrect_confidences
            else None
        ),
        "error_faces": errors,
    }


def markdown_report(
    summaries: dict[str, dict[str, Any]],
    new_model_id: str,
    total_faces: int,
) -> str:
    lines = [
        "# Klavuz ONNX all-face comparison",
        "",
        f"Corrected reference contains {total_faces} faces. "
        "Rows are ground truth and columns are predictions in the order "
        "`Stock, Thread, Text`.",
        "",
        "| Model | Accuracy | Errors | mIoU | Stock→Thread | Stock→Text | "
        "Thread misses | Text misses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, summary in summaries.items():
        lines.append(
            f"| {model_id} | {summary['accuracy']:.4f} | "
            f"{summary['error_count']} | {summary['miou']:.4f} | "
            f"{len(summary['stock_as_thread'])} | "
            f"{len(summary['stock_as_text'])} | "
            f"{len(summary['thread_misses'])} | "
            f"{len(summary['text_misses'])} |"
        )

    new = summaries[new_model_id]
    lines.extend(
        [
            "",
            f"## Model {new_model_id}",
            "",
            f"- Prediction counts: `{new['prediction_counts']}`",
            f"- Known original Model A errors still wrong: "
            f"`{new['known_errors_remaining']}`",
            f"- Stock→Thread faces: `{new['stock_as_thread']}`",
            f"- Stock→Text faces: `{new['stock_as_text']}`",
            f"- Thread misses: `{new['thread_misses']}`",
            f"- Text misses: `{new['text_misses']}`",
            f"- Mean confidence, correct faces: "
            f"{fmt_metric(new['mean_confidence_correct'])}",
            f"- Mean confidence, incorrect faces: "
            f"{fmt_metric(new['mean_confidence_incorrect'])}",
            "",
            "Confusion matrix:",
            "",
            "| true\\pred | Stock | Thread | Text |",
            "|---|---:|---:|---:|",
        ]
    )
    for class_name, values in zip(CLASSES, new["confusion_matrix"]):
        lines.append(
            f"| {class_name} | {values[0]} | {values[1]} | {values[2]} |"
        )

    lines.extend(["", "Per-class metrics:", "", "| Class | Support | Precision | Recall | IoU |", "|---|---:|---:|---:|---:|"])
    for class_name in CLASSES:
        metrics = new["per_class"][class_name]
        lines.append(
            f"| {class_name} | {metrics['support']} | "
            f"{fmt_metric(metrics['precision'])} | "
            f"{fmt_metric(metrics['recall'])} | "
            f"{fmt_metric(metrics['iou'])} |"
        )
    lines.extend(
        [
            "",
            "> This is a diagnostic on one real part, not a model-selection "
            "benchmark. Use it as a regression test together with a broader "
            "real-part suite and the strict stock-only holdout.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    model_id = args.model_id.strip().upper()
    if not model_id or not model_id.isalnum():
        raise ValueError("--model-id must be a simple alphanumeric identifier")

    rows: list[dict[str, Any]] = read_csv(args.base_comparison)
    predictions = read_csv(args.new_predictions)
    by_face = {int(row["face_index"]): row for row in predictions}
    base_faces = {int(row["face_index"]) for row in rows}
    if set(by_face) != base_faces:
        missing = sorted(base_faces - set(by_face))
        extra = sorted(set(by_face) - base_faces)
        raise ValueError(
            f"Face-index mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    for row in rows:
        pred = by_face[int(row["face_index"])]
        row[f"model_{model_id}_class"] = pred["predicted_label"].title()
        row[f"model_{model_id}_confidence"] = f"{float(pred['confidence']):.8f}"
        row[f"model_{model_id}_prob_stock"] = f"{float(pred['prob_Stock']):.8f}"
        row[f"model_{model_id}_prob_thread"] = f"{float(pred['prob_Thread']):.8f}"
        row[f"model_{model_id}_prob_text"] = f"{float(pred['prob_Text']):.8f}"
        row[f"model_{model_id}_correct"] = int(
            row[f"model_{model_id}_class"] == row["corrected_ground_truth"]
        )

    model_ids = []
    for key in rows[0]:
        if key.startswith("model_") and key.endswith("_class"):
            model_ids.append(key[len("model_") : -len("_class")])
    summaries = {mid: model_summary(rows, mid) for mid in model_ids}
    for mid in model_ids:
        if mid == model_id:
            continue
        summaries[model_id].setdefault("disagreement_counts", {})[mid] = sum(
            row[f"model_{model_id}_class"] != row[f"model_{mid}_class"]
            for row in rows
        )

    for path in (args.output_csv, args.output_json, args.output_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(
        markdown_report(summaries, model_id, len(rows)), encoding="utf-8"
    )

    new = summaries[model_id]
    print(
        f"Model {model_id}: accuracy={new['accuracy']:.6f}, "
        f"errors={new['error_count']}, mIoU={new['miou']:.6f}"
    )
    print(f"Confusion: {new['confusion_matrix']}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
