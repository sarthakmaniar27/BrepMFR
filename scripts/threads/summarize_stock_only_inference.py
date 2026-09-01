#!/usr/bin/env python3
"""Summarize false Thread/Text predictions on Stock-only inference CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def _prediction_id(row: dict[str, str]) -> int:
    for key in ("predicted_class", "predicted_class_id"):
        value = (row.get(key) or "").strip()
        if value:
            return int(value)
    raise ValueError("prediction CSV has no predicted_class/predicted_class_id value")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown (default: <inference-dir>/stock_false_positive_summary.md).",
    )
    args = parser.parse_args()

    root = args.inference_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Inference directory not found: {root}")

    # Prefer *_predictions.csv (ONNX / newer writers). Fall back to plain
    # {stem}.csv from run_thread_pyg_inference.py, skipping aggregate metrics.
    skip_names = {
        "confusion_matrix.csv",
        "per_class.csv",
        "onnx_inference_summary.csv",
        "onnx_json_inference_summary.csv",
        "stock_label_manifest.csv",
    }
    paths = sorted(root.glob("*_predictions.csv"))
    if not paths:
        paths = sorted(
            p
            for p in root.glob("*.csv")
            if p.name.lower() not in skip_names and p.is_file()
        )
    if not paths:
        raise SystemExit(
            f"No per-graph prediction CSVs found under: {root}\n"
            "Expected *_predictions.csv or {stem}.csv from run_thread_pyg_inference.py."
        )

    total_faces = thread_faces = text_faces = 0
    parts_with_thread = parts_with_text = parts_with_any_feature = 0
    failures: list[str] = []
    for path in paths:
        part_thread = part_text = 0
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    predicted = _prediction_id(row)
                    total_faces += 1
                    if predicted == 1:
                        thread_faces += 1
                        part_thread += 1
                    elif predicted == 2:
                        text_faces += 1
                        part_text += 1
            parts_with_thread += int(part_thread > 0)
            parts_with_text += int(part_text > 0)
            parts_with_any_feature += int(part_thread > 0 or part_text > 0)
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")

    if failures:
        print("Failed prediction CSVs:")
        for failure in failures[:20]:
            print(f"  {failure}")
        return 1

    part_count = len(paths)
    stock_faces = total_faces - thread_faces - text_faces

    def rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    lines = [
        "# Stock-only false-positive summary",
        "",
        f"- Parts: {part_count:,}",
        f"- Faces: {total_faces:,}",
        f"- Predicted Stock faces: {stock_faces:,}",
        f"- Predicted Thread faces: {thread_faces:,}",
        f"- Predicted Text faces: {text_faces:,}",
        f"- Stock→Thread face rate: {100.0 * rate(thread_faces, total_faces):.6f}%",
        f"- Stock→Text face rate: {100.0 * rate(text_faces, total_faces):.6f}%",
        f"- Parts with any false Thread: {parts_with_thread:,}/{part_count:,} "
        f"({100.0 * rate(parts_with_thread, part_count):.4f}%)",
        f"- Parts with any false Text: {parts_with_text:,}/{part_count:,} "
        f"({100.0 * rate(parts_with_text, part_count):.4f}%)",
        f"- Parts with any false feature: {parts_with_any_feature:,}/{part_count:,} "
        f"({100.0 * rate(parts_with_any_feature, part_count):.4f}%)",
        "",
    ]
    out = (
        args.out.resolve()
        if args.out is not None
        else root / "stock_false_positive_summary.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, out)
    print("\n".join(lines))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
