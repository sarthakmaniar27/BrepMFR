#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the exported lite ONNX model on one PyG graph or a directory of graphs.

This is a testing/deployment-reference runner, not a training script. It accepts
only the ``lite`` PyG layout used by the Thread + Text model: A1, A2, and A3
tensors must be absent. Each graph is inferred separately because the ONNX
wrapper was exported for one graph per call.

Examples (from repository root):
    conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_pyg_inference.py ^
      --input Z:\thread_and_text\lite\pyg\some_part.pt

    conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_pyg_inference.py ^
      --input Z:\thread_and_text\lite\pyg --output-dir C:\onnx_results

Each graph produces a CSV with one row per face and one probability column per
class. A summary CSV is also written to the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_EXPORTED_DIR = _THIS.parent / "exported"
_ASSET_DIR = _EXPORTED_DIR if _EXPORTED_DIR.is_dir() else _THIS.parent


LITE_REQUIRED_INPUTS = {
    "node_data",
    "face_area",
    "face_type",
    "face_loop",
    "in_degree",
    "attn_bias",
    "padding_mask",
}
FLOAT32_INPUTS = {"node_data", "face_area", "attn_bias"}
INT64_INPUTS = {"face_type", "face_loop", "in_degree"}
BOOL_INPUTS = {"padding_mask"}
OPTIONAL_LITE_TENSORS = (
    "spatial_pos",
    "d2_distance",
    "angle_distance",
    "edge_path",
)


class SkipGraph(Exception):
    """A graph that is structurally valid but cannot produce face predictions."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exported lite ONNX model on PyG .pt graph files."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="One PyG .pt file or a directory containing PyG .pt files.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=_ASSET_DIR / "brepmfr_lite.onnx",
        help="Path to the exported ONNX model.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=_ASSET_DIR / "label_map.json",
        help="JSON object mapping predicted class IDs to display names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV predictions (default: <input-parent>/onnx_inference).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a directory, include .pt files in subdirectories.",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ORT execution provider. auto prefers CUDA when installed.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit directory inference to this many files (testing only).",
    )
    return parser.parse_args()


def resolve_graphs(input_path: Path, recursive: bool, max_files: int | None) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".pt":
            raise ValueError(f"--input file must end in .pt: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"--input does not exist: {input_path}")

    iterator = input_path.rglob("*.pt") if recursive else input_path.glob("*.pt")
    if max_files is not None:
        if max_files < 1:
            raise ValueError("--max-files must be at least 1")
        # Avoid scanning a very large network dataset just to run a small smoke test.
        graphs = sorted(path for path in islice(iterator, max_files) if path.is_file())
    else:
        graphs = sorted(path for path in iterator if path.is_file())
    if not graphs:
        raise FileNotFoundError(f"No .pt files found under: {input_path}")
    return graphs


def load_label_map(path: Path) -> dict[int, str]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Label map must be a JSON object: {path}")
    return {int(key): str(value) for key, value in raw.items()}


def select_providers(choice: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Install a CUDA-enabled ONNX Runtime "
                f"build; available providers are: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def ensure_lite_graph(graph: Any, graph_path: Path) -> None:
    missing = [
        name
        for name in ("node_data", "face_area", "face_type", "face_loop", "node_degree")
        if not hasattr(graph, name)
    ]
    if missing:
        raise ValueError(f"{graph_path.name}: not a recognized PyG B-rep graph; missing {missing}")
    if int(graph.node_data.size(0)) == 0:
        raise SkipGraph(
            "zero faces: no per-face prediction is possible "
            "(the training pipeline also drops these invalid graphs)"
        )

    enabled = []
    for name in OPTIONAL_LITE_TENSORS:
        value = getattr(graph, name, None)
        if value is not None:
            enabled.append(name)
    for flag in ("has_a1", "has_a2", "has_a3"):
        if bool(getattr(graph, flag, False)):
            enabled.append(flag)
    if enabled:
        raise ValueError(
            f"{graph_path.name}: this is not a lite graph ({', '.join(enabled)} is present). "
            "The exported ONNX model requires graphs created with --inference_profile lite."
        )


def make_lite_batch(graph: Any) -> dict[str, torch.Tensor]:
    """Reproduce the production collator for one unpadded lite PyG graph.

    The model does not use edge or A1/A2/A3 tensors in its lite profile, so the
    seven tensors below are the complete ONNX input contract.
    """
    node_data = graph.node_data
    n_faces = int(node_data.size(0))
    if node_data.ndim != 4 or tuple(node_data.shape[1:]) != (5, 5, 7):
        raise ValueError(
            "node_data must have shape [N, 5, 5, 7], "
            f"received {tuple(node_data.shape)}"
        )

    flat_features: dict[str, torch.Tensor] = {
        "face_area": graph.face_area,
        "face_type": graph.face_type,
        "face_loop": graph.face_loop,
        "in_degree": graph.node_degree,
    }
    for name, tensor in flat_features.items():
        if not torch.is_tensor(tensor) or tensor.numel() != n_faces:
            shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__
            raise ValueError(
                f"{name} must contain exactly {n_faces} values, received {shape}"
            )
        flat_features[name] = tensor.reshape(-1)

    attn_bias = getattr(graph, "attn_bias", None)
    if attn_bias is None:
        attn_bias = torch.zeros(n_faces + 1, n_faces + 1, dtype=torch.float32)
    if tuple(attn_bias.shape) != (n_faces + 1, n_faces + 1):
        raise ValueError(
            "attn_bias must have shape "
            f"[{n_faces + 1}, {n_faces + 1}], received {tuple(attn_bias.shape)}"
        )

    return {
        "node_data": node_data,
        **flat_features,
        "attn_bias": attn_bias.unsqueeze(0),
        "padding_mask": torch.zeros(1, n_faces, dtype=torch.bool),
    }


def batch_to_ort_feed(batch: dict[str, Any], input_names: set[str]) -> dict[str, np.ndarray]:
    missing = sorted(input_names - set(batch))
    if missing:
        raise ValueError(f"Lite batch did not produce required ONNX inputs: {missing}")

    feed: dict[str, np.ndarray] = {}
    for name in input_names:
        tensor = batch[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"ONNX input {name!r} is not a tensor")
        if name in FLOAT32_INPUTS:
            tensor = tensor.float()
        elif name in INT64_INPUTS:
            tensor = tensor.long()
        elif name in BOOL_INPUTS:
            tensor = tensor.bool()
        feed[name] = tensor.detach().cpu().numpy()
    return feed


def write_predictions(
    output_path: Path,
    probabilities: np.ndarray,
    label_map: dict[int, str],
) -> tuple[list[int], list[float]]:
    predicted_ids = probabilities.argmax(axis=1).astype(int).tolist()
    confidences = probabilities.max(axis=1).astype(float).tolist()
    class_ids = list(range(probabilities.shape[1]))
    probability_columns = [f"prob_{label_map.get(class_id, f'class_{class_id}')}" for class_id in class_ids]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["face_index", "predicted_class_id", "predicted_label", "confidence"]
            + probability_columns,
        )
        writer.writeheader()
        for face_index, (class_id, confidence, probs) in enumerate(
            zip(predicted_ids, confidences, probabilities)
        ):
            row = {
                "face_index": face_index,
                "predicted_class_id": class_id,
                "predicted_label": label_map.get(class_id, f"class_{class_id}"),
                "confidence": f"{confidence:.8f}",
            }
            row.update(
                {
                    column: f"{float(probability):.8f}"
                    for column, probability in zip(probability_columns, probs)
                }
            )
            writer.writerow(row)
    return predicted_ids, confidences


def main() -> int:
    args = parse_args()
    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    graphs = resolve_graphs(args.input, args.recursive, args.max_files)
    label_map = load_label_map(args.label_map)
    input_root = args.input.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (input_root.parent if input_root.is_file() else input_root) / "onnx_inference"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort

    providers = select_providers(args.provider)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = {item.name for item in session.get_inputs()}
    unsupported_inputs = sorted(input_names - LITE_REQUIRED_INPUTS)
    missing_inputs = sorted(LITE_REQUIRED_INPUTS - input_names)
    if unsupported_inputs or missing_inputs:
        raise RuntimeError(
            "Unexpected ONNX input contract. "
            f"Missing lite inputs={missing_inputs}; unsupported inputs={unsupported_inputs}."
        )
    if len(session.get_outputs()) != 1:
        raise RuntimeError("Expected exactly one ONNX output")
    output_name = session.get_outputs()[0].name

    print(f"[INFO] ONNX: {onnx_path}")
    print(f"[INFO] Providers in use: {session.get_providers()}")
    print(f"[INFO] Processing {len(graphs)} graph(s); CSV output: {output_dir}")

    summary_rows: list[dict[str, str]] = []
    failures = 0
    for graph_path in graphs:
        try:
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            ensure_lite_graph(graph, graph_path)
            batch = make_lite_batch(graph)
            ort_feed = batch_to_ort_feed(batch, input_names)
            probabilities = session.run([output_name], ort_feed)[0]
            if probabilities.ndim != 2:
                raise RuntimeError(f"Unexpected output shape: {probabilities.shape}")
            if probabilities.shape[1] != len(label_map):
                raise RuntimeError(
                    f"Output has {probabilities.shape[1]} classes but label map has {len(label_map)}"
                )

            csv_path = output_dir / f"{graph_path.stem}_predictions.csv"
            predicted_ids, confidences = write_predictions(csv_path, probabilities, label_map)
            counts = Counter(predicted_ids)
            count_text = ", ".join(
                f"{label_map.get(class_id, f'class_{class_id}')}={counts[class_id]}"
                for class_id in sorted(counts)
            )
            print(
                f"[PASS] {graph_path.name}: faces={len(predicted_ids)} "
                f"mean_confidence={np.mean(confidences):.4f}  {count_text}"
            )
            summary_rows.append(
                {
                    "graph": str(graph_path),
                    "prediction_csv": str(csv_path),
                    "faces": str(len(predicted_ids)),
                    "mean_confidence": f"{np.mean(confidences):.8f}",
                    "class_counts": count_text,
                    "status": "PASS",
                    "error": "",
                }
            )
        except SkipGraph as exc:
            print(f"[SKIP] {graph_path.name}: {exc}")
            summary_rows.append(
                {
                    "graph": str(graph_path),
                    "prediction_csv": "",
                    "faces": "0",
                    "mean_confidence": "",
                    "class_counts": "",
                    "status": "SKIP",
                    "error": str(exc),
                }
            )
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {graph_path}: {exc}", file=sys.stderr)
            summary_rows.append(
                {
                    "graph": str(graph_path),
                    "prediction_csv": "",
                    "faces": "",
                    "mean_confidence": "",
                    "class_counts": "",
                    "status": "FAIL",
                    "error": str(exc),
                }
            )

    summary_path = output_dir / "onnx_inference_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "graph",
                "prediction_csv",
                "faces",
                "mean_confidence",
                "class_counts",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[INFO] Summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
