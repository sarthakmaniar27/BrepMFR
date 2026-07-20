#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage ONNX inference over raw B-rep JSONs in C:\\jsons.

Stage 1 — For each ``*.json`` in the JSON folder (not under ``inference/``):
  1. Convert to a lite PyG graph (same path as training ingest).
  2. Run ``brepmfr_lite.onnx`` from ``BrepMFR_lite_onnx_pyg_demo_v2``.
  3. Write per-face CSV under ``C:\\jsons\\inference\\``.

Stage 2 — Scan those CSVs and list JSONs where **no face** has Thread or Text
probability above a confidence threshold (default 0.80).

Example:
    conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_json_batch_inference.py

    python standalone_scripts/run_onnx_json_batch_inference.py ^
      --json-dir C:\\jsons --max-files 5 --provider cpu
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
_DEMO_DIR = _THIS.parent / "BrepMFR_lite_onnx_pyg_demo_v2"

# Repo root on path for scripts.inference.json_to_brepmfr_pyg
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
boot = _REPO / "bootstrap_path.py"
if boot.is_file():
    spec = importlib.util.spec_from_file_location("_brep_bootstrap", boot)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.setup(str(_THIS))

from scripts.inference.json_to_brepmfr_pyg import build_pyg_from_json_path  # noqa: E402

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

# label_map.json: 0=Stock, 1=Thread, 2=Text
THREAD_CLASS_ID = 1
TEXT_CLASS_ID = 2


class SkipGraph(Exception):
    """Valid JSON but no face predictions possible."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch ONNX inference on C:\\jsons + filter low Thread/Text confidence."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path(r"C:\jsons"),
        help="Folder containing raw B-rep *.json files.",
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=None,
        help="CSV output folder (default: <json-dir>/inference).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_DEMO_DIR,
        help="Folder with brepmfr_lite.onnx, label_map.json, model_config.json.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.80,
        help="Stage-2 threshold for Thread/Text probability (default: 0.80).",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX Runtime execution provider.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip Stage-1 for JSONs that already have a predictions CSV.",
    )
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Skip inference; only scan existing CSVs in the inference folder.",
    )
    return parser.parse_args()


def load_label_map(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(k): str(v) for k, v in raw.items()}


def select_providers(choice: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider unavailable; have: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def list_json_files(json_dir: Path, max_files: int | None) -> list[Path]:
    """List top-level *.json only (exclude inference/ and other subfolders)."""
    files = sorted(
        json_dir / name
        for name in os.listdir(json_dir)
        if name.lower().endswith(".json") and (json_dir / name).is_file()
    )
    if max_files is not None:
        if max_files < 1:
            raise ValueError("--max-files must be at least 1")
        files = files[:max_files]
    return files


def make_lite_batch(graph: Any) -> dict[str, torch.Tensor]:
    node_data = graph.node_data
    if node_data.dtype == torch.float16:
        node_data = node_data.float()
    n_faces = int(node_data.size(0))
    if n_faces == 0:
        raise SkipGraph("zero faces")
    if node_data.ndim != 4 or tuple(node_data.shape[1:]) != (5, 5, 7):
        raise ValueError(f"node_data must be [N,5,5,7], got {tuple(node_data.shape)}")

    face_area = graph.face_area
    if face_area.dtype == torch.float16:
        face_area = face_area.float()

    flat = {
        "face_area": face_area.reshape(-1),
        "face_type": graph.face_type.reshape(-1),
        "face_loop": graph.face_loop.reshape(-1),
        "in_degree": graph.node_degree.reshape(-1),
    }
    for name, tensor in flat.items():
        if tensor.numel() != n_faces:
            raise ValueError(f"{name} length {tensor.numel()} != n_faces {n_faces}")

    attn_bias = getattr(graph, "attn_bias", None)
    if attn_bias is None:
        attn_bias = torch.zeros(n_faces + 1, n_faces + 1, dtype=torch.float32)
    if tuple(attn_bias.shape) != (n_faces + 1, n_faces + 1):
        raise ValueError(
            f"attn_bias must be [{n_faces + 1},{n_faces + 1}], got {tuple(attn_bias.shape)}"
        )

    return {
        "node_data": node_data,
        **flat,
        "attn_bias": attn_bias.unsqueeze(0),
        "padding_mask": torch.zeros(1, n_faces, dtype=torch.bool),
    }


def batch_to_ort_feed(batch: dict[str, Any], input_names: set[str]) -> dict[str, np.ndarray]:
    missing = sorted(input_names - set(batch))
    if missing:
        raise ValueError(f"Missing ONNX inputs: {missing}")
    feed: dict[str, np.ndarray] = {}
    for name in input_names:
        tensor = batch[name]
        if name in FLOAT32_INPUTS:
            tensor = tensor.float()
        elif name in INT64_INPUTS:
            tensor = tensor.long()
        elif name in BOOL_INPUTS:
            tensor = tensor.bool()
        feed[name] = tensor.detach().cpu().numpy()
    return feed


def write_predictions_csv(
    output_path: Path,
    probabilities: np.ndarray,
    label_map: dict[int, str],
    json_stem: str,
) -> tuple[list[int], list[float]]:
    predicted_ids = probabilities.argmax(axis=1).astype(int).tolist()
    confidences = probabilities.max(axis=1).astype(float).tolist()
    class_ids = list(range(probabilities.shape[1]))
    prob_cols = [f"prob_{label_map.get(cid, f'class_{cid}')}" for cid in class_ids]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "json_stem",
                "face_index",
                "predicted_class_id",
                "predicted_label",
                "confidence",
            ]
            + prob_cols,
        )
        writer.writeheader()
        for face_index, (cid, conf, probs) in enumerate(
            zip(predicted_ids, confidences, probabilities)
        ):
            row = {
                "json_stem": json_stem,
                "face_index": face_index,
                "predicted_class_id": cid,
                "predicted_label": label_map.get(cid, f"class_{cid}"),
                "confidence": f"{conf:.8f}",
            }
            for col, p in zip(prob_cols, probs):
                row[col] = f"{float(p):.8f}"
            writer.writerow(row)
    return predicted_ids, confidences


def csv_path_for_json(inference_dir: Path, json_path: Path) -> Path:
    return inference_dir / f"{json_path.stem}_predictions.csv"


def stage1_infer(
    json_files: list[Path],
    inference_dir: Path,
    onnx_path: Path,
    label_map: dict[int, str],
    provider: str,
    skip_existing: bool,
) -> list[dict[str, str]]:
    import onnxruntime as ort

    inference_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(onnx_path), providers=select_providers(provider))
    input_names = {item.name for item in session.get_inputs()}
    missing = sorted(LITE_REQUIRED_INPUTS - input_names)
    extra = sorted(input_names - LITE_REQUIRED_INPUTS)
    if missing or extra:
        raise RuntimeError(
            f"Unexpected ONNX inputs. missing={missing} unsupported={extra}"
        )
    output_name = session.get_outputs()[0].name

    print(f"[INFO] ONNX: {onnx_path}")
    print(f"[INFO] Providers: {session.get_providers()}")
    print(f"[INFO] Stage 1: {len(json_files)} JSON(s) -> {inference_dir}")

    summary: list[dict[str, str]] = []
    for i, json_path in enumerate(json_files, start=1):
        out_csv = csv_path_for_json(inference_dir, json_path)
        if skip_existing and out_csv.is_file():
            print(f"[SKIP-EXISTING] ({i}/{len(json_files)}) {json_path.name}")
            summary.append(
                {
                    "json": str(json_path),
                    "prediction_csv": str(out_csv),
                    "faces": "",
                    "mean_confidence": "",
                    "class_counts": "",
                    "status": "SKIP_EXISTING",
                    "error": "",
                }
            )
            continue

        try:
            graph = build_pyg_from_json_path(json_path, inference_profile="lite")
            batch = make_lite_batch(graph)
            feed = batch_to_ort_feed(batch, input_names)
            probabilities = session.run([output_name], feed)[0]
            if probabilities.ndim != 2:
                raise RuntimeError(f"Unexpected output shape {probabilities.shape}")
            if probabilities.shape[1] != len(label_map):
                raise RuntimeError(
                    f"Output classes={probabilities.shape[1]} label_map={len(label_map)}"
                )

            predicted_ids, confidences = write_predictions_csv(
                out_csv, probabilities, label_map, json_path.stem
            )
            counts = Counter(predicted_ids)
            count_text = ", ".join(
                f"{label_map.get(cid, f'class_{cid}')}={counts[cid]}"
                for cid in sorted(counts)
            )
            print(
                f"[PASS] ({i}/{len(json_files)}) {json_path.name}: "
                f"faces={len(predicted_ids)} mean_conf={np.mean(confidences):.4f} {count_text}"
            )
            summary.append(
                {
                    "json": str(json_path),
                    "prediction_csv": str(out_csv),
                    "faces": str(len(predicted_ids)),
                    "mean_confidence": f"{float(np.mean(confidences)):.8f}",
                    "class_counts": count_text,
                    "status": "PASS",
                    "error": "",
                }
            )
        except SkipGraph as exc:
            print(f"[SKIP] ({i}/{len(json_files)}) {json_path.name}: {exc}")
            summary.append(
                {
                    "json": str(json_path),
                    "prediction_csv": "",
                    "faces": "0",
                    "mean_confidence": "",
                    "class_counts": "",
                    "status": "SKIP",
                    "error": str(exc),
                }
            )
        except Exception as exc:
            print(f"[FAIL] ({i}/{len(json_files)}) {json_path.name}: {exc}", file=sys.stderr)
            summary.append(
                {
                    "json": str(json_path),
                    "prediction_csv": "",
                    "faces": "",
                    "mean_confidence": "",
                    "class_counts": "",
                    "status": "FAIL",
                    "error": str(exc),
                }
            )

    summary_path = inference_dir / "onnx_json_inference_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "json",
                "prediction_csv",
                "faces",
                "mean_confidence",
                "class_counts",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"[INFO] Stage-1 summary: {summary_path}")
    return summary


def _prob_columns(fieldnames: list[str], label_map: dict[int, str]) -> tuple[str | None, str | None]:
    thread_name = label_map.get(THREAD_CLASS_ID, "Thread")
    text_name = label_map.get(TEXT_CLASS_ID, "Text")
    thread_col = f"prob_{thread_name}" if f"prob_{thread_name}" in fieldnames else None
    text_col = f"prob_{text_name}" if f"prob_{text_name}" in fieldnames else None
    return thread_col, text_col


def stage2_filter(
    json_dir: Path,
    inference_dir: Path,
    label_map: dict[int, str],
    confidence: float,
) -> list[dict[str, str]]:
    """Find JSONs with no Thread/Text face probability above ``confidence``."""
    csv_files = sorted(inference_dir.glob("*_predictions.csv"))
    print(
        f"[INFO] Stage 2: scanning {len(csv_files)} CSV(s); "
        f"flag if max(prob_Thread, prob_Text) never exceeds {confidence:.2f}"
    )

    flagged: list[dict[str, str]] = []
    checked = 0

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            thread_col, text_col = _prob_columns(list(reader.fieldnames), label_map)
            if thread_col is None or text_col is None:
                print(f"[WARN] Missing Thread/Text prob columns in {csv_path.name}")
                continue

            max_thread = 0.0
            max_text = 0.0
            n_faces = 0
            for row in reader:
                n_faces += 1
                max_thread = max(max_thread, float(row[thread_col]))
                max_text = max(max_text, float(row[text_col]))

        checked += 1
        has_confident = (max_thread > confidence) or (max_text > confidence)
        if has_confident:
            continue

        # Map CSV stem back to JSON path
        stem = csv_path.name[: -len("_predictions.csv")]
        json_path = json_dir / f"{stem}.json"
        flagged.append(
            {
                "json": str(json_path) if json_path.is_file() else stem,
                "prediction_csv": str(csv_path),
                "faces": str(n_faces),
                "max_prob_Thread": f"{max_thread:.8f}",
                "max_prob_Text": f"{max_text:.8f}",
                "confidence_threshold": f"{confidence:.2f}",
            }
        )

    out_path = inference_dir / "no_confident_thread_or_text.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "json",
            "prediction_csv",
            "faces",
            "max_prob_Thread",
            "max_prob_Text",
            "confidence_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flagged)

    list_path = inference_dir / "no_confident_thread_or_text.txt"
    with list_path.open("w", encoding="utf-8") as handle:
        for row in flagged:
            handle.write(row["json"] + "\n")

    print(f"[INFO] Stage 2 checked CSVs: {checked}")
    print(f"[INFO] Flagged (no Thread/Text > {confidence:.0%}): {len(flagged)}")
    print(f"[INFO] Flagged CSV:  {out_path}")
    print(f"[INFO] Flagged list: {list_path}")
    return flagged


def main() -> int:
    args = parse_args()
    json_dir = args.json_dir.expanduser().resolve()
    if not json_dir.is_dir():
        print(f"ERROR: JSON folder not found: {json_dir}", file=sys.stderr)
        return 1

    inference_dir = (
        args.inference_dir.expanduser().resolve()
        if args.inference_dir is not None
        else json_dir / "inference"
    )
    model_dir = args.model_dir.expanduser().resolve()
    onnx_path = model_dir / "brepmfr_lite.onnx"
    label_map_path = model_dir / "label_map.json"

    if not onnx_path.is_file():
        print(f"ERROR: ONNX not found: {onnx_path}", file=sys.stderr)
        return 1
    if not label_map_path.is_file():
        print(f"ERROR: label_map not found: {label_map_path}", file=sys.stderr)
        return 1

    label_map = load_label_map(label_map_path)
    if THREAD_CLASS_ID not in label_map or TEXT_CLASS_ID not in label_map:
        print(
            f"ERROR: label_map must include ids {THREAD_CLASS_ID} (Thread) "
            f"and {TEXT_CLASS_ID} (Text); got {label_map}",
            file=sys.stderr,
        )
        return 1

    if not args.stage2_only:
        json_files = list_json_files(json_dir, args.max_files)
        if not json_files:
            print(f"ERROR: No *.json files in {json_dir}", file=sys.stderr)
            return 1
        stage1_infer(
            json_files,
            inference_dir,
            onnx_path,
            label_map,
            args.provider,
            args.skip_existing,
        )
    elif not inference_dir.is_dir():
        print(f"ERROR: inference dir missing for --stage2-only: {inference_dir}", file=sys.stderr)
        return 1

    stage2_filter(json_dir, inference_dir, label_map, args.confidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
