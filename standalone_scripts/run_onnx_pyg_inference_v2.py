#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the exported lite ONNX model (Thread + Text, 3 classes) on a PyG dataset.

v2 of ``run_onnx_pyg_inference.py``. Two main differences from v1:

1. **Three-class label map baked in.** The model now predicts
   ``Stock=0``, ``Thread=1`` and the new ``Text=2`` class. The default
   label map is the built-in 3-class map below; ``--label-map`` can still
   override it (e.g. to use ``exported/label_map.json``).

2. **Dataset-root oriented CLI** matching the production inference
   scripts. Point ``--dataset-path`` at a folder laid out like::

       <dataset-path>/
           pyg/            # *.pt graphs   (also accepts --pyg-subdir pug)
           label/          # optional <stem>.json sidecars with {"labels": [...]}
           test.txt        # optional split list, one stem per line

   Per-graph CSVs and an ``onnx_inference_summary.csv`` are written to
   ``--output-dir``. When ground-truth labels are available (either
   ``graph.label_feature`` or a ``label/<stem>.json`` sidecar) the script
   also emits ``confusion_matrix.csv``, ``per_class.csv`` and
   ``summary.md`` under ``--metrics-dir`` (defaults to ``<output-dir>``).

Examples (PowerShell, from the repo root):

    conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_pyg_inference_v2.py ^
      --dataset-path \\\\Gr-sw66464\\d\\brepmfr_sw_inference\\pyg_lite ^
      --output-dir   \\\\Gr-sw66464\\d\\brepmfr_sw_inference\\csv_inference

    # If the graph sub-folder is literally named "pug":
    conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_pyg_inference_v2.py ^
      --dataset-path \\\\Gr-sw66464\\d\\brepmfr_sw_inference\\pyg_lite ^
      --pyg-subdir pug ^
      --output-dir   \\\\Gr-sw66464\\d\\brepmfr_sw_inference\\csv_inference

A single ``.pt`` file is still accepted via ``--input`` for backward
compatibility with v1 smoke tests.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_EXPORTED_DIR = _THIS.parent / "exported"
_ASSET_DIR = _EXPORTED_DIR if _EXPORTED_DIR.is_dir() else _THIS.parent

# 3-class Thread + Text model. Text is the new class with id = 2.
DEFAULT_LABEL_MAP: dict[int, str] = {0: "Stock", 1: "Thread", 2: "Text"}
DEFAULT_NUM_CLASSES = 3

DEFAULT_OUTPUT_DIR = Path(r"\\Gr-sw66464\d\brepmfr_sw_inference\csv_inference")

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
        description=(
            "Run the exported lite ONNX model (Thread + Text, 3 classes) on a "
            "PyG dataset root or a single .pt graph."
        )
    )
    src = parser.add_argument_group("Input source (pick one)")
    src.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Dataset root containing a pyg/ sub-folder of *.pt graphs, an "
            "optional label/ sub-folder of sidecars, and an optional test.txt."
        ),
    )
    src.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Backward-compatible single .pt file or a flat directory of *.pt "
            "files. Mutually exclusive with --dataset-path."
        ),
    )

    parser.add_argument(
        "--pyg-subdir",
        default="pyg",
        help=(
            "Sub-folder under --dataset-path that holds the *.pt graphs "
            "(default: 'pyg'; use 'pug' if that is the actual folder name)."
        ),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help=(
            "Split list, one stem per line (default: "
            "<dataset-path>/test.txt if it exists). Pass --no-split to ignore."
        ),
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Ignore any test.txt under --dataset-path; infer every *.pt.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a directory, include *.pt in sub-folders.",
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
        default=None,
        help=(
            "JSON object mapping class IDs to display names. Overrides the "
            "built-in 3-class {0:Stock, 1:Thread, 2:Text} map."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-graph and summary CSV outputs.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help=(
            "Directory for confusion_matrix.csv / per_class.csv / summary.md "
            "(default: <output-dir>). Only written when GT labels exist."
        ),
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
        help="Limit inference to this many graphs (testing only).",
    )
    args = parser.parse_args()

    if args.dataset_path is None and args.input is None:
        parser.error("Provide --dataset-path or --input.")
    if args.dataset_path is not None and args.input is not None:
        parser.error("--dataset-path and --input are mutually exclusive.")
    return args


def read_split_stems(split_file: Path) -> list[str]:
    stems: list[str] = []
    for raw in split_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        stems.append(line.removesuffix(".pt"))
    return stems


def resolve_graphs(args: argparse.Namespace) -> tuple[list[Path], Optional[Path], Path]:
    """Return (graph_files, split_file_used, graph_root_for_sidecars).

    graph_root_for_sidecars is the parent of the pyg sub-folder, so that
    ``graph_root / "label" / "<stem>.json"`` resolves correctly.
    """
    if args.input is not None:
        input_path = args.input.expanduser().resolve()
        if input_path.is_file():
            if input_path.suffix.lower() != ".pt":
                raise ValueError(f"--input file must end in .pt: {input_path}")
            return [input_path], None, input_path.parent
        if not input_path.is_dir():
            raise FileNotFoundError(f"--input does not exist: {input_path}")
        iterator = (
            input_path.rglob("*.pt") if args.recursive else input_path.glob("*.pt")
        )
        files = sorted(p for p in iterator if p.is_file())
        if args.max_files is not None:
            files = files[: max(0, args.max_files)]
        if not files:
            raise FileNotFoundError(f"No .pt files found under: {input_path}")
        return files, None, input_path

    ds_root = args.dataset_path.expanduser().resolve()
    if not ds_root.is_dir():
        raise FileNotFoundError(f"--dataset-path is not a directory: {ds_root}")

    pyg_dir = ds_root / args.pyg_subdir
    if not pyg_dir.is_dir():
        # Be forgiving about the common 'pug'/'pyg' typo.
        alt = ds_root / ("pug" if args.pyg_subdir == "pyg" else "pyg")
        if alt.is_dir():
            print(f"[warn] '{args.pyg_subdir}/' not found; using '{alt.name}/' instead.")
            pyg_dir = alt
        else:
            raise FileNotFoundError(
                f"PyG sub-folder not found under {ds_root}: looked for "
                f"'{args.pyg_subdir}' and '{alt.name}'."
            )

    split_file: Optional[Path] = None
    if args.split_file is not None:
        split_file = args.split_file.expanduser().resolve()
        if not split_file.is_file():
            raise FileNotFoundError(f"--split-file not found: {split_file}")
    elif not args.no_split:
        candidate = ds_root / "test.txt"
        if candidate.is_file():
            split_file = candidate
        else:
            print(f"[info] No test.txt under {ds_root}; inferring every *.pt in {pyg_dir}.")

    if split_file is not None:
        stems = read_split_stems(split_file)
        files: list[Path] = []
        missing = 0
        for stem in stems:
            p = pyg_dir / f"{stem}.pt"
            if p.is_file():
                files.append(p)
            else:
                missing += 1
        if missing:
            print(
                f"[warn] {missing}/{len(stems)} split stems have no .pt under {pyg_dir}.",
                file=sys.stderr,
            )
    else:
        files = sorted(pyg_dir.glob("*.pt"))

    if args.max_files is not None:
        files = files[: max(0, args.max_files)]
    if not files:
        raise FileNotFoundError(f"No .pt graphs to infer under {pyg_dir}.")
    return files, split_file, ds_root


def load_label_map(path: Optional[Path]) -> dict[int, str]:
    if path is None:
        return dict(DEFAULT_LABEL_MAP)
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
                "CUDAExecutionProvider is unavailable. Install a CUDA-enabled ONNX "
                f"Runtime build; available providers are: {available}"
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
        raise ValueError(
            f"{graph_path.name}: not a recognized PyG B-rep graph; missing {missing}"
        )
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
    """Reproduce the production collator for one unpadded lite PyG graph."""
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


def batch_to_ort_feed(
    batch: dict[str, Any], input_names: set[str]
) -> dict[str, np.ndarray]:
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


def resolve_gt(
    graph: Any,
    stem: str,
    dataset_root: Path,
    num_classes: int,
) -> Optional[np.ndarray]:
    n_faces = int(graph.node_data.size(0))
    lf = getattr(graph, "label_feature", None)
    if lf is not None:
        arr = lf.detach().cpu().numpy().flatten()
        if (
            arr.size == n_faces
            and np.isfinite(arr).all()
            and np.all(arr >= 0)
            and np.all(arr < num_classes)
        ):
            return arr.astype(np.int64)

    for sub in ("label", "labels"):
        sidecar = dataset_root / sub / f"{stem}.json"
        if not sidecar.is_file():
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError):
            continue
        labels = data.get("labels")
        if labels is None or len(labels) != n_faces:
            continue
        arr = np.asarray(labels, dtype=np.int64).flatten()
        if arr.size == n_faces and np.all(arr >= 0) and np.all(arr < num_classes):
            return arr
    return None


def write_predictions(
    output_path: Path,
    probabilities: np.ndarray,
    label_map: dict[int, str],
    gt: Optional[np.ndarray],
) -> tuple[list[int], list[float]]:
    predicted_ids = probabilities.argmax(axis=1).astype(int).tolist()
    confidences = probabilities.max(axis=1).astype(float).tolist()
    class_ids = list(range(probabilities.shape[1]))
    probability_columns = [
        f"prob_{label_map.get(cid, f'class_{cid}')}" for cid in class_ids
    ]

    fields = (
        ["face_index", "predicted_class_id", "predicted_label", "confidence"]
        + probability_columns
    )
    if gt is not None:
        fields += ["ground_truth_class_id", "ground_truth_label", "correct_top1"]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for face_index, (cid, conf, probs) in enumerate(
            zip(predicted_ids, confidences, probabilities)
        ):
            row = {
                "face_index": face_index,
                "predicted_class_id": cid,
                "predicted_label": label_map.get(cid, f"class_{cid}"),
                "confidence": f"{conf:.8f}",
            }
            row.update(
                {
                    col: f"{float(p):.8f}"
                    for col, p in zip(probability_columns, probs)
                }
            )
            if gt is not None:
                yi = int(gt[face_index])
                row["ground_truth_class_id"] = yi
                row["ground_truth_label"] = label_map.get(yi, f"class_{yi}")
                row["correct_top1"] = int(cid == yi)
            writer.writerow(row)
    return predicted_ids, confidences


def confusion_matrix(
    preds: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, y in zip(preds, labels):
        if 0 <= int(y) < num_classes and 0 <= int(p) < num_classes:
            cm[int(y), int(p)] += 1
    return cm


def per_class_rows(cm: np.ndarray, label_map: dict[int, str]) -> list[dict[str, Any]]:
    n = cm.shape[0]
    rows: list[dict[str, Any]] = []
    for c in range(n):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        support = int(cm[c, :].sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        rows.append(
            {
                "class_id": c,
                "class_name": label_map.get(c, f"class_{c}"),
                "support_faces": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "iou": iou,
            }
        )
    return rows


def write_confusion_csv(path: Path, cm: np.ndarray, label_map: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = cm.shape[0]
    header = ["true\\pred"] + [label_map.get(j, f"class_{j}") for j in range(n)]
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for i in range(n):
            w.writerow([label_map.get(i, f"class_{i}")] + [int(cm[i, j]) for j in range(n)])


def write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "class_id",
        "class_name",
        "support_faces",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "iou",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = dict(row)
            for k in ("precision", "recall", "iou"):
                v = out[k]
                out[k] = "" if v != v else f"{float(v):.6g}"
            w.writerow(out)


def write_summary_md(
    path: Path,
    *,
    onnx_path: Path,
    output_dir: Path,
    dataset_root: Optional[Path],
    split_file: Optional[Path],
    pyg_dir: Path,
    n_listed: int,
    n_missing: int,
    n_candidates: int,
    n_graphs: int,
    n_faces: int,
    labelled_faces: int,
    mean_top1: Optional[float],
    per_class: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ONNX PyG inference summary (v2, Thread + Text)",
        "",
        f"- **onnx**: `{onnx_path}`",
        f"- **pyg_dir**: `{pyg_dir}`",
        f"- **output_dir**: `{output_dir}`",
    ]
    if dataset_root is not None:
        lines.append(f"- **dataset_path**: `{dataset_root}`")
    if split_file is not None:
        lines.append(f"- **split_file**: `{split_file}`")
        lines.append(f"- **split_listed**: {n_listed}")
        lines.append(f"- **split_missing_on_disk**: {n_missing}")
    lines += [
        f"- **pt_candidates**: {n_candidates}",
        f"- **graphs_written**: {n_graphs}",
        f"- **total_faces**: {n_faces}",
        f"- **labelled_faces** (GT available): {labelled_faces}",
    ]
    if mean_top1 is not None:
        lines.append(f"- **per_face_accuracy**: {100.0 * mean_top1:.4f}%")
    ious = [float(r["iou"]) for r in per_class if r["iou"] == r["iou"]]
    if ious:
        miou = float(np.mean(ious))
        lines.append(f"- **mIoU** (macro mean of per-class IoU): {miou:.6f} ({100.0 * miou:.4f}%)")
    lines += ["", "## Per-class (face-level)", ""]
    for r in per_class:
        p, rcl, iou = r["precision"], r["recall"], r["iou"]
        ps = "n/a" if p != p else f"{100.0 * float(p):.2f}%"
        rs = "n/a" if rcl != rcl else f"{100.0 * float(rcl):.2f}%"
        is_ = "n/a" if iou != iou else f"{float(iou):.6f}"
        lines.append(
            f"- **{r['class_name']}** (id={r['class_id']}): "
            f"support={r['support_faces']} IoU={is_} precision={ps} recall={rs}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    label_map = load_label_map(args.label_map)
    num_classes = len(label_map)
    if num_classes < 2:
        raise ValueError(f"Label map must have at least 2 classes; got {num_classes}.")

    graphs, split_file, dataset_root = resolve_graphs(args)
    if args.input is not None:
        pyg_dir = dataset_root  # for sidecars, fall back to the input parent
        dataset_root_for_sidecars = dataset_root
    else:
        pyg_dir = dataset_root / args.pyg_subdir
        dataset_root_for_sidecars = dataset_root

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = (
        args.metrics_dir.expanduser().resolve() if args.metrics_dir is not None else output_dir
    )

    import onnxruntime as ort

    providers = select_providers(args.provider)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = {item.name for item in session.get_inputs()}
    missing_inputs = sorted(LITE_REQUIRED_INPUTS - input_names)
    unsupported_inputs = sorted(input_names - LITE_REQUIRED_INPUTS)
    if missing_inputs or unsupported_inputs:
        raise RuntimeError(
            "Unexpected ONNX input contract. "
            f"Missing lite inputs={missing_inputs}; unsupported inputs={unsupported_inputs}."
        )
    if len(session.get_outputs()) != 1:
        raise RuntimeError("Expected exactly one ONNX output")
    output_name = session.get_outputs()[0].name

    print(f"[INFO] ONNX: {onnx_path}")
    print(f"[INFO] Providers in use: {session.get_providers()}")
    print(f"[INFO] Label map: {label_map}")
    if split_file is not None:
        print(f"[INFO] Split file: {split_file}")
    print(f"[INFO] PyG dir: {pyg_dir}")
    print(f"[INFO] Processing {len(graphs)} graph(s); CSV output: {output_dir}")

    n_listed = len(read_split_stems(split_file)) if split_file is not None else 0
    n_missing = n_listed - len(graphs) if split_file is not None else 0

    summary_rows: list[dict[str, str]] = []
    failures = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    labelled_faces = 0
    correct = 0
    total_faces = 0
    graphs_written = 0

    for graph_path in graphs:
        try:
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            ensure_lite_graph(graph, graph_path)
            batch = make_lite_batch(graph)
            ort_feed = batch_to_ort_feed(batch, input_names)
            probabilities = session.run([output_name], ort_feed)[0]
            if probabilities.ndim != 2:
                raise RuntimeError(f"Unexpected output shape: {probabilities.shape}")
            if probabilities.shape[1] != num_classes:
                raise RuntimeError(
                    f"Output has {probabilities.shape[1]} classes but label map has {num_classes}"
                )

            gt = resolve_gt(graph, graph_path.stem, dataset_root_for_sidecars, num_classes)

            csv_path = output_dir / f"{graph_path.stem}_predictions.csv"
            predicted_ids, confidences = write_predictions(csv_path, probabilities, label_map, gt)
            counts = Counter(predicted_ids)
            count_text = ", ".join(
                f"{label_map.get(cid, f'class_{cid}')}={counts[cid]}"
                for cid in sorted(counts)
            )
            total_faces += len(predicted_ids)
            graphs_written += 1
            preds_np = np.asarray(predicted_ids, dtype=np.int64)
            if gt is not None:
                labelled_faces += len(predicted_ids)
                correct += int((preds_np == gt).sum())
                all_preds.extend(predicted_ids)
                all_labels.extend(gt.tolist())
                gt_extra = f" acc={100.0 * float((preds_np == gt).mean()):.2f}%"
            else:
                gt_extra = ""
            print(
                f"[PASS] {graph_path.name}: faces={len(predicted_ids)} "
                f"mean_confidence={float(np.mean(confidences)):.4f}  {count_text}{gt_extra}"
            )
            summary_rows.append(
                {
                    "graph": str(graph_path),
                    "prediction_csv": str(csv_path),
                    "faces": str(len(predicted_ids)),
                    "mean_confidence": f"{float(np.mean(confidences)):.8f}",
                    "class_counts": count_text,
                    "has_gt": "yes" if gt is not None else "no",
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
                    "has_gt": "",
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
                    "has_gt": "",
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
                "has_gt",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[INFO] Summary: {summary_path}")

    if labelled_faces > 0:
        preds_np = np.asarray(all_preds, dtype=np.int64)
        labels_np = np.asarray(all_labels, dtype=np.int64)
        cm = confusion_matrix(preds_np, labels_np, num_classes)
        per_class = per_class_rows(cm, label_map)
        write_confusion_csv(metrics_dir / "confusion_matrix.csv", cm, label_map)
        write_per_class_csv(metrics_dir / "per_class.csv", per_class)
        write_summary_md(
            metrics_dir / "summary.md",
            onnx_path=onnx_path,
            output_dir=output_dir,
            dataset_root=dataset_root_for_sidecars if args.dataset_path is not None else None,
            split_file=split_file,
            pyg_dir=pyg_dir,
            n_listed=n_listed,
            n_missing=n_missing,
            n_candidates=len(graphs),
            n_graphs=graphs_written,
            n_faces=total_faces,
            labelled_faces=labelled_faces,
            mean_top1=(correct / labelled_faces) if labelled_faces > 0 else None,
            per_class=per_class,
        )
        print(f"[INFO] Metrics: {metrics_dir}")
    elif metrics_dir != output_dir:
        print(
            "[warn] --metrics-dir set but no ground-truth labels were found; "
            f"expected sidecars at {dataset_root_for_sidecars / 'label' / '<stem>.json'} "
            "with key 'labels' (length = num faces).",
            file=sys.stderr,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
