#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP → ONNX lite inference (no SOLIDWORKS, pure Python).

For every .step and .stp file in the input folder:
  1. Parses the STEP file via occwl + pythonocc.
  2. Converts the B-rep solid to a PyG-style graph.
  3. Strips non-lite tensors so the lite ONNX model accepts it.
  4. Runs the ONNX model via ONNX Runtime.
  5. Writes a per-face predictions CSV to the output folder.

A summary CSV (onnx_step_inference_summary.csv) is also written.

Usage (from repo root):
    conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_step_inference.py

    # Or with custom paths:
    conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_step_inference.py ^
      --input-dir  \\\\GR-SW65551\\abc_steps ^
      --output-dir C:\\Users\\RZA2\\Desktop\\onnx_inference ^
      --model-dir  standalone_scripts\\BrepMFR_lite_onnx_pyg_demo_v2

Requirements:
    pythonocc-core, occwl, torch, torch_geometric, onnxruntime, numpy
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Bootstrap: make sure the repo root and tools/pipeline are on sys.path
# so that occwl_to_brep_tensors and the occwl patch can be imported.
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent  # BrepMFR_PyG root

_pipeline_dir = _REPO_ROOT / "tools" / "pipeline"
for _p in [str(_REPO_ROOT), str(_pipeline_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Defaults matching the user's setup
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR = Path(r"\\GR-SW65551\abc_steps")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\RZA2\Desktop\onnx_inference")
DEFAULT_MODEL_DIR = _THIS.parent / "BrepMFR_lite_onnx_pyg_demo_v2"

# ---------------------------------------------------------------------------
# ONNX lite model input contract
# ---------------------------------------------------------------------------
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


class SkipFile(Exception):
    """A STEP file that cannot produce predictions (e.g. no solids, zero faces)."""


# ---- CLI -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the lite ONNX model on STEP files (pure Python, no SOLIDWORKS). "
            "Converts each STEP to a PyG graph via occwl, then runs ONNX Runtime."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing .step / .stp files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for per-file prediction CSVs and the summary CSV.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Folder containing brepmfr_lite.onnx, label_map.json, "
            "and model_config.json."
        ),
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Override: path to the .onnx model (default: <model-dir>/brepmfr_lite.onnx).",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Override: path to label_map.json.",
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
        help="Limit to this many STEP files (for testing).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search sub-folders for STEP files.",
    )
    return parser.parse_args()


# ---- Helpers ---------------------------------------------------------------

def find_step_files(input_dir: Path, recursive: bool, max_files: Optional[int]) -> list[Path]:
    """Collect .step and .stp files from input_dir."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    glob_fn = input_dir.rglob if recursive else input_dir.glob
    files: list[Path] = []
    for ext in ("*.step", "*.stp", "*.STEP", "*.STP"):
        files.extend(p for p in glob_fn(ext) if p.is_file())
    # Deduplicate (case-insensitive match on Windows might double-count)
    seen: set[str] = set()
    unique: list[Path] = []
    for f in sorted(files):
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    if max_files is not None:
        unique = unique[: max(0, max_files)]
    if not unique:
        raise FileNotFoundError(f"No .step or .stp files found in: {input_dir}")
    return unique


def load_label_map(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def select_providers(choice: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDA provider unavailable; available: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # auto
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ---- STEP → PyG (direct OCC, no occwl.compound/graph) ----------------------
#
# occwl.compound and occwl.graph both import OCC.Extend.DataExchange which
# pulls in OCC.Core.XCAFDoc — a module whose DLLs are broken on this system.
# We work around this by:
#   1. Loading STEP files with OCC.Core.STEPControl directly.
#   2. Wrapping the result in occwl.solid.Solid (which imports cleanly).
#   3. Inlining the face_adjacency graph construction.
#   4. Re-using tensor_dict_from_face_adjacency for feature extraction
#      by importing it carefully (it only triggers the problem if occwl
#      is already imported via its broken path, but we pre-patch here).

def _load_step_as_solids(step_path: Path):
    """Load a STEP file and return a list of occwl.solid.Solid objects."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopoDS import topods_Solid
    from occwl.solid import Solid

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEPControl_Reader failed with status {status}")

    reader.TransferRoots()
    shape = reader.OneShape()

    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(Solid(topods_Solid(explorer.Current())))
        explorer.Next()

    return solids


def _face_adjacency(solid):
    """Build a face adjacency nx.DiGraph from an occwl.solid.Solid.

    This is equivalent to occwl.graph.face_adjacency but avoids importing
    occwl.graph (which transitively imports the broken XCAFDoc module).
    """
    import networkx as nx
    from occwl.entity_mapper import EntityMapper

    mapper = EntityMapper(solid)
    graph = nx.DiGraph()

    for face in solid.faces():
        face_idx = mapper.face_index(face)
        graph.add_node(face_idx, face=face)

    for edge in solid.edges():
        if not edge.has_curve():
            continue
        connected_faces = list(solid.faces_from_edge(edge))
        if len(connected_faces) < 2:
            # Skip seam / free edges
            continue
        elif len(connected_faces) == 2:
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue
            if not mapper.oriented_edge_exists(edge):
                continue
            edge_idx = mapper.oriented_edge_index(edge)
            edge_reversed = edge.reversed_edge()
            if not mapper.oriented_edge_exists(edge_reversed):
                continue
            edge_reversed_idx = mapper.oriented_edge_index(edge_reversed)
            left_index = mapper.face_index(left_face)
            right_index = mapper.face_index(right_face)
            graph.add_edge(left_index, right_index, edge=edge, edge_index=edge_idx)
            graph.add_edge(right_index, left_index, edge=edge_reversed, edge_index=edge_reversed_idx)
        else:
            raise RuntimeError("Non-manifold edge incident on >2 faces")

    return graph


def _solid_to_pyg_data(solid, data_id: int = 0):
    """Convert an occwl Solid → PyG Data using tensor_dict_from_face_adjacency.

    We import occwl_to_brep_tensors.tensor_dict_from_face_adjacency directly;
    it only uses occwl_pythonocc_patch (safe) and occwl.uvgrid (safe).
    """
    from torch import FloatTensor
    from torch_geometric.data import Data as PYGGraph

    # Import just the tensor builder (not convert_stp_path_to_pyg which
    # would pull in occwl.compound).
    from occwl_to_brep_tensors import tensor_dict_from_face_adjacency

    adj = _face_adjacency(solid)
    t = tensor_dict_from_face_adjacency(adj, 5, 5, 5)

    n = int(t["num_nodes"])
    pyg = PYGGraph()
    pyg.node_data = t["node_data"].type(FloatTensor)
    pyg.face_type = t["face_type"].type(torch.int)
    pyg.face_area = t["face_area"].type(torch.float)
    pyg.face_loop = t["face_loop"].type(torch.int)
    pyg.face_adj = t["face_adj"].type(torch.int)
    pyg.label_feature = t["label_feature"].type(torch.int)
    pyg.edge_index = t["edge_index"].long()
    pyg.attn_bias = torch.zeros([n + 1, n + 1], dtype=torch.float)

    row, col = pyg.edge_index
    deg = torch.zeros(n, dtype=torch.long)
    if row.numel() > 0:
        deg.index_add_(0, row, torch.ones_like(row))
        deg.index_add_(0, col, torch.ones_like(col))
    pyg.node_degree = deg
    pyg.data_id = int(data_id)
    return pyg


def step_to_lite_pyg(step_path: Path) -> Any:
    """
    Parse a STEP file and return a PyG-like graph object suitable for the
    lite ONNX model. Uses STEPControl_Reader directly (no XCAFDoc dependency).
    """
    solids = _load_step_as_solids(step_path)
    if not solids:
        raise SkipFile(f"No solids found in STEP file: {step_path.name}")

    stem = step_path.stem
    try:
        data_id = int(stem.split("_")[-1])
    except ValueError:
        data_id = 0

    pyg = _solid_to_pyg_data(solids[0], data_id=data_id)

    n_faces = int(pyg.node_data.size(0))
    if n_faces == 0:
        raise SkipFile(f"Zero faces in STEP file: {step_path.name}")

    return pyg


# ---- Lite batch + ORT feed -------------------------------------------------

def make_lite_batch(graph: Any) -> dict[str, torch.Tensor]:
    """Build the flat-tensor dict the ONNX model expects from one PyG graph."""
    node_data = graph.node_data
    n_faces = int(node_data.size(0))

    if node_data.ndim != 4 or tuple(node_data.shape[1:]) != (5, 5, 7):
        raise ValueError(
            f"node_data shape must be [N, 5, 5, 7], got {tuple(node_data.shape)}"
        )

    flat: dict[str, torch.Tensor] = {
        "face_area": graph.face_area,
        "face_type": graph.face_type,
        "face_loop": graph.face_loop,
        "in_degree": graph.node_degree,
    }
    for name, tensor in flat.items():
        if not torch.is_tensor(tensor) or tensor.numel() != n_faces:
            raise ValueError(
                f"{name} must have {n_faces} elements, "
                f"got {tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor)}"
            )
        flat[name] = tensor.reshape(-1)

    attn_bias = getattr(graph, "attn_bias", None)
    if attn_bias is None:
        attn_bias = torch.zeros(n_faces + 1, n_faces + 1, dtype=torch.float32)
    if tuple(attn_bias.shape) != (n_faces + 1, n_faces + 1):
        raise ValueError(
            f"attn_bias shape must be [{n_faces+1}, {n_faces+1}], "
            f"got {tuple(attn_bias.shape)}"
        )

    return {
        "node_data": node_data,
        **flat,
        "attn_bias": attn_bias.unsqueeze(0),
        "padding_mask": torch.zeros(1, n_faces, dtype=torch.bool),
    }


def batch_to_ort_feed(
    batch: dict[str, Any], input_names: set[str]
) -> dict[str, np.ndarray]:
    """Cast tensors to the dtypes ONNX Runtime expects and convert to numpy."""
    missing = sorted(input_names - set(batch))
    if missing:
        raise ValueError(f"Missing ONNX inputs: {missing}")

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


# ---- CSV writing -----------------------------------------------------------

def write_predictions_csv(
    output_path: Path,
    probabilities: np.ndarray,
    label_map: dict[int, str],
    part_name: str,
    body_index: int = 1,
) -> tuple[list[int], list[float]]:
    """Write per-face predictions CSV. Returns (predicted_ids, confidences)."""
    predicted_ids = probabilities.argmax(axis=1).astype(int).tolist()
    confidences = probabilities.max(axis=1).astype(float).tolist()

    fields = [
        "part_name",
        "body_index",
        "face_index",
        "original_face_id",
        "predicted_class",
        "predicted_class_name",
        "predicted_probability",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for face_idx, (cid, conf) in enumerate(
            zip(predicted_ids, confidences)
        ):
            row = {
                "part_name": part_name,
                "body_index": body_index,
                "face_index": face_idx,
                "original_face_id": face_idx,
                "predicted_class": cid,
                "predicted_class_name": label_map.get(cid, f"class_{cid}"),
                "predicted_probability": f"{conf:.8f}",
            }
            writer.writerow(row)

    return predicted_ids, confidences


# ---- Main ------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Resolve model assets
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    onnx_path = (
        args.onnx.expanduser().resolve()
        if args.onnx
        else model_dir / "brepmfr_lite.onnx"
    )
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    label_map_path = (
        args.label_map.expanduser().resolve()
        if args.label_map
        else model_dir / "label_map.json"
    )
    if not label_map_path.is_file():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    label_map = load_label_map(label_map_path)
    num_classes = len(label_map)

    # Resolve I/O directories
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step_files = find_step_files(input_dir, args.recursive, args.max_files)

    # Create ONNX Runtime session
    import onnxruntime as ort

    providers = select_providers(args.provider)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_names = {item.name for item in session.get_inputs()}

    missing_inputs = sorted(LITE_REQUIRED_INPUTS - input_names)
    extra_inputs = sorted(input_names - LITE_REQUIRED_INPUTS)
    if missing_inputs or extra_inputs:
        raise RuntimeError(
            f"ONNX input mismatch. Missing: {missing_inputs}; Extra: {extra_inputs}"
        )
    output_name = session.get_outputs()[0].name

    print(f"[INFO] ONNX model:  {onnx_path}")
    print(f"[INFO] Label map:   {label_map}")
    print(f"[INFO] Providers:   {session.get_providers()}")
    print(f"[INFO] Input dir:   {input_dir}")
    print(f"[INFO] Output dir:  {output_dir}")
    print(f"[INFO] STEP files:  {len(step_files)}")
    print()

    # Process each STEP file
    summary_rows: list[dict[str, str]] = []
    successes = 0
    failures = 0
    skipped = 0
    total_start = time.time()

    for idx, step_path in enumerate(step_files, 1):
        file_start = time.time()
        tag = f"[{idx}/{len(step_files)}]"

        try:
            # 1. STEP → PyG
            pyg = step_to_lite_pyg(step_path)

            # 2. PyG → flat tensors
            batch = make_lite_batch(pyg)

            # 3. Flat tensors → numpy feed
            ort_feed = batch_to_ort_feed(batch, input_names)

            # 4. Run ONNX inference
            probabilities = session.run([output_name], ort_feed)[0]

            if probabilities.ndim != 2:
                raise RuntimeError(
                    f"Unexpected output shape: {probabilities.shape}"
                )
            if probabilities.shape[1] != num_classes:
                raise RuntimeError(
                    f"Output has {probabilities.shape[1]} classes "
                    f"but label map has {num_classes}"
                )

            # 5. Write predictions CSV
            csv_path = output_dir / f"{step_path.stem}_predictions.csv"
            predicted_ids, confidences = write_predictions_csv(
                csv_path, probabilities, label_map, step_path.stem
            )

            elapsed = time.time() - file_start
            counts = Counter(predicted_ids)
            count_text = ", ".join(
                f"{label_map.get(cid, f'class_{cid}')}={counts[cid]}"
                for cid in sorted(counts)
            )

            print(
                f"[PASS] {tag} {step_path.name}: "
                f"faces={len(predicted_ids)}  "
                f"mean_conf={np.mean(confidences):.4f}  "
                f"{count_text}  "
                f"({elapsed:.1f}s)"
            )

            summary_rows.append({
                "step_file": str(step_path),
                "prediction_csv": str(csv_path),
                "faces": str(len(predicted_ids)),
                "mean_confidence": f"{np.mean(confidences):.8f}",
                "class_counts": count_text,
                "elapsed_seconds": f"{elapsed:.2f}",
                "status": "PASS",
                "error": "",
            })
            successes += 1

        except SkipFile as exc:
            elapsed = time.time() - file_start
            print(f"[SKIP] {tag} {step_path.name}: {exc}")
            skipped += 1
            summary_rows.append({
                "step_file": str(step_path),
                "prediction_csv": "",
                "faces": "0",
                "mean_confidence": "",
                "class_counts": "",
                "elapsed_seconds": f"{elapsed:.2f}",
                "status": "SKIP",
                "error": str(exc),
            })

        except Exception as exc:
            elapsed = time.time() - file_start
            print(
                f"[FAIL] {tag} {step_path.name}: {exc}",
                file=sys.stderr,
            )
            failures += 1
            summary_rows.append({
                "step_file": str(step_path),
                "prediction_csv": "",
                "faces": "",
                "mean_confidence": "",
                "class_counts": "",
                "elapsed_seconds": f"{elapsed:.2f}",
                "status": "FAIL",
                "error": str(exc),
            })

    # Write summary CSV
    summary_path = output_dir / "onnx_step_inference_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "step_file",
                "prediction_csv",
                "faces",
                "mean_confidence",
                "class_counts",
                "elapsed_seconds",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    total_elapsed = time.time() - total_start
    print()
    print("=" * 70)
    print(f"STEP files found:  {len(step_files)}")
    print(f"Successful:        {successes}")
    print(f"Failed:            {failures}")
    print(f"Skipped:           {skipped}")
    print(f"Total time:        {total_elapsed:.1f}s")
    print(f"Summary CSV:       {summary_path}")
    print(f"Predictions dir:   {output_dir}")
    print("=" * 70)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
