#!/usr/bin/env python3
"""Validate A1+A3 checkpoint/ONNX parity on one real PyG graph."""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from standalone_scripts.run_onnx_a1_a3_inference import (
    batch_to_ort_feed,
    ensure_a1_a3_graph,
    make_a1_a3_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def load_export_module(repo_root: Path):
    path = repo_root / "migration_to_c++" / "migration_to_c" / "export_a1_a3_onnx.py"
    spec = importlib.util.spec_from_file_location("export_a1_a3_onnx", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import exporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    exporter = load_export_module(repo_root)

    model, _ = exporter.load_brepseg(args.checkpoint.resolve(), device="cpu")
    wrapper = exporter.BrepMFRONNXWrapper(model).eval()

    graph_path = args.graph.resolve()
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    ensure_a1_a3_graph(graph, graph_path)
    batch = make_a1_a3_batch(graph)

    session = ort.InferenceSession(
        str(args.onnx.resolve()), providers=["CPUExecutionProvider"]
    )
    input_names = {item.name for item in session.get_inputs()}
    feed = batch_to_ort_feed(batch, input_names)

    with torch.no_grad():
        torch_probs = wrapper(**batch).detach().cpu().numpy()
    onnx_probs = session.run(None, feed)[0]

    if torch_probs.shape != onnx_probs.shape:
        raise RuntimeError(
            f"Shape mismatch: PyTorch={torch_probs.shape}, ONNX={onnx_probs.shape}"
        )
    max_diff = float(np.max(np.abs(torch_probs - onnx_probs)))
    mean_diff = float(np.mean(np.abs(torch_probs - onnx_probs)))
    torch_labels = np.argmax(torch_probs, axis=1)
    onnx_labels = np.argmax(onnx_probs, axis=1)
    label_matches = int(np.sum(torch_labels == onnx_labels))
    total = int(torch_labels.size)

    print(f"Graph: {graph_path}")
    print(f"Faces: {total}")
    print(f"Max probability difference: {max_diff:.3e}")
    print(f"Mean probability difference: {mean_diff:.3e}")
    print(f"Label matches: {label_matches}/{total}")
    print(f"PyTorch counts: {dict(Counter(map(int, torch_labels)))}")
    print(f"ONNX counts: {dict(Counter(map(int, onnx_labels)))}")
    if label_matches != total or max_diff > args.atol:
        print("FAIL: checkpoint and ONNX predictions are not within tolerance")
        return 1
    print("PASS: checkpoint and ONNX agree on the real graph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
