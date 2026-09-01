# -*- coding: utf-8 -*-
"""
BrepMFR A1+A3 Fine-Tuned Model → ONNX Export & Validation
==========================================================

Exports the A1+A3 fine-tuned BrepSeg checkpoint (trained with
train_a1_a3_from_lite.ps1) to ONNX, validates numerical parity,
and creates the supporting config files for C++ deployment.

Usage (from repo root):
    conda activate brep_mfr_pyg

    python migration_to_c++/migration_to_c/export_a1_a3_onnx.py ^
        --checkpoint model_checkpoints/abc_with_no_a2/last-v1.ckpt

Outputs:
    migration_to_c++/migration_to_c/exported_a1_a3/brepmfr_a1_a3.onnx
    migration_to_c++/migration_to_c/exported_a1_a3/model_config.json
    migration_to_c++/migration_to_c/exported_a1_a3/label_map.json

Requirements:
    pip install onnx onnxruntime  (or onnxruntime-gpu)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Bootstrap repo root so `models.*` and `data.*` resolve ──
_THIS = Path(__file__).resolve()
for _anc in _THIS.parents:
    _bst = _anc / "bootstrap_path.py"
    if _bst.is_file():
        _spec = importlib.util.spec_from_file_location("__brepmfr_bootstrap", _bst)
        _bm = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_bm)
        _bm.setup(str(_THIS))
        break
else:
    # Fallback: assume script is two levels inside repo
    _repo = str(_THIS.parent.parent.parent)
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.brepseg_model import BrepSeg, Attention, NonLinearClassifier

# ═══════════════════════════════════════════════════════════════════
# Label map — 3-class Thread/Text model
# ═══════════════════════════════════════════════════════════════════
LABEL_MAP = {
    0: "Stock",
    1: "Thread",
    2: "Text",
}

# ═══════════════════════════════════════════════════════════════════
# 1. ONNX WRAPPER
# ═══════════════════════════════════════════════════════════════════

class BrepMFRONNXWrapper(nn.Module):
    """Flat-tensor wrapper for ONNX export.

    Accepts explicit tensor arguments (no Python dicts, no PyG objects).
    Uses the ``no_a2`` inference path (A1 spatial_pos + A3 edge_path, no
    A2 d2_distance / angle_distance).
    Single-graph inference (batch_size = 1).

    ONNX export calls ``forward()`` with named tensors; the wrapper
    re-packs them into the dict the encoder expects and runs:
        encoder → strip VNode → attention fusion → classifier → softmax probs
    """

    def __init__(self, brepseg: BrepSeg):
        super().__init__()
        self.brep_encoder = brepseg.brep_encoder
        self.attention_layer = brepseg.attention  # Attention module
        self.classifier = brepseg.classifier

    def forward(
        self,
        # ── Node (face) tensors ──
        node_data: torch.Tensor,        # [total_nodes, U, V, C]     float32
        face_area: torch.Tensor,        # [total_nodes]              float32
        face_type: torch.Tensor,        # [total_nodes]              int64
        face_loop: torch.Tensor,        # [total_nodes]              int64
        in_degree: torch.Tensor,        # [total_nodes]              int64
        # ── Edge tensors ──
        edge_data: torch.Tensor,        # [total_edges, L, C]        float32
        edge_type: torch.Tensor,        # [total_edges]              int64
        edge_len: torch.Tensor,         # [total_edges]              float32
        edge_ang: torch.Tensor,         # [total_edges]              float32
        edge_conv: torch.Tensor,        # [total_edges]              int64
        edge_index: torch.Tensor,       # [2, total_edges]           int64
        # ── Graph structure (batch_size=1) ──
        attn_bias: torch.Tensor,        # [1, N+1, N+1]             float32
        spatial_pos: torch.Tensor,      # [1, N, N]                 int64
        edge_path: torch.Tensor,        # [1, N, N, K]              int64
        padding_mask: torch.Tensor,     # [1, N]                    bool
        edge_padding_mask: torch.Tensor,# [1, max_edges]            bool
    ) -> torch.Tensor:
        """Returns probabilities [total_nodes, num_classes] (after softmax)."""

        batch_data = {
            "node_data": node_data,
            "face_area": face_area,
            "face_type": face_type,
            "face_loop": face_loop,
            "in_degree": in_degree,
            "edge_data": edge_data,
            "edge_type": edge_type,
            "edge_len": edge_len,
            "edge_ang": edge_ang,
            "edge_conv": edge_conv,
            "edge_index": edge_index,
            "attn_bias": attn_bias,
            "spatial_pos": spatial_pos,
            "edge_path": edge_path,
            "padding_mask": padding_mask,
            "edge_padding_mask": edge_padding_mask,
            # A2 tensors are None → no_a2 profile
            "d2_distance": None,
            "angle_distance": None,
        }

        # ── Encoder ──
        node_emb, graph_emb = self.brep_encoder(batch_data, last_state_only=True)

        # node_emb is a list; [0] has shape [N+1, B, D]
        node_emb = node_emb[0].permute(1, 0, 2)   # → [B, N+1, D]
        node_emb = node_emb[:, 1:, :]              # → [B, N, D]  strip virtual node

        # ── Extract non-padded nodes ──
        node_pos = torch.where(~padding_mask)  # noqa: E712
        node_z = node_emb[node_pos]                    # [total_real_nodes, D]

        num_nodes_per_graph = (~padding_mask).sum(dim=-1)  # [B]
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0)

        # ── Attention fusion + classifier ──
        z = self.attention_layer([node_z, graph_z])
        probs = self.classifier(z)  # classifier.forward() applies softmax

        return probs


# ═══════════════════════════════════════════════════════════════════
# 2. CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════

def _namespace_from_ckpt(ckpt: Dict[str, Any]) -> Namespace:
    """Extract hyperparameters from a Lightning checkpoint."""
    h = ckpt.get("hyper_parameters")
    if not h:
        raise ValueError("Checkpoint missing hyper_parameters")
    if "args" in h:
        a = h["args"]
        if isinstance(a, Namespace):
            return Namespace(**vars(a))
        elif isinstance(a, dict):
            return Namespace(**a)
        return Namespace(**vars(a))
    return Namespace(**{k: v for k, v in h.items() if k != "args"})


def load_brepseg(ckpt_path: Path, device: str = "cpu") -> Tuple[BrepSeg, Namespace]:
    """Load a BrepSeg model from a Lightning checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise ValueError("Not a Lightning checkpoint (no state_dict key)")

    args = _namespace_from_ckpt(ckpt)

    # Disable training-only features
    args.pre_train = None
    args.warmup_freeze_epochs = 0
    # No A3 cap for export — we want full A3 attention at inference
    args.max_nodes_for_a3 = None

    # Handle class weights path that may not exist on this machine
    cw = getattr(args, "class_weights_path", None)
    if cw and not pathlib.Path(cw).expanduser().is_file():
        args.class_weights_path = None

    model = BrepSeg(args)
    state = ckpt["state_dict"]

    # Load weights (filter to encoder/attention/classifier)
    seg_sd = {
        k: v for k, v in state.items()
        if k.startswith(("brep_encoder.", "attention.", "classifier."))
    }
    if "class_weights" in state:
        seg_sd["class_weights"] = state["class_weights"]

    # Also load A1/A3 scale buffer if present
    a1_a3_key = "brep_encoder.graph_attn_bias.a1_a3_scale"
    if a1_a3_key in state:
        seg_sd[a1_a3_key] = state[a1_a3_key]

    if not seg_sd:
        raise ValueError("No brep_encoder/attention/classifier weights found")

    incompatible = model.load_state_dict(seg_sd, strict=False)
    ignorable = {"class_weights", "_val_confusion"}
    bad_missing = [k for k in incompatible.missing_keys
                   if not k.startswith("_") and k not in ignorable]
    if bad_missing:
        print(f"  WARNING: Missing keys: {bad_missing[:10]}")

    if incompatible.unexpected_keys:
        print(f"  NOTE: Unexpected keys (ignored): {incompatible.unexpected_keys[:5]}")

    # Force A1/A3 scale to 1.0 for inference (training ramps it from 0.1)
    if hasattr(model.brep_encoder, "graph_attn_bias"):
        model.brep_encoder.graph_attn_bias.set_a1_a3_scale(1.0)
        print("  Set A1/A3 scale to 1.0 for inference")

    model.eval()
    model.to(device)

    num_classes = int(getattr(args, "num_classes", 3))
    print(f"  Model loaded: num_classes={num_classes}, "
          f"dim_node={getattr(args, 'dim_node', '?')}, "
          f"d_model={getattr(args, 'd_model', '?')}, "
          f"n_heads={getattr(args, 'n_heads', '?')}, "
          f"n_layers={getattr(args, 'n_layers_encode', '?')}")

    return model, args


# ═══════════════════════════════════════════════════════════════════
# 3. DUMMY INPUT GENERATION
# ═══════════════════════════════════════════════════════════════════

def create_dummy_inputs(
    N: int = 50,
    E: int = 120,
    K: int = 16,
    spatial_pos_max: int = 32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Create a set of realistic dummy inputs for a single graph.

    Args:
        N: Number of faces (nodes)
        E: Number of directed edges
        K: max_edge_path_len (multi_hop_max_dist)
        spatial_pos_max: Max spatial position value
        device: Target device
    """
    # Node features
    node_data = torch.randn(N, 5, 5, 7, device=device)
    face_area = torch.rand(N, device=device) * 100.0
    face_type = torch.randint(0, 7, (N,), device=device, dtype=torch.int64)
    face_loop = torch.randint(0, 10, (N,), device=device, dtype=torch.int64)
    in_degree = torch.randint(1, 20, (N,), device=device, dtype=torch.int64)

    # Edge features — build valid edge_index first
    src_indices = torch.randint(0, N, (E,), device=device, dtype=torch.int64)
    dst_indices = torch.randint(0, N, (E,), device=device, dtype=torch.int64)
    edge_index = torch.stack([src_indices, dst_indices], dim=0)

    edge_data = torch.randn(E, 5, 7, device=device)
    edge_type = torch.randint(0, 5, (E,), device=device, dtype=torch.int64)
    edge_len = torch.rand(E, device=device) * 50.0
    edge_ang = (torch.rand(E, device=device) * 2 - 1) * 3.14159
    edge_conv = torch.randint(0, 3, (E,), device=device, dtype=torch.int64)

    # Graph structure — batch_size=1
    attn_bias = torch.zeros(1, N + 1, N + 1, device=device)
    spatial_pos = torch.randint(0, spatial_pos_max, (1, N, N),
                                device=device, dtype=torch.int64)
    # Self-distance = 0
    for i in range(N):
        spatial_pos[0, i, i] = 0

    edge_path = torch.randint(-1, E, (1, N, N, K), device=device, dtype=torch.int64)

    padding_mask = torch.zeros(1, N, device=device, dtype=torch.bool)
    edge_padding_mask = torch.zeros(1, E, device=device, dtype=torch.bool)

    return {
        "node_data": node_data,
        "face_area": face_area,
        "face_type": face_type,
        "face_loop": face_loop,
        "in_degree": in_degree,
        "edge_data": edge_data,
        "edge_type": edge_type,
        "edge_len": edge_len,
        "edge_ang": edge_ang,
        "edge_conv": edge_conv,
        "edge_index": edge_index,
        "attn_bias": attn_bias,
        "spatial_pos": spatial_pos,
        "edge_path": edge_path,
        "padding_mask": padding_mask,
        "edge_padding_mask": edge_padding_mask,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. PYTORCH SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════

def run_pytorch_checks(
    wrapper: BrepMFRONNXWrapper,
    num_classes: int,
    device: str = "cpu",
):
    """Run basic sanity checks on the PyTorch model before ONNX export."""
    print("\n" + "=" * 60)
    print("STEP 1: PyTorch Sanity Checks")
    print("=" * 60)

    test_sizes = [(20, 48), (50, 120), (100, 250)]
    all_passed = True

    for N, E in test_sizes:
        print(f"\n  Testing N={N} faces, E={E} edges...")
        inputs = create_dummy_inputs(N=N, E=E, device=device)

        with torch.no_grad():
            try:
                probs = wrapper(**inputs)
            except Exception as e:
                print(f"    ❌ FAILED: {e}")
                all_passed = False
                continue

        # Check output shape
        expected_shape = (N, num_classes)
        if probs.shape != expected_shape:
            print(f"    ❌ Shape mismatch: got {probs.shape}, expected {expected_shape}")
            all_passed = False
            continue

        # Check output is valid probabilities (classifier has softmax)
        prob_sum = probs.sum(dim=-1)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-3):
            print(f"    ⚠️  Output may not be probabilities (sum range: "
                  f"{prob_sum.min():.4f} - {prob_sum.max():.4f})")
        else:
            print(f"    ✅ Output shape correct: {probs.shape}")
            print(f"    ✅ Probabilities sum to 1.0 (within tolerance)")

        # Check no NaN/Inf
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"    ❌ Output contains NaN or Inf!")
            all_passed = False
            continue

        preds = probs.argmax(dim=-1)
        unique_preds = preds.unique()
        print(f"    ✅ No NaN/Inf in output")
        labels = [LABEL_MAP.get(int(c), f"class_{c}") for c in unique_preds.tolist()[:10]]
        print(f"    ℹ️  Predicted {len(unique_preds)} unique classes: {unique_preds.tolist()[:10]} ({labels})")

    # Determinism check
    print(f"\n  Testing determinism...")
    inputs = create_dummy_inputs(N=30, E=70, device=device)
    with torch.no_grad():
        out1 = wrapper(**inputs).clone()
        out2 = wrapper(**inputs).clone()
    if torch.allclose(out1, out2, atol=1e-7):
        print(f"    ✅ Deterministic: two runs produce identical output")
    else:
        max_diff = (out1 - out2).abs().max().item()
        print(f"    ⚠️  Non-deterministic: max diff = {max_diff:.2e}")

    return all_passed


# ═══════════════════════════════════════════════════════════════════
# 5. ONNX EXPORT
# ═══════════════════════════════════════════════════════════════════

def export_to_onnx(
    wrapper: BrepMFRONNXWrapper,
    output_path: Path,
    N: int = 50,
    E: int = 120,
    K: int = 16,
    opset_version: int = 17,
    device: str = "cpu",
) -> Path:
    """Export the wrapper to ONNX with dynamic axes for variable graph sizes."""
    print("\n" + "=" * 60)
    print("STEP 2: ONNX Export")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = create_dummy_inputs(N=N, E=E, K=K, device=device)

    # Ordered argument tuple for torch.onnx.export
    input_tuple = (
        inputs["node_data"],
        inputs["face_area"],
        inputs["face_type"],
        inputs["face_loop"],
        inputs["in_degree"],
        inputs["edge_data"],
        inputs["edge_type"],
        inputs["edge_len"],
        inputs["edge_ang"],
        inputs["edge_conv"],
        inputs["edge_index"],
        inputs["attn_bias"],
        inputs["spatial_pos"],
        inputs["edge_path"],
        inputs["padding_mask"],
        inputs["edge_padding_mask"],
    )

    input_names = [
        "node_data",
        "face_area",
        "face_type",
        "face_loop",
        "in_degree",
        "edge_data",
        "edge_type",
        "edge_len",
        "edge_ang",
        "edge_conv",
        "edge_index",
        "attn_bias",
        "spatial_pos",
        "edge_path",
        "padding_mask",
        "edge_padding_mask",
    ]

    output_names = ["probabilities"]

    # Dynamic axes: allow variable N (nodes) and E (edges)
    dynamic_axes = {
        "node_data":         {0: "total_nodes"},
        "face_area":         {0: "total_nodes"},
        "face_type":         {0: "total_nodes"},
        "face_loop":         {0: "total_nodes"},
        "in_degree":         {0: "total_nodes"},
        "edge_data":         {0: "total_edges"},
        "edge_type":         {0: "total_edges"},
        "edge_len":          {0: "total_edges"},
        "edge_ang":          {0: "total_edges"},
        "edge_conv":         {0: "total_edges"},
        "edge_index":        {1: "total_edges"},
        "attn_bias":         {1: "num_nodes_plus_one", 2: "num_nodes_plus_one"},
        "spatial_pos":       {1: "num_nodes", 2: "num_nodes"},
        "edge_path":         {1: "num_nodes", 2: "num_nodes"},
        "padding_mask":      {1: "num_nodes"},
        "edge_padding_mask": {1: "max_edges"},
        "probabilities":     {0: "total_nodes"},
    }

    print(f"  Exporting with dummy graph: N={N}, E={E}, K={K}")
    print(f"  ONNX opset version: {opset_version}")
    print(f"  Output: {output_path}")

    t0 = time.perf_counter()
    torch.onnx.export(
        wrapper,
        input_tuple,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  ✅ Export completed in {elapsed:.1f}s")

    # Validate ONNX model structure
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ✅ ONNX model passes onnx.checker validation")

        # Print model info
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ℹ️  File size: {file_size_mb:.1f} MB")
        print(f"  ℹ️  IR version: {onnx_model.ir_version}")
        print(f"  ℹ️  Opset: {[o.version for o in onnx_model.opset_import]}")
        print(f"  ℹ️  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"  ℹ️  Outputs: {[out.name for out in onnx_model.graph.output]}")

    except ImportError:
        print(f"  ⚠️  `onnx` package not installed — skipping structural validation")
    except Exception as e:
        print(f"  ❌ ONNX validation failed: {e}")
        raise

    return output_path


# ═══════════════════════════════════════════════════════════════════
# 6. ONNX RUNTIME VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_onnx_vs_pytorch(
    wrapper: BrepMFRONNXWrapper,
    onnx_path: Path,
    device: str = "cpu",
    num_tests: int = 5,
    atol: float = 1e-4,
) -> bool:
    """Compare ONNX Runtime outputs against PyTorch outputs."""
    print("\n" + "=" * 60)
    print("STEP 3: ONNX Runtime vs PyTorch Validation")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("  ❌ onnxruntime not installed. Install with:")
        print("     pip install onnxruntime")
        print("     # or: pip install onnxruntime-gpu")
        return False

    # Create ONNX Runtime session
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]
    if device != "cpu":
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"  Using CUDA execution provider")
        else:
            print(f"  CUDA EP not available, falling back to CPU")

    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)

    print(f"  ONNX Runtime version: {ort.__version__}")
    print(f"  Providers: {session.get_providers()}")

    # List expected inputs
    ort_inputs = {inp.name: inp.shape for inp in session.get_inputs()}
    print(f"  Model inputs: {list(ort_inputs.keys())}")

    test_configs = [
        (20, 48, "small"),
        (50, 120, "medium"),
        (100, 250, "large"),
        (200, 500, "xl"),
        (30, 70, "odd"),
    ][:num_tests]

    all_passed = True
    for N, E, label in test_configs:
        print(f"\n  Test '{label}': N={N}, E={E}")

        inputs = create_dummy_inputs(N=N, E=E, device="cpu")

        # ── PyTorch forward ──
        with torch.no_grad():
            # Move inputs to model device for PyTorch
            pt_inputs = {k: v.to(device) for k, v in inputs.items()}
            pt_probs = wrapper(**pt_inputs).cpu().numpy()

        # ── ONNX Runtime forward ──
        ort_feed = {}
        for name in ort_inputs:
            tensor = inputs[name]
            ort_feed[name] = tensor.numpy()

        ort_probs = session.run(["probabilities"], ort_feed)[0]

        # ── Compare ──
        if pt_probs.shape != ort_probs.shape:
            print(f"    ❌ Shape mismatch: PyTorch={pt_probs.shape}, ORT={ort_probs.shape}")
            all_passed = False
            continue

        abs_diff = np.abs(pt_probs - ort_probs)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()

        # Compare argmax predictions
        pt_preds = pt_probs.argmax(axis=-1)
        ort_preds = ort_probs.argmax(axis=-1)
        pred_match = (pt_preds == ort_preds).sum()
        pred_total = len(pt_preds)

        if max_diff <= atol:
            print(f"    ✅ PASS  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  "
                  f"labels={pred_match}/{pred_total}")
        elif pred_match == pred_total and max_diff <= 1e-2:
            print(f"    ⚠️  WARN  max_diff={max_diff:.2e} (> atol={atol}) but "
                  f"labels match {pred_match}/{pred_total}")
        else:
            print(f"    ❌ FAIL  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  "
                  f"labels={pred_match}/{pred_total}")
            all_passed = False

    # Latency comparison
    print(f"\n  Latency comparison (N=50, E=120, 10 runs):")
    inputs = create_dummy_inputs(N=50, E=120, device="cpu")

    # PyTorch latency
    pt_inputs_dev = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            wrapper(**pt_inputs_dev)
        t0 = time.perf_counter()
        for _ in range(10):
            wrapper(**pt_inputs_dev)
        pt_time = (time.perf_counter() - t0) / 10 * 1000

    # ORT latency
    ort_feed = {name: inputs[name].numpy() for name in ort_inputs}
    for _ in range(3):
        session.run(["probabilities"], ort_feed)
    t0 = time.perf_counter()
    for _ in range(10):
        session.run(["probabilities"], ort_feed)
    ort_time = (time.perf_counter() - t0) / 10 * 1000

    print(f"    PyTorch:      {pt_time:.1f} ms/inference")
    print(f"    ONNX Runtime: {ort_time:.1f} ms/inference")
    speedup = pt_time / ort_time if ort_time > 0 else float("inf")
    print(f"    Speedup:      {speedup:.2f}x")

    return all_passed


# ═══════════════════════════════════════════════════════════════════
# 7. CONFIG EXPORT
# ═══════════════════════════════════════════════════════════════════

def export_configs(output_dir: Path, args: Namespace, num_classes: int):
    """Export model_config.json and label_map.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "num_classes": num_classes,
        "uv_grid_u": 5,
        "uv_grid_v": 5,
        "uv_channels": 7,
        "edge_grid_size": 5,
        "edge_channels": 7,
        "max_edge_path_len": 16,
        "spatial_pos_max": 32,
        "inference_profile": "no_a2",
        "model_variant": "a1_a3_finetuned",
        "training_info": {
            "base_model": "lite (no A1/A2/A3)",
            "finetuned_with": "A1 (spatial_pos) + A3 (edge_path)",
            "a2_status": "excluded (no d2_distance / angle_distance)",
        },
        "model_hyperparams": {
            "dim_node": getattr(args, "dim_node", None),
            "d_model": getattr(args, "d_model", None),
            "n_heads": getattr(args, "n_heads", None),
            "n_layers_encode": getattr(args, "n_layers_encode", None),
            "dropout": getattr(args, "dropout", None),
            "attention_dropout": getattr(args, "attention_dropout", None),
            "act_dropout": getattr(args, "act_dropout", None),
        },
        "input_dtype_policy": {
            "float_inputs": "float32",
            "index_inputs": "int64",
        },
    }

    config_path = output_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"  Saved: {config_path}")

    label_path = output_dir / "label_map.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in LABEL_MAP.items()}, f, indent=2)
    print(f"  Saved: {label_path}")


# ═══════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Export A1+A3 fine-tuned BrepMFR to ONNX and validate parity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to Lightning .ckpt file",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for exported files (default: <script_dir>/exported_a1_a3/)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for PyTorch inference (ONNX RT uses CPU by default)",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--skip_validation", action="store_true",
        help="Skip ONNX Runtime validation step",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _THIS.parent / "exported_a1_a3"

    onnx_path = output_dir / "brepmfr_a1_a3.onnx"

    # Use cpu for export to avoid CUDA issues during tracing
    export_device = "cpu"

    # ── Load model ──
    model, model_args = load_brepseg(ckpt_path, device=export_device)
    num_classes = int(getattr(model_args, "num_classes", 3))

    # Sanity check: num_classes must match label map
    if num_classes != len(LABEL_MAP):
        print(f"⚠️  WARNING: Checkpoint num_classes={num_classes} but label map has "
              f"{len(LABEL_MAP)} entries. Using num_classes from checkpoint.")

    # ── Create wrapper ──
    wrapper = BrepMFRONNXWrapper(model)
    wrapper.eval()
    wrapper.to(export_device)

    # ── Step 1: PyTorch sanity checks ──
    checks_passed = run_pytorch_checks(wrapper, num_classes, device=export_device)
    if not checks_passed:
        print("\n❌ PyTorch sanity checks failed. Fix issues before exporting.")
        sys.exit(1)

    # ── Step 2: ONNX export ──
    export_to_onnx(
        wrapper,
        onnx_path,
        opset_version=args.opset,
        device=export_device,
    )

    # ── Step 3: Validate ONNX vs PyTorch ──
    if not args.skip_validation:
        validation_passed = validate_onnx_vs_pytorch(
            wrapper, onnx_path, device=export_device,
        )
        if validation_passed:
            print("\n" + "=" * 60)
            print("✅ ALL CHECKS PASSED — A1+A3 ONNX model is ready for C++ deployment")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("⚠️  VALIDATION ISSUES DETECTED — review output above")
            print("=" * 60)
    else:
        print("\n  Skipping ONNX Runtime validation (--skip_validation)")

    # ── Step 4: Export configs ──
    print("\n" + "=" * 60)
    print("STEP 4: Export Configuration Files")
    print("=" * 60)
    export_configs(output_dir, model_args, num_classes)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"  ONNX model:    {onnx_path}")
    print(f"  Config:        {output_dir / 'model_config.json'}")
    print(f"  Label map:     {output_dir / 'label_map.json'}")
    print(f"  Profile:       no_a2 (A1 spatial + A3 edge path, no A2 histograms)")
    print(f"  Model variant: A1+A3 fine-tuned from lite")
    print(f"  Num classes:   {num_classes}")
    print(f"  Classes:       {', '.join(f'{k}={v}' for k, v in LABEL_MAP.items())}")
    print(f"\n  Next steps:")
    print(f"    1. Copy these files to the C++ project")
    print(f"    2. Build the C++ tensor converter (Phase 2)")
    print(f"    3. Integrate with ONNX Runtime C++ API (Phase 3)")


if __name__ == "__main__":
    main()
