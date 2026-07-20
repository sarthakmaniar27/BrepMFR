# -*- coding: utf-8 -*-
"""
BrepMFR -> ONNX Export & Validation
===================================

Loads a Stage-1 BrepSeg Lightning checkpoint, wraps it in a flat-tensor
inference interface (no Python dicts, no PyG), exports to ONNX, and
validates numerical parity between PyTorch and ONNX Runtime outputs.

This script specifically targets the "lite" inference profile used for
the Thread + Text model (num_classes=3), which omits A1, A2, and A3 tensors
(spatial_pos, d2_distance, angle_distance, edge_path).

Usage (from repo root):
    conda activate brep_mfr_pyg
    python standalone_scripts/model_conversion_onnx.py --checkpoint results/model_to_onnx/last.ckpt

Outputs:
    standalone_scripts/exported/brepmfr_lite.onnx
    standalone_scripts/exported/model_config.json
    standalone_scripts/exported/label_map.json
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
from typing import Any, Dict, Tuple

# Bootstrap repo root so models.* and data.* resolve
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
    _repo = str(_THIS.parent.parent)
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.brepseg_model import BrepSeg

# -------------------------------------------------------------------
# Label map for Thread + Text (3 classes)
# -------------------------------------------------------------------
LABEL_MAP = {
    0: "Stock",
    1: "Thread",
    2: "Text"
}

# -------------------------------------------------------------------
# 1. ONNX WRAPPER
# -------------------------------------------------------------------

class BrepMFRONNXWrapper(nn.Module):
    """Flat-tensor wrapper for ONNX export (lite profile).

    Accepts explicit tensor arguments (no Python dicts, no PyG objects).
    Hardcodes the `lite` inference path (no A1, A2, or A3 tensors).
    Single-graph inference (batch_size = 1).
    """

    def __init__(self, brepseg: BrepSeg):
        super().__init__()
        self.brep_encoder = brepseg.brep_encoder
        self.attention_layer = brepseg.attention
        self.classifier = brepseg.classifier

    def forward(
        self,
        # Node (face) tensors
        node_data: torch.Tensor,        # [total_nodes, U, V, C]     float32
        face_area: torch.Tensor,        # [total_nodes]              float32
        face_type: torch.Tensor,        # [total_nodes]              int64
        face_loop: torch.Tensor,        # [total_nodes]              int64
        in_degree: torch.Tensor,        # [total_nodes]              int64
        # Edge tensors
        edge_data: torch.Tensor,        # [total_edges, L, C]        float32
        edge_type: torch.Tensor,        # [total_edges]              int64
        edge_len: torch.Tensor,         # [total_edges]              float32
        edge_ang: torch.Tensor,         # [total_edges]              float32
        edge_conv: torch.Tensor,        # [total_edges]              int64
        edge_index: torch.Tensor,       # [2, total_edges]           int64
        # Graph structure (batch_size=1)
        attn_bias: torch.Tensor,        # [1, N+1, N+1]             float32
        padding_mask: torch.Tensor,     # [1, N]                    bool
        edge_padding_mask: torch.Tensor,# [1, max_edges]            bool
    ) -> torch.Tensor:
        """Returns logits [total_nodes, num_classes] (before softmax)."""

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
            "padding_mask": padding_mask,
            "edge_padding_mask": edge_padding_mask,
            # 'lite' profile omits these tensors
            "spatial_pos": None,
            "edge_path": None,
            "d2_distance": None,
            "angle_distance": None,
        }

        # Encoder
        node_emb, graph_emb = self.brep_encoder(batch_data, last_state_only=True)

        # node_emb is a list; [0] has shape [N+1, B, D]
        node_emb = node_emb[0].permute(1, 0, 2)   # -> [B, N+1, D]
        node_emb = node_emb[:, 1:, :]              # -> [B, N, D] (strip virtual node)

        # Extract non-padded nodes
        node_pos = torch.where(~padding_mask)
        node_z = node_emb[node_pos]                    # [total_real_nodes, D]

        num_nodes_per_graph = (~padding_mask).sum(dim=-1)  # [B]
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0)

        # Attention fusion + classifier
        z = self.attention_layer([node_z, graph_z])
        logits = self.classifier(z)  # classifier already applies softmax

        return logits


# -------------------------------------------------------------------
# 2. CHECKPOINT LOADING
# -------------------------------------------------------------------

def _namespace_from_ckpt(ckpt: Dict[str, Any]) -> Namespace:
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
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise ValueError("Not a Lightning checkpoint (no state_dict key)")

    args = _namespace_from_ckpt(ckpt)

    # Disable training-only features
    args.pre_train = None
    args.warmup_freeze_epochs = 0
    args.max_nodes_for_a3 = None

    cw = getattr(args, "class_weights_path", None)
    if cw and not pathlib.Path(cw).expanduser().is_file():
        args.class_weights_path = None

    model = BrepSeg(args)
    state = ckpt["state_dict"]

    seg_sd = {
        k: v for k, v in state.items()
        if k.startswith(("brep_encoder.", "attention.", "classifier."))
    }
    if "class_weights" in state:
        seg_sd["class_weights"] = state["class_weights"]

    if not seg_sd:
        raise ValueError("No brep_encoder/attention/classifier weights found")

    incompatible = model.load_state_dict(seg_sd, strict=False)
    ignorable = {"class_weights"}
    bad_missing = [k for k in incompatible.missing_keys if not k.startswith("_") and k not in ignorable]
    if bad_missing:
        print(f"[WARN] Missing keys: {bad_missing[:10]}")

    if incompatible.unexpected_keys:
        print(f"[INFO] Unexpected keys (ignored): {incompatible.unexpected_keys[:5]}")

    model.eval()
    model.to(device)

    num_classes = int(getattr(args, "num_classes", 3))
    print(f"[INFO] Model loaded: num_classes={num_classes}, dim_node={getattr(args, 'dim_node', '?')}")

    return model, args


# -------------------------------------------------------------------
# 3. DUMMY INPUT GENERATION
# -------------------------------------------------------------------

def create_dummy_inputs(N: int = 50, E: int = 120, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Create dummy inputs matching the lite profile."""
    # Node features
    node_data = torch.randn(N, 5, 5, 7, device=device)
    face_area = torch.rand(N, device=device) * 100.0
    face_type = torch.randint(0, 7, (N,), device=device, dtype=torch.int64)
    face_loop = torch.randint(0, 10, (N,), device=device, dtype=torch.int64)
    in_degree = torch.randint(1, 20, (N,), device=device, dtype=torch.int64)

    # Edge features
    src_indices = torch.randint(0, N, (E,), device=device, dtype=torch.int64)
    dst_indices = torch.randint(0, N, (E,), device=device, dtype=torch.int64)
    edge_index = torch.stack([src_indices, dst_indices], dim=0)

    edge_data = torch.randn(E, 5, 7, device=device)
    edge_type = torch.randint(0, 5, (E,), device=device, dtype=torch.int64)
    edge_len = torch.rand(E, device=device) * 50.0
    edge_ang = (torch.rand(E, device=device) * 2 - 1) * 3.14159
    edge_conv = torch.randint(0, 3, (E,), device=device, dtype=torch.int64)

    # Graph structure
    attn_bias = torch.zeros(1, N + 1, N + 1, device=device)
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
        "padding_mask": padding_mask,
        "edge_padding_mask": edge_padding_mask,
    }


# -------------------------------------------------------------------
# 4. PYTORCH SANITY CHECKS
# -------------------------------------------------------------------

def run_pytorch_checks(wrapper: BrepMFRONNXWrapper, num_classes: int, device: str = "cpu"):
    print("\n" + "=" * 60)
    print("STEP 1: PyTorch Sanity Checks")
    print("=" * 60)

    test_sizes = [(20, 48), (50, 120), (100, 250)]
    all_passed = True

    for N, E in test_sizes:
        print(f"\n[INFO] Testing N={N} faces, E={E} edges...")
        inputs = create_dummy_inputs(N=N, E=E, device=device)

        with torch.no_grad():
            try:
                logits = wrapper(**inputs)
            except Exception as e:
                print(f"[FAIL] Exception during forward pass: {e}")
                all_passed = False
                continue

        expected_shape = (N, num_classes)
        if logits.shape != expected_shape:
            print(f"[FAIL] Shape mismatch: got {logits.shape}, expected {expected_shape}")
            all_passed = False
            continue

        prob_sum = logits.sum(dim=-1)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-3):
            print(f"[WARN] Output may not be probabilities (sum range: {prob_sum.min():.4f} - {prob_sum.max():.4f})")
        else:
            print(f"[PASS] Output shape correct: {logits.shape}")
            print(f"[PASS] Probabilities sum to 1.0")

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[FAIL] Output contains NaN or Inf!")
            all_passed = False
            continue

        preds = logits.argmax(dim=-1)
        unique_preds = preds.unique()
        print(f"[PASS] No NaN/Inf in output")
        print(f"[INFO] Predicted {len(unique_preds)} unique classes: {unique_preds.tolist()[:10]}")

    print(f"\n[INFO] Testing determinism...")
    inputs = create_dummy_inputs(N=30, E=70, device=device)
    with torch.no_grad():
        out1 = wrapper(**inputs).clone()
        out2 = wrapper(**inputs).clone()
    if torch.allclose(out1, out2, atol=1e-7):
        print(f"[PASS] Deterministic: two runs produce identical output")
    else:
        max_diff = (out1 - out2).abs().max().item()
        print(f"[WARN] Non-deterministic: max diff = {max_diff:.2e}")

    return all_passed


# -------------------------------------------------------------------
# 5. ONNX EXPORT
# -------------------------------------------------------------------

def export_to_onnx(wrapper: BrepMFRONNXWrapper, output_path: Path, N: int = 50, E: int = 120, opset_version: int = 17, device: str = "cpu") -> Path:
    print("\n" + "=" * 60)
    print("STEP 2: ONNX Export")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = create_dummy_inputs(N=N, E=E, device=device)

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
        inputs["padding_mask"],
        inputs["edge_padding_mask"],
    )

    input_names = [
        "node_data", "face_area", "face_type", "face_loop", "in_degree",
        "edge_data", "edge_type", "edge_len", "edge_ang", "edge_conv", "edge_index",
        "attn_bias", "padding_mask", "edge_padding_mask"
    ]

    output_names = ["logits"]

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
        "padding_mask":      {1: "num_nodes"},
        "edge_padding_mask": {1: "max_edges"},
        "logits":            {0: "total_nodes"},
    }

    print(f"[INFO] Exporting with dummy graph: N={N}, E={E}")
    print(f"[INFO] ONNX opset version: {opset_version}")
    print(f"[INFO] Output: {output_path}")

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
    print(f"[PASS] Export completed in {elapsed:.1f}s")

    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"[PASS] ONNX model passes onnx.checker validation")
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[INFO] File size: {file_size_mb:.1f} MB")
    except ImportError:
        print(f"[WARN] `onnx` package not installed - skipping validation")
    except Exception as e:
        print(f"[FAIL] ONNX validation failed: {e}")
        raise

    return output_path


# -------------------------------------------------------------------
# 6. ONNX RUNTIME VALIDATION
# -------------------------------------------------------------------

def validate_onnx_vs_pytorch(wrapper: BrepMFRONNXWrapper, onnx_path: Path, device: str = "cpu", num_tests: int = 5, atol: float = 1e-4) -> bool:
    print("\n" + "=" * 60)
    print("STEP 3: ONNX Runtime vs PyTorch Validation")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("[FAIL] onnxruntime not installed.")
        return False

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]
    if device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)
    ort_inputs = {inp.name: inp.shape for inp in session.get_inputs()}

    test_configs = [
        (20, 48, "small"), (50, 120, "medium"), (100, 250, "large"),
        (200, 500, "xl"), (30, 70, "odd")
    ][:num_tests]

    all_passed = True
    for N, E, label in test_configs:
        print(f"\n[INFO] Test '{label}': N={N}, E={E}")
        inputs = create_dummy_inputs(N=N, E=E, device="cpu")

        with torch.no_grad():
            pt_inputs = {k: v.to(device) for k, v in inputs.items()}
            pt_logits = wrapper(**pt_inputs).cpu().numpy()

        ort_feed = {name: inputs[name].numpy() for name in ort_inputs}
        ort_logits = session.run(["logits"], ort_feed)[0]

        if pt_logits.shape != ort_logits.shape:
            print(f"  [FAIL] Shape mismatch: PyTorch={pt_logits.shape}, ORT={ort_logits.shape}")
            all_passed = False
            continue

        abs_diff = np.abs(pt_logits - ort_logits)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()

        pt_preds = pt_logits.argmax(axis=-1)
        ort_preds = ort_logits.argmax(axis=-1)
        pred_match = (pt_preds == ort_preds).sum()
        pred_total = len(pt_preds)

        if max_diff <= atol:
            print(f"  [PASS] max_diff={max_diff:.2e}  labels={pred_match}/{pred_total}")
        elif pred_match == pred_total and max_diff <= 1e-2:
            print(f"  [WARN] max_diff={max_diff:.2e} (> atol={atol}) but labels match")
        else:
            print(f"  [FAIL] max_diff={max_diff:.2e}  labels={pred_match}/{pred_total}")
            all_passed = False

    print(f"\n[INFO] Latency comparison (N=50, E=120, 10 runs):")
    inputs = create_dummy_inputs(N=50, E=120, device="cpu")
    pt_inputs_dev = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        for _ in range(3): wrapper(**pt_inputs_dev)
        t0 = time.perf_counter()
        for _ in range(10): wrapper(**pt_inputs_dev)
        pt_time = (time.perf_counter() - t0) / 10 * 1000

    ort_feed = {name: inputs[name].numpy() for name in ort_inputs}
    for _ in range(3): session.run(["logits"], ort_feed)
    t0 = time.perf_counter()
    for _ in range(10): session.run(["logits"], ort_feed)
    ort_time = (time.perf_counter() - t0) / 10 * 1000

    print(f"  PyTorch:      {pt_time:.1f} ms/inference")
    print(f"  ONNX Runtime: {ort_time:.1f} ms/inference")

    return all_passed


# -------------------------------------------------------------------
# 7. CONFIG EXPORT
# -------------------------------------------------------------------

def export_configs(output_dir: Path, args: Namespace, num_classes: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    # After lite export, unused edge/A3 inputs are optimized out of the ONNX graph.
    model_config = {
        "num_classes": num_classes,
        "uv_grid_u": 5,
        "uv_grid_v": 5,
        "uv_channels": 7,
        "edge_grid_size": 5,
        "edge_channels": 7,
        "inference_profile": "lite",
        "output_name": "logits",
        "output_semantics": "softmax_probabilities",
        "model_hyperparams": {
            "dim_node": getattr(args, "dim_node", None),
            "d_model": getattr(args, "d_model", None),
            "n_heads": getattr(args, "n_heads", None),
            "n_layers_encode": getattr(args, "n_layers_encode", None),
        },
        "onnx_inputs": {
            "node_data": {"shape": ["total_nodes", 5, 5, 7], "dtype": "float32"},
            "face_area": {"shape": ["total_nodes"], "dtype": "float32"},
            "face_type": {"shape": ["total_nodes"], "dtype": "int64"},
            "face_loop": {"shape": ["total_nodes"], "dtype": "int64"},
            "in_degree": {"shape": ["total_nodes"], "dtype": "int64"},
            "attn_bias": {"shape": [1, "num_nodes_plus_one", "num_nodes_plus_one"], "dtype": "float32", "note": "zeros for lite"},
            "padding_mask": {"shape": [1, "num_nodes"], "dtype": "bool", "note": "False=real face"},
        },
        "onnx_inputs_optimized_out": [
            "edge_data",
            "edge_type",
            "edge_len",
            "edge_ang",
            "edge_conv",
            "edge_index",
            "edge_padding_mask",
        ],
        "input_dtype_policy": {
            "float_inputs": "float32",
            "index_inputs": "int64",
            "note": "Cast face_type/face_loop/in_degree to int64 before ORT; .pt files may store int32",
        },
    }

    config_path = output_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"[INFO] Saved config: {config_path}")

    label_path = output_dir / "label_map.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in LABEL_MAP.items()}, f, indent=2)
    print(f"[INFO] Saved labels: {label_path}")


# -------------------------------------------------------------------
# 8. MAIN
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export BrepMFR to ONNX (lite profile)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for exported files")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--skip_validation", action="store_true", help="Skip ONNX Runtime validation")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"[FAIL] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else _THIS.parent / "exported"
    onnx_path = output_dir / "brepmfr_lite.onnx"
    export_device = "cpu"

    model, model_args = load_brepseg(ckpt_path, device=export_device)
    num_classes = int(getattr(model_args, "num_classes", 3))

    wrapper = BrepMFRONNXWrapper(model)
    wrapper.eval()
    wrapper.to(export_device)

    checks_passed = run_pytorch_checks(wrapper, num_classes, device=export_device)
    if not checks_passed:
        print("\n[FAIL] PyTorch sanity checks failed. Fix issues before exporting.")
        sys.exit(1)

    export_to_onnx(wrapper, onnx_path, opset_version=args.opset, device=export_device)

    if not args.skip_validation:
        if validate_onnx_vs_pytorch(wrapper, onnx_path, device=export_device):
            print("\n[PASS] ALL CHECKS PASSED - ONNX model is ready for C++ deployment")
        else:
            print("\n[WARN] VALIDATION ISSUES DETECTED - review output above")
    
    print("\n" + "=" * 60)
    print("STEP 4: Export Configuration Files")
    print("=" * 60)
    export_configs(output_dir, model_args, num_classes)

if __name__ == "__main__":
    main()
