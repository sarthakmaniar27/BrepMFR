# -*- coding: utf-8 -*-
"""Real-graph PyTorch vs ONNX parity for lite Thread+Text model."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

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

from data.collator import collator

# Same folder as this script (not installed as a package)
sys.path.insert(0, str(_THIS.parent))
from model_conversion_onnx import BrepMFRONNXWrapper, load_brepseg  # noqa: E402

LITE_ROOT = Path(r"Z:\thread_and_text\lite")
CKPT = Path(r"C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG\results\model_to_onnx\last.ckpt")
ONNX_PATH = _THIS.parent / "exported" / "brepmfr_lite.onnx"

INPUT_NAMES = [
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
    "padding_mask",
    "edge_padding_mask",
]

# Lite ONNX drops unused A3/edge inputs during export; feed only what ORT requires.
ONNX_EXPECTED_MIN_INPUTS = {
    "node_data",
    "face_area",
    "face_type",
    "face_loop",
    "in_degree",
    "attn_bias",
    "padding_mask",
}


def _load_pyg(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def batch_to_wrapper_inputs(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {k: batch[k] for k in INPUT_NAMES}


def _resolve_pt(pyg_root: Path, stem: str) -> Path | None:
    direct = pyg_root / f"{stem}.pt"
    if direct.is_file():
        return direct
    # One-level nesting common in exports
    for child in pyg_root.iterdir():
        if child.is_dir():
            cand = child / f"{stem}.pt"
            if cand.is_file():
                return cand
    return None


def pick_graphs(
    pyg_root: Path,
    split_file: Path,
    *,
    max_nodes: int = 400,
    max_edges: int = 1200,
    want: int = 12,
    max_probe: int = 400,
) -> List[Tuple[Path, int, int]]:
    """Resolve stems from split list (no full-tree rglob — Z: is slow)."""
    stems = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Spread probes across the split list
    if len(stems) > max_probe:
        step = max(1, len(stems) // max_probe)
        stems = stems[::step][:max_probe]

    buckets: Dict[str, List[Tuple[Path, int, int]]] = {"small": [], "med": [], "large": []}
    for stem in stems:
        if sum(len(v) for v in buckets.values()) >= want * 4:
            break
        path = _resolve_pt(pyg_root, stem)
        if path is None:
            continue
        try:
            g = _load_pyg(path)
            n = int(g.node_data.size(0))
            e = int(g.edge_data.size(0))
        except Exception:
            continue
        if n > max_nodes or e > max_edges or n < 3:
            continue
        if n <= 40:
            buckets["small"].append((path, n, e))
        elif n <= 120:
            buckets["med"].append((path, n, e))
        else:
            buckets["large"].append((path, n, e))

    picked: List[Tuple[Path, int, int]] = []
    for key in ("small", "med", "large"):
        for item in buckets[key][: max(1, want // 3)]:
            picked.append(item)
            if len(picked) >= want:
                return picked
    return picked


def main() -> int:
    print("=" * 60)
    print("REAL-GRAPH PYTORCH vs ONNX PARITY")
    print("=" * 60)

    if not CKPT.is_file():
        print(f"[FAIL] checkpoint missing: {CKPT}")
        return 1
    if not ONNX_PATH.is_file():
        print(f"[FAIL] onnx missing: {ONNX_PATH}")
        return 1

    import onnxruntime as ort

    print(f"[INFO] ckpt: {CKPT}")
    print(f"[INFO] onnx: {ONNX_PATH}")
    print(f"[INFO] lite: {LITE_ROOT}")

    model, args = load_brepseg(CKPT, device="cpu")
    num_classes = int(getattr(args, "num_classes", -1))
    print(
        f"[INFO] ckpt hyperparams: num_classes={num_classes} "
        f"dim_node={getattr(args,'dim_node',None)} d_model={getattr(args,'d_model',None)} "
        f"n_heads={getattr(args,'n_heads',None)} n_layers_encode={getattr(args,'n_layers_encode',None)}"
    )
    if num_classes != 3:
        print(f"[FAIL] expected num_classes=3, got {num_classes}")
        return 1

    wrapper = BrepMFRONNXWrapper(model)
    wrapper.eval()

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    ort_input_names = [i.name for i in session.get_inputs()]
    ort_input_set = set(ort_input_names)
    print(f"[INFO] ONNX inputs ({len(ort_input_names)}): {ort_input_names}")
    print(f"[INFO] ONNX outputs: {[o.name for o in session.get_outputs()]}")

    missing_min = sorted(ONNX_EXPECTED_MIN_INPUTS - ort_input_set)
    if missing_min:
        print(f"[FAIL] ONNX missing required lite inputs: {missing_min}")
        return 1

    dropped_edges = sorted(
        {
            "edge_data",
            "edge_type",
            "edge_len",
            "edge_ang",
            "edge_conv",
            "edge_index",
            "edge_padding_mask",
        }
        - ort_input_set
    )
    if dropped_edges:
        print(
            f"[INFO] Edge inputs optimized out of ONNX (expected for lite/A3-off): {dropped_edges}"
        )
    print("[PASS] ONNX has required lite node/attn inputs")

    # Also compare against full Lightning test_step path on same batch
    graphs = pick_graphs(LITE_ROOT / "pyg", LITE_ROOT / "test.txt", want=12)
    if len(graphs) < 3:
        print(f"[FAIL] too few graphs picked: {len(graphs)}")
        return 1
    print(f"[INFO] picked {len(graphs)} real test graphs")

    all_ok = True
    total_faces = 0
    total_match = 0
    max_abs_all = 0.0

    for idx, (path, n, e) in enumerate(graphs, 1):
        g = _load_pyg(path)
        batch = collator([g], multi_hop_max_dist=16, spatial_pos_max=32)

        # Confirm lite packing
        if batch["spatial_pos"] is not None or batch["edge_path"] is not None:
            print(f"  [FAIL] {path.name}: collator produced A1/A3 (not lite)")
            all_ok = False
            continue
        if batch["d2_distance"] is not None or batch["angle_distance"] is not None:
            print(f"  [FAIL] {path.name}: collator produced A2 (not lite)")
            all_ok = False
            continue

        inputs = batch_to_wrapper_inputs(batch)

        with torch.no_grad():
            # Lightning-equivalent path (same as validation_step)
            node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True)
            node_emb = node_emb[0].permute(1, 0, 2)[:, 1:, :]
            padding_mask = batch["padding_mask"]
            node_pos = torch.where(padding_mask == False)  # noqa: E712
            node_z = node_emb[node_pos]
            num_nodes_per_graph = (~padding_mask).sum(dim=-1)
            graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0)
            z = model.attention([node_z, graph_z])
            lightning_out = model.classifier(z)

            wrapper_out = wrapper(**inputs)

        # Softmax probs should sum ~1
        prob_sum = wrapper_out.sum(dim=-1)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-3):
            print(
                f"  [WARN] {path.name}: probs sum range "
                f"{float(prob_sum.min()):.4f}-{float(prob_sum.max()):.4f}"
            )

        lw_diff = (lightning_out - wrapper_out).abs().max().item()
        if lw_diff > 1e-6:
            print(f"  [FAIL] {path.name}: Lightning vs wrapper max_diff={lw_diff:.2e}")
            all_ok = False
            continue

        ort_feed = {}
        for name in ort_input_names:
            t = inputs[name].detach().cpu()
            if name in ("face_type", "face_loop", "in_degree", "edge_type", "edge_conv", "edge_index"):
                t = t.long()
            elif name in ("node_data", "face_area", "edge_data", "edge_len", "edge_ang", "attn_bias"):
                t = t.float()
            elif name in ("padding_mask", "edge_padding_mask"):
                t = t.bool()
            ort_feed[name] = t.numpy()

        ort_out = session.run(["logits"], ort_feed)[0]
        pt_out = wrapper_out.detach().cpu().numpy()

        if ort_out.shape != pt_out.shape:
            print(f"  [FAIL] {path.name}: shape PT={pt_out.shape} ORT={ort_out.shape}")
            all_ok = False
            continue

        abs_diff = np.abs(pt_out - ort_out)
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())
        pt_pred = pt_out.argmax(axis=-1)
        ort_pred = ort_out.argmax(axis=-1)
        match = int((pt_pred == ort_pred).sum())
        total = int(len(pt_pred))
        total_faces += total
        total_match += match
        max_abs_all = max(max_abs_all, max_diff)

        labels = batch["label_feature"].long().numpy()
        known = labels < num_classes
        if known.any():
            acc = float((pt_pred[known] == labels[known]).mean())
        else:
            acc = float("nan")

        status = "PASS" if max_diff <= 1e-4 or (match == total and max_diff <= 1e-2) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(
            f"  [{status}] {idx:02d} {path.name} N={n} E={e} "
            f"max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} "
            f"labels={match}/{total} face_acc={acc:.3f} "
            f"lw_vs_wrap={lw_diff:.1e}"
        )

    print("\n" + "=" * 60)
    print(
        f"SUMMARY: label_match={total_match}/{total_faces} "
        f"({(100.0 * total_match / max(total_faces,1)):.2f}%) "
        f"max_abs_diff_all={max_abs_all:.2e}"
    )
    if all_ok and total_match == total_faces:
        print("[PASS] Real-graph Lightning == wrapper == ONNX Runtime")
        return 0
    print("[FAIL] Parity issues detected")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
