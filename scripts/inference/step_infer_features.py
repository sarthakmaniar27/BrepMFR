#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer machining feature class (25-way) per B-rep face from a STEP file.

Pipeline
  STEP  --(occwl,``occwl_to_brep_tensors``)-->  PyG ``Data``  --(collator + BrepSeg)-->  softmax

  ``--graph *.bin`` still uses DGL once to load legacy bins; ``*.pt`` and STEP do not need DGL.

Checkpoints: Stage-1 ``BrepSeg`` *.ckpt or Stage-2 ``DomainAdapt`` *.ckpt (uses brep_encoder +
attention + classifier weights only).

Examples
  # From STEP (occwl + PyG + torch; no DGL)
  python scripts/inference/step_infer_features.py model.ckpt part.step

  # From an existing graph (.pt, or .bin if you have dgl)
  python scripts/inference/step_infer_features.py model.ckpt --graph part.pt

  # All 25 probabilities per face
  python scripts/inference/step_infer_features.py model.ckpt part.step --all-proclasses
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_REPO_ROOT = None
_bf = Path(__file__).resolve()
for _ancestor in _bf.parents:
    _bst = _ancestor / "bootstrap_path.py"
    if _bst.is_file():
        _spec = importlib.util.spec_from_file_location("__brepmfr_bootstrap", _bst)
        _bm = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_bm)
        _REPO_ROOT = _bm.setup(str(_bf))
        break
if _REPO_ROOT is None:
    raise RuntimeError(
        "bootstrap_path.py not found; keep scripts inside the BrepMFR_PyG repository."
    )
_pipeline_dir = _REPO_ROOT / "tools" / "pipeline"
if _pipeline_dir.is_dir() and str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

import numpy as np
import torch
import torch.nn.functional as F

from data.collator import collator  # noqa: E402
from models.brepseg_model import BrepSeg  # noqa: E402
from occwl_to_brep_tensors import convert_stp_path_to_pyg  # noqa: E402

# Authoritative 25-class names (CADSynth)
CLASS_NAMES = {
    0: "Stock",
    1: "Rectangular through slot",
    2: "Triangular through slot",
    3: "Rectangular passage",
    4: "Triangular passage",
    5: "6-sided passage",
    6: "Rectangular through step",
    7: "2-sided through step",
    8: "Slanted through step",
    9: "Rectangular blind step",
    10: "Triangular blind step",
    11: "Rectangular blind slot",
    12: "Rectangular pocket",
    13: "Triangular pocket",
    14: "6-sided pocket",
    15: "Chamfer",
    16: "Circular through slot",
    17: "Through hole",
    18: "Circular blind step",
    19: "Horizontal circular end blind slot",
    20: "Vertical circular end blind slot",
    21: "Circular end pocket",
    22: "O-ring",
    23: "Blind hole",
    24: "Round",
}


def _load_pyg_from_disk(path: pathlib.Path):
    suffix = path.suffix.lower()
    if suffix == ".bin":
        # Lazy: keeps `conda activate brep_mfr_pyg` free of optional `dgl` unless `.bin` is used.
        from data.dgl_bin_to_pyg import bin_to_pyg

        return bin_to_pyg(path)
    if suffix == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(obj, "edge_index") and hasattr(obj, "node_data"):
            return obj
        raise ValueError(f"Unrecognized .pt layout: {path}")
    raise ValueError(f"Expected .bin or .pt, got {path}")


def _namespace_from_ckpt(ckpt: Dict[str, Any]) -> Namespace:
    h = ckpt.get("hyper_parameters")
    if not h:
        raise ValueError("Checkpoint missing hyper_parameters")
    if "args" in h:
        a = h["args"]
        if isinstance(a, Namespace):
            d = vars(a).copy()
        elif isinstance(a, dict):
            d = dict(a)
        else:
            d = vars(a)
        return Namespace(**d)
    return Namespace(**{k: v for k, v in h.items() if k != "args"})


def load_brepseg_for_inference(
    ckpt_path: pathlib.Path,
    device: torch.device,
) -> Tuple[BrepSeg, int]:
    """
    Build BrepSeg and load segmentation weights from either BrepSeg or DomainAdapt ckpt.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError("Not a Lightning checkpoint (no state_dict)")

    args = _namespace_from_ckpt(ckpt)
    args.pre_train = None
    args.warmup_freeze_epochs = 0
    cw = getattr(args, "class_weights_path", None)
    if cw and not pathlib.Path(cw).is_file():
        args.class_weights_path = None

    num_classes = int(getattr(args, "num_classes", 25))
    model = BrepSeg(args)
    state = ckpt["state_dict"]
    seg_sd = {
        k: v
        for k, v in state.items()
        if k.startswith(("brep_encoder.", "attention.", "classifier."))
    }
    if not seg_sd:
        raise ValueError("No brep_encoder / attention / classifier weights in checkpoint")

    # Class weights are CE loss only; checkpoints sliced to encoder/attention/classifier omit this buffer.
    ignorable_missing = frozenset({"class_weights"})
    if "class_weights" in state and "class_weights" not in seg_sd:
        seg_sd = {**seg_sd, "class_weights": state["class_weights"]}

    incompatible = model.load_state_dict(seg_sd, strict=False)
    if incompatible.missing_keys:
        bad = [
            k
            for k in incompatible.missing_keys
            if not k.startswith("_") and k not in ignorable_missing
        ]
        if bad:
            raise RuntimeError(f"Missing required keys: {bad[:8]}...")
    model.eval()
    model.to(device)
    return model, num_classes


def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.inference_mode()
def predict_probs(
    model: BrepSeg,
    batch: Dict,
    num_classes: int,
) -> torch.Tensor:
    """Returns [N_faces, C] **probabilities** (classifier already applies softmax)."""
    node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True)
    node_emb = node_emb[0].permute(1, 0, 2)
    node_emb = node_emb[:, 1:, :]
    padding_mask = batch["padding_mask"]
    node_pos = torch.where(padding_mask == False)  # noqa: E712
    node_z = node_emb[node_pos]
    num_nodes_per_graph = (~padding_mask).sum(dim=-1)
    graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
    z = model.attention([node_z, graph_z])
    node_seg = model.classifier(z)
    # NonLinearClassifier uses softmax; re-normalize for numeric safety
    if (node_seg < 0).any() or (node_seg > 1.01).any():
        node_seg = F.softmax(node_seg, dim=-1)
    else:
        node_seg = node_seg.clamp(min=1e-12)
        node_seg = node_seg / node_seg.sum(dim=-1, keepdim=True)
    return node_seg


def main() -> None:
    ap = argparse.ArgumentParser(description="STEP / graph → per-face machining feature probabilities")
    ap.add_argument("checkpoint", type=pathlib.Path, help="BrepSeg or DomainAdapt Lightning .ckpt")
    ap.add_argument("step", type=pathlib.Path, nargs="?", help="Input .step / .stp (if not using --graph)")
    ap.add_argument(
        "--graph",
        type=pathlib.Path,
        default=None,
        help="Skip STEP: load this .pt (PyG) or .bin (DGL; requires dgl)",
    )
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument(
        "--all-proclasses",
        action="store_true",
        help="Print all num_classes probabilities per face (wide lines)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=5,
        help="When not --all-proclasses, show this many classes per line",
    )
    args = ap.parse_args()

    if args.graph is None and args.step is None:
        ap.error("Provide a STEP path or --graph")
    if args.graph is not None and args.step is not None:
        ap.error("Use either a STEP path or --graph, not both")

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA requested but not available; using CPU.", file=sys.stderr)

    if args.graph is not None:
        pyg = _load_pyg_from_disk(args.graph.resolve())
    else:
        step_path = args.step.resolve()
        if not step_path.is_file():
            raise FileNotFoundError(step_path)

        print(f"STEP → PyG via occwl (no DGL): {step_path}", file=sys.stderr)
        pyg = convert_stp_path_to_pyg(step_path)
        if pyg is None:
            raise RuntimeError(f"No solids read from STEP: {step_path}")

    model, num_classes = load_brepseg_for_inference(args.checkpoint.resolve(), device)
    if num_classes != len(CLASS_NAMES):
        print(
            f"Warning: model num_classes={num_classes} vs built-in names {len(CLASS_NAMES)}",
            file=sys.stderr,
        )

    batch = collator([pyg], multi_hop_max_dist=16, spatial_pos_max=32)
    batch = _batch_to_device(batch, device)

    probs = predict_probs(model, batch, num_classes)
    probs_np = probs.float().cpu().numpy()
    preds = probs_np.argmax(axis=1)
    labels = batch["label_feature"].cpu().numpy()

    n_face = probs_np.shape[0]
    print(
        f"faces={n_face}  checkpoint={args.checkpoint}  device={device}  num_classes={num_classes}",
        flush=True,
    )

    topk = max(1, min(args.topk, num_classes))
    for i in range(n_face):
        gt = int(labels[i]) if i < len(labels) else -1
        gt_s = (
            f"  gt_label={gt} ({CLASS_NAMES.get(gt, '?')})"
            if 0 <= gt < num_classes
            else ("  gt_label=(n/a)" if gt < 0 else f"  gt_label={gt}")
        )
        p = probs_np[i]
        pred = int(p.argmax())
        if args.all_proclasses:
            line_extra = " ".join(f"{j}:{p[j]:.4f}" for j in range(len(p)))
        else:
            order = np.argsort(-p)[:topk]
            line_extra = ", ".join(
                f"{int(j)}:{CLASS_NAMES.get(int(j), str(j))}={p[j]:.4f}" for j in order
            )

        print(
            f"face[{i:4d}]  pred={pred:2d} ({CLASS_NAMES.get(pred, '?')})  "
            f"p(pred)={p[pred]:.6f}{gt_s}",
            flush=True,
        )
        print(f"           probs: {line_extra}", flush=True)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
