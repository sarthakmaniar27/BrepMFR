#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inference on manual PyG ``.pt`` graphs (three dataset layouts under ``Y:\\new_dataset\\test``).

Loads a Stage‑1 ``BrepSeg`` or Stage‑2 ``DomainAdapt`` checkpoint via partial ``state_dict``
(encoder / attention / classifier only). Writes one CSV per input graph under each dataset's
``inference`` folder.

Usage (PowerShell, from repo root):

  conda activate brep_mfr_pyg
  python scripts/inference/run_pyg_inference.py --checkpoint results/.../best.ckpt

Optional overrides::

  python scripts/inference/run_pyg_inference.py --checkpoint model.ckpt --device cpu --batch_size 4 ^
    --dataset_root Z:\\new_dataset\\test --only abc

After CSVs exist, export predicted-label UV JSON (``uv_json_pred``) with
``python scripts/inference/export_uv_json_pred.py --dataset_root ...`` (reads ``inference/*.csv``).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_bf = Path(__file__).resolve()
for _ancestor in _bf.parents:
    _bst = _ancestor / "bootstrap_path.py"
    if _bst.is_file():
        _spec = importlib.util.spec_from_file_location("__brepmfr_bootstrap", _bst)
        _bm = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_bm)
        _bm.setup(str(_bf))
        break
else:
    raise RuntimeError(
        "bootstrap_path.py not found; keep scripts inside the BrepMFR_PyG repository."
    )

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.collator import collator  # noqa: E402
from models.brepseg_model import BrepSeg  # noqa: E402

FACE_LABEL_NAME = {
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
    24: "Fillet",
}


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
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.inference_mode()
def predict_probs_per_node(
    model: BrepSeg,
    batch: Dict,
    num_classes: int,
) -> torch.Tensor:
    """[total_nodes, C] softmax probabilities (same path as validation forward)."""
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
    if (node_seg < 0).any() or (node_seg > 1.01).any():
        node_seg = F.softmax(node_seg, dim=-1)
    else:
        node_seg = node_seg.clamp(min=1e-12)
        node_seg = node_seg / node_seg.sum(dim=-1, keepdim=True)
    return node_seg


def _num_faces(pyg: Any) -> int:
    return int(pyg.node_data.shape[0])


def _labels_from_tensor(
    lf: Optional[torch.Tensor],
    num_faces: int,
    num_classes: int,
) -> Optional[np.ndarray]:
    if lf is None:
        return None
    arr = lf.detach().cpu().numpy().flatten()
    if arr.size != num_faces:
        return None
    if not np.isfinite(arr).all():
        return None
    if np.any(arr < 0) or np.any(arr >= num_classes):
        return None
    return arr.astype(np.int64)


def _load_sidecar_labels(
    graph_parent: pathlib.Path,
    stem: str,
    num_faces: int,
    num_classes: int,
) -> Optional[np.ndarray]:
    for sub in ("label", "labels"):
        p = graph_parent / sub / f"{stem}.json"
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            labels = data.get("labels")
            if labels is None or len(labels) != num_faces:
                return None
            arr = np.asarray(labels, dtype=np.int64).flatten()
            if arr.size != num_faces:
                return None
            if np.any(arr < 0) or np.any(arr >= num_classes):
                return None
            return arr
        except (json.JSONDecodeError, OSError, TypeError):
            return None
    return None


def _resolve_gt(
    pyg: Any,
    stem: str,
    graph_parent: pathlib.Path,
    num_classes: int,
) -> Optional[np.ndarray]:
    nf = _num_faces(pyg)
    lf = getattr(pyg, "label_feature", None)
    gt = _labels_from_tensor(lf, nf, num_classes)
    if gt is not None:
        return gt
    return _load_sidecar_labels(graph_parent, stem, nf, num_classes)


def _class_name(cid: int, num_classes: int) -> str:
    if cid in FACE_LABEL_NAME and cid < num_classes:
        return FACE_LABEL_NAME[int(cid)]
    return f"class_{cid}"


def write_graph_csv(
    out_csv: pathlib.Path,
    probs: np.ndarray,
    num_classes: int,
    gt: Optional[np.ndarray],
) -> None:
    """probs [N, C]; optional gt [N]."""
    n = probs.shape[0]
    preds = probs.argmax(axis=1).astype(np.int64)
    p_hat = probs[np.arange(n), preds].astype(float)

    with_gt = gt is not None

    fields = [
        "face_index",
        "predicted_class",
        "predicted_class_name",
        "predicted_probability",
    ]
    if with_gt:
        fields += ["ground_truth_class", "ground_truth_class_name", "correct_top1"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            row = {
                "face_index": i,
                "predicted_class": int(preds[i]),
                "predicted_class_name": _class_name(int(preds[i]), num_classes),
                "predicted_probability": f"{float(p_hat[i]):.10g}",
            }
            if with_gt:
                yi = int(gt[i])  # type: ignore[index]
                row["ground_truth_class"] = yi
                row["ground_truth_class_name"] = _class_name(yi, num_classes)
                row["correct_top1"] = int(preds[i] == yi)
            w.writerow(row)


def inference_one_dataset(
    name: str,
    pyg_dir: pathlib.Path,
    inference_out: pathlib.Path,
    device: torch.device,
    batch_size: int,
    multi_hop_max_dist: int,
    spatial_pos_max: int,
    model: BrepSeg,
    num_classes: int,
    max_files: Optional[int],
) -> Tuple[int, int, int, Optional[float]]:
    """
    Returns (pyg_candidates, graphs_written_ok, total_faces, mean_top1_or_None).
    """
    files = sorted(pyg_dir.glob("*.pt"))
    if max_files is not None:
        files = files[: max_files]

    graph_parent = pyg_dir.parent
    inference_out.mkdir(parents=True, exist_ok=True)

    graphs_written = 0
    total_faces = 0
    correct = 0
    labelled_faces = 0

    batch_paths: List[pathlib.Path] = []
    batch_pygs: List[Any] = []
    batch_gt: List[Optional[np.ndarray]] = []

    def flush_chunk() -> None:
        nonlocal batch_paths, batch_pygs, batch_gt, total_faces, correct, labelled_faces
        nonlocal graphs_written
        if not batch_pygs:
            return
        b = collator(batch_pygs, multi_hop_max_dist, spatial_pos_max)
        b = _batch_to_device(b, device)
        probs_t = predict_probs_per_node(model, b, num_classes)
        probs = probs_t.float().cpu().numpy()

        offset = 0
        for k, pt_path in enumerate(batch_paths):
            n = _num_faces(batch_pygs[k])
            pslice = probs[offset : offset + n]
            offset += n
            gt_arr = batch_gt[k]
            if gt_arr is not None:
                labelled_faces += n
                correct += int((pslice.argmax(axis=1) == gt_arr).sum())
            out_csv = inference_out / f"{pt_path.stem}.csv"
            write_graph_csv(out_csv, pslice, num_classes, gt_arr)
            total_faces += n
            graphs_written += 1
        batch_paths = []
        batch_pygs = []
        batch_gt = []

    for pt_path in tqdm(files, desc=f"{name}"):
        try:
            pyg = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[skip] load failed {pt_path}: {e}")
            continue
        if not hasattr(pyg, "edge_index") or not hasattr(pyg, "node_data"):
            print(f"[skip] not BrepMFR PyG layout: {pt_path}")
            continue

        gt = _resolve_gt(pyg, pt_path.stem, graph_parent, num_classes)

        batch_paths.append(pt_path)
        batch_pygs.append(pyg)
        batch_gt.append(gt)

        if len(batch_pygs) >= batch_size:
            flush_chunk()

    flush_chunk()

    mean_acc = (correct / labelled_faces) if labelled_faces > 0 else None
    return len(files), graphs_written, total_faces, mean_acc


def main() -> None:
    default_root = pathlib.Path(r"Y:\new_dataset\test")

    ap = argparse.ArgumentParser(
        description="Batch PyG inference: cadsynth / mfcadpp / abc trees under dataset_root.",
    )
    ap.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="BrepSeg or DomainAdapt Lightning .ckpt (segmentation head only).",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=1, help="Graphs per collator batch.")
    ap.add_argument(
        "--dataset_root",
        type=pathlib.Path,
        default=default_root,
        help=f"Folder containing cadsynth, mfcadpp, abc (default: {default_root}).",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset: abc,mfcadpp,cadsynth",
    )
    ap.add_argument(
        "--multi_hop_max_dist",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--spatial_pos_max",
        type=int,
        default=32,
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Max .pt files per dataset (debug).",
    )

    args = ap.parse_args()
    sets = {"cadsynth", "mfcadpp", "abc"}
    if args.only:
        sel = {x.strip().lower() for x in args.only.split(",") if x.strip()}
        unknown = sel - sets
        if unknown:
            raise SystemExit(f"Unknown --only entries: {unknown}")
        run_sets = [(n, args.dataset_root / n) for n in ("cadsynth", "mfcadpp", "abc") if n in sel]
    else:
        run_sets = [(n, args.dataset_root / n) for n in ("cadsynth", "mfcadpp", "abc")]

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.", flush=True)
        device = torch.device("cpu")

    ckpt_path = args.checkpoint.resolve()
    model, num_classes = load_brepseg_for_inference(ckpt_path, device)
    print(f"Loaded segmentation head num_classes={num_classes} from {ckpt_path}", flush=True)

    nc_names = len(FACE_LABEL_NAME)
    if num_classes != nc_names:
        print(
            f"Warning: ckpt num_classes={num_classes} vs built-in CADSynth "
            f"name table length {nc_names}; unknown ids use literal class_<id>",
            flush=True,
        )

    for name, base in run_sets:
        pyg_dir = base / "graph" / "pyg"
        infer_dir = base / "inference"
        if not pyg_dir.is_dir():
            print(f"[skip] {name}: missing pyg dir {pyg_dir}")
            continue
        n_found, n_graph, n_face, mean_acc = inference_one_dataset(
            name=name,
            pyg_dir=pyg_dir,
            inference_out=infer_dir,
            device=device,
            batch_size=max(1, int(args.batch_size)),
            multi_hop_max_dist=args.multi_hop_max_dist,
            spatial_pos_max=args.spatial_pos_max,
            model=model,
            num_classes=num_classes,
            max_files=args.max_files,
        )
        extra = ""
        if mean_acc is not None:
            extra = f"  top1 (faces with GT)={100.0 * mean_acc:.2f}%"
        print(
            f"{name}: pt_candidates={n_found} graphs_ok={n_graph} "
            f"total_faces_written={n_face}{extra}",
            flush=True,
        )


if __name__ == "__main__":
    main()
