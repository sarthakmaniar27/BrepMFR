#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export predicted-label UV JSON from PyG ``.pt`` graphs.

Standalone script — does **not** import ``extract_uv_points``. Per-face UV grids are read from
``Data.node_data`` (already stored when the ``.pt`` was built). Predictions default to
``inference/<stem>.csv`` (``predicted_class`` / ``face_index`` columns), same layout as
``run_pyg_inference.py``. JSON schema matches ``extract_uv_points`` output / ``Y:\\uv_json\\*.json``.

Faces with predicted label ``0`` (Stock) are omitted (same filter as the extractor on GT).

Fallback: pass ``--checkpoint`` to run partial ``BrepSeg`` forward when CSV is missing.

Usage::

  conda activate brep_mfr_pyg
  python scripts/inference/export_uv_json_pred.py --dataset_root Y:\\new_dataset\\test

  python scripts/inference/export_uv_json_pred.py --checkpoint results/.../best.ckpt

**Single PyG folder** (after ``run_pyg_inference`` wrote CSVs)::

  python scripts/inference/export_uv_json_pred.py ^
    --pyg_dir Y:\\new_dataset\\test\\abc\\abc_brepmfr_test_inference\\pyg_lite ^
    --inference_dir Y:\\new_dataset\\test\\abc\\abc_brepmfr_test_inference\\inference_lite ^
    --uv_json_dir Y:\\new_dataset\\test\\abc\\abc_brepmfr_test_inference\\uv_json_pred_lite
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
from typing import Any, Dict, List, Optional, Tuple

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


def tensor_to_nested_list(t: torch.Tensor) -> Any:
    if not isinstance(t, torch.Tensor):
        return t
    return t.detach().cpu().tolist()


def infer_uv_grid(uv_tensor: torch.Tensor) -> Tuple[List, Dict[str, Any]]:
    """Mirror ``extract_uv_points._infer_uv_grid`` (logic duplicated here on purpose)."""
    meta: Dict[str, Any] = {"original_shape": list(uv_tensor.shape)}
    t = uv_tensor
    if t.ndim == 3 and t.shape[0] == 5 and t.shape[1] == 5:
        meta["interpreted_as"] = "[5,5,C]"
        return tensor_to_nested_list(t), meta

    if t.ndim == 2:
        if t.shape[0] == 25:
            c = t.shape[1]
            meta["interpreted_as"] = "[25,C] -> [5,5,C]"
            return tensor_to_nested_list(t.reshape(5, 5, c)), meta

        if t.shape[1] == 25:
            c = t.shape[0]
            meta["interpreted_as"] = "[C,25] -> [25,C] -> [5,5,C]"
            t2 = t.transpose(0, 1).contiguous()
            return tensor_to_nested_list(t2.reshape(5, 5, c)), meta

    if t.ndim == 3:
        if t.shape[1] == 5 and t.shape[2] == 5:
            meta["interpreted_as"] = "[C,5,5] -> [5,5,C]"
            t2 = t.permute(1, 2, 0).contiguous()
            return tensor_to_nested_list(t2), meta

        if 25 in t.shape:
            meta["interpreted_as"] = "squeezed_fallback"
            t2 = t.squeeze()
            if isinstance(t2, torch.Tensor) and t2.ndim >= 2:
                return infer_uv_grid(t2)

    meta["interpreted_as"] = "raw_fallback"
    return tensor_to_nested_list(t), meta


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
    max_nodes_for_a3: Optional[int] = 768,
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

    if max_nodes_for_a3 is not None and max_nodes_for_a3 <= 0:
        args.max_nodes_for_a3 = None
    else:
        args.max_nodes_for_a3 = max_nodes_for_a3

    num_classes = int(getattr(args, "num_classes", 25))
    model = BrepSeg(args)
    state = ckpt["state_dict"]
    seg_sd = {
        k: v for k, v in state.items() if k.startswith(("brep_encoder.", "attention.", "classifier."))
    }
    if not seg_sd:
        raise ValueError("No brep_encoder / attention / classifier weights in checkpoint")

    ignorable_missing = frozenset({"class_weights"})
    if "class_weights" in state and "class_weights" not in seg_sd:
        seg_sd = {**seg_sd, "class_weights": state["class_weights"]}

    incompatible = model.load_state_dict(seg_sd, strict=False)
    bad = [
        k for k in incompatible.missing_keys if not k.startswith("_") and k not in ignorable_missing
    ]
    if bad:
        raise RuntimeError(f"Missing required keys: {bad[:8]}...")
    model.eval()
    model.to(device)
    return model, num_classes


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.inference_mode()
def predict_argmax_classes(model: BrepSeg, batch: Dict[str, Any]) -> torch.Tensor:
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
    return torch.argmax(node_seg, dim=-1).long()


def read_predicted_classes_csv(csv_path: pathlib.Path, num_faces: int) -> Optional[np.ndarray]:
    if not csv_path.is_file():
        return None
    preds = np.full(num_faces, -1, dtype=np.int64)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if not reader.fieldnames or "predicted_class" not in reader.fieldnames:
                print(f"[ERROR] missing predicted_class column: {csv_path}")
                return None
            if "face_index" not in reader.fieldnames:
                print(f"[ERROR] missing face_index column: {csv_path}")
                return None
            filled = np.zeros(num_faces, dtype=np.bool_)
            for row in reader:
                fi = int(row["face_index"])
                if fi < 0 or fi >= num_faces:
                    print(f"[WARN] illegal face_index {fi} in {csv_path}")
                    continue
                preds[fi] = int(row["predicted_class"])
                filled[fi] = True
        if not filled.all():
            missing = (~filled).nonzero()[0][:8].tolist()
            print(f"[WARN] incomplete predictions in {csv_path} (e.g. missing faces {missing})")
            return None
        return preds
    except Exception as e:
        print(f"[ERROR] read CSV failed {csv_path}: {e}")
        return None


def build_uv_json_payload(
    pt_path: pathlib.Path,
    inference_csv_path: pathlib.Path,
    preds: np.ndarray,
    pyg: Any,
) -> Optional[Dict[str, Any]]:
    node_data = pyg.node_data
    num_faces = int(node_data.shape[0])
    if preds.size != num_faces:
        print(f"[WARN] preds len {preds.size} != faces {num_faces} for {pt_path.name}")
        return None

    faces_out: List[Dict[str, Any]] = []
    for face_idx in range(num_faces):
        lab = int(preds[face_idx])
        if lab == 0:
            continue
        uv_tensor = node_data[face_idx]
        uv_grid, meta = infer_uv_grid(uv_tensor.float())
        faces_out.append(
            {
                "face_index": face_idx,
                "label": lab,
                "uv_grid": uv_grid,
                "uv_meta": meta,
            }
        )

    return {
        "file": pt_path.stem,
        "bin_path": str(pt_path.resolve()),
        "label_path": str(inference_csv_path.resolve()) if inference_csv_path.is_file() else "",
        "num_faces_in_graph": num_faces,
        "num_labels_in_json": num_faces,
        "num_labeled_faces": len(faces_out),
        "faces": faces_out,
    }


def torch_load_pt(path: pathlib.Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def process_one_graph(
    pt_path: pathlib.Path,
    inference_csv: pathlib.Path,
    uv_out: pathlib.Path,
    pyg: Any,
    model: Optional[BrepSeg],
    num_classes: int,
    device: torch.device,
    multi_hop: int,
    spatial_max: int,
    skip_existing: bool,
) -> bool:
    if skip_existing and uv_out.is_file():
        return True

    if not hasattr(pyg, "node_data"):
        print(f"[skip] invalid PyG {pt_path}")
        return False

    n_faces = int(pyg.node_data.shape[0])
    preds = read_predicted_classes_csv(inference_csv, n_faces)

    if preds is None:
        if model is None:
            print(f"[skip] no preds and no --checkpoint for {pt_path.name}")
            return False
        b = collator([pyg], multi_hop, spatial_max)
        b = _batch_to_device(b, device)
        preds = predict_argmax_classes(model, b).cpu().numpy().astype(np.int64)

    payload = build_uv_json_payload(pt_path, inference_csv, preds, pyg)
    if payload is None:
        return False

    uv_out.parent.mkdir(parents=True, exist_ok=True)
    with uv_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return True


def main() -> None:
    default_root = pathlib.Path(r"Y:\new_dataset\test")

    ap = argparse.ArgumentParser(description="Export uv_json_pred from PyG + inference CSV.")
    ap.add_argument("--dataset_root", type=pathlib.Path, default=default_root)
    ap.add_argument("--only", type=str, default=None, help="Comma subset: abc,mfcadpp,cadsynth")
    ap.add_argument("--inference_subdir", type=str, default="inference")
    ap.add_argument("--uv_json_subdir", type=str, default="uv_json_pred")
    ap.add_argument("--checkpoint", type=pathlib.Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--multi_hop_max_dist", type=int, default=16)
    ap.add_argument("--spatial_pos_max", type=int, default=32)
    ap.add_argument(
        "--max_nodes_for_a3",
        type=int,
        default=768,
        help="Same as run_pyg_inference: cap A3 for huge graphs (0 = no cap).",
    )
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument(
        "--pyg_dir",
        type=pathlib.Path,
        default=None,
        help="Process one folder of *.pt (overrides --dataset_root / --only). Pair with --inference_dir.",
    )
    ap.add_argument(
        "--inference_dir",
        type=pathlib.Path,
        default=None,
        help="CSV dir when using --pyg_dir (default: <parent of pyg_dir>/inference).",
    )
    ap.add_argument(
        "--uv_json_dir",
        type=pathlib.Path,
        default=None,
        help="Output JSON dir when using --pyg_dir (default: <parent of pyg_dir>/uv_json_pred).",
    )
    args = ap.parse_args()

    if args.pyg_dir is not None:
        pyg_dir = args.pyg_dir.expanduser().resolve()
        if not pyg_dir.is_dir():
            raise SystemExit(f"--pyg_dir is not a directory: {pyg_dir}")
        inf_dir = (
            args.inference_dir.expanduser().resolve()
            if args.inference_dir is not None
            else (pyg_dir.parent / "inference")
        )
        uv_dir = (
            args.uv_json_dir.expanduser().resolve()
            if args.uv_json_dir is not None
            else (pyg_dir.parent / "uv_json_pred")
        )

        device = torch.device(args.device)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA unavailable; using CPU.", flush=True)
            device = torch.device("cpu")

        model: Optional[BrepSeg] = None
        num_classes = 25
        if args.checkpoint is not None:
            ck = args.checkpoint.resolve()
            print(f"Loading checkpoint (CSV fallback): {ck}", flush=True)
            model, num_classes = load_brepseg_for_inference(ck, device, int(args.max_nodes_for_a3))

        files = sorted(pyg_dir.glob("*.pt"))
        if args.max_files is not None:
            files = files[: args.max_files]

        grand_ok = grand_fail = 0
        for pt_path in tqdm(files, desc="uv_json_pred(single_dir)"):
            stem = pt_path.stem
            csv_path = inf_dir / f"{stem}.csv"
            json_path = uv_dir / f"{stem}.json"
            try:
                pyg = torch_load_pt(pt_path)
            except Exception as e:
                print(f"[skip] load failed {pt_path}: {e}")
                grand_fail += 1
                continue
            ok = process_one_graph(
                pt_path,
                csv_path,
                json_path,
                pyg,
                model,
                num_classes,
                device,
                args.multi_hop_max_dist,
                args.spatial_pos_max,
                args.skip_existing,
            )
            if ok:
                grand_ok += 1
            else:
                grand_fail += 1
        print(f"single_dir: ok={grand_ok} fail={grand_fail} -> {uv_dir}", flush=True)
        return

    sets = {"cadsynth", "mfcadpp", "abc"}
    if args.only:
        sel = {x.strip().lower() for x in args.only.split(",") if x.strip()}
        bad = sel - sets
        if bad:
            raise SystemExit(f"Bad --only: {bad}")
        triple = [(n, args.dataset_root / n) for n in ("cadsynth", "mfcadpp", "abc") if n in sel]
    else:
        triple = [(n, args.dataset_root / n) for n in ("cadsynth", "mfcadpp", "abc")]

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.", flush=True)
        device = torch.device("cpu")

    model: Optional[BrepSeg] = None
    num_classes = 25
    if args.checkpoint is not None:
        ck = args.checkpoint.resolve()
        print(f"Loading checkpoint (CSV fallback): {ck}", flush=True)
        model, num_classes = load_brepseg_for_inference(ck, device, int(args.max_nodes_for_a3))

    grand_ok = grand_fail = 0

    for name, base in triple:
        pyg_dir = base / "graph" / "pyg"
        inf_dir = base / args.inference_subdir
        uv_dir = base / args.uv_json_subdir
        if not pyg_dir.is_dir():
            print(f"[skip] {name}: no {pyg_dir}")
            continue

        files = sorted(pyg_dir.glob("*.pt"))
        if args.max_files is not None:
            files = files[: args.max_files]

        local_ok = 0
        for pt_path in tqdm(files, desc=f"{name}:uv_json_pred"):
            stem = pt_path.stem
            csv_path = inf_dir / f"{stem}.csv"
            json_path = uv_dir / f"{stem}.json"

            try:
                pyg = torch_load_pt(pt_path)
            except Exception as e:
                print(f"[skip] load failed {pt_path}: {e}")
                grand_fail += 1
                continue

            ok = process_one_graph(
                pt_path,
                csv_path,
                json_path,
                pyg,
                model,
                num_classes,
                device,
                args.multi_hop_max_dist,
                args.spatial_pos_max,
                args.skip_existing,
            )
            if ok:
                local_ok += 1
                grand_ok += 1
            else:
                grand_fail += 1

        print(f"{name}: uv_json ok={local_ok}/{len(files)} -> {uv_dir}", flush=True)

    print(f"TOTAL ok={grand_ok} fail={grand_fail}", flush=True)


if __name__ == "__main__":
    main()
