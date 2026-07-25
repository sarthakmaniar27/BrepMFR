#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch PyG inference for **thread identification** (2 classes: stock=0, thread=1).

Mirrors ``scripts/inference/run_pyg_inference.py`` but uses thread class names,
defaults for the lite Stage-1 checkpoint, optional ``train.txt`` / ``val.txt`` /
``test.txt`` filtering, and writes a small evaluation bundle (confusion matrix,
per-class CSV, summary.md) when ground truth is available.

Usage (PowerShell, from repo root)::

  conda run -n brep_mfr_pyg python scripts/threads/run_thread_pyg_inference.py `
    --checkpoint results/stage1/thread_lite_ce_weighted_exp1_memsafe/best.ckpt `
    --dataset_path D:/threads/lite `
    --split test

Single folder of ``*.pt`` (no split list)::

  conda run -n brep_mfr_pyg python scripts/threads/run_thread_pyg_inference.py `
    --checkpoint results/stage1/thread_lite_ce_weighted_exp1_memsafe/best.ckpt `
    --pyg_dir D:/threads/lite/pyg `
    --inference_dir D:/threads/lite/inference_test
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

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

_inf_path = Path(__file__).resolve().parents[1] / "inference" / "run_pyg_inference.py"
_spec = importlib.util.spec_from_file_location("run_pyg_inference", _inf_path)
assert _spec is not None and _spec.loader is not None
_ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ri)

from data.collator import collator  # noqa: E402

THREAD_FACE_LABEL_NAME: Dict[int, str] = {
    0: "stock",
    1: "thread",
    2: "text",
}

DEFAULT_CHECKPOINT = (
    Path("results") / "stage1" / "thread_lite_ce_weighted_exp1_memsafe" / "best.ckpt"
)


def _class_name(cid: int) -> str:
    return THREAD_FACE_LABEL_NAME.get(int(cid), f"class_{cid}")


def write_thread_graph_csv(
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
                "predicted_class_name": _class_name(int(preds[i])),
                "predicted_probability": f"{float(p_hat[i]):.10g}",
            }
            if with_gt:
                yi = int(gt[i])  # type: ignore[index]
                row["ground_truth_class"] = yi
                row["ground_truth_class_name"] = _class_name(yi)
                row["correct_top1"] = int(preds[i] == yi)
            w.writerow(row)


def _read_split_stems(split_file: pathlib.Path) -> List[str]:
    lines = split_file.read_text(encoding="utf-8").splitlines()
    stems: List[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        stems.append(s.removesuffix(".pt"))
    return stems


def _resolve_pt_files(
    pyg_dir: pathlib.Path,
    split_file: Optional[pathlib.Path],
    max_files: Optional[int],
) -> Tuple[List[pathlib.Path], int, int]:
    """
    Returns (files_to_run, n_listed_in_split, n_missing_on_disk).
    """
    if split_file is not None:
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        stems = _read_split_stems(split_file)
        files: List[pathlib.Path] = []
        missing = 0
        for stem in stems:
            p = pyg_dir / f"{stem}.pt"
            if p.is_file():
                files.append(p)
            else:
                missing += 1
        if max_files is not None:
            files = files[: max_files]
        return files, len(stems), missing

    files = sorted(pyg_dir.glob("*.pt"))
    if max_files is not None:
        files = files[: max_files]
    return files, len(files), 0


def _confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, y in zip(preds, labels):
        if 0 <= int(y) < num_classes and 0 <= int(p) < num_classes:
            cm[int(y), int(p)] += 1
    return cm


def _per_class_rows(
    cm: np.ndarray,
    num_classes: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in range(num_classes):
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
                "class_name": _class_name(c),
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


def _write_confusion_csv(path: pathlib.Path, cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = cm.shape[0]
    header = ["true\\pred"] + [_class_name(j) for j in range(n)]
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for i in range(n):
            w.writerow([_class_name(i)] + [int(cm[i, j]) for j in range(n)])


def _write_per_class_csv(path: pathlib.Path, rows: Sequence[Dict[str, Any]]) -> None:
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
                out[k] = "" if v != v else f"{float(v):.6g}"  # NaN -> blank
            w.writerow(out)


def _write_summary_md(
    path: pathlib.Path,
    *,
    pyg_dir: pathlib.Path,
    inference_dir: pathlib.Path,
    checkpoint: pathlib.Path,
    split_file: Optional[pathlib.Path],
    n_listed: int,
    n_missing: int,
    n_candidates: int,
    n_graphs: int,
    n_faces: int,
    labelled_faces: int,
    mean_top1: Optional[float],
    per_class: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Thread PyG inference summary",
        "",
        f"- **checkpoint**: `{checkpoint}`",
        f"- **pyg_dir**: `{pyg_dir}`",
        f"- **inference_dir**: `{inference_dir}`",
    ]
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
    ious = [float(row["iou"]) for row in per_class if row["iou"] == row["iou"]]
    if ious:
        mean_iou = float(np.mean(ious))
        lines.append(f"- **mIoU** (macro mean of per-class IoU): {mean_iou:.6f} ({100.0 * mean_iou:.4f}%)")
    lines += ["", "## Per-class (face-level)", ""]
    for row in per_class:
        p = row["precision"]
        r = row["recall"]
        iou = row["iou"]
        ps = "n/a" if p != p else f"{100.0 * float(p):.2f}%"
        rs = "n/a" if r != r else f"{100.0 * float(r):.2f}%"
        iou_s = "n/a" if iou != iou else f"{float(iou):.6f}"
        lines.append(
            f"- **{row['class_name']}** (id={row['class_id']}): "
            f"support={row['support_faces']} IoU={iou_s} precision={ps} recall={rs}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def inference_thread_dataset(
    name: str,
    pyg_dir: pathlib.Path,
    inference_out: pathlib.Path,
    device: torch.device,
    batch_size: int,
    multi_hop_max_dist: int,
    spatial_pos_max: int,
    model: Any,
    num_classes: int,
    split_file: Optional[pathlib.Path],
    max_files: Optional[int],
    metrics_dir: Optional[pathlib.Path],
    checkpoint: pathlib.Path,
) -> Tuple[int, int, int, Optional[float], int, int]:
    """
    Returns (pt_candidates, graphs_written, total_faces, mean_top1_or_None,
    n_listed_in_split, n_missing_on_disk).
    """
    files, n_listed, n_missing = _resolve_pt_files(pyg_dir, split_file, max_files)
    if split_file is not None and n_missing:
        print(
            f"[warn] {name}: {n_missing}/{n_listed} split stems have no .pt under {pyg_dir}",
            flush=True,
        )

    graph_parent = pyg_dir.parent
    inference_out.mkdir(parents=True, exist_ok=True)

    graphs_written = 0
    total_faces = 0
    correct = 0
    labelled_faces = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    batch_paths: List[pathlib.Path] = []
    batch_pygs: List[Any] = []
    batch_gt: List[Optional[np.ndarray]] = []

    def flush_chunk() -> None:
        nonlocal batch_paths, batch_pygs, batch_gt, total_faces, correct, labelled_faces
        nonlocal graphs_written, all_preds, all_labels
        if not batch_pygs:
            return
        if sum(_ri._num_faces(g) for g in batch_pygs) == 0:
            print(f"[skip] batch has zero total faces: {[p.name for p in batch_paths]}")
            batch_paths = []
            batch_pygs = []
            batch_gt = []
            return
        b = collator(batch_pygs, multi_hop_max_dist, spatial_pos_max)
        b = _ri._batch_to_device(b, device)
        probs_t = _ri.predict_probs_per_node(model, b, num_classes)
        probs = probs_t.float().cpu().numpy()

        offset = 0
        for k, pt_path in enumerate(batch_paths):
            n = _ri._num_faces(batch_pygs[k])
            pslice = probs[offset : offset + n]
            offset += n
            gt_arr = batch_gt[k]
            preds = pslice.argmax(axis=1).astype(np.int64)
            if gt_arr is not None:
                labelled_faces += n
                correct += int((preds == gt_arr).sum())
                all_preds.extend(preds.tolist())
                all_labels.extend(gt_arr.tolist())
            out_csv = inference_out / f"{pt_path.stem}_predictions.csv"
            write_thread_graph_csv(out_csv, pslice, num_classes, gt_arr)
            total_faces += n
            graphs_written += 1
        batch_paths = []
        batch_pygs = []
        batch_gt = []

    for pt_path in tqdm(files, desc=name):
        try:
            pyg = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[skip] load failed {pt_path}: {e}")
            continue
        if not hasattr(pyg, "edge_index") or not hasattr(pyg, "node_data"):
            print(f"[skip] not BrepMFR PyG layout: {pt_path}")
            continue
        if _ri._num_faces(pyg) == 0:
            print(f"[skip] zero faces: {pt_path}")
            continue

        gt = _ri._resolve_gt(pyg, pt_path.stem, graph_parent, num_classes)
        batch_paths.append(pt_path)
        batch_pygs.append(pyg)
        batch_gt.append(gt)
        if len(batch_pygs) >= batch_size:
            flush_chunk()

    flush_chunk()

    mean_acc = (correct / labelled_faces) if labelled_faces > 0 else None

    if metrics_dir is not None and labelled_faces == 0:
        print(
            "[warn] metrics_dir set but no labelled faces — check sidecar "
            f"{pyg_dir.parent / 'label' / '<stem>.json'} with key 'labels' (length = num faces).",
            file=sys.stderr,
        )

    if metrics_dir is not None and labelled_faces > 0:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        preds_np = np.asarray(all_preds, dtype=np.int64)
        labels_np = np.asarray(all_labels, dtype=np.int64)
        cm = _confusion_matrix(preds_np, labels_np, num_classes)
        per_class = _per_class_rows(cm, num_classes)
        _write_confusion_csv(metrics_dir / "confusion_matrix.csv", cm)
        _write_per_class_csv(metrics_dir / "per_class.csv", per_class)
        _write_summary_md(
            metrics_dir / "summary.md",
            pyg_dir=pyg_dir,
            inference_dir=inference_out,
            checkpoint=checkpoint,
            split_file=split_file,
            n_listed=n_listed,
            n_missing=n_missing,
            n_candidates=len(files),
            n_graphs=graphs_written,
            n_faces=total_faces,
            labelled_faces=labelled_faces,
            mean_top1=mean_acc,
            per_class=per_class,
        )

    return len(files), graphs_written, total_faces, mean_acc, n_listed, n_missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Thread (stock vs thread) PyG batch inference with optional split lists.",
    )
    ap.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=DEFAULT_CHECKPOINT,
        help=f"BrepSeg Lightning .ckpt (default: {DEFAULT_CHECKPOINT}).",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=1, help="Graphs per collator batch.")
    ap.add_argument(
        "--dataset_path",
        type=pathlib.Path,
        default=None,
        help="Folder with *.pt and train.txt/val.txt/test.txt (used with --split).",
    )
    ap.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        default="test",
        help="Which split list under --dataset_path (default: test). Ignored if --pyg_dir only.",
    )
    ap.add_argument(
        "--split_file",
        type=pathlib.Path,
        default=None,
        help="Explicit split list (one stem per line). Overrides --split.",
    )
    ap.add_argument(
        "--pyg_dir",
        type=pathlib.Path,
        default=None,
        help="Folder of *.pt graphs. Default: --dataset_path when set.",
    )
    ap.add_argument(
        "--inference_dir",
        type=pathlib.Path,
        default=None,
        help="Per-graph CSV output dir (default: <dataset_path>/inference_<split> or parent/inference).",
    )
    ap.add_argument(
        "--metrics_dir",
        type=pathlib.Path,
        default=None,
        help="Write confusion_matrix.csv, per_class.csv, summary.md here when GT exists.",
    )
    ap.add_argument("--multi_hop_max_dist", type=int, default=16)
    ap.add_argument("--spatial_pos_max", type=int, default=32)
    ap.add_argument(
        "--max_nodes_for_a3",
        type=int,
        default=768,
        help="Skip A3 edge bias when padded nodes exceed this (0 = no cap).",
    )
    ap.add_argument("--max_files", type=int, default=None, help="Debug cap on graphs.")
    args = ap.parse_args()

    if args.pyg_dir is None and args.dataset_path is None:
        ap.error("Provide --dataset_path and/or --pyg_dir")

    if args.pyg_dir is not None:
        pyg_dir = args.pyg_dir.expanduser().resolve()
    else:
        pyg_dir = args.dataset_path.expanduser().resolve()

    if not pyg_dir.is_dir():
        raise SystemExit(f"pyg_dir is not a directory: {pyg_dir}")

    split_file: Optional[pathlib.Path] = None
    if args.split_file is not None:
        split_file = args.split_file.expanduser().resolve()
    elif args.dataset_path is not None:
        ds_root = args.dataset_path.expanduser().resolve()
        candidate = ds_root / f"{args.split}.txt"
        if candidate.is_file():
            split_file = candidate
        elif args.pyg_dir is None:
            print(
                f"[warn] no split file {candidate}; inferring all *.pt under {pyg_dir}",
                flush=True,
            )

    if args.inference_dir is not None:
        infer_dir = args.inference_dir.expanduser().resolve()
    elif args.dataset_path is not None:
        infer_dir = args.dataset_path.expanduser().resolve() / f"inference_{args.split}"
    else:
        infer_dir = pyg_dir.parent / "inference"

    if args.metrics_dir is not None:
        metrics_dir = args.metrics_dir.expanduser().resolve()
    elif args.dataset_path is not None:
        metrics_dir = infer_dir.parent / f"metrics_{args.split}"
    else:
        metrics_dir = None

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.", flush=True)
        device = torch.device("cpu")

    ckpt_path = args.checkpoint.expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    model, num_classes = _ri.load_brepseg_for_inference(
        ckpt_path, device, int(args.max_nodes_for_a3)
    )
    if num_classes != 2:
        print(
            f"Warning: checkpoint num_classes={num_classes} (expected 2 for thread/stock).",
            flush=True,
        )
    print(f"Loaded num_classes={num_classes} from {ckpt_path}", flush=True)

    n_found, n_graph, n_face, mean_acc, n_listed, n_missing = inference_thread_dataset(
        name=args.split if split_file else "pyg",
        pyg_dir=pyg_dir,
        inference_out=infer_dir,
        device=device,
        batch_size=max(1, int(args.batch_size)),
        multi_hop_max_dist=args.multi_hop_max_dist,
        spatial_pos_max=args.spatial_pos_max,
        model=model,
        num_classes=num_classes,
        split_file=split_file,
        max_files=args.max_files,
        metrics_dir=metrics_dir,
        checkpoint=ckpt_path,
    )

    extra = ""
    if mean_acc is not None:
        extra = f"  per_face_accuracy={100.0 * mean_acc:.2f}%"
    split_note = ""
    if split_file is not None:
        split_note = f" split_listed={n_listed} missing_pt={n_missing}"
    print(
        f"pyg_dir={pyg_dir}{split_note}: pt_candidates={n_found} graphs_ok={n_graph} "
        f"faces={n_face}{extra}\nCSV -> {infer_dir}",
        flush=True,
    )
    if metrics_dir is not None:
        print(f"Metrics -> {metrics_dir}", flush=True)


if __name__ == "__main__":
    main()
