# -*- coding: utf-8 -*-
"""
Post-hoc logit adjustment + tau sweep for a Stage 2 DomainAdapt checkpoint.

Runs inference on a target split (default t_test.txt), caches per-face softmax
probs, sweeps tau, and writes per-class metrics (accuracy/recall, precision,
F1, IoU) with human-readable class names.

Usage (PowerShell):

  cd C:\\Users\\D58\\Desktop\\BrepMFR_PyG
  # Use brep_mfr_pyg (torch_geometric). Plain `python` may be base/other env.
  conda activate brep_mfr_pyg
  python scripts/diagnostics/stage2_logit_adjust_eval.py ^
    --checkpoint results/.../best.ckpt ^
    --source_path Z:/Experiment6_PyG/source_dataset ^
    --target_path Z:/Experiment6_PyG/target_dataset ^
    --target_split test ^
    --out_dir results/diagnostics/stage2_logit_adjust_test
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import sys
from pathlib import Path

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TransferDataset, _dataloader_kw
from diagnose_stage1_target import (
    FilelistDataset,
    make_loader,
    per_class_metrics,
    write_confusion_csv,
)
from models.transfer_model import DomainAdapt

from logit_adjust_eval import (
        adjusted_predictions,
        confusion_matrix,
        metrics_summary,
        priors_from_counts,
    )


# Authoritative 25-class names (CADSynth / BrepMFR machining features)
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
    24: "Round",
}


def _move_batch_to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def count_labels_filelist(
    root: str,
    filelist: str,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    pt_subdir: str | None = None,
):
    """Count face labels using label-only path via FilelistDataset batches."""
    ds = FilelistDataset(root, filelist, pt_subdir)
    loader = make_loader(ds, batch_size, num_workers)
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in tqdm(loader, desc=f"count {filelist}", dynamic_ncols=True):
        labels = batch["label_feature"].long().numpy()
        labels = labels[(labels >= 0) & (labels < num_classes)]
        counts += np.bincount(labels, minlength=num_classes)
    return counts


@torch.no_grad()
def collect_target_probs_stage2(model: DomainAdapt, loader, num_classes: int, device: torch.device):
    """Run Stage-2 model on TransferDataset batches; return target probs + labels."""
    model.eval()
    prob_chunks = []
    label_chunks = []
    for batch in tqdm(loader, desc="infer target (Stage2)", dynamic_ncols=True):
        batch = _move_batch_to_device(batch, device)

        node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True)
        node_emb = node_emb[0].permute(1, 0, 2)
        node_emb = node_emb[:, 1:, :]
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)

        node_pos_s = torch.where(padding_mask_s == False)
        node_pos_t = torch.where(padding_mask_t == False)
        node_z_s = node_emb_s[node_pos_s]
        node_z_t = node_emb_t[node_pos_t]

        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)

        num_nodes_per_graph_s = torch.sum(~padding_mask_s, dim=-1)
        graph_z_s = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
        _ = model.attention([node_z_s, graph_z_s])

        num_nodes_per_graph_t = torch.sum(~padding_mask_t, dim=-1)
        graph_z_t = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
        z_t = model.attention([node_z_t, graph_z_t])

        node_seg_t = model.classifier(z_t)
        probs = node_seg_t.detach().float().cpu().numpy()

        num_node_s = node_z_s.size(0)
        label_t = batch["label_feature"][num_node_s:].long().cpu().numpy()

        mask = (label_t >= 0) & (label_t < num_classes)
        prob_chunks.append(probs[mask])
        label_chunks.append(label_t[mask])

    return np.concatenate(prob_chunks, axis=0), np.concatenate(label_chunks, axis=0)


def per_class_iou(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> list[float]:
    """Same construction as DomainAdapt.on_test_epoch_end."""
    out = []
    for i in range(num_classes):
        label_pos = np.where(labels == i)[0]
        pred_pos = np.where(preds == i)[0]
        if len(pred_pos) > 0 and len(label_pos) > 0:
            class_i_preds = preds[label_pos]
            class_i_label = labels[label_pos]
            inter = (class_i_preds == class_i_label).astype(np.float64)
            union_missed = (class_i_preds != class_i_label).astype(np.float64)
            class_i_preds_ = preds[pred_pos]
            class_i_label_ = labels[pred_pos]
            union_fp = (class_i_preds_ != class_i_label_).astype(np.float64)
            denom = np.sum(union_missed) + np.sum(inter) + np.sum(union_fp)
            out.append(float(np.sum(inter) / denom) if denom > 0 else 0.0)
        else:
            out.append(float("nan"))
    return out


def write_per_class_table_csv(path: pathlib.Path, rep: list, ious: list[float]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "class_id",
            "class_name",
            "support",
            "accuracy_recall",
            "precision",
            "f1",
            "iou",
            "predicted_count",
        ])
        for c, row in enumerate(rep):
            w.writerow([
                c,
                FACE_LABEL_NAME.get(c, ""),
                row["support"],
                f"{row['recall']:.6f}",
                f"{row['precision']:.6f}",
                f"{row['f1']:.6f}",
                f"{ious[c]:.6f}" if np.isfinite(ious[c]) else "",
                row["predicted_count"],
            ])


def main():
    ap = argparse.ArgumentParser("Stage 2 logit adjustment + tau sweep")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--target_path", required=True)
    ap.add_argument(
        "--target_split",
        choices=("val", "test"),
        default="test",
        help="Which TransferDataset split for target (pairs s_val/t_val or s_test/t_test).",
    )
    ap.add_argument("--source_prior_filelist", default="s_val.txt",
                    help="Filelist for estimating P_source(y) counts (default: val).")
    ap.add_argument("--num_classes", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--taus",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0",
    )
    ap.add_argument("--out_dir", default="results/diagnostics/stage2_logit_adjust")
    ap.add_argument("--max_batches", type=int, default=0, help="If >0, truncates infer (smoke test).")
    ap.add_argument(
        "--pt_subdir",
        default=None,
        help=(
            "Relative subgraph dir under source/target roots (e.g. output/bin_skip_a2)."
        ),
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading Stage 2 checkpoint: {args.checkpoint}")
    model = DomainAdapt.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device)
    model.eval()

    # --- Priors: source from filelist; target empirical on same split we evaluate ---
    print("\nCounting source priors:", args.source_prior_filelist)
    src_counts = count_labels_filelist(
        args.source_path,
        args.source_prior_filelist,
        args.num_classes,
        args.batch_size,
        args.num_workers,
        args.pt_subdir,
    )
    tgt_filelist = "t_val.txt" if args.target_split == "val" else "t_test.txt"
    print("Counting target priors (same split as infer):", tgt_filelist)
    tgt_counts = count_labels_filelist(
        args.target_path,
        tgt_filelist,
        args.num_classes,
        args.batch_size,
        args.num_workers,
        args.pt_subdir,
    )

    src_priors = priors_from_counts(src_counts)
    tgt_priors = priors_from_counts(tgt_counts)
    log_ratio = np.log(tgt_priors) - np.log(src_priors)

    # --- Transfer loader (drop_last=False so all target graphs covered) ---
    split = args.target_split
    test_ds = TransferDataset(
        root_dir_source=args.source_path,
        root_dir_target=args.target_path,
        split=split,
        random_rotate=False,
        num_class=args.num_classes,
        open_set=0,
        pt_subdir=args.pt_subdir,
    )
    dl_kw = _dataloader_kw(args.num_workers)
    dl_kw["drop_last"] = False
    infer_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_ds._collate,
        **dl_kw,
    )

    if args.max_batches > 0:
        def _trunc(loader, n):
            for i, b in enumerate(loader):
                if i >= n:
                    break
                yield b

        infer_loader = _trunc(infer_loader, args.max_batches)

    print("\n=== Single inference pass (target only, cached probs) ===")
    target_probs, target_labels = collect_target_probs_stage2(
        model, infer_loader, args.num_classes, device
    )
    print(f"  collected {target_probs.shape[0]:,} target faces")

    taus = [float(t) for t in args.taus.split(",")]
    sweep_rows = []
    best_tau = 0.0
    best_per_class = -1.0
    baseline_pf, baseline_pc, _ = metrics_summary(
        target_probs.argmax(axis=1), target_labels, args.num_classes
    )
    print("\n=== Tau sweep (per-class acc = mean of class recalls) ===")
    print(f"  {'tau':>6} {'per_face':>10} {'per_class':>10}")
    for tau in taus:
        preds = adjusted_predictions(target_probs, log_ratio, tau)
        pf, pc, _ = metrics_summary(preds, target_labels, args.num_classes)
        sweep_rows.append({"tau": tau, "per_face": pf, "per_class": pc})
        mark = ""
        if pc > best_per_class:
            best_per_class = pc
            best_tau = tau
            mark = " *"
        print(f"  {tau:6.2f} {pf:10.4f} {pc:10.4f}{mark}")

    # Baseline (tau=0) full metrics table
    pred_b = target_probs.argmax(axis=1)
    cm_b = confusion_matrix(pred_b, target_labels, args.num_classes)
    rep_b = per_class_metrics(cm_b, args.num_classes)
    iou_b = per_class_iou(pred_b, target_labels, args.num_classes)

    best_preds = adjusted_predictions(target_probs, log_ratio, best_tau)
    cm_best = confusion_matrix(best_preds, target_labels, args.num_classes)
    rep_best = per_class_metrics(cm_best, args.num_classes)
    iou_best = per_class_iou(best_preds, target_labels, args.num_classes)

    write_per_class_table_csv(out_dir / "per_class_baseline_tau0.csv", rep_b, iou_b)
    write_per_class_table_csv(
        out_dir / f"per_class_best_tau{best_tau:g}.csv", rep_best, iou_best
    )

    write_confusion_csv(out_dir / "confusion_target_baseline.csv", cm_b)
    write_confusion_csv(out_dir / f"confusion_target_tau{best_tau:g}.csv", cm_best)

    with open(out_dir / "tau_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tau", "per_face_acc", "per_class_acc"])
        for r in sweep_rows:
            w.writerow([f"{r['tau']:.4f}", f"{r['per_face']:.6f}", f"{r['per_class']:.6f}"])

    mean_iou_b = np.nanmean(np.array(iou_b, dtype=np.float64))
    mean_iou_bt = np.nanmean(np.array(iou_best, dtype=np.float64))

    # Markdown: wide table
    md_path = out_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Stage 2 + post-hoc logit adjustment\n\n")
        f.write(f"- Checkpoint: `{args.checkpoint}`\n")
        f.write(f"- Target split: `{args.target_split}` (filelist `{tgt_filelist}`)\n")
        f.write(f"- Faces evaluated: **{target_probs.shape[0]:,}**\n")
        f.write(f"- Source priors from: `{args.source_prior_filelist}`\n\n")

        f.write("## Tau sweep\n\n")
        f.write("| tau | per-face acc | mean per-class recall |\n|---:|---:|---:|\n")
        for r in sweep_rows:
            star = " *" if r["tau"] == best_tau else ""
            f.write(f"| {r['tau']:.2f} | {r['per_face']:.4f} | {r['per_class']:.4f}{star} |\n")
        f.write(f"\n**Best tau:** {best_tau:g} (highest mean per-class recall)\n\n")

        pf0, pc0, _ = metrics_summary(pred_b, target_labels, args.num_classes)
        pfb, pcb, _ = metrics_summary(best_preds, target_labels, args.num_classes)
        f.write("## Global metrics\n\n")
        f.write("| Setting | per-face acc | mean per-class recall | mean IoU (finite classes) |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| Baseline τ=0 | {pf0:.4f} | {pc0:.4f} | {mean_iou_b:.4f} |\n")
        f.write(f"| Best τ={best_tau:g} | {pfb:.4f} | {pcb:.4f} | {mean_iou_bt:.4f} |\n\n")

        f.write("## Per-class metrics at baseline (τ=0)\n\n")
        f.write("| ID | Class | Support | Recall | Precision | F1 | IoU |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for c in range(args.num_classes):
            r = rep_b[c]
            iu = iou_b[c]
            iu_s = f"{iu:.4f}" if np.isfinite(iu) else "—"
            name = FACE_LABEL_NAME.get(c, "").replace("|", "\\|")
            f.write(
                f"| {c} | {name} | {r['support']} | {r['recall']:.4f} | "
                f"{r['precision']:.4f} | {r['f1']:.4f} | {iu_s} |\n"
            )

        f.write(f"\n## Per-class metrics at best τ={best_tau:g}\n\n")
        f.write("| ID | Class | Support | Recall | Precision | F1 | IoU |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for c in range(args.num_classes):
            r = rep_best[c]
            iu = iou_best[c]
            iu_s = f"{iu:.4f}" if np.isfinite(iu) else "—"
            name = FACE_LABEL_NAME.get(c, "").replace("|", "\\|")
            f.write(
                f"| {c} | {name} | {r['support']} | {r['recall']:.4f} | "
                f"{r['precision']:.4f} | {r['f1']:.4f} | {iu_s} |\n"
            )

    print(f"\nWrote: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
