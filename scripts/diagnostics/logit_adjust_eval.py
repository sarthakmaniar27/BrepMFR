# -*- coding: utf-8 -*-
"""
Option 1: post-hoc logit adjustment on the Stage 1 checkpoint.

Background:
    Stage 1 source-only training learned a class prior of P_source(y), which
    is dominated by class 0 (stock = 57.65%). On target val, true P_target(0)
    is only 22.36%, so the implicit class-0 prior in the softmax pushes ~41k
    target faces into the wrong class 0 bin. This is *label shift*, not
    covariate shift. DANN cannot fix label shift by construction (Zhao et al.
    2019). The standard correction is post-hoc logit adjustment:

        logits'[k] = logits[k] + tau * (log P_target(k) - log P_source(k))
        pred       = argmax_k logits'[k]

    Equivalently, multiply softmax probs by (P_target/P_source)^tau and argmax.
    With tau=1 and known target priors, this recovers the Bayes-optimal
    classifier under label shift (Saerens et al. 2002).

This script:
    1. Loads Stage 1 checkpoint (no DA, no retraining).
    2. Computes source priors from source val labels.
    3. Computes target priors from target val labels (we cheat for the POC;
       in production we'd estimate via BBSE).
    4. Runs model inference on target val ONCE, caches per-face probs.
    5. Sweeps tau in {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0} and reports
       per-face acc, per-class acc for each.
    6. Saves best-tau confusion matrix, per-class report, and a markdown
       summary to --out_dir.

Usage (PowerShell, single line):

  python scripts/diagnostics/logit_adjust_eval.py `
    --checkpoint "C:/Users/D58/Desktop/BrepMFR_PyG/results/BrepMFR/0425/183526/best.ckpt" `
    --source_path "Z:/Experiment6_PyG/source_dataset" `
    --target_path "Z:/Experiment6_PyG/target_dataset" `
    --num_classes 25 `
    --batch_size 32 `
    --num_workers 2 `
    --out_dir "results/diagnostics/logit_adjust"
"""

import argparse
import csv
import importlib.util
import pathlib
import sys
from collections import Counter
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

from diagnose_stage1_target import (
    FilelistDataset,
    load_stage1_model,
    make_loader,
    per_class_metrics,
    write_confusion_csv,
    write_per_class_csv,
)


# ---------------------------------------------------------------------------
# Inference (caches per-face probs + labels so we can sweep tau cheaply)
# ---------------------------------------------------------------------------


def _move_batch_to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def collect_probs(model, loader, num_classes, device, name):
    """Run model on `loader`; return (probs [N,C] float32, labels [N] int64)."""
    prob_chunks = []
    label_chunks = []
    for batch in tqdm(loader, desc=f"infer {name}", dynamic_ncols=True):
        batch = _move_batch_to_device(batch, device)

        node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True)
        node_emb = node_emb[0].permute(1, 0, 2)
        node_emb = node_emb[:, 1:, :]
        padding_mask = batch["padding_mask"]
        node_pos = torch.where(padding_mask == False)
        node_z = node_emb[node_pos]
        num_nodes_per_graph = (~padding_mask).sum(dim=-1)
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        z = model.attention([node_z, graph_z])
        # NonLinearClassifier ends in F.softmax — node_seg is already probs.
        probs = model.classifier(z).detach().float().cpu().numpy()
        labels = batch["label_feature"].long().detach().cpu().numpy()

        mask = (labels >= 0) & (labels < num_classes)
        prob_chunks.append(probs[mask])
        label_chunks.append(labels[mask])

    return np.concatenate(prob_chunks, axis=0), np.concatenate(label_chunks, axis=0)


# ---------------------------------------------------------------------------
# Counts / priors (no model needed)
# ---------------------------------------------------------------------------


def count_labels(loader, num_classes, name):
    """Iterate dataset to count labels — no model inference."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in tqdm(loader, desc=f"count {name}", dynamic_ncols=True):
        labels = batch["label_feature"].long().numpy()
        labels = labels[(labels >= 0) & (labels < num_classes)]
        counts += np.bincount(labels, minlength=num_classes)
    return counts


def priors_from_counts(counts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = counts.astype(np.float64)
    p = p / max(1, p.sum())
    return np.maximum(p, eps)  # avoid log(0)


# ---------------------------------------------------------------------------
# Adjustment + evaluation
# ---------------------------------------------------------------------------


def adjusted_predictions(
    probs: np.ndarray,
    log_ratio: np.ndarray,  # log(P_target) - log(P_source), shape [C]
    tau: float,
) -> np.ndarray:
    """argmax of (log_probs + tau * log_ratio). Numerically stable."""
    if tau == 0.0:
        return probs.argmax(axis=1)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    adjusted = log_probs + tau * log_ratio[None, :]
    return adjusted.argmax(axis=1)


def confusion_matrix(preds, labels, num_classes):
    idx = labels * num_classes + preds
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_summary(preds, labels, num_classes):
    """Returns (per_face_acc, per_class_acc_mean, per_class_acc_array)."""
    per_face = float((preds == labels).mean()) if len(labels) else 0.0
    per_class = []
    for c in range(num_classes):
        mask = labels == c
        n = int(mask.sum())
        if n > 0:
            per_class.append(float((preds[mask] == c).mean()))
    pca = np.array(per_class) if per_class else np.zeros(1)
    return per_face, float(pca.mean()), pca


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser("Stage 1 + post-hoc logit adjustment")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--target_path", required=True)
    parser.add_argument("--source_filelist", default="s_val.txt")
    parser.add_argument("--target_filelist", default="t_val.txt")
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--taus",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated tau values to sweep.",
    )
    parser.add_argument(
        "--uniform_target",
        action="store_true",
        help="Use uniform target prior (Menon-style) instead of empirical target priors.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/diagnostics/logit_adjust",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="If >0, stop after N batches per loader (smoke-test mode).",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_stage1_model(args.checkpoint, device)

    # ---------------- Source priors (label counts only) --------------------
    print("\nBuilding source val dataset (priors only)")
    src_ds = FilelistDataset(args.source_path, args.source_filelist)
    src_loader = make_loader(src_ds, args.batch_size, args.num_workers)

    print("Building target val dataset")
    tgt_ds = FilelistDataset(args.target_path, args.target_filelist)
    tgt_loader = make_loader(tgt_ds, args.batch_size, args.num_workers)

    if args.max_batches > 0:
        def _trunc(loader, n):
            for i, b in enumerate(loader):
                if i >= n:
                    break
                yield b
        src_loader = _trunc(src_loader, args.max_batches)
        tgt_loader_for_count = _trunc(tgt_loader, args.max_batches)
        # We need a separate iterator for inference; rebuild a fresh one.
        tgt_loader_for_infer = _trunc(
            make_loader(tgt_ds, args.batch_size, args.num_workers), args.max_batches
        )
    else:
        tgt_loader_for_count = make_loader(tgt_ds, args.batch_size, args.num_workers)
        tgt_loader_for_infer = tgt_loader

    print("\n=== Counting source priors ===")
    src_counts = count_labels(src_loader, args.num_classes, "source")
    print("\n=== Counting target priors ===")
    tgt_counts = count_labels(tgt_loader_for_count, args.num_classes, "target")

    src_priors = priors_from_counts(src_counts)
    if args.uniform_target:
        tgt_priors = np.full(args.num_classes, 1.0 / args.num_classes)
    else:
        tgt_priors = priors_from_counts(tgt_counts)

    log_ratio = np.log(tgt_priors) - np.log(src_priors)

    print("\nClass priors:")
    print(f"  {'class':>5} {'src %':>8} {'tgt %':>8} {'log ratio':>10}")
    for c in range(args.num_classes):
        print(f"  {c:5d} {100*src_priors[c]:8.3f} {100*tgt_priors[c]:8.3f} {log_ratio[c]:+10.3f}")

    # ---------------- Target inference (probs cache) -----------------------
    print("\n=== Running model on target val (single pass) ===")
    target_probs, target_labels = collect_probs(
        model, tgt_loader_for_infer, args.num_classes, device, "target"
    )
    print(f"  collected {target_probs.shape[0]:,} target faces")

    # ---------------- Tau sweep --------------------------------------------
    taus = [float(t) for t in args.taus.split(",")]
    sweep_rows = []
    best_tau = 0.0
    best_per_class = -1.0

    print("\n=== Tau sweep ===")
    print(f"  {'tau':>6} {'per_face':>10} {'per_class':>10}  {'delta_per_class':>15}")
    baseline_pf, baseline_pc, _ = metrics_summary(
        target_probs.argmax(axis=1), target_labels, args.num_classes
    )
    for tau in taus:
        preds = adjusted_predictions(target_probs, log_ratio, tau)
        pf, pc, _ = metrics_summary(preds, target_labels, args.num_classes)
        sweep_rows.append({"tau": tau, "per_face": pf, "per_class": pc})
        delta_pc = pc - baseline_pc
        marker = ""
        if pc > best_per_class:
            best_per_class = pc
            best_tau = tau
            marker = " <-- best"
        print(f"  {tau:6.2f} {pf:10.4f} {pc:10.4f}  {delta_pc:+15.4f}{marker}")

    # ---------------- Save outputs at best tau -----------------------------
    best_preds = adjusted_predictions(target_probs, log_ratio, best_tau)
    cm_best = confusion_matrix(best_preds, target_labels, args.num_classes)
    cm_baseline = confusion_matrix(target_probs.argmax(axis=1), target_labels, args.num_classes)
    rep_best = per_class_metrics(cm_best, args.num_classes)
    rep_baseline = per_class_metrics(cm_baseline, args.num_classes)

    write_confusion_csv(out_dir / "confusion_matrix_target_baseline.csv", cm_baseline)
    write_confusion_csv(out_dir / f"confusion_matrix_target_tau{best_tau:g}.csv", cm_best)

    # Per-class CSV (using baseline as "src" slot for diff display)
    with open(out_dir / "per_class_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "class", "support_tgt",
            "acc_baseline", "acc_adjusted", "delta",
            "precision_baseline", "precision_adjusted",
            "f1_baseline", "f1_adjusted",
            "predicted_count_baseline", "predicted_count_adjusted",
            "src_prior_pct", "tgt_prior_pct", "log_ratio",
        ])
        for c in range(args.num_classes):
            b = rep_baseline[c]
            a = rep_best[c]
            w.writerow([
                c, b["support"],
                f"{b['recall']:.4f}", f"{a['recall']:.4f}",
                f"{a['recall'] - b['recall']:+.4f}",
                f"{b['precision']:.4f}", f"{a['precision']:.4f}",
                f"{b['f1']:.4f}", f"{a['f1']:.4f}",
                b["predicted_count"], a["predicted_count"],
                f"{100*src_priors[c]:.4f}", f"{100*tgt_priors[c]:.4f}",
                f"{log_ratio[c]:+.4f}",
            ])

    with open(out_dir / "tau_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tau", "per_face_acc", "per_class_acc"])
        for r in sweep_rows:
            w.writerow([f"{r['tau']:.4f}", f"{r['per_face']:.4f}", f"{r['per_class']:.4f}"])

    # ---------------- Markdown summary -------------------------------------
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Logit-adjustment evaluation (post-hoc, no retraining)\n\n")
        f.write(f"- Checkpoint: `{args.checkpoint}`\n")
        f.write(f"- Source val: `{args.source_path}` ({src_counts.sum():,} faces)\n")
        f.write(f"- Target val: `{args.target_path}` ({tgt_counts.sum():,} faces)\n")
        f.write(f"- Target priors: {'UNIFORM' if args.uniform_target else 'empirical (oracle for POC)'}\n\n")

        f.write("## Tau sweep\n\n")
        f.write("| tau | per-face acc | per-class acc | Δ per-class vs baseline |\n")
        f.write("|-----|--------------|---------------|-------------------------|\n")
        for r in sweep_rows:
            d = r["per_class"] - baseline_pc
            mark = " **(best)**" if r["tau"] == best_tau else ""
            f.write(f"| {r['tau']:.2f} | {r['per_face']:.4f} | {r['per_class']:.4f} | {d:+.4f}{mark} |\n")
        f.write("\n")

        f.write(f"## Best config: tau = {best_tau:g}\n\n")
        baseline_pf_summary, baseline_pc_summary, _ = metrics_summary(
            target_probs.argmax(axis=1), target_labels, args.num_classes
        )
        best_pf, best_pc, _ = metrics_summary(best_preds, target_labels, args.num_classes)
        f.write(f"- Baseline (no adjust): per-face **{baseline_pf_summary:.4f}**, per-class **{baseline_pc_summary:.4f}**\n")
        f.write(f"- Adjusted (tau={best_tau:g}): per-face **{best_pf:.4f}**, per-class **{best_pc:.4f}**\n")
        f.write(f"- Delta: per-face **{best_pf - baseline_pf_summary:+.4f}**, per-class **{best_pc - baseline_pc_summary:+.4f}**\n\n")

        f.write("## Per-class change at best tau\n\n")
        f.write("| class | support | acc baseline | acc adjusted | Δ | top-1 pred share src→tgt |\n")
        f.write("|-------|---------|--------------|--------------|---|---------------------------|\n")
        for c in range(args.num_classes):
            b = rep_baseline[c]
            a = rep_best[c]
            ratio_str = (
                f"src {100*src_priors[c]:.2f}% / tgt {100*tgt_priors[c]:.2f}% (log {log_ratio[c]:+.2f})"
            )
            f.write(
                f"| {c} | {b['support']} | {b['recall']:.3f} | {a['recall']:.3f} | "
                f"{a['recall'] - b['recall']:+.3f} | {ratio_str} |\n"
            )
        f.write("\n")

        f.write("## Reading the result\n\n")
        f.write(
            "- If best per-class acc lands within a few points of the paper's headline "
            "while per-face acc rises ~5–10 pp, the gap was almost entirely **label shift**.\n"
            "- If best tau == 0 (no adjustment helps), it's NOT label shift — the encoder "
            "is producing miscalibrated probabilities even after re-weighting.\n"
            "- A best tau between 0.5 and 1.0 with a clear improvement is the typical signature "
            "of label shift fixed by Saerens-style correction.\n\n"
            "If this works: bake the fix into Stage 1 by retraining with class-balanced loss "
            "(see Option 2 in the previous discussion). Logit adjustment alone works at inference "
            "but doesn't change the encoder's internal feature geometry, so DA on top will "
            "still benefit from a balanced backbone.\n"
        )

    print(f"\nWrote outputs to: {out_dir.resolve()}")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
