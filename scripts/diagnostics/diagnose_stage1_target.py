# -*- coding: utf-8 -*-
"""
Stage 1 diagnostic: is the gap to the paper caused by the encoder or the labels?

This script loads a Stage 1 BrepSeg checkpoint and runs pure inference (no
domain adaptation, no training) on:
  - the source val set  (CADSynth)
  - the target val set  (MFCAD++)

It then writes out:
  - confusion_matrix_source.csv
  - confusion_matrix_target.csv
  - per_class_report.csv      (support, recall, precision, f1, top-3 confused)
  - label_distribution.csv    (true vs predicted counts per class)
  - summary.md                (human-readable diagnosis)

The summary buckets each class into one of:
  - "class collapse"      : never predicted on target (model has given up on it)
  - "domain shift"        : high source acc, low target acc (DA target)
  - "encoder weak"        : low on BOTH (Stage 1 itself can't learn it; not a DA problem)
  - "label inconsistency" : suspicious top-1 confusion pattern (e.g. always predicted as one wrong class)

That bucketing is exactly what tells us whether more DA tuning will help, or
whether we have a data/label problem upstream.

Usage (PowerShell, single line):

  python scripts/diagnostics/diagnose_stage1_target.py `
    --checkpoint "C:/Users/D58/Desktop/BrepMFR_PyG/results/BrepMFR/0425/183526/best.ckpt" `
    --source_path "Z:/Experiment6_PyG/source_dataset" `
    --target_path "Z:/Experiment6_PyG/target_dataset" `
    --num_classes 25 `
    --batch_size 32 `
    --num_workers 2 `
    --out_dir "results/diagnostics/stage1_audit"
"""

import argparse
import csv
import importlib.util
import pathlib
import sys
from argparse import Namespace
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Reuse existing project utilities so we go through the SAME data pipeline as
# training. If the migration introduced a subtle preprocessing bug, we want it
# reflected here too — that is precisely what we're auditing.
from data.collator import collator
from data.dataset import _load_pyg_sample, _resolve_dataset_split_list
from models.brepseg_model import BrepSeg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FilelistDataset(Dataset):
    """Single-domain dataset that takes an explicit filelist filename.

    We can't reuse CADSynth directly because its split filenames are hard-coded
    to '<split>.txt', and the transfer experiment uses 's_val.txt' / 't_val.txt'.
    """

    def __init__(self, root_dir: str, filelist: str):
        path = pathlib.Path(root_dir)
        list_path = _resolve_dataset_split_list(path, filelist)
        with open(list_path, "r", encoding="utf-8") as f:
            wanted = set(line.strip() for line in f if line.strip())

        self.file_paths = [p for p in path.rglob("*[0-9].pt") if p.stem in wanted]
        print(f"[{filelist}] resolved {list_path} -> {len(self.file_paths)} files")
        if len(self.file_paths) == 0:
            raise RuntimeError(
                f"No .pt files matched filelist '{filelist}' under {path}. "
                "Check that the converted dataset and split file are present."
            )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return _load_pyg_sample(self.file_paths[idx])

    @staticmethod
    def _collate(batch):
        return collator(batch, multi_hop_max_dist=16, spatial_pos_max=32)


def make_loader(ds: FilelistDataset, batch_size: int, num_workers: int) -> DataLoader:
    kw = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=ds._collate,
        num_workers=num_workers,
        pin_memory=False,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 1
        kw["persistent_workers"] = False
    return DataLoader(ds, **kw)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_stage1_model(checkpoint_path: str, device: torch.device) -> BrepSeg:
    """Load BrepSeg without re-triggering Stage 0 pre_train loading.

    BrepSeg.__init__ unconditionally tries to load `args.pre_train` when set.
    For diagnostic inference we don't want or need that — the Stage 1 weights
    in the checkpoint are already the final values. We override pre_train=None
    before constructing the module, then load state_dict explicitly.
    """
    print(f"\nLoading Stage 1 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})

    if "args" in hparams:
        saved_args = hparams["args"]
        if not isinstance(saved_args, Namespace):
            saved_args = Namespace(**saved_args)
    else:
        saved_args = Namespace(**dict(hparams))

    # Disable side effects of __init__
    saved_args.pre_train = None
    saved_args.warmup_freeze_epochs = 0

    print(f"  num_classes={getattr(saved_args, 'num_classes', '?')} "
          f"dim_node={getattr(saved_args, 'dim_node', '?')} "
          f"n_heads={getattr(saved_args, 'n_heads', '?')} "
          f"n_layers={getattr(saved_args, 'n_layers_encode', '?')}")

    model = BrepSeg(saved_args)
    msg = model.load_state_dict(ckpt["state_dict"], strict=False)
    if msg.missing_keys:
        print(f"  state_dict missing keys: {len(msg.missing_keys)} (showing first 5)")
        for k in msg.missing_keys[:5]:
            print(f"    - {k}")
    if msg.unexpected_keys:
        print(f"  state_dict unexpected keys: {len(msg.unexpected_keys)} (showing first 5)")
        for k in msg.unexpected_keys[:5]:
            print(f"    - {k}")

    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=False)
        else:
            moved[k] = v
    return moved


@torch.no_grad()
def evaluate(
    model: BrepSeg,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (confusion_matrix [C, C], all_preds, all_labels)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    pred_chunks = []
    label_chunks = []

    for batch in tqdm(loader, desc=f"eval {name}", dynamic_ncols=True):
        batch = _move_batch_to_device(batch, device)

        # Forward — copied from BrepSeg.validation_step ----------------------
        node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True)
        node_emb = node_emb[0].permute(1, 0, 2)
        node_emb = node_emb[:, 1:, :]  # drop global virtual node
        padding_mask = batch["padding_mask"]
        node_pos = torch.where(padding_mask == False)
        node_z = node_emb[node_pos]
        num_nodes_per_graph = (~padding_mask).sum(dim=-1)
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        z = model.attention([node_z, graph_z])
        node_seg = model.classifier(z)
        preds = node_seg.argmax(dim=-1).detach().cpu().numpy()
        labels = batch["label_feature"].long().detach().cpu().numpy()

        # Filter to known label range — same convention as test_step ---------
        mask = (labels >= 0) & (labels < num_classes)
        preds = preds[mask]
        labels = labels[mask]

        # Vectorized confusion matrix update
        idx = labels * num_classes + preds
        bins = np.bincount(idx, minlength=num_classes * num_classes)
        cm += bins.reshape(num_classes, num_classes)

        pred_chunks.append(preds)
        label_chunks.append(labels)

    return cm, np.concatenate(pred_chunks), np.concatenate(label_chunks)


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------


def per_class_metrics(cm: np.ndarray, num_classes: int):
    out = []
    for c in range(num_classes):
        support = int(cm[c, :].sum())          # number of true class-c samples
        pred_count = int(cm[:, c].sum())       # number predicted as class c
        tp = int(cm[c, c])
        recall = tp / support if support > 0 else 0.0
        precision = tp / pred_count if pred_count > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Top-3 most common confusions for this true class
        confused = []
        if support > 0:
            row = cm[c, :].copy()
            row[c] = 0  # ignore correct predictions
            order = np.argsort(row)[::-1]
            for idx in order[:3]:
                if row[idx] > 0:
                    confused.append(
                        (int(idx), int(row[idx]), float(row[idx] / support))
                    )

        out.append({
            "class": c,
            "support": support,
            "predicted_count": pred_count,
            "tp": tp,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "top_confused": confused,
        })
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_confusion_csv(path: pathlib.Path, cm: np.ndarray):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["true\\pred"] + [str(i) for i in range(cm.shape[1])]
        w.writerow(header)
        for i, row in enumerate(cm):
            w.writerow([i] + [int(x) for x in row])


def write_per_class_csv(path: pathlib.Path, rep_s, rep_t, num_classes):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "class",
            "support_src", "acc_src",
            "support_tgt", "acc_tgt",
            "precision_tgt", "f1_tgt",
            "predicted_count_tgt",
            "delta_src_minus_tgt",
            "top1_class", "top1_count", "top1_frac",
            "top2_class", "top2_count", "top2_frac",
            "top3_class", "top3_count", "top3_frac",
        ])
        for c in range(num_classes):
            s, t = rep_s[c], rep_t[c]
            row = [
                c,
                s["support"], f"{s['recall']:.4f}",
                t["support"], f"{t['recall']:.4f}",
                f"{t['precision']:.4f}", f"{t['f1']:.4f}",
                t["predicted_count"],
                f"{s['recall'] - t['recall']:+.4f}",
            ]
            for k in range(3):
                if k < len(t["top_confused"]):
                    cls, cnt, frac = t["top_confused"][k]
                    row.extend([cls, cnt, f"{frac:.4f}"])
                else:
                    row.extend(["", "", ""])
            w.writerow(row)


def write_label_distribution_csv(
    path: pathlib.Path,
    labels_s, preds_s, labels_t, preds_t,
    num_classes: int,
):
    src_true = Counter(labels_s.tolist())
    src_pred = Counter(preds_s.tolist())
    tgt_true = Counter(labels_t.tolist())
    tgt_pred = Counter(preds_t.tolist())
    n_src = max(1, len(labels_s))
    n_tgt = max(1, len(labels_t))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "class",
            "true_src", "true_src_pct",
            "pred_src", "pred_src_pct",
            "true_tgt", "true_tgt_pct",
            "pred_tgt", "pred_tgt_pct",
            "pred_to_true_ratio_tgt",
        ])
        for c in range(num_classes):
            ts, ps = src_true.get(c, 0), src_pred.get(c, 0)
            tt, pt = tgt_true.get(c, 0), tgt_pred.get(c, 0)
            ratio = (pt / tt) if tt > 0 else float("nan")
            w.writerow([
                c,
                ts, f"{100*ts/n_src:.3f}",
                ps, f"{100*ps/n_src:.3f}",
                tt, f"{100*tt/n_tgt:.3f}",
                pt, f"{100*pt/n_tgt:.3f}",
                f"{ratio:.4f}",
            ])


def bucket_classes(rep_s, rep_t, tgt_pred_counts: Counter, num_classes: int):
    """Categorise each class to localise the gap to data vs encoder vs DA."""
    collapsed, domain_shift, encoder_weak, label_suspect = [], [], [], []
    for c in range(num_classes):
        s = rep_s[c]
        t = rep_t[c]
        s_acc, t_acc = s["recall"], t["recall"]

        if t["support"] > 0 and tgt_pred_counts.get(c, 0) == 0:
            collapsed.append(c)

        if s_acc >= 0.5 and t_acc < 0.3 and t["support"] > 0:
            domain_shift.append(c)

        if s["support"] > 0 and s_acc < 0.3 and t_acc < 0.3:
            encoder_weak.append(c)

        # "Label suspect": top-1 confusion is overwhelmingly to one wrong class
        # AND that confusion accounts for >70% of all errors. That's the
        # signature of a remap (e.g. target uses label 7 where source uses 8).
        if t["support"] >= 50 and t["top_confused"]:
            top_cls, top_cnt, top_frac = t["top_confused"][0]
            err_total = t["support"] - t["tp"]
            if err_total > 0 and top_cnt / err_total >= 0.7 and top_frac >= 0.3:
                label_suspect.append((c, top_cls, top_frac))

    return collapsed, domain_shift, encoder_weak, label_suspect


def write_summary_md(
    path: pathlib.Path,
    args,
    overall_src_acc: float,
    overall_tgt_acc: float,
    n_src: int,
    n_tgt: int,
    rep_s, rep_t,
    tgt_pred_counts: Counter,
    num_classes: int,
):
    collapsed, domain_shift, encoder_weak, label_suspect = bucket_classes(
        rep_s, rep_t, tgt_pred_counts, num_classes
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Stage 1 diagnostic: encoder vs labels\n\n")
        f.write(f"- Checkpoint: `{args.checkpoint}`\n")
        f.write(f"- Source val: `{args.source_path}` (filelist `{args.source_filelist}`)\n")
        f.write(f"- Target val: `{args.target_path}` (filelist `{args.target_filelist}`)\n")
        f.write(f"- Num classes: {num_classes}\n\n")

        f.write("## Overall (Stage 1 weights, no DA)\n\n")
        f.write(f"- Source per-face acc: **{overall_src_acc:.4f}**  ({n_src:,} faces)\n")
        f.write(f"- Target per-face acc: **{overall_tgt_acc:.4f}**  ({n_tgt:,} faces)\n")
        f.write(f"- Domain gap: **{overall_src_acc - overall_tgt_acc:+.4f}**\n\n")

        f.write("## Class collapse (never predicted on target)\n\n")
        f.write("If a class exists in target val but the model never picks it, ")
        f.write("the classifier head has zero recall for it. DA cannot recover this; ")
        f.write("it points to either Stage 1 weakness or a label-mapping bug.\n\n")
        if collapsed:
            f.write("| class | true count | top-3 wrongly assigned predictions |\n")
            f.write("|-------|------------|--------------------------------------|\n")
            for c in collapsed:
                t = rep_t[c]
                # Where do those targets actually go instead?
                conf_str = ", ".join(
                    f"-> {cls} ({frac:.0%})" for cls, _, frac in t["top_confused"]
                ) or "-"
                f.write(f"| {c} | {t['support']} | {conf_str} |\n")
        else:
            f.write("- (none)\n")
        f.write("\n")

        f.write("## Label-mapping suspects\n\n")
        f.write("Classes where >=70% of all errors collapse onto a single wrong ")
        f.write("class. That is the signature of a **label remap mismatch** ")
        f.write("between source and target — not a learning problem.\n\n")
        if label_suspect:
            f.write("| true class | predicted-as | fraction of true samples |\n")
            f.write("|------------|--------------|--------------------------|\n")
            for c, top, frac in label_suspect:
                f.write(f"| {c} | {top} | {frac:.2%} |\n")
        else:
            f.write("- (none — errors are diffuse, not concentrated on one wrong class)\n")
        f.write("\n")

        f.write("## Domain-shift candidates (high src acc, low tgt acc)\n\n")
        f.write("These are the classes DA *should* fix. ")
        f.write("If this list is large and DA isn't moving them, the discriminator isn't doing its job.\n\n")
        if domain_shift:
            f.write("| class | src_acc | tgt_acc | delta | top-1 confused (frac of true) |\n")
            f.write("|-------|---------|---------|-------|--------------------------------|\n")
            for c in domain_shift:
                s, t = rep_s[c], rep_t[c]
                conf = t["top_confused"]
                conf_str = f"{conf[0][0]} ({conf[0][2]:.2f})" if conf else "-"
                f.write(
                    f"| {c} | {s['recall']:.3f} | {t['recall']:.3f} | "
                    f"{s['recall'] - t['recall']:+.3f} | {conf_str} |\n"
                )
        else:
            f.write("- (none)\n")
        f.write("\n")

        f.write("## Encoder-weak classes (low on BOTH source and target)\n\n")
        f.write("These are NOT DA-fixable. Stage 1 itself never learned them. ")
        f.write("Either the synthetic source dataset doesn't represent them well, ")
        f.write("or class imbalance starved them. DA is the wrong tool here.\n\n")
        if encoder_weak:
            f.write("| class | src_acc | tgt_acc | src_support | tgt_support |\n")
            f.write("|-------|---------|---------|-------------|-------------|\n")
            for c in encoder_weak:
                s, t = rep_s[c], rep_t[c]
                f.write(
                    f"| {c} | {s['recall']:.3f} | {t['recall']:.3f} | "
                    f"{s['support']} | {t['support']} |\n"
                )
        else:
            f.write("- (none)\n")
        f.write("\n")

        f.write("## Per-class detail (target)\n\n")
        f.write("| class | support | tgt_acc | precision | f1 | top-1 confused (frac) | top-2 confused (frac) |\n")
        f.write("|-------|---------|---------|-----------|-----|-----------------------|------------------------|\n")
        for c in range(num_classes):
            t = rep_t[c]
            conf = t["top_confused"]
            c1 = f"{conf[0][0]} ({conf[0][2]:.2f})" if len(conf) >= 1 else "-"
            c2 = f"{conf[1][0]} ({conf[1][2]:.2f})" if len(conf) >= 2 else "-"
            f.write(
                f"| {c} | {t['support']} | {t['recall']:.3f} | "
                f"{t['precision']:.3f} | {t['f1']:.3f} | {c1} | {c2} |\n"
            )
        f.write("\n")

        f.write("## How to read this report\n\n")
        f.write("1. **If `class collapse` or `label-mapping suspects` is non-empty:** ")
        f.write("the data side is broken. No amount of DA hyperparameter tuning will fix it. ")
        f.write("Re-check the DGL->PyG conversion and label mapping for those class IDs.\n")
        f.write("2. **If `domain shift` is the dominant bucket:** DA *should* work; we need ")
        f.write("better adversarial pressure or a stronger discriminator. Iterate on Stage 2.\n")
        f.write("3. **If `encoder weak` is the dominant bucket:** the gap is upstream of ")
        f.write("Stage 2. Either add those classes to the synthetic source set or use class-balanced loss in Stage 1.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser("Stage 1 encoder-vs-labels diagnostic")
    parser.add_argument("--checkpoint", required=True, help="Path to Stage 1 best.ckpt")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--target_path", required=True)
    parser.add_argument("--source_filelist", default="s_val.txt")
    parser.add_argument("--target_filelist", default="t_val.txt")
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--out_dir",
        default="results/diagnostics/stage1_audit",
        help="Where to write CSVs and summary.md",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="If >0, stop after this many batches per domain (smoke-test mode)",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_stage1_model(args.checkpoint, device)

    print("\nBuilding source val dataset")
    src_ds = FilelistDataset(args.source_path, args.source_filelist)
    src_loader = make_loader(src_ds, args.batch_size, args.num_workers)

    print("Building target val dataset")
    tgt_ds = FilelistDataset(args.target_path, args.target_filelist)
    tgt_loader = make_loader(tgt_ds, args.batch_size, args.num_workers)

    if args.max_batches > 0:
        # Wrap loaders to stop early — useful for smoke-testing the script.
        def _truncated(loader, n):
            for i, b in enumerate(loader):
                if i >= n:
                    break
                yield b
        src_loader = _truncated(src_loader, args.max_batches)
        tgt_loader = _truncated(tgt_loader, args.max_batches)

    print("\n=== Evaluating on SOURCE val ===")
    cm_s, preds_s, labels_s = evaluate(model, src_loader, args.num_classes, device, "source")
    print("\n=== Evaluating on TARGET val ===")
    cm_t, preds_t, labels_t = evaluate(model, tgt_loader, args.num_classes, device, "target")

    overall_src_acc = float((preds_s == labels_s).mean()) if len(preds_s) else 0.0
    overall_tgt_acc = float((preds_t == labels_t).mean()) if len(preds_t) else 0.0
    print(f"\nOverall src acc: {overall_src_acc:.4f}  ({len(labels_s):,} faces)")
    print(f"Overall tgt acc: {overall_tgt_acc:.4f}  ({len(labels_t):,} faces)")
    print(f"Domain gap     : {overall_src_acc - overall_tgt_acc:+.4f}")

    rep_s = per_class_metrics(cm_s, args.num_classes)
    rep_t = per_class_metrics(cm_t, args.num_classes)

    write_confusion_csv(out_dir / "confusion_matrix_source.csv", cm_s)
    write_confusion_csv(out_dir / "confusion_matrix_target.csv", cm_t)
    write_per_class_csv(out_dir / "per_class_report.csv", rep_s, rep_t, args.num_classes)
    write_label_distribution_csv(
        out_dir / "label_distribution.csv",
        labels_s, preds_s, labels_t, preds_t,
        args.num_classes,
    )

    tgt_pred_counts = Counter(preds_t.tolist())
    write_summary_md(
        out_dir / "summary.md",
        args,
        overall_src_acc, overall_tgt_acc,
        len(labels_s), len(labels_t),
        rep_s, rep_t,
        tgt_pred_counts,
        args.num_classes,
    )

    print(f"\nDiagnostic outputs written to: {out_dir.resolve()}")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
