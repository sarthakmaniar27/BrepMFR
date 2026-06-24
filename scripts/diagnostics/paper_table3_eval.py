# -*- coding: utf-8 -*-
"""
Paper Table 3 (last column CADSynth → MFCAD++): three inference runs with full metrics.

1) Stage 1 on CADSynth **test** (s_test.txt) — in-domain source test.
2) Stage 1 on MFCAD++ **test** (t_test.txt) — "Source only" / no DA on target.
3) Stage 2 on MFCAD++ **test** — domain adaptation on target test split.

Writes per run: summary.md, per_class.csv, confusion_matrix.csv

Usage (activate brep_mfr_pyg first — plain ``python`` may lack torch_geometric):

  conda activate brep_mfr_pyg
  python scripts/diagnostics/paper_table3_eval.py --run_all ^
    --stage1_ckpt results/stage1/ce_weighted_balanced__2026-05-04_163109/best.ckpt ^
    --stage2_ckpt results/stage2/transfer_iwdan_weighted__2026-05-05_134214/best.ckpt ^
    --source_path Z:/Experiment6_PyG/source_dataset ^
    --target_path Z:/Experiment6_PyG/target_dataset
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from argparse import Namespace
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

from data.dataset import TransferDataset, _dataloader_kw
from diagnose_stage1_target import (
    FilelistDataset,
    evaluate,
    load_stage1_model,
    make_loader,
    per_class_metrics,
    write_confusion_csv,
)
from logit_adjust_eval import confusion_matrix as cm_from_preds
from logit_adjust_eval import metrics_summary
from models.transfer_model import DomainAdapt
from stage2_logit_adjust_eval import (
    FACE_LABEL_NAME,
    collect_target_probs_stage2,
    per_class_iou,
    write_per_class_table_csv,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _namespace_clone(ns: Namespace) -> Namespace:
    return Namespace(**vars(ns))


def _resolve_path_if_missing(path_str: str | None, repo_root: Path) -> str | None:
    """Turn relocated-relative paths and basenames into an existing file when possible."""
    if not path_str:
        return path_str
    p = Path(path_str).expanduser()
    if p.is_file():
        return str(p.resolve())
    rel = repo_root / path_str
    if rel.is_file():
        return str(rel.resolve())
    name = Path(path_str).name
    for sub in (
        "artifacts/class_weights/stage1",
        "artifacts/class_weights/stage2_iwdan",
        "results/class_weights",
    ):
        c = repo_root / sub / name
        if c.is_file():
            return str(c.resolve())
    return path_str


def load_domainadapt_for_eval(
    checkpoint_path: str,
    device: torch.device,
    stage1_ckpt: str,
) -> DomainAdapt:
    """Rebuild `DomainAdapt` so stale `pre_train` / IWDAN JSON paths from another layout still work."""
    repo = _repo_root()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    h = ckpt.get("hyper_parameters") or {}
    raw_args = h.get("args")
    if raw_args is None:
        raise ValueError("Checkpoint missing hyper_parameters.args")
    args = _namespace_clone(raw_args) if isinstance(raw_args, Namespace) else Namespace(**dict(raw_args))

    s1 = Path(stage1_ckpt).expanduser()
    if s1.is_file():
        args.pre_train = str(s1.resolve())
    else:
        fixed = _resolve_path_if_missing(getattr(args, "pre_train", None), repo)
        if fixed and Path(fixed).is_file():
            args.pre_train = fixed
        else:
            raise FileNotFoundError(
                f"Cannot resolve Stage 1 pre_train (ckpt had {getattr(raw_args, 'pre_train', None)!r}, "
                f"--stage1_ckpt={stage1_ckpt!r})"
            )

    if getattr(args, "iwdan", False):
        for attr in ("iwdan_source_priors", "iwdan_target_priors"):
            v = getattr(args, attr, None)
            if not v:
                continue
            fixed = _resolve_path_if_missing(v, repo)
            if not Path(fixed).is_file():
                raise FileNotFoundError(
                    f"{attr} JSON not found (saved {v!r}); expected under artifacts/class_weights/"
                )
            setattr(args, attr, fixed)

    model = DomainAdapt(args)
    incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
    if incompatible.missing_keys:
        print(f"  load_state_dict missing_keys (first 5): {incompatible.missing_keys[:5]}")
    if incompatible.unexpected_keys:
        print(f"  load_state_dict unexpected_keys (first 5): {incompatible.unexpected_keys[:5]}")
    model.eval()
    model.to(device)
    return model


def _mean_iou(iou_list: list[float]) -> float:
    arr = np.array(iou_list, dtype=np.float64)
    return float(np.nanmean(arr))


def run_stage1_on_filelist(
    checkpoint: str,
    dataset_root: str,
    filelist: str,
    out_dir: pathlib.Path,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    title: str,
    pt_subdir: str | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"\n{'=' * 60}\n{title}\nDevice: {device}\nCheckpoint: {checkpoint}\n"
        f"Root: {dataset_root}\nFilelist: {filelist}\n{'=' * 60}"
    )

    model = load_stage1_model(checkpoint, device)
    ds = FilelistDataset(dataset_root, filelist, pt_subdir)
    loader = make_loader(ds, batch_size, num_workers)
    if max_batches > 0:

        def _trunc(loader_, n):
            for i, b in enumerate(loader_):
                if i >= n:
                    break
                yield b

        loader = _trunc(loader, max_batches)

    _, preds, labels = evaluate(model, loader, num_classes, device, "eval")

    per_face, mean_per_class, _ = metrics_summary(preds, labels, num_classes)
    cm2 = cm_from_preds(preds, labels, num_classes)
    rep = per_class_metrics(cm2, num_classes)
    ious = per_class_iou(preds, labels, num_classes)
    m_iou = _mean_iou(ious)

    write_confusion_csv(out_dir / "confusion_matrix.csv", cm2)
    write_per_class_table_csv(out_dir / "per_class.csv", rep, ious)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"- Checkpoint: `{checkpoint}`\n")
        f.write(f"- Dataset root: `{dataset_root}`\n")
        f.write(f"- Filelist: `{filelist}`\n")
        if pt_subdir:
            f.write(f"- `--pt_subdir`: `{pt_subdir}`\n")
        f.write(f"- Faces evaluated: **{len(labels):,}**\n\n")
        f.write("## Global metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|------:|\n")
        f.write(f"| Per-face accuracy | {per_face:.4f} |\n")
        f.write(f"| Mean per-class accuracy (mean recall) | {mean_per_class:.4f} |\n")
        f.write(f"| Mean IoU (nan-mean over classes) | {m_iou:.4f} |\n\n")
        f.write("## Per-class accuracy (recall) — all 25 classes\n\n")
        f.write("| ID | Name | Support | Acc (recall) | Precision | F1 | IoU |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for c in range(num_classes):
            r = rep[c]
            iou_s = f"{ious[c]:.4f}" if np.isfinite(ious[c]) else ""
            f.write(
                f"| {c} | {FACE_LABEL_NAME.get(c, '')} | {r['support']} | "
                f"{r['recall']:.4f} | {r['precision']:.4f} | {r['f1']:.4f} | {iou_s} |\n"
            )

    print(
        f"  per_face_acc={per_face:.4f}  mean_per_class={mean_per_class:.4f}  mean_IoU={m_iou:.4f}"
    )
    print(f"  Wrote: {out_dir.resolve()}")


def run_stage2_mfcad_test(
    checkpoint: str,
    stage1_ckpt: str,
    source_path: str,
    target_path: str,
    out_dir: pathlib.Path,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    title: str,
    pt_subdir: str | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"\n{'=' * 60}\n{title}\nDevice: {device}\nCheckpoint: {checkpoint}\n"
        f"split=test (s_test + t_test)\n{'=' * 60}"
    )

    model = load_domainadapt_for_eval(checkpoint, device, stage1_ckpt)

    test_ds = TransferDataset(
        root_dir_source=source_path,
        root_dir_target=target_path,
        split="test",
        random_rotate=False,
        num_class=num_classes,
        open_set=0,
        pt_subdir=pt_subdir,
    )
    dl_kw = _dataloader_kw(num_workers)
    dl_kw["drop_last"] = False
    infer_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_ds._collate,
        **dl_kw,
    )
    if max_batches > 0:

        def _trunc(loader_, n):
            for i, b in enumerate(loader_):
                if i >= n:
                    break
                yield b

        infer_loader = _trunc(infer_loader, max_batches)

    probs, labels = collect_target_probs_stage2(model, infer_loader, num_classes, device)
    preds = probs.argmax(axis=1)

    per_face, mean_per_class, _ = metrics_summary(preds, labels, num_classes)
    cm2 = cm_from_preds(preds, labels, num_classes)
    rep = per_class_metrics(cm2, num_classes)
    ious = per_class_iou(preds, labels, num_classes)
    m_iou = _mean_iou(ious)

    write_confusion_csv(out_dir / "confusion_matrix.csv", cm2)
    write_per_class_table_csv(out_dir / "per_class.csv", rep, ious)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"- Checkpoint: `{checkpoint}`\n")
        if pt_subdir:
            f.write(f"- `--pt_subdir`: `{pt_subdir}` (source + target graph scan roots)\n")
        f.write("- Split: **test** (`t_test.txt` on target; paired `s_test.txt` on source)\n")
        f.write(f"- Faces evaluated: **{len(labels):,}**\n\n")
        f.write("## Global metrics (target only)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|------:|\n")
        f.write(f"| Per-face accuracy | {per_face:.4f} |\n")
        f.write(f"| Mean per-class accuracy (mean recall) | {mean_per_class:.4f} |\n")
        f.write(f"| Mean IoU (nan-mean over classes) | {m_iou:.4f} |\n\n")
        f.write("## Per-class accuracy (recall) — all 25 classes\n\n")
        f.write("| ID | Name | Support | Acc (recall) | Precision | F1 | IoU |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for c in range(num_classes):
            r = rep[c]
            iou_s = f"{ious[c]:.4f}" if np.isfinite(ious[c]) else ""
            f.write(
                f"| {c} | {FACE_LABEL_NAME.get(c, '')} | {r['support']} | "
                f"{r['recall']:.4f} | {r['precision']:.4f} | {r['f1']:.4f} | {iou_s} |\n"
            )

    print(
        f"  per_face_acc={per_face:.4f}  mean_per_class={mean_per_class:.4f}  mean_IoU={m_iou:.4f}"
    )
    print(f"  Wrote: {out_dir.resolve()}")


def main():
    p = argparse.ArgumentParser("Paper Table 3 style eval (CADSynth / MFCAD++)")
    p.add_argument("--stage1_ckpt", required=True, help="Stage 1 BrepSeg best.ckpt")
    p.add_argument("--stage2_ckpt", help="Stage 2 DomainAdapt best.ckpt (for stage2 scenario)")
    p.add_argument("--source_path", required=True, help="CADSynth root (contains s_*.txt, .pt)")
    p.add_argument("--target_path", required=True, help="MFCAD++ root (contains t_*.txt, .pt)")
    p.add_argument("--num_classes", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0, help="Truncate each run (smoke test)")
    p.add_argument(
        "--scenario",
        choices=("stage1_cadsynth_test", "stage1_mfcadpp_test", "stage2_mfcadpp_test"),
        help="Single scenario (omit if --run_all)",
    )
    p.add_argument("--run_all", action="store_true", help="Run all three scenarios")
    p.add_argument(
        "--out_root",
        default="results/diagnostics/paper_table3_cadsynth_to_mfcadpp",
        help="Output directory root",
    )
    p.add_argument(
        "--pt_subdir",
        default=None,
        help=(
            "Relative subgraph dir under source/target roots (e.g. output/bin_skip_a2); "
            "passed to Stage 1 filelist scan and Stage 2 TransferDataset."
        ),
    )
    args = p.parse_args()

    out_root = pathlib.Path(args.out_root)

    def _run(name: str):
        if name == "stage1_cadsynth_test":
            run_stage1_on_filelist(
                args.stage1_ckpt,
                args.source_path,
                "s_test.txt",
                out_root / "01_stage1_cadsynth_test",
                args.num_classes,
                args.batch_size,
                args.num_workers,
                args.max_batches,
                "Table 3 style — Stage 1 on CADSynth test (s_test.txt)",
                args.pt_subdir,
            )
        elif name == "stage1_mfcadpp_test":
            run_stage1_on_filelist(
                args.stage1_ckpt,
                args.target_path,
                "t_test.txt",
                out_root / "02_stage1_mfcadpp_test_source_only",
                args.num_classes,
                args.batch_size,
                args.num_workers,
                args.max_batches,
                "Table 3 style — Stage 1 on MFCAD++ test (source-only / no DA)",
                args.pt_subdir,
            )
        elif name == "stage2_mfcadpp_test":
            if not args.stage2_ckpt:
                raise SystemExit("--stage2_ckpt required for stage2_mfcadpp_test")
            run_stage2_mfcad_test(
                args.stage2_ckpt,
                args.stage1_ckpt,
                args.source_path,
                args.target_path,
                out_root / "03_stage2_mfcadpp_test_domain_adapt",
                args.num_classes,
                args.batch_size,
                args.num_workers,
                args.max_batches,
                "Table 3 style — Stage 2 DA on MFCAD++ test",
                args.pt_subdir,
            )

    if args.run_all:
        if not args.stage2_ckpt:
            raise SystemExit("--run_all requires --stage2_ckpt")
        for s in ("stage1_cadsynth_test", "stage1_mfcadpp_test", "stage2_mfcadpp_test"):
            _run(s)
        idx = out_root / "README.md"
        with open(idx, "w", encoding="utf-8") as f:
            f.write("# Paper Table 3 (CADSynth → MFCAD++ last column) — reproduced metrics\n\n")
            f.write("Three runs:\n\n")
            f.write("1. `01_stage1_cadsynth_test/` — Stage 1 on **CADSynth** test.\n")
            f.write(
                "2. `02_stage1_mfcadpp_test_source_only/` — Stage 1 on **MFCAD++** test "
                "(paper *Source only* analogue).\n"
            )
            f.write(
                "3. `03_stage2_mfcadpp_test_domain_adapt/` — **Stage 2** on **MFCAD++** test "
                "(paper *Domain adaptation* analogue).\n\n"
            )
            f.write("Each folder has `summary.md`, `per_class.csv`, `confusion_matrix.csv`.\n")
        print(f"\nIndex written: {idx.resolve()}")
    elif args.scenario:
        _run(args.scenario)
    else:
        p.error("Pass --run_all or --scenario")


if __name__ == "__main__":
    main()
