#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load ``.pt`` graphs from JSON conversion and run ``collator`` on one mini-batch.

Writes only logs under optional ``--out_log`` (default Experiment_test/logs).

Example:
  conda activate brep_mfr_pyg
  cd BrepMFR_PyG
  python scripts/diagnostics/json_pyg_collator_smoke.py \\
    --root Z:/Experiment_test/out_pyg \\
    --batch_size 2
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
from pathlib import Path

import torch

_script = Path(__file__).resolve()
for _ancestor in _script.parents:
    _bst = _ancestor / "bootstrap_path.py"
    if _bst.is_file():
        _spec = importlib.util.spec_from_file_location("__brepmfr_bootstrap", _bst)
        _bm = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_bm)
        _bm.setup(str(_script))
        break
else:
    raise RuntimeError("bootstrap_path.py not found")

from data.collator import collator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"Z:\Experiment_test\out_pyg"),
        help="Folder containing *.pt produced by json_to_brepmfr_pyg",
    )
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument(
        "--split_file",
        type=Path,
        default=None,
        help="Explicit train.txt (default writes root/train.txt covering all *[0-9].pt)",
    )
    ap.add_argument(
        "--out_log",
        type=Path,
        default=None,
        help="Log path (under Experiment_test/logs by default)",
    )
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        raise FileNotFoundError(root)

    pt_files = sorted(root.glob("*[0-9].pt"))
    if not pt_files:
        raise FileNotFoundError(f"No *[0-9].pt under {root}")

    if args.split_file is None:
        split_path = root / "train_smoke.txt"
        split_path.write_text("\n".join(p.stem for p in pt_files) + "\n", encoding="utf-8")
    else:
        split_path = args.split_file

    stems = [
        ln.strip()
        for ln in split_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    objs = []
    stem_set = frozenset(stems)
    for p in pt_files:
        if p.stem not in stem_set:
            continue
        objs.append(torch.load(p, map_location="cpu", weights_only=False))
        if len(objs) >= args.batch_size:
            break

    if len(objs) < args.batch_size:
        raise RuntimeError(f"Need at least batch_size graphs; got {len(objs)}")

    batched = collator(objs[: args.batch_size], multi_hop_max_dist=16, spatial_pos_max=32)
    keys_expected = {"padding_mask", "edge_padding_mask", "edge_index", "node_data", "edge_data", "label_feature"}
    missing = keys_expected - set(batched.keys())
    assert not missing, f"collator missing keys: {missing}"

    log_txt = []
    log_txt.append("json_pyg_collator_smoke OK")
    log_txt.append(f"  root: {root.resolve()}")
    log_txt.append(f"  graphs in batch: {len(objs[: args.batch_size])}")
    log_txt.append(f"  batched keys: {sorted(batched.keys())}")
    for k in sorted(keys_expected):
        v = batched[k]
        if hasattr(v, "shape"):
            log_txt.append(f"  {k} shape: {tuple(v.shape)} dtype={v.dtype}")
        else:
            log_txt.append(f"  {k}: {type(v)}")

    txt = "\n".join(log_txt) + "\n"
    print(txt)
    out = args.out_log or Path(r"Z:\Experiment_test\logs\collator_smoke.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
