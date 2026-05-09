#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two ``torch.save``\'d PyG ``Data`` files (same field order as parity tooling).

Typical pairs:
  - ``Experiment6_PyG/.../*.pt`` (from ``convert_dgl_bins_to_pyg`` / ``bin_to_pyg``)
  - ``your_out/*.pt`` from ``json_to_brepmfr_pyg.py``

Exit 0 iff all tensors match within tolerances.

Example:
  conda activate brep_mfr_pyg
  python scripts/diagnostics/compare_pt_vs_pt.py \\
    --ref Z:/Experiment6_PyG/source_dataset/output/bin/00000000_101.pt \\
    --cand Z:/Experiment_test/out_pyg/00000000_101.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

_py = Path(__file__).resolve()
_REPO = None
for _ancestor in _py.parents:
    _bst = _ancestor / "bootstrap_path.py"
    if _bst.is_file():
        _spec = importlib.util.spec_from_file_location("__brepmfr_bootstrap", _bst)
        _bm = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_bm)
        _REPO = _bm.setup(str(_py))
        break
else:
    raise RuntimeError("bootstrap_path.py not found")

_inference = _REPO / "scripts" / "inference"
if str(_inference) not in sys.path:
    sys.path.insert(0, str(_inference))

import json_to_brepmfr_pyg as j2p  # noqa: E402

PYG_ATTRS_ORDER = j2p.PYG_ATTRS_ORDER


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _compare_tensor(name: str, ref: torch.Tensor, cand: torch.Tensor, rtol: float, atol: float) -> list[str]:
    errs: list[str] = []
    if ref.shape != cand.shape:
        errs.append(f"{name}: shape mismatch ref={tuple(ref.shape)} cand={tuple(cand.shape)}")
        return errs
    if ref.dtype != cand.dtype:
        errs.append(f"{name}: dtype mismatch ref={_dtype_name(ref)} cand={_dtype_name(cand)}")
        return errs
    if ref.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        if not torch.allclose(ref, cand, rtol=rtol, atol=atol):
            d = (ref.float() - cand.float()).abs()
            errs.append(f"{name}: float mismatch max_abs={d.max().item():.6g}")
    else:
        if not torch.equal(ref, cand):
            ne = torch.sum(ref != cand).item()
            errs.append(f"{name}: int mismatch mismatched_elems={ne}")
    return errs


def compare_two_pyg(ref, cand, rtol: float, atol: float) -> list[str]:
    errs: list[str] = []
    if ref.data_id != cand.data_id:
        errs.append(f"data_id mismatch ref={ref.data_id} cand={cand.data_id}")
    for name in PYG_ATTRS_ORDER:
        if name == "data_id":
            continue
        if not hasattr(ref, name):
            errs.append(f"ref missing attribute `{name}`")
            continue
        if not hasattr(cand, name):
            errs.append(f"cand missing attribute `{name}`")
            continue
        rr = getattr(ref, name)
        cc = getattr(cand, name)
        errs.extend(_compare_tensor(name, rr, cc, rtol=rtol, atol=atol))
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    cand = torch.load(args.cand, map_location="cpu", weights_only=False)

    errs = compare_two_pyg(ref, cand, rtol=args.rtol, atol=args.atol)
    if errs:
        print(f"Compared {args.ref}\nvs     {args.cand}\n")
        for line in errs:
            print(line)
        raise SystemExit(1)
    print("OK: all fields match")


if __name__ == "__main__":
    main()
