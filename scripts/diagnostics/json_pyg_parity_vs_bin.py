#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify JSON→PyG ingest matches legacy **``.bin``** loaded through ``bin_to_pyg``. Reference
``.bin`` files must be in the **post-``append_angle_7th_channel``** state—the same lineage as JSON
converted with ``json_to_brepmfr_pyg.py`` (UV channel 7 wrapped). Requires ``dgl`` only to ``load_graphs``.

**Roots:** ``Z:\\Experiment6`` produces reference ``*.bin``; ``Z:\\Experiment6_PyG`` holds matching
canonical ``*.pt``; ``Z:\\Experiment_test`` should use the **same** JSON filenames as ``Experiment6``
(typically ``input_json/`` mirrors ``Experiment6/*/input/`` while ``ref_bin/`` mirrors
``Experiment6/*/output/bin/``) so parity is apples-to-apples.

Default parity root: ``Z:\\Experiment_test`` (writable; only this tree needs copies/symlinks).

Example:
  conda activate brep_mfr
  python scripts/diagnostics/json_pyg_parity_vs_bin.py \\
    --root Z:/Experiment_test \\
    --patterns "part1_*.json"
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

from data.dgl_bin_to_pyg import bin_to_pyg  # noqa: E402

import json_to_brepmfr_pyg as j2p  # noqa: E402

PYG_ATTRS_ORDER = j2p.PYG_ATTRS_ORDER
build_pyg_from_json_path = j2p.build_pyg_from_json_path


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _compare_tensor(name: str, ref: torch.Tensor, cand: torch.Tensor, rtol: float, atol: float) -> list[str]:
    errs = []
    if ref.shape != cand.shape:
        errs.append(f"{name}: shape mismatch ref={tuple(ref.shape)} cand={tuple(cand.shape)}")
        return errs
    if ref.dtype != cand.dtype:
        errs.append(
            f"{name}: dtype mismatch ref={_dtype_name(ref)} cand={_dtype_name(cand)}"
        )
    if errs:
        return errs
    if ref.dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ):
        if not torch.allclose(ref, cand, rtol=rtol, atol=atol):
            d = (ref.float() - cand.float()).abs()
            errs.append(f"{name}: float mismatch max_abs={d.max().item():.6g}")
    else:
        if not torch.equal(ref, cand):
            ne = torch.sum(ref != cand).item()
            errs.append(f"{name}: int mismatch mismatched_elems={ne}")
    return errs


def compare_two_pyg(ref, cand, rtol: float, atol: float) -> list[str]:
    """Compare BrepMFR ``Data`` objects (custom attrs live in PyG `_store`, not ``vars()``)."""
    all_errors: list[str] = []
    if ref.data_id != cand.data_id:
        all_errors.append(f"data_id mismatch ref={ref.data_id} cand={cand.data_id}")
    for name in PYG_ATTRS_ORDER:
        if name == "data_id":
            continue
        if not hasattr(ref, name):
            all_errors.append(f"ref missing attribute `{name}`")
            continue
        if not hasattr(cand, name):
            all_errors.append(f"cand missing attribute `{name}`")
            continue
        rr = getattr(ref, name)
        cc = getattr(cand, name)
        all_errors.extend(_compare_tensor(name, rr, cc, rtol=rtol, atol=atol))
    return all_errors


def main():
    ap = argparse.ArgumentParser("Parity JSON→PyG vs reference DGL .bin → PyG")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"Z:\Experiment_test"),
        help="Test root containing ref_bin/, input_json/ (writable log area optional)",
    )
    ap.add_argument(
        "--json_dir",
        type=Path,
        default=None,
        help="Explicit JSON folder (default root/input_json)",
    )
    ap.add_argument(
        "--bin_dir",
        type=Path,
        default=None,
        help="Explicit .bin folder (default root/ref_bin)",
    )
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=["*.json"],
        help='Glob stem patterns under json_dir e.g. "model_*.json"',
    )
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument(
        "--write_log",
        type=Path,
        default=None,
        help="Append summary markdown (under Experiment_test preferred)",
    )
    args = ap.parse_args()

    root = args.root
    jdir = args.json_dir or root / "input_json"
    bdir = args.bin_dir or root / "ref_bin"

    if not jdir.is_dir():
        raise FileNotFoundError(f"Missing json_dir {jdir}")
    if not bdir.is_dir():
        raise FileNotFoundError(f"Missing bin_dir {bdir}")

    json_paths: list[Path] = []
    for pat in args.patterns:
        json_paths.extend(sorted(jdir.glob(pat)))
    json_paths = list(dict.fromkeys(json_paths))

    print(f"Paired parity: JSON from {jdir}\nRefs from {bdir}\nFiles: {len(json_paths)}\n")

    fail = 0
    rows = []
    for jp in json_paths:
        stem = jp.stem
        bin_p = bdir / f"{stem}.bin"
        if not bin_p.is_file():
            print(f"[SKIP no bin] {jp.name}")
            rows.append((stem, "skip_no_bin", []))
            continue
        try:
            ref = bin_to_pyg(bin_p)
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", "") != "dgl":
                raise
            raise SystemExit(
                "Parity loads reference `.bin` via `dgl` (inside `data.dgl_bin_to_pyg.bin_to_pyg`). "
                "Run under `conda activate brep_mfr`, or install a `dgl` build compatible "
                "with your `.bin` producer."
            ) from exc
        cand = build_pyg_from_json_path(jp, spatial_pos_max=32)
        errs = compare_two_pyg(ref, cand, rtol=args.rtol, atol=args.atol)
        if errs:
            fail += 1
            print(f"[FAIL] {stem}")
            for e in errs[:40]:
                print(f"       {e}")
            if len(errs) > 40:
                print(f"       ... +{len(errs) - 40} lines")
            rows.append((stem, "fail", errs))
        else:
            print(f"[OK]   {stem}")
            rows.append((stem, "ok", []))

    lines = []
    if args.write_log:
        args.write_log.parent.mkdir(parents=True, exist_ok=True)
        lines.append("| stem | status | issues |\n|------|------|-------|")
        for stem, status, errs in rows:
            if status == "ok":
                lines.append(f"| {stem} | ok | — |")
            elif status == "skip_no_bin":
                lines.append(f"| {stem} | skip_no_bin | |")
            else:
                snippet = errs[0][:200].replace("|", "\\|") if errs else ""
                lines.append(f"| {stem} | FAIL | {snippet} ... |")

    summary = (
        f"\nSummary: {len(json_paths) - fail} matched, {fail} failed, "
        f"{sum(1 for r in rows if r[1] == 'skip_no_bin')} skipped (no bin)\n"
    )
    print(summary)
    if args.write_log:
        Path(args.write_log).write_text(
            "## json_pyg_parity_vs_bin\n\n" + "\n".join(lines) + "\n\n" + summary.strip(),
            encoding="utf-8",
        )
        print(f"Wrote {args.write_log}")

    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
