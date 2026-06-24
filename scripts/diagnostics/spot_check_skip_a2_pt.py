# -*- coding: utf-8 -*-
"""Spot-check paired full-A2 vs skip-A2 `.pt`: skip omits dense A2; other tensors match."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

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
    # Repo checkout may omit bootstrap_path.py; PYTHONPATH-less runs still common.
    _repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repo))

_COMPARE_ATTRS = (
    "edge_index",
    "node_data",
    "edge_data",
    "face_type",
    "face_area",
    "face_loop",
    "face_adj",
    "label_feature",
    "edge_type",
    "edge_len",
    "edge_ang",
    "edge_conv",
    "node_degree",
    "attn_bias",
    "edge_path",
    "spatial_pos",
)


def _pt_stems(scan: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in scan.rglob("*[0-9].pt"):
        out.setdefault(p.stem, p)
    return out


def _tensor_match(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        if not torch.allclose(a, b, rtol=1e-5, atol=1e-6):
            d = (a.float() - b.float()).abs().max().item()
            raise AssertionError(f"{name}: allclose failed (max abs diff={d})")
    else:
        if not torch.equal(a, b):
            raise AssertionError(f"{name}: tensors differ")


def _check_pair(full_pt: Path, skip_pt: Path) -> None:
    full_d = torch.load(full_pt, map_location="cpu", weights_only=False)
    skip_d = torch.load(skip_pt, map_location="cpu", weights_only=False)

    d2_skip = getattr(skip_d, "d2_distance", None)
    ang_skip = getattr(skip_d, "angle_distance", None)
    if d2_skip is not None or ang_skip is not None:
        raise AssertionError(f"{full_pt.name}: skip graph should omit d2_distance / angle_distance")

    d2_full = getattr(full_d, "d2_distance", None)
    ang_full = getattr(full_d, "angle_distance", None)
    if d2_full is not None:
        mx_full = d2_full.abs().max().item()
        ang_full_mx = (
            ang_full.abs().max().item() if ang_full is not None else float("nan")
        )
        if mx_full == 0.0 and ang_full_mx == 0.0:
            sys.stderr.write(
                f"[warn] Full-A2 tensors are also zero for {full_pt.name}; "
                "comparison is non-diagnostic.\n"
            )

    for name in _COMPARE_ATTRS:
        ta = getattr(full_d, name, None)
        tb = getattr(skip_d, name, None)
        if ta is None or tb is None:
            raise AssertionError(f"missing tensor {name!r} on one side")
        _tensor_match(ta, tb, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full_dir",
        required=True,
        help="Scan root containing full-A2 graphs (same stem as skip tree).",
    )
    ap.add_argument("--skip_dir", required=True, help="Scan root for skip-A2 / zero-A2 graphs.")
    ap.add_argument(
        "--max_checks",
        type=int,
        default=8,
        help="Max stems to validate (pairs present in BOTH directories).",
    )
    args = ap.parse_args()

    full_scan = Path(args.full_dir)
    skip_scan = Path(args.skip_dir)
    if not full_scan.is_dir() or not skip_scan.is_dir():
        raise SystemExit("full_dir and skip_dir must exist as directories.")

    stems_full = _pt_stems(full_scan)
    stems_skip = _pt_stems(skip_scan)
    common = sorted(set(stems_full) & set(stems_skip))

    if not common:
        print("No overlapping stems between full_dir and skip_dir.")
        sys.exit(2)

    picked = common[: args.max_checks]
    print(f"Checking {len(picked)} stem(s): {picked[:5]}{'...' if len(picked) > 5 else ''}")

    ok = 0
    for stem in picked:
        _check_pair(stems_full[stem], stems_skip[stem])
        ok += 1

    print(f"OK: validated {ok} pair(s); skip-A2 omits dense A2; non-A2 tensors match.")
    sys.exit(0)


if __name__ == "__main__":
    main()
