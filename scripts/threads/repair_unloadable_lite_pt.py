#!/usr/bin/env python3
"""Rebuild lite ``.pt`` files that torch.load cannot open.

The parallel lite converter can report ``converted=OK`` while leaving a truncated
Torch ZIP on disk. A1/A3 upgrade then fails with ``unexpected pos``. This script
checks every JSON stem, deletes unloadable lite graphs, and reconverts them
sequentially with the legacy pickle serializer.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_REPO = Path(__file__).resolve().parents[2]
_INFERENCE = _REPO / "scripts" / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

import torch
from tqdm import tqdm

from json_to_brepmfr_pyg_optimized import convert_one_json  # noqa: E402


def _loadable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            torch.load(path, map_location="cpu")
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--lite-pyg-dir", required=True, type=Path)
    parser.add_argument("--lite-label-dir", required=True, type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Rebuild at most N unloadable files (0 = all).",
    )
    args = parser.parse_args()

    json_dir = args.json_dir.resolve()
    pyg_dir = args.lite_pyg_dir.resolve()
    label_dir = args.lite_label_dir.resolve()
    pyg_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    jsons = sorted(json_dir.glob("*.json"))
    to_fix: list[Path] = []
    loadable = 0
    for jp in jsons:
        pt = pyg_dir / f"{jp.stem}.pt"
        if _loadable(pt):
            loadable += 1
        else:
            to_fix.append(jp)
    if args.limit > 0:
        to_fix = to_fix[: args.limit]

    print(f"JSON files:          {len(jsons):,}")
    print(f"Loadable lite graphs:{loadable:,}")
    print(f"Unloadable/missing:  {len(to_fix):,}")
    if not to_fix:
        return 0

    ok = failed = 0
    failures: list[str] = []
    for jp in tqdm(to_fix, desc="rebuild lite", unit="file"):
        pt = pyg_dir / f"{jp.stem}.pt"
        lab = label_dir / f"{jp.stem}.json"
        pt.unlink(missing_ok=True)
        lab.unlink(missing_ok=True)
        for tmp in pyg_dir.glob(f"{jp.stem}.pt*.tmp"):
            tmp.unlink(missing_ok=True)
        try:
            convert_one_json(
                jp,
                pyg_dir,
                label_dir,
                spatial_pos_max=32,
                inference_profile="lite",
                max_edge_path_len=16,
                shortest_path_workers=0,
            )
            if not _loadable(pt):
                raise RuntimeError("wrote a lite graph that torch.load cannot open")
            ok += 1
        except Exception as exc:
            failed += 1
            failures.append(f"{jp.stem}: {exc}")
            pt.unlink(missing_ok=True)

    print(f"\nRebuilt ok={ok:,} failed={failed:,}")
    for line in failures[:20]:
        print(f"  - {line}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
