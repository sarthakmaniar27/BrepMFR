#!/usr/bin/env python3
"""Convert a JSON folder to lite PyG graphs with file-level parallelism."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
_INFERENCE = _REPO / "scripts" / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

from json_to_brepmfr_pyg_optimized import convert_one_json  # noqa: E402


def _pt_is_loadable(path: Path) -> bool:
    try:
        import torch

        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            torch.load(path, map_location="cpu")
        return True
    except Exception:
        return False


def _worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def _convert_one(job: tuple[str, str, str]) -> tuple[str, str, float]:
    json_path_s, pt_out_dir_s, label_out_dir_s = job
    json_path = Path(json_path_s)
    pt_out_dir = Path(pt_out_dir_s)
    label_out_dir = Path(label_out_dir_s)
    out_pt = pt_out_dir / f"{json_path.stem}.pt"
    out_label = label_out_dir / f"{json_path.stem}.json"
    started = time.perf_counter()
    try:
        if out_pt.is_file() and out_label.is_file():
            if _pt_is_loadable(out_pt):
                return json_path.stem, "skip", time.perf_counter() - started
            # Truncated ZIP from a killed/crashed worker. Remove it so this
            # stem is converted again instead of poisoning A1/A3 upgrade.
            out_pt.unlink(missing_ok=True)
            out_label.unlink(missing_ok=True)
        convert_one_json(
            json_path,
            pt_out_dir,
            label_out_dir,
            spatial_pos_max=32,
            inference_profile="lite",
            max_edge_path_len=16,
            shortest_path_workers=0,
        )
        return json_path.stem, "ok", time.perf_counter() - started
    except Exception as exc:
        return json_path.stem, f"fail:{exc}", time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--pt-out-dir", required=True, type=Path)
    parser.add_argument("--label-out-dir", required=True, type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4) - 2)),
        help="Parallel JSON conversion workers (default: CPU-2, capped at 8).",
    )
    args = parser.parse_args()

    json_dir = args.json_dir.resolve()
    if not json_dir.is_dir():
        raise SystemExit(f"JSON directory not found: {json_dir}")
    paths = sorted(json_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No top-level JSON files found in: {json_dir}")

    pt_out_dir = args.pt_out_dir.resolve()
    label_out_dir = args.label_out_dir.resolve()
    pt_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    jobs = [(str(path), str(pt_out_dir), str(label_out_dir)) for path in paths]

    print(f"JSON files:      {len(paths):,}")
    print(f"Lite PT output:  {pt_out_dir}")
    print(f"Label output:    {label_out_dir}")
    print(f"File workers:    {workers}")

    ok = skipped = failed = 0
    failures: list[str] = []
    started = time.perf_counter()

    def consume(stem: str, status: str, _seconds: float) -> None:
        nonlocal ok, skipped, failed
        if status == "ok":
            ok += 1
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
            failures.append(f"{stem}: {status}")

    if workers == 1:
        _worker_init()
        for job in tqdm(jobs, desc="JSON -> lite", unit="file"):
            consume(*_convert_one(job))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            for result in tqdm(
                pool.map(_convert_one, jobs, chunksize=4),
                total=len(jobs),
                desc="JSON -> lite",
                unit="file",
            ):
                consume(*result)

    elapsed = time.perf_counter() - started
    print(
        f"\nDone. converted={ok:,} skipped={skipped:,} failed={failed:,} "
        f"total={len(jobs):,} wall={elapsed / 60.0:.2f} min"
    )
    if failures:
        print("First conversion failures:")
        for failure in failures[:20]:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
