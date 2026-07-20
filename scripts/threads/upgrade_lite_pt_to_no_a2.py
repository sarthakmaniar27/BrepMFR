#!/usr/bin/env python3
"""
Upgrade existing lite ``.pt`` graphs to ``no_a2`` (A1+A3) without re-parsing JSON.

This is the fast path for recovering A1/A3 after a lite training run:
  lite/pyg/*.pt  →  compute spatial_pos + edge_path  →  no_a2/pyg/*.pt

Why this is much faster than ``json_to_brepmfr_pyg_optimized.py --inference_profile no_a2``:
  1. Skips JSON parse / UV packing (reuse lite tensors).
  2. Uses the NumPy all-pairs BFS (≈50–80× faster than the old torch cell-write BFS).
  3. Processes many files in a persistent process pool (file-level parallelism).
  4. Does not spawn a ProcessPool per graph for BFS sources.

Typical target: ~40–50k graphs in well under 2 hours on a 16–24 core machine when
reading/writing a local or reasonably fast network path.
"""
from __future__ import annotations

# Cap BLAS/OMP threads before NumPy/Torch init (critical for process pools).
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
from tqdm import tqdm

# Allow `python scripts/threads/upgrade_lite_pt_to_no_a2.py` from repo root.
_REPO = Path(__file__).resolve().parents[2]
_INFERENCE = _REPO / "scripts" / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

from json_to_brepmfr_pyg_optimized import (  # noqa: E402
    _compute_shortest_paths_edge_indices,
    _write_label_json,
    _HAS_CYTHON_BFS,
)


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(obj, path: Path) -> None:
    """Save via temp-then-replace.  Skips the extra local→network copy when the
    destination is on a local drive (not a UNC path)."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    dest_str = str(path)
    is_unc = dest_str.startswith("\\\\") or dest_str.startswith("//")

    if is_unc:
        # Network (SMB) path: write to local temp first, then copy to network.
        fd, tmp_name = tempfile.mkstemp(prefix=f"brepmfr_{os.getpid()}_", suffix=".pt")
        os.close(fd)
        tmp_local = Path(tmp_name)
        dest_tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        try:
            torch.save(obj, tmp_local)
            shutil.copyfile(tmp_local, dest_tmp)
            os.replace(dest_tmp, path)
        finally:
            for p in (tmp_local, dest_tmp):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
    else:
        # Local drive: write temp directly in the target directory (no extra copy).
        dest_tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        try:
            torch.save(obj, dest_tmp)
            os.replace(dest_tmp, path)
        except BaseException:
            try:
                dest_tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def upgrade_one_graph(
    graph,
    *,
    spatial_pos_max: int = 32,
    max_edge_path_len: int = 16,
) -> tuple[object, list[int]]:
    """Attach A1/A3 tensors to a lite (or already-partial) PyG graph; strip A2 if present."""
    n = int(graph.node_data.size(0))
    edge_index = graph.edge_index
    if edge_index is None or edge_index.numel() == 0:
        src = np.empty(0, dtype=np.int64)
        dst = np.empty(0, dtype=np.int64)
    else:
        _ei_np = edge_index.detach().cpu().numpy()
        src = _ei_np[0]
        dst = _ei_np[1]
    e = len(src)

    spatial_pos, edges_path = _compute_shortest_paths_edge_indices(
        src,
        dst,
        n,
        max_dist=max_edge_path_len,
        num_workers=0,
        legacy_bfs=False,
    )
    if n > 1:
        reachable = spatial_pos < 10**8
        max_p = int(spatial_pos[reachable].max().item()) if bool(reachable.any()) else 0
    else:
        max_p = 0
    edges_path = edges_path[:, :, :max_p]
    spatial_pos = spatial_pos.clamp(max=spatial_pos_max)

    if e <= 32767:
        edges_path_i = edges_path.to(torch.int16)
        edge_path_storage = "int16"
    else:
        edges_path_i = edges_path.to(torch.int32)
        edge_path_storage = "int32"

    graph.spatial_pos = spatial_pos.to(torch.uint8)
    graph.edge_path = edges_path_i
    graph.edge_path_storage_dtype = edge_path_storage
    graph.spatial_pos_storage_dtype = "uint8"
    graph.has_a1 = True
    graph.has_a2 = False
    graph.has_a3 = True
    graph.inference_profile = "no_a2"
    graph.has_stored_attn_bias = False

    # Ensure A2 is absent on disk for the no_a2 profile.
    if hasattr(graph, "d2_distance"):
        delattr(graph, "d2_distance")
    if hasattr(graph, "angle_distance"):
        delattr(graph, "angle_distance")
    if hasattr(graph, "attn_bias"):
        # Collator synthesizes attn_bias; drop any stale stored copy.
        delattr(graph, "attn_bias")

    labels = graph.label_feature.detach().cpu().flatten().tolist()
    return graph, labels


_WORKER_SPATIAL_POS_MAX = 32
_WORKER_MAX_EDGE_PATH_LEN = 16
_WORKER_WRITE_LABELS = True


def _worker_init(spatial_pos_max: int, max_edge_path_len: int, write_labels: bool) -> None:
    global _WORKER_SPATIAL_POS_MAX, _WORKER_MAX_EDGE_PATH_LEN, _WORKER_WRITE_LABELS
    # File-level parallelism: one BLAS/OMP thread per process or RAM/CPU explode.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch as _torch

        _torch.set_num_threads(1)
    except Exception:
        pass
    _WORKER_SPATIAL_POS_MAX = int(spatial_pos_max)
    _WORKER_MAX_EDGE_PATH_LEN = int(max_edge_path_len)
    _WORKER_WRITE_LABELS = bool(write_labels)


def _upgrade_one_path(args: tuple[str, str, str]) -> tuple[str, str, float]:
    """Worker entry: (in_pt, out_pt, out_label_or_empty) → (stem, status, seconds)."""
    in_pt_s, out_pt_s, out_label_s = args
    in_pt = Path(in_pt_s)
    out_pt = Path(out_pt_s)
    stem = in_pt.stem
    t0 = time.perf_counter()
    try:
        # Resume-safe: existing destination .pt is kept. Labels are optional for training.
        if out_pt.is_file():
            return stem, "skip", time.perf_counter() - t0

        graph = _load_pt(in_pt)
        graph, labels = upgrade_one_graph(
            graph,
            spatial_pos_max=_WORKER_SPATIAL_POS_MAX,
            max_edge_path_len=_WORKER_MAX_EDGE_PATH_LEN,
        )
        _atomic_torch_save(graph, out_pt)
        if out_label_s and _WORKER_WRITE_LABELS:
            _write_label_json(Path(out_label_s).parent, stem, labels)
        return stem, "ok", time.perf_counter() - t0
    except Exception as exc:
        return stem, f"fail:{exc}", time.perf_counter() - t0


def _split_stems(dataset_root: Path) -> list[str]:
    stems: list[str] = []
    for split in ("train", "val", "test"):
        path = dataset_root / f"{split}.txt"
        if path.is_file():
            stems.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    # Preserve order, drop dups.
    seen: set[str] = set()
    unique: list[str] = []
    for stem in stems:
        if stem not in seen:
            seen.add(stem)
            unique.append(stem)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade lite PyG .pt graphs to no_a2 (A1+A3) with file-level parallelism."
    )
    parser.add_argument("--lite-root", type=Path, required=True, help="Existing lite dataset root (has pyg/).")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination no_a2 dataset root.")
    parser.add_argument("--pt-subdir", default="pyg")
    parser.add_argument("--label-subdir", default="label")
    parser.add_argument("--spatial-pos-max", type=int, default=32)
    parser.add_argument("--max-edge-path-len", type=int, default=16)
    parser.add_argument(
        "--file-workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 4) - 2)),
        help="Persistent process pool size (file-level). Default ≈ CPU-2 capped at 12.",
    )
    parser.add_argument(
        "--write-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write label JSON beside graphs (default: true).",
    )
    parser.add_argument(
        "--copy-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy train/val/test.txt from lite root (default: true).",
    )
    parser.add_argument(
        "--only-split-listed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only convert stems listed in train/val/test.txt (default: true).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests (0 = all).")
    args = parser.parse_args()

    lite_root = args.lite_root.resolve()
    output_root = args.output_root.resolve()
    if lite_root == output_root:
        raise SystemExit("output-root must differ from lite-root")

    in_pyg = lite_root / args.pt_subdir
    if not in_pyg.is_dir():
        raise SystemExit(f"lite pyg dir not found: {in_pyg}")

    out_pyg = output_root / args.pt_subdir
    out_label = output_root / args.label_subdir
    out_pyg.mkdir(parents=True, exist_ok=True)
    if args.write_labels:
        out_label.mkdir(parents=True, exist_ok=True)

    print(f"lite_root={lite_root}", flush=True)
    print(f"output_root={output_root}", flush=True)

    if args.copy_splits:
        for name in ("train.txt", "val.txt", "test.txt"):
            src = lite_root / name
            if not src.is_file():
                raise SystemExit(f"missing split list: {src}")
            shutil.copy2(src, output_root / name)

    if args.only_split_listed:
        stems = _split_stems(lite_root)
        if not stems:
            raise SystemExit(f"no stems found in split lists under {lite_root}")
        if args.limit > 0:
            stems = stems[: args.limit]
        print(f"Building {len(stems):,} jobs from split lists (flat pyg/<stem>.pt)...", flush=True)
        jobs_meta = []
        for stem in stems:
            label_out = str(out_label / f"{stem}.json") if args.write_labels else ""
            jobs_meta.append(
                (
                    str(in_pyg / f"{stem}.pt"),
                    str(out_pyg / f"{stem}.pt"),
                    label_out,
                )
            )
    else:
        print(f"Scanning {in_pyg} recursively for *.pt ...", flush=True)
        paths = sorted(in_pyg.rglob("*.pt"))
        if args.limit > 0:
            paths = paths[: args.limit]
        jobs_meta = []
        for path in paths:
            label_out = str(out_label / f"{path.stem}.json") if args.write_labels else ""
            jobs_meta.append((str(path), str(out_pyg / f"{path.stem}.pt"), label_out))

    n_workers = max(1, int(args.file_workers))
    print(
        f"Upgrading {len(jobs_meta):,} lite graphs -> no_a2 under {output_root}\n"
        f"  file_workers={n_workers}  spatial_pos_max={args.spatial_pos_max}  "
        f"max_edge_path_len={args.max_edge_path_len}  cython_bfs={'yes' if _HAS_CYTHON_BFS else 'no (python fallback)'}",
        flush=True,
    )

    ok = skipped = failed = 0
    times: list[float] = []
    failures: list[str] = []
    wall0 = time.perf_counter()

    def _consume(stem: str, status: str, dt: float) -> None:
        nonlocal ok, skipped, failed
        times.append(dt)
        if status == "ok":
            ok += 1
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
            failures.append(f"{stem}: {status}")
            if len(failures) <= 10:
                print(f"\n[FAIL] {stem}: {status}", flush=True)

    if n_workers == 1:
        _worker_init(args.spatial_pos_max, args.max_edge_path_len, args.write_labels)
        for job in tqdm(jobs_meta, desc="upgrade lite->no_a2", unit="file"):
            _consume(*_upgrade_one_path(job))
    else:
        # Persistent pool + chunked map avoids submitting 40k Future objects at once.
        chunksize = max(1, len(jobs_meta) // (n_workers * 32))
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(args.spatial_pos_max, args.max_edge_path_len, args.write_labels),
        ) as pool:
            for stem, status, dt in tqdm(
                pool.map(_upgrade_one_path, jobs_meta, chunksize=chunksize),
                total=len(jobs_meta),
                desc="upgrade lite->no_a2",
                unit="file",
            ):
                _consume(stem, status, dt)

    wall = time.perf_counter() - wall0
    done = ok + skipped + failed
    rate = (ok / wall) if wall > 0 and ok else 0.0
    print(
        f"\nDone. ok={ok:,} skipped={skipped:,} failed={failed:,} total_jobs={done:,}\n"
        f"Wall {wall:.1f}s ({wall/60:.2f} min)  throughput={rate:.2f} newly-converted files/s",
        flush=True,
    )
    if times:
        print(
            f"Per-job wall time (includes skips): "
            f"avg={sum(times)/len(times):.3f}s  min={min(times):.3f}s  max={max(times):.3f}s",
            flush=True,
        )
    if failures:
        print("First failures:")
        for line in failures[:20]:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
