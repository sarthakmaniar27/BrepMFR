# -*- coding: utf-8 -*-
"""Verify every CADSynth split .pt under ``--pt_subdir`` loads with ``torch.load`` (ZIP corruption probe).

Uses the same path resolution as ``data.dataset.CADSynth`` (split list + scan root).

UNC/SMB deployments can throw transient ``OSError(22)`` or partial zip reads when many
workers hit the share; each file is retried a few times before we report failure.

Exit codes: ``0`` all OK; ``2`` one or more files failed load or stems missing.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm


def _load_one(path_str: str) -> tuple[str | None, str | None]:
    """Return (failed_path, repr(err)) on failure, else (None, None).

    UNC/SMB can raise transient ``OSError(22, ...)`` or partial zip reads under
    parallel load; retry a few times before failing.
    """
    for attempt in range(5):
        try:
            torch.load(path_str, map_location="cpu", weights_only=False)
            return None, None
        except Exception as e:
            msg = repr(e).lower()
            transient = isinstance(e, OSError) or (
                "pytorchstreamreader" in msg and "read" in msg
            )
            if transient and attempt < 4:
                time.sleep(0.08 * (2**attempt))
                continue
            return path_str, repr(e)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="CADSynth dataset root (contains train.txt / val.txt or under output/).",
    )
    ap.add_argument(
        "--pt_subdir",
        type=str,
        default=None,
        help="Same as training --pt_subdir (e.g. output/bin_skip_a2). Omit to scan entire root.",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated CADSynth splits to check (subset of train,val,test).",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "Parallel torch.load workers (default 4; higher on fast local disks; "
            "0 = sequential)."
        ),
    )
    ap.add_argument(
        "--stop_after",
        type=int,
        default=0,
        help="Stop after this many load failures (0 = report all).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    rp = str(repo_root)
    if rp not in sys.path:
        sys.path.insert(0, rp)

    from data.dataset import (
        _resolve_dataset_split_list,
        _resolve_graph_pt_scan_root,
    )

    nw = args.num_workers
    if nw is None:
        nw = 4
    if nw < 0:
        ap.error("--num_workers must be >= 0")

    root = Path(args.dataset_path).resolve()
    scan_root = _resolve_graph_pt_scan_root(root, args.pt_subdir)
    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in {"train", "val", "test"}:
            ap.error(f"invalid split '{s}', expected train,val,test")

    stem_set: set[str] = set()
    for s in splits:
        lp = _resolve_dataset_split_list(root, f"{s}.txt")
        with open(lp, "r", encoding="utf-8") as fh:
            for line in fh:
                t = line.strip()
                if t:
                    stem_set.add(t)

    matched: list[Path] = []
    print(f"Scanning: {scan_root}")
    for p in tqdm(
        sorted(scan_root.rglob("*[0-9].pt")),
        desc="rglob stems",
        unit="file",
    ):
        if p.stem in stem_set:
            matched.append(p)

    by_stem: dict[str, list[Path]] = {}
    for p in matched:
        by_stem.setdefault(p.stem, []).append(p)

    dup = sum(max(0, len(v) - 1) for v in by_stem.values())

    print(f"Splits ({', '.join(splits)}): {len(stem_set)} stems in txt lists")
    print(
        f".pt graphs matched under scan root: {len(matched)} "
        f"(unique stems present: {len(by_stem)})"
    )
    missing = stem_set.difference(by_stem.keys())
    if missing:
        print(
            f"ERROR: {len(missing)} split stems have no *.pt under {scan_root}",
            file=sys.stderr,
        )
        preview = sorted(missing)
        for st in preview[:40]:
            print(f"  missing stem: {st}", file=sys.stderr)
        if len(preview) > 40:
            print(f"  ... and {len(preview) - 40} more", file=sys.stderr)
        return 2
    if dup:
        print(
            f"WARN: duplicate .pt paths for the same stem (extra paths): {dup}",
            file=sys.stderr,
        )

    failures: list[tuple[Path, str]] = []
    path_strs = [str(x) for x in matched]

    if nw == 0:
        for ps in tqdm(path_strs, desc="torch.load", unit="graph"):
            fail_p, err = _load_one(ps)
            if fail_p:
                failures.append((Path(fail_p), err))
                if args.stop_after and len(failures) >= args.stop_after:
                    break
    else:
        multiprocessing.freeze_support()
        chunksize = max(8, len(path_strs) // max(64, nw * 32))
        with ProcessPoolExecutor(max_workers=nw) as ex:
            for fail_p, err in tqdm(
                ex.map(_load_one, path_strs, chunksize=chunksize),
                total=len(path_strs),
                desc=f"torch.load x{nw}",
                unit="graph",
            ):
                if fail_p:
                    failures.append((Path(fail_p), err))
                    if args.stop_after and len(failures) >= args.stop_after:
                        break

    if failures:
        print(f"FAILED loads: {len(failures)}", file=sys.stderr)
        for fp, err in failures:
            try:
                sz = fp.stat().st_size if fp.is_file() else -1
            except OSError:
                sz = -1
            print(f"  {fp}\n    size_bytes={sz} err={err}", file=sys.stderr)
        return 2

    print("OK: every matched .pt file loaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
