#!/usr/bin/env python3
"""Remap labels only for JSONs that do not yet have a no_a2 PyG graph."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

try:
    import orjson
except ImportError:
    orjson = None

from repair_json_face_labels import load_remap


_WORKER_REMAP: dict[int, int] = {}


def _worker_init(remap: dict[int, int]) -> None:
    global _WORKER_REMAP
    _WORKER_REMAP = remap


def _loads(path: Path) -> dict:
    raw = path.read_bytes()
    return orjson.loads(raw) if orjson is not None else json.loads(raw)


def _dumps(data: dict) -> bytes:
    if orjson is not None:
        return orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _labels(data: dict) -> list[int]:
    values: list[int] = []
    for face in data.get("faces") or []:
        if not isinstance(face, dict) or "label" not in face:
            continue
        values.append(int(face["label"]))
    return values


def _scan_one(path_string: str) -> tuple[dict[int, int], dict[int, int], int, str]:
    path = Path(path_string)
    try:
        values = _labels(_loads(path))
        raw = Counter(values)
        unknown = Counter(value for value in values if value not in _WORKER_REMAP)
        return dict(raw), dict(unknown), len(values), ""
    except Exception as exc:
        return {}, {}, 0, f"{path.name}: {exc}"


def _rewrite_one(
    path_string: str,
) -> tuple[int, int, dict[int, int], str]:
    path = Path(path_string)
    temp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        data = _loads(path)
        rewritten = 0
        post = Counter()
        for face in data.get("faces") or []:
            if not isinstance(face, dict) or "label" not in face:
                continue
            old = int(face["label"])
            if old not in _WORKER_REMAP:
                raise ValueError(f"unmapped label {old}")
            new = _WORKER_REMAP[old]
            post[new] += 1
            if new != old:
                face["label"] = new
                rewritten += 1
        if rewritten:
            temp_path.write_bytes(_dumps(data))
            os.replace(temp_path, path)
        return int(rewritten > 0), rewritten, dict(post), ""
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return 0, 0, {}, f"{path.name}: {exc}"


def _run_parallel(paths: list[Path], function, workers: int, description: str):
    strings = [str(path) for path in paths]
    if workers <= 1:
        return [
            function(path)
            for path in tqdm(strings, desc=description, unit="file", dynamic_ncols=True)
        ]
    chunksize = max(1, len(strings) // (workers * 32))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(_WORKER_REMAP,),
    ) as pool:
        return list(
            tqdm(
                pool.map(function, strings, chunksize=chunksize),
                total=len(strings),
                desc=description,
                unit="file",
                dynamic_ncols=True,
            )
        )


def _scan(
    paths: list[Path],
    remap: dict[int, int],
    workers: int,
) -> tuple[Counter, Counter, int, list[str]]:
    _worker_init(remap)
    raw: Counter = Counter()
    unknown: Counter = Counter()
    total = 0
    errors: list[str] = []
    for item_raw, item_unknown, item_total, error in _run_parallel(
        paths, _scan_one, workers, "Audit labels"
    ):
        raw.update(item_raw)
        unknown.update(item_unknown)
        total += item_total
        if error:
            errors.append(error)
    return raw, unknown, total, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--pyg-dir", required=True, type=Path)
    parser.add_argument("--map-json", required=True, type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4) - 2)),
        help="Parallel JSON audit/remap workers (default: CPU-2, capped at 8).",
    )
    parser.add_argument(
        "--require-no-missing",
        action="store_true",
        help="Exit non-zero when any JSON still lacks a matching .pt.",
    )
    parser.add_argument(
        "--skip-prewrite-audit",
        action="store_true",
        help="With --yes-write, trust a just-completed dry run and avoid parsing twice.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--yes-write", action="store_true")
    args = parser.parse_args()

    json_dir = args.json_dir.resolve()
    pyg_dir = args.pyg_dir.resolve()
    map_path = args.map_json.resolve()
    if not json_dir.is_dir():
        raise SystemExit(f"JSON directory not found: {json_dir}")
    if not pyg_dir.is_dir():
        raise SystemExit(f"PyG directory not found: {pyg_dir}")
    if not map_path.is_file():
        raise SystemExit(f"Map file not found: {map_path}")
    if args.skip_prewrite_audit and not args.yes_write:
        raise SystemExit("--skip-prewrite-audit requires --yes-write")

    print(f"Indexing top-level JSON files under: {json_dir}", flush=True)
    json_paths = sorted(json_dir.glob("*.json"))
    print(f"Found {len(json_paths):,} JSON files; indexing existing .pt stems...", flush=True)
    pt_paths = sorted(pyg_dir.glob("*.pt"))
    if not pt_paths:
        pt_paths = sorted(pyg_dir.rglob("*.pt"))
    existing_stems = {path.stem for path in pt_paths}
    missing_paths = [path for path in json_paths if path.stem not in existing_stems]

    remap = load_remap(map_path)
    effective_remap = dict(remap)
    for target in remap.values():
        effective_remap.setdefault(int(target), int(target))

    print(f"Existing no_a2 .pt stems: {len(existing_stems):,}")
    print(f"Missing JSONs selected:   {len(missing_paths):,}")
    print(f"Workers:                  {int(args.workers)}")
    print(f"JSON parser:              {'orjson' if orjson is not None else 'stdlib json'}")
    if args.require_no_missing and missing_paths:
        print("Coverage check failed; first missing stems:")
        for path in missing_paths[:20]:
            print(f"  {path.stem}")
        return 1
    if not missing_paths:
        print("Nothing to remap; every JSON already has a no_a2 graph.")
        return 0

    workers = max(1, int(args.workers))
    if args.dry_run or not args.skip_prewrite_audit:
        raw, unknown, total_faces, errors = _scan(missing_paths, effective_remap, workers)
        print(f"\nSelected faces with labels: {total_faces:,}")
        print(f"Selected raw-label counts: {dict(sorted(raw.items()))}")
        if errors:
            print("JSON read/parse failures:")
            for error in errors[:20]:
                print(f"  {error}")
            return 1
        if unknown:
            print(f"Unknown labels: {dict(sorted(unknown.items()))}")
            return 1
        if args.dry_run:
            print("Dry run passed; no files were modified.")
            return 0

    _worker_init(effective_remap)
    modified = rewritten = 0
    post: Counter = Counter()
    errors: list[str] = []
    for item_modified, item_rewritten, item_post, error in _run_parallel(
        missing_paths, _rewrite_one, workers, "Remap labels"
    ):
        modified += item_modified
        rewritten += item_rewritten
        post.update(item_post)
        if error:
            errors.append(error)
    if errors:
        print("Remap failures:")
        for error in errors[:20]:
            print(f"  {error}")
        return 1
    print(
        f"Remap complete: modified {modified:,} JSON files; "
        f"rewrote {rewritten:,} labels; post-labels={dict(sorted(post.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
