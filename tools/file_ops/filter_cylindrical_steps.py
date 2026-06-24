#!/usr/bin/env python3
"""
Filter STEP files whose companion JSON has at least one cylindrical face (z == 1).

Reads CadSynth, MFCAD++, and ABC JSON corpora (read-only), maps each JSON stem to its
STEP basename by stripping a trailing _### suffix (e.g. 00000000_101.json → 00000000),
then copies matching originals into a threads dataset STEP folder.

CadSynth is processed first, then MFCAD++, then ABC (when enabled). Source trees are
never modified.

ABC layout (defaults):

  JSON  : D:\\abc\\sw_jsons\\*.json  (all STEP chunks in one folder)
  STEP  : D:\\abc\\step_extracted\\abc_0000_step_v00\\, abc_0001_step_v00\\, ...
  OUT   : Y:\\threads_dataset\\abc\\step

Use ``--only abc`` to run ABC only. STEP indexing under ``--abc-step-src`` is
recursive by default so all chunk subfolders are searched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

CYLINDER_FACE_TYPE = 1
_JSON_SUFFIX_RE = re.compile(r"^(.+)_\d+$")
_STEP_EXTENSIONS = (".stp", ".step", ".STP", ".STEP")


@dataclass
class DatasetStats:
    json_files_scanned: int = 0
    json_parse_errors: int = 0
    json_missing_faces: int = 0
    json_with_cylinder: int = 0
    unique_step_ids_with_cylinder: int = 0
    steps_copied: int = 0
    steps_missing: int = 0
    steps_copy_errors: int = 0
    steps_already_present: int = 0
    missing_step_ids: list[str] = field(default_factory=list)


def strip_json_suffix(stem: str) -> str:
    """00000000_101 → 00000000; mfcad_3236_102 → mfcad_3236."""
    match = _JSON_SUFFIX_RE.match(stem)
    return match.group(1) if match else stem


def has_cylindrical_face(json_path: Path) -> tuple[bool, str | None]:
    """
    Return (has_cylinder, error_message).
    error_message is one of: None, 'parse', 'no_faces'.
    """
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False, "parse"

    faces = data.get("faces")
    if not faces:
        return False, "no_faces"

    for face in faces:
        if not isinstance(face, dict):
            continue
        z = face.get("z")
        if z == CYLINDER_FACE_TYPE or z == float(CYLINDER_FACE_TYPE):
            return True, None
    return False, None


def build_step_index(step_dir: Path, recursive: bool) -> dict[str, Path]:
    """Map STEP stem → first matching path (case-insensitive extension)."""
    index: dict[str, Path] = {}
    iterator = step_dir.rglob("*") if recursive else step_dir.iterdir()
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix not in _STEP_EXTENSIONS:
            continue
        stem = path.stem
        if stem not in index:
            index[stem] = path
    return index


def build_step_index_multi(
    step_dirs: list[Path],
    recursive: bool,
) -> dict[str, Path]:
    """Merge STEP indices from several roots (e.g. ABC chunk folders). First path wins."""
    index: dict[str, Path] = {}
    for step_dir in step_dirs:
        sub = build_step_index(step_dir, recursive=recursive)
        for stem, path in sub.items():
            if stem not in index:
                index[stem] = path
    return index


def discover_abc_step_dirs(step_root: Path) -> list[Path]:
    """
    Return STEP source directories under *step_root*.

    If *step_root* contains chunk subfolders (``abc_*_step_v*``), use those;
    otherwise treat *step_root* itself as the only STEP directory.
    """
    if not step_root.is_dir():
        raise FileNotFoundError(f"ABC: STEP root not found: {step_root}")
    chunks = sorted(
        p
        for p in step_root.iterdir()
        if p.is_dir() and p.name.lower().startswith("abc_") and "step" in p.name.lower()
    )
    if chunks:
        return chunks
    return [step_root]


def _scan_one_json(json_path_str: str) -> tuple[str | None, str | None]:
    """Worker: return (step_id, error) where error is 'parse' | 'no_faces' | None."""
    path = Path(json_path_str)
    has_cyl, err = has_cylindrical_face(path)
    if err:
        return None, err
    if has_cyl:
        return strip_json_suffix(path.stem), None
    return None, None


def scan_json_directory(
    json_dir: Path,
    desc: str,
    workers: int,
    *,
    recursive: bool = False,
) -> tuple[set[str], DatasetStats]:
    """Scan ``*.json`` under *json_dir*; return unique STEP stems with z == 1."""
    stats = DatasetStats()
    if recursive:
        json_paths = sorted(json_dir.rglob("*.json"))
    else:
        json_paths = sorted(json_dir.glob("*.json"))
    cylinder_by_step: dict[str, bool] = defaultdict(bool)

    if workers <= 1:
        iterator = (
            _scan_one_json(str(p))
            for p in json_paths
        )
        progress = tqdm(json_paths, desc=desc, unit="json", file=sys.stderr)
        for json_path, (step_id, err) in zip(progress, iterator):
            stats.json_files_scanned += 1
            if err == "parse":
                stats.json_parse_errors += 1
            elif err == "no_faces":
                stats.json_missing_faces += 1
            elif step_id is not None:
                stats.json_with_cylinder += 1
                cylinder_by_step[step_id] = True
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_scan_one_json, str(p)): p for p in json_paths
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=desc,
                unit="json",
                file=sys.stderr,
            ):
                stats.json_files_scanned += 1
                step_id, err = future.result()
                if err == "parse":
                    stats.json_parse_errors += 1
                elif err == "no_faces":
                    stats.json_missing_faces += 1
                elif step_id is not None:
                    stats.json_with_cylinder += 1
                    cylinder_by_step[step_id] = True

    step_ids = {sid for sid, flag in cylinder_by_step.items() if flag}
    stats.unique_step_ids_with_cylinder = len(step_ids)
    return step_ids, stats


def copy_step_files(
    step_ids: set[str],
    step_index: dict[str, Path],
    out_dir: Path,
    desc: str,
    stats: DatasetStats,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered_ids = sorted(step_ids)

    for step_id in tqdm(ordered_ids, desc=desc, unit="step", file=sys.stderr):
        src = step_index.get(step_id)
        if src is None:
            stats.steps_missing += 1
            if len(stats.missing_step_ids) < 50:
                stats.missing_step_ids.append(step_id)
            continue

        dst = out_dir / src.name
        if dst.exists():
            try:
                if dst.stat().st_size == src.stat().st_size:
                    stats.steps_already_present += 1
                    continue
            except OSError:
                pass

        try:
            shutil.copy2(src, dst)
            stats.steps_copied += 1
        except OSError:
            stats.steps_copy_errors += 1


def process_dataset(
    name: str,
    json_dir: Path,
    step_src: Path,
    step_dst: Path,
    recursive_steps: bool,
    workers: int,
    *,
    recursive_json: bool = False,
    step_src_dirs: list[Path] | None = None,
) -> DatasetStats:
    if not json_dir.is_dir():
        raise FileNotFoundError(f"{name}: JSON directory not found: {json_dir}")
    if not step_src.is_dir():
        raise FileNotFoundError(f"{name}: STEP source directory not found: {step_src}")

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"{name}", file=sys.stderr)
    print(f"  JSON : {json_dir}", file=sys.stderr)
    print(f"  STEP : {step_src}", file=sys.stderr)
    print(f"  OUT  : {step_dst}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    step_ids, stats = scan_json_directory(
        json_dir,
        desc=f"{name}: scan JSON",
        workers=workers,
        recursive=recursive_json,
    )

    if not step_ids:
        print(f"{name}: no cylindrical models found.", file=sys.stderr)
        return stats

    roots = step_src_dirs if step_src_dirs is not None else [step_src]
    if len(roots) == 1:
        print(f"{name}: indexing STEP files in {roots[0]} ...", file=sys.stderr)
        step_index = build_step_index(roots[0], recursive=recursive_steps)
    else:
        print(
            f"{name}: indexing STEP files in {len(roots)} chunk folder(s) under {step_src} ...",
            file=sys.stderr,
        )
        step_index = build_step_index_multi(roots, recursive=recursive_steps)
    print(f"{name}: {len(step_index):,} STEP files indexed.", file=sys.stderr)

    copy_step_files(
        step_ids,
        step_index,
        step_dst,
        desc=f"{name}: copy STEP",
        stats=stats,
    )
    return stats


def print_summary(name: str, stats: DatasetStats, out_dir: Path) -> None:
    print(f"\n--- {name} summary ---")
    print(f"  JSON scanned              : {stats.json_files_scanned:,}")
    print(f"  JSON with cylinder (z=1)  : {stats.json_with_cylinder:,}")
    print(f"  Unique STEP ids (cylinder): {stats.unique_step_ids_with_cylinder:,}")
    print(f"  STEP files copied         : {stats.steps_copied:,}")
    print(f"  STEP already in output    : {stats.steps_already_present:,}")
    print(f"  STEP missing at source    : {stats.steps_missing:,}")
    print(f"  STEP copy errors          : {stats.steps_copy_errors:,}")
    print(f"  JSON parse errors         : {stats.json_parse_errors:,}")
    print(f"  JSON without faces        : {stats.json_missing_faces:,}")
    print(f"  Output directory          : {out_dir}")
    if stats.missing_step_ids:
        preview = ", ".join(stats.missing_step_ids[:10])
        suffix = " ..." if stats.steps_missing > 10 else ""
        print(f"  Missing STEP id sample    : {preview}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy STEP files that have at least one cylindrical face (z=1) "
            "according to companion JSON labels."
        )
    )
    parser.add_argument(
        "--cadsynth-json-dir",
        type=Path,
        default=Path(r"Z:\Experiment6\source_dataset\input"),
        help="CadSynth JSON directory (read-only).",
    )
    parser.add_argument(
        "--mfcadpp-json-dir",
        type=Path,
        default=Path(
            r"Z:\Experiment6\target_dataset\input\json_new_labels_cadsynth_label_indices"
        ),
        help="MFCAD++ JSON directory (read-only).",
    )
    parser.add_argument(
        "--cadsynth-step-src",
        type=Path,
        default=Path(r"Y:\orginal_authors\step"),
        help="Original CadSynth STEP directory (read-only).",
    )
    parser.add_argument(
        "--mfcadpp-step-src",
        type=Path,
        default=Path(r"X:\step"),
        help="Original MFCAD++ STEP directory (read-only).",
    )
    parser.add_argument(
        "--cadsynth-step-dst",
        type=Path,
        default=Path(r"Y:\threads_dataset\cadsynth\steps"),
        help="Output directory for filtered CadSynth STEP files.",
    )
    parser.add_argument(
        "--mfcadpp-step-dst",
        type=Path,
        default=Path(r"Y:\threads_dataset\mfcadpp\steps"),
        help="Output directory for filtered MFCAD++ STEP files.",
    )
    parser.add_argument(
        "--abc-json-dir",
        type=Path,
        default=Path(r"D:\abc\sw_jsons"),
        help="ABC SolidWorks-style JSON directory (all chunks, read-only).",
    )
    parser.add_argument(
        "--abc-step-src",
        type=Path,
        default=Path(r"D:\abc\step_extracted"),
        help=(
            "ABC STEP root: either one chunk folder or parent of abc_*_step_v* chunk folders."
        ),
    )
    parser.add_argument(
        "--abc-step-dst",
        type=Path,
        default=Path(r"Y:\threads_dataset\abc\step"),
        help="Output directory for filtered ABC STEP files.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="NAMES",
        help="Comma-separated subset to run: cadsynth, mfcadpp, abc. Default: all three.",
    )
    parser.add_argument(
        "--recursive-steps",
        action="store_true",
        help="Search STEP source trees recursively (CadSynth/MFCAD++; default: top-level only).",
    )
    parser.add_argument(
        "--no-abc-recursive-steps",
        action="store_true",
        help="Only list STEP at the top level of each ABC chunk (default: recursive).",
    )
    parser.add_argument(
        "--abc-recursive-json",
        action="store_true",
        help="Scan JSON recursively under --abc-json-dir (default: top-level *.json only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel JSON scan workers (default: CPU count - 1). Use 1 to disable.",
    )
    return parser.parse_args()


def _parse_only(raw: str | None) -> set[str]:
    allowed = {"cadsynth", "mfcadpp", "abc"}
    if raw is None:
        return allowed
    sel = {x.strip().lower() for x in raw.split(",") if x.strip()}
    unknown = sel - allowed
    if unknown:
        raise SystemExit(f"Unknown --only entries: {sorted(unknown)}; use: {sorted(allowed)}")
    return sel


def main() -> int:
    args = parse_args()
    run = _parse_only(args.only)

    workers = max(1, args.workers)
    print(f"Using {workers} worker(s) for JSON scan.", file=sys.stderr)
    print(f"Datasets to run: {', '.join(sorted(run))}", file=sys.stderr)

    summaries: list[tuple[str, DatasetStats, Path]] = []

    if "cadsynth" in run:
        cadsynth_stats = process_dataset(
            "CadSynth",
            args.cadsynth_json_dir,
            args.cadsynth_step_src,
            args.cadsynth_step_dst,
            args.recursive_steps,
            workers,
        )
        summaries.append(("CadSynth", cadsynth_stats, args.cadsynth_step_dst))

    if "mfcadpp" in run:
        mfcadpp_stats = process_dataset(
            "MFCAD++",
            args.mfcadpp_json_dir,
            args.mfcadpp_step_src,
            args.mfcadpp_step_dst,
            args.recursive_steps,
            workers,
        )
        summaries.append(("MFCAD++", mfcadpp_stats, args.mfcadpp_step_dst))

    if "abc" in run:
        abc_chunks = discover_abc_step_dirs(args.abc_step_src.expanduser().resolve())
        print(
            f"ABC: using {len(abc_chunks)} STEP chunk folder(s): "
            + ", ".join(p.name for p in abc_chunks[:5])
            + (" ..." if len(abc_chunks) > 5 else ""),
            file=sys.stderr,
        )
        abc_recursive_steps = not args.no_abc_recursive_steps
        abc_stats = process_dataset(
            "ABC",
            args.abc_json_dir,
            args.abc_step_src,
            args.abc_step_dst,
            abc_recursive_steps,
            workers,
            recursive_json=args.abc_recursive_json,
            step_src_dirs=abc_chunks,
        )
        summaries.append(("ABC", abc_stats, args.abc_step_dst))

    print("\n" + "=" * 60)
    total_copied = 0
    for name, stats, out_dir in summaries:
        print_summary(name, stats, out_dir)
        total_copied += stats.steps_copied
    print("=" * 60)
    if summaries:
        parts = "  |  ".join(f"{name}: {st.steps_copied:,}" for name, st, _ in summaries)
        print(f"\nTotal STEP files added — {parts}  |  Combined: {total_copied:,}")

    if any(st.steps_copy_errors for _, st, _ in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
