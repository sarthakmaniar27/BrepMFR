#!/usr/bin/env python3
"""Audit training JSONs and optionally quarantine parse/structure failures.

Dry-run is the default. Pass ``--apply`` only after reviewing the complete
report. Valid files are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

try:
    import orjson
except ImportError:
    orjson = None


def _audit(path: Path) -> tuple[Path, int, str]:
    try:
        raw = path.read_bytes()
        if not raw.strip():
            raise ValueError("empty file")
        data = orjson.loads(raw) if orjson is not None else json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON value is not an object")
        faces = data.get("faces")
        if not isinstance(faces, list) or not faces:
            raise ValueError("top-level 'faces' must be a non-empty list")
        for index, face in enumerate(faces):
            if not isinstance(face, dict):
                raise ValueError(f"faces[{index}] is not an object")
            if "label" not in face:
                raise ValueError(f"faces[{index}] has no label")
            int(face["label"])
        return path, len(faces), ""
    except Exception as exc:
        return path, 0, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--quarantine-dir", required=True, type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move invalid JSONs after the complete scan. Default is dry-run.",
    )
    args = parser.parse_args()

    root = args.json_dir.resolve()
    quarantine = args.quarantine_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"JSON directory not found: {root}")
    if root == quarantine:
        raise SystemExit("quarantine directory must differ from JSON directory")

    paths = sorted(root.glob("*.json"))
    if not paths:
        raise SystemExit(f"No top-level JSON files found under: {root}")

    print(f"JSON directory:     {root}")
    print(f"JSON files:         {len(paths):,}")
    print(f"Quarantine:         {quarantine}")
    print(f"Mode:               {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Parser:             {'orjson' if orjson is not None else 'stdlib json'}")

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        results = list(
            tqdm(
                pool.map(_audit, paths),
                total=len(paths),
                desc="Auditing training JSONs",
                unit="file",
                dynamic_ncols=True,
            )
        )

    invalid = [(path, error) for path, _, error in results if error]
    valid_faces = sum(face_count for _, face_count, error in results if not error)
    print(f"Valid JSONs:        {len(paths) - len(invalid):,}")
    print(f"Invalid JSONs:      {len(invalid):,}")
    print(f"Faces in valid:     {valid_faces:,}")
    if invalid:
        print("Invalid files:")
        for path, error in invalid:
            print(f"  {path.name}: {error}")

    report = {
        "json_dir": str(root),
        "quarantine_dir": str(quarantine),
        "mode": "apply" if args.apply else "dry_run",
        "total_jsons": len(paths),
        "valid_jsons": len(paths) - len(invalid),
        "invalid_jsons": len(invalid),
        "valid_faces": valid_faces,
        "invalid": [{"path": str(path), "error": error} for path, error in invalid],
    }

    if not args.apply:
        print("Dry run complete; no files were moved.")
        if invalid:
            print("Rerun with --apply after reviewing the list.")
        return 0

    quarantine.mkdir(parents=True, exist_ok=True)
    collisions = [
        quarantine / path.name
        for path, _ in invalid
        if (quarantine / path.name).exists()
    ]
    if collisions:
        print("Refusing move because quarantine destinations already exist:")
        for path in collisions[:20]:
            print(f"  {path}")
        return 1

    moved: list[tuple[Path, Path]] = []
    try:
        for source, _ in invalid:
            destination = quarantine / source.name
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        raise

    report["moved"] = len(moved)
    report_path = quarantine / "invalid_training_jsons_report.json"
    temporary = report_path.with_suffix(report_path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    print(f"Moved invalid JSONs: {len(moved):,}")
    print(f"Report:              {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
