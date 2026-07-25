#!/usr/bin/env python3
"""Create Stock-only JSON copies from the approved ABC inference list.

The source ABC JSONs are never modified.  Every input path must come from the
``no_confident_thread_or_text.txt`` allowlist produced by the inference filter.
The script first audits the complete list and refuses to write anything when:

* an input file is missing or invalid;
* a JSON has no non-empty top-level ``faces`` list;
* a face is missing an integer ``label``;
* a source label is outside ``--expected-source-labels``; or
* two approved paths would produce the same output filename.

After a successful audit, pass ``--write`` to copy each JSON to ``--output-dir``
while setting every ``face["label"]`` to class 0 (Stock by default).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

try:
    import orjson
except ImportError:
    orjson = None


@dataclass(frozen=True)
class AuditResult:
    source: Path
    output_name: str
    face_count: int
    labels: dict[int, int]
    error: str = ""


def _loads(path: Path) -> dict:
    raw = path.read_bytes()
    data = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value is not an object")
    return data


def _dumps(data: dict) -> bytes:
    if orjson is not None:
        return orjson.dumps(
            data,
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
        )
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _parse_int_set(value: str) -> set[int]:
    try:
        parsed = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one expected source label is required")
    return parsed


def _approved_paths(list_path: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        list_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        value = raw_line.strip().strip('"')
        if not value or value.startswith("#"):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = list_path.parent / path
        path = path.resolve()
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        if path.suffix.lower() != ".json":
            raise ValueError(
                f"{list_path}:{line_number}: approved path is not a .json file: {path}"
            )
        paths.append(path)
    if not paths:
        raise ValueError(f"approved list contains no JSON paths: {list_path}")
    return paths


def _audit_one(path: Path, expected_labels: set[int]) -> AuditResult:
    try:
        if not path.is_file():
            raise FileNotFoundError("source JSON does not exist")
        data = _loads(path)
        faces = data.get("faces")
        if not isinstance(faces, list) or not faces:
            raise ValueError("top-level 'faces' must be a non-empty list")
        counts: Counter[int] = Counter()
        for index, face in enumerate(faces):
            if not isinstance(face, dict):
                raise ValueError(f"faces[{index}] is not an object")
            if "label" not in face:
                raise ValueError(f"faces[{index}] has no 'label'")
            raw_label = face["label"]
            if isinstance(raw_label, bool):
                raise ValueError(f"faces[{index}].label is boolean, not an integer")
            try:
                label = int(raw_label)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"faces[{index}].label is not an integer: {raw_label!r}"
                ) from exc
            if label not in expected_labels:
                raise ValueError(
                    f"faces[{index}].label={label} is outside expected "
                    f"{sorted(expected_labels)}"
                )
            counts[label] += 1
        return AuditResult(
            source=path,
            output_name=path.name,
            face_count=len(faces),
            labels=dict(counts),
        )
    except Exception as exc:
        return AuditResult(
            source=path,
            output_name=path.name,
            face_count=0,
            labels={},
            error=str(exc),
        )


def _write_one(
    result: AuditResult,
    output_dir: Path,
    stock_label: int,
    overwrite: bool,
) -> tuple[Path, int, str]:
    destination = output_dir / result.output_name
    temporary = output_dir / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"destination exists (use --overwrite deliberately): {destination}"
            )
        data = _loads(result.source)
        faces = data["faces"]
        for face in faces:
            face["label"] = stock_label
        temporary.write_bytes(_dumps(data))
        os.replace(temporary, destination)
        return destination, len(faces), ""
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return destination, 0, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approved-list",
        required=True,
        type=Path,
        help="Path to no_confident_thread_or_text.txt (one approved JSON path per line).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Separate destination for Stock-labeled JSON copies.",
    )
    parser.add_argument(
        "--expected-source-labels",
        type=_parse_int_set,
        default={-10, -1, 0},
        help=(
            "Comma-separated labels permitted in approved source JSONs "
            "(default: -10,-1,0). Labels 70/101 are rejected."
        ),
    )
    parser.add_argument(
        "--stock-label",
        type=int,
        default=0,
        help="Target Stock class id (default: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
        help="Parallel audit/write workers (default: CPU count capped at 8).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write Stock-labeled copies after the complete audit passes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination JSONs; requires --write.",
    )
    args = parser.parse_args()

    approved_list = args.approved_list.resolve()
    output_dir = args.output_dir.resolve()
    if not approved_list.is_file():
        raise SystemExit(f"approved list not found: {approved_list}")
    if args.overwrite and not args.write:
        parser.error("--overwrite requires --write")
    if args.stock_label < 0:
        parser.error("--stock-label must be non-negative")

    try:
        paths = _approved_paths(approved_list)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    duplicate_names = [
        name
        for name, count in Counter(path.name.lower() for path in paths).items()
        if count > 1
    ]
    if duplicate_names:
        raise SystemExit(
            "approved inputs contain output filename collisions; first examples: "
            + ", ".join(duplicate_names[:20])
        )

    workers = max(1, int(args.workers))
    expected = set(args.expected_source_labels)
    print(f"Approved list:          {approved_list}")
    print(f"Approved JSON paths:    {len(paths):,}")
    print(f"Expected source labels: {sorted(expected)}")
    print(f"Stock target label:     {args.stock_label}")
    print(f"Output directory:       {output_dir}")
    print(f"Mode:                   {'WRITE' if args.write else 'DRY RUN'}")
    print(f"JSON parser:            {'orjson' if orjson is not None else 'stdlib json'}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            tqdm(
                pool.map(lambda path: _audit_one(path, expected), paths),
                total=len(paths),
                desc="Auditing approved JSONs",
                unit="file",
                dynamic_ncols=True,
            )
        )

    errors = [result for result in results if result.error]
    total_faces = sum(result.face_count for result in results)
    label_counts: Counter[int] = Counter()
    for result in results:
        label_counts.update(result.labels)
    print(f"Audited faces:          {total_faces:,}")
    print(f"Source label counts:    {dict(sorted(label_counts.items()))}")
    if errors:
        print(f"Audit failed for {len(errors):,} JSON(s); nothing was written.")
        for result in errors[:20]:
            print(f"  {result.source}: {result.error}")
        return 1

    if not args.write:
        print("Dry run passed; no source or destination files were modified.")
        print("Rerun with --write to create the Stock-labeled copies.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        existing = [
            output_dir / result.output_name
            for result in results
            if (output_dir / result.output_name).exists()
        ]
        if existing:
            print(
                f"Refusing write: {len(existing):,} destination JSON(s) already exist. "
                "Use an empty output directory or pass --overwrite deliberately."
            )
            for path in existing[:20]:
                print(f"  {path}")
            return 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        written = list(
            tqdm(
                pool.map(
                    lambda result: _write_one(
                        result,
                        output_dir,
                        int(args.stock_label),
                        bool(args.overwrite),
                    ),
                    results,
                ),
                total=len(results),
                desc="Writing Stock JSONs",
                unit="file",
                dynamic_ncols=True,
            )
        )
    write_errors = [item for item in written if item[2]]
    if write_errors:
        print(
            f"Write failed for {len(write_errors):,} JSON(s). "
            "Successful outputs remain valid and can be replaced with --overwrite."
        )
        for destination, _, error in write_errors[:20]:
            print(f"  {destination}: {error}")
        return 1

    manifest_path = output_dir / "stock_label_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source_json", "output_json", "face_count", "assigned_label"]
        )
        for result, (destination, face_count, _) in zip(results, written):
            writer.writerow(
                [str(result.source), str(destination), face_count, args.stock_label]
            )

    print(f"Wrote Stock JSONs:      {len(written):,}")
    print(f"Wrote Stock faces:      {sum(item[1] for item in written):,}")
    print(f"Manifest:               {manifest_path}")
    print("The source JSONs were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
