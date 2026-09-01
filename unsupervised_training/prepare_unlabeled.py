#!/usr/bin/env python3
"""Audit label-free SolidWorks JSONs and convert them to strict no-A2 PyG graphs.

Original JSON files are never modified. Any face labels present in the input are
ignored, and generated graph labels are set to ``-100`` solely to satisfy the
existing collator contract. The training module asserts that these values never
enter supervised cross-entropy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inference.json_to_brepmfr_pyg_optimized import (  # noqa: E402
    load_json_fast,
    tensors_from_brep_json_dict,
)
from unsupervised_training.constants import (  # noqa: E402
    IGNORE_LABEL,
    NO_A2_PROFILE,
    SPATIAL_POS_MAX,
)


STEP_FAMILY_RE = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)
REQUIRED_FACE_KEYS = frozenset({"id", "uv", "z", "y", "l", "a"})
REQUIRED_EDGE_KEYS = frozenset({"nf", "pt"})


@dataclass(frozen=True)
class ConversionResult:
    source: str
    stem: str
    family: str
    split: str
    status: str
    faces: int = 0
    edges: int = 0
    sha256: str = ""
    output: str = ""
    error: str = ""


def family_key(stem: str) -> str:
    match = STEP_FAMILY_RE.match(stem)
    return (match.group("key") if match else stem).casefold()


def stable_split(family: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{family}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if fraction < validation_fraction else "train"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_json_schema(data: dict[str, Any], source: Path) -> tuple[int, int]:
    faces = data.get("faces")
    edges = data.get("edges")
    if not isinstance(faces, list) or not faces:
        raise ValueError("missing or empty faces list")
    if not isinstance(edges, list):
        raise ValueError("missing edges list")
    face_ids: set[int] = set()
    for index, face in enumerate(faces):
        if not isinstance(face, dict):
            raise ValueError(f"face[{index}] is not an object")
        missing = REQUIRED_FACE_KEYS - set(face)
        if missing:
            raise ValueError(f"face[{index}] missing {sorted(missing)}")
        face_id = int(face["id"])
        if face_id in face_ids:
            raise ValueError(f"duplicate face id {face_id}")
        face_ids.add(face_id)
        if len(face["uv"]) != 5 * 5 * 7:
            raise ValueError(f"face[{index}].uv length={len(face['uv'])}, expected 175")
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(f"edge[{index}] is not an object")
        missing = REQUIRED_EDGE_KEYS - set(edge)
        if missing:
            raise ValueError(f"edge[{index}] missing {sorted(missing)}")
        if len(edge["nf"]) != 2 or any(int(value) not in face_ids for value in edge["nf"]):
            raise ValueError(f"edge[{index}] references invalid faces: {edge['nf']}")
        if len(edge["pt"]) != 5 * 7:
            raise ValueError(f"edge[{index}].pt length={len(edge['pt'])}, expected 35")
    return len(faces), len(edges)


def _atomic_torch_save(graph: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    torch.save(graph, temporary)
    os.replace(temporary, destination)


def convert_one(
    source_text: str,
    output_text: str,
    split: str,
    family: str,
    overwrite: bool,
    float16_storage: bool,
) -> ConversionResult:
    source = Path(source_text)
    output = Path(output_text)
    try:
        digest = sha256_file(source)
        if output.is_file() and not overwrite:
            try:
                existing = torch.load(output, map_location="cpu", weights_only=False)
            except TypeError:
                existing = torch.load(output, map_location="cpu")
            labels = getattr(existing, "label_feature", None)
            if labels is None or labels.numel() == 0 or not torch.all(labels == IGNORE_LABEL):
                raise ValueError("existing output is not strict sentinel-only unlabeled data")
            recorded_digest = getattr(existing, "source_json_sha256", None)
            if recorded_digest != digest:
                raise ValueError(
                    "source JSON changed after conversion; rerun with --overwrite after review"
                )
            return ConversionResult(
                str(source),
                source.stem,
                family,
                split,
                "existing",
                int(existing.node_data.shape[0]),
                int(existing.edge_data.shape[0]),
                digest,
                str(output),
            )
        data = load_json_fast(source)
        faces, edges = validate_json_schema(data, source)
        graph, _ = tensors_from_brep_json_dict(
            data,
            spatial_pos_max=SPATIAL_POS_MAX,
            inference_profile=NO_A2_PROFILE,
            max_edge_path_len=16,
            float16_storage=float16_storage,
            shortest_path_workers=0,
        )
        graph.label_feature = torch.full((faces,), IGNORE_LABEL, dtype=torch.int32)
        graph.is_unlabeled = True
        graph.unlabeled_sentinel = IGNORE_LABEL
        graph.source_json_sha256 = digest
        graph.source_json_name = source.name
        try:
            graph.data_id = int(source.stem.split("_")[-1])
        except ValueError:
            graph.data_id = 0
        _atomic_torch_save(graph, output)
        return ConversionResult(
            str(source), source.stem, family, split, "converted", faces, edges, digest, str(output)
        )
    except Exception as exc:
        return ConversionResult(
            str(source), source.stem, family, split, "failed", error=f"{type(exc).__name__}: {exc}"
        )


def existing_labeled_families(dataset_root: Path) -> tuple[set[str], set[str]]:
    stems: set[str] = set()
    families: set[str] = set()
    for split in ("train", "val", "test"):
        path = dataset_root / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Labeled split not found: {path}")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stem = line.strip()
            if stem:
                stems.add(stem.casefold())
                families.add(family_key(stem))
    return stems, families


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", required=True, help="Folder containing raw SolidWorks JSONs")
    parser.add_argument("--output-root", required=True, help="New unlabeled dataset root")
    parser.add_argument(
        "--labeled-dataset-root",
        required=True,
        help="Labeled dataset whose train/val/test STEP families must be excluded",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="Conversion smoke limit; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--float16-storage", action="store_true")
    parser.add_argument(
        "--allow-labeled-family-overlap",
        action="store_true",
        help="Allow overlap with labeled STEP families (not recommended; explicit opt-in only)",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    labeled_root = Path(args.labeled_dataset_root).expanduser().resolve()
    if not json_dir.is_dir():
        raise SystemExit(f"JSON directory not found: {json_dir}")
    if not 0.0 < args.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be in (0, 0.5)")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    labeled_stems, labeled_families = existing_labeled_families(labeled_root)
    sources = sorted(json_dir.glob("*.json"), key=lambda path: path.name.casefold())
    duplicate_names = [name for name, count in Counter(path.stem.casefold() for path in sources).items() if count > 1]
    if duplicate_names:
        raise SystemExit(f"Duplicate JSON stems: {duplicate_names[:20]}")

    selected: list[tuple[Path, str, str]] = []
    exclusions: Counter[str] = Counter()
    seen_families: dict[str, str] = {}
    for source in sources:
        family = family_key(source.stem)
        if source.stem.casefold() in labeled_stems:
            exclusions["exact_labeled_stem"] += 1
            continue
        if family in labeled_families and not args.allow_labeled_family_overlap:
            exclusions["labeled_step_family"] += 1
            continue
        split = seen_families.setdefault(
            family, stable_split(family, args.seed, args.validation_fraction)
        )
        selected.append((source, family, split))
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No new unlabeled JSONs remain after overlap exclusions")

    graph_root = output_root / "pyg"
    graph_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            str(source),
            str(graph_root / f"{source.stem}.pt"),
            split,
            family,
            bool(args.overwrite),
            bool(args.float16_storage),
        )
        for source, family, split in selected
    ]

    results: list[ConversionResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one, *job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 250 == 0 or index == len(futures):
                print(f"Converted/audited {index:,}/{len(futures):,}", flush=True)

    results.sort(key=lambda result: result.stem.casefold())
    failed = [result for result in results if result.status == "failed"]
    successful = [result for result in results if result.status in {"converted", "existing"}]
    if failed:
        failure_path = output_root / "conversion_failures.jsonl"
        write_atomic(failure_path, "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in failed))
        raise SystemExit(
            f"Conversion failed for {len(failed):,} JSON(s); see {failure_path}. "
            f"First: {failed[0].source}: {failed[0].error}"
        )

    by_split = {split: [item.stem for item in successful if item.split == split] for split in ("train", "val")}
    for split, stems in by_split.items():
        write_atomic(output_root / f"{split}.txt", "".join(f"{stem}\n" for stem in stems))
    manifest_path = output_root / "manifest.jsonl"
    write_atomic(
        manifest_path,
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in successful),
    )
    summary = {
        "schema_version": 1,
        "input_json_dir": str(json_dir),
        "output_root": str(output_root),
        "labeled_dataset_root": str(labeled_root),
        "profile": NO_A2_PROFILE,
        "unlabeled_label_sentinel": IGNORE_LABEL,
        "input_jsons": len(sources),
        "selected_jsons": len(selected),
        "converted_or_existing": len(successful),
        "train_graphs": len(by_split["train"]),
        "val_graphs": len(by_split["val"]),
        "faces": sum(item.faces for item in successful),
        "excluded": dict(exclusions),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "labels_used_from_source_json": False,
    }
    write_atomic(output_root / "preparation_summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

