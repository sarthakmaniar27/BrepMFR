#!/usr/bin/env python3
"""Validate a no_a2 (A1+A3) dataset before fine-tuning a lite checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _index_pt(root: Path) -> tuple[dict[str, Path], list[str]]:
    by_stem: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in root.rglob("*.pt"):
        if path.stem in by_stem:
            duplicates.append(path.stem)
        else:
            by_stem[path.stem] = path
    return by_stem, duplicates


def _split_stems(root: Path) -> list[str]:
    stems: list[str] = []
    for split in ("train", "val", "test"):
        path = root / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split list: {path}")
        stems.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    duplicates = [stem for stem, count in Counter(stems).items() if count > 1]
    if duplicates:
        raise ValueError(f"Stems occur in multiple split lists; first examples: {duplicates[:10]}")
    return stems


def _rewrite_stem_file(path: Path, rejected: set[str]) -> int:
    if not path.is_file():
        return 0
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [stem for stem in lines if stem not in rejected]
    removed = len(lines) - len(kept)
    if removed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(f"{stem}\n" for stem in kept), encoding="utf-8")
        os.replace(temporary, path)
    return removed


def _quarantine_invalid(
    dataset_root: Path,
    graph_paths: dict[str, Path],
    errors: dict[str, str],
) -> Path:
    quarantine = dataset_root / "quarantine_invalid_graphs"
    quarantine.mkdir(parents=True, exist_ok=True)
    overwritten = 0
    skipped_missing = 0
    moved: list[tuple[Path, Path]] = []
    try:
        for stem in sorted(errors):
            source = graph_paths.get(stem)
            destination = quarantine / f"{stem}.pt"
            if source is None or not source.is_file():
                skipped_missing += 1
                continue
            if source.resolve() == destination.resolve():
                continue
            if destination.exists():
                destination.unlink()
                overwritten += 1
            source.replace(destination)
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                destination.replace(source)
        raise

    rejected = set(errors)
    split_removals = {
        name: _rewrite_stem_file(dataset_root / f"{name}.txt", rejected)
        for name in ("train", "val", "test")
    }
    abc_removed = _rewrite_stem_file(dataset_root / "abc_stems.txt", rejected)
    report = {
        "quarantined_count": len(errors),
        "moved_count": len(moved),
        "overwritten_existing_quarantine": overwritten,
        "already_absent_from_pyg": skipped_missing,
        "split_removals": split_removals,
        "abc_manifest_removals": abc_removed,
        "graphs": [{"stem": stem, "reason": errors[stem]} for stem in sorted(errors)],
    }
    report_path = quarantine / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check A1/A3 tensors, split coverage, and optional label parity with the lite dataset."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--reference-lite-root", type=Path, default=None)
    parser.add_argument("--pt-subdir", default="pyg")
    parser.add_argument("--max-files", type=int, default=0, help="0 validates every split-listed graph.")
    parser.add_argument("--report-a3-cap", type=int, default=768)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument(
        "--quarantine-invalid",
        action="store_true",
        help="Move every invalid graph out of pyg and remove its stem from split/ABC lists.",
    )
    args = parser.parse_args()
    if args.quarantine_invalid and args.max_files > 0:
        raise SystemExit("--quarantine-invalid requires a complete scan; do not use --max-files")

    dataset_root = args.dataset_root.resolve()
    graph_root = dataset_root / args.pt_subdir
    if not graph_root.is_dir():
        raise FileNotFoundError(f"Graph directory not found: {graph_root}")

    stems = _split_stems(dataset_root)
    graph_paths, duplicate_paths = _index_pt(graph_root)
    if duplicate_paths:
        raise ValueError(f"Duplicate .pt stems under {graph_root}; first examples: {duplicate_paths[:10]}")

    missing = [stem for stem in stems if stem not in graph_paths]
    if missing:
        raise FileNotFoundError(
            f"{len(missing):,} split-listed graphs are missing from {graph_root}; first: {missing[:10]}"
        )

    reference_paths: dict[str, Path] = {}
    if args.reference_lite_root is not None:
        reference_graph_root = args.reference_lite_root.resolve() / args.pt_subdir
        reference_paths, reference_duplicates = _index_pt(reference_graph_root)
        if reference_duplicates:
            raise ValueError(
                f"Duplicate reference .pt stems under {reference_graph_root}; first: {reference_duplicates[:10]}"
            )
        missing_reference = [stem for stem in stems if stem not in reference_paths]
        if missing_reference:
            raise FileNotFoundError(
                f"{len(missing_reference):,} split-listed lite references are missing; first: {missing_reference[:10]}"
            )

    selected = sorted(stems)
    if args.max_files > 0:
        selected = selected[: args.max_files]

    errors: dict[str, str] = {}
    max_nodes = 0
    above_cap = 0
    cap = int(args.report_a3_cap)

    for stem in tqdm(selected, desc="validating A1+A3 graphs", unit="graph"):
        try:
            graph = _load(graph_paths[stem])
            n = int(graph.node_data.size(0))
            labels = getattr(graph, "label_feature", None)
            if labels is None or labels.numel() == 0:
                raise ValueError("label_feature is missing or empty")
            if int(labels.numel()) != n:
                raise ValueError(
                    f"label_feature has {int(labels.numel())} values for {n} faces"
                )
            label_min = int(labels.min().item())
            label_max = int(labels.max().item())
            if label_min < 0 or label_max >= int(args.num_classes):
                raise ValueError(
                    f"label range [{label_min}, {label_max}] is outside "
                    f"[0, {int(args.num_classes) - 1}]"
                )
            max_nodes = max(max_nodes, n)
            if cap > 0 and n > cap:
                above_cap += 1

            has_a1 = bool(getattr(graph, "has_a1", getattr(graph, "spatial_pos", None) is not None))
            has_a2 = bool(
                getattr(
                    graph,
                    "has_a2",
                    getattr(graph, "d2_distance", None) is not None
                    and getattr(graph, "angle_distance", None) is not None,
                )
            )
            has_a3 = bool(getattr(graph, "has_a3", getattr(graph, "edge_path", None) is not None))
            spatial_pos = getattr(graph, "spatial_pos", None)
            edge_path = getattr(graph, "edge_path", None)

            if not has_a1 or has_a2 or not has_a3:
                raise ValueError(f"flags are has_a1={has_a1}, has_a2={has_a2}, has_a3={has_a3}")
            if spatial_pos is None or tuple(spatial_pos.shape) != (n, n):
                raise ValueError(
                    f"spatial_pos shape is {None if spatial_pos is None else tuple(spatial_pos.shape)}, expected {(n, n)}"
                )
            if edge_path is None or edge_path.ndim != 3 or tuple(edge_path.shape[:2]) != (n, n):
                raise ValueError(
                    f"edge_path shape is {None if edge_path is None else tuple(edge_path.shape)}, expected ({n}, {n}, D)"
                )
            if getattr(graph, "d2_distance", None) is not None or getattr(graph, "angle_distance", None) is not None:
                raise ValueError("no_a2 graph unexpectedly stores A2 tensors")

            if reference_paths:
                lite = _load(reference_paths[stem])
                if not torch.equal(graph.label_feature.cpu(), lite.label_feature.cpu()):
                    raise ValueError("label_feature differs from lite reference")
                if not torch.equal(graph.edge_index.cpu(), lite.edge_index.cpu()):
                    raise ValueError("edge_index differs from lite reference")
                if tuple(graph.node_data.shape) != tuple(lite.node_data.shape):
                    raise ValueError("node_data shape differs from lite reference")
        except Exception as exc:
            errors[stem] = str(exc)

    print(f"\nValid graphs: {len(selected) - len(errors):,} / {len(selected):,}")
    print(f"Maximum face count observed: {max_nodes:,}")
    if cap > 0:
        print(f"Graphs above suggested A3 cap ({cap}): {above_cap:,}")

    if errors:
        print(f"\nInvalid graphs: {len(errors):,}")
        for stem, reason in list(errors.items())[:20]:
            print(f"  - {stem}: {reason}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20:,} more (all are included in the report).")
        if args.quarantine_invalid:
            report_path = _quarantine_invalid(dataset_root, graph_paths, errors)
            print(
                f"\nQuarantined all {len(errors):,} invalid graphs and removed them from "
                f"split/ABC lists (existing quarantine copies were replaced).\n"
                f"Report: {report_path}"
            )
            print("All remaining split-listed graphs passed validation.")
            return 0
        print("\nValidation failed. Rerun with --quarantine-invalid to exclude unusable graphs.")
        return 1

    print("A1+A3 profile, split coverage, and reference parity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
