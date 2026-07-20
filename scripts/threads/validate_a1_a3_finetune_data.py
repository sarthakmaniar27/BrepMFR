#!/usr/bin/env python3
"""Validate a no_a2 (A1+A3) dataset before fine-tuning a lite checkpoint."""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check A1/A3 tensors, split coverage, and optional label parity with the lite dataset."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--reference-lite-root", type=Path, default=None)
    parser.add_argument("--pt-subdir", default="pyg")
    parser.add_argument("--max-files", type=int, default=0, help="0 validates every split-listed graph.")
    parser.add_argument("--report-a3-cap", type=int, default=768)
    args = parser.parse_args()

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

    errors: list[str] = []
    max_nodes = 0
    above_cap = 0
    cap = int(args.report_a3_cap)

    for stem in tqdm(selected, desc="validating A1+A3 graphs", unit="graph"):
        try:
            graph = _load(graph_paths[stem])
            n = int(graph.node_data.size(0))
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
            errors.append(f"{stem}: {exc}")
            if len(errors) >= 20:
                break

    print(f"\nValidated graphs: {len(selected) - len(errors):,} / {len(selected):,}")
    print(f"Maximum face count observed: {max_nodes:,}")
    if cap > 0:
        print(f"Graphs above suggested A3 cap ({cap}): {above_cap:,}")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("A1+A3 profile, split coverage, and reference parity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
