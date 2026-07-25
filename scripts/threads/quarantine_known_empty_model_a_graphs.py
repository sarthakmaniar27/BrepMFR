#!/usr/bin/env python3
"""Quarantine known empty legacy Model A graphs without rescanning the dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import torch


SPLITS = ("train", "val", "test")


def _key(value: str) -> str:
    return value.casefold()


def _index_flat(root: Path, suffix: str) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    indexed: dict[str, Path] = {}
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.casefold().endswith(suffix):
                continue
            path = Path(entry.path)
            normalized = _key(path.stem)
            if normalized in indexed:
                raise ValueError(f"Duplicate stem under {root}: {path.stem}")
            indexed[normalized] = path
    return indexed


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find Model A graphs absent from the trusted expanded graph store, "
            "prove they have empty labels, then quarantine only those links."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model-a-root", required=True, type=Path)
    parser.add_argument("--expanded-root", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=25)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_pyg = dataset_root / "pyg"
    model_a_graphs = _index_flat(args.model_a_root.resolve() / "pyg", ".pt")
    expanded_graphs = _index_flat(args.expanded_root.resolve() / "pyg", ".pt")
    output_graphs = _index_flat(output_pyg, ".pt")

    candidates = sorted(set(model_a_graphs) - set(expanded_graphs))
    if len(candidates) != args.expected_count:
        raise ValueError(
            "Safety check failed: Model-A-minus-expanded candidate count is "
            f"{len(candidates)}, expected {args.expected_count}."
        )

    split_lines: dict[str, list[str]] = {}
    split_by_stem: dict[str, str] = {}
    for split in SPLITS:
        split_lines[split] = _read_lines(dataset_root / f"{split}.txt")
        for stem in split_lines[split]:
            normalized = _key(stem)
            if normalized in split_by_stem:
                raise ValueError(f"Stem appears in multiple splits: {stem}")
            split_by_stem[normalized] = split

    audit: list[dict[str, str]] = []
    for normalized in candidates:
        output_path = output_graphs.get(normalized)
        if output_path is None:
            raise FileNotFoundError(
                f"Candidate is missing from prepared output: {model_a_graphs[normalized].stem}"
            )
        if normalized not in split_by_stem:
            raise ValueError(f"Candidate is not split-listed: {output_path.stem}")
        graph = _load(output_path)
        labels = getattr(graph, "label_feature", None)
        if labels is not None and labels.numel() > 0:
            raise ValueError(
                f"Refusing to quarantine a graph with non-empty labels: {output_path.stem}"
            )
        audit.append(
            {
                "stem": output_path.stem,
                "split": split_by_stem[normalized],
                "reason": "label_feature is missing or empty",
            }
        )

    removals_by_split = Counter(item["split"] for item in audit)
    print(f"Verified empty-label candidates: {len(audit):,}")
    for split in SPLITS:
        print(f"  {split}: remove {removals_by_split[split]:,}")
    for item in audit:
        print(f"  - {item['stem']} ({item['split']})")

    if not args.apply:
        print("Dry run passed; no files were modified.")
        return 0

    quarantine = dataset_root / "quarantine_invalid_graphs"
    quarantine.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    try:
        for item in audit:
            source = output_graphs[_key(item["stem"])]
            destination = quarantine / source.name
            if destination.exists():
                raise FileExistsError(f"Quarantine target already exists: {destination}")
            source.replace(destination)
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                destination.replace(source)
        raise

    rejected = {_key(item["stem"]) for item in audit}
    final_split_counts: dict[str, int] = {}
    for split in SPLITS:
        kept = [stem for stem in split_lines[split] if _key(stem) not in rejected]
        final_split_counts[split] = len(kept)
        _atomic_text(
            dataset_root / f"{split}.txt",
            "".join(f"{stem}\n" for stem in kept),
        )

    sources_csv = dataset_root / "graph_sources.csv"
    if sources_csv.is_file():
        with sources_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        kept_rows = [row for row in rows if _key(row["stem"]) not in rejected]
        temporary = sources_csv.with_name(sources_csv.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
        os.replace(temporary, sources_csv)

    report = {
        "quarantined_count": len(audit),
        "graphs": audit,
        "final_split_counts": final_split_counts,
        "final_training_graphs": sum(final_split_counts.values()),
        "note": "Only output hard links were moved; source .pt files were not modified.",
    }
    _atomic_text(
        quarantine / "report.json",
        json.dumps(report, indent=2) + "\n",
    )

    summary_path = dataset_root / "preparation_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["post_validation_cleanup"] = report
        summary["counts"]["training_graphs"] = report["final_training_graphs"]
        summary["counts"]["total_output_graph_links"] -= len(audit)
        for split in SPLITS:
            summary["splits"][split]["model_a"] -= removals_by_split[split]
            summary["splits"][split]["final"] = final_split_counts[split]
        _atomic_text(summary_path, json.dumps(summary, indent=2) + "\n")

    print(
        f"Quarantined {len(audit):,} empty graphs. "
        f"Final training graphs: {report['final_training_graphs']:,}."
    )
    print(f"Report: {quarantine / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
