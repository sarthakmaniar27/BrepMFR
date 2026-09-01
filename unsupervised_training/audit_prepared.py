#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unsupervised_training.data import UnlabeledGraphDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict audit of prepared unlabeled no-A2 graphs")
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    summary_path = root / "preparation_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Preparation summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("labels_used_from_source_json") is not False:
        raise SystemExit("Preparation summary does not certify labels_used_from_source_json=false")

    datasets = {
        split: UnlabeledGraphDataset(root, split, scan_graphs=True)
        for split in ("train", "val")
    }
    overlap = {
        path.stem.casefold() for path in datasets["train"].file_paths
    } & {path.stem.casefold() for path in datasets["val"].file_paths}
    if overlap:
        raise SystemExit(f"Train/val stem leakage: {sorted(overlap)[:20]}")
    report = {
        "status": "passed",
        "dataset_root": str(root),
        "train_graphs": len(datasets["train"]),
        "val_graphs": len(datasets["val"]),
        "train_faces": sum(datasets["train"]._actual_node_counts),
        "val_faces": sum(datasets["val"]._actual_node_counts),
        "sentinel_only": True,
        "profile": "no_a2",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

