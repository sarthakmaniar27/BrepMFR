#!/usr/bin/env python3
"""Build a Model-A replay dataset with only full-stem-unique new ABC graphs.

The output is materialized with hard links by default. This gives training a
single, clean ``pyg`` directory without duplicating the (large) graph bytes.
Source graphs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm


SPLITS = ("train", "val", "test")
STEP_FAMILY_RE = re.compile(r"^(?P<key>.+?_step_\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class GraphSource:
    stem: str
    path: Path
    kind: str
    split: str


def _key(value: str) -> str:
    return value.casefold()


def _family_key(stem: str) -> str:
    match = STEP_FAMILY_RE.match(stem)
    return _key(match.group("key") if match else stem)


def _index_files(root: Path, suffix: str) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    indexed: dict[str, Path] = {}
    duplicates: list[str] = []
    with os.scandir(root) as entries:
        paths = [
            Path(entry.path)
            for entry in entries
            if entry.is_file() and entry.name.casefold().endswith(suffix.casefold())
        ]
    for path in paths:
        normalized = _key(path.stem)
        if normalized in indexed:
            duplicates.append(path.stem)
        else:
            indexed[normalized] = path
    if duplicates:
        raise ValueError(
            f"Duplicate {suffix} stems under {root}; first examples: {duplicates[:20]}"
        )
    return indexed


def _read_stems(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required stem list not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _load_model_a_splits(
    model_a_root: Path,
    old_graphs: dict[str, Path],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    split_stems: dict[str, list[str]] = {}
    stem_split: dict[str, str] = {}
    family_split: dict[str, str] = {}

    for split in SPLITS:
        stems = _read_stems(model_a_root / f"{split}.txt")
        split_stems[split] = stems
        for stem in stems:
            normalized = _key(stem)
            if normalized not in old_graphs:
                raise ValueError(
                    f"Model A {split}.txt references a missing graph: {stem}"
                )
            if normalized in stem_split:
                raise ValueError(
                    f"Model A stem occurs in more than one split: {stem}"
                )
            stem_split[normalized] = split

            family = _family_key(stem)
            previous = family_split.get(family)
            if previous is not None and previous != split:
                raise ValueError(
                    "Model A already has STEP-family leakage: "
                    f"family={family!r}, splits={previous}/{split}"
                )
            family_split[family] = split

    old_keys = set(old_graphs)
    listed_keys = set(stem_split)
    missing_from_splits = sorted(old_keys - listed_keys)
    missing_on_disk = sorted(listed_keys - old_keys)
    if missing_from_splits or missing_on_disk:
        raise ValueError(
            "Model A split lists must cover its graph directory exactly. "
            f"unlisted_graphs={len(missing_from_splits)}, "
            f"missing_graphs={len(missing_on_disk)}, "
            f"first_unlisted={missing_from_splits[:10]}, "
            f"first_missing={missing_on_disk[:10]}"
        )
    return split_stems, family_split


def _stable_new_family_split(family: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{family}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < 0.80:
        return "train"
    if fraction < 0.90:
        return "val"
    return "test"


def _load_graph(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _stock_label_check(path: Path) -> tuple[bool, str]:
    try:
        graph = _load_graph(path)
    except Exception as exc:  # pragma: no cover - depends on external files
        return False, f"load_error:{type(exc).__name__}:{exc}"
    labels = getattr(graph, "label_feature", None)
    if labels is None:
        return False, "missing_label_feature"
    labels = labels.detach().cpu().flatten()
    if labels.numel() == 0:
        return False, "empty_label_feature"
    unique = sorted(int(value) for value in torch.unique(labels).tolist())
    if unique != [0]:
        return False, f"non_stock_labels:{unique}"
    return True, ""


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_lines(path: Path, values: list[str]) -> None:
    _write_text_atomic(path, "".join(f"{value}\n" for value in values))


def _same_hard_link(source: Path, destination: Path) -> bool:
    try:
        return os.path.samefile(source, destination)
    except OSError:
        return False


def _materialize_graphs(
    graph_sources: list[GraphSource],
    output_pyg: Path,
    link_mode: str,
) -> tuple[int, int]:
    output_pyg.mkdir(parents=True, exist_ok=True)
    expected_names = {_key(item.path.name) for item in graph_sources}
    unexpected = [
        path.name
        for path in output_pyg.rglob("*.pt")
        if _key(path.name) not in expected_names
    ]
    if unexpected:
        raise ValueError(
            "Output pyg contains stale/unexpected graphs. Use a new empty output "
            f"directory or remove them explicitly. First examples: {unexpected[:20]}"
        )

    created = 0
    reused = 0
    for item in tqdm(graph_sources, desc=f"Materializing ({link_mode})", unit="graph"):
        destination = output_pyg / item.path.name
        if destination.exists():
            if link_mode == "hardlink":
                valid_existing = _same_hard_link(item.path, destination)
            else:
                valid_existing = (
                    destination.stat().st_size == item.path.stat().st_size
                    and destination.stat().st_mtime_ns == item.path.stat().st_mtime_ns
                )
            if not valid_existing:
                raise FileExistsError(
                    f"Existing output graph does not match its source: {destination}"
                )
            reused += 1
            continue

        try:
            if link_mode == "hardlink":
                os.link(item.path, destination)
            else:
                shutil.copy2(item.path, destination)
        except OSError as exc:
            if link_mode == "hardlink":
                raise OSError(
                    f"Hard-link creation failed for {item.path} -> {destination}. "
                    "The source and output must be on the same volume and the share "
                    "must support hard links. Use --link-mode copy only if enough "
                    "free space is available."
                ) from exc
            raise
        created += 1
    return created, reused


def _write_source_manifest(path: Path, sources: list[GraphSource]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "source_kind", "split", "source_path"))
        for item in sources:
            writer.writerow((item.stem, item.kind, item.split, str(item.path)))
    os.replace(temporary, path)


def _write_exclusion_manifest(
    path: Path,
    exclusions: list[tuple[str, str]],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "reason"))
        writer.writerows(exclusions)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve Model A exactly, append only full-stem-unique new ABC graphs, "
            "and create a strict stock-only evaluation holdout."
        )
    )
    parser.add_argument("--model-a-root", required=True, type=Path)
    parser.add_argument("--expanded-root", required=True, type=Path)
    parser.add_argument("--new-abc-json-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hard links consume no duplicate graph bytes and are recommended.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create the output. Without this flag the script performs a dry run.",
    )
    parser.add_argument(
        "--audit-stock-labels",
        action="store_true",
        help=(
            "Load every stock-holdout graph and require label_feature == 0. "
            "The PowerShell wrapper enables this automatically with -Apply."
        ),
    )
    args = parser.parse_args()

    model_a_root = args.model_a_root.resolve()
    expanded_root = args.expanded_root.resolve()
    new_json_root = args.new_abc_json_dir.resolve()
    output_root = args.output_root.resolve()
    model_a_pyg = model_a_root / "pyg"
    expanded_pyg = expanded_root / "pyg"

    if output_root in (model_a_root, expanded_root, model_a_pyg, expanded_pyg):
        raise ValueError("Output root must be different from every source directory.")
    if output_root.is_relative_to(model_a_pyg) or output_root.is_relative_to(expanded_pyg):
        raise ValueError("Output root cannot be inside a source pyg directory.")

    print("Indexing source files...")
    old_graphs = _index_files(model_a_pyg, ".pt")
    expanded_graphs = _index_files(expanded_pyg, ".pt")
    new_jsons = _index_files(new_json_root, ".json")
    old_splits, old_family_split = _load_model_a_splits(model_a_root, old_graphs)

    new_json_keys = set(new_jsons)
    expanded_keys = set(expanded_graphs)
    old_keys = set(old_graphs)
    missing_new_graphs = sorted(new_json_keys - expanded_keys)
    if missing_new_graphs:
        raise ValueError(
            "Some new ABC JSONs have no graph in the expanded dataset. "
            f"count={len(missing_new_graphs)}, first={missing_new_graphs[:20]}"
        )

    exact_duplicates = sorted(new_json_keys & old_keys)
    unique_new_keys = sorted(new_json_keys - old_keys)

    additions: dict[str, list[str]] = {split: [] for split in SPLITS}
    new_family_split: dict[str, str] = {}
    inherited_family_count = 0
    new_family_count = 0
    for normalized in unique_new_keys:
        stem = expanded_graphs[normalized].stem
        family = _family_key(stem)
        split = old_family_split.get(family)
        if split is not None:
            inherited_family_count += int(family not in new_family_split)
        else:
            split = new_family_split.get(family)
            if split is None:
                split = _stable_new_family_split(family, args.seed)
                new_family_count += 1
        new_family_split[family] = split
        additions[split].append(stem)

    final_splits: dict[str, list[str]] = {}
    final_stem_split: dict[str, str] = {}
    final_family_split: dict[str, str] = dict(old_family_split)
    for split in SPLITS:
        final_splits[split] = old_splits[split] + sorted(
            additions[split], key=str.casefold
        )
        for stem in final_splits[split]:
            normalized = _key(stem)
            if normalized in final_stem_split:
                raise ValueError(f"Final stem leakage detected: {stem}")
            final_stem_split[normalized] = split
            family = _family_key(stem)
            previous = final_family_split.get(family)
            if previous is not None and previous != split:
                raise ValueError(
                    f"Final STEP-family leakage detected: {family} in {previous}/{split}"
                )
            final_family_split[family] = split

    stock_list_path = expanded_root / "stock_only_test.txt"
    stock_list = _read_stems(stock_list_path)
    stock_exclusions: list[tuple[str, str]] = []
    strict_stock: list[str] = []
    seen_stock: set[str] = set()
    for listed_stem in tqdm(stock_list, desc="Auditing stock holdout", unit="graph"):
        normalized = _key(listed_stem)
        if normalized in seen_stock:
            stock_exclusions.append((listed_stem, "duplicate_in_stock_list"))
            continue
        seen_stock.add(normalized)
        source = expanded_graphs.get(normalized)
        if source is None:
            stock_exclusions.append((listed_stem, "missing_expanded_graph"))
            continue
        if normalized in new_json_keys:
            stock_exclusions.append(
                (listed_stem, "stem_collision_with_new_labeled_abc")
            )
            continue
        if normalized in final_stem_split:
            stock_exclusions.append((listed_stem, "training_split_leakage"))
            continue
        if args.audit_stock_labels:
            is_stock, reason = _stock_label_check(source)
            if not is_stock:
                stock_exclusions.append((listed_stem, reason))
                continue
        strict_stock.append(source.stem)

    graph_sources: list[GraphSource] = []
    for split in SPLITS:
        for stem in old_splits[split]:
            path = old_graphs[_key(stem)]
            graph_sources.append(GraphSource(path.stem, path, "model_a_replay", split))
        for stem in sorted(additions[split], key=str.casefold):
            path = expanded_graphs[_key(stem)]
            graph_sources.append(GraphSource(path.stem, path, "unique_new_abc", split))
    for stem in sorted(strict_stock, key=str.casefold):
        path = expanded_graphs[_key(stem)]
        graph_sources.append(GraphSource(path.stem, path, "stock_eval", "stock_eval"))

    source_key_counts = Counter(_key(item.stem) for item in graph_sources)
    duplicate_sources = [key for key, count in source_key_counts.items() if count > 1]
    if duplicate_sources:
        raise ValueError(
            f"Duplicate output graph stems detected: {duplicate_sources[:20]}"
        )

    summary = {
        "mode": "apply" if args.apply else "dry_run",
        "model_a_root": str(model_a_root),
        "expanded_root": str(expanded_root),
        "new_abc_json_dir": str(new_json_root),
        "output_root": str(output_root),
        "link_mode": args.link_mode,
        "seed": args.seed,
        "stock_graph_labels_audited": args.audit_stock_labels,
        "counts": {
            "model_a_graphs": len(old_graphs),
            "new_abc_jsons": len(new_jsons),
            "new_abc_present_in_expanded": len(new_json_keys),
            "new_abc_exact_duplicates_with_model_a": len(exact_duplicates),
            "unique_new_abc_added": len(unique_new_keys),
            "training_graphs": sum(len(final_splits[s]) for s in SPLITS),
            "stock_eval_listed": len(stock_list),
            "stock_eval_strict": len(strict_stock),
            "stock_eval_excluded": len(stock_exclusions),
            "total_output_graph_links": len(graph_sources),
            "inherited_new_step_families": inherited_family_count,
            "entirely_new_step_families": new_family_count,
        },
        "splits": {
            split: {
                "model_a": len(old_splits[split]),
                "unique_new_abc": len(additions[split]),
                "final": len(final_splits[split]),
            }
            for split in SPLITS
        },
        "stock_eval_exclusions": [
            {"stem": stem, "reason": reason} for stem, reason in stock_exclusions
        ],
    }

    print(json.dumps(summary, indent=2))
    if not args.apply:
        print("\nDry run passed; no files were modified.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    created, reused = _materialize_graphs(
        graph_sources, output_root / "pyg", args.link_mode
    )
    for split in SPLITS:
        _write_lines(output_root / f"{split}.txt", final_splits[split])
    _write_lines(
        output_root / "new_abc_unique_stems.txt",
        [expanded_graphs[key].stem for key in unique_new_keys],
    )
    _write_lines(
        output_root / "new_abc_exact_duplicates_with_model_a.txt",
        [new_jsons[key].stem for key in exact_duplicates],
    )
    _write_lines(
        output_root / "stock_only_test.txt",
        sorted(strict_stock, key=str.casefold),
    )
    _write_source_manifest(output_root / "graph_sources.csv", graph_sources)
    _write_exclusion_manifest(
        output_root / "stock_eval_exclusions.csv", stock_exclusions
    )
    summary["materialization"] = {
        "created": created,
        "reused": reused,
        "source_graph_logical_gib": round(
            sum(item.path.stat().st_size for item in graph_sources) / (1024**3), 3
        ),
        "note": (
            "Hard links share source bytes; treat linked source/output .pt files as "
            "immutable."
            if args.link_mode == "hardlink"
            else "Graphs were physically copied."
        ),
    }
    _write_text_atomic(
        output_root / "preparation_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    print(
        f"\nPrepared {len(graph_sources):,} graphs at {output_root}; "
        f"created={created:,}, reused={reused:,}."
    )
    print("The source datasets were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
