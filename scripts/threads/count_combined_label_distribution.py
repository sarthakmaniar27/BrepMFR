#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face-label distribution for **multiple** PyG roots (e.g. thread-only + thread+text).

Each source is scanned independently; a **combined** histogram is printed plus optional
stem overlap between sources (same ``.pt`` stem in two folders).

Typical use (PowerShell, from repo root)::

  conda run -n brep_mfr_pyg python scripts/threads/count_combined_label_distribution.py `
    --source threads_only=D:/threads/lite/pyg `
    --source thread_text=D:/thread_and_text/lite/pyg `
    --group "0:stock,1:thread,2:text"

If a path has no ``*.pt`` at the top level, the script also tries ``<path>/pyg``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_kw):
        return it


DEFAULT_GROUP = "0:stock,1:thread,2:text"


@dataclass
class SourceStats:
    name: str
    root: Path
    graph_count: int = 0
    graphs_empty_labels: int = 0
    graphs_bad_labels: int = 0
    face_counter: Counter = field(default_factory=Counter)
    stems: Set[str] = field(default_factory=set)


def _parse_group_spec(spec: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --group segment (need id:name): {part!r}")
        lid_s, name = part.split(":", 1)
        out[int(lid_s.strip())] = name.strip()
    if not out:
        raise ValueError("Empty --group")
    return out


def _resolve_pyg_root(path: Path) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")
    top_pts = list(path.glob("*.pt"))
    if top_pts:
        return path
    nested = path / "pyg"
    if nested.is_dir() and list(nested.glob("*.pt")):
        return nested
    # allow rglob-only trees (no top-level pt but files deeper)
    if any(path.rglob("*.pt")):
        return path
    raise FileNotFoundError(f"No .pt graphs under {path} or {path / 'pyg'}")


def _parse_source_arg(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"--source must be NAME=PATH, got: {raw!r}")
    name, p = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty source name in: {raw!r}")
    return name, Path(p.strip())


def _iter_pt_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.pt"))


def scan_pyg_source(name: str, root: Path, *, max_files: int = 0) -> SourceStats:
    import torch

    scan_root = _resolve_pyg_root(root)
    paths = _iter_pt_files(scan_root)
    if max_files > 0:
        paths = paths[: max_files]

    stats = SourceStats(name=name, root=scan_root)
    for pp in tqdm(paths, desc=name, unit="graph"):
        stats.graph_count += 1
        stats.stems.add(pp.stem)
        try:
            g = torch.load(pp, map_location="cpu", weights_only=False)
            lf = getattr(g, "label_feature", None)
            if lf is None or lf.numel() == 0:
                stats.graphs_empty_labels += 1
                continue
            arr = lf.detach().cpu().numpy().ravel()
            for v in arr.tolist():
                stats.face_counter[int(v)] += 1
        except Exception as e:
            stats.graphs_bad_labels += 1
            print(f"[WARN] {name} skip {pp}: {e}", file=sys.stderr)
    return stats


def _sort_key(k):
    if isinstance(k, str):
        return (1, k)
    return (0, k)


def _print_counter(title: str, counter: Counter, group: Optional[dict[int, str]]) -> int:
    total = sum(counter.values())
    print(f"\n{'=' * 60}")
    print(title)
    print(f"Total labeled faces: {total:,}\n")
    for k in sorted(counter.keys(), key=_sort_key):
        pct = 100.0 * counter[k] / total if total else 0.0
        print(f"  label {k!s:>8}: {counter[k]:>12,}  ({pct:5.2f}%)")
    if group:
        print("\n--- Grouped ---")
        for lid in sorted(group.keys()):
            n = int(counter.get(lid, 0))
            pct = 100.0 * n / total if total else 0.0
            print(f"  {group[lid]} ({lid}): {n:>12,}  ({pct:5.2f}%)")
        other = sum(
            int(counter[k])
            for k in counter
            if isinstance(k, int) and k not in group
        )
        if other:
            pct = 100.0 * other / total if total else 0.0
            print(f"  (other int labels): {other:>12,}  ({pct:5.2f}%)")
    return total


def _print_source_meta(stats: SourceStats) -> None:
    print(f"\n--- {stats.name} ---")
    print(f"  scan root:     {stats.root}")
    print(f"  graphs:        {stats.graph_count:,}")
    print(f"  unique stems:  {len(stats.stems):,}")
    if stats.graphs_empty_labels:
        print(f"  empty labels:  {stats.graphs_empty_labels:,}")
    if stats.graphs_bad_labels:
        print(f"  load errors:   {stats.graphs_bad_labels:,}")


def _stem_overlap(all_stats: List[SourceStats]) -> None:
    if len(all_stats) < 2:
        return
    stem_to_sources: Dict[str, List[str]] = defaultdict(list)
    for st in all_stats:
        for s in st.stems:
            stem_to_sources[s].append(st.name)
    dupes = {s: names for s, names in stem_to_sources.items() if len(names) > 1}
    print(f"\n{'=' * 60}")
    print("Stem overlap (same graph stem in multiple sources)")
    print(f"  duplicated stems: {len(dupes):,}")
    if dupes:
        preview = list(dupes.items())[:10]
        for stem, names in preview:
            print(f"    {stem}: {', '.join(names)}")
        if len(dupes) > 10:
            print(f"    ... and {len(dupes) - 10:,} more")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combined face-label distribution across multiple PyG directories."
    )
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeatable. Example: --source threads_only=D:/threads/lite/pyg",
    )
    ap.add_argument(
        "--group",
        type=str,
        default=DEFAULT_GROUP,
        help=f'Named buckets (default: "{DEFAULT_GROUP}").',
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Cap graphs per source (0 = all). Smoke-test only.",
    )
    args = ap.parse_args()

    try:
        group = _parse_group_spec(args.group) if args.group else None
    except ValueError as e:
        raise SystemExit(f"--group: {e}") from e

    parsed: List[Tuple[str, Path]] = []
    for raw in args.source:
        try:
            parsed.append(_parse_source_arg(raw))
        except ValueError as e:
            raise SystemExit(str(e)) from e

    names = [n for n, _ in parsed]
    if len(names) != len(set(names)):
        raise SystemExit("Duplicate --source names are not allowed.")

    all_stats: List[SourceStats] = []
    for name, path in parsed:
        try:
            st = scan_pyg_source(name, path, max_files=int(args.max_files))
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from e
        all_stats.append(st)
        _print_source_meta(st)
        _print_counter(f"{name} — face labels", st.face_counter, group)

    combined: Counter = Counter()
    for st in all_stats:
        combined.update(st.face_counter)
    total_graphs = sum(st.graph_count for st in all_stats)
    print(f"\n{'=' * 60}")
    print(f"COMBINED ({len(all_stats)} sources, {total_graphs:,} graphs)")
    _print_counter("Combined face labels (union of all sources)", combined, group)

    _stem_overlap(all_stats)

    if len(all_stats) == 2 and group:
        a, b = all_stats[0], all_stats[1]
        ta = sum(a.face_counter.values())
        tb = sum(b.face_counter.values())
        tc = sum(combined.values())
        if tc:
            t1_a = int(a.face_counter.get(1, 0))
            t1_b = int(b.face_counter.get(1, 0))
            print(f"\n--- Merge impact (thread class 1 only) ---")
            print(f"  {a.name}: thread faces {t1_a:,} / {ta:,} ({100*t1_a/ta:.2f}% of its faces)")
            print(f"  {b.name}: thread faces {t1_b:,} / {tb:,} ({100*t1_b/tb:.2f}% of its faces)")
            print(
                f"  combined: thread faces {t1_a+t1_b:,} / {tc:,} "
                f"({100*(t1_a+t1_b)/tc:.2f}% of all faces)"
            )


if __name__ == "__main__":
    main()
