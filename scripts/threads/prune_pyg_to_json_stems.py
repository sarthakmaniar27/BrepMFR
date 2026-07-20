#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delete ``.pt`` / label JSON whose stems are NOT in the given JSON folders.

Use after a conversion when ``lite/pyg`` still contains leftover graphs from
older runs (those inflate train/val/test and can include corrupt ``.pt`` files).

Example:

  python scripts/threads/prune_pyg_to_json_stems.py `
    --pyg-dir Z:/thread_and_text/lite/pyg `
    --label-dir Z:/thread_and_text/lite/label `
    --json-dir Z:/thread_and_text/root_json `
    --json-dir Z:/thread_and_text/abc_json
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _json_stems(dirs: list[Path]) -> set[str]:
    stems: set[str] = set()
    for d in dirs:
        paths = sorted(d.glob("*.json"))
        if not paths:
            paths = sorted(d.rglob("*.json"))
        stems.update(p.stem for p in paths)
    return stems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pyg-dir", type=Path, required=True)
    ap.add_argument("--label-dir", type=Path, default=None)
    ap.add_argument(
        "--json-dir",
        type=Path,
        action="append",
        required=True,
        help="Keep stems that appear in these folders (repeatable).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pyg = args.pyg_dir.resolve()
    if not pyg.is_dir():
        raise SystemExit(f"Not a directory: {pyg}")

    json_dirs = [p.resolve() for p in args.json_dir]
    for d in json_dirs:
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")

    keep = _json_stems(json_dirs)
    print(f"Keep stems from JSON dirs: {len(keep):,}")

    pts = sorted(pyg.glob("*.pt"))
    orphan_pts = [p for p in pts if p.stem not in keep]
    print(f"PyG .pt total: {len(pts):,}  |  orphan (delete): {len(orphan_pts):,}  |  keep: {len(pts) - len(orphan_pts):,}")

    orphan_labels: list[Path] = []
    if args.label_dir is not None:
        lab = args.label_dir.resolve()
        if lab.is_dir():
            labels = sorted(lab.glob("*.json"))
            orphan_labels = [p for p in labels if p.stem not in keep]
            print(
                f"Label JSON total: {len(labels):,}  |  orphan (delete): {len(orphan_labels):,}  |  "
                f"keep: {len(labels) - len(orphan_labels):,}"
            )

    if args.dry_run:
        print("Dry-run only; no deletes.")
        if orphan_pts:
            print("First orphan .pt:", orphan_pts[0].name)
        return

    deleted = 0
    for p in orphan_pts:
        p.unlink(missing_ok=True)
        deleted += 1
    for p in orphan_labels:
        p.unlink(missing_ok=True)
        deleted += 1
    print(f"Deleted {deleted:,} files.")


if __name__ == "__main__":
    main()
