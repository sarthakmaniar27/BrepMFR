# -*- coding: utf-8 -*-
"""
merge_thread_json.py
====================

Merge two thread-JSON folders into a single destination:

    SOURCE 1 (new):  \\\\Gr-sw66464\\d\\thread_and_text\\root_json
    SOURCE 2 (old):  \\\\Gr-sw66464\\d\\threads\\json
    DEST (merged):   \\\\Gr-sw66464\\d\\thread_and_text\\thread_text_merged

Naming conventions (discovered by scanning both folders)
---------------------------------------------------------

The 8-digit zero-padded prefix is the **"main file"** identifier — i.e. the
source CAD part.  Everything after it identifies a **variation** that was
generated FROM that main part.  Duplicate detection therefore keys on the
8-digit part ID, not on the full filename.

NEW folder (root_json) — 42,074 files, 14,923 unique part IDs
    Pattern A  (3 tokens, 14,917 files):  {id:08d}_engrave_{faces}.json
        e.g. 00000000_engrave_102.json
        One file per part ID, no version suffix.  ``engrave`` is the category.

    Pattern B  (4 tokens, 27,157 files):  {id:08d}_{type}_v{N}_{faces}.json
        e.g. 00000000_both_v3_104.json
             00000000_thread_v4_104.json
        ``type`` ∈ {both, thread};  ``v{N}`` is the variation index (v1..v6).

OLD folder (threads\\json) — 34,531 files, 5,335 unique part IDs
    Pattern C  (3 tokens, 34,531 files):  {id:08d}_{index}_{faces}.json
        e.g. 00000000_1_104.json
        ``index`` ∈ {1..6};  every file follows this single pattern.

The trailing ``{faces}`` token (102..544) is the per-part face count and
should be consistent for a given part ID across all of its files AND across
both folders — a mismatch flags a corrupt or mismatched part.

How to use
----------

1. Run the script.  It first prints the duplicate / overlap report to
   stdout and writes a full per-part overlap list to
   ``thread_text_merged/../overlap_report.txt``.

2. The merge then runs automatically: every ``.json`` from both source
   folders is copied into ``thread_text_merged`` (existing files skipped).
   To disable the merge and run *only* the duplicate report, comment out
   the ``merge_folders()`` call at the bottom of the script.

Label-scheme compatibility (verified 2026-06-26)
-------------------------------------------------

Both folders use the same ``face[].label`` field but with different class
counts:

- OLD (``threads\\json``): labels ``{0=stock, 1=thread}``  — 2-class
  threads-only dataset.  Source SolidWorks labels ``{-1, 0}``
  (stock) and ``70`` (thread) were remapped via
  ``repair_thread_json_labels.py``.  The text label ``101`` was never
  present in the OLD source parts.
- NEW (``root_json``): labels ``{0=stock, 1=thread, 2=text}``  — 3-class
  thread+text dataset.  Source labels ``{-1, 0, 70, 101}`` were remapped
  via ``repair_json_face_labels.py`` using
  ``remap_maps/thread_text_sw_to_brep.json``.

The two schemes are **compatible**: OLD ``label=1`` and NEW ``label=1``
both mean *genuine thread*.  OLD data simply has no text faces, so it
contributes nothing to class 2 — which is the intended behaviour given
that the goal is to add more *thread* training data.

Train the merged dataset with ``--num_classes 3``.  Do NOT use
``--num_classes 2``: NEW files contain ``label=2`` which would make
``F.one_hot(labels, 2)`` index out of bounds and crash the model.

After merging, regenerate ``train.txt`` / ``val.txt`` / ``test.txt``
from the merged folder (e.g. via ``scripts/threads/make_random_splits.py``)
before starting training.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_JSON_DIR = Path(r"\\Gr-sw66464\d\thread_and_text\root_json")   # NEW
OLD_JSON_DIR  = Path(r"\\Gr-sw66464\d\threads\json")                # OLD
MERGED_DIR    = Path(r"\\Gr-sw66464\d\thread_and_text\thread_text_merged")
REPORT_PATH   = Path(r"\\Gr-sw66464\d\thread_and_text\overlap_report.txt")

# ---------------------------------------------------------------------------
# Filename patterns
# ---------------------------------------------------------------------------
# NEW pattern A: 00000000_engrave_102.json
NEW_ENGRAVE_RE = re.compile(r"^(\d{8})_engrave_(\d+)\.json$")
# NEW pattern B: 00000000_both_v3_104.json  /  00000000_thread_v4_104.json
NEW_VARIANT_RE = re.compile(r"^(\d{8})_(both|thread)_v(\d+)_(\d+)\.json$")
# OLD pattern  : 00000000_1_104.json
OLD_INDEX_RE   = re.compile(r"^(\d{8})_(\d+)_(\d+)\.json$")


# ---------------------------------------------------------------------------
# Scanners — build {part_id: [file_records]} for each folder
# ---------------------------------------------------------------------------
def scan_new(folder: Path):
    """Return (by_id, unrecognized).

    by_id[part_id] = list of (filename, category, variation_or_None, faces)
    """
    by_id: dict[str, list[tuple[str, str, int | None, int]]] = defaultdict(list)
    unrecognized: list[str] = []
    for entry in os.scandir(folder):
        name = entry.name
        if not name.endswith(".json"):
            continue
        m = NEW_ENGRAVE_RE.match(name)
        if m:
            pid, faces = m.group(1), int(m.group(2))
            by_id[pid].append((name, "engrave", None, faces))
            continue
        m = NEW_VARIANT_RE.match(name)
        if m:
            pid, typ, var, faces = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            by_id[pid].append((name, typ, var, faces))
            continue
        unrecognized.append(name)
    return by_id, unrecognized


def scan_old(folder: Path):
    """Return (by_id, unrecognized).

    by_id[part_id] = list of (filename, index, faces)
    """
    by_id: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    unrecognized: list[str] = []
    for entry in os.scandir(folder):
        name = entry.name
        if not name.endswith(".json"):
            continue
        m = OLD_INDEX_RE.match(name)
        if m:
            pid, idx, faces = m.group(1), int(m.group(2)), int(m.group(3))
            by_id[pid].append((name, idx, faces))
            continue
        unrecognized.append(name)
    return by_id, unrecognized


# ---------------------------------------------------------------------------
# Duplicate / overlap report  (ACTIVE)
# ---------------------------------------------------------------------------
def report_duplicates() -> None:
    print("=" * 78)
    print("  THREAD JSON — DUPLICATE / OVERLAP REPORT")
    print("=" * 78)

    # ---- scan new ----------------------------------------------------------
    print(f"\n[1/2] Scanning NEW  : {ROOT_JSON_DIR}")
    t0 = time.time()
    new_by_id, new_unrec = scan_new(ROOT_JSON_DIR)
    new_ids = set(new_by_id.keys())
    new_total = sum(len(v) for v in new_by_id.values())
    print(f"      scanned {new_total:,} files in {time.time()-t0:.1f}s")
    print(f"      unique part IDs : {len(new_ids):,}")
    print(f"      unrecognized    : {len(new_unrec)}")
    if new_unrec:
        print(f"      sample unrecognized: {new_unrec[:5]}")

    new_cat = Counter()
    new_var_per_cat: dict[str, Counter] = defaultdict(Counter)
    for files in new_by_id.values():
        for _, cat, var, _ in files:
            new_cat[cat] += 1
            new_var_per_cat[cat][var] += 1
    print(f"      category counts : {dict(new_cat)}")
    for cat, vars in new_var_per_cat.items():
        if cat != "engrave":
            print(f"        {cat} variation ids: {dict(sorted(vars.items()))}")

    # ---- scan old ----------------------------------------------------------
    print(f"\n[2/2] Scanning OLD  : {OLD_JSON_DIR}")
    t0 = time.time()
    old_by_id, old_unrec = scan_old(OLD_JSON_DIR)
    old_ids = set(old_by_id.keys())
    old_total = sum(len(v) for v in old_by_id.values())
    print(f"      scanned {old_total:,} files in {time.time()-t0:.1f}s")
    print(f"      unique part IDs : {len(old_ids):,}")
    print(f"      unrecognized    : {len(old_unrec)}")
    if old_unrec:
        print(f"      sample unrecognized: {old_unrec[:5]}")

    old_idx = Counter()
    for files in old_by_id.values():
        for _, idx, _ in files:
            old_idx[idx] += 1
    print(f"      index counts    : {dict(sorted(old_idx.items()))}")

    # ---- overlap -----------------------------------------------------------
    overlap_ids   = new_ids & old_ids
    only_new_ids  = new_ids - old_ids
    only_old_ids  = old_ids - new_ids

    print("\n" + "=" * 78)
    print("  OVERLAP SUMMARY  (a 'duplicate' = same 8-digit part ID in both)")
    print("=" * 78)
    print(f"  Parts in BOTH folders : {len(overlap_ids):>8,}")
    print(f"  Parts only in NEW     : {len(only_new_ids):>8,}")
    print(f"  Parts only in OLD     : {len(only_old_ids):>8,}")
    print(f"  Unique parts after merge : {len(new_ids | old_ids):>6,}")

    overlap_new_files = sum(len(new_by_id[pid]) for pid in overlap_ids)
    overlap_old_files = sum(len(old_by_id[pid]) for pid in overlap_ids)
    print(f"\n  Files belonging to overlapping parts:")
    print(f"      in NEW : {overlap_new_files:,}")
    print(f"      in OLD : {overlap_old_files:,}")
    print(f"      total  : {overlap_new_files + overlap_old_files:,}"
          f"  (these are the 'duplicate' files — same part, two pipelines)")

    # ---- files-per-ID distribution for overlapping parts ------------------
    print("\n  Files-per-part for overlapping IDs (NEW vs OLD):")
    new_counts = Counter(len(new_by_id[pid]) for pid in overlap_ids)
    old_counts = Counter(len(old_by_id[pid]) for pid in overlap_ids)
    print(f"      NEW files-per-part histogram: {dict(sorted(new_counts.items()))}")
    print(f"      OLD files-per-part histogram: {dict(sorted(old_counts.items()))}")

    # ---- face-count consistency -------------------------------------------
    print("\n  Face-count consistency check on overlapping parts:")
    mismatches = []
    for pid in overlap_ids:
        new_faces = {f[3] for f in new_by_id[pid]}
        old_faces = {f[2] for f in old_by_id[pid]}
        if new_faces != old_faces:
            mismatches.append((pid, sorted(new_faces), sorted(old_faces)))
    print(f"      parts with mismatched face counts: {len(mismatches)}")
    for pid, nf, of in mismatches[:10]:
        print(f"        {pid}: new={nf}  old={of}")

    # ---- sample overlapping parts -----------------------------------------
    print("\n  Sample of 20 overlapping parts:")
    print(f"    {'PartID':<10} {'#NEW':>5} {'#OLD':>5}  {'NEW categories':<28} {'OLD indices'}")
    for pid in sorted(overlap_ids)[:20]:
        nf = new_by_id[pid]
        of = old_by_id[pid]
        cats = sorted({c for _, c, _, _ in nf})
        idxs = sorted({i for _, i, _ in of})
        print(f"    {pid:<10} {len(nf):>5} {len(of):>5}  {','.join(cats):<28} {idxs}")

    # ---- write full overlap list to a file --------------------------------
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            fh.write("overlap_report.txt\n")
            fh.write("=" * 78 + "\n")
            fh.write(f"NEW folder : {ROOT_JSON_DIR}\n")
            fh.write(f"OLD folder : {OLD_JSON_DIR}\n")
            fh.write(f"NEW files  : {new_total:,}  ({len(new_ids):,} parts)\n")
            fh.write(f"OLD files  : {old_total:,}  ({len(old_ids):,} parts)\n")
            fh.write(f"Overlap    : {len(overlap_ids):,} parts\n")
            fh.write(f"  -> {overlap_new_files:,} files in NEW\n")
            fh.write(f"  -> {overlap_old_files:,} files in OLD\n")
            fh.write(f"Unique parts after merge : {len(new_ids | old_ids):,}\n\n")
            fh.write(f"{'PartID':<10}{'#NEW':>6}{'#OLD':>6}  {'NEW cats':<28}{'OLD idxs'}\n")
            fh.write("-" * 78 + "\n")
            for pid in sorted(overlap_ids):
                nf = new_by_id[pid]
                of = old_by_id[pid]
                cats = sorted({c for _, c, _, _ in nf})
                idxs = sorted({i for _, i, _ in of})
                fh.write(f"{pid:<10}{len(nf):>6}{len(of):>6}  {','.join(cats):<28}{idxs}\n")
        print(f"\n  Full per-part overlap list written to:\n    {REPORT_PATH}")
    except Exception as exc:
        print(f"\n  WARN: could not write report file: {exc}")

    print("\n" + "=" * 78)
    print("  Duplicate report done.  Merge will start next — see output below.")
    print("  (To skip the merge, comment out merge_folders() at the bottom")
    print("   of this script and re-run.)")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Copy / merge  (ACTIVE — runs after the duplicate report)
# ---------------------------------------------------------------------------
def merge_folders() -> None:
    """
    Copy every .json from ROOT_JSON_DIR and OLD_JSON_DIR into MERGED_DIR.
    Filenames from the two sources do not collide (different patterns),
    but we still guard against accidental overwrites by skipping files
    that already exist in the destination.
    """
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    copied, skipped, failed = 0, 0, 0
    t0 = time.time()

    for src_dir, label in [(ROOT_JSON_DIR, "NEW"), (OLD_JSON_DIR, "OLD")]:
        print(f"\n--- Copying from {label}: {src_dir}")
        files = [e for e in os.scandir(src_dir) if e.name.endswith(".json")]
        n_files = len(files)
        for i, entry in enumerate(files, 1):
            dst = MERGED_DIR / entry.name
            try:
                if dst.exists():
                    skipped += 1
                    continue
                shutil.copy2(entry.path, dst)
                copied += 1
            except Exception as exc:
                failed += 1
                if failed <= 10:
                    print(f"  FAIL: {entry.name} -> {exc}")
            if i % 5000 == 0:
                print(f"  {label}: {i:,}/{n_files:,} processed "
                      f"({copied:,} copied, {skipped:,} skipped)")

    print(f"\nMerge complete in {time.time()-t0:.1f}s")
    print(f"  copied   : {copied:,}")
    print(f"  skipped  : {skipped:,}  (already existed in destination)")
    print(f"  failed   : {failed:,}")
    print(f"  destination : {MERGED_DIR}")
    final = sum(1 for _ in os.scandir(MERGED_DIR) if _.name.endswith(".json"))
    print(f"  final file count : {final:,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report_duplicates()
    merge_folders()
