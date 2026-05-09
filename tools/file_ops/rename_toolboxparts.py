"""
rename_toolbox_parts.py
=======================
Renames all .SLDPRT files and their corresponding .json files
across all 5 category folders under the root dataset directory.

Naming convention after rename:
  Bearings → bearing_1.sldprt, bearing_2.sldprt, ...
  Bolts    → bolt_1.sldprt,    bolt_2.sldprt,    ...
  Gears    → gear_1.sldprt,    gear_2.sldprt,    ...
  Nuts     → nut_1.sldprt,     nut_2.sldprt,     ...
  Screws   → screw_1.sldprt,   screw_2.sldprt,   ...

JSON files are renamed to match their corresponding part:
  bearing_1_104.json, bolt_3_102.json, etc.

A CSV mapping file is saved next to this script for full traceability.

Usage:
  1. Set DRY_RUN = True  first to preview all renames without touching files.
  2. Set DRY_RUN = False to execute the actual rename.
"""

import os
import re
import csv
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
ROOT     = r"C:\Users\smr52\Desktop\ToolBoxParts_Dataset"
DRY_RUN  = False   # ← Set to False to actually rename files

CATEGORIES = {
    "Bearings": "bearing",
    "Bolts":    "bolt",
    "Gears":    "gear",
    "Nuts":     "nut",
    "Screws":   "screw",
}

# Subfolder names inside each category
PARTS_SUBDIR = "parts"
JSONS_SUBDIR = "jsons"

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def parse_json_name(json_filename):
    """
    Split a JSON filename into (original_part_stem, number_suffix).

    e.g.
      "AS_ISO 15 RBB - 12 - 10,DE,AC,10_68_104.json"
        → ("AS_ISO 15 RBB - 12 - 10,DE,AC,10_68", "104")

      "CSBOLT 0.2500-20x0.5x0.5-NS-C_102.json"
        → ("CSBOLT 0.2500-20x0.5x0.5-NS-C", "102")

    Returns (None, None) if the pattern doesn't match.
    """
    name_no_ext = Path(json_filename).stem          # strip .json
    match = re.match(r'^(.+)_(\d+)$', name_no_ext) # split at last _<digits>
    if match:
        return match.group(1), match.group(2)
    return None, None


def safe_rename(src: Path, dst: Path, dry_run: bool):
    """Rename src → dst, skipping if src == dst or dst already exists."""
    if src == dst:
        return "same"
    if dst.exists():
        return f"CONFLICT — destination already exists: {dst.name}"
    if not dry_run:
        src.rename(dst)
    return "ok"

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def process_category(category: str, prefix: str, log_rows: list, dry_run: bool):
    parts_dir = Path(ROOT) / category / PARTS_SUBDIR
    jsons_dir = Path(ROOT) / category / JSONS_SUBDIR

    print(f"\n{'='*65}")
    print(f"  Category : {category}  (prefix='{prefix}')")
    print(f"  Parts    : {parts_dir}")
    print(f"  JSONs    : {jsons_dir}")
    print(f"{'='*65}")

    # ── Collect & sort all SLDPRT files ─────────────────────
    if not parts_dir.exists():
        print(f"  [ERROR] Parts folder not found — skipping.")
        return

    all_parts = sorted(
        [f for f in parts_dir.iterdir() if f.suffix.lower() == ".sldprt"],
        key=lambda f: f.name.lower()
    )
    print(f"  Parts found : {len(all_parts)}")

    # ── Build stem → new_name mapping ───────────────────────
    # key   = original stem, lower-cased  (for case-insensitive match)
    # value = (new_stem, original_Path)
    stem_map = {}
    for idx, part_file in enumerate(all_parts, start=1):
        original_stem = part_file.stem
        new_stem      = f"{prefix}_{idx}"
        stem_map[original_stem.lower()] = (new_stem, part_file)

    # ── Rename JSONs first ───────────────────────────────────
    json_ok      = 0
    json_warn    = 0
    json_skipped = 0

    if jsons_dir.exists():
        json_files = sorted(
            [f for f in jsons_dir.iterdir() if f.suffix.lower() == ".json"],
            key=lambda f: f.name.lower()
        )
        print(f"  JSONs found : {len(json_files)}")

        for jf in json_files:
            orig_part_stem, num_suffix = parse_json_name(jf.name)

            if orig_part_stem is None:
                print(f"  [WARN ] Cannot parse JSON name: {jf.name}")
                log_rows.append([category, "json", jf.name, "", "WARN: unparseable name"])
                json_warn += 1
                continue

            lookup = orig_part_stem.lower()
            if lookup not in stem_map:
                print(f"  [WARN ] No matching part for JSON: {jf.name}")
                log_rows.append([category, "json", jf.name, "", "WARN: no matching part"])
                json_warn += 1
                continue

            new_stem, _ = stem_map[lookup]
            new_json_name = f"{new_stem}_{num_suffix}.json"
            new_json_path = jsons_dir / new_json_name

            result = safe_rename(jf, new_json_path, dry_run)
            if result == "same":
                json_skipped += 1
                log_rows.append([category, "json", jf.name, new_json_name, "already correct"])
            elif result == "ok":
                json_ok += 1
                log_rows.append([category, "json", jf.name, new_json_name, "renamed"])
            else:
                print(f"  [WARN ] {result}")
                log_rows.append([category, "json", jf.name, new_json_name, result])
                json_warn += 1

        print(f"  JSONs → renamed:{json_ok}  already_correct:{json_skipped}  warnings:{json_warn}")
    else:
        print(f"  [INFO ] JSON folder not found — skipping JSON rename.")

    # ── Rename parts ─────────────────────────────────────────
    parts_ok      = 0
    parts_skipped = 0
    parts_warn    = 0

    for idx, part_file in enumerate(all_parts, start=1):
        new_stem      = f"{prefix}_{idx}"
        new_part_name = f"{new_stem}.SLDPRT"
        new_part_path = parts_dir / new_part_name

        result = safe_rename(part_file, new_part_path, dry_run)
        if result == "same":
            parts_skipped += 1
            log_rows.append([category, "part", part_file.name, new_part_name, "already correct"])
        elif result == "ok":
            parts_ok += 1
            log_rows.append([category, "part", part_file.name, new_part_name, "renamed"])
        else:
            print(f"  [WARN ] {result}")
            log_rows.append([category, "part", part_file.name, new_part_name, result])
            parts_warn += 1

    print(f"  Parts → renamed:{parts_ok}  already_correct:{parts_skipped}  warnings:{parts_warn}")


def main():
    print(f"\n{'#'*65}")
    print(f"  ToolBoxParts Rename Script")
    print(f"  ROOT     : {ROOT}")
    print(f"  DRY_RUN  : {DRY_RUN}")
    if DRY_RUN:
        print(f"  ** DRY RUN — no files will be changed **")
    print(f"{'#'*65}")

    log_rows = []  # for CSV

    for category, prefix in CATEGORIES.items():
        process_category(category, prefix, log_rows, DRY_RUN)

    # ── Write mapping CSV ────────────────────────────────────
    csv_path = Path(ROOT) / "rename_mapping.csv"
    if not DRY_RUN:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Type", "Original Name", "New Name", "Status"])
            writer.writerows(log_rows)
        print(f"\n  Mapping CSV saved → {csv_path}")
    else:
        # Save dry-run preview CSV with a different name
        csv_path = Path(ROOT) / "rename_mapping_PREVIEW.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Type", "Original Name", "New Name", "Status"])
            writer.writerows(log_rows)
        print(f"\n  Preview CSV saved → {csv_path}")

    print(f"\n  Total files logged : {len(log_rows)}")
    print(f"  Done.\n")


if __name__ == "__main__":
    main()