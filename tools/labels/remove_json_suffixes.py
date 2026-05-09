"""
remove_json_suffixes.py
=======================
Renames all JSON files across all 5 category folders by removing
the trailing numeric suffix.

  nut_1_104.json   → nut_1.json
  bolt_2_102.json  → bolt_2.json
  bearing_5_103.json → bearing_5.json
"""

import os
import re
from pathlib import Path

ROOT = r"C:\Users\smr52\Desktop\ToolBoxParts_Dataset"
DRY_RUN = False  # Set to False to actually rename

CATEGORIES = ["Bearings", "Bolts", "Gears", "Nuts", "Screws"]

def main():
    print(f"DRY_RUN = {DRY_RUN}")
    print("=" * 60)

    for category in CATEGORIES:
        jsons_dir = Path(ROOT) / category / "jsons"

        if not jsons_dir.exists():
            print(f"[SKIP] Not found: {jsons_dir}")
            continue

        print(f"\n{category}:")
        renamed = 0
        skipped = 0
        conflicts = 0

        for f in sorted(jsons_dir.glob("*.json")):
            # Match "anything_digits.json" — strip the last _digits
            match = re.match(r'^(.+)_(\d+)\.json$', f.name)
            if not match:
                print(f"  [SKIP] No suffix found: {f.name}")
                skipped += 1
                continue

            new_name = match.group(1) + ".json"  # e.g. nut_1.json
            new_path = f.parent / new_name

            if new_path.exists():
                print(f"  [CONFLICT] {f.name} → {new_name} already exists!")
                conflicts += 1
                continue

            if DRY_RUN:
                print(f"  [PREVIEW] {f.name} → {new_name}")
            else:
                f.rename(new_path)
                renamed += 1

        print(f"  Renamed: {renamed}  Skipped: {skipped}  Conflicts: {conflicts}")

    print("\nDone.")

if __name__ == "__main__":
    main()