"""
generate_remaining_ids.py

Computes the set of priority IDs that still need STEP → SLDPRT processing:
    remaining = matched_files.json  MINUS  (IDs already in NEW sldprt folders)

Saves the result as a plain text file (one ID per line) to a network share
so Jenkins can read it during distribution.

Usage:
    python generate_remaining_ids.py
"""

import json
import os

# =====================================================================
# Configuration
# =====================================================================
JSON_FILE_PATH = r"C:\Users\RZA2\Downloads\matched_files.json"

NEW_DIRECTORIES = [
    r"\\Gr-sw26877\d\brepmfr_sldprts\cadsynth",
    r"\\Gr-sw34959\d\brepmfr_sldprts\cadsynth",
]

# Output: network share path all machines can access
OUTPUT_PATH = r"\\DZ4-SMR52-DSA\cadsynth_data\remaining_step_ids.txt"


def load_target_ids(json_path):
    """Load the JSON and extract the set of lowercase target IDs."""
    with open(json_path, "r") as f:
        data = json.load(f)

    target_ids = set()
    for j_file in data.get("matched_files", []):
        file_id = os.path.splitext(j_file)[0]
        target_ids.add(file_id.lower())

    return target_ids


def scan_sldprt_ids(directories, label):
    """Scan directories for SLDPRT files, return set of prefix IDs."""
    found_ids = set()
    print(f"\nScanning {label} folders...")
    for directory in directories:
        if not os.path.exists(directory):
            print(f"  [!] WARNING: Cannot access {directory}. Skipping.")
            continue
        print(f"  [-] Scanning {directory} ...")
        try:
            for f_name in os.listdir(directory):
                if f_name.lower().endswith(".sldprt"):
                    base_name = os.path.splitext(f_name)[0]
                    file_id = base_name.split("_")[0].lower()
                    found_ids.add(file_id)
        except Exception as e:
            print(f"  [!] Error scanning {directory}: {e}")
    return found_ids


def main():
    print(f"Loading target IDs from: {JSON_FILE_PATH}")
    target_ids = load_target_ids(JSON_FILE_PATH)
    print(f"  Total target IDs: {len(target_ids)}")

    new_ids = scan_sldprt_ids(NEW_DIRECTORIES, "NEW")

    remaining = target_ids - new_ids

    print(f"\n{'='*50}")
    print(f"  REMAINING IDs COMPUTATION")
    print(f"{'='*50}")
    print(f"  Total priority targets     : {len(target_ids)}")
    print(f"  Already done (NEW)         : {len(target_ids & new_ids)}")
    print(f"  ─────────────────────────────")
    print(f"  REMAINING (need processing): {len(remaining)}")
    print(f"{'='*50}\n")

    # Write to network share
    sorted_remaining = sorted(remaining)
    with open(OUTPUT_PATH, "w") as f:
        for rid in sorted_remaining:
            f.write(f"{rid}\n")

    print(f"Saved {len(sorted_remaining)} IDs to: {OUTPUT_PATH}")
    print("Jenkins can now read this file during distribution.\n")


if __name__ == "__main__":
    main()
