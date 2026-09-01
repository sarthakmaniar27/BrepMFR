"""
transfer_uv_jsons.py

Copies UV JSON files from a network source to a destination share
using robocopy for fast, multi-threaded transfers.

First determines the ~5K matching prefix IDs by intersecting
matched_files.json with the NEW sldprt directories, then transfers
the corresponding .json files from the UV JSON source.

Usage:
    python transfer_uv_jsons.py              # Dry run (default) — just counts
    python transfer_uv_jsons.py --execute    # Actually copies via robocopy
"""

import argparse
import json
import os
import subprocess
import time

# =====================================================================
# Configuration
# =====================================================================
JSON_FILE_PATH = r"C:\Users\RZA2\Downloads\matched_files.json"

# NEW sldprt directories (used to determine the ~5K matching IDs)
SLDPRT_DIRECTORIES = [
    r"\\Gr-sw26877\d\brepmfr_sldprts\cadsynth",
    r"\\Gr-sw34959\d\brepmfr_sldprts\cadsynth",
]

# UV JSON source
UV_JSON_SOURCE = r"\\dz4-smr52-dsa\cadsynth_data\sw_cadsynth\uv_json"

# Destination
DESTINATION = r"\\Gr-sw36912\c\Threads\conversion\uv_jsons"

# Robocopy settings
ROBOCOPY_THREADS = 16
ROBOCOPY_RETRIES = 3
ROBOCOPY_WAIT    = 5
BATCH_SIZE       = 200


def load_target_ids(json_path):
    """Load the JSON and extract the set of lowercase target IDs."""
    with open(json_path, "r") as f:
        data = json.load(f)

    target_ids = set()
    for j_file in data.get("matched_files", []):
        file_id = os.path.splitext(j_file)[0]
        target_ids.add(file_id.lower())

    return target_ids


def get_new_sldprt_ids(directories):
    """Scan NEW sldprt directories and return the set of prefix IDs found."""
    found_ids = set()

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


def run_robocopy_batches(source_dir, filenames, destination, dry_run=True):
    """
    Run robocopy in batches for a single source directory.
    In dry-run mode, adds /L (list-only, no copy).
    Returns (copied, skipped, errors) counts.
    """
    total_copied = 0
    total_skipped = 0
    total_errors = 0

    batches = [filenames[i:i + BATCH_SIZE] for i in range(0, len(filenames), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches, 1):
        cmd = [
            "robocopy",
            source_dir,
            destination,
        ] + batch + [
            f"/MT:{ROBOCOPY_THREADS}",
            f"/R:{ROBOCOPY_RETRIES}",
            f"/W:{ROBOCOPY_WAIT}",
            "/NP",
            "/NDL",
            "/NJH",
            "/NJS",
            "/COPY:DAT",
        ]

        if dry_run:
            cmd.append("/L")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            for line in result.stdout.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                lower = stripped.lower()
                if lower.startswith("new file") or lower.startswith("newer"):
                    total_copied += 1
                elif lower.startswith("*extra file") or lower.startswith("same"):
                    total_skipped += 1
                elif lower.startswith("*failed"):
                    total_errors += 1

            if result.returncode >= 8:
                print(f"    [!] Robocopy batch {batch_num} returned error code {result.returncode}")
                if result.stderr:
                    print(f"        {result.stderr.strip()}")

            if len(batches) > 1:
                print(f"    [batch {batch_num}/{len(batches)}] processed {len(batch)} files")

        except subprocess.TimeoutExpired:
            print(f"    [!] Robocopy batch {batch_num} timed out!")
            total_errors += len(batch)
        except Exception as e:
            print(f"    [!] Error running robocopy batch {batch_num}: {e}")
            total_errors += len(batch)

    return total_copied, total_skipped, total_errors


def main():
    parser = argparse.ArgumentParser(
        description="Transfer matched UV JSON files to a network share using robocopy."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually copy the files. Without this flag, a dry run is performed.",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    mode = "DRY RUN" if dry_run else "EXECUTE"

    print(f"\n{'='*60}")
    print(f"  UV JSON Transfer Script (robocopy)  —  Mode: {mode}")
    print(f"{'='*60}")
    print(f"  UV JSON source : {UV_JSON_SOURCE}")
    print(f"  Destination    : {DESTINATION}")
    print(f"  Threads        : {ROBOCOPY_THREADS}")
    print(f"{'='*60}\n")

    # --- Step 1: Load target IDs from JSON ---
    print(f"Loading target IDs from: {JSON_FILE_PATH}")
    target_ids = load_target_ids(JSON_FILE_PATH)
    print(f"  Target IDs from JSON: {len(target_ids)}\n")

    # --- Step 2: Scan NEW sldprt dirs to find which IDs exist ---
    print("Scanning NEW sldprt directories to find matching IDs...")
    new_sldprt_ids = get_new_sldprt_ids(SLDPRT_DIRECTORIES)
    print(f"  Unique IDs in NEW folders: {len(new_sldprt_ids)}\n")

    # --- Step 3: Intersect to get the ~5K matching IDs ---
    matching_ids = target_ids.intersection(new_sldprt_ids)
    print(f"  Matching IDs (JSON ∩ NEW): {len(matching_ids)}\n")

    if not matching_ids:
        print("No matching IDs found. Exiting.")
        return

    # --- Step 4: Build the file list (prefix.json for each ID) ---
    filenames = sorted([f"{fid}.json" for fid in matching_ids])
    print(f"  UV JSON files to transfer: {len(filenames)}\n")

    # --- Step 5: Transfer via robocopy ---
    print(f"{'— DRY RUN (robocopy /L) —' if dry_run else '— COPYING VIA ROBOCOPY —'}\n")
    print(f"  {UV_JSON_SOURCE}  →  {DESTINATION}")
    start = time.time()

    copied, skipped, errors = run_robocopy_batches(
        UV_JSON_SOURCE, filenames, DESTINATION, dry_run
    )

    elapsed = time.time() - start

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  TRANSFER SUMMARY  ({mode})")
    print(f"{'='*60}")
    print(f"  Matching prefix IDs       : {len(matching_ids)}")
    print(f"  UV JSON files to transfer : {len(filenames)}")
    print(f"  {'Would copy' if dry_run else 'Copied'}              : {copied}")
    print(f"  Skipped (already exist)   : {skipped}")
    print(f"  Errors                    : {errors}")
    print(f"  Time elapsed              : {elapsed:.1f}s")
    print(f"{'='*60}\n")

    if dry_run:
        print("  ► To actually copy, re-run with:  python transfer_uv_jsons.py --execute\n")


if __name__ == "__main__":
    main()
