"""
transfer_sldprts.py

Copies SLDPRT files from NEW network directories to a destination share
using robocopy for fast, multi-threaded transfers.

Matches files by the prefix ID (everything before the first '_') against
the target IDs in matched_files.json.

All filename variations sharing the same prefix are included.

Usage:
    python transfer_sldprts.py              # Dry run (default) — just counts
    python transfer_sldprts.py --execute    # Actually copies via robocopy
"""

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict

# =====================================================================
# Configuration
# =====================================================================
JSON_FILE_PATH = r"C:\Users\RZA2\Downloads\matched_files.json"

# Source directories (the NEW folders from check.py)
SOURCE_DIRECTORIES = [
    r"\\Gr-sw26877\d\brepmfr_sldprts\cadsynth",
    r"\\Gr-sw34959\d\brepmfr_sldprts\cadsynth",
]

# Destination network share
DESTINATION = r"\\Gr-sw36912\c\Threads\conversion\sldprts"

# Robocopy settings
ROBOCOPY_THREADS = 16       # /MT:n  — number of parallel threads
ROBOCOPY_RETRIES = 3        # /R:n   — retries on failed copies
ROBOCOPY_WAIT    = 5        # /W:n   — seconds between retries
BATCH_SIZE       = 200      # max files per robocopy invocation (cmd-line limit)


def load_target_ids(json_path):
    """Load the JSON and extract the set of lowercase target IDs."""
    with open(json_path, "r") as f:
        data = json.load(f)

    target_ids = set()
    for j_file in data.get("matched_files", []):
        file_id = os.path.splitext(j_file)[0]
        target_ids.add(file_id.lower())

    return target_ids


def collect_files_to_copy(source_dirs, target_ids):
    """
    Walk the source directories and collect every SLDPRT whose prefix ID
    (text before the first '_') is in target_ids.

    Returns:
        files_by_source: dict  {source_dir: [filename, ...]}
        total_count: int
    """
    files_by_source = defaultdict(list)
    total_count = 0

    for directory in source_dirs:
        if not os.path.exists(directory):
            print(f"  [!] WARNING: Cannot access {directory}. Skipping.")
            continue

        print(f"  [-] Scanning {directory} ...")
        try:
            for f_name in os.listdir(directory):
                if not f_name.lower().endswith(".sldprt"):
                    continue

                base_name = os.path.splitext(f_name)[0]
                file_id = base_name.split("_")[0].lower()

                if file_id in target_ids:
                    files_by_source[directory].append(f_name)
                    total_count += 1
        except Exception as e:
            print(f"  [!] Error scanning {directory}: {e}")

    return files_by_source, total_count


def run_robocopy_batches(source_dir, filenames, destination, dry_run=True):
    """
    Run robocopy in batches for a single source directory.

    robocopy <src> <dst> file1 file2 ... /MT:16 /R:3 /W:5 /NP /NDL /NJH /NJS

    In dry-run mode, adds /L (list-only, no copy).

    Returns (copied, skipped, errors) counts.
    """
    total_copied = 0
    total_skipped = 0
    total_errors = 0

    # Split filenames into batches
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
            "/NP",          # No progress percentage
            "/NDL",         # No directory list
            "/NJH",         # No job header
            "/NJS",         # No job summary
            "/COPY:DAT",    # Copy Data, Attributes, Timestamps
        ]

        if dry_run:
            cmd.append("/L")  # List only — don't actually copy

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout per batch
            )

            # Parse robocopy output to count results
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue

                # Robocopy prefixes lines with status tags
                lower = stripped.lower()
                if lower.startswith("new file") or lower.startswith("newer"):
                    total_copied += 1
                elif lower.startswith("*extra file") or lower.startswith("same"):
                    total_skipped += 1
                elif lower.startswith("*failed"):
                    total_errors += 1

            # Robocopy exit codes:
            #   0 = no files copied (all matched/skipped)
            #   1 = files copied successfully
            #   2 = extra files/dirs detected
            #   3 = 1+2
            #   >=8 = errors
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
        description="Transfer matched SLDPRT files to a network share using robocopy."
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
    print(f"  SLDPRT Transfer Script (robocopy)  —  Mode: {mode}")
    print(f"{'='*60}")
    print(f"  Source dirs : {SOURCE_DIRECTORIES}")
    print(f"  Destination : {DESTINATION}")
    print(f"  Threads     : {ROBOCOPY_THREADS}")
    print(f"  Batch size  : {BATCH_SIZE} files per robocopy call")
    print(f"{'='*60}\n")

    # --- Load targets ---
    print(f"Loading target IDs from: {JSON_FILE_PATH}")
    target_ids = load_target_ids(JSON_FILE_PATH)
    print(f"  Target IDs (unique prefixes): {len(target_ids)}\n")

    # --- Collect matching files ---
    print("Collecting matching SLDPRT files from source directories...")
    files_by_source, total_count = collect_files_to_copy(SOURCE_DIRECTORIES, target_ids)

    # Count unique prefix IDs across all sources
    all_unique_ids = set()
    for src_dir, fnames in files_by_source.items():
        for f in fnames:
            base = os.path.splitext(f)[0]
            fid = base.split("_")[0].lower()
            all_unique_ids.add(fid)

    print(f"\n  Unique prefix IDs matched             : {len(all_unique_ids)}")
    print(f"  Total SLDPRT files (all variations)   : {total_count}")
    for src_dir, fnames in files_by_source.items():
        print(f"    └─ {src_dir}: {len(fnames)} files")
    print()

    if total_count == 0:
        print("Nothing to transfer. Exiting.")
        return

    # --- Transfer / Dry Run ---
    print(f"{'— DRY RUN (robocopy /L) —' if dry_run else '— COPYING VIA ROBOCOPY —'}\n")
    start = time.time()

    grand_copied = 0
    grand_skipped = 0
    grand_errors = 0

    for src_dir, fnames in files_by_source.items():
        print(f"  [{len(fnames)} files] {src_dir}  →  {DESTINATION}")
        copied, skipped, errors = run_robocopy_batches(src_dir, fnames, DESTINATION, dry_run)
        grand_copied += copied
        grand_skipped += skipped
        grand_errors += errors
        print(f"    Done: {copied} copied, {skipped} skipped, {errors} errors\n")

    elapsed = time.time() - start

    # --- Summary ---
    print(f"{'='*60}")
    print(f"  TRANSFER SUMMARY  ({mode})")
    print(f"{'='*60}")
    print(f"  Unique prefix IDs matched : {len(all_unique_ids)}")
    print(f"  Total files to transfer   : {total_count}")
    print(f"  {'Would copy' if dry_run else 'Copied'}              : {grand_copied}")
    print(f"  Skipped (already exist)   : {grand_skipped}")
    print(f"  Errors                    : {grand_errors}")
    print(f"  Time elapsed              : {elapsed:.1f}s")
    print(f"{'='*60}\n")

    if dry_run:
        print("  ► To actually copy, re-run with:  python transfer_sldprts.py --execute\n")


if __name__ == "__main__":
    main()
