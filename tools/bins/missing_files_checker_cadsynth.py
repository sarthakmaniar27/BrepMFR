import os
import shutil
import re
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\chunks")
CHUNKS_TO_PROCESS = [f"chunk{i}" for i in range(1, 13)]  # Processes chunk1 through chunk12

# Destination Paths (All missing files from all chunks go here)
MISS_DIR = BASE_DIR / "chunk_miss"
MISS_STEP_DIR = MISS_DIR / "step"
MISS_LABELS_DIR = MISS_DIR / "uv_json" / "labels"

def isolate_missing_all_chunks():
    # 1. Ensure the destination folder structure exists once
    os.makedirs(MISS_STEP_DIR, exist_ok=True)
    os.makedirs(MISS_LABELS_DIR, exist_ok=True)

    print(f"Starting analysis for {len(CHUNKS_TO_PROCESS)} chunks...")

    for chunk_name in CHUNKS_TO_PROCESS:
        SOURCE_DIR = BASE_DIR / chunk_name
        STEP_DIR = SOURCE_DIR / "step"
        JSON_PARENT_DIR = SOURCE_DIR / "uv_json"
        SOURCE_LABELS_DIR = JSON_PARENT_DIR / "labels"

        # Check if chunk folder exists before processing
        if not SOURCE_DIR.exists():
            print(f"\n[!] Skipping {chunk_name}: Folder not found.")
            continue

        print(f"\n--- Processing {chunk_name} ---")

        # 2. Identify STEP files in this chunk
        step_files = {f.stem: f.name for f in STEP_DIR.glob("*") if f.suffix.lower() in [".step", ".stp"]}
        
        # 3. Identify JSON IDs using Regex to handle any _### suffix
        # This matches the ID before the last underscore followed by numbers
        json_ids_in_parent = set()
        for f in JSON_PARENT_DIR.glob("*.json"):
            # Regex explanation: (.+?) looks for the ID, _\d+$ looks for underscore + numbers at end
            match = re.match(r"(.+?)_\d+$", f.stem)
            if match:
                json_ids_in_parent.add(match.group(1))
            else:
                # Fallback: if there's no suffix, just take the stem
                json_ids_in_parent.add(f.stem)

        # 4. Compare
        missing_ids = sorted(list(set(step_files.keys()) - json_ids_in_parent))

        if not missing_ids:
            print(f"  [+] All files in {chunk_name} are in sync.")
            continue

        print(f"  [!] Found {len(missing_ids)} missing samples in {chunk_name}.")

        # 5. Selective Copy
        for m_id in missing_ids:
            # Copy STEP
            step_filename = step_files[m_id]
            src_step = STEP_DIR / step_filename
            dst_step = MISS_STEP_DIR / step_filename
            shutil.copy2(src_step, dst_step)
            
            # Copy corresponding label from 'labels' subfolder
            found_label = False
            if SOURCE_LABELS_DIR.exists():
                # We search for the ID with any suffix in the labels folder
                for json_file in SOURCE_LABELS_DIR.glob(f"{m_id}*.json"):
                    shutil.copy2(json_file, MISS_LABELS_DIR / json_file.name)
                    found_label = True
            
            status = "with Label" if found_label else "STEP ONLY (Label missing)"
            print(f"    -> Isolated {m_id} ({status})")

    print(f"\n{'='*40}")
    print(f"DONE: All chunks processed.")
    print(f"Missing items isolated to: {MISS_DIR}")
    print(f"{'='*40}")

if __name__ == "__main__":
    isolate_missing_all_chunks()

# import os
# from collections import Counter
# from pathlib import Path

# # Path to the folder where you have 99,942 files
# FINAL_FOLDER = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\input\json")

# def find_the_imposter():
#     # 1. Get all filenames
#     all_files = os.listdir(FINAL_FOLDER)
    
#     # 2. Check for non-JSON files (the Thumbs.db suspect)
#     non_json = [f for f in all_files if not f.endswith('.json')]
#     if non_json:
#         print(f"[!] Found non-JSON files: {non_json}")

#     # 3. Check for Duplicate IDs (The "Overwriter" suspect)
#     # Stripping suffixes to get pure IDs
#     ids = [f.split('_')[0] for f in all_files if f.endswith('.json')]
#     counts = Counter(ids)
#     duplicates = [item for item, count in counts.items() if count > 1]
    
#     if duplicates:
#         print(f"[!] Found Duplicate IDs: {duplicates}")
#     else:
#         print("[+] No duplicate IDs found.")

#     print(f"\nTotal files checked: {len(all_files)}")

# if __name__ == "__main__":
#     find_the_imposter()

# import os
# from pathlib import Path

# FINAL_FOLDER = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\input\json")

# # Find any file that starts with the duplicate ID
# culprits = [f for f in os.listdir(FINAL_FOLDER) if f.startswith('00056014')]

# print("The duplicate files are:")
# for f in culprits:
#     print(f" - {f}")