import os
import shutil
import json
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
SOURCE_BASE = Path(r"C:\Users\smr52\Desktop\MFCAD++_dataset")
DEST_BASE = Path(r"C:\Users\smr52\Desktop\MFCAD++_dataset\mfcad_miss")
SPLITS = ["train", "test", "val"]

def process_mfcad_dataset():
    print(f"Starting Multi-Body Analysis and Quality Check...")
    
    total_isolated = 0

    for split in SPLITS:
        STEP_DIR = SOURCE_BASE / "step" / split
        JSON_DIR = SOURCE_BASE / "json" / split
        MISS_STEP_DIR = DEST_BASE / "step" / split
        
        os.makedirs(MISS_STEP_DIR, exist_ok=True)

        print(f"\n--- Processing {split.upper()} split ---")

        # 1. Clean up empty JSON files and map valid ones
        # Mapping: ID -> list of non-empty JSON filenames
        valid_json_map = defaultdict(list)
        empty_files_deleted = 0

        if JSON_DIR.exists():
            for json_file in JSON_DIR.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if JSON is empty (no faces or no edges)
                    if not data.get("faces") or len(data["faces"]) == 0:
                        f.close() # Ensure file is closed before removal
                        os.remove(json_file)
                        empty_files_deleted += 1
                        continue
                    
                    # Extract ID (e.g., mfcad_3236 from mfcad_3236_101.json)
                    # Splitting by last underscore to handle the suffix
                    base_id = json_file.stem.rsplit('_', 1)[0]
                    valid_json_map[base_id].append(json_file.name)
                
                except Exception as e:
                    print(f"Error reading {json_file.name}: {e}")

        # 2. Map all STEP files
        step_files = {f.stem: f.name for f in STEP_DIR.glob("*") if f.suffix.lower() in [".step", ".stp"]}

        # 3. Analyze Counts
        multi_body_ids = [m_id for m_id, jsons in valid_json_map.items() if len(jsons) > 1]
        missing_ids = [s_id for s_id in step_files.keys() if s_id not in valid_json_map]

        print(f" [+] Deleted {empty_files_deleted} empty JSON files.")
        print(f" [+] Found {len(multi_body_ids)} Multi-Body models (IDs with >1 valid JSON).")
        print(f" [!] Found {len(missing_ids)} STEP files with ZERO valid JSONs.")

        # 4. Isolate Missing STEPs
        for m_id in missing_ids:
            step_name = step_files[m_id]
            shutil.copy2(STEP_DIR / step_name, MISS_STEP_DIR / step_name)
            total_isolated += 1

    print(f"\n{'='*50}")
    print(f"PROCESS COMPLETE")
    print(f"Total STEP files with no valid JSON isolated: {total_isolated}")
    print(f"Check isolated files at: {DEST_BASE}")
    print(f"{'='*50}")

if __name__ == "__main__":
    process_mfcad_dataset()