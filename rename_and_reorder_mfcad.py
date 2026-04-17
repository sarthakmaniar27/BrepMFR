import os
import re
from pathlib import Path

# --- CONFIGURATION ---
STEP_DIR = r"C:\Users\smr52\Desktop\MFCAD++_dataset\step"
JSON_DIR = r"C:\Users\smr52\Desktop\MFCAD++_dataset\json"
# ---------------------

def rename_dataset():
    step_path = Path(STEP_DIR)
    json_path = Path(JSON_DIR)

    # 1. Get all step files and extract their numeric ID
    print("Scanning files...")
    files = []
    for f in step_path.glob("mfcad_*.step"):
        match = re.search(r'mfcad_(\d+)\.step', f.name)
        if match:
            files.append((int(match.group(1)), f))

    # 2. Sort numerically to ensure consistent ordering
    files.sort(key=lambda x: x[0])
    total_files = len(files)
    print(f"Found {total_files} step files. Starting renaming...")

    for index, (old_id, f_path) in enumerate(files):
        new_base = f"{index:08d}" # Produces 00000000, 00000001, etc.
        
        # New names
        new_step_name = f"{new_base}.step"
        new_json_name = f"{new_base}_101.json"
        
        # Old JSON name pattern
        old_json_name = f"mfcad_{old_id}_101.json"
        old_json_path = json_path / old_json_name
        
        # Rename STEP
        f_path.rename(step_path / new_step_name)
        
        # Rename JSON if it exists
        if old_json_path.exists():
            old_json_path.rename(json_path / new_json_name)

        if index % 5000 == 0:
            print(f"Progress: {index}/{total_files}...")

    print(f"Success. Renamed {total_files} file sets.")
    print(f"New range is 00000000.step to {total_files-1:08d}.step")

if __name__ == "__main__":
    rename_dataset()