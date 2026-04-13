import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---
# The folder containing all your chunks (chunk1, chunk2, etc.)
CHUNKS_ROOT = Path(r"Z:")

# The final destination for all aggregated JSONs
DEST_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\source_dataset\input\json")

def aggregate_jsons():
    # 1. Ensure the destination folder exists
    if not DEST_DIR.exists():
        print(f"Creating destination directory: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)

    total_copied = 0
    
    # 2. Iterate through every item in the chunks folder
    # This will find 'chunk1', 'chunk2', etc.
    chunk_folders = [d for d in CHUNKS_ROOT.iterdir() if d.is_dir() and d.name.startswith("chunk")]
    
    print(f"Found {len(chunk_folders)} chunk folders. Starting copy...")

    for chunk in chunk_folders:
        # Target the uv_json folder inside each chunk
        uv_json_path = chunk / "uv_json"
        
        if not uv_json_path.exists():
            print(f"  [!] Skipping {chunk.name}: uv_json folder not found.")
            continue
        
        # Find all JSON files directly in uv_json 
        # (If you also need the ones inside 'labels', use .rglob('*.json') instead)
        json_files = list(uv_json_path.glob("*.json"))
        
        if not json_files:
            continue

        print(f"  Processing {chunk.name}: Copying {len(json_files)} files...")
        
        for json_file in json_files:
            try:
                # shutil.copy2 preserves file metadata (timestamps)
                shutil.copy2(json_file, DEST_DIR / json_file.name)
                total_copied += 1
            except Exception as e:
                print(f"    [!] Error copying {json_file.name}: {e}")

    print(f"\n{'='*40}")
    print(f"SUCCESS: Total JSON files aggregated: {total_copied}")
    print(f"Destination: {DEST_DIR}")
    print(f"{'='*40}")

if __name__ == "__main__":
    aggregate_jsons()