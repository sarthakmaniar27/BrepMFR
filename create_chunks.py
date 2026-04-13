import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---
SRC_STEP = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\authors_data\step"
SRC_LABELS = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\input\uv_json\labels"
CHUNK_ROOT = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\chunks"
CHUNK_SIZE = 9000

def create_pure_chunks():
    # 1. Get all 100,000 STEP files
    all_steps = sorted([f for f in os.listdir(SRC_STEP) if f.lower().endswith('.stp')])
    total_files = len(all_steps)
    print(f"Total files found: {total_files}")

    if total_files == 0:
        print("No STEP files found in the source directory. Please check your path.")
        return

    # 2. Process into Chunks
    for i in range(0, total_files, CHUNK_SIZE):
        chunk_id = (i // CHUNK_SIZE) + 1
        chunk_path = Path(CHUNK_ROOT) / f"chunk{chunk_id}"
        step_target = chunk_path / "step"
        label_target = chunk_path / "uv_json" / "labels"

        # Create the directories
        step_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)

        batch = all_steps[i : i + CHUNK_SIZE]
        print(f"Organizing Chunk {chunk_id} (Files {i} to {min(i + CHUNK_SIZE, total_files)})...")

        for file_name in batch:
            # Copy STEP file
            shutil.copy2(os.path.join(SRC_STEP, file_name), os.path.join(step_target, file_name))
            
            # Copy corresponding Label file
            # Assuming label matches: 00000000.json
            label_name = os.path.splitext(file_name)[0] + ".json" 
            label_src_path = os.path.join(SRC_LABELS, label_name)
            
            if os.path.exists(label_src_path):
                shutil.copy2(label_src_path, os.path.join(label_target, label_name))
            else:
                # Optional: Print warning if a label is missing
                pass

    print(f"\nSuccess! 100,000 files divided into {chunk_id} chunks.")
    print(f"Root folder: {CHUNK_ROOT}")

if __name__ == "__main__":
    create_pure_chunks()