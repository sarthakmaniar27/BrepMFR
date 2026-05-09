import os
import time
import subprocess
import argparse

# --- PATHS ---
SLDWORKS_PATH = r"C:\images\2025_SP05_release64\d260209.004.INT\WinRel64\WinRel64\sldworks.exe"
MACRO_PATH = r"C:\Users\smr52\Desktop\Projects\Satish\testing\process_chunks.swp" 
COMM_FILE = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\chunks\current_chunk.txt"
TEST_MODE_FILE = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\chunks\test_mode.txt"
CHUNK_ROOT = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\chunks"

def run_loop(is_test, start_from):
    # Get and filter chunks
    all_chunks = sorted([d for d in os.listdir(CHUNK_ROOT) if d.startswith("chunk")], 
                        key=lambda x: int(x.replace("chunk", "")))
    
    chunks = [d for d in all_chunks if int(d.replace("chunk", "")) >= start_from]

    if is_test:
        print(f"!!! TEST MODE: Starting from Chunk {start_from} !!!")
        with open(TEST_MODE_FILE, "w") as f: f.write("TRUE")
    else:
        if os.path.exists(TEST_MODE_FILE): os.remove(TEST_MODE_FILE)

    for chunk_dir in chunks:
        chunk_id = chunk_dir.replace("chunk", "")
        print(f"\n>>> Processing Chunk {chunk_id} at {time.strftime('%H:%M:%S')}")
        
        with open(COMM_FILE, "w") as f:
            f.write(chunk_id)
        
        cmd = f'"{SLDWORKS_PATH}" /m "{MACRO_PATH}"'
        process = subprocess.Popen(cmd, shell=True)
        
        while process.poll() is None:
            time.sleep(5)
            
        print(f"Chunk {chunk_id} session ended.")
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--start', type=int, default=1, help='Chunk ID to start from')
    args = parser.parse_args()
    run_loop(args.test, args.start)

    
# import os
# import shutil
# import time
# import subprocess
# from pathlib import Path

# # --- CONFIGURATION (VERIFY THESE PATHS!) ---
# SRC_STEP = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\authors_data\step"
# SRC_LABELS = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\input\uv_json\labels"
# CHUNK_ROOT = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\chunks"
# COMM_FILE = os.path.join(CHUNK_ROOT, "current_chunk.txt") 

# SLDWORKS_PATH = r"C:\images\2025_SP05_release64\d260209.004.INT\WinRel64\WinRel64\sldworks.exe" 
# MACRO_PATH = r"C:\Users\smr52\Desktop\Projects\Satish\testing\process_chunks.swp" 

# CHUNK_SIZE = 9000
# TIMEOUT_SECONDS = 3600  # 1 hour per chunk max

# def setup_chunks_if_needed():
#     # CHECKPOINT: If chunk1 exists, assume chunking is already done
#     if os.path.exists(os.path.join(CHUNK_ROOT, "chunk1")):
#         print("Checkpoint: Chunks already exist. Skipping file copy...")
#         # Count how many chunk folders there are
#         existing_chunks = [d for d in os.listdir(CHUNK_ROOT) if d.startswith("chunk")]
#         return len(existing_chunks)
    
#     print("No chunks found. Starting chunking process...")
#     all_steps = sorted([f for f in os.listdir(SRC_STEP) if f.lower().endswith('.stp')])
#     total_chunks = 0
#     for i in range(0, len(all_steps), CHUNK_SIZE):
#         chunk_id = (i // CHUNK_SIZE) + 1
#         chunk_path = Path(CHUNK_ROOT) / f"chunk{chunk_id}"
#         (chunk_path / "step").mkdir(parents=True, exist_ok=True)
#         (chunk_path / "uv_json" / "labels").mkdir(parents=True, exist_ok=True)
        
#         for file_name in all_steps[i : i + CHUNK_SIZE]:
#             shutil.copy2(os.path.join(SRC_STEP, file_name), os.path.join(chunk_path, "step", file_name))
#         total_chunks = chunk_id
#     return total_chunks

# def run_overnight():
#     total_chunks = setup_chunks_if_needed()
#     print(f"Ready to process {total_chunks} chunks.")

#     for chunk_id in range(1, total_chunks + 1):
#         print(f"\n>>> Starting Chunk {chunk_id} at {time.strftime('%H:%M:%S')}")
        
#         with open(COMM_FILE, "w") as f:
#             f.write(str(chunk_id))
        
#         # Start SolidWorks
#         start_time = time.time()
#         #process = subprocess.Popen([SLDWORKS_PATH, "/m", MACRO_PATH])
#         # Change from Popen([PATH, ...]) to this:
#         process = subprocess.Popen(f'"{SLDWORKS_PATH}" /m "{MACRO_PATH}"', shell=True)
        
#         # Monitor Loop
#         finished_cleanly = False
#         while time.time() - start_time < TIMEOUT_SECONDS:
#             if process.poll() is not None:
#                 finished_cleanly = True
#                 break
#             time.sleep(15) # Check every 15 seconds
            
#         if not finished_cleanly:
#             print(f"WARNING: Chunk {chunk_id} timed out. Forcing closure.")
#             process.terminate()

#         # Clean up processes to ensure a fresh start for the next chunk
#         os.system("taskkill /F /IM sldworks.exe /T") 
#         time.sleep(10) # Wait for RAM to clear

#     print("\n--- ALL CHUNKS PROCESSED ---")

# if __name__ == "__main__":
#     run_overnight()