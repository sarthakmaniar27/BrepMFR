import os
import time
import subprocess
import win32com.client

SLDPRT_FOLDER = r"\\GR-SW36912\Threads\conversion\sldprts"
OUTPUT_JSON_FOLDER = r"\\GR-SW36912\Threads\conversion\jsons"
UV_JSON_FOLDER = r"\\GR-SW36912\Threads\conversion\uv_jsons"
BATCH_SIZE = 100  # Restart SolidWorks every 100 files to clear memory

def get_all_sldprts(folder):
    sldprts = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".sldprt"):
                sldprts.append(os.path.join(root, f))
    return sldprts

def kill_solidworks():
    """Force close SolidWorks to flush memory completely."""
    subprocess.run(["taskkill", "/F", "/IM", "SLDWORKS.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

def process_batch(file_list):
    kill_solidworks()  # Ensure clean state
    
    # Start SolidWorks in background
    sw_app = win32com.client.Dispatch("SldWorks.Application")
    sw_app.Visible = False  # HIDE GUI (Massive speed boost)
    sw_internal = sw_app

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        json_path = os.path.join(OUTPUT_JSON_FOLDER, f"{base_name}.json")

        # Skip if already converted
        if os.path.exists(json_path):
            continue

        try:
            print(f"Processing: {file_name}")
            # 1 = swDocPART, 1 = swOpenDocOptions_Silent
            model = sw_app.OpenDoc6(file_path, 1, 1, "", 0, 0)
            
            if model:
                # Trigger command
                sw_internal.BaselineOutputCmd(100040, f"{OUTPUT_JSON_FOLDER}|1|{UV_JSON_FOLDER}")
                sw_app.CloseDoc(file_path)
            else:
                print(f"FAILED to open: {file_name}")

        except Exception as e:
            print(f"ERROR on {file_name}: {e}")
            # If error occurs, break and let main loop restart SW
            break

    # Close SW gracefully at end of batch
    try:
        sw_app.ExitApp()
    except:
        pass
    kill_solidworks()

def main():
    all_files = get_all_sldprts(SLDPRT_FOLDER)
    print(f"Found {len(all_files)} total files.")

    # Process in chunks of BATCH_SIZE
    for i in range(0, len(all_files), BATCH_SIZE):
        batch = all_files[i:i + BATCH_SIZE]
        print(f"\n--- Starting Batch {i // BATCH_SIZE + 1} ({len(batch)} files) ---")
        process_batch(batch)

if __name__ == "__main__":
    main()