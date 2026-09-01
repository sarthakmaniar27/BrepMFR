import json
import os

# =====================================================================
# 1. Configuration
# =====================================================================
JSON_FILE_PATH = r"C:\Users\RZA2\Downloads\matched_files.json"

# Separate the directories into NEW and OLD
NEW_DIRECTORIES = [
    r"\\Gr-sw26877\d\brepmfr_sldprts\cadsynth",
    r"\\Gr-sw34959\d\brepmfr_sldprts\cadsynth"
]

OLD_DIRECTORIES = [
    r"\\Gr-sw26877\d\brepmfr_sldprts_old\cadsynth",
    r"\\Gr-sw34959\d\brepmfr_sldprts_old\cadsynth",
    r"\\Gr-sw65237\d\brepmfr_sldprts_old\cadsynth"
]

def scan_directories(directories, folder_type):
    """Helper function to scan a list of directories and return a set of base IDs."""
    found_ids = set()
    print(f"\nScanning {folder_type} folders...")
    for directory in directories:
        if not os.path.exists(directory):
            print(f"  [!] WARNING: Cannot access {directory}. Check network or permissions.")
            continue
            
        print(f"  [-] Scanning {directory}...")
        try:
            files_in_dir = os.listdir(directory)
            for f_name in files_in_dir:
                if f_name.lower().endswith(".sldprt"):
                    base_name = os.path.splitext(f_name)[0]
                    file_id = base_name.split('_')[0]
                    found_ids.add(file_id.lower())
        except Exception as e:
            print(f"  [!] Error scanning {directory}: {e}")
            
    return found_ids

def main():
    print(f"Loading target JSON file: {JSON_FILE_PATH}")
    
    # =====================================================================
    # 2. Extract the base IDs from the JSON files
    # =====================================================================
    with open(JSON_FILE_PATH, 'r') as f:
        data = json.load(f)
    
    json_filenames = data.get("matched_files", [])
    
    target_ids = set()
    for j_file in json_filenames:
        file_id = os.path.splitext(j_file)[0] 
        target_ids.add(file_id.lower())
        
    print(f"Total target IDs to look for: {len(target_ids)}")
    
    # =====================================================================
    # 3. Read the Network Folders
    # =====================================================================
    new_ids_available = scan_directories(NEW_DIRECTORIES, "NEW")
    old_ids_available = scan_directories(OLD_DIRECTORIES, "OLD")
    
    # Combine them to get the total unique IDs available across all 5 shares
    all_available_ids = new_ids_available.union(old_ids_available)

    print(f"\nUnique IDs in NEW folders: {len(new_ids_available)}")
    print(f"Unique IDs in OLD folders: {len(old_ids_available)}")
    print(f"Total unique across all  : {len(all_available_ids)}")

    # =====================================================================
    # 4. Compare the sets to see what matches
    # =====================================================================
    found_in_new  = target_ids.intersection(new_ids_available)
    found_in_old  = target_ids.intersection(old_ids_available)
    found_in_both = found_in_new.intersection(found_in_old) # Just in case there are duplicates
    
    total_found = target_ids.intersection(all_available_ids)
    missing_ids = target_ids - all_available_ids
    
    print("\n==============================================")
    print("                    RESULTS")
    print("==============================================")
    print(f" Total JSON targets      : {len(target_ids)}")
    print(f" SLDPRTs FOUND (Total)   : {len(total_found)}")
    print(f"   ├─ From NEW folders   : {len(found_in_new)}")
    print(f"   ├─ From OLD folders   : {len(found_in_old)}")
    print(f"   └─ Found in BOTH      : {len(found_in_both)}")
    print(f" SLDPRTs MISSING         : {len(missing_ids)}")
    print("==============================================\n")
    
    # =====================================================================
    # 5. Output a report of the missing files
    # =====================================================================
    if missing_ids:
        out_file = "missing_sldprts_report.txt"
        with open(out_file, "w") as out_f:
            out_f.write(f"Total missing base IDs: {len(missing_ids)}\n")
            out_f.write("-" * 35 + "\n")
            for m_id in sorted(missing_ids):
                out_f.write(f"{m_id}.json\n")
        print(f"Saved a list of the {len(missing_ids)} missing files to: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    main()