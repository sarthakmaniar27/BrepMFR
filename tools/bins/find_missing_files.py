from pathlib import Path

def find_missing_files(step_folder, json_folder):
    step_path = Path(step_folder)
    json_path = Path(json_folder)

    print("Scanning directories... (this may take a minute for 100k files)")
    
    # Get all stems (filenames without extensions)
    # Using .stem handles .step, .stp, and .json automatically
    step_stems = {f.stem for f in step_path.glob("*") if f.suffix.lower() in [".step", ".stp"]}
    json_stems = {f.stem for f in json_path.glob("*.json")}

    # 1. Find totally missing files
    missing = step_stems - json_stems
    
    # 2. Find empty (corrupted) files that exist but are 0 bytes
    empty_jsons = {f.stem for f in json_path.glob("*.json") if f.stat().st_size == 0}

    print("-" * 30)
    print(f"Total STEP files found: {len(step_stems)}")
    print(f"Total JSON files found: {len(json_stems)}")
    print("-" * 30)

    if missing:
        print(f"❌ Missing JSON files ({len(missing)}):")
        for m in sorted(missing):
            print(f"  - {m}.json")
    else:
        print("✅ No files are missing from the directory.")

    if empty_jsons:
        print(f"\n⚠️  Corrupted (0-byte) JSON files ({len(empty_jsons)}):")
        for e in sorted(empty_jsons):
            print(f"  - {e}.json")

if __name__ == "__main__":
    # Update these paths to your actual drive locations
    STEP_DIR = r"Z:\step"
    JSON_DIR = r"Z:\uv_json"
    
    find_missing_files(STEP_DIR, JSON_DIR)