import json
from pathlib import Path

# ---------------------------------
# CONFIG
# ---------------------------------

INPUT_DIR = Path(
    r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment3\target_dataset\input\json_old_labels"
)

# If you want overwrite in place:
OUTPUT_DIR = INPUT_DIR

# If you prefer separate folder, use:
# OUTPUT_DIR = Path(r"...\json_fixed_labels")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPLACEMENT_LABEL = 24  # Replace -1 with this

# ---------------------------------
# PROCESS
# ---------------------------------

json_files = sorted(INPUT_DIR.glob("*.json"))

total_files = 0
files_modified = 0
total_replacements = 0

for file_path in json_files:
    total_files += 1
    modified = False

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))

        if "faces" not in data:
            print(f"[SKIP] No faces key: {file_path.name}")
            continue

        for face in data["faces"]:
            if "label" in face and int(face["label"]) == -1:
                face["label"] = REPLACEMENT_LABEL
                total_replacements += 1
                modified = True

        if modified:
            out_path = OUTPUT_DIR / file_path.name
            out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            files_modified += 1
            print(f"[UPDATED] {file_path.name}")

    except Exception as e:
        print(f"[ERROR] {file_path.name} -> {e}")

print("\n=================================")
print("Total files scanned:", total_files)
print("Files modified:", files_modified)
print("Total -1 labels replaced:", total_replacements)
print("=================================")
