import json
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------
INPUT_DIR = Path(r"C:\Users\smr52\Desktop\ToolBoxParts_Dataset\data\json_old")
OUTPUT_DIR = Path(r"C:\Users\smr52\Desktop\ToolBoxParts_Dataset\data\json_new")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# TOOLBOXPARTS LABELS
# old sparse label -> new consecutive label
# -----------------------------
REMAP = {
    0: 0,
    26: 1,
    27: 2,
    32: 3,
    33: 4,
    34: 5,
    35: 6,
    38: 7,
    39: 8,
    40: 9,
    41: 10,
    42: 11,
    43: 12,
    44: 13,
    45: 14,
    46: 15,
    47: 16,
    48: 17,
    49: 18,
    50: 19,
    51: 20,
    52: 21,
    53: 22,
    54: 23,
    55: 24,
    56: 25,
    57: 26,
    58: 27,
    59: 28,
    60: 29,
    61: 30,
    62: 31,
    63: 32,
}

# Optional readable names for reporting
NEW_LABEL_NAMES = {
    0: "Stock",
    1: "Radial Ball Bearing Outer Face",
    2: "Radial Ball Bearing Inner Face",
    3: "Hex Head Bolt Tail",
    4: "Hex Head Bolt Head",
    5: "Spur Gear Teeth",
    6: "Spur Gear Hub",
    7: "Hex Nut Inner Face",
    8: "Hex Nut Side Face",
    9: "Helical Gear Teeth",
    10: "Helical Gear Hub",
    11: "Straight Bevel Gear Teeth",
    12: "Straight Bevel Gear Hub",
    13: "Straight Miter Gear Teeth",
    14: "Straight Miter Gear Hub",
    15: "Square Head Bolt Tail",
    16: "Square Head Bolt Head",
    17: "Countersunk Head Bolt Tail",
    18: "Countersunk Head Bolt Head",
    19: "Round Head Bolt Tail",
    20: "Round Head Bolt Head",
    21: "Hex Screw Tail",
    22: "Hex Screw Head",
    23: "Thrust Ball Bearing Outer Face",
    24: "Thrust Ball Bearing Inner Face",
    25: "Radial Cylinder Roller Bearing Outer Face",
    26: "Radial Cylinder Roller Bearing Inner Face",
    27: "Hex Flat Nut Inner Face",
    28: "Hex Flat Nut Side Face",
    29: "Hex Slotted Nut Inner Face",
    30: "Hex Slotted Nut Side Face",
    31: "Hex Nut Prevailing Torque Inner Face",
    32: "Hex Nut Prevailing Torque Side Face",
}

# -----------------------------
# VALIDATION
# -----------------------------
if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

json_files = sorted(INPUT_DIR.rglob("*.json"))

if not json_files:
    raise FileNotFoundError(f"No JSON files found inside: {INPUT_DIR}")

print(f"Found {len(json_files):,} JSON files.")
print("Starting remap...\n")

# -----------------------------
# PROCESS FILES
# -----------------------------
total_files = 0
success_files = 0
failed_files = 0
total_faces = 0
total_remapped_faces = 0
all_new_labels_seen = set()
error_messages = []

for file_path in tqdm(json_files, desc="Remapping JSON labels", unit="file"):
    total_files += 1

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))

        if "faces" not in data:
            raise ValueError("Missing top-level key 'faces'")

        if not isinstance(data["faces"], list):
            raise ValueError("'faces' must be a list")

        file_face_count = 0
        file_labels_seen = set()

        for face_idx, face in enumerate(data["faces"]):
            if not isinstance(face, dict):
                raise ValueError(f"faces[{face_idx}] is not a dictionary")

            if "label" not in face:
                raise ValueError(f"Face at index {face_idx} missing 'label'")

            old_label = face["label"]

            try:
                old_label = int(old_label)
            except Exception:
                raise ValueError(
                    f"Invalid non-integer label at face index {face_idx}: {face['label']}"
                )

            if old_label not in REMAP:
                raise ValueError(
                    f"Invalid label index {old_label} at face index {face_idx}"
                )

            new_label = REMAP[old_label]
            face["label"] = new_label

            file_face_count += 1
            file_labels_seen.add(new_label)

        # preserve input folder structure
        relative_path = file_path.relative_to(INPUT_DIR)
        out_path = OUTPUT_DIR / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        success_files += 1
        total_faces += file_face_count
        total_remapped_faces += file_face_count
        all_new_labels_seen.update(file_labels_seen)

    except Exception as e:
        failed_files += 1
        error_messages.append(
            f"ERROR PROCESSING: {file_path}\nReason: {str(e)}\n"
        )

# -----------------------------
# WRITE ERROR LOG
# -----------------------------
if error_messages:
    error_log_path = OUTPUT_DIR / "remap_errors.txt"
    error_log_path.write_text("\n".join(error_messages), encoding="utf-8")
else:
    error_log_path = None

# -----------------------------
# WRITE SUMMARY
# -----------------------------
summary = {
    "input_dir": str(INPUT_DIR),
    "output_dir": str(OUTPUT_DIR),
    "total_files": total_files,
    "successfully_processed": success_files,
    "failed": failed_files,
    "total_faces_processed": total_faces,
    "total_faces_remapped": total_remapped_faces,
    "new_labels_seen": sorted(all_new_labels_seen),
    "new_label_names_seen": {
        str(label): NEW_LABEL_NAMES[label]
        for label in sorted(all_new_labels_seen)
    },
    "remap": REMAP,
}

summary_path = OUTPUT_DIR / "remap_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

# -----------------------------
# FINAL REPORT
# -----------------------------
print("\n=================================")
print(f"Total files           : {total_files}")
print(f"Successfully processed: {success_files}")
print(f"Failed                : {failed_files}")
print(f"Total faces processed : {total_faces}")
print(f"Summary JSON          : {summary_path}")

if error_log_path is not None:
    print(f"Error log             : {error_log_path}")

print("Labels seen after remap:", sorted(all_new_labels_seen))
print("=================================")