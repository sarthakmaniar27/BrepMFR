import json
from pathlib import Path

# -----------------------------
# LABEL NAME MAPS
# -----------------------------

CADSYNTH_LABELS = {
    0: "Stock",
    1: "Rectangular through slot",
    2: "Triangular through slot",
    3: "Rectangular passage",
    4: "Triangular passage",
    5: "6-sided passage",
    6: "Rectangular through step",
    7: "2-sided through step",
    8: "Slanted through step",
    9: "Rectangular blind step",
    10: "Triangular blind step",
    11: "Rectangular blind slot",
    12: "Rectangular pocket",
    13: "Triangular pocket",
    14: "6-sided pocket",
    15: "Chamfer",
    16: "Circular through slot",
    17: "Through hole",
    18: "Circular blind step",
    19: "Horizontal circular end blind slot",
    20: "Vertical circular end blind slot",
    21: "Circular end pocket",
    22: "O-ring",
    23: "Blind hole",
    24: "Round"
}

MFCADPP_LABELS = {
    0: "Chamfer",
    1: "Through hole",
    2: "Triangular passage",
    3: "Rectangular passage",
    4: "6-sided passage",
    5: "Triangular through slot",
    6: "Rectangular through slot",
    7: "Circular through slot",
    8: "Rectangular through step",
    9: "2-sided through step",
    10: "Slanted through step",
    11: "O-ring",
    12: "Blind hole",
    13: "Triangular pocket",
    14: "Rectangular pocket",
    15: "6-sided pocket",
    16: "Circular end pocket",
    17: "Rectangular blind slot",
    18: "Vertical circular end blind slot",
    19: "Horizontal circular end blind slot",
    20: "Triangular blind step",
    21: "Circular blind step",
    22: "Rectangular blind step",
    23: "Round",
    24: "Stock"
}

# -----------------------------
# BUILD REMAP DICTIONARY
# -----------------------------

name_to_cadsynth = {v: k for k, v in CADSYNTH_LABELS.items()}

REMAP = {}
for old_idx, name in MFCADPP_LABELS.items():
    if name not in name_to_cadsynth:
        raise ValueError(f"Label name not found in CadSynth: {name}")
    REMAP[old_idx] = name_to_cadsynth[name]

print("Label mapping built successfully.")

# -----------------------------
# PATHS
# -----------------------------

INPUT_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\target_dataset\input\json_old_labels_og_mfcad_label_indices")
OUTPUT_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\target_dataset\input\json_new_labels_cadsynth_label_indices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# PROCESS FILES
# -----------------------------

json_files = sorted(INPUT_DIR.glob("*.json"))

total = 0
failed = 0

for file_path in json_files:
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))

        if "faces" not in data:
            raise ValueError("Missing 'faces' key")

        for face in data["faces"]:
            if "label" not in face:
                raise ValueError("Face missing 'label'")

            old_label = int(face["label"])

            if old_label not in REMAP:
                raise ValueError(f"Invalid label index {old_label}")

            face["label"] = REMAP[old_label]

        out_path = OUTPUT_DIR / file_path.name
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        total += 1

    except Exception as e:
        failed += 1
        print("\nERROR PROCESSING:", file_path.name)
        print("Reason:", str(e))

print("\n=================================")
print("Total files:", len(json_files))
print("Successfully processed:", total)
print("Failed:", failed)
print("=================================")
