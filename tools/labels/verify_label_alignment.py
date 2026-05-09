import json
from pathlib import Path

INPUT_JSON = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment\uv_json\input_json_to_bin\00000000_101.json")
LABEL_JSON = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment\sw_kernel\label\00000000_101.json")

inp = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
out = json.loads(LABEL_JSON.read_text(encoding="utf-8"))

faces_sorted = sorted(inp["faces"], key=lambda f: int(f["id"]))
expected = [int(f.get("label", 0)) for f in faces_sorted]
got = out["labels"]

print("file_name:", out["file_name"])
print("num_faces:", len(faces_sorted), "num_labels_out:", len(got))

# show a small table
print("\nidx | face_id | expected_label | output_label | ok")
for i, f in enumerate(faces_sorted[:min(25, len(faces_sorted))]):
    face_id = int(f["id"])
    e = expected[i]
    g = got[i]
    print(f"{i:>3} | {face_id:>7} | {e:>14} | {g:>12} | {e==g}")

# final verdict
if expected == got:
    print("\n✅ Alignment OK: output labels match sorted(face.id) labels.")
else:
    # find first mismatch
    for i, (e, g) in enumerate(zip(expected, got)):
        if e != g:
            print(f"\n❌ Mismatch at idx={i}: expected={e}, got={g}, face_id={int(faces_sorted[i]['id'])}")
            break
