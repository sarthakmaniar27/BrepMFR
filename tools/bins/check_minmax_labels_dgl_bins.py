import json
from pathlib import Path
import torch
from dgl.data.utils import load_graphs

# 🔁 CHANGE THIS
BIN_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment3\target_dataset\output\bin")
# If your label JSONs are stored separately, set this. If not, we’ll rely on g.ndata['f'].
LABEL_DIR = None  # e.g., Path(r"...\output\label")

def get_labels_from_bin(bin_path: Path) -> torch.Tensor:
    graphs, _ = load_graphs(str(bin_path))
    g = graphs[0]
    if "f" not in g.ndata:
        raise KeyError("g.ndata['f'] not found (expected labels stored in node feature 'f').")
    return g.ndata["f"].view(-1).to(torch.int64).cpu()

def get_labels_from_json(bin_path: Path, label_dir: Path) -> torch.Tensor:
    label_json = label_dir / (bin_path.stem + ".json")
    out = json.loads(label_json.read_text(encoding="utf-8"))
    return torch.tensor(out["labels"], dtype=torch.int64)

global_min = None
global_max = None
neg_total = 0
files_scanned = 0
bad_files = []

print(f"Scanning: {BIN_DIR}")

for bin_path in sorted(BIN_DIR.glob("*.bin")):
    try:
        labels = get_labels_from_bin(bin_path)

        # Optional: cross-check against JSON labels if you have them
        if LABEL_DIR is not None:
            labels_json = get_labels_from_json(bin_path, LABEL_DIR)
            if labels.numel() != labels_json.numel():
                raise ValueError(f"Label count mismatch: bin={labels.numel()} json={labels_json.numel()}")
            if not torch.equal(labels, labels_json):
                # find first mismatch
                idx = int((labels != labels_json).nonzero()[0].item())
                raise ValueError(f"Label mismatch at idx {idx}: bin={int(labels[idx])} json={int(labels_json[idx])}")

        mn = int(labels.min().item()) if labels.numel() else 0
        mx = int(labels.max().item()) if labels.numel() else 0
        neg = int((labels < 0).sum().item())

        global_min = mn if global_min is None else min(global_min, mn)
        global_max = mx if global_max is None else max(global_max, mx)
        neg_total += neg
        files_scanned += 1

    except Exception as e:
        bad_files.append((bin_path.name, str(e)))

print("\n---------------- RESULTS ----------------")
print(f"Files scanned: {files_scanned}")
print(f"Global MIN label: {global_min}")
print(f"Global MAX label: {global_max}")
print(f"Total negative labels (<0): {neg_total}")

if bad_files:
    print("\n❌ Files with errors:")
    for name, err in bad_files[:20]:
        print(f"  - {name}: {err}")
    if len(bad_files) > 20:
        print(f"  ... and {len(bad_files) - 20} more")
else:
    print("\n✅ All files loaded successfully and scanned.")