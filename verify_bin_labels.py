import json
from pathlib import Path
import torch
from dgl.data.utils import load_graphs

BIN_PATH   = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment\sw_kernel\bin\00000000_101.bin")
LABEL_JSON = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment\sw_kernel\label\00000000_101.json")

graphs, aux = load_graphs(str(BIN_PATH))
g = graphs[0]

out = json.loads(LABEL_JSON.read_text(encoding="utf-8"))
labels = torch.tensor(out["labels"], dtype=torch.int64)

print("graph nodes:", g.num_nodes())
print("labels len:", len(labels))
print("g.ndata['f'] exists:", "f" in g.ndata)

assert g.num_nodes() == len(labels), "❌ Node count != label count"

if "f" in g.ndata:
    f = g.ndata["f"].cpu().long()
    same = torch.equal(f, labels)
    print("g.ndata['f'] == label_json.labels:", same)
    if not same:
        # show first mismatch
        idx = int((f != labels).nonzero()[0].item())
        print("first mismatch idx:", idx, "bin_f:", int(f[idx]), "json_label:", int(labels[idx]))
else:
    print("Note: g.ndata['f'] missing; only count check done.")

print("✅ BIN/label JSON consistency check done.")
