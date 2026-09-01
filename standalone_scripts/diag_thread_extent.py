"""How large is a thread region in training, compared to the Klavuz thread region?

Klavuz reference values, measured on the part:
  thread faces               249
  thread-thread edge_len     median 0.10335
  thread face_area           median 0.00106
"""

import os
import random

import numpy as np
import torch

PT = r"Z:\thread_and_text\abc_for_modelA_finetuning"
KLAVUZ = {
    "thread_face_count": 249.0,
    "thread_edge_len_median": 0.10335,
    "thread_face_area_median": 0.0010647,
}

files = []
for root, _dd, fns in os.walk(PT):
    files += [os.path.join(root, f) for f in fns if f.endswith(".pt")]
    if len(files) > 30000:
        break
random.seed(11)
pick = random.sample(files, 700)

counts, elens, areas, diams = [], [], [], []
for p in pick:
    try:
        g = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        continue
    lab = g.label_feature.numpy().reshape(-1)
    tm = lab == 1
    if tm.sum() < 5:
        continue
    ei = g.edge_index.numpy()
    both = tm[ei[0]] & tm[ei[1]]
    if not both.any():
        continue
    counts.append(int(tm.sum()))
    elens.append(float(np.median(g.edge_len.float().numpy().reshape(-1)[both])))
    areas.append(float(np.median(g.face_area.float().numpy().reshape(-1)[tm])))
    if hasattr(g, "spatial_pos"):
        idx = np.where(tm)[0]
        sub = g.spatial_pos.numpy()[np.ix_(idx, idx)].astype(np.float64)
        sub[sub > 1e8] = np.nan
        if np.isfinite(sub).any():
            diams.append(float(np.nanmax(sub)))

print(f"parts with a thread region: {len(counts)}")
for nm, v in [
    ("thread_face_count", np.array(counts, float)),
    ("thread_edge_len_median", np.array(elens, float)),
    ("thread_face_area_median", np.array(areas, float)),
]:
    k = KLAVUZ[nm]
    print(
        f"{nm:24s} p50={np.median(v):10.5f} p90={np.percentile(v,90):10.5f} "
        f"p99={np.percentile(v,99):10.5f} max={v.max():10.5f}   "
        f"KLAVUZ={k:.5f}  pct_of_training_below_klavuz={100*(v<k).mean():.2f}%"
    )

if diams:
    d = np.array(diams, float)
    print(
        f"{'thread_hop_diameter':24s} p50={np.median(d):10.2f} "
        f"p90={np.percentile(d,90):10.2f} p99={np.percentile(d,99):10.2f} "
        f"max={d.max():10.2f}   (A1 clamp is 32)"
    )
    print(f"  fraction of training thread regions with hop diameter >= 32: "
          f"{100*(d >= 32).mean():.2f}%")
