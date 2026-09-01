"""Face-level class prior of the Model A dataset vs the fine-tuning dataset.

Models D and E were fine-tuned from Model A on the replay+uniqueABC set with
near-uniform class weights. If that set is far more Stock-dominated than Model A's
original data, the fine-tune applies steady pressure toward Stock on every
marginal face.
"""

import os
import random

import numpy as np
import torch

DIRS = [
    (r"Z:\thread_and_text\no_a2", "Model A training data"),
    (r"Z:\thread_and_text\abc_for_modelA_finetuning", "D/E/F replay+uniqueABC data"),
]
SAMPLE = 900

for d, tag in DIRS:
    if not os.path.isdir(d):
        print(f"(skip {d})")
        continue
    files = []
    for root, _dd, fns in os.walk(d):
        files += [os.path.join(root, f) for f in fns if f.endswith(".pt")]
        if len(files) > 30000:
            break
    random.seed(5)
    pick = random.sample(files, min(SAMPLE, len(files)))
    counts = np.zeros(3, dtype=np.int64)
    parts_pure_stock = 0
    parts_ok = 0
    for p in pick:
        try:
            g = torch.load(p, map_location="cpu", weights_only=False)
            lab = g.label_feature.numpy().reshape(-1)
        except Exception:
            continue
        parts_ok += 1
        for c in range(3):
            counts[c] += int((lab == c).sum())
        if (lab == 0).all():
            parts_pure_stock += 1
    tot = counts.sum()
    print(f"\n=== {tag}  ({parts_ok} parts sampled, {tot} faces) ===")
    for c, nm in enumerate(["Stock", "Thread", "Text"]):
        print(f"  {nm:7s}: {counts[c]:9d}  {100.0*counts[c]/tot:6.2f}%")
    print(f"  parts that are 100% Stock: {parts_pure_stock}/{parts_ok} "
          f"({100.0*parts_pure_stock/parts_ok:.1f}%)")
    print(f"  Stock:Thread face ratio  : {counts[0]/max(counts[1],1):.2f} : 1")
