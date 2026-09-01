"""Re-score every Klavuz model after relabelling the third thread land (faces
274-361) from Stock to Thread.

Two references are reported:
  pseudo   = the existing reference (Model A predictions + 13 manual corrections)
  land3    = pseudo, with faces 274-361 changed Stock -> Thread
  land3+rim= land3, plus the thread-radius runout faces 108-112, 192, 271, 272
"""

import csv

import numpy as np

CSV = r"artifacts\klavuz_full_a1_a3_scratch_abc70k_last_onnx\all_face_comparison.csv"
MODELS = {
    "A  (lite->A1A3, small data)": "A",
    "B  (scratch 72K)": "B",
    "C  (B -> +new ABC)": "C",
    "D  (A -> finetune, BN update)": "D",
    "E  (A -> finetune, BN frozen)": "FrozenBestV3",
    "F  (scratch, replay+uniqueABC)": "Scratch70K",
}
LAND3 = list(range(274, 362))
RIM = [108, 109, 110, 111, 112, 192, 271, 272]


def metrics(gt, pred):
    acc = float((gt == pred).mean())
    ious, recs, precs = [], {}, {}
    for cid, cname in [(0, "Stock"), (1, "Thread"), (2, "Text")]:
        g, p = gt == cid, pred == cid
        inter = (g & p).sum()
        union = (g | p).sum()
        ious.append(inter / union if union else np.nan)
        recs[cname] = inter / g.sum() if g.sum() else np.nan
        precs[cname] = inter / p.sum() if p.sum() else np.nan
    return acc, float(np.nanmean(ious)), recs, precs


with open(CSV, newline="") as fh:
    rows_raw = list(csv.DictReader(fh))
df = {k: [r[k] for r in rows_raw] for k in rows_raw[0]}

lab2id = {"Stock": 0, "Thread": 1, "Text": 2}
gt_pseudo = np.array([int(v) for v in df["ground_truth_id"]])

refs = {"pseudo (Model A based)": gt_pseudo.copy()}
g2 = gt_pseudo.copy()
g2[LAND3] = 1
refs["land3 corrected"] = g2
g3 = g2.copy()
g3[RIM] = 1
refs["land3 + thread rim"] = g3

for rname, gt in refs.items():
    dist = {k: int((gt == v).sum()) for k, v in lab2id.items()}
    print(f"\n{'='*82}\nREFERENCE: {rname}    distribution {dist}\n{'='*82}")
    print(f"{'model':32s} {'acc':>7s} {'mIoU':>7s} {'StockR':>7s} {'ThrP':>7s} {'ThrR':>7s} {'errs':>5s}")
    rows = []
    for name, col in MODELS.items():
        pred = np.array([lab2id[v] for v in df[f"{col}_pred"]])
        acc, miou, rec, prec = metrics(gt, pred)
        rows.append((acc, name, miou, rec, prec, int((gt != pred).sum())))
    for acc, name, miou, rec, prec, err in sorted(rows, reverse=True):
        print(f"{name:32s} {100*acc:6.2f}% {100*miou:6.2f}% "
              f"{100*rec['Stock']:6.2f}% {100*prec['Thread']:6.2f}% "
              f"{100*rec['Thread']:6.2f}% {err:5d}")

# Where does the best model still fail under the corrected reference?
gt = refs["land3 corrected"]
pred = np.array([lab2id[v] for v in df["Scratch70K_pred"]])
bad = np.where(gt != pred)[0]
print(f"\nModel F remaining errors under 'land3 corrected' ({len(bad)} faces):")
inv = {v: k for k, v in lab2id.items()}
for i in bad:
    print(f"  face {i:3d}: gt={inv[gt[i]]:6s} pred={inv[pred[i]]:6s} "
          f"conf={float(df['Scratch70K_confidence'][i]):.3f}")

pred = np.array([lab2id[v] for v in df["A_pred"]])
bad = np.where(gt != pred)[0]
print(f"\nModel A remaining errors under 'land3 corrected': {len(bad)} faces")
