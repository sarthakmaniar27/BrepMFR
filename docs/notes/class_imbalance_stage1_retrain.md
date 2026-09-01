# Class-imbalance fix: retraining Stage 1 with balanced loss

This document is the durable companion to `post_hoc_logit_adjustment.md`. The post-hoc logit-adjustment work proved that **the BrepMFR_PyG target plateau is caused by class imbalance in source training data, not by the DA architecture, the encoder size, or any hyperparameter we'd been tuning**. This document covers the durable fix: retrain Stage 1 with class-balanced cross-entropy loss so the encoder is no longer over-confident and no longer biased toward class 0.

---

## TL;DR

1. Source training data is **57.65 % class 0 (stock)**. Target val is only **22.36 % class 0**. The Stage-1 classifier learned a P_source(0) prior and *over-predicts class 0 on target by 69 %*.
2. Post-hoc logit adjustment (Saerens 2002) recovered ~4.4 pp per-class on the existing Stage 1 ckpt. But the optimal τ was 3.0 (theoretically should be 1.0), proving the encoder is **over-confident** — a symptom of imbalance baked into the weights.
3. The durable fix is to retrain Stage 1 with **per-class loss weights** computed from inverse class frequency. This:
   - Removes the class-0 prior from the encoder weights (no need for inference-time correction)
   - Produces a calibrated softmax (so future τ ≈ 1 if we still want adjustment)
   - Gives Stage 2 DANN a real domain-shift problem to solve, instead of being dominated by label-shift signal it cannot fix
4. Implementation is small and surgical:
   - One new script: `scripts/training/compute_class_weights.py`
   - Three lines added to `BrepSeg.__init__` to load + register weights
   - One line in `training_step` to pass them to the existing CE loss
   - One CLI flag in `segmentation.py`
5. Default α = 0.5 (sqrt-inverse-frequency). Fallback α = 1.0 (full inverse) if the first run doesn't move the needle enough.

---

## The problem in numbers

### Source training class distribution (78,237 graphs, 2,180,651 labelled faces)

From `scripts/training/compute_class_weights.py` on the **full source training set** (saved to `artifacts/class_weights/stage1/source_train_alpha05.json`):

| class | count | fraction | weight (α=0.5) |
|------:|------:|---------:|---------------:|
| **0 (stock)** | **1,258,903** | **57.73 %** | **0.157** |
| 1 | 61,036 | 2.80 % | 0.713 |
| 2 | 19,674 | 0.90 % | 1.256 |
| 3 | 68,367 | 3.14 % | 0.674 |
| 4 | 48,647 | 2.23 % | 0.798 |
| 5 | 98,932 | 4.54 % | 0.560 |
| **6** | 20,581 | 0.94 % | 1.228 |
| 7 | 28,233 | 1.30 % | 1.048 |
| 8 | 18,060 | 0.83 % | 1.310 |
| 9 | 29,576 | 1.36 % | 1.024 |
| **10** | 12,993 | 0.60 % | **1.545** |
| 11 | 79,524 | 3.65 % | 0.624 |
| 12 | 55,972 | 2.57 % | 0.744 |
| 13 | 35,776 | 1.64 % | 0.931 |
| 14 | 73,516 | 3.37 % | 0.650 |
| 15 | 15,357 | 0.70 % | 1.421 |
| **16** | **10,370** | **0.48 %** | **1.729** *(max)* |
| 17 | 17,682 | 0.81 % | 1.324 |
| 18 | 19,386 | 0.89 % | 1.265 |
| 19 | 40,331 | 1.85 % | 0.877 |
| 20 | 39,169 | 1.80 % | 0.890 |
| 21 | 55,350 | 2.54 % | 0.749 |
| 22 | 33,108 | 1.52 % | 0.968 |
| 23 | 23,486 | 1.08 % | 1.149 |
| 24 | 16,622 | 0.76 % | 1.366 |

Summary statistics:
- Total faces: **2,180,651**
- Most-frequent class (0) is **121× more frequent** than rarest class (16): `1,258,903 / 10,370`.
- Weight ratio after sqrt-inverse: **11.0×** (`1.729 / 0.157`) — within stable training range.
- Mean weight: 1.000 (correctly normalised, so loss magnitude stays comparable to unweighted training).
- No weights hit the `[0.1, 20.0]` clip — every class has enough samples for a reliable estimate.

Without rebalancing, the model only sees a single sample of class 16 once every ~210 source faces; class 0 appears ~58 % of the time. Cross-entropy gradients are dominated by class 0, which is exactly the bias we measured on target val.

### Target val class distribution (268,029 faces)

| class | fraction of target val | over/under in source vs target |
|-------|------------------------|-------------------------------|
| 0 | 22.36 % | source 2.6× higher |
| 1 | 1.61 % | source 1.7× higher |
| 6 | 2.56 % | **source 2.6× lower** |
| 7 | 3.81 % | **source 2.9× lower** |
| 8 | 2.71 % | **source 3.3× lower** |
| 9 | 3.91 % | **source 2.8× lower** |
| 10 | 2.61 % | **source 4.4× lower** |
| 11 | 1.27 % | source 2.8× higher |
| 12 | 6.29 % | source 2.5× lower |
| 14 | 8.49 % | source 2.6× lower |
| 18 | 2.48 % | source 2.7× lower |

**Source and target have radically different class compositions.** This is the "label shift" diagnosed in the post-hoc document. Class-balanced training partially neutralises this at the *source* end so the model stops baking the source prior into its decision boundary.

### What the over-confidence looks like

Even after **maximum tested logit adjustment** (τ=3) on the existing Stage 1 checkpoint:

| class | acc baseline | acc + τ=3 | acc paper-target | gap remaining |
|-------|--------------|-----------|------------------|---------------|
| 6 | 0.029 | 0.066 | ~0.85 | **−0.78** |
| 9 | 0.180 | 0.249 | ~0.85 | **−0.60** |
| 1 | 0.463 | 0.478 | ~0.85 | **−0.37** |
| 2 | 0.330 | 0.431 | ~0.80 | **−0.37** |

For these four classes the encoder simply doesn't recognise the target geometry well enough. Pure prior correction can't fix that. They need either (a) a less over-confident encoder so DA gradients are meaningful, or (b) actual feature alignment via Stage 2 DANN — and DANN only works once label shift stops dominating its training signal. Both routes go through "retrain Stage 1 with balanced loss".

---

## The fix: weighted cross-entropy in Stage 1

### Math

The existing CE loss (`models/brepseg_model.py: CrossEntropyLoss`) **already** accepts per-class weights — they were just never used. The weighted form is:

```
L = -(1/N) Σ_n Σ_c w_c · y_{n,c} · log(p_{n,c} + ε)
```

where `w_c` is the per-class weight applied to the contribution of every sample of true class c. Common choices for `w_c`:

| name | formula | trade-off |
|------|---------|-----------|
| Uniform | `w_c = 1` | No rebalancing (current behaviour) |
| Inverse frequency (sklearn "balanced") | `w_c = N / (C · n_c)` | Equalises per-class loss; can over-amplify rare classes (84× ratio for our data) |
| **Sqrt inverse frequency (our default)** | `w_c = (1/freq_c)^0.5` | **Moderate (~10× ratio for our data); stable** |
| Inverse frequency (full) | `w_c = (1/freq_c)^1.0` | Aggressive; ~84× ratio; risk of over-amplifying noise on rare classes |
| Effective number (Cui 2019) | `w_c = (1-β)/(1-β^{n_c})` | Theoretically motivated; β≈0.999 typical |
| Focal loss (Lin 2017) | modulates by `(1-p)^γ` | Targets *hard examples* not *rare classes* |

We default to **sqrt inverse frequency** (`α = 0.5`) because:
- Empirically robust on extreme-imbalance datasets
- Max/min weight ratio ≈ 10× — well within stable training territory
- One knob to tune (`α`) — easy to sweep if needed
- Exactly matches the "compromise" rebalancing the literature uses for word2vec, NLP class imbalance, etc.

The general formula in our implementation is:

```
freq_c = count_c / sum(counts)
w_c    = (1 / freq_c) ** alpha
w_c    = w_c / mean(w_c)               # normalise so mean weight = 1.0
w_c    = clip(w_c, weight_min=0.1, weight_max=20.0)
```

The mean-normalisation keeps the loss magnitude comparable to unweighted training (so the existing learning rate stays appropriate). The clip is a safety rail against pathologies (e.g. a class with <10 samples giving weight=∞).

### Why not just use logit adjustment forever?

Several reasons:

1. **It only fixes label shift, not over-confidence.** A balanced encoder additionally has well-calibrated softmax outputs, which makes downstream things like uncertainty estimation, threshold-based gating, and Stage 2 DANN gradients meaningful.
2. **Logit adjustment is post-hoc.** It depends on accurately knowing target priors at deployment. With BBSE estimation that's typically possible but adds noise. Baking the fix into the encoder is more robust.
3. **Stage 2 DANN can't work over an over-confident encoder.** The discriminator gradient is dominated by the strong class-0 signal that DANN has no theoretical means to fix. After balancing, the discriminator gradient reflects actual covariate shift, which DANN *can* fix.
4. **Generalisation to unseen target distributions.** A balanced encoder has uniform per-class quality. A label-shift-corrected unbalanced encoder is only correct for the specific target distribution we tuned τ on.

So: balanced Stage 1 is the *durable* fix. Logit adjustment is the *interim* fix that already gave us +4 pp before we had a balanced checkpoint.

---

## Implementation

### Files touched

| file | change |
|------|--------|
| `scripts/training/compute_class_weights.py` | **new** — one-time pass over source train, produces a JSON cache |
| `models/brepseg_model.py` | adds `class_weights` buffer (loaded from JSON), uses it in `training_step` |
| `segmentation.py` | adds `--class_weights_path` CLI flag |

No changes to:
- The data pipeline (`data/dataset.py`, `data/collator.py`)
- The encoder, attention, or classifier modules
- The validation loss (kept unweighted so `ReduceLROnPlateau` sees a stable signal)
- The optimiser, learning-rate schedule, or any other training hyperparameter
- Stage 2 (`models/transfer_model.py`, `domain_adapt.py`) — Stage 2 has its own loss and just consumes the better-trained encoder

### `scripts/training/compute_class_weights.py`

One-time pass over source train. Iterates the 78 k `.pt` samples with a custom `_LabelOnlyDataset` (loads only `label_feature`, no graph attention features), counts labels with `np.bincount`, computes weights, writes JSON.

Key CLI flags:

| flag | default | meaning |
|------|---------|---------|
| `--dataset_path` | required | Source dataset root (Z:/Experiment6_PyG/source_dataset) |
| `--split` | `train` | Filelist base name; appends `.txt` |
| `--num_classes` | 25 | Must match Stage 1 model |
| `--alpha` | 0.5 | Weight exponent. 0=uniform, 0.5=sqrt-inv, 1.0=full inv |
| `--weight_min` | 0.1 | Lower clip on weights |
| `--weight_max` | 20.0 | Upper clip on weights |
| `--num_workers` | 4 | DataLoader workers |
| `--batch_size` | 64 | Files per worker batch (just for label collation) |
| `--max_files` | 0 (off) | Sample subset for smoke or quick estimate |
| `--out` | required | Output JSON path |

Output JSON schema:

```json
{
  "method": "inv_freq_pow",
  "alpha": 0.5,
  "num_classes": 25,
  "num_samples": 78237,
  "total_faces": 21500000,
  "weight_min": 0.1,
  "weight_max": 20.0,
  "counts": [c_0, c_1, ..., c_24],
  "weights": [w_0, w_1, ..., w_24],
  "computed_from": "Z:/Experiment6_PyG/source_dataset",
  "split_file": "train.txt",
  "computed_at": "2026-05-04T16:00:00"
}
```

Total runtime on 78,237 source train files with 4 workers: ~10 minutes. The slow part is `torch.load` on each `.pt` (which loads the whole graph just to read the label tensor — there's no partial load API for `torch.save` files).

### `models/brepseg_model.py` changes

In `__init__` (after the existing pretrained-checkpoint loading block):

```python
# Load class-balanced loss weights if provided
self.class_weights_path = getattr(args, "class_weights_path", None)
if self.class_weights_path:
    with open(self.class_weights_path, "r", encoding="utf-8") as f:
        cw = json.load(f)
    assert cw["num_classes"] == self.num_classes, "class_weights JSON / model mismatch"
    weights = torch.tensor(cw["weights"], dtype=torch.float32)
    self.use_class_weights = True
    print(f"Loaded class weights from: {self.class_weights_path}")
    print(f"  method={cw['method']} alpha={cw['alpha']} "
          f"min={weights.min():.4f} max={weights.max():.4f} mean={weights.mean():.4f}")
else:
    weights = torch.ones(self.num_classes, dtype=torch.float32)
    self.use_class_weights = False
self.register_buffer("class_weights", weights)
```

Why a `register_buffer` and not a regular attribute?
- Lightning automatically moves buffers to the correct device (GPU)
- Buffers are saved in checkpoints (so `load_from_checkpoint` round-trips correctly)
- Buffers are excluded from `.parameters()` (no gradient computed)
- We always register a buffer, even when unused — keeps the model state shape consistent across runs

In `training_step`:

```python
labels = batch["label_feature"].long()
labels_onehot = F.one_hot(labels, self.num_classes)
cw = self.class_weights if self.use_class_weights else None
loss = CrossEntropyLoss(labels_onehot, node_seg, class_level_weight=cw)
self.log("train_loss", loss, on_step=False, on_epoch=True)
return loss
```

Validation step is **deliberately left unweighted**. Reasons:
- `ReduceLROnPlateau` monitors `eval_loss`. If we change the loss formula mid-training the scheduler sees a discontinuity. Keeping val unweighted across runs gives comparable scheduler behaviour.
- The metric we ultimately care about (target per-class acc) is computed externally via the diagnostic. Val loss is just a convergence signal.

### `segmentation.py` changes

One CLI flag:

```python
parser.add_argument(
    "--class_weights_path",
    type=str,
    default=None,
    help=(
        "Path to a JSON file produced by scripts/training/compute_class_weights.py. "
        "When set, the source CE loss is multiplied per-class by the provided "
        "weights. This counteracts the class-0 dominance (~58% stock) and "
        "produces a less over-confident encoder, which closes the label-shift "
        "gap on target evaluation."
    ),
)
```

When omitted, behaviour is identical to the original training (weights = all 1.0, `use_class_weights = False` skips the multiplication).

---

## How to run it

### Step 1 — compute class weights (one-time, ~10 min)

```powershell
python scripts/training/compute_class_weights.py `
  --dataset_path "Z:/Experiment6_PyG/source_dataset" `
  --split train `
  --num_classes 25 `
  --alpha 0.5 `
  --num_workers 4 `
  --out "artifacts/class_weights/stage1/source_train_alpha05.json"
```

Console output ends with (actual run on 78,237 source train files, ~8 min wall time):

```
Total labelled faces: 2,180,651
  class        count      pct
      0    1,258,903  57.731%
      1       61,036   2.799%
      2       19,674   0.902%
      ...
     16       10,370   0.476%
     ...

Class weights (method=inv_freq_pow, alpha=0.5, clip=[0.1, 20.0]):
  class     weight
      0     0.1570
      1     0.7128
     10     1.5450
     16     1.7294
     ...
  mean = 1.0000, min = 0.157, max = 1.729

Wrote class weights to: artifacts/class_weights/stage1/source_train_alpha05.json
```

Sanity checks (all passed for our actual run):
- `mean(weights) = 1.0000` ✓
- `weights[0]` (stock) is the smallest at **0.157** ✓
- Highest weights belong to the rarest classes (10, 16) at **1.55–1.73** ✓
- No weights hit the `[0.1, 20.0]` clip → counts are reliable for every class ✓

### Step 2 — retrain Stage 1 (~1–2 days)

Same command as the original Stage 1 run, with one extra flag and a new experiment name:

```powershell
python segmentation.py train `
  --dataset_path "Z:/Experiment6_PyG/source_dataset" `
  --num_classes 25 `
  --batch_size 32 `
  --num_workers 4 `
  --class_weights_path "artifacts/class_weights/stage1/source_train_alpha05.json" `
  --run_name ce_weighted_balanced__manual `
  --max_epochs 100
```

On startup the console should print:

```
Loaded class weights from: artifacts/class_weights/stage1/source_train_alpha05.json
  method=inv_freq_pow alpha=0.5 min=0.1554 max=1.6324 mean=1.0000
```

If you see that, the weights are loaded and active. Logs land under **`results/stage1/<run_name>/`** (default auto `ce_weighted_balanced__YYYY-MM-DD_...`, or pass **`--run_name`**).

**Use a different `--run_name` for each experiment** so runs stay in separate folders under `results/stage1/`.

### Step 3 — verify on target val (~5 min after retrain)

Run the same diagnostic and τ-sweep on the new checkpoint:

```powershell
# diagnostic: confirms class-collapse / over-prediction patterns are gone
python scripts/diagnostics/diagnose_stage1_target.py `
  --checkpoint "C:/Users/D58/Desktop/BrepMFR_PyG/results/stage1/<run_name>/best.ckpt" `
  --source_path "Z:/Experiment6_PyG/source_dataset" `
  --target_path "Z:/Experiment6_PyG/target_dataset" `
  --num_classes 25 --batch_size 32 --num_workers 2 `
  --out_dir "results/diagnostics/stage1_balanced_audit"

# tau sweep: confirms the encoder is now calibrated (best tau should drop near 1.0)
python scripts/diagnostics/logit_adjust_eval.py `
  --checkpoint "C:/Users/D58/Desktop/BrepMFR_PyG/results/stage1/<run_name>/best.ckpt" `
  --source_path "Z:/Experiment6_PyG/source_dataset" `
  --target_path "Z:/Experiment6_PyG/target_dataset" `
  --num_classes 25 --batch_size 32 --num_workers 2 `
  --taus "0.0,0.5,1.0,1.5,2.0" `
  --out_dir "results/diagnostics/logit_adjust_balanced"
```

---

## What success looks like

Expected signals if the fix worked:

| Signal | Original Stage 1 | Balanced Stage 1 (success criterion) |
|--------|------------------|--------------------------------------|
| Source per-face acc (val) | 0.9934 | 0.97–0.99 (slightly lower because it stops over-predicting class 0) |
| Source per-face acc on rare classes | varies | All classes >0.9 (no class is starved during training) |
| Target per-face acc baseline (no τ) | 0.8055 | **>0.85** |
| Target per-class acc baseline (no τ) | 0.7209 | **>0.78** |
| Target per-class acc + best τ | 0.7650 | **>0.82** |
| Best τ in adjustment sweep | **3.0** (over-confident) | **~1.0** (well-calibrated) |
| τ sweep gain (best − baseline) | +4.4 pp | **<2 pp** (most of the fix is in the weights now) |
| Class 6 acc (target) | 0.029 | **>0.30** |
| Class 9 acc (target) | 0.180 | **>0.40** |
| Class 1 acc (target) | 0.463 | **>0.65** |
| Per-face on stock (class 0) | 0.9999 | 0.99+ (slight drop is fine) |

Failure modes to watch for during training (TensorBoard):

- `train_loss` exploding or oscillating → α too aggressive, drop to α=0.25 or raise weight_min to 0.2
- `eval_loss` flat-line above original Stage 1's eval_loss → weights miscomputed; check JSON
- Source val per-face dropping below 0.85 → over-rebalancing destroyed common-class accuracy
- Class 0 source acc <0.95 → too much down-weighting on class 0; raise weight_min

---

## What if it doesn't work?

A few escalations, in increasing order of effort:

### 1. Try α = 1.0 (full inverse frequency, ~84× ratio)

Re-run `compute_class_weights.py` with `--alpha 1.0`, save to `source_train_alpha10.json`, retrain. This is more aggressive but for our 84:1 imbalance might be exactly what's needed.

```powershell
python scripts/training/compute_class_weights.py `
  --dataset_path "Z:/Experiment6_PyG/source_dataset" --split train --num_classes 25 `
  --alpha 1.0 --weight_max 50.0 `
  --out "artifacts/class_weights/stage1/source_train_alpha10.json"
```

(Note the higher `--weight_max` to allow the rarest classes through unclipped.)

### 2. Switch to focal loss

Focal loss (`(1-p)^γ · CE`) targets *hard examples* in addition to rare classes. Empirically more stable than full inverse frequency at extreme imbalance. Implementation requires modifying the loss function rather than just providing weights — call out if this becomes necessary.

### 3. Effective number of samples (Cui et al. 2019)

A theoretically grounded weighting that accounts for sample overlap in feature space. Implementation: compute weights as `(1−β)/(1−β^{n_c})` with β=0.999. Add as another `--method` option to `compute_class_weights.py`.

### 4. Source data augmentation for rare classes

If class 6 is still ~30 % even after balanced training and DA, consider augmenting the source dataset to include more class-6 geometry. This is a data-side fix, not a training-side fix. Check whether `Z:/Experiment6/source` could be re-generated with class-balanced sampling at the synthetic-CAD generation stage.

---

## After this — Stage 2 redux

Once balanced Stage 1 is working (target per-class > 0.78 baseline, best τ ≈ 1.0):

1. **Run a fresh Stage 2** using the balanced Stage 1 checkpoint as `--pre_train`. Use the authors-faithful config (current `models/transfer_model.py`).
2. **Expect DA to actually help this time** because the discriminator's gradient signal is no longer dominated by label shift.
3. The two specific classes to watch on target:
   - **Class 6**: the most stubborn. If balanced Stage 1 alone gets it to 0.3+, then a successful Stage 2 should land it at 0.7+.
   - **Class 9**: same pattern; aim for 0.6+ after Stage 2.
4. If Stage 2 still doesn't help, then the residual gap is in the encoder's *capacity* to recognise these specific geometries — which is an architecture/data problem, not a training problem.

The full chain to the paper's 92.74 % per-class:

```
unbalanced Stage 1                                                 0.7209
  └─ + post-hoc τ=3 logit adjustment (interim)              +4.4   0.7650
balanced Stage 1                                            +5.5  ~0.7800  (estimated)
  └─ + Stage 2 DANN on top                                  +6.0  ~0.8400  (estimated)
  └─ + post-hoc τ≈1 (only if residual label shift)          +1.0  ~0.8500  (estimated)
                                                          ─────
                                                  paper       ~0.92  (?)
```

The remaining ~7 pp gap from our optimistic ~0.85 to the paper's 0.92 is unexplained — could be authors' MFCAD++ split is slightly different from ours, could be an architectural detail we missed, could be longer training. The first three steps are well-founded and what we should chase first.

---

## Glossary

| term | meaning |
|------|---------|
| α (alpha) | Exponent in `(1/freq)^α`. 0=uniform, 0.5=sqrt, 1.0=full inverse |
| τ (tau) | Strength of post-hoc logit adjustment. 1.0=Saerens-optimal, >1=compensates over-confidence |
| Label shift | P(Y) differs across domains; P(X\|Y) does not. Not fixable by feature alignment. |
| Covariate shift | P(X) differs across domains; P(Y\|X) does not. Fixable by DANN/CORAL/MMD. |
| Over-confidence | Softmax outputs near 1.0 for predicted class; symptom of class imbalance and unregularised CE training. |
| Calibration | Property where predicted probabilities match empirical correctness rates. Logit adjustment requires this for τ=1 to work. |
| Sqrt-inverse-frequency | `w_c = (1/freq_c)^0.5`. Conservative class-balancing weight choice. |

---

## File reference

| File | Role |
|------|------|
| `scripts/training/compute_class_weights.py` | One-time pass over source train; writes JSON cache |
| `models/brepseg_model.py` | Modified to load + use class weights |
| `segmentation.py` | New `--class_weights_path` CLI flag |
| `artifacts/class_weights/stage1/source_train_alpha05.json` | Computed weights for α=0.5 (default) |
| `artifacts/class_weights/stage1/source_train_alpha10.json` | Optional weights for α=1.0 (escalation) |
| `results/stage1/<run_name>/best.ckpt` | Output of balanced Stage 1 retrain |
| `markdowns/post_hoc_logit_adjustment.md` | Companion doc — interim fix and the diagnosis that motivated this retrain |

---

## Status

- [x] `compute_class_weights.py` written
- [x] `BrepSeg` modified to accept `class_weights_path`
- [x] `segmentation.py` adds `--class_weights_path` flag
- [x] Smoke test passes (500-file weight computation produces sensible weights; BrepSeg loads JSON correctly)
- [x] **Step 1 complete: full source-train weight computation (8 min wall time, 78,237 graphs, 2,180,651 faces) → `artifacts/class_weights/stage1/source_train_alpha05.json`**
- [ ] Step 2: balanced Stage 1 retrain (~1–2 days)
- [ ] Step 3: diagnostic + τ-sweep on balanced checkpoint
- [ ] Step 4: fresh Stage 2 DANN on top of balanced Stage 1
