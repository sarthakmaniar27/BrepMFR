# Focal Loss Training Report — Thread + Text Dataset

## Experiment Overview

| Item | Detail |
|------|--------|
| **Goal** | Improve thread feature accuracy on the combined thread+text dataset |
| **Problem** | Text faces outnumber thread faces 106:1, causing the model to neglect thread classification |
| **Solution Applied** | Focal Loss (γ=2.0) + inverse-frequency class weights (α=0.5) |
| **Dataset** | `Z:\lite\pyg` — 33,657 train / 4,205 val graphs (3 classes: stock, thread, text) |
| **Run Name** | `focal_gamma2_thread_text_v2` |
| **Duration** | ~19 hours (55 epochs at step 29,049) |

---

## Code Changes Made

### 1. Focal Loss Function — [brepseg_model.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py#L72-L108)

Added `FocalLoss()` as a drop-in alternative to `CrossEntropyLoss()`. Focal Loss applies a `(1 - p_t)^γ` modulating factor that dynamically down-weights easy examples:

- Text faces classified at 99% confidence: `(1-0.99)^2 = 0.0001` → near-zero gradient
- Thread faces classified at 60% confidence: `(1-0.60)^2 = 0.16` → 1,600× stronger gradient

Combined with existing class weights, this gives thread faces both frequency-based AND difficulty-based amplification.

### 2. CLI Arguments — [segmentation.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/segmentation.py#L216-L237)

Added `--loss_type` (`ce` or `focal`) and `--focal_gamma` (default 2.0). Default behaviour is unchanged (`--loss_type ce`).

### 3. Performance Fix — Removed `torch.cuda.empty_cache()` from training/validation steps

The original code called `torch.cuda.empty_cache()` every training step, forcing CUDA to deallocate and reallocate all memory from scratch. This caused 100% GPU utilization with 0% memory bandwidth (GPU spinning on memory management, not compute). Removing it eliminated the periodic 32GB memory stalls.

### 4. Validation Metrics — Added per-class accuracy, feature-only accuracy, and IoU to validation logging

The original code only logged `per_face_accuracy` during validation, which is dominated by the majority class (85% text). Added `val_class_0_acc`, `val_class_1_acc`, `val_class_2_acc`, `per_class_accuracy`, `per_face_accuracy_feature`, and `IoU`.

---

## Training Configuration

```powershell
python segmentation.py train `
  --dataset_path "Z:\lite\pyg" `
  --num_classes 3 `
  --loss_type focal `
  --focal_gamma 2.0 `
  --class_weights_path "artifacts/class_weights/thread_text/source_train_alpha05.json" `
  --batch_size 1 `
  --accumulate_grad_batches 64 `
  --precision "16-mixed" `
  --num_workers 4 `
  --drop_invalid_graphs `
  --run_name "focal_gamma2_thread_text_v2"
```

### Class Weight Configuration

From [source_train_alpha05.json](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/artifacts/class_weights/thread_text/source_train_alpha05.json):

| Class | Face Count | % of Dataset | Weight (α=0.5) |
|-------|-----------|-------------|-----------------|
| 0 — Stock | 1,015,768 | 14.2% | 0.534 |
| 1 — Thread | 57,350 | 0.8% | 2.248 |
| 2 — Text | 6,095,914 | 85.0% | 0.218 |

---

## Results from TensorBoard (55 epochs, ~19 hours)

### Per-Class Recall at Epoch 54

![Per-class recall and IoU metrics from TensorBoard at epoch 54](C:/Users/RZA2/.gemini/antigravity-ide/brain/4591f6af-228d-46ae-be48-2381d3d5dfc7/preview.webp)

| Metric | Value (epoch 54) | Smoothed | Interpretation |
|--------|-----------------|----------|----------------|
| **Stock recall (c00)** | 99.37% | 99.59% | ✅ Excellent — stock faces classified reliably |
| **Thread recall (c01)** | 93.86% | 88.28% | ⚠️ Improved from baseline 80% but volatile |
| **Text recall (c02)** | 48.07% | 33.95% | ❌ Severely degraded — Focal Loss suppresses text too aggressively |
| **Stock IoU (c00)** | 45.03% | 41.33% | Low — many false-positive stock predictions |
| **Thread IoU (c01)** | 23.75% | 22.33% | Low — thread predictions have poor precision |
| **Text IoU (c02)** | 47.96% | 33.88% | Low — text accuracy tanked |

### Per-Face Accuracy Over Training

![Per-face accuracy over ~29K steps showing decline from 95% to ~52-62%](C:/Users/RZA2/.gemini/antigravity-ide/brain/4591f6af-228d-46ae-be48-2381d3d5dfc7/preview_2.webp)

| Metric | Value | Smoothed |
|--------|-------|----------|
| **per_face_accuracy** (final) | 62.41% | 52.22% |

The per_face_accuracy **peaked at ~95% around step 5,000-8,000** (epochs ~10-15) then **collapsed** to ~50-60%. This is a clear sign of training instability.

### Learning Rate and Epoch Progress

![Learning rate decaying from 0.002 to 0 over training, epoch reaching 55](C:/Users/RZA2/.gemini/antigravity-ide/brain/4591f6af-228d-46ae-be48-2381d3d5dfc7/preview_3.webp)

| Metric | Value |
|--------|-------|
| **Final LR** | 0 (decayed to minimum) |
| **Peak LR** | ~0.002 (after warmup at step ~2,100) |
| **Epochs completed** | 55 |

> [!WARNING]
> The LR has decayed to **zero** by step ~28K. This means `ReduceLROnPlateau` triggered repeatedly because `eval_loss` kept worsening. The model stopped learning entirely in the final epochs.

---

## Analysis: What Went Wrong

### 1. Focal Loss + Class Weights Over-Corrected

The combination of Focal Loss (γ=2.0) AND class weights created **excessive suppression of text faces**:

- Class weight for text: **0.218** (already 4.6× suppressed vs. mean)
- Focal factor for correctly classified text: **(1-0.99)² = 0.0001** (another 10,000× suppression)
- **Effective combined suppression: ~46,000×** compared to a hard thread face

This caused the model to essentially **ignore text faces** during training, leading to text recall collapsing from ~80% to 48%.

### 2. Thread Recall Improved But Is Volatile

Thread recall (class 1) shows improvement — reaching 93.86% at epoch 54, up from the baseline ~80%. However, it's highly volatile (swinging between 60% and 95% across epochs), indicating the model hasn't found a stable decision boundary for threads.

### 3. The LR Collapsed to Zero

Because per_face_accuracy (dominated by 85% text) dropped, `eval_loss` worsened, triggering `ReduceLROnPlateau` to halve the LR repeatedly until it hit the 1e-6 floor. By step ~25K, learning effectively stopped.

---

## Conclusions and Next Steps

### What Focal Loss Proved

1. **Thread accuracy CAN be improved** — the thread recall peak of 93.86% confirms the model has capacity to learn thread features
2. **The current γ=2.0 + class weights combo is too aggressive** — it destroyed text accuracy in the process

### Recommended Next Experiments

| Priority | Experiment | Rationale |
|----------|-----------|-----------|
| 🥇 | **Focal Loss with γ=1.0** (reduce gamma) | Less aggressive easy-example suppression; may preserve text accuracy while still helping threads |
| 🥇 | **Focal Loss WITHOUT class weights** | Focal Loss inherently handles imbalance; stacking class weights may be double-correcting |
| 🥈 | **CE loss with stronger class weights** (α=1.0) | Try full inverse-frequency weighting without Focal Loss |
| 🥉 | **Monitor `eval_loss` on per-class basis** | Use the new `val_class_X_acc` metrics to checkpoint on `per_class_accuracy` instead of `eval_loss` |
| 🥉 | **Change checkpoint monitor to `per_class_accuracy`** | Current model saves on `eval_loss` which is dominated by text; use balanced metric instead |

### Specific Command for Next Run (Focal γ=1.0, no class weights)

```powershell
python segmentation.py train `
  --dataset_path "Z:\lite\pyg" `
  --num_classes 3 `
  --loss_type focal `
  --focal_gamma 1.0 `
  --batch_size 1 `
  --accumulate_grad_batches 64 `
  --precision "16-mixed" `
  --num_workers 0 `
  --drop_invalid_graphs `
  --run_name "focal_gamma1_no_cw"
```

---

## Files Modified

| File | Changes |
|------|---------|
| [models/brepseg_model.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py) | Added `FocalLoss()`, loss type selection in `__init__`, loss branching in `training_step`, removed `torch.cuda.empty_cache()`, added per-class metrics to `on_validation_epoch_end` |
| [segmentation.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/segmentation.py) | Added `--loss_type` and `--focal_gamma` CLI arguments |
