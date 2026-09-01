# Class Imbalance Deep Analysis — BrepMFR_PyG

## Problem Statement

Your BrepMFR project performs **per-face classification** on B-rep CAD models using a GNN (EdgeConv) backbone with 3 classes:

| Class | Label | Distribution | Description |
|-------|-------|-------------|-------------|
| Stock | 0 | ~18% | Base geometry |
| Thread | 1 | **~2%** | Added via SolidWorks |
| Text | 2 | **~80%** | Added via SolidWorks |

This is an extreme imbalance — a model predicting "text" everywhere achieves 80% accuracy. Focal Loss (`α=[0.15, 0.50, 0.35]`, `γ=2.0`) has been tried but is **unstable**.

---

## Current Setup Analysis

> [!WARNING]
> Several issues with the current configuration likely contribute to instability.

### Issue 1: Focal Loss Alpha Values Are Inverted
Your current `focal_alpha = [0.15, 0.50, 0.35]` assigns:
- stock (18% of data) → α = 0.15 (lowest weight)
- thread (2% of data) → α = 0.50 
- text (80% of data) → α = 0.35

The α for **stock** is set very low despite it being a minority class. Typically, α should be **inversely proportional** to class frequency. A better α would be something like `[0.40, 0.50, 0.10]`.

### Issue 2: No Sampling Strategy
The dataloader uses a standard random split with no stratified sampling, no weighted sampling, and no oversampling. Batches are dominated by text faces.

### Issue 3: Augmentation Is Disabled
Despite being "enabled" in config, all augmentation parameters are set to identity values (`noise_std: 0.0`, `scale_range: [1.0, 1.0]`, `random_flip: false`, `feature_dropout: 0.0`). This means the model sees the same data every epoch.

### Issue 4: No Stratified Splitting
Random splits may result in the tiny thread class being poorly represented in the validation/test sets.

---

## Why Focal Loss Is Unstable Here

Focal Loss was designed for **object detection** (RetinaNet) where the imbalance is between foreground/background — a binary problem. In your multi-class scenario with a 40:1 ratio between text and thread:

1. **Gradient noise**: With γ=2.0, the loss heavily down-weights "easy" text examples. But with 80% of faces being text, the remaining gradient signal comes disproportionately from the ~2% thread faces — this is **noisy** and can cause oscillations.
2. **α and γ interact non-trivially**: The α weighting and the (1-pt)^γ modulation both affect the same gradient, and their joint tuning is difficult with 3 classes at extreme ratios.
3. **Batch-level variance**: In a batch of 32 graphs, some batches may have zero or very few thread faces. The loss contribution swings wildly between batches.

---

## Recommended Strategies (Ranked)

### 🥇 Strategy 1: Decoupled Training + Class-Balanced Loss *(Highest Confidence)*

**Concept** (Kang et al., ICLR 2020): Learn representations first, then fix the classifier.

The key insight from the paper *"Decoupling Representation and Classifier for Long-Tailed Recognition"* is that **GNN representations learned with standard CE loss are already good** — the problem is the linear classifier becoming biased toward the majority class.

**Implementation:**
```
Stage 1 (70-80% of epochs):
  - Standard CrossEntropyLoss (no class weights)
  - Instance-balanced sampling (standard dataloader)
  - Train full model (backbone + classifier)
  
Stage 2 (20-30% of epochs):
  - Freeze GNN backbone
  - Re-initialize classification head
  - Use Class-Balanced Focal Loss (CB Loss, Cui et al., 2019)
  - Optionally use weighted sampling
```

**Why this works for you:**
- Stage 1 lets the EdgeConv GNN learn rich geometric features from **all** the data without distortion
- Stage 2 fixes the classifier bias with a small, focused re-training
- Avoids the instability of focal loss during backbone training
- The most principled approach for your scenario

| Aspect | Rating |
|--------|--------|
| Expected effectiveness | ⭐⭐⭐⭐⭐ |
| Stability | ⭐⭐⭐⭐⭐ |
| Implementation complexity | Medium |
| Validated for GNN | Architecture-agnostic ✅ |

**Reference:** Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition," ICLR 2020.

---

### 🥈 Strategy 2: LDAM Loss + Deferred Re-Weighting (DRW)

**Concept** (Cao et al., NeurIPS 2019): Enforce larger classification margins for minority classes.

The margin for class $j$ is set to $\Delta_j = C / n_j^{1/4}$, where $n_j$ is the number of samples. This means thread (with fewest samples) gets the **largest margin**, making the model more confident before predicting thread.

**Implementation:**
```
Phase 1 (first 70% of epochs):
  - LDAM Loss with NO re-weighting
  - Standard sampling
  
Phase 2 (last 30% of epochs - "DRW"):
  - LDAM Loss WITH class-balanced re-weighting
  - Weights = 1 / effective_number_of_samples
```

**Why this works:**
- Theoretically motivated by generalization bounds
- The margin prevents the decision boundary from being pushed into minority class space
- DRW timing prevents early-training instability (which is likely what you're seeing with focal loss)

| Aspect | Rating |
|--------|--------|
| Expected effectiveness | ⭐⭐⭐⭐⭐ |
| Stability | ⭐⭐⭐⭐ |
| Implementation complexity | Medium |

**Reference:** Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS 2019.

---

### 🥉 Strategy 3: Dice-CE Hybrid Loss + Weighted Sampling

**Concept:** Combine Generalized Dice Loss (from medical image segmentation) with Cross-Entropy. Your per-face classification is essentially a **segmentation task** on B-rep surfaces, making Dice loss a natural fit.

$$L = \lambda \cdot L_{CE}^{weighted} + (1-\lambda) \cdot L_{GDice}$$

**Generalized Dice Loss** (Sudre et al., 2017) weights each class by the inverse of its volume squared:
$$w_l = \frac{1}{\left(\sum_n r_{ln}\right)^2}$$

This automatically handles imbalance without manual α tuning.

**Implementation:**
```
Loss:
  - λ = 0.5 (tunable)
  - CE component: class weights = inverse frequency
  - Dice component: Generalized Dice Loss (auto-weighted)
  
Sampling:
  - WeightedRandomSampler at graph level
  - Weight per graph = max(class_weight of faces in graph)
```

| Aspect | Rating |
|--------|--------|
| Expected effectiveness | ⭐⭐⭐⭐ |
| Stability | ⭐⭐⭐⭐ |
| Implementation complexity | Medium |
| Validated for GNN | Yes (point cloud segmentation) ✅ |

---

### Strategy 4: Generate More Thread Data *(Most Direct Fix)*

Since you control the data pipeline (SolidWorks + CadSynth), the most impactful thing you can do is **generate more thread samples**:

- Add threads to more base geometries from CadSynth
- Vary thread parameters (pitch, diameter, length, position)
- Create models with multiple thread features
- Aim for at least **10-15% thread representation**

This is the only approach that adds **real geometric diversity** for the minority class.

> [!TIP]
> Even going from 2% → 8% thread representation would dramatically improve all other techniques.

---

### Strategy 5: Quick Fixes (Low-Hanging Fruit)

These can be applied **immediately** alongside any of the above:

#### 5a. Enable Augmentation
Your config has augmentation disabled. Enable it:
```yaml
augmentation:
  enabled: true
  noise_std: 0.01      # was 0.0
  scale_range: [0.95, 1.05]  # was [1.0, 1.0]
  random_flip: true     # was false
  feature_dropout: 0.1  # was 0.0
```

#### 5b. Stratified Splitting
Replace random splits with stratified splits to ensure thread faces appear proportionally in train/val/test:
```python
from sklearn.model_selection import StratifiedShuffleSplit
```

#### 5c. Weighted Random Sampling
Add per-graph weighted sampling to the dataloader:
```python
# Weight each graph by presence of minority classes
sample_weights = []
for data in dataset:
    has_thread = (data.y == 1).any().float()
    has_stock = (data.y == 0).any().float()
    weight = 1.0 + 5.0 * has_thread + 2.0 * has_stock
    sample_weights.append(weight)
sampler = WeightedRandomSampler(sample_weights, len(dataset))
```

#### 5d. Post-hoc Temperature Scaling
After training, calibrate on a balanced validation set:
```python
# Learn temperature T that minimizes NLL on balanced val set
temperature = nn.Parameter(torch.ones(1) * 1.5)
calibrated_logits = logits / temperature
```

---

## What I'd Recommend You Do

> [!IMPORTANT]
> **My top recommendation: Strategy 1 (Decoupled Training) + Strategy 5 (Quick Fixes)**

Here's the concrete plan:

```
1. IMMEDIATE (Quick Fixes):
   ├─ Enable augmentation (noise, scale, flip, feature dropout)
   ├─ Add stratified splitting
   └─ Add weighted random sampling per graph

2. CORE CHANGE (Decoupled Training):
   ├─ Stage 1: Standard CE, full model, ~100 epochs
   └─ Stage 2: Freeze backbone, retrain classifier head with
      Class-Balanced Focal Loss, ~50 epochs

3. EVALUATION:
   ├─ Primary metric: Macro F1
   ├─ Secondary: Per-class F1, balanced accuracy, MCC
   └─ Always report confusion matrix

4. IF STILL INSUFFICIENT:
   └─ Generate more thread data via SolidWorks pipeline
```

---

## Comparison Table

| Technique | Effectiveness (80/18/2) | Stability | Complexity | GNN Validated |
|-----------|:----------------------:|:---------:|:----------:|:-------------:|
| Focal Loss *(current)* | ⭐⭐ | ⭐⭐ | Low | Partial |
| **Decoupled Training** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | ✅ |
| **LDAM + DRW** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | ✅ |
| Dice-CE Hybrid | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | ✅ |
| CB Focal Loss | ⭐⭐⭐⭐ | ⭐⭐⭐ | Low | Partial |
| Weighted Sampling | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | ✅ |
| More Thread Data | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Varies | N/A |
| Post-hoc Calibration | ⭐⭐⭐ (add-on) | ⭐⭐⭐⭐⭐ | Low | ✅ |

---

## Key References

1. Kang et al., *"Decoupling Representation and Classifier for Long-Tailed Recognition,"* ICLR 2020.
2. Cao et al., *"Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss,"* NeurIPS 2019.
3. Cui et al., *"Class-Balanced Loss Based on Effective Number of Samples,"* CVPR 2019.
4. Sudre et al., *"Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations,"* DLMIA 2017.
5. Lin et al., *"Focal Loss for Dense Object Detection,"* ICCV 2017.
6. Leng et al., *"PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions,"* ICLR 2022.
7. Menon et al., *"Long-tail learning via logit adjustment,"* ICLR 2021.
