# Analysis: Thread+Text Class Imbalance & Pipeline Improvement Plan

## The Problem Statement

Thread-only training achieved **~98% accuracy** across 2 classes (stock + thread). When text features were added (3 classes: stock, thread, text), text accuracy jumped to **99%** but thread accuracy **dropped to ~80%**. The user wants solutions **beyond** the obvious "increase class weights."

---

## Root Cause Analysis

### 1. The Numbers Tell the Story

From [source_train_alpha05.json](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/artifacts/class_weights/thread_text/source_train_alpha05.json):

| Class | Face Count | Percentage | Weight (α=0.5) |
|-------|-----------|------------|-----------------|
| 0 — Stock | 1,015,768 | **14.2%** | 0.534 |
| 1 — Thread | 57,350 | **0.8%** | 2.248 |
| 2 — Text | 6,095,914 | **85.0%** | 0.218 |

Compare to thread-only from [lite_source_train_alpha05.json](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/artifacts/class_weights/thread/lite_source_train_alpha05.json):

| Class | Face Count | Percentage | Weight (α=0.5) |
|-------|-----------|------------|-----------------|
| 0 — Stock | 1,578,970 | **59.7%** | 0.902 |
| 1 — Thread | 1,067,512 | **40.3%** | 1.098 |

> [!CAUTION]
> **The real problem is not just imbalance — it's a 106× ratio** between text faces (6.1M) and thread faces (57K). Text faces outnumber thread faces by **two orders of magnitude**. No standard loss weighting scheme can fix this cleanly because:
> - Weight of 2.248 for threads vs 0.218 for text is only a **10× correction**, leaving a **10× residual gap**
> - Pushing α higher (e.g., α=1.0 for full inverse-frequency) would give thread weights ~106× — this makes gradients explode on the rare class and destabilises training

### 2. Why Text Generates So Many Faces (The Geometric Root Cause)

When SolidWorks engraves text on a surface, each character creates dozens of B-Rep faces:
- Each letter stroke has **walls** (vertical faces of the engraving)
- Each letter has a **bottom** face
- Each curved letter (O, S, B, etc.) gets **tessellated** into many faces
- A single word like "SOLIDWORKS" could generate **100+ faces**, all labelled as class 2

Meanwhile, a thread feature creates only **~4-8 faces** (the helical sweep surface + end faces).

**This means text is geometrically verbose**: a single text operation produces far more labelled faces than a single thread operation, even though functionally they are equally "one feature."

### 3. Why This Is Fundamentally Different from the Original 25-Class Problem

The original BrepMFR 25-class training had Stock at 58% dominance — bad, but manageable with α=0.5 weighting. Here, text at **85%** is far worse, AND the minority class (thread) is geometrically distinct from text. The model learns "if it's not stock, it's probably text" as an excellent heuristic — 85% of non-stock faces ARE text.

---

## Solutions (Beyond Loss Weighting)

### Solution 1: Per-Graph Balanced Sampling (Data-Level)

> [!IMPORTANT]
> **Impact: High. Complexity: Low. No architecture changes needed.**

**The Idea:** Instead of weighting the loss, control which **graphs** the DataLoader serves. Sample graphs so that each batch sees roughly equal total faces from each class.

**Why this is better than loss weighting:**
- Loss weighting changes gradient magnitudes but not gradient **directions** — the model still sees 106× more text faces per epoch, just with smaller gradients on them
- Sampling changes what the model **actually sees**, which is a stronger signal

**Implementation (two variants):**

**A) Class-stratified graph sampling:** Precompute per-graph class distribution. At each epoch, build a sampling weight per graph proportional to the inverse frequency of its rarest non-stock class. Graphs rich in thread faces get sampled more often.

```python
# Pseudocode for data/dataset.py
class BalancedCADSynth(CADSynth):
    def __init__(self, ...):
        super().__init__(...)
        self.sampling_weights = self._compute_sampling_weights()
    
    def _compute_sampling_weights(self):
        weights = []
        for path in self.file_paths:
            g = torch.load(path)
            labels = g.label_feature.numpy()
            # Weight by inverse-frequency of the rarest class present
            counts = np.bincount(labels, minlength=self.num_class)
            # Boost graphs that contain threads
            thread_ratio = counts[1] / max(1, counts.sum())
            weights.append(1.0 + 10.0 * thread_ratio)  # tunable
        return weights
    
    def get_dataloader(self, batch_size, ...):
        sampler = WeightedRandomSampler(self.sampling_weights, 
                                         num_samples=len(self), 
                                         replacement=True)
        return DataLoader(self, batch_size=batch_size, sampler=sampler, ...)
```

**B) Class-balanced mini-batch construction:** Each batch is required to contain a minimum number of thread-bearing graphs. This is the approach used by DetectoRS, BalancedGroupSoftmax, and other detection frameworks for rare objects.

**Downstream consequence:** The model will see some thread-heavy graphs multiple times per epoch. This is fine as long as you maintain standard augmentation (random rotation is already in `data/utils.py`).

---

### Solution 2: Focal Loss (Gradient-Level)

> [!IMPORTANT]
> **Impact: Medium-High. Complexity: Low. Drop-in replacement for CE loss.**

**The Idea:** Focal Loss (Lin et al., ICCV 2017) dynamically down-weights **easy examples** regardless of class. Text faces that the model already classifies at 99% confidence get near-zero gradient. Thread faces that the model struggles with get amplified gradient.

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

p_t = predicted probability of the TRUE class
γ   = focusing parameter (2.0 recommended)
α_t = optional class-level weight
```

**Why this is better than plain CE weighting for your case:**
- CE weighting gives ALL text faces weight 0.218, even the ones the model already gets right trivially
- Focal Loss gives near-zero weight to easy text faces **and** easy stock faces, concentrating learning capacity on the **hard boundary cases** (text-vs-thread confusion, stock-vs-thread confusion)

**Implementation:**

```python
# Drop-in for models/brepseg_model.py
def FocalLoss(label_onehot, predict_prob, gamma=2.0, class_weight=None, epsilon=1e-12):
    p_t = (label_onehot * predict_prob).sum(dim=-1)  # Prob of true class
    focal_weight = (1.0 - p_t) ** gamma             # ↓ for easy, ↑ for hard
    ce = -torch.log(p_t + epsilon)
    loss = focal_weight * ce
    if class_weight is not None:
        true_class = label_onehot.argmax(dim=-1)
        cw = class_weight[true_class]
        loss = loss * cw
    return loss.mean()
```

**Critical note for this project:** The current classifier outputs `F.softmax(x, dim=-1)` not raw logits. Focal Loss works on softmax probabilities directly (the formula above handles that). No architecture change needed.

**Downstream consequence:** Focal Loss with γ=2.0 will effectively "ignore" 85% of easy text faces during training. Combined with class weights, this gives thread faces ~1000× effective gradient amplification over easy text faces, without destabilising training.

---

### Solution 3: Feature-Level Data Generation Balancing (Pipeline-Level)

> [!WARNING]
> **This is the most impactful solution but requires changes to the SolidWorks macro.**

**The Idea:** The imbalance is created at data generation time. Fix it there.

**Current macro logic:** The macro adds threads to ALL cylindrical edges on a STEP file. If a separate text macro runs on the same file, it adds text to ALL flat faces. Since most parts have far more flat faces than cylindrical edges, text dominates.

**Proposed fixes:**

**A) Cap text face count per model:** In the VBA macro, after adding text features, count the total text-labelled faces. If they exceed a threshold (e.g., 50), randomly select which text operations to keep and suppress the rest. This prevents any single model from flooding the dataset with text faces.

**B) Generate text-only and thread-only models separately, then mix at the dataset level:** Instead of adding both features to every model, generate three pools:
- Pool A: Stock-only models (baseline geometry)
- Pool B: Thread-augmented models (same pool you already had at 98%)  
- Pool C: Text-augmented models
- Pool D (optional): Thread+text combined models

At training time, mix pools B, C, D in controlled ratios. This gives you **independent control** over the class distribution without touching loss functions.

**C) Generate multi-text-density variations:** Like the 6 thread variations, create text variations with different densities (1 text feature, 3 text features, 8 text features). This naturally produces a distribution of text face counts instead of always maxing out.

**Downstream consequence for production:** If the production model will encounter parts with varying amounts of text vs. threads, the training distribution should roughly match the expected production distribution. Solution B gives you the knob to tune this.

---

### Solution 4: Hierarchical / Coarse-to-Fine Classification

> [!IMPORTANT]
> **Impact: High. Complexity: Medium. Architecture change.**

**The Idea:** Instead of a flat 3-class (or eventually N-class) classifier, use a two-stage classification:

```
Stage 1: Stock vs. Feature  (binary, 99%+ accuracy expected)
                |
          (if Feature)
                |
Stage 2: Thread vs. Text vs. NewFeature1 vs. ...  (only on feature faces)
```

**Why this helps:**
- The binary Stock-vs-Feature classifier sees a ~14% vs 86% split — much more balanced
- The feature-type classifier operates on a **pre-filtered subset** where stock faces are removed, giving thread 57K/(57K+6.1M) ≈ 0.9% — still imbalanced, but now Focal Loss + class weights can handle it because the easy stock majority is gone
- Each classifier specialises on its own decision boundary

**Implementation approach:** Add a second classifier head in `BrepSeg`:

```python
class BrepSeg(pl.LightningModule):
    def __init__(self, args):
        ...
        self.binary_classifier = NonLinearClassifier(args.dim_node, 2, args.dropout)  # stock vs feature
        self.feature_classifier = NonLinearClassifier(args.dim_node, num_feature_classes, args.dropout)
    
    def training_step(self, batch, batch_idx):
        ...
        z = self.attention([node_z, graph_z])
        
        # Stage 1: binary classification
        binary_pred = self.binary_classifier(z)
        binary_label = (labels > 0).long()  # 0=stock, 1=feature
        loss_binary = CrossEntropyLoss(F.one_hot(binary_label, 2), binary_pred)
        
        # Stage 2: feature-type classification (only on non-stock faces)
        feature_mask = (labels > 0)
        if feature_mask.any():
            z_features = z[feature_mask]
            feature_pred = self.feature_classifier(z_features)
            feature_labels = labels[feature_mask] - 1  # shift to 0-indexed
            loss_feature = FocalLoss(F.one_hot(feature_labels, nf), feature_pred)
        else:
            loss_feature = 0.0
        
        loss = loss_binary + loss_feature
```

**Downstream consequence for production:** At inference time, you run both heads. Only faces predicted as "Feature" by the binary head get sent to the feature classifier. This reduces false-positive feature predictions on stock faces (a common production failure mode).

**Scalability advantage:** When you add new feature types (chamfers, pockets, etc.), you only retrain the feature classifier, not the entire model. The binary Stock-vs-Feature boundary is stable.

---

### Solution 5: Graph-Level Contrastive Pre-Training

> [!TIP]
> **Impact: Medium-High. Complexity: Medium-High. Separate pre-training phase.**

**The Idea:** Before supervised training, pre-train the `BrepEncoder` with a self-supervised contrastive objective (e.g., SimCLR/MoCo-style) on **all available B-Rep data** — including unlabelled parts. The encoder learns general geometric representations without being biased by class distribution.

Then, fine-tune the classifier head on the labelled data. Because the encoder already understands geometry, the classifier needs far fewer examples per class to achieve high accuracy.

**Why this helps with imbalance:**
- The encoder's representation quality is decoupled from label distribution
- Thread faces and text faces will naturally cluster in different embedding regions because their geometry is fundamentally different
- Even with 57K thread faces, fine-tuning a classifier on top of a good representation is trivial

**Downstream consequence:** This is a significant engineering effort but would be the most robust foundation for scaling to many feature types. It's the approach used by state-of-the-art 3D understanding models (Point-MAE, PointGPT, etc.).

---

## Pipeline Improvements for Scaling to More Feature Types

### Improvement 1: Modular Feature Generation Architecture

**Current limitation:** Thread generation is hardcoded in `ThreadCreationScript8.bas`. Adding text required a separate macro. Adding chamfers, pockets, etc. would require yet another macro.

**Proposed architecture:**

```
Master Orchestrator (Python)
    │
    ├── Feature Plugin: Threads (threadgen.swp)
    │     └── Config: types, sizes, variations
    ├── Feature Plugin: Text (textgen.swp)  
    │     └── Config: fonts, sizes, positions
    ├── Feature Plugin: Chamfers (chamfergen.swp)
    │     └── Config: angles, widths
    └── Feature Plugin: Fillets (filletgen.swp)
          └── Config: radii ranges

Per STEP file:
1. Master selects which plugins to apply (configurable mix ratio)
2. Each plugin runs independently
3. Master controls total feature density per model
4. Final BrepJson export happens once
```

**Key benefit:** Feature mix ratios are controlled at the orchestrator level, preventing any single feature type from dominating the dataset.

---

### Improvement 2: Adaptive Class-Weight Computation During Training

**Current limitation:** Class weights are precomputed from the dataset and frozen throughout training. As the model learns, the "difficulty" of each class changes — early in training, all classes are hard; late in training, text is trivially easy but thread is still hard.

**Proposed improvement:** Recompute effective class weights every N epochs based on per-class accuracy from the last validation:

```python
# In on_validation_epoch_end:
for c in range(num_classes):
    if per_class_acc[c] < target_accuracy:
        # Increase weight for struggling classes
        self.class_weights[c] *= 1.1
    elif per_class_acc[c] > 0.99:
        # Decrease weight for already-mastered classes
        self.class_weights[c] *= 0.9
```

This is called **curriculum-weighted training** and naturally focuses training capacity where it's needed most.

---

### Improvement 3: Replace the Custom CE Loss with Standard PyTorch Losses

**Current code issue in [brepseg_model.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py#L49-L69):**

```python
def CrossEntropyLoss(label, predict_prob, ...):
    ce = -label * torch.log(predict_prob + epsilon)
```

This applies `log()` to **softmax probabilities**, which is numerically less stable than `torch.nn.CrossEntropyLoss` which uses `log_softmax` internally (the LogSumExp trick avoids underflow). The `epsilon=1e-12` partially mitigates this, but:

- When a confident prediction gives `predict_prob ≈ 0` for the true class, `log(1e-12) = -27.6`, creating a massive loss spike
- This is especially problematic for thread faces where the model is uncertain — the loss can spike by 10-100× on individual faces

**Proposed fix:** Switch the classifier to output raw logits (remove `F.softmax` from [NonLinearClassifier.forward](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py#L37-L46)) and use `torch.nn.functional.cross_entropy` which combines log-softmax + NLL in a numerically stable way.

```python
# In NonLinearClassifier:
def forward(self, inp):
    ...
    x = self.linear4(x)
    return x  # Raw logits, NOT softmax

# In training_step:
loss = F.cross_entropy(logits, labels, weight=self.class_weights)
```

> [!WARNING]
> This requires careful migration because the current `DomainAdversarialLoss` and `EntropyLoss` in Stage 2 also operate on softmax probabilities. You'd need to apply softmax only where needed (entropy loss, domain adversarial input) and keep raw logits for CE.

---

### Improvement 4: Multi-Task Learning with Feature Detection Head

**Idea:** Add an auxiliary task: for each face, predict **"does this face belong to ANY machining feature?"** (binary) in addition to the primary N-class classification. This auxiliary signal is class-balanced (since it's binary) and helps the encoder learn a strong feature-vs-stock boundary.

```python
loss = loss_primary + 0.3 * loss_binary_detection
```

This is lightweight and composable with any of the other solutions.

---

### Improvement 5: Test-Time Augmentation (TTA) for Thread Faces

At inference time, run the model on multiple random rotations of the same graph and **average predictions**. This is free (no retraining) and empirically improves accuracy on geometrically challenging faces by 2-5%.

Already partially supported via `random_rotate=True` in the dataset, but TTA at inference is not implemented.

---

## Recommended Priority Order

| Priority | Solution | Effort | Expected Thread Acc Improvement |
|----------|---------|--------|-------------------------------|
| 🥇 1 | **Focal Loss** (Solution 2) | 1-2 hours | +5-10% (to ~85-90%) |
| 🥈 2 | **Per-Graph Balanced Sampling** (Solution 1) | 2-4 hours | +5-8% (compounds with #1) |
| 🥉 3 | **Fix Numerical Stability** (Improvement 3) | 2-3 hours | +2-3% (reduces gradient noise) |
| 4 | **Hierarchical Classification** (Solution 4) | 1-2 days | +8-15% (to ~93-98%) |
| 5 | **Data Generation Balancing** (Solution 3) | 1-2 days (macro changes) | +10-15% (most sustainable long-term) |
| 6 | **Adaptive Weights** (Improvement 2) | 3-4 hours | +2-5% |
| 7 | **Modular Feature Plugins** (Improvement 1) | 1-2 weeks | N/A (infrastructure) |
| 8 | **Contrastive Pre-Training** (Solution 5) | 1-2 weeks | +5-10% |

## Open Questions

1. **What is the target production distribution?** If real-world parts have roughly equal numbers of thread and text features, IWDAN with the correct priors would help. If real-world parts are also text-heavy, then 80% thread accuracy might be acceptable.

2. **Are thread and text faces ever geometrically confusable?** Or is the 80% thread accuracy caused by thread faces being misclassified as stock (not text)? A confusion matrix would answer this and determine which solutions to prioritise.

3. **Can you share the actual confusion matrix from the thread+text run?** This would reveal whether threads are being confused with text (feature boundary problem) or with stock (detection problem) — two very different failure modes requiring different fixes.

4. **Is the text generation macro (`threadplustextgen8.swp`) adding text to every possible flat face?** If so, capping the number of text operations per model (Solution 3A) would be the single highest-impact change.

5. **How many more feature types are planned?** If >3 more, the hierarchical approach (Solution 4) and modular plugin architecture (Improvement 1) should be prioritised now to avoid repeated rework.
