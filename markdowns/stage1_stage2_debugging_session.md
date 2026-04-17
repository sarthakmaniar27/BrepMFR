# BrepMFR Stage 1 & Stage 2 Debugging Session

**Date:** 2026-04-16 / 2026-04-17  
**Goal:** Diagnose why Stage 2 domain adaptation achieves ~77% on MFCAD++ vs paper's 90.32%

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Paper: Key Specifications](#2-paper-key-specifications)
3. [Stage 1 — Code Audit & Fixes](#3-stage-1--code-audit--fixes)
4. [Stage 2 — Architecture & Concepts](#4-stage-2--architecture--concepts)
5. [Stage 2 — All Bugs Found & Fixed](#5-stage-2--all-bugs-found--fixed)
6. [domain_adapt.py — Fixes](#6-domain_adaptpy--fixes)
7. [grl.py — Fix](#7-grlpy--fix)
8. [TensorBoard Interpretation — Stage 1 (port 6007)](#8-tensorboard-interpretation--stage-1-port-6007)
9. [TensorBoard Interpretation — Stage 2 (port 6008)](#9-tensorboard-interpretation--stage-2-port-6008)
10. [Stage 2 Failure Diagnosis & Final Fixes](#10-stage-2-failure-diagnosis--final-fixes)
11. [Complete Fix Summary](#11-complete-fix-summary)
12. [Key Concepts Explained](#12-key-concepts-explained)

---

## 1. Project Overview

**BrepMFR** (B-rep Machining Feature Recognition) is a graph transformer model that classifies machining features (e.g. holes, slots, pockets) on B-rep CAD models.

**Two-stage training:**
- **Stage 1:** Supervised training on labeled synthetic CAD (`CADSynth`) using `segmentation.py` → `BrepSeg` model
- **Stage 2:** Domain adaptation from `CADSynth` (source, labeled) → `MFCAD++` (target, unlabeled) using `domain_adapt.py` → `DomainAdapt` model

**Paper results (Table 3):**
- CADSynth source-only: 81.50%
- Domain adaptation: **90.32%**
- User was achieving: ~77% (no improvement from source-only)

---

## 2. Paper: Key Specifications

### Network Settings (§5.1.1)
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| β1, β2 | 0.9, 0.999 |
| ε | 1e-8 |
| weight_decay | 0.01 |
| Initial LR | 0.001 |
| Warmup steps | 5,000 |
| Scheduler | ReduceLROnPlateau (when loss stops decreasing) |
| Max epochs | 200 |
| Batch size | 64 |

### Stage 2 Objective (Eq. 17)
```
L_total = L_label + α·L_entropy + β·L_adv
α = 0.1,  β = 0.3
```

- **L_label**: Cross-entropy on source nodes (supervised)
- **L_entropy**: Entropy minimization on target predictions (unsupervised)
- **L_adv**: Adversarial domain discrimination loss (via GRL)

### Ablation (Table 6)
| Loss | Target Acc |
|---|---|
| L_label only | 61.82% |
| L_label + L_adv | 91.75% |
| L_label + L_entropy + L_adv | 92.74% |

---

## 3. Stage 1 — Code Audit & Fixes

**File:** `models/brepseg_model.py`

### What was wrong

The user had modified `configure_optimizers()` to use **differential learning rates** (three separate param groups with different LRs):
```python
# BEFORE (user's modified version)
encoder_params → lr=1e-4
attention_params → lr=3e-4
classifier_params → lr=5e-4
# plus custom warmup targeting each group's lr
patience=15, cooldown=5, min_lr=1e-5
```

This is a fine-tuning strategy, not appropriate for Stage 1 which trains from scratch.

### Evidence from TensorBoard (Stage 1, port 6007)

The original author's TensorBoard showed:
- Peak LR = **0.002** (not 0.001 as paper states)
- Warmup: 0 → 0.002 over 5000 steps (~4 epochs)
- ReduceLROnPlateau fired once (0.002 → 0.001) around step 80k
- Accuracy: **99.69%** after 50 epochs, still rising

This confirmed the author used `lr=0.002` as peak LR.

### Fix Applied

```python
# AFTER — matches what author actually ran
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),   # single param group
        lr=0.002,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=1e-4,
        threshold_mode='rel',
        min_lr=1e-6,
        cooldown=2,
        verbose=False,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "eval_loss",
        },
    }

def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    optimizer.step(closure=optimizer_closure)
    # Warmup: linearly ramp LR from 0 → 0.002 over first 5000 steps
    if self.trainer.global_step < 5000:
        lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_scale * 0.002
```

### Stage 1 Current Status (from TensorBoard at epoch 38)

```
train_loss:        0.029  (steadily falling)
per_face_accuracy: 99.16% (still climbing, warmup done at epoch 4)
LR:                0.002  (ReduceLROnPlateau not fired yet)
Steps/epoch:       1248   (at batch_size=64)
```

Training normally. Should reach ~99.7%+ by epoch 200.

---

## 4. Stage 2 — Architecture & Concepts

### What Stage 2 Does

Stage 2 loads the Stage 1 checkpoint and fine-tunes with adversarial domain adaptation:

```
Source graphs (CADSynth, labeled)  ─┐
                                     ├→ BrepEncoder → Attention Fusion → Classifier → L_label
Target graphs (MFCAD++, unlabeled) ─┘                              ↓
                                                              DomainDiscriminator
                                                                    ↑
                                                               GRL (reverses gradients)
```

**Three modules loaded from Stage 1 checkpoint:**
- `brep_encoder` — graph transformer
- `attention` — inter-graph attention fusion (Eq. 10: `Z = att_n·Z_n + att_g·Z_g`)
- `classifier` — 4-layer MLP (256→512→512→256→num_classes)

**New module added for Stage 2:**
- `domain_adv` = `DomainAdversarialLoss(DomainDiscriminator(...))` — trained from scratch

### GRL (Gradient Reversal Layer) — How Lambda Works

```
λ(i) = 2/(1 + exp(-α × i/N)) - 1

With α=1, N=max_iters:
  i=0:         λ = 0.00   (no adversarial signal)
  i=0.1×N:     λ ≈ 0.05
  i=0.5×N:     λ ≈ 0.25
  i=N:         λ ≈ 0.46   (NOT 1.0 — sigmoid asymptote)
  i=∞:         λ → 1.00
```

In **forward**: GRL is identity (`output = input × 1.0`)  
In **backward**: gradients are multiplied by `-λ` (reversed sign)  
Effect: encoder is trained to fool the discriminator (feature alignment)

### BatchNorm Behavior

- **During training**: BN updates running stats from mixed source+target batches
- **During validation/test**: BN frozen at eval mode, uses running stats
- This is an inherent domain adaptation limitation (mixed BN stats)

---

## 5. Stage 2 — All Bugs Found & Fixed

### Bug 1: `weight_decay=0.01` on Pre-trained Modules (CRITICAL)

**File:** `models/transfer_model.py` → `configure_optimizers()`

**What was wrong:**
```python
# BEFORE — weight decay on ALL modules
optimizer = AdamW(self.brep_encoder.parameters(), lr=0.0001,
                  betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)
# + attention and classifier also had weight_decay=0.01
```

**Why this is fatal:**  
AdamW applies weight decay as: `w ← w × (1 - lr × wd)` per step.

```
lr=1e-4, wd=0.01:  per-step factor = (1 - 1e-4 × 0.01) = 0.999999
200 epochs × 2800 steps = 560,000 steps:
  w_final = w_init × 0.999999^560000 ≈ w_init × 0.57
```

**Stage 1 weights shrink to 57%** of their original values — systematically destroying the learned representations before domain adaptation can work.

**Fix:**
```python
# AFTER — no weight decay on pre-trained modules
optimizer = torch.optim.AdamW(
    self.brep_encoder.parameters(),
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,      # ← no decay on pre-trained weights
)
optimizer.add_param_group({
    "params": self.attention.parameters(),
    "lr": 0.0001,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,   # ← no decay on pre-trained weights
})
optimizer.add_param_group({
    "params": self.classifier.parameters(),
    "lr": 0.0001,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,   # ← no decay on pre-trained weights
})
optimizer.add_param_group({
    "params": self.domain_adv.parameters(),
    "lr": 0.001,           # discriminator trained from scratch — higher LR
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,  # ← regularize the NEW discriminator only
})
```

### Bug 2: Attention Module Missing from Optimizer (CRITICAL)

**File:** `models/transfer_model.py` (original author's code)

**What was wrong:**
```python
# ORIGINAL author code — attention MISSING
optimizer = AdamW(self.brep_encoder.parameters(), lr=0.0001, betas=(0.99, 0.999))
optimizer.add_param_group({'params': self.classifier.parameters(), ...})
optimizer.add_param_group({'params': self.domain_adv.parameters(), ...})
# self.attention NOT INCLUDED → frozen during Stage 2
```

The inter-graph attention fusion (which all three losses flow through) was frozen throughout Stage 2. The fusion module couldn't adapt to target domain features.

**Fix:** Added `self.attention.parameters()` as a param group (shown above).

### Bug 3: GRL `estimated_steps_per_epoch` Hard-coded

**File:** `models/transfer_model.py` → `DomainAdapt.__init__()`

**What was wrong:**
```python
# BEFORE
estimated_steps_per_epoch = 1400  # calibrated for batch_size=64
# Running at batch_size=32 → actual steps = 2800 → GRL ramps 2× too fast
```

**Fix:**
```python
# AFTER — dynamic based on actual batch_size
estimated_steps_per_epoch = max(1, round(1400 * 64 / args.batch_size))
max_training_iters = args.max_epochs * estimated_steps_per_epoch
print(f"[GRL] batch_size={args.batch_size}, "
      f"estimated_steps_per_epoch={estimated_steps_per_epoch}, "
      f"max_training_iters={max_training_iters}")
```

### Bug 4: `eval_loss = 1.0 / accuracy` (Original Author Code)

**File:** `models/transfer_model.py` → original `validation_step()`

**What was wrong:**
```python
# ORIGINAL author code
loss = 1.0 / np.mean(per_face_comp_t)   # blows up if accuracy=0
self.log("eval_loss", loss)
# ReduceLROnPlateau monitored this → fires when accuracy was briefly perfect
```

**Fix:** Replaced with the actual combined validation objective:
```python
val_obj = loss_s + 0.3 * loss_adv + 0.1 * loss_t
self.log("eval_loss", val_obj)
```

### Bug 5: ReduceLROnPlateau Premature Decay (DISCOVERED FROM TENSORBOARD)

**File:** `models/transfer_model.py` → `configure_optimizers()`

**What happened (observed in TensorBoard at epoch 22):**
- `eval_loss_s` started at 0.01696 (epoch 1) — already near-optimal from Stage 1
- Never improved below epoch 1 value (source CE was already trained in Stage 1)
- With `patience=5`, scheduler fired at epoch 7 → LR halved to 0.00005
- Fired again at epoch 16 → LR = 0.000025 (1/4 of target)
- Encoder was barely moving by epoch 22

**Root cause:** ReduceLROnPlateau is inappropriate for Stage 2 because:
1. Source CE is already optimal from Stage 1 — it never improves in Stage 2
2. The adversarial objective is non-monotonic by design

**Fix:**
```python
# AFTER — no scheduler for Stage 2
# Fixed LR throughout: encoder stays at 0.0001 for all 200 epochs
return {"optimizer": optimizer}
```

### Bug 6: Entropy Confirmation Bias (DISCOVERED FROM TENSORBOARD)

**File:** `models/transfer_model.py` → `training_step()`

**What happened (observed in TensorBoard at epoch 22):**
```
train_loss_t:  0.047 (epoch 1) → 0.006 (epoch 22)   [7× drop]
per_face_accuracy_target: 76.95% → 74.08%             [declining]
```

Entropy loss collapsed while accuracy fell — predictions became **confidently wrong**.  
Classic confirmation bias: entropy minimization sharpens whatever the model currently predicts, even if incorrect.

**Fix:**
```python
# AFTER — entropy disabled until alignment stabilises
loss = loss_s + 0.3 * loss_adv   # α=0 entropy

# Re-enable with small coefficient (e.g. 0.01) only after
# train_transfer_acc approaches ~50%
```

---

## 6. domain_adapt.py — Fixes

### Fix 1: `num_workers` Default on Windows
```python
# BEFORE
default=12  # crashes/stalls on Windows

# AFTER
default=0   # Windows-safe
```

### Fix 2: `max_epochs` Default Handling
```python
# BEFORE — no validation; PL defaults max_epochs=None (unlimited)
# → GRL max_iters = None × steps → crash or wrong schedule

# AFTER
if args.traintest == "train":
    assert args.pre_train is not None, \
        "Stage 2 training requires --pre_train (path to Stage 1 checkpoint)"
    if args.max_epochs is None:
        args.max_epochs = 200
        print("[domain_adapt] --max_epochs not specified, defaulting to 200 for GRL schedule")
```

### Fix 3: Double Checkpoint Load in Test Mode
```python
# BEFORE — checkpoint loaded twice
model = DomainAdapt.load_from_checkpoint(args.checkpoint)
trainer.test(model, dataloaders=[test_loader], ckpt_path=args.checkpoint)

# AFTER — load once
model = DomainAdapt.load_from_checkpoint(args.checkpoint)
trainer.test(model, dataloaders=[test_loader])   # no ckpt_path
```

### Fix 4: ModelCheckpoint Filename
```python
# BEFORE — ambiguous filenames (best.ckpt, best-v1.ckpt, ...)
filename="best",
save_top_k=10,

# AFTER — informative filenames, less disk waste
filename="best-{epoch:03d}-{per_face_accuracy_target:.4f}",
save_top_k=3,
```

---

## 7. grl.py — Fix

### `np.float` Removed in NumPy 1.24

**File:** `models/modules/domain_adv/grl.py`

```python
# BEFORE — crashes on NumPy >= 1.24
coeff = np.float(
    2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
    - (self.hi - self.lo) + self.lo
)

# AFTER
coeff = float(
    2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
    - (self.hi - self.lo) + self.lo
)
```

---

## 8. TensorBoard Interpretation — Stage 1 (port 6007)

**At epoch 38/200:**

| Metric | Value | Interpretation |
|---|---|---|
| `per_face_accuracy` | 99.16% | Strong, still climbing |
| `train_loss` | 0.029 | Healthy, still decreasing |
| `eval_loss` | 0.022 | Low, stable |
| `current_lr` | 0.002 | Warmup done at epoch 4, no plateau decay yet |
| Steps/epoch | 1248 | Consistent with CADSynth training split |

**LR warmup confirmed working:**
```
Epoch 1 (step 1247):  lr = 0.000499 = 0.002 × (1247/5000)
Epoch 4 (step 4991):  lr = 0.001997 ≈ 0.002 (almost done)
Epoch 5+ (step 6239): lr = 0.002000 (flat — warmup complete)
```

**Anomaly at epoch 18:** eval_loss spiked (0.027 → 0.059), accuracy dropped (98.9% → 97.6%), recovered fully next epoch. Transient event, not concerning.

**Status:** Healthy Stage 1 training. Let run until accuracy plateaus and ReduceLROnPlateau decays LR. Use best checkpoint as `--pre_train` for Stage 2.

---

## 9. TensorBoard Interpretation — Stage 2 (port 6008)

**At epoch 22/200:**

### Full Metric Snapshot

| Metric | Epoch 1 | Epoch 22 | Trend | Healthy? |
|---|---|---|---|---|
| `per_face_accuracy_target` | 76.95% | 74.08% | ↓ declining | **NO** |
| `per_face_accuracy_source` | 99.51% | 99.53% | → stable | ✓ |
| `train_loss` | 0.102 | 0.165 | ↑ rising | Expected (adv dominates) |
| `train_loss_s` | 0.011 | 0.010 | → flat | ✓ |
| `train_loss_t` | 0.047 | 0.006 | ↓ collapsed | **NO** |
| `train_loss_transfer` | 0.286 | 0.516 | ↑ rising | Expected |
| `train_transfer_acc` | 85.9% | 71.3% | ↓ slowly | Progressing but too slow |
| `grl_lambda` | 0.0011 | 0.0479 | ↑ growing | Too small (4.8%) |
| `current_lr` | 0.000025 | 0.000025 | ↓ already decayed 4× | **CRITICAL** |

### LR Decay Timeline (catastrophic)
```
Epochs  1-4:  lr = warmup (0.000025 → 0.0001)
Epochs  5-7:  lr = 0.0001   (target reached)
Epoch   8:    lr = 0.00005  ← ReduceLROnPlateau fired (patience=5 hit)
Epoch  17:    lr = 0.000025 ← fired again
Epoch  22:    lr = 0.000025 ← 1/4 of intended encoder LR
```

### Key Observations

**1. Discrimination declining correctly** — `train_transfer_acc` falling from 86% → 71% confirms the encoder IS slowly learning to confuse the discriminator. GRL is working, just very slowly.

**2. But LR already dead** — By the time alignment would have produced results, the encoder LR was already at 0.000025, too small to make meaningful weight updates.

**3. Entropy collapse** — `train_loss_t` at 0.006 while accuracy falling = confidently wrong predictions being reinforced.

**4. GRL lambda at 4.8%** — Adversarial gradient is only 4.8% of full strength. The encoder barely receives alignment signal.

---

## 10. Stage 2 Failure Diagnosis & Final Fixes

### Root Cause Summary

```
Stage 2 failing (74% and falling) is caused by THREE simultaneous issues:

1. LR scheduler fires at epoch 7 (and again at epoch 16)
   → eval_loss_s was already optimal from Stage 1, never improves
   → encoder LR = 0.000025 by epoch 22 (barely moving)

2. Entropy minimization confirmation bias
   → train_loss_t collapses to 0.006
   → predictions become confidently WRONG on target
   → target accuracy actively decreases

3. GRL lambda too small (4.8% at epoch 22)
   → adversarial signal too weak
   → discriminator still at 71% accuracy (should be ~50%)
   → BUT this alone wouldn't cause decline — it would just mean slow progress
```

### Fixes Applied

**Fix A — Remove ReduceLROnPlateau (transfer_model.py):**
```python
# No scheduler — fixed LR throughout Stage 2
return {"optimizer": optimizer}
```

**Fix B — Disable entropy loss (transfer_model.py):**
```python
# Training loss
loss = loss_s + 0.3 * loss_adv   # entropy disabled (α=0)

# Validation objective (same)
val_obj = loss_s + 0.3 * loss_adv
```

### When to Re-enable Entropy

Once `train_transfer_acc` approaches ~50% (discriminator confused = good alignment), you can re-enable entropy at a small coefficient:
```python
loss = loss_s + 0.3 * loss_adv + 0.02 * loss_t   # start small
```

---

## 11. Complete Fix Summary

### Files Modified

#### `models/transfer_model.py`

| Fix | Before | After | Impact |
|---|---|---|---|
| `weight_decay` on pre-trained | 0.01 for encoder/attention/classifier | **0.0** for encoder/attention/classifier | Prevents 43% weight decay of Stage 1 knowledge |
| Attention in optimizer | Missing (frozen) | Added as param group | Attention can adapt to target domain |
| Scheduler | ReduceLROnPlateau (fires at epoch 7) | **None — fixed LR** | Encoder stays at 0.0001 throughout |
| Entropy loss | `0.1 * loss_t` | **Disabled (0.0)** | Stops confirmation-bias collapse |
| GRL max_iters | Hard-coded 1400 steps/epoch | `round(1400 * 64 / batch_size)` | Correct lambda schedule for any batch_size |
| Validation objective | `1/accuracy` (original) | `loss_s + 0.3*loss_adv` | Real loss, not unstable inverse accuracy |

#### `models/modules/domain_adv/grl.py`

| Fix | Before | After |
|---|---|---|
| NumPy compatibility | `np.float(...)` | `float(...)` |

#### `models/brepseg_model.py`

| Fix | Before | After |
|---|---|---|
| `configure_optimizers` | Differential LRs (1e-4, 3e-4, 5e-4), patience=15 | Single LR=0.002, patience=5 |
| `optimizer_step` | Per-group warmup to (1e-4, 3e-4, 5e-4) | Single warmup to 0.002 |

#### `domain_adapt.py`

| Fix | Before | After |
|---|---|---|
| `num_workers` default | 12 (crashes Windows) | **0** |
| `max_epochs` guard | None | Defaults to 200 if not specified |
| `--pre_train` guard | No check | Assert at startup |
| Double checkpoint load | Loaded twice in test | Once only |
| Checkpoint filename | `"best"` (ambiguous) | `"best-{epoch:03d}-{per_face_accuracy_target:.4f}"` |
| `save_top_k` | 10 | **3** |

---

## 12. Key Concepts Explained

### What is an Optimizer?

The optimizer adjusts model weights based on gradients. **AdamW**:
- Keeps momentum (β1=0.9: 90% old momentum, 10% new gradient)
- Adapts per-weight step size (β2=0.999)
- Weight decay: multiplies weights by `(1 - lr × wd)` each step (regularization)

### What is a Learning Rate Scheduler?

Changes LR over training. **ReduceLROnPlateau**: watches a metric; if no improvement for `patience` epochs, multiply LR by `factor` (e.g. 0.5 = halve it). **Warmup**: ramp LR linearly from 0 to target over N steps.

### What is Domain Adaptation?

Training a model to work on a new data distribution (target domain) when you only have labeled data from a different distribution (source domain).

**DANN approach (used here):**
1. Train encoder to produce features that look the same for source and target
2. A discriminator tries to tell source from target
3. GRL reverses discriminator gradients flowing into encoder → encoder learns to fool discriminator
4. Equilibrium: discriminator at 50% accuracy = perfect domain alignment

### GRL Lambda Formula

```
λ(i) = 2*(hi - lo) / (1 + exp(-α * i/N)) - (hi - lo) + lo

With α=1, lo=0, hi=1:
λ = 2/(1 + exp(-i/N)) - 1

Key values:
  i=0:    λ = 0.000  (warm start — no adversarial pressure)
  i=0.1N: λ ≈ 0.050
  i=0.5N: λ ≈ 0.246
  i=N:    λ ≈ 0.462  (NOT 1.0 — sigmoid asymptote)
  i=∞:    λ → 1.000
```

With α=2, lambda would reach ~0.71 at end of training — more aggressive alignment.

### Why Weight Decay Destroys Pre-trained Weights in Fine-tuning

```
Stage 2: 200 epochs × 2800 steps/epoch = 560,000 steps
AdamW:   w ← w × (1 - lr × wd) each step
         w_final = w_init × (1 - 1e-4 × 0.01)^560,000
                 = w_init × 0.999999^560,000
                 = w_init × e^(-0.56)
                 ≈ w_init × 0.57

Stage 1 weights shrink to 57% — destroying learned representations.
Solution: weight_decay=0.0 for pre-trained modules in Stage 2.
```

### What `train_transfer_acc` Tells You

This is the domain discriminator's accuracy at classifying source vs target features.

| Value | Meaning |
|---|---|
| ~50% | Perfect alignment — discriminator can't tell source from target |
| ~100% | No alignment — features completely separable |
| Declining toward 50% | Alignment is working |
| Stuck at 70%+ | Encoder not fooling discriminator — alignment weak |

In our run: 85.9% → 71.3% over 22 epochs. Declining (good sign) but too slowly.

### Confirmation Bias in Entropy Minimization

Entropy minimization (`-p·log(p)`) pushes predictions to be more confident.  
**Problem:** If the model is initially wrong on a target sample, minimizing entropy makes it MORE confidently wrong.  
This is "confirmation bias" — the model gets stuck in wrong confident predictions.  
**Solution:** Only use entropy once alignment has progressed (discriminator near 50%), or use a small coefficient.

---

## Recommended Next Steps

1. **Wait for Stage 1** to converge (accuracy plateaus, LR decays to min)
2. **Take best Stage 1 checkpoint** (highest `per_face_accuracy` on validation)
3. **Start Stage 2** with all fixes applied:
   ```bash
   python domain_adapt.py train \
     --source_path <path_to_CADSynth> \
     --target_path <path_to_MFCAD++> \
     --pre_train results/<stage1_best>.ckpt \
     --max_epochs 200 \
     --batch_size 64 \
     --num_workers 0 \
     --num_classes 25
   ```
4. **Monitor TensorBoard** for:
   - `train_transfer_acc` → should decline toward ~50%
   - `per_face_accuracy_target` → should increase above 77%
   - `current_lr` → should stay constant at 0.0001 (no scheduler decay)
   - `grl_lambda` → should grow slowly from 0 to ~0.46 over 200 epochs
5. **If target accuracy plateaus below 85%**, try:
   - Increasing GRL `alpha` from 1.0 to 2.0 or 3.0
   - Re-enabling entropy at small coefficient: `0.02 * loss_t`
   - Increasing adversarial coefficient β from 0.3 to 0.5
