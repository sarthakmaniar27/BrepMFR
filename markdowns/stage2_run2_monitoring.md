# Stage 2 Training — Run 2 Monitoring Log

**Date:** 2026-04-17  
**Run:** Restarted after diagnosing Run 1 failure (see `stage1_stage2_debugging_session.md`)  
**Fixes applied before this run:**
- `weight_decay=0.0` for encoder/attention/classifier
- ReduceLROnPlateau **removed** (fixed LR throughout)
- Entropy loss **disabled** (`loss = loss_s + 0.3 * loss_adv`)
- GRL `estimated_steps_per_epoch` dynamic from `args.batch_size`
- `np.float` → `float()` in grl.py

---

## Snapshot at Epoch 10/200

### Full Metric Table

| ep | target_acc | source_acc | train_loss | loss_s  | loss_t  | loss_adv | disc_acc% | grl_lambda | lr         |
|----|-----------|-----------|-----------|---------|---------|----------|-----------|------------|------------|
|  1 | 77.34%    | 99.52%    | 0.09466   | 0.01077 | 0.05937 | 0.27964  | 87.54     | 0.00112    | 0.0000250  |
|  2 | 76.64%    | 99.51%    | 0.07320   | 0.01092 | 0.05877 | 0.20759  | 91.17     | 0.00334    | 0.0000499  |
|  3 | 77.11%    | 99.49%    | 0.07584   | 0.01113 | 0.05904 | 0.21571  | 90.68     | 0.00557    | 0.0000749  |
|  4 | 77.71%    | 99.50%    | 0.08473   | 0.01153 | 0.05751 | 0.24403  | 89.26     | 0.00780    | 0.0000998  |
|  5 | 77.57%    | 99.49%    | 0.09416   | 0.01185 | 0.05781 | 0.27438  | 87.59     | 0.01003    | 0.0001000  |
|  6 | **78.16%**| 99.49%    | 0.10125   | 0.01188 | 0.05656 | 0.29790  | 86.33     | 0.01226    | 0.0001000  |
|  7 | 77.65%    | 99.51%    | 0.10659   | 0.01187 | 0.05723 | 0.31574  | 85.31     | 0.01449    | 0.0001000  |
|  8 | 76.14%    | 99.51%    | 0.11357   | 0.01159 | 0.05507 | 0.33993  | 83.93     | 0.01671    | 0.0001000  |
|  9 | 76.95%    | 99.48%    | 0.12261   | 0.01183 | 0.05367 | 0.36927  | 82.14     | 0.01894    | 0.0001000  |
| 10 | 77.18%    | 99.51%    | 0.13360   | 0.01176 | 0.05193 | 0.40613  | 79.74     | 0.02117    | 0.0001000  |

> **Best target accuracy so far:** 78.16% at epoch 6

---

## Signal-by-Signal Interpretation

### 1. Learning Rate — Fix Confirmed Working

```
Epochs 1–4:  warmup  0.000025 → 0.0001  (over 5000 steps, ~4 epochs)
Epoch 5+:    0.0001  STABLE — ReduceLROnPlateau removed, no decay
```

In Run 1, LR had already halved to 0.00005 by epoch 8 and was at 0.000025 by epoch 17.
This run stays at 0.0001 throughout. **Fix confirmed.**

### 2. Discriminator Accuracy — Declining (Alignment Working)

```
Epoch  1:  87.5%   (discriminator has easy time separating domains)
Epoch  2:  91.2%   (brief spike as disc learns quickly while lambda~0)
Epoch  5:  87.6%   (starts falling)
Epoch  9:  82.1%
Epoch 10:  79.7%   ← fastest decline yet — encoder gaining momentum
Target:    ~50%    (perfect alignment = discriminator at random chance)
```

The discriminator is progressively losing its ability to separate source from target features.
This is exactly what DANN alignment looks like. Declining from 91% → 80% in 10 epochs is a healthy trend.

### 3. Target Accuracy — Oscillating, Not Collapsing

```
Epoch  1:  77.34%
Epoch  6:  78.16%  ← best (above source-only baseline of ~77%)
Epoch  8:  76.14%  ← temporary dip
Epoch 10:  77.18%  ← recovering
```

Not declining monotonically like Run 1 (which went 77% → 74% without recovery).
The oscillation at epoch 9–10 is normal: lambda is only 2.1%, adversarial pressure is minimal.
Real improvement expected in epochs 30–100 as lambda grows past 0.05–0.20.

### 4. Entropy Loss — Logged, Not Trained

```
train_loss_t:  0.059 (ep1) → 0.052 (ep10)  [gentle decline, no collapse]
```

Disabled from training objective but still logged for monitoring.
In Run 1, this collapsed to 0.006 by epoch 22 while accuracy fell — the confirmation bias.
This run shows only a gentle natural decline — no forced sharpening of wrong predictions.

### 5. GRL Lambda — Tiny But Growing

```
Epoch  1:   0.00112  (0.1% strength)
Epoch 10:   0.02117  (2.1% strength)
```

Lambda grows ~0.002 per epoch. Projected milestone epochs:
```
Epoch  25:  lambda ≈ 0.050  (5% — alignment signal starts mattering)
Epoch  50:  lambda ≈ 0.099  (10%)
Epoch 100:  lambda ≈ 0.197  (20%)
Epoch 200:  lambda ≈ 0.390  (39% — max at end of training with alpha=1)
```

Real accuracy improvement expected as lambda crosses 0.05–0.10 (around epoch 25–50).

### 6. Adversarial Loss (loss_adv) — Rising Correctly

```
Epoch  1:  0.280
Epoch 10:  0.406
```

Rising because as the encoder improves at confusing the discriminator, the discriminator
finds it harder and its BCE loss increases. This is the correct DANN dynamic.
A rising adversarial loss paired with a falling disc_acc is a healthy sign.

### 7. Source CE — Stable (Stage 1 Knowledge Preserved)

```
train_loss_s:  0.011 → 0.012  (essentially flat)
source_acc:    99.52% → 99.51%  (no degradation)
```

Weight decay fix (wd=0.0 on pre-trained modules) is preserving Stage 1 knowledge.

---

## Comparison: Run 1 (failed) vs Run 2 (this run) at Epoch 9

| Metric              | Run 1 (ep 9)          | Run 2 (ep 9)          | Better? |
|---------------------|-----------------------|-----------------------|---------|
| Target accuracy     | 76.09%                | **76.95%**            | ✓       |
| LR (encoder)        | 0.00005 *(halved)*    | **0.0001 (stable)**   | ✓       |
| Discriminator acc   | 84.6%                 | **82.1%**             | ✓       |
| Entropy loss_t      | 0.054 *(collapsing)*  | 0.054 *(stable)*      | ✓       |
| Scheduler fires?    | Yes (at ep 7)         | **No**                | ✓       |

---

## What to Watch Next

| Checkpoint epoch | Expected state | Action |
|---|---|---|
| Epoch 25 | lambda ≈ 0.05, disc_acc ≈ 65–70%, target_acc ≈ 78–80% | Continue if improving |
| Epoch 50 | lambda ≈ 0.10, disc_acc ≈ 55–65%, target_acc ≈ 80–84% | Consider re-enabling entropy at 0.02 |
| Epoch 100 | lambda ≈ 0.20, disc_acc ≈ 50–55%, target_acc ≈ 85–88%? | Evaluate vs paper's 90.32% |
| Epoch 200 | lambda ≈ 0.39, target_acc = final | Final benchmark |

### Warning Signs to Watch For

| Signal | Concern | Action |
|---|---|---|
| `disc_acc` stops falling (plateaus > 70%) | Alignment stalled | Increase GRL alpha from 1.0 to 2.0 |
| `target_acc` starts monotonically falling | New instability | Stop and diagnose |
| `train_loss_t` collapses to < 0.01 | Entropy re-enabled accidentally, or collapse | Check loss formula |
| `current_lr` drops below 0.0001 | Scheduler re-introduced somehow | Check configure_optimizers |

### If accuracy plateaus below 85% after epoch 100

Try these in order:
```python
# Option 1: Stronger GRL
grl = WarmStartGradientReverseLayer(alpha=2., lo=0., hi=1., ...)

# Option 2: Re-enable entropy at small coefficient
loss = loss_s + 0.3 * loss_adv + 0.02 * loss_t

# Option 3: Stronger adversarial coefficient
loss = loss_s + 0.5 * loss_adv
```

---

## Training Command Reference

```bash
python domain_adapt.py train \
  --source_path <path_to_CADSynth> \
  --target_path <path_to_MFCAD++> \
  --pre_train results/<stage1_best>.ckpt \
  --max_epochs 200 \
  --batch_size 64 \
  --num_workers 0 \
  --num_classes 25 \
  --experiment_name BrepToSeq-segmentation
```

---

*Last updated: epoch 10/200*
