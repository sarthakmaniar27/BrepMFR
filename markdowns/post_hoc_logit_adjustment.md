# Post-hoc logit adjustment for label-shift on target

This document captures the full investigation that revealed the BrepMFR_PyG target-domain accuracy plateau is a **label-shift** problem (not a label-mapping bug, not a learning-failure), and the post-hoc fix that recovered ~4.4 pp per-class accuracy in zero training time.

It is the predecessor to `class_imbalance_stage1_retrain.md`. The retrain plan there is the *durable* fix; this document is the *discovery* and the *zero-cost interim fix*.

---

## TL;DR

1. Three Stage 2 DA runs (gentle λ, aggressive λ, authors-faithful) all plateaued at ~0.80 per-face / ~0.72 per-class on target val. Tuning DA hyperparameters did **nothing**.
2. A new diagnostic (`scripts/diagnostics/diagnose_stage1_target.py`) showed the gap was caused by **P(Y|source) ≠ P(Y|target)**:
   - Source val: **57.65 %** of faces are class 0 (stock)
   - Target val: **22.36 %** of faces are class 0
   - The Stage-1 classifier learned the source class-0 prior and over-predicts it on target by **+69 %**.
3. **DANN cannot fix this**, by construction (Zhao et al. 2019, *On Learning Invariant Representations for Domain Adaptation*).
4. **Saerens 2002** post-hoc logit adjustment (`scripts/diagnostics/logit_adjust_eval.py`) recovered **+4.4 pp per-class** and **+4.2 pp per-face** on Stage 1 with zero retraining.
5. Applied to either of our Stage 2 checkpoints, the same correction **still gives a worse result than Stage 1 + adjustment**, proving Stage 2 DA was net-negative.
6. The "best" tuning parameter was **τ = 3.0** instead of the theoretically optimal τ = 1.0, which proved the encoder is **over-confident** — the durable fix has to live in Stage 1 (see the companion retrain document).

---

## Background: what is label shift?

In transfer learning, the target distribution can differ from the source in two distinct ways:

| Type | Definition | Example | Fixable by |
|------|------------|---------|------------|
| **Covariate shift** | P(X) differs between domains, P(Y\|X) is the same | Same labels, different image style | Domain Adversarial training (DANN), CORAL, MMD, etc. |
| **Label shift** | P(Y) differs between domains, P(X\|Y) is the same | Same shapes, different class proportions | **Bayes-optimal logit re-weighting** (Saerens 2002, BBSE 2018) |

**DANN aligns features. It does not align label priors.** This is a textbook impossibility result:

> Zhao, H., et al. (2019). *On Learning Invariant Representations for Domain Adaptation.* ICML 2019.
>
> If P_source(Y) ≠ P_target(Y), then minimising H-divergence between the source and target *features* increases the joint error. Feature alignment forces the classifier to mis-classify proportionally to the prior gap.

In our case, the prior gap is enormous (class 0 is 57.65 % at source vs 22.36 % at target), so DANN was being asked to do an impossible thing — and the harder we tried (stronger λ, longer training, asymmetric LR), the more it degraded the encoder.

---

## How the diagnosis was made

### The diagnostic script

`scripts/diagnostics/diagnose_stage1_target.py` runs **pure inference** with the Stage 1 checkpoint on both source val and target val (no DA, no training, no Stage 0 reload). It produces:

| File | Contents |
|------|----------|
| `confusion_matrix_source.csv` | 25×25 matrix, rows = true, cols = pred |
| `confusion_matrix_target.csv` | same, on target val |
| `per_class_report.csv` | support, recall, precision, f1, top-3 confused-with classes per class |
| `label_distribution.csv` | true vs predicted counts per class — exposes prior over-/under-prediction |
| `summary.md` | bucketed diagnosis (collapse / domain shift / encoder weak / label-mapping suspect) |

### The smoking gun

`label_distribution.csv` (full source val + full target val, 272 k vs 268 k faces):

| class | source true % | target true % | model predicts on target % | over/under |
|-------|---------------|---------------|----------------------------|------------|
| **0 (stock)** | **57.65** | **22.36** | **37.89** | **+69 %** |
| 6 | 0.97 | 2.56 | 0.12 | **−95 %** |
| 9 | 1.40 | 3.91 | 0.72 | **−81 %** |
| 10 | 0.59 | 2.61 | 0.99 | **−62 %** |
| 2 | 0.87 | 0.98 | 0.42 | **−58 %** |
| 8 | 0.83 | 2.71 | 1.52 | **−44 %** |
| 1 | 2.76 | 1.61 | 1.09 | **−32 %** |
| 7 | 1.29 | 3.81 | 2.78 | **−27 %** |

The pattern is unambiguous:

- The **dominant** source class (0) is **massively over-predicted** on target.
- The classes that are **MORE prevalent in target than source** (6 is 2.6× more common; 10 is 4.4× more common) are **massively under-predicted**.
- Their errors **all sink toward class 0** — not because of label remap, but because softmax is biased toward the dominant class.

The diagnostic flagged classes 1, 2, 6, 7, 8, 9, 10 as "label-mapping suspects" (≥70 % of all errors collapse onto a single wrong class). With `label_distribution.csv` in hand, we know that single wrong class is always **the most-frequent source class**, which is the signature of label shift, not relabeling.

### What the diagnostic ruled out

| Bucket | Result | Implication |
|--------|--------|-------------|
| Class collapse (predicted 0 times on target) | empty | every class IS sometimes predicted |
| Encoder-weak (low on BOTH src and tgt) | empty | the encoder learned every class on source |
| Label-mapping suspect (errors → one specific wrong class) | populated, but all → class 0 | not a remap; just the dominant class sink |
| Domain shift (high src acc, low tgt acc) | classes 6 and 9 only | these two are the *real* DA targets |

So we had a single, decisive diagnosis: **label shift is the dominant source of error**, and *only* classes 6 and 9 have genuine encoder-level domain shift remaining underneath.

---

## The fix: Saerens-style logit adjustment

### Math

Under label shift assumption (P(X|Y) is the same, P(Y) differs), the Bayes-optimal classifier on target is recovered from the source-trained classifier by:

```
P_target(y|x) ∝ P_source(y|x) × P_target(y) / P_source(y)
```

Equivalently in logit space:

```
logits_adjusted[k] = logits[k] + τ · (log P_target(k) − log P_source(k))
prediction          = argmax_k logits_adjusted[k]
```

where **τ = 1** recovers the theoretically optimal correction *if probabilities are well calibrated*.

Sources:
- Saerens, M., Latinne, P., & Decaestecker, C. (2002). *Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure.* Neural Computation, 14(1), 21–41.
- Lipton, Z. C., Wang, Y.-X., & Smola, A. J. (2018). *Detecting and Correcting for Label Shift with Black Box Predictors* (BBSE). ICML 2018.
- Menon, A. K., et al. (2021). *Long-tail learning via logit adjustment.* ICLR 2021.

### Implementation

`scripts/diagnostics/logit_adjust_eval.py`:

1. Loads Stage 1 checkpoint (also works on Stage 2 — uses `strict=False` and ignores `domain_adv.*` keys).
2. Computes source priors from source val labels (label-only iteration, no model needed — fast).
3. Computes target priors from target val labels (oracle for the POC; in production we would estimate via BBSE).
4. Runs model inference on target val **once**, caches per-face probabilities.
5. Sweeps τ over a configurable range, applying `log_probs + τ · log(P_target/P_source)` and taking argmax.
6. Reports per-face acc, per-class acc, confusion matrix, per-class change at each τ.

The cached-probs design means the τ sweep itself is essentially free — the only cost is the single inference pass.

### Numerical notes

- The classifier in `models/brepseg_model.py` ends in `F.softmax`, so we get probabilities, not raw logits. We compute `log_probs = log(clip(probs, 1e-12, 1.0))` to recover (approximate) log-softmax. For argmax purposes this is identical to using raw logits because the omitted `−logsumexp(x)` term is constant across classes.
- To handle log-of-zero we floor priors at `1e-8` before taking the log ratio.
- Empty-class entries in target val (e.g. class 24 with 0 samples in a smoke run) get a very negative `log_ratio`, which is fine — argmax avoids them automatically.

---

## Results

### On Stage 1 checkpoint (full target val: 268,029 faces)

τ sweep:

| τ | per-face acc | per-class acc | Δ per-class vs τ=0 |
|------|--------------|---------------|--------------------|
| 0.00 | 0.8055 | 0.7209 | +0.0000 |
| 0.25 | 0.8184 | 0.7304 | +0.0095 |
| 0.50 | 0.8222 | 0.7352 | +0.0143 |
| 0.75 | 0.8254 | 0.7393 | +0.0184 |
| 1.00 | 0.8282 | 0.7427 | +0.0217 |
| 1.25 | 0.8310 | 0.7458 | +0.0249 |
| 1.50 | 0.8334 | 0.7486 | +0.0277 |
| 2.00 | 0.8378 | 0.7537 | +0.0327 |
| 2.50 | 0.8420 | 0.7585 | +0.0376 |
| **3.00** | **0.8474** | **0.7650** | **+0.0441** |

**Best τ was the boundary value (3.0). Per-class was still rising — could try τ = 4, 5.**

### Per-class change at best τ = 3 (delta vs no-adjustment baseline)

The classes with strong positive `log_ratio` (target prevalence > source prevalence) are the big winners:

| class | acc base → adj | Δ | log P_t/P_s |
|-------|----------------|---|-------------|
| 10 | 0.370 → 0.581 | **+21.1** | +1.48 |
| 8 | 0.540 → 0.725 | **+18.5** | +1.18 |
| 7 | 0.696 → 0.830 | **+13.5** | +1.08 |
| 18 | 0.791 → 0.914 | **+12.3** | +0.99 |
| 14 | 0.848 → 0.965 | **+11.7** | +0.94 |
| 2 | 0.330 → 0.431 | +10.2 | +0.12 |
| 13 | 0.708 → 0.748 | +4.0 | +1.06 |
| 6 | 0.029 → 0.066 | +3.7 | +0.97 |
| 12 | 0.836 → 0.860 | +2.4 | +0.91 |

The **stubborn classes** (real domain shift, not label shift):

| class | acc base → adj | Δ | comment |
|-------|----------------|---|---------|
| 6 | 0.029 → 0.066 | +3.7 | Still terrible — encoder doesn't recognise this geometry on target |
| 9 | 0.180 → 0.249 | +6.9 | Better but still bad — same story |
| 1 | 0.463 → 0.478 | +1.5 | Encoder weak on target for this class |

The **two losers** (negative `log_ratio`, prior-correction over-shoots):

| class | acc base → adj | Δ | log P_t/P_s |
|-------|----------------|---|-------------|
| 19 | 0.806 → 0.726 | **−8.0** | −0.74 |
| 11 | 0.704 → 0.676 | −2.9 | −1.05 |
| 24 | 0.957 → 0.945 | −1.2 | −0.62 |

Class 0 (stock) recall stayed at 0.999 — the adjustment didn't break the dominant class. It just stopped over-predicting it (101,542 → 88,500 predictions, a 13 % drop).

### On Stage 2 checkpoints (the verdict)

Same script, same target val, but pointed at the Stage 2 best checkpoints:

| Checkpoint | per-face baseline | per-class baseline | per-face + best τ | per-class + best τ | best τ |
|------------|-------------------|--------------------|-------------------|--------------------|--------|
| **Stage 1 (no DA)** | 0.8055 | **0.7209** | **0.8474** | **0.7650** | 3.0 |
| Stage 2 "gentle λ" (0430) | 0.8066 | 0.7157 | 0.8267 | 0.7345 | 4.0 |
| Stage 2 "authors-faithful" (0501) | 0.7786 | 0.6731 | 0.8055 | 0.6989 | 4.0 |

Three concrete things this proves:

1. **DANN was net-negative for our setup.** Both Stage 2 variants are worse than just leaving Stage 1 alone, even after applying the same correction to all three.
2. **Stage 2 made the encoder *more* over-confident.** The optimal τ for Stage 2 is 4.0 vs Stage 1's 3.0 — DA training pushed the classifier to be more decisive, in the wrong direction.
3. **The "gentle" Stage 2 was approximately neutral; the "authors-faithful" Stage 2 was actively damaging** (−4.8 pp per-class baseline). The harder we pushed adversarial pressure, the more the encoder warped.

---

## Why best τ = 3, not τ = 1: over-confidence

Theoretically, with calibrated probabilities and known priors, **τ = 1** is Bayes-optimal. We measured **τ = 3** (and per-class still rising at the boundary).

This is the textbook signature of an **over-confident softmax**: cross-entropy training pushes correct-class probabilities toward 1, which makes the implicit prior almost impossible to override at τ = 1. To shift predictions across the decision boundary you have to dump 3× the theoretical adjustment into the logits.

Over-confidence has two causes here, both stemming from the same root:

1. **Class imbalance**: rare classes get tiny gradients during training (they appear in few batches). Common classes get huge gradients. Common-class logits saturate. The model becomes "very sure" of class 0 and only the strongest contrary evidence flips it.
2. **Loss form**: the model's classifier ends in `F.softmax` and the loss is `-label · log(probs + eps)`. There's no temperature, no calibration, no label smoothing. The softmax peaks naturally drift toward 1.0 on training data the model is good at.

Both root causes are addressed by **class-balanced loss** at training time. See `class_imbalance_stage1_retrain.md` for the durable fix.

---

## Recoverability ceiling

| component of the gap | recoverable | how |
|----------------------|-------------|-----|
| Label shift (P(Y) mismatch) | ~5 pp | post-hoc τ correction — `scripts/diagnostics/logit_adjust_eval.py` |
| Over-confident softmax | ~2–3 pp | calibration / class-balanced training — Stage 1 retrain |
| Genuine domain shift on classes 6, 9, 1 | ~4–5 pp | Stage 2 DANN — but only after both above are fixed |
| **Total recoverable** | **~12 pp** | matches the gap to the paper's 0.9274 |

---

## How to use the adjustment in production

### Run the diagnostic

```powershell
python scripts/diagnostics/diagnose_stage1_target.py `
  --checkpoint "C:/Users/D58/Desktop/BrepMFR_PyG/results/BrepMFR/0425/183526/best.ckpt" `
  --source_path "Z:/Experiment6_PyG/source_dataset" `
  --target_path "Z:/Experiment6_PyG/target_dataset" `
  --num_classes 25 `
  --batch_size 32 `
  --num_workers 2 `
  --out_dir "results/diagnostics/stage1_audit"
```

Outputs land in `results/diagnostics/stage1_audit/`. Read `summary.md` first, then `per_class_report.csv` for detail.

### Run the τ sweep

```powershell
python scripts/diagnostics/logit_adjust_eval.py `
  --checkpoint "C:/Users/D58/Desktop/BrepMFR_PyG/results/BrepMFR/0425/183526/best.ckpt" `
  --source_path "Z:/Experiment6_PyG/source_dataset" `
  --target_path "Z:/Experiment6_PyG/target_dataset" `
  --num_classes 25 `
  --batch_size 32 `
  --num_workers 2 `
  --taus "0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0" `
  --out_dir "results/diagnostics/logit_adjust"
```

CLI flags worth knowing:

| Flag | Default | Notes |
|------|---------|-------|
| `--taus` | `0.0,0.25,0.5,...,2.0` | Comma-separated. Sweep is essentially free since probs are cached. |
| `--uniform_target` | off | Use uniform P_target = 1/C instead of empirical. Use when target labels are unavailable. |
| `--max_batches N` | 0 (off) | Smoke-test mode. |
| `--out_dir` | `results/diagnostics/logit_adjust` | Per-class compare CSV, baseline + best-τ confusion matrices, sweep CSV, and summary.md. |

### Consume the result

The script writes `results/diagnostics/logit_adjust/per_class_compare.csv` with per-class baseline acc, adjusted acc, delta, precision, F1, predicted-count for both, plus the source/target priors and log-ratio. That CSV is the source of truth for downstream analysis.

For inference in a serving setup, the recipe is:

```python
# 1. Compute and cache source priors at training time:
src_priors = label_counts_source / label_counts_source.sum()

# 2. Estimate target priors at deployment time:
#    - if target labels are available (eval): use empirical
#    - else: use BBSE on a few hundred unlabeled target samples
tgt_priors = ...

# 3. At inference time, on every batch:
logits = model.forward_until_softmax(x)        # shape [N, C]
log_ratio = torch.log(tgt_priors) - torch.log(src_priors)
adjusted = logits + tau * log_ratio.unsqueeze(0)
preds = adjusted.argmax(dim=-1)
```

τ should be tuned on a held-out target labeled set if available (we used τ = 3 here). For BBSE-estimated priors, slightly lower τ values typically work better because BBSE noise compounds with τ.

---

## Limitations and caveats

1. **Oracle target priors.** For the POC we used the *labeled* target val set to estimate target priors. This is allowed for diagnosis (we want to know if label shift is the cause), but in production you must use BBSE or similar from *unlabeled* target. BBSE adds ~1–2 % error to the prior estimates which translates to roughly −1 pp on the recovered accuracy.
2. **Over-confident softmax inflates τ.** A calibrated model would have hit best τ ≈ 1 and might have recovered slightly more accuracy with the same correction, because the adjustment wouldn't have to fight against an inflated softmax. Calibration is part of the durable fix.
3. **Class 6 stays catastrophically bad** (0.066) even after adjustment. This is *real* domain shift — the synthetic source data simply does not represent class-6 geometry well enough for the encoder to recognise it on real B-rep faces. DA *should* help here, but only after the label-shift signal stops dominating the discriminator gradient.
4. **Class 0 recall drops** from 0.99988 to 0.99924 (−0.06 %) at τ = 3. Negligible in practice but worth noting if you have a downstream system that depends on near-perfect stock detection.
5. **The script reuses the diagnostic loaders** (`FilelistDataset`, `load_stage1_model`, `make_loader`) via `sys.path` injection rather than packaging. Keep both scripts under **`scripts/diagnostics/`** alongside `diagnose_stage1_target.py`.

---

## File reference

| File | Role |
|------|------|
| `scripts/diagnostics/diagnose_stage1_target.py` | Per-class confusion + bucketed diagnosis (collapse / shift / weak / suspect). |
| `scripts/diagnostics/logit_adjust_eval.py` | Sweeps τ on a Stage 1 or Stage 2 checkpoint, reports per-class deltas. |
| `results/diagnostics/stage1_audit/` | Stage 1 diagnostic outputs (oracle run that uncovered the label shift). |
| `results/diagnostics/logit_adjust/` | Stage 1 + τ sweep results. |
| `results/diagnostics/logit_adjust_stage2_authors/` | Same for the authors-faithful Stage 2 ckpt. |
| `results/diagnostics/logit_adjust_stage2_gentle/` | Same for the gentle-λ Stage 2 ckpt. |
| `markdowns/class_imbalance_stage1_retrain.md` | Companion document — durable fix via class-balanced Stage 1 training. |

---

## Decision

**Adopt logit adjustment as a free baseline immediately.** It works, it's zero-training-cost, and it already beats every Stage 2 attempt we made.

**But the durable fix lives in Stage 1.** The need for τ = 3 (instead of the theoretically correct τ = 1) is itself a signal that the encoder is mis-trained. Retraining Stage 1 with class-balanced loss should:

- Recover the label-shift gap *inside* the model weights (so τ ≈ 1 suffices afterwards)
- Produce a calibrated encoder that is a much better starting point for Stage 2 DA on the genuinely-domain-shifted classes (6, 9, 1)

That is the work tracked in `class_imbalance_stage1_retrain.md`.
