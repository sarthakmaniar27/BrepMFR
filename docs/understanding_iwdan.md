# Understanding IWDAN (Importance-Weighted DANN) in BrepMFR_PyG

This note explains **what** IWDAN is, **why** we use it in Stage 2, and **how** it is implemented in this repository. It parallels the style of [`post_hoc_logit_adjustment.md`](post_hoc_logit_adjustment.md). Primary code: [`models/transfer_model.py`](../models/transfer_model.py), [`domain_adapt.py`](../domain_adapt.py), [`models/modules/domain_adv/dann.py`](../models/modules/domain_adv/dann.py).

---

## 1. The setting: Stage 2 domain adaptation

Stage 2 trains **`DomainAdapt`**: the same **BrepSeg** backbone (encoder + attention + classifier) as Stage 1, plus a **domain discriminator** and a **gradient reversal layer (GRL)** on features fed to that discriminator.

Each training batch contains **paired** synthetic (source) and real (target) graphs. We optimize:

- **Supervised CE** on **source** face labels (CADSynth).
- **Entropy minimization** on **target** predictions (no target labels in that term).
- **Domain adversarial loss (DANN)**: the discriminator tries to tell source vs target features apart; the encoder is trained (via GRL) to **fool** it so features look domain-invariant.

Vanilla DANN assumes a particular relationship between domains. When **class proportions differ** between source and target (**label shift**), the **marginal** distribution of source features seen by the discriminator is **not** the same as what you would get if target labels followed target priors. Plain DANN can be **mis-specified** and may **hurt** target accuracy (see Zhao et al., ICML 2019).

**IWDAN** (Tachet des Combes et al., NeurIPS 2020) corrects the **domain adversarial** objective by **importance weighting the source side** using class priors \(P_S(y)\), \(P_T(y)\).

---

## 2. What IWDAN does (intuition)

Think of the discriminator as matching two **mixtures** of class-conditional feature distributions. If “Stock” is 60% of source faces but only 20% on target, the **raw** source stream over-represents Stock-like features. IWDAN **down-weights** each source face’s contribution to the discriminator loss in proportion to how **over-** or **under-represented** its true class is on source vs target.

Roughly, each **labeled source** face with class \(y\) gets a weight proportional to

\[
\frac{P_T(y)}{P_S(y)}
\]

so that, **in expectation**, the weighted source empirical distribution aligns with the **target** label distribution—**for the purpose of training the discriminator**, not for changing the classifier’s softmax at inference time.

---

## 3. How we compute the weights (frozen at init)

Implementation: [`_load_priors_json`](../models/transfer_model.py) and [`_compute_iwdan_weights`](../models/transfer_model.py).

### Step A — Load priors from JSON

At model construction, we read two paths from the CLI:

- `--iwdan_source_priors`
- `--iwdan_target_priors`

Each file is **`compute_class_weights.py` format**: a `counts` array of length 25. We convert counts to **normalized priors**:

\[
\hat{P}_S(c) = \frac{\text{count}_S(c)}{\sum_k \text{count}_S(k)}, \quad
\hat{P}_T(c) = \frac{\text{count}_T(c)}{\sum_k \text{count}_T(k)}
\]

(with a small floor to avoid zeros).

Canonical copies in-repo live under [`artifacts/class_weights/stage2_iwdan/`](../artifacts/class_weights/README.md). Those JSONs use the **same schema** as Stage 1 weight files, but **Stage 1 CE** uses the `weights` field; **IWDAN** uses **`counts` → priors** only.

### Step B — Raw importance ratios

\[
\tilde{w}_c = \frac{\hat{P}_T(c)}{\hat{P}_S(c)}
\]

### Step C — Clip (stability)

\[
w_c^{\text{clip}} = \mathrm{clip}\bigl(\tilde{w}_c,\; 1/\texttt{iwdan\_clip},\; \texttt{iwdan\_clip}\bigr)
\]

Default **`--iwdan_clip`** is **10.0** (so ratios lie in \([0.1, 10]\)). This stops a few extremely rare-on-source classes from **dominating** the discriminator gradient.

### Step D — Renormalize so expectation under source is 1

\[
w_c = \frac{w_c^{\text{clip}}}{\sum_k \hat{P}_S(k)\, w_k^{\text{clip}}}
\]

So \(\sum_c \hat{P}_S(c)\, w_c = 1\): **source mass is preserved** after clipping (see comment in `_compute_iwdan_weights`).

The resulting vector `w` is stored once as a buffer:

```text
self.register_buffer("iwdan_weights", torch.from_numpy(w))  # shape [25]
```

---

## 4. Where the weights are applied (training only)

In **`DomainAdapt.training_step`** ([`transfer_model.py`](../models/transfer_model.py)), after building per-node features `z_s`, `z_t` for domain loss:

- **Target** branch BCE weights: **1** on every real target node (unchanged).
- **Source** branch: if `iwdan_enabled`, each source node with label \(y_i\) gets weight **`iwdan_weights[y_i]`**; else **1.0** for all source nodes.

Those weights are passed into **`DomainAdversarialLoss.forward(f_s, f_t, w_s, w_t)`** as `w_s`, `w_t` ([`dann.py`](../models/modules/domain_adv/dann.py)): they scale **`binary_cross_entropy`** on the discriminator outputs for padded batches (padding rows stay at weight 0).

**What is *not* reweighted by IWDAN in our code**

- **Source classification CE** (`CrossEntropyLoss` on `node_seg_s`) — uniform per face.
- **Target entropy loss** — no per-class IWDAN weights in the current implementation.

So IWDAN specifically shapes **\(L_{\text{adv}}\)** (the adversarial alignment term), not the whole `loss = loss_s + 0.3 * loss_adv + 0.1 * loss_t` with the same weights everywhere.

### Validation vs training (important detail)

In **`validation_step`**, the logged **`eval_loss_transfer`** uses **uniform** `weight_s[:num_node_s] = 1.0` (IWDAN weights are **not** applied there). Training **does** use IWDAN on `loss_adv`. So TensorBoard’s train vs val “transfer” loss are **not** the same objective weighting; val is mainly for monitoring and **`eval_loss`** is driven by target accuracy.

---

## 5. GRL schedule (orthogonal but coupled in practice)

The GRL multiplier \(\lambda\) ramps over **`max_iters`**, derived from:

- `--estimated_steps_per_epoch` (default 2444),
- `--max_epochs`,
- `--grl_ramp_frac` (fraction of total steps for the ramp),

unless **`--grl_max_iters`** is set explicitly. See [`DomainAdapt.__init__`](../models/transfer_model.py). This controls **how fast** the discriminator’s reversed gradient turns on; it is **not** part of the IWDAN paper’s ratio formula but strongly affects stability.

---

## 6. Enabling IWDAN from the command line

[`domain_adapt.py`](../domain_adapt.py) flags:

| Flag | Role |
|------|------|
| `--iwdan` | Turn on IWDAN (must supply both JSON paths). |
| `--iwdan_source_priors` | Path to source prior JSON (`counts` → \( \hat{P}_S \)). |
| `--iwdan_target_priors` | Path to target prior JSON (`counts` → \( \hat{P}_T \)). |
| `--iwdan_clip` | Clip bound \(C\); ratios clipped to \([1/C, C]\). Default **10**. |

Example (paths illustrative):

```powershell
python domain_adapt.py train ... --iwdan `
  --iwdan_source_priors artifacts/class_weights/stage2_iwdan/source_train_priors.json `
  --iwdan_target_priors artifacts/class_weights/stage2_iwdan/target_train_priors.json
```

Without `--iwdan`, Stage 2 is **vanilla DANN**-style adversarial weighting (all ones on source for the domain BCE).

---

## 7. IWDAN vs Stage 1 class weights vs post-hoc logit adjustment

| Mechanism | **When** | **What it changes** | **Uses labels on target?** |
|-----------|----------|----------------------|-----------------------------|
| **Stage 1 CE weights** (`--class_weights_path`) | Stage 1 training | Per-class **loss** weights in cross-entropy on **source** | No |
| **IWDAN** (`--iwdan`, prior JSONs) | Stage 2 training | Per-source-node **BCE weights** on **domain discriminator** only | No (priors from JSON counts, not from adapting at runtime) |
| **Post-hoc logit adjustment** | After any training | **Inference-time** rescoring of softmax using \(\log \hat{P}_T - \log \hat{P}_S\) and \(\tau\) | Diagnostic scripts use empirical target counts (oracle) |

They can **all** use similar-looking JSON files; the **role** of `counts` / `weights` differs. See [`artifacts/class_weights/README.md`](../artifacts/class_weights/README.md).

---

## 8. One-line summary

**IWDAN** in this repo: at Stage 2 **training**, each **labeled source** face’s contribution to the **domain adversarial (DANN) loss** is multiplied by a **fixed per-class weight** derived from \(\hat{P}_T(y)/\hat{P}_S(y)\) (clipped and renormalized), so the discriminator is trained as if source and target shared compatible **label marginals**—mitigating **label shift** in the adversarial term.

---

## 9. References (for your paper’s bibliography)

1. **Ganin et al.** — Domain-Adversarial Training of Neural Networks (DANN / gradient reversal).
2. **Zhao et al. (ICML 2019)** — Impossibility / negative results for DANN under certain **label shift** conditions; motivates explicit corrections.
3. **Tachet des Combes et al. (NeurIPS 2020)** — **Importance-Weighted DANN** (IWDAN): the theoretical basis for reweighting the source side of the domain classifier under label shift.

Add full citation metadata when you freeze the manuscript.
