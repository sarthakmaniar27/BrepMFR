# Post-hoc logit adjustment (Stage 1 and Stage 2)

This note explains **what** post-hoc logit adjustment is, **why** we use it in BrepMFR_PyG, and **how** our Stage 1 and Stage 2 diagnostic scripts implement it. The math matches [`scripts/diagnostics/logit_adjust_eval.py`](../scripts/diagnostics/logit_adjust_eval.py) and [`scripts/diagnostics/stage2_logit_adjust_eval.py`](../scripts/diagnostics/stage2_logit_adjust_eval.py).

---

## 1. The problem: label shift

Stage 1 trains a face classifier on **CADSynth** (synthetic source). Class frequencies there are not the same as on **MFCAD++** (real target). For example, “Stock” (class 0) can dominate the source split while being much rarer on the target.

A softmax classifier trained on source tends to **internalize the source class prior** \(P_S(y)\): it becomes optimistic about frequent source classes even when the true target prior \(P_T(y)\) differs. That mismatch is **label shift** (shift in \(P(y)\) while \(P(x \mid y)\) is often assumed stable enough to reason about). Plain domain adaptation (DANN / IWDAN) targets **covariate shift** in the representation; it does **not** automatically fix arbitrary label-prior mismatch (see e.g. Zhao et al., ICML 2019, on limitations of DANN under label shift).

**Post-hoc logit adjustment** is a **decision-rule change at inference time**: we keep the same trained network and **re-score** its outputs using estimated class priors on source and target, without backprop or retraining.

---

## 2. What “post-hoc” means here

- **Post-hoc** = after training. The checkpoint is frozen.
- We run **one** forward pass over the target split, cache **per-face class probabilities** \(p \in \mathbb{R}^{C}\) (here \(C = 25\)).
- We then **sweep a scalar** \(\tau\) and pick predictions that maximize an adjusted score. No extra GPU training loops.

---

## 3. How the adjustment works (mathematics)

Let \(p_{ik}\) be the model’s probability for face \(i\) and class \(k\). In code we work in **log-probability space** for stability:

\[
\tilde{z}_{ik} = \log p_{ik} + \tau \cdot \bigl(\log \hat{P}_T(k) - \log \hat{P}_S(k)\bigr)
\]

The predicted class is \(\arg\max_k \tilde{z}_{ik}\).

Equivalently (up to normalization), this behaves like **reweighting** odds by \(\bigl(\hat{P}_T(k) / \hat{P}_S(k)\bigr)^{\tau}\). Intuition:

- \(\tau = 0\): no change — raw argmax of the model’s probabilities.
- \(\tau = 1\) with **correct** \(\hat{P}_S, \hat{P}_T\): standard “prior correction” / **Bayes rule under label shift** (see Saerens et al., 2002; Menon et al. discuss logit adjustment in the long-tail setting).
- \(\tau \in (0, 1)\) or \(\tau > 1\): a **partial or stronger** correction. Our scripts sweep a grid of \(\tau\) values because plug-in priors are imperfect and a softer correction sometimes generalizes better on finite data.

Implementation detail: [`adjusted_predictions`](../scripts/diagnostics/logit_adjust_eval.py) computes `log_ratio[k] = log P̂_T(k) - log P̂_S(k)` once, then for each \(\tau\):

```text
adjusted_logits[i, :] = log(probs[i, :]) + tau * log_ratio[:]
pred[i] = argmax(adjusted_logits[i, :])
```

---

## 4. Where \(\hat{P}_S\) and \(\hat{P}_T\) come from

We estimate priors from **empirical label counts** on chosen filelists:

\[
\hat{P}(k) = \frac{\text{count}(y = k)}{\sum_j \text{count}(y = j)}
\]

with a small floor to avoid \(\log 0\) (`priors_from_counts` in `logit_adjust_eval.py`).

### Stage 1 script (`logit_adjust_eval.py`)

- Default: **`s_val.txt`** on the source root → \(\hat{P}_S\).
- Default: **`t_val.txt`** on the target root → \(\hat{P}_T\).

You can change filelists via CLI flags (`--source_filelist`, `--target_filelist`). There is also `--uniform_target` for a Menon-style uniform target prior (ablation).

**Caveat (honest evaluation):** Using **target val/test labels** to set \(\hat{P}_T\) is **oracle** for analysis and paper-style diagnostics. In deployment you would estimate target priors without labels (e.g. Black Box Shift Estimation or other methods). The scripts document this trade-off in their module docstrings.

### Stage 2 script (`stage2_logit_adjust_eval.py`)

- **Source prior counts:** default **`s_val.txt`** on `source_path` (`--source_prior_filelist`).
- **Target prior counts:** the **same split** as inference — **`t_val.txt`** or **`t_test.txt`** according to `--target_split` (so priors match the evaluated distribution).

Stage 2 inference uses [`TransferDataset`](../data/dataset.py) (paired source/target batches); the script collects **target-side** face probabilities and labels only.

---

## 5. Stage 1 vs Stage 2 in this repository

| Aspect | Stage 1 (`logit_adjust_eval.py`) | Stage 2 (`stage2_logit_adjust_eval.py`) |
|--------|----------------------------------|------------------------------------------|
| **Model** | `BrepSeg` checkpoint | `DomainAdapt` checkpoint |
| **Inference** | Source-style `FilelistDataset` on target filelist | `TransferDataset` test/val split; target branch of encoder + classifier |
| **Cached tensor** | Per-face softmax on target | Per-face softmax on target |
| **Adjustment** | Same `log_ratio` + `adjusted_predictions` | Same (imported from `logit_adjust_eval`) |
| **Typical use** | “Source-only” model on MFCAD++ with prior correction | After DA: same correction on top of adapted logits |

**Important:** IWDAN during Stage 2 training already uses **importance weights** derived from prior JSONs in the discriminator loss. Post-hoc logit adjustment is **another**, **inference-time** knob on the **classifier outputs**; it is not redundant theory-wise, but you should interpret both together when writing ablations.

---

## 6. Purpose in our workflow

1. **Diagnostics:** See how much of the remaining target error is consistent with **prior mismatch** vs other errors.
2. **Reporting:** Report **τ = 0** (raw checkpoint) vs **best τ** on a grid for mean per-class recall (or another KPI) alongside Table 3.
3. **Cheap sweep:** One inference pass + many \(\tau\) values is fast compared to retraining.

Outputs are written under `--out_dir`: `summary.md`, `tau_sweep.csv`, confusion matrices, per-class CSVs.

---

## 7. How to run (reminder)

**Stage 1**

```powershell
conda activate brep_mfr_pyg
python scripts/diagnostics/logit_adjust_eval.py `
  --checkpoint results/stage1/<run>/best.ckpt `
  --source_path Z:/Experiment6_PyG/source_dataset `
  --target_path Z:/Experiment6_PyG/target_dataset `
  --out_dir results/diagnostics/logit_adjust_<name>
```

**Stage 2**

```powershell
python scripts/diagnostics/stage2_logit_adjust_eval.py `
  --checkpoint results/stage2/<run>/best.ckpt `
  --source_path Z:/Experiment6_PyG/source_dataset `
  --target_path Z:/Experiment6_PyG/target_dataset `
  --target_split test `
  --out_dir results/diagnostics/stage2_logit_adjust_<name>
```

If your graphs live under a subgraph directory (e.g. `output/bin_skip_a2`), pass the same **`--pt_subdir`** as for `paper_table3_eval.py` so counts and inference align with that experiment.

---

## 8. References (reading order)

1. **Saerens et al.** — Adjusting posterior probabilities when training and application priors differ (EM framework for prior correction).
2. **Menon et al.** — Long-tail classification and logit adjustment (related rebalancing of logits at train or test time).
3. **Zhao et al. (ICML 2019)** — On why DANN does not address arbitrary label shift; motivates explicit prior / importance handling (connected to our IWDAN + post-hoc layers).

For exact citation strings and paper links, add them to your manuscript bibliography when you freeze the text.

---

## 9. One-line summary

**Post-hoc logit adjustment** re-scores each face’s class probabilities using \(\log \hat{P}_T(y) - \log \hat{P}_S(y)\), scaled by \(\tau\), to reduce bias from **different class priors** on source vs target—**after** the model is trained, using **`logit_adjust_eval.py`** (Stage 1) or **`stage2_logit_adjust_eval.py`** (Stage 2).
