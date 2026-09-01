# BrepMFR on PyG: Stage 1, Stage 2, and the Path from Plateau to Strong Target Performance

**Document type:** engineering retrospective and results summary  
**Scope:** PyTorch Geometric stack, MFCAD++-style source/target B-rep face segmentation, two-stage training (supervised source pre-training + domain adaptation).  
**Audience:** teammates and future you—anyone asking whether “we finally broke the ceiling.”

---

## Executive summary

Over an extended effort (on the order of **months** of iteration), the project moved from a frustrating **target-domain plateau**—where Stage 2 domain adaptation (DA) seemed unable to improve accuracy, and in several configurations actively hurt it—to a regime where **class imbalance and label shift were diagnosed**, **Stage 1 was retrained with balanced loss**, and a **fresh Stage 2** run using a **slow gradient-reversal ramp** and **importance-weighted DANN (IWDAN)** produced **much stronger target metrics** on the evaluated split.

**Did we “break the ceiling”?**

- **Relative to the old PyG Stage 2 story (DANN net-negative vs. Stage 1 + post-hoc correction):** **Yes.** Earlier checkpoints under “gentle” or “authors-faithful” DA were **worse** than taking Stage 1 alone and applying Saerens-style logit adjustment on target val. The new pipeline reverses that pattern: a strong Stage 2 checkpoint exists, and post-hoc adjustment still yields incremental gains on top.
- **Relative to the original unbalanced Stage 1 on target val:** **Yes, for balanced Stage 1 alone** (target per-face accuracy **0.8055 → 0.8208** baseline; **0.8617** with τ on val—see tables below). **Stage 2 on target test** reports **0.9397** per-face (τ=0) and **0.9487** (best τ), which is a large jump—but **this is not on the same filelist as the classic Stage 1 val numbers** (see Limitations).
- **Relative to a published headline number (e.g. ~92.7% mean per-class in the paper narrative used internally):** **Close but not automatically identical.** The latest **mean per-class recall** on **target test** after Stage 2 + best τ is **0.9113**. That is excellent, but protocol details (split definitions, exact metric, and whether priors are oracle vs. estimated) must match before claiming parity with any external benchmark.

**Bottom line:** The struggle was real; the diagnosis (label shift + imbalance + over-confidence) was correct; **retraining Stage 1 and redesigning Stage 2 were necessary**. The numbers on **target test** are the strongest evidence so far. **One more apples-to-apples eval** (same split for all rows in a single table) is the right next step for absolute certainty in reports and papers.

---

## Narrative timeline (phased, not calendar-dated)

This section is intentionally **phase-oriented**. Exact calendar weeks are not required for the argument; the repository artifacts and diagnostics anchor the story.

### Phase A — Migration to PyG (engineering unblock)

The training stack was moved from **DGL** to **PyTorch Geometric** so that training could run on **modern CUDA** (including **Blackwell-class** GPUs) using **Windows-friendly** wheels. Data under `Z:\Experiment6` (DGL `.bin`) was converted to **`*.pt`** graphs and paired filelists under `Z:\Experiment6_PyG`. Code changes centered on `edge_index` batching instead of DGL graphs, Lightning 2.x trainers, and small compatibility shims.

**Reference:** [markdowns/BREP_MFR_PYG.md](BREP_MFR_PYG.md).

**Effect on the “Stage 2 struggle”:** Migration was a prerequisite to keep experimenting; it did not by itself fix domain gap. It changed *where* we trained, not *why* the target metric was stuck.

### Phase B — Reproducing Stage 1 and hitting the Stage 2 plateau

Stage 1 (supervised on **source**) achieved **very high source validation accuracy** with the original (class-imbalanced) training recipe. Stage 2 (DANN-style **domain adaptation**) was then tuned—adversarial weight schedules, learning rates, “gentle” vs. “aggressive” setups. **Target accuracy did not respond** in the way one expects if covariate shift were the only problem.

The post-hoc analysis phase later showed something sharper: **Several Stage 2 checkpoints were worse than “Stage 1 + zero-training logit adjustment.”** In other words, DA was not only plateauing—it could be **net harmful** under label shift and an over-confident encoder.

**Reference:** [markdowns/post_hoc_logit_adjustment.md](post_hoc_logit_adjustment.md) (Stage 2 vs. Stage 1+τ comparison table).

### Phase C — Diagnosis: label shift dominates; a few classes show real domain shift

A dedicated diagnostic ([scripts/diagnostics/diagnose_stage1_target.py](../scripts/diagnostics/diagnose_stage1_target.py)) compared **source val** vs. **target val** with pure Stage 1 inference—no DA. It bucketed failure modes: class collapse, encoder weakness, suspected label remapping, vs. genuine domain shift.

**Key outcomes on the original Stage 1 checkpoint** ([results/diagnostics/stage1_audit/summary.md](../results/diagnostics/stage1_audit/summary.md) — `BrepMFR/0425/183526/best.ckpt`):

| Metric | Source val (`s_val.txt`) | Target val (`t_val.txt`) | Gap |
|--------|--------------------------|--------------------------|-----|
| Per-face accuracy | **0.9934** (272,344 faces) | **0.8055** (268,029 faces) | **+0.1879** |

The report highlighted **label-mapping suspects** that, on closer inspection via label priors, were **errors collapsing into class 0 (stock)**—consistent with **label shift**, not a silent relabeling bug. Under **domain-shift candidates**, **class 6** and **class 9** stood out as **high source acc, very low target acc** (e.g. class 6 target acc **0.029** vs. source **0.964** on that diagnostic).

### Phase D — Interim fix: Saerens / BBSE-style post-hoc logit adjustment

**Method (conceptual):** If \(P_s(Y) \neq P_t(Y)\) but \(P(X\mid Y)\) is stable enough, predictions can be reweighted using \(\log P_t(c) - \log P_s(c)\) (possibly scaled by **τ** when probabilities are mis-calibrated).

**Result on original Stage 1, target val** (from [markdowns/post_hoc_logit_adjustment.md](post_hoc_logit_adjustment.md)):

| τ | Per-face acc | Mean per-class acc |
|---:|---:|---:|
| 0.00 | 0.8055 | 0.7209 |
| … | … | … |
| **3.00** | **0.8474** | **0.7650** |

**Interpretation:**

- A **large τ** was required for best results vs. the theoretical τ≈1, implying **over-confident softmax** and **imbalance-driven saturation**—not a perfectly calibrated classifier.
- Applying the **same** idea to **earlier Stage 2 checkpoints** still left **Stage 1 + τ** on top in those experiments—DA had **warped** features/classifier in an unhelpful direction under the dominant label-shift signal.

### Phase Durable fix — Class-balanced Stage 1

Rather than relying only on inference-time correction, the durable fix **bakes balance into training**:

1. **Count** class frequencies on **source train** (`scripts/training/compute_class_weights.py`).
2. Compute **sqrt-inverse-frequency** weights (α=0.5), mean-normalised, lightly clipped—stable for extreme imbalance (see companion doc).
3. Pass **`--class_weights_path`** into Stage 1 so **training** cross-entropy uses per-class weights; **validation loss** stays unweighted for a stable plateau signal.

**Reference:** [markdowns/class_imbalance_stage1_retrain.md](class_imbalance_stage1_retrain.md).

**Checkpoint audited:** `results/stage1/ce_weighted_balanced__2026-05-04_163109/best.ckpt` — [results/diagnostics/stage1_balanced_final/summary.md](../results/diagnostics/stage1_balanced_final/summary.md).

| Metric | Source val | Target val | Gap |
|--------|------------|------------|-----|
| Per-face accuracy | **0.9891** | **0.8208** | **+0.1683** |

Balanced training **improved target per-face** vs. unbalanced Stage 1 at τ=0 (**0.8055 → 0.8208**) and changed the nature of errors, but **hard classes remained hard on val**: e.g. **class 6** still **~0.050** target accuracy on that diagnostic—exactly the scenario where **Stage 2 should matter**, *if* the adversarial objective is no longer drowned out by label shift.

**Post-hoc τ on balanced Stage 1, target val** ([results/diagnostics/logit_adjust_balanced_final/summary.md](../results/diagnostics/logit_adjust_balanced_final/summary.md)):

| Setting | Per-face acc | Mean per-class acc |
|---------|-------------|-------------------|
| Baseline τ=0 | 0.8208 | 0.7310 |
| Best τ=4 | **0.8617** | **0.7732** |

So even after balanced training, **residual label shift** still rewards a modest τ sweep on val—though the story is closer to “calibration + priors” than the extreme τ needed before.

### Phase E — Stage 2 redux: slow GRL + IWDAN from balanced Stage 1

A new Stage 2 experiment initialized from **balanced Stage 1** and addressed two known failure modes:

1. **Gradient reversal warmup:** CLI support in [domain_adapt.py](../domain_adapt.py) for **`--grl_ramp_frac`**, **`--grl_max_iters`**, and **`--estimated_steps_per_epoch`** so λ does not spike according to a tiny fixed iteration budget (the classic “λ saturates in \<1 epoch” pitfall).

2. **Importance-weighted DANN (IWDAN):** When source and target label priors differ, **plain DANN is misspecified**. IWDAN applies per-class importance ratios \(w[c] \propto P_T(c)/P_S(c)\) on the **source side** of the discriminator loss (Tachet des Combes et al., NeurIPS 2020), aligned with the theory cited in Zhao et al. (2019). Flags: **`--iwdan`**, **`--iwdan_source_priors`**, **`--iwdan_target_priors`**, **`--iwdan_clip`**.

**Best checkpoint evaluated in this report (Stage 2):**  
`C:/Users/D58/Desktop/BrepMFR_PyG/results/stage2/transfer_iwdan_weighted__2026-05-05_134214/best.ckpt`

---

## Methodology notes (how to read the metrics)

- **Per-face accuracy:** fraction of **faces** predicted correctly (macro volume is dominated by frequent classes like stock).
- **Mean per-class recall (reported as “mean per-class” in several tables):** average of **per-class recall** over classes—**more informative for imbalance** than raw per-face accuracy.
- **IoU:** reported as **mean IoU over finite classes** in the Stage 2 + τ summary (same construction as training code paths).
- **τ sweep:** applies **post-hoc** log correction **after** forward pass; **source priors** for adjustment were taken from **`s_val.txt`** counts; **target priors** from the **same target split** used for scoring (oracle POC, as in the original logit-adjustment doc).

---

## Results tables

### 1. Original Stage 1 (unbalanced) — target val

| Setting | Checkpoint / doc | Per-face acc | Mean per-class acc |
|---------|------------------|-------------|-------------------|
| Baseline (τ=0) | `BrepMFR/0425/183526` — stage1_audit | **0.8055** | **0.7209** (from logit sweep doc) |
| Post-hoc best τ | τ=3 — post_hoc_logit_adjustment.md | **0.8474** | **0.7650** |

**Earlier Stage 2 (for context):** same document reports **Stage 2 gentle** and **Stage 2 authors-faithful** **below** Stage 1+τ on those runs—DA was not helping until the root cause was handled.

### 2. Balanced Stage 1 — target val

| Setting | Checkpoint | Per-face acc | Mean per-class acc |
|---------|------------|-------------|-------------------|
| Baseline (τ=0) | `stage1/ce_weighted_balanced__2026-05-04_163109` | **0.8208** | **0.7310** |
| Post-hoc best τ | τ=4 — logit_adjust_balanced_final | **0.8617** | **0.7732** |

### 3. New Stage 2 (IWDAN + slow GRL) — target **test**

**Primary source:** [results/diagnostics/stage2_logit_adjust_t_test/summary.md](../results/diagnostics/stage2_logit_adjust_t_test/summary.md) — filelist **`t_test.txt`**, **267,964** faces.

| Setting | Per-face acc | Mean per-class recall | Mean IoU (finite classes) |
|---------|-------------|------------------------|----------------------------|
| Baseline τ=0 | **0.9397** | **0.9028** | **0.8584** |
| Best τ=4 | **0.9487** | **0.9113** | **0.8678** |

**Full τ sweep (target test):**

| τ | Per-face acc | Mean per-class recall |
|---:|---:|---:|
| 0.00 | 0.9397 | 0.9028 |
| 1.00 | 0.9458 | 0.9077 |
| 2.00 | 0.9470 | 0.9091 |
| 3.00 | 0.9478 | 0.9101 |
| **4.00** | **0.9487** | **0.9113** |

### 4. Illustrative per-class contrast (why “same split” matters)

On **balanced Stage 1, target val**, **class 6 (rectangular through step)** had target accuracy **~0.050** ([stage1_balanced_final/summary.md](../results/diagnostics/stage1_balanced_final/summary.md)).

On **new Stage 2, target test**, **class 6 recall** is **0.716** (τ=0) and **0.744** (τ=4) ([stage2_logit_adjust_t_test/summary.md](../results/diagnostics/stage2_logit_adjust_t_test/summary.md)).

Both statements are **true in their artifacts**—but **val ≠ test**, so they must **not** be quoted as a controlled A/B without rerunning one model on the other split.

**Per-class detail** for Stage 2 + τ is available in the same summary file (25-class tables) and in sibling CSVs under `results/diagnostics/stage2_logit_adjust_t_test/`.

---

## Limitations and professional skepticism

1. **Val vs. test:** Most Stage 1 headline metrics are on **`t_val.txt`**. The flagship Stage 2 block above is on **`t_test.txt`**. **Do not** compare **0.82** and **0.94** as if they were the same experiment without a unified eval.

2. **Oracle target priors for τ:** The sweep uses **empirical** \(P_t(c)\) on the **labeled** evaluation split. Deployment or strict blind evaluation should use **BBSE** or another estimator from **unlabeled** target data (as noted in [post_hoc_logit_adjustment.md](post_hoc_logit_adjustment.md)).

3. **τ selection:** Choosing **τ** on the **same** labeled set you report inflates optimism slightly. Prefer a **held-out** target fold for τ.

4. **Stubborn classes:** Even at strong global metrics, some classes remain weak (e.g. **Chamfer** / class 15 in the Stage 2 test table: recall **~0.66**). **“Success” is uneven** across the taxonomy.

5. **Protocol parity with literature:** Any external **92.x%** figure must match **split, metric definition, and adjustment** before claiming equality. Internal mean per-class recall **0.911** is in the same ballpark but is **not automatically** the paper number.

6. **Class-weight validation:** Keeping val loss **unweighted** is deliberate for training stability; reporting should continue to rely on **target diagnostics**, not val loss alone.

---

## Deliverables map (reproducibility)

| Artifact | Role |
|----------|------|
| [scripts/diagnostics/diagnose_stage1_target.py](../scripts/diagnostics/diagnose_stage1_target.py) | Stage 1 vs. target confusion + bucketing |
| [scripts/diagnostics/logit_adjust_eval.py](../scripts/diagnostics/logit_adjust_eval.py) | Stage 1 + τ (val/test configurable) |
| [scripts/diagnostics/stage2_logit_adjust_eval.py](../scripts/diagnostics/stage2_logit_adjust_eval.py) | Stage 2 + cached probs + τ sweep |
| [scripts/training/compute_class_weights.py](../scripts/training/compute_class_weights.py) | Source (or target) prior JSON for weights / IWDAN |
| [artifacts/class_weights/stage1/source_train_alpha05.json](../artifacts/class_weights/stage1/source_train_alpha05.json) | Default balanced weights (α=0.5) |
| [results/diagnostics/stage1_audit/](../results/diagnostics/stage1_audit/) | Unbalanced Stage 1 diagnosis |
| [results/diagnostics/stage1_balanced_final/](../results/diagnostics/stage1_balanced_final/) | Balanced Stage 1 diagnosis |
| [results/diagnostics/logit_adjust_balanced_final/](../results/diagnostics/logit_adjust_balanced_final/) | Balanced Stage 1 τ sweep |
| [results/diagnostics/stage2_logit_adjust_t_test/](../results/diagnostics/stage2_logit_adjust_t_test/) | New Stage 2 + τ on **test** |

---

## Conclusion

The **three-month struggle** was not “DA cannot work on B-reps.” It was **DA cannot fix label shift by feature alignment alone**, and under extreme **class imbalance** the **Stage 1 encoder absorbed a destructive source prior**, which made **GradRev-driven training** fight the wrong war. **Post-hoc logit adjustment** proved the point cheaply; **class-balanced Stage 1** made the representation trainable again; **IWDAN + slow GRL** made **Stage 2** align with theory instead of amplifying the failure mode.

**The latest numbers on target test are strong enough to treat this as a real breakthrough for this codebase**—with the **explicit homework** of one **unified table** (same split, same priors protocol) for publication-grade claims.

---

## Appendix: journey flowchart

```mermaid
flowchart LR
  migrate[PyG_migration] --> s1a[Stage1_unbalanced]
  s1a --> plateau[Stage2_DA_plateau]
  plateau --> diag[Label_shift_diagnosis]
  diag --> logit[Post_hoc_logit_adj]
  logit --> s1b[Stage1_class_balanced]
  s1b --> s2new[Stage2_IWDAN_slow_grl]
  s2new --> eval[Eval_plus_tau_sweep]
```

---

## Appendix: apples-to-apples eval (target val, Stage 2)

To populate **Stage 2 + τ** on **`t_val.txt`** (same convention as Stage 1 val tables), run from repo root (adjust paths to your machine):

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
python scripts/diagnostics/stage2_logit_adjust_eval.py `
  --checkpoint "C:/Users/D58/Desktop/BrepMFR_PyG/results/stage2/transfer_iwdan_weighted__2026-05-05_134214/best.ckpt" `
  --source_path Z:/Experiment6_PyG/source_dataset `
  --target_path Z:/Experiment6_PyG/target_dataset `
  --target_split val `
  --batch_size 32 --num_workers 4 `
  --out_dir results/diagnostics/stage2_logit_adjust_t_val
```

Compare the resulting `summary.md` directly to `results/diagnostics/stage1_balanced_final/summary.md` and `logit_adjust_balanced_final/summary.md`.

**Environment (important on Windows):** These scripts require the **`brep_mfr_pyg`** interpreter. A plain `python` in a fresh shell often resolves to **base Anaconda** or **another Python** that does **not** have `torch_geometric`, which yields `ModuleNotFoundError: No module named 'torch_geometric'`.

Use any one of:

```powershell
conda activate brep_mfr_pyg
python scripts/diagnostics/stage2_logit_adjust_eval.py ...
```

```powershell
conda run -n brep_mfr_pyg python scripts/diagnostics/stage2_logit_adjust_eval.py ...
```

```powershell
& "$env:LOCALAPPDATA\anaconda3\envs\brep_mfr_pyg\python.exe" scripts/diagnostics/stage2_logit_adjust_eval.py ...
```

*(On this machine the env path is `C:\Users\D58\AppData\Local\anaconda3\envs\brep_mfr_pyg`. Automated tools that only probe `%USERPROFILE%\anaconda3\envs\` will miss a **per-user Local** Anaconda install.)*

If `results/diagnostics/stage2_logit_adjust_t_val/` is not in the repo yet, the full val eval has not been run to completion with the correct interpreter and dataset mounted.

---

*This report synthesizes repository markdowns and numeric summaries; cite those files when forwarding excerpts externally.*
