# Training runs: layout and canonical balanced pipelines

## Folder layout (from 2026-05 on)

All **new** Lightning training writes under:

| Stage | Path pattern |
|-------|----------------|
| Stage 1 (`segmentation.py train`) | `results/stage1/<run_name>/` |
| Stage 2 (`domain_adapt.py train`) | `results/stage2/<run_name>/` |

**Logs** (TensorBoard, optional CSV / W&B): `results/logs/stage{1,2}/<run_name>/`.

| Kind | Path |
|------|------|
| Checkpoints | `results/stage1/<run_name>/` or `results/stage2/<run_name>/` → `best.ckpt`, `best-v*.ckpt`, `last.ckpt` |
| TensorBoard (+ optional CSV / W&B artifacts) | `results/logs/stage{1,2}/<run_name>/` → **`tensorboard/version_0`** and **`csv_metrics/version_0`** (fixed version so each resume keeps one TB/CSV subtree). Older duplicate `version_1`… dirs can be moved under `_archived_lightning_versions/` if you already have clutter. Optional `wandb/` when enabled. |

Hyperparameters remain in Lightning checkpoints and in TensorBoard `meta/hparams_json` (TrainingMetaLoggerCallback).

**Resume Stage 1:** use `--resume_from_checkpoint results/stage1/<run_name>/last.ckpt` with the **same** `--run_name`, dataset paths, `--class_weights_path`, and architecture flags as the stopped run so checkpoints and TB logs stay in one folder.

**See also:** [Experiment tracking and HPO tutorial](experiment_tracking_and_hpo.md) — Optuna, Weights & Biases, CSV logging, and related tools mapped to this repo.

**See also:** [PyTorch profiling guide](pytorch_profiling_guide.md) — Profiler concepts, TensorBoard PROFILE, `--tb_profile` / `--tb_profile_cuda_only`, and Stage 1/2 smoke commands.

### Naming convention

Controlled by `--run_name` (optional). If omitted, a default is generated:

| Script | Default prefix | Example auto name |
|--------|------------------|-------------------|
| Segmentation | `ce_weighted_balanced__` | `ce_weighted_balanced__2026-05-10_143022_041` |
| Domain adapt + `--iwdan` | `transfer_iwdan_weighted__` | `transfer_iwdan_weighted__2026-05-10_091200_512` |
| Domain adapt without `--iwdan` | `transfer_dann__` | `transfer_dann__2026-05-10_091200_512` |

Override examples:

- Unweighted Stage 1: `--run_name ce_unweighted_baseline__2026-05-10`
- Stage 2 ablation: `--run_name transfer_iwdan_weighted__slow_grl_ablation`

### Canonical “balanced class weights” runs (this workspace)

These are the **weighted CE Stage 1** and **IWDAN + priors Stage 2** trainings (checkpoint + stored hyperparameters / TB meta):

| Role | Run folder (after migration) | Evidence |
|------|------------------------------|----------|
| **Stage 1 — weighted CE** | Checkpoints: `results/stage1/ce_weighted_balanced__2026-05-04_163109/`. Logs (TensorBoard + `hparams.yaml`): `results/logs/stage1/ce_weighted_balanced__2026-05-04_163109/`. | `class_weights_path` set to inverse-frequency weights |
| **Stage 2 — IWDAN + priors** | Checkpoints: `results/stage2/transfer_iwdan_weighted__2026-05-05_134214/`. Logs (TensorBoard + `hparams.yaml`): `results/logs/stage2/transfer_iwdan_weighted__2026-05-05_134214/`. | `iwdan: true`, `iwdan_source_priors` / `iwdan_target_priors` set; `pre_train` pointed at balanced Stage 1 |
| **Stage 1 — alpha=1.0 ablation** | Checkpoints: `results/stage1/ce_alpha100__2026-05-12_002501178/` (when finished). Logs + CSV + console: `results/logs/stage1/ce_alpha100__2026-05-12_002501178/` plus sibling `*_stdout.log` / `*_stderr.log`. | Same architecture/hparams as canonical Stage 1; `class_weights_path` = `artifacts/class_weights/ablation/source_train_alpha100.json` (counts aligned with alpha=0.5 JSON via derive script). |
| **Stage 1 — alpha=1.0 + profiler** | Prefer **`ce_alpha100_profile__*`**: checkpoints under `results/stage1/<run_name>/`; logs include **`--tb_profile`** traces (`*.pt.trace.json`, `fit-pytorch_profiler.txt`) beside TB — see [pytorch_profiling_guide.md §7b–7c](pytorch_profiling_guide.md). | Restart superseded runs without `--tb_profile` manually if both were started; only one job should hold the GPU. |

**Stage 2 without IWDAN** (e.g. old `BrepMFR_DA_from_balanced_stage1`) is *not* the same as the balanced-priors pipeline; it is archived under `results/_archive/` if you chose to zip legacy folders.

### Legacy training folders (zipped)

Older experiment directories were archived with lossless ZIP under **`results/_archive/`** (one zip per former top-level folder). Filenames are like `BrepMFR_balanced_2026-05-08.zip`. Restore by unzipping into a temp location; canonical **balanced** checkpoints already live under **`results/stage1/`** and **`results/stage2/`** as described above.

A prior copy of **`results/class_weights/`** was also zipped (`class_weights_*.zip`); **canonical** weight JSON belongs under **`artifacts/class_weights/`** in the repo.


### TensorBoard

Scalars, images, and text live under `tensorboard/version_*` inside each run’s logs folder.

```powershell
tensorboard --logdir results/logs/stage1/<run_name>/
tensorboard --logdir results/logs/stage2/<run_name>/
```

Pointing **`--logdir` at the run folder** (not only `.../tensorboard`) lets TensorBoard pick up both Lightning’s `tensorboard/version_*` events and optional **PyTorch Profiler** traces written next to them when you pass **`--tb_profile`** to `segmentation.py` or `domain_adapt.py`. Open the **PROFILE** tab after a run that captured traces (profiling adds overhead and larger files on disk).

Optional **`--tb_full_graph`** logs an extra **GRAPH** trace for **BrepEncoder + attention + classifier** using one **small real graph** from the dataset (tight caps first, then a relaxed scan). Graphs are **fixed-shape surrogates** for visualization; very large graphs may still fail to trace—check TB TEXT keys `model/graph_brep_encoder_pipeline_note` / `*_error`. Stage 2 also logs domain **discriminator** and **GRL→discriminator** graphs when `domain_adv` is present.

Profiler schedule knobs: `--tb_profile_wait`, `--tb_profile_warmup`, `--tb_profile_active`, `--tb_profile_repeat` (defaults profile only a few steps). On Windows, CUDA profiling depends on your PyTorch/Kineto build; CPU-only profiling still produces useful traces.

Legacy runs sometimes kept TensorBoard **`events.out.tfevents*`** beside checkpoints under `<run_ckpt_folder>/` (single file); those have been relocated to **`results/logs/stage{1,2}/<run>/tensorboard/version_0/`** when applicable (baseline Stage 1 / Stage 2 balanced runs migrated). Older runs may still keep events under `<run_ckpt_folder>/tensorboard/`; `scripts/hpo/tb_metrics.py` prefers `results/logs/...` and falls back automatically.
