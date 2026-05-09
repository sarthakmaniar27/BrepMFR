# Training runs: layout and canonical balanced pipelines

## Folder layout (from 2026-05 on)

All **new** Lightning training writes under:

| Stage | Path pattern |
|-------|----------------|
| Stage 1 (`segmentation.py train`) | `results/stage1/<run_name>/` |
| Stage 2 (`domain_adapt.py train`) | `results/stage2/<run_name>/` |

Each `<run_name>` is a **single directory** for one training job. It contains:

- `best.ckpt`, `best-v*.ckpt`, `last.ckpt` — checkpoints (top-level in that folder)
- `tensorboard/` — TensorBoard event files (subfolder created by PyTorch Lightning)
- `hparams.yaml` — saved hyperparameters

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

These are the **weighted CE Stage 1** and **IWDAN + priors Stage 2** trainings (verified from saved `hparams.yaml`):

| Role | Run folder (after migration) | Evidence |
|------|------------------------------|----------|
| **Stage 1 — weighted CE** | `results/stage1/ce_weighted_balanced__2026-05-04_163109/` | `class_weights_path` set to inverse-frequency weights |
| **Stage 2 — IWDAN + priors** | `results/stage2/transfer_iwdan_weighted__2026-05-05_134214/` | `iwdan: true`, `iwdan_source_priors` / `iwdan_target_priors` set; `pre_train` pointed at balanced Stage 1 |

**Stage 2 without IWDAN** (e.g. old `BrepMFR_DA_from_balanced_stage1`) is *not* the same as the balanced-priors pipeline; it is archived under `results/_archive/` if you chose to zip legacy folders.

### Legacy training folders (zipped)

Older experiment directories were archived with lossless ZIP under **`results/_archive/`** (one zip per former top-level folder). Filenames are like `BrepMFR_balanced_2026-05-08.zip`. Restore by unzipping into a temp location; canonical **balanced** checkpoints already live under **`results/stage1/`** and **`results/stage2/`** as described above.

A prior copy of **`results/class_weights/`** was also zipped (`class_weights_*.zip`); **canonical** weight JSON belongs under **`artifacts/class_weights/`** in the repo.


### TensorBoard

```powershell
tensorboard --logdir results/stage1/<run_name>/tensorboard
tensorboard --logdir results/stage2/<run_name>/tensorboard
```
