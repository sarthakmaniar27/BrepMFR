# Training Recipes

This document is the **single reference** for all training modes supported by
`segmentation.py`.  Each recipe below corresponds to an old `.ps1` launcher
script that has been archived under `scripts/threads/_archive/`.

> **Quick start** — pick a profile, provide your dataset path and checkpoint,
> and override anything you need on the command line:
>
> ```powershell
> python segmentation.py train \
>     --training_profile a1_a3_finetune_from_lite \
>     --dataset_path Z:\thread_and_text\no_a2 \
>     --pre_train model_checkpoints\lite\best.ckpt \
>     --class_weights_path artifacts/class_weights/thread_text/source_train_alpha05.json
> ```

---

## Profile summary

| Profile | Replaces | Init | LR | A1/A3 LR | Epochs | Description |
|---|---|---|---|---|---|---|
| `a1_a3_finetune_from_lite` | `train_a1_a3_from_lite.ps1` | `--pre_train` | 1e-4 | 1e-3 | 30 | Fine-tune A1/A3 branches from a lite checkpoint |
| `no_a2_from_scratch` | `train_no_a2_from_scratch.ps1` | random or `--pre_train` | 2e-3 | 2e-3 | 100 | Train a no-A2 model from scratch (or resume) |
| `new_abc_finetune` | `train_new_abc_finetune.ps1` | `--pre_train` | 1e-4 | 1e-4 | 15 | Fine-tune on newly labeled ABC data |
| `model_a_unique_abc_finetune` | `train_model_a_unique_abc_finetune.ps1` | `--pre_train` | 2e-5 | 2e-5 | 8 | Specialized Model A fine-tune on unique ABC |

---

## Recipe 1: A1/A3 Fine-tune from Lite

**Purpose:** Take an existing "lite" checkpoint (trained without A1/A3 attention
bias branches) and fine-tune it with the full A1+A3 attention bias enabled.

**When to use:** You have a good lite model and want to add geometric attention
(A1 = edge-path shortest-path distance, A3 = spatial position encoding).

**Prerequisites:**
- A1+A3 dataset under `<dataset_root>/pyg/` — prepared with
  `scripts/threads/prepare_a1_a3_finetune.ps1`
- Split files: `train.txt`, `val.txt`, `test.txt` at `<dataset_root>/`
- Class weights JSON (e.g., `artifacts/class_weights/thread_text/source_train_alpha05.json`)
- A lite `.ckpt` checkpoint

### Fresh fine-tune from lite weights

```powershell
python segmentation.py train `
    --training_profile a1_a3_finetune_from_lite `
    --dataset_path "Z:\thread_and_text\no_a2" `
    --pre_train "model_checkpoints\lite\best.ckpt" `
    --class_weights_path "artifacts/class_weights/thread_text/source_train_alpha05.json" `
    --run_name "thread_text_a1_a3_finetune_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
```

### Resume an interrupted A1/A3 fine-tune

```powershell
python segmentation.py train `
    --training_profile a1_a3_finetune_from_lite `
    --dataset_path "Z:\thread_and_text\no_a2" `
    --resume_from_checkpoint "results/stage1/<run_name>/last.ckpt" `
    --class_weights_path "artifacts/class_weights/thread_text/source_train_alpha05.json" `
    --run_name "<same_run_name>" `
    --check_val_every_n_epoch 1
```

### Key hyperparameters
| Parameter | Value | Notes |
|---|---|---|
| `learning_rate` | 0.0001 | Conservative for pre-trained backbone |
| `a1_a3_learning_rate` | 0.001 | 10× higher for newly initialized A1/A3 |
| `a1_a3_ramp_epochs` | 5 | Gradually scale A1/A3 from 0.1 → 1.0 |
| `max_nodes_for_a3` | 768 | Skip dense A3 for very large graphs |
| `batch_node_sq_budget` | 4,000,000 | Adaptive batching by padded N² cost |
| `precision` | 16-mixed | FP16 mixed precision for speed |

---

## Recipe 2: No-A2 From Scratch

**Purpose:** Train a completely new model from random initialization using
no-A2 graphs (A1 + A3 features active from epoch 0).

**When to use:** Building a fresh model on a new or expanded dataset.

**Prerequisites:**
- Expanded no_a2 dataset — prepared with
  `scripts/threads/prepare_no_a2_scratch_delta.ps1`
- Class weights JSON
- No checkpoint needed (random init)

### Fresh training from scratch

```powershell
python segmentation.py train `
    --training_profile no_a2_from_scratch `
    --dataset_path "D:\thread_and_text\no_a2_large" `
    --class_weights_path "artifacts/class_weights/thread_text/no_a2_large_70k_train_alpha05.json" `
    --run_name "thread_text_no_a2_70k_scratch_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
```

### Resume an interrupted scratch run

```powershell
python segmentation.py train `
    --training_profile no_a2_from_scratch `
    --dataset_path "D:\thread_and_text\no_a2_large" `
    --resume_from_checkpoint "results/stage1/<run_name>/last.ckpt" `
    --class_weights_path "artifacts/class_weights/thread_text/no_a2_large_70k_train_alpha05.json" `
    --run_name "<same_run_name>"
```

### Initialize from pre-trained weights (not exact resume)

```powershell
python segmentation.py train `
    --training_profile no_a2_from_scratch `
    --dataset_path "D:\thread_and_text\no_a2_large" `
    --pre_train "model_checkpoints\some_model\best.ckpt" `
    --class_weights_path "artifacts/class_weights/thread_text/no_a2_large_70k_train_alpha05.json"
```

### Key hyperparameters
| Parameter | Value | Notes |
|---|---|---|
| `learning_rate` | 0.002 | Standard for training from scratch |
| `a1_a3_learning_rate` | 0.002 | Same as backbone (everything is new) |
| `a1_a3_ramp_epochs` | 0 | No ramp — A1/A3 fully active from epoch 0 |
| `max_epochs` | 100 | Longer for from-scratch training |
| `check_val_every_n_epoch` | 2 | Less frequent validation |

---

## Recipe 3: New ABC Fine-tune

**Purpose:** Fine-tune an existing model on newly labeled ABC (synthetic) data
that has been merged with the original training dataset.

**When to use:** You received new labeled ABC JSONs and want to incorporate them
without retraining from scratch.

**Prerequisites:**
- Combined no_a2 dataset — prepared with
  `scripts/threads/prepare_new_abc_finetune_data.ps1`
- A base checkpoint to fine-tune from

### Fresh fine-tune

```powershell
python segmentation.py train `
    --training_profile new_abc_finetune `
    --dataset_path "D:\thread_and_text\no_a2_combined" `
    --pre_train "results/stage1/<base_run>/best.ckpt" `
    --run_name "thread_text_new_abc_finetune_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
```

### With class weights

```powershell
python segmentation.py train `
    --training_profile new_abc_finetune `
    --dataset_path "D:\thread_and_text\no_a2_combined" `
    --pre_train "results/stage1/<base_run>/best.ckpt" `
    --class_weights_path "artifacts/class_weights/thread_text/new_abc_finetune_alpha05.json"
```

### Key hyperparameters
| Parameter | Value | Notes |
|---|---|---|
| `learning_rate` | 0.0001 | Conservative for fine-tuning |
| `a1_a3_learning_rate` | 0.0001 | Same as backbone (both already trained) |
| `max_epochs` | 15 | Short fine-tuning run |
| `check_val_every_n_epoch` | 1 | Validate every epoch |
| `csv_log` | True | CSV logging enabled by default |

---

## Recipe 4: Model A → Unique ABC Fine-tune

**Purpose:** Specialized fine-tuning that takes the "Model A" checkpoint and
fine-tunes it on unique new ABC data not seen during Model A's original training.

**When to use:** Incremental model improvement with carefully curated ABC data.

**Prerequisites:**
- Prepared dataset — created by
  `scripts/threads/prepare_model_a_unique_abc_finetune_data.ps1`
- Model A checkpoint (defaults to `model_checkpoints/abc_with_no_a2/last-v1.ckpt`)

### Default fine-tune

```powershell
python segmentation.py train `
    --training_profile model_a_unique_abc_finetune `
    --dataset_path "Z:\thread_and_text\abc_for_modelA_finetuning" `
    --pre_train "model_checkpoints\abc_with_no_a2\last-v1.ckpt" `
    --run_name "thread_text_model_a_unique_abc_finetune_v1"
```

### Key hyperparameters
| Parameter | Value | Notes |
|---|---|---|
| `learning_rate` | 0.00002 | Very conservative (2e-5) |
| `a1_a3_learning_rate` | 0.00002 | Same for all branches |
| `max_epochs` | 8 | Very short fine-tuning |
| Class weights | disabled | No class weighting for this recipe |

---

## Overriding profile defaults

Any CLI argument you provide explicitly **always wins** over the profile default.
For example, to use the `no_a2_from_scratch` profile but with a shorter run:

```powershell
python segmentation.py train `
    --training_profile no_a2_from_scratch `
    --dataset_path "D:\thread_and_text\no_a2_large" `
    --class_weights_path "artifacts/class_weights/thread_text/no_a2_large_70k_train_alpha05.json" `
    --max_epochs 20 `
    --learning_rate 0.001
```

Here `--max_epochs 20` and `--learning_rate 0.001` replace the profile's 100 and
0.002, while all other profile defaults still apply.

---

## Data preparation scripts

The data preparation `.ps1` scripts remain in `scripts/threads/` because they
orchestrate multi-step pipelines (JSON label remapping, PyG conversion, split
generation, class weight computation, validation) that don't reduce to a single
`segmentation.py` call:

| Script | Purpose |
|---|---|
| `prepare_a1_a3_finetune.ps1` | Upgrade lite .pt graphs to A1+A3 (no_a2) profile |
| `prepare_no_a2_scratch_delta.ps1` | Build expanded no_a2 dataset from new JSONs |
| `prepare_new_abc_finetune_data.ps1` | Audit + combine new ABC data with old dataset |
| `prepare_model_a_unique_abc_finetune_data.ps1` | Build Model A unique-ABC dataset |
| `post_pyg_export.ps1` | Post-conversion splits + class weights (2-class thread) |
| `post_thread_text_pyg_export.ps1` | Post-conversion splits + class weights (3-class thread+text) |

---

## Legacy `.ps1` archive

The original PowerShell training launchers have been archived under
`scripts/threads/_archive/` for reference. They are functionally replaced by
the `--training_profile` argument documented above.
