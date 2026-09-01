# Stock + thread + text + chamfer + fillet (5-class) pipeline

SolidWorks-style face labels → BrepMFR Stage 1 with **`num_classes=5`**:

| Meaning | Raw `label` in JSON | After remap |
|---------|---------------------|-------------|
| Stock | `0`, `-1`, `-10` | `0` |
| Thread | `70` | `1` |
| Text (emboss) | `101` | `2` |
| Chamfer | `15` | `3` |
| Fillet | `24` | `4` |

Identity-safe map (also keeps already-remapped `1`–`4` unchanged):  
[`remap_maps/thread_text_sw_to_brep_with_identity.json`](remap_maps/thread_text_sw_to_brep_with_identity.json).

Strict raw map (no identity for `1`–`4`):  
[`remap_maps/thread_text_sw_to_brep.json`](remap_maps/thread_text_sw_to_brep.json).

Use **`conda run -n brep_mfr_pyg python ...`** on Windows if base Python lacks PyG.

---

## Current 39,450-file run: A1+A3 from scratch, no ABC

This is the canonical path for:

- source: `Z:\thread_and_text\cadsynth_with_fillets_and_champer\root_json`
- exactly five output classes (`0..4`)
- A1 (`spatial_pos`) + A3 (`edge_path`), with no dense A2
- random model initialization (no `--pre_train`)
- no `abc_jsons` input and no ABC split quota

The preparation launcher intentionally uses a two-stage conversion:

1. remapped JSON → `lite` graphs with file-level workers (fast geometry/topology extraction);
2. `lite` → `no_a2` using file-level workers to calculate A1/A3.

This is faster and more resume-friendly than calculating all-pairs A1/A3 serially
inside the JSON converter. The lite graphs are only staging data; no lite model is
trained or fine-tuned.

### A. Read-only label audit

```powershell
cd C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG

powershell -ExecutionPolicy Bypass -File scripts/threads/prepare_5class_a1_a3_scratch.ps1
```

The script scans all JSON labels and stops without changing files. Unknown labels
cause a non-zero exit.

The `Z:` path is a mapped network share on this workstation. A serial test read
took roughly 3–5 seconds per file, so the launcher uses 12 parallel remap workers,
8 parallel lite-conversion workers, and up to 12 A1/A3 workers by default. Tune
these only if the file server or host memory becomes saturated:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/threads/prepare_5class_a1_a3_scratch.ps1 `
  -RemapWorkers 16 `
  -LiteWorkers 8 `
  -FileWorkers 12
```

### B. Remap and build the complete A1+A3 dataset

The next command changes `face[].label` in the source JSON files in place. Copy or
snapshot the source folder first if the raw ids must be preserved.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/threads/prepare_5class_a1_a3_scratch.ps1 `
  -ApplyLabelRemap
```

The launcher then verifies one output graph per source JSON, makes STEP-aware
80/10/10 splits, validates every A1/A3 graph with `--num-classes 5`, and computes
class weights from the train split only.

Default outputs:

```text
Z:\thread_and_text\cadsynth_with_fillets_and_champer\five_class_a1_a3\
  lite\                    # intermediate graphs and split lists
  no_a2\pyg\               # final A1+A3 training graphs
  no_a2\train.txt
  no_a2\val.txt
  no_a2\test.txt

artifacts\class_weights\thread_text\
  cadsynth_5class_a1_a3_train_alpha05.json
```

### C. Start true scratch training

```powershell
powershell -ExecutionPolicy Bypass -File scripts/threads/train_5class_a1_a3_from_scratch.ps1
```

The launcher passes `--num_classes 5` and
`--full_a1_a3_from_scratch`. It does not pass `--pre_train`, so the model is
randomly initialized and A1/A3 are active at scale `1.0` from epoch 0.

The safe default `-MaxNodesForA3 768` still uses A1 on every graph but skips the
dense A3 tensor for batches whose padded face count exceeds 768. After preparation,
the validator reports how many graphs exceed this threshold. To force A3 on every
graph (substantially higher peak VRAM), use `-MaxNodesForA3 0`.

Quick one-batch smoke run before committing to 100 epochs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/threads/train_5class_a1_a3_from_scratch.ps1 `
  -RunName five_class_a1_a3_smoke `
  -MaxEpochs 1 `
  -LimitTrainBatches 1 `
  -DataLoaderWorkers 0
```

Do not resume the smoke run for real training. Start the normal command with a new
run name after the smoke run succeeds.

### D. Resume an interrupted scratch run

```powershell
powershell -ExecutionPolicy Bypass -File scripts/threads/train_5class_a1_a3_from_scratch.ps1 `
  -ResumeFromCheckpoint "results\stage1\<run_name>\last.ckpt"
```

This is exact continuation (model + optimizer + epoch), not fine-tuning.

### E. Test the best checkpoint

```powershell
conda run --no-capture-output -n brep_mfr_pyg python segmentation.py test `
  --dataset_path Z:\thread_and_text\cadsynth_with_fillets_and_champer\five_class_a1_a3\no_a2 `
  --pt_subdir pyg `
  --num_classes 5 `
  --drop_invalid_graphs `
  --batch_size 4 `
  --num_workers 0 `
  --max_nodes_for_a3 768 `
  --checkpoint "results\stage1\<run_name>\best.ckpt"
```

### F. Generate a manager-friendly training report

The report reader can run during or after training. It generates a self-contained
HTML report with plots, a concise Markdown summary, a scalar CSV export, and a JSON
summary:

```powershell
python scripts/training/analyze_tensorboard_run.py `
  --run-name five_class_a1_a3_scratch_20260806_214444
```

Default output:

```text
results/reports/five_class_a1_a3_scratch_20260806_214444/
  manager_report.html
  manager_summary.md
  tensorboard_scalars.csv
  summary.json
```

Open the HTML report:

```powershell
Start-Process results/reports/five_class_a1_a3_scratch_20260806_214444/manager_report.html
```

For logs outside the repository's normal run layout:

```powershell
python scripts/training/analyze_tensorboard_run.py `
  --log-dir "D:\path\to\tensorboard" `
  --output-dir "D:\path\to\training_report"
```

The report prioritizes macro class accuracy, mean IoU, and per-class behavior over
overall face accuracy because the five-class corpus is imbalanced. Its conclusions
describe validation behavior; held-out test metrics are still required for a
production-readiness claim.

---

## 1. Repair / remap JSON labels (data-driven map)

**Prefer the identity map** when JSON folders may mix raw SolidWorks ids (`15`, `24`, `70`, `101`) and already-normalized ids (`1`–`4`).

Inspect only:

```powershell
cd C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG
conda run -n brep_mfr_pyg python scripts/threads/repair_json_face_labels.py `
  --json-dir D:\thread_and_text\root_json `
  --map-json scripts/threads/remap_maps/thread_text_sw_to_brep_with_identity.json `
  --dry-run
```

Optional strict check (exit 1 if any label is not in the map):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/repair_json_face_labels.py `
  --json-dir D:\thread_and_text\root_json `
  --map-json scripts/threads/remap_maps/thread_text_sw_to_brep_with_identity.json `
  --dry-run --fail-on-unknown
```

Apply remaps and rewrite JSON (`indent=2`):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/repair_json_face_labels.py `
  --json-dir D:\thread_and_text\root_json `
  --map-json scripts/threads/remap_maps/thread_text_sw_to_brep_with_identity.json `
  --yes-write
```

Also remap **`abc_jsons`** the same way:

```powershell
conda run -n brep_mfr_pyg python scripts/threads/repair_json_face_labels.py `
  --json-dir D:\thread_and_text\abc_jsons `
  --map-json scripts/threads/remap_maps/thread_text_sw_to_brep_with_identity.json `
  --yes-write
```

Writes are refused if any face label is missing from the map (unless you pass **`--allow-unmapped`**, not recommended).

---

## 2. Class distribution

**JSON** (raw or after repair):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/count_thread_label_distribution.py `
  --json-dir D:\thread_and_text\root_json `
  --group "0:stock,1:thread,2:text,3:chamfer,4:fillet"
```

**PyG** (after conversion):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/count_thread_label_distribution.py `
  --pyg-dir D:\thread_and_text\lite\pyg `
  --group "0:stock,1:thread,2:text,3:chamfer,4:fillet"
```

Optional raw-id scan before remap (counts SolidWorks `15`/`24`/`70`/`101`):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/count_5class_distribution.py
```

---

## 3. JSON → PyG (lite)

Long-running. Adjust workers / paths to your machine.

Primary folder only:

```powershell
cd C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG
conda run -n brep_mfr_pyg python scripts/inference/json_to_brepmfr_pyg_optimized.py `
  --json_dir D:\thread_and_text\root_json `
  --pt_out_dir D:\thread_and_text\lite\pyg `
  --label_out_dir D:\thread_and_text\lite\label `
  --spatial_pos_max 32 `
  --inference_profile lite `
  --shortest_path_workers 8
```

**Also convert `abc_jsons` into the same PyG/label dirs** (this training run):

```powershell
conda run -n brep_mfr_pyg python scripts/inference/json_to_brepmfr_pyg_optimized.py `
  --json_dir D:\thread_and_text\root_json `
  --abc_json_dir D:\thread_and_text\abc_jsons `
  --pt_out_dir D:\thread_and_text\lite\pyg `
  --label_out_dir D:\thread_and_text\lite\label `
  --spatial_pos_max 32 `
  --inference_profile lite `
  --shortest_path_workers 8
```

Writes `lite/abc_stems.txt` (stems that came from `--abc_json_dir`) for bookkeeping.

---

## 4. Splits + class weights + recount

Splits are **STEP-key aware**: all variants sharing `..._step_NNN` go to the **same** split (no train/test leakage across bodies/variants).

Pass `--abc-json-dir` so **≥80%** of ABC stems land in **train** (leftover ABC groups go to val/test only):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/make_random_splits.py `
  --pyg-dir D:\thread_and_text\lite\pyg `
  --out-dir D:\thread_and_text\lite `
  --abc-json-dir D:\thread_and_text\abc_jsons `
  --abc-min-train-frac 0.8 `
  --seed 42
```

### Class weights from train distribution (`alpha=0.5` sqrt-inverse)

```powershell
New-Item -ItemType Directory -Force -Path "artifacts/class_weights/thread_text" | Out-Null

conda run -n brep_mfr_pyg python scripts/training/compute_class_weights.py `
  --dataset_path D:\thread_and_text\lite `
  --split train `
  --num_classes 5 `
  --alpha 0.5 `
  --num_workers 0 `
  --out artifacts/class_weights/thread_text/source_train_5class_alpha05.json
```

Or edit paths inside the post-export script and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/threads/_archive/post_thread_text_pyg_export.ps1
```

Recount after weights:

```powershell
conda run -n brep_mfr_pyg python scripts/threads/count_thread_label_distribution.py `
  --pyg-dir D:\thread_and_text\lite\pyg `
  --group "0:stock,1:thread,2:text,3:chamfer,4:fillet"
```

### Dataset path layout

- **Recommended:** `--dataset_path D:\thread_and_text\lite` plus **`--pt_subdir pyg`** so split files can live in `lite\` while graphs stay in `lite\pyg\`.
- **Also supported:** `--dataset_path D:\thread_and_text\lite\pyg` only — `CADSynth` will look for `train.txt` in the **parent** folder (`lite\`) when the dataset folder is named `pyg` (see [`data/dataset.py`](../../data/dataset.py) `_resolve_dataset_split_list`).

---

## 5. Stage 1 training (5-class, CE + distribution class weights)

Only change vs older 3-class runs: **`--num_classes 5`**, dataset path, 5-class weights JSON, and **`--run_name`**. Example:

```powershell
cd C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG
conda run -n brep_mfr_pyg python segmentation.py train `
  --dataset_path D:\thread_and_text\lite `
  --pt_subdir pyg `
  --num_classes 5 `
  --drop_invalid_graphs `
  --class_weights_path artifacts/class_weights/thread_text/source_train_5class_alpha05.json `
  --batch_size 1 `
  --accumulate_grad_batches 32 `
  --precision 16-mixed `
  --max_epochs 100 `
  --num_workers 0 `
  --log_every_n_steps 50 `
  --dropout 0.3 --attention_dropout 0.3 --act-dropout 0.3 `
  --d_model 512 --dim_node 256 --n_heads 32 --n_layers_encode 8 --warmup_freeze_epochs 3 `
  --loss_type ce `
  --run_name stock_thread_text_chamfer_fillet_lite_ce_weighted_exp1
```

Length-bucket / larger-batch style (adjust paths to your machine):

```powershell
conda run -n brep_mfr_pyg python segmentation.py train `
  --dataset_path Z:\thread_and_text\lite `
  --pt_subdir pyg `
  --num_classes 5 `
  --drop_invalid_graphs `
  --class_weights_path artifacts/class_weights/thread_text/source_train_5class_alpha05.json `
  --batch_size 8 --accumulate_grad_batches 2 `
  --precision 16-mixed `
  --max_epochs 100 --warmup_freeze_epochs 3 `
  --d_model 512 --dim_node 256 --n_heads 32 --n_layers_encode 8 `
  --num_workers 4 --pin_memory `
  --dropout 0.2 --attention_dropout 0.3 `
  --loss_type ce `
  --length_bucket_batching `
  --run_name stock_thread_text_chamfer_fillet_lite_ce_weighted_exp1
```

Tune `--batch_size` / `--accumulate_grad_batches` / `--precision` / `--max_graph_nodes` for VRAM like the [thread README](README.md) OOM section.

### Test

```powershell
conda run -n brep_mfr_pyg python segmentation.py test `
  --dataset_path Z:\thread_and_text\lite `
  --pt_subdir pyg `
  --num_classes 5 `
  --drop_invalid_graphs `
  --batch_size 4 `
  --num_workers 0 `
  --checkpoint results/stage1/<run_name>/best.ckpt
```

---

## 6. Recover a trained lite checkpoint by introducing A1 + A3

The `lite` profile used above contains **no A1, A2, or A3 tensors**. A trained lite checkpoint is still reusable because the model shape is unchanged, but its A1/A3 attention-bias modules were never exercised.

The safe recovery path is:

1. Keep the existing `lite` dataset and checkpoint unchanged.
2. Reconvert the same repaired JSON files into a separate `no_a2` dataset (`A1+A3`, no dense A2).
3. Copy the original split lists unchanged so the experiment has identical train/val/test membership.
4. Validate labels/topology against the lite graphs (**`--num-classes 5`**).
5. Start a **new fine-tuning run with `--pre_train`**, a low backbone LR, a higher A1/A3 LR, and a five-epoch A1/A3 contribution ramp.

Do **not** use the old lite checkpoint with `--resume_from_checkpoint` for this transition. Exact resume restores the late optimizer/scheduler state. Use `--pre_train` once to create a fresh fine-tuning run; use `--resume_from_checkpoint` only to resume that new run after an interruption.

### 6.1 Finish and preserve the lite run

If lite training is currently in the middle of an epoch, let the epoch finish so Lightning updates `last.ckpt`. Keep both `best.ckpt` and `last.ckpt`; start from `best.ckpt` unless the final epochs clearly improved the desired validation metrics.

### 6.2 Build and validate the A1+A3 dataset

**Do not re-convert from JSON for this step.** The lite `.pt` graphs already have the geometry, labels, and `edge_index`. The fast path upgrades those files in place into a separate `no_a2` tree by attaching A1 (`spatial_pos`) + A3 (`edge_path`) only.

```powershell
cd C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG

# Recommended (~12 file workers, NumPy BFS, resume-safe). Aim: ~48k graphs in well under 2 hours.
powershell -ExecutionPolicy Bypass -File scripts/threads/_archive/prepare_a1_a3_finetune.ps1 `
  -LiteRoot Z:\thread_and_text\lite `
  -OutputRoot Z:\thread_and_text\no_a2 `
  -FileWorkers 12
```

Direct Python equivalent:

```powershell
conda run -n brep_mfr_pyg python -u scripts/threads/upgrade_lite_pt_to_no_a2.py `
  --lite-root Z:\thread_and_text\lite `
  --output-root Z:\thread_and_text\no_a2 `
  --file-workers 12
```

Validate with 5 classes:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts/threads/validate_a1_a3_finetune_data.py `
  --dataset-root Z:\thread_and_text\no_a2 `
  --num-classes 5 `
  --report-a3-cap 768
```

### 6.3 Start the new fine-tuning run

Use the archived helper (or mirror its flags) with **`--num_classes 5`** and the 5-class weights file from step 4.

### 6.4 Test the recovered model

```powershell
conda run -n brep_mfr_pyg python segmentation.py test `
  --dataset_path Z:\thread_and_text\no_a2 `
  --pt_subdir pyg `
  --num_classes 5 `
  --drop_invalid_graphs `
  --batch_size 4 `
  --num_workers 0 `
  --max_nodes_for_a3 768 `
  --checkpoint results/stage1/<new-run-name>/best.ckpt
```

---

## Subgraph Training (k-hop neighborhoods) — Optional for Severe Imbalance

```powershell
conda run -n brep_mfr_pyg python segmentation.py train `
  --dataset_path D:\thread_and_text\lite `
  --pt_subdir pyg `
  --num_classes 5 `
  --drop_invalid_graphs `
  --class_weights_path artifacts/class_weights/thread_text/source_train_5class_alpha05.json `
  --batch_size 1 --accumulate_grad_batches 32 --precision 16-mixed `
  --max_epochs 100 --num_workers 0 `
  --loss_type focal `
  --subgraph_training --subgraph_k_hop 2 --subgraph_seeds_per_class "2,3,3,2,2" `
  --run_name fiveclass_subgraph_k2_s23322_$(Get-Date -Format 'yyyyMMdd_HHmmss')
```

`--subgraph_seeds_per_class "2,3,3,2,2"` = stock, thread, text, chamfer, fillet seed budgets per part.

---

## Related

- 2-class thread-only flow: [README.md](README.md)
- Generic label repair: [`repair_json_face_labels.py`](repair_json_face_labels.py)
- Class weights: [`scripts/training/compute_class_weights.py`](../training/compute_class_weights.py)
- Combined corpus stats: [`count_combined_label_distribution.py`](count_combined_label_distribution.py)
