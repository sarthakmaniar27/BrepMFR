# Thread + text (3-class) pipeline

SolidWorks-style face labels → BrepMFR Stage 1 with **`num_classes=3`**:

| Meaning | Raw `label` in JSON | After remap |
|---------|---------------------|-------------|
| Stock | `0`, `-1`, `-10` | `0` |
| Thread | `70` | `1` |
| Text (emboss) | `101` | `2` |

Use **`conda run -n brep_mfr_pyg python ...`** on Windows if base Python lacks PyG.

## 1. Repair / remap JSON labels (data-driven map)

Map file (repo): [`remap_maps/thread_text_sw_to_brep.json`](remap_maps/thread_text_sw_to_brep.json).

Inspect only:

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
conda run -n brep_mfr_pyg python scripts/threads/repair_json_face_labels.py `
  --json-dir D:\thread_and_text\root_json `
  --map-json scripts/threads/remap_maps/thread_text_sw_to_brep.json `
  --dry-run
```

Optional strict check (exit 1 if any label is not in the map):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/repair_json_face_labels.py `
  --json-dir D:\thread_and_text\root_json `
  --map-json scripts/threads/remap_maps/thread_text_sw_to_brep.json `
  --dry-run --fail-on-unknown
```

Apply remaps and rewrite JSON (`indent=2`):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/repair_json_face_labels.py `
  --json-dir D:\thread_and_text\root_json `
  --map-json scripts/threads/remap_maps/thread_text_sw_to_brep.json `
  --yes-write
```

Also remap **`abc_jsons`** the same way (same map includes `-10` → `0`):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/repair_json_face_labels.py `
  --json-dir D:\thread_and_text\abc_jsons `
  --map-json scripts/threads/remap_maps/thread_text_sw_to_brep.json `
  --yes-write
```

Writes are refused if any face label is missing from the map (unless you pass **`--allow-unmapped`**, not recommended).

Future class sets: add a new JSON map under `remap_maps/` and point `--map-json` at it (no code change).

## 2. Class distribution
**JSON** (raw or after repair):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/count_thread_label_distribution.py `
  --json-dir D:\thread_and_text\root_json `
  --group "0:stock,1:thread,2:text"
```

**PyG** (after conversion):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/count_thread_label_distribution.py `
  --pyg-dir D:\thread_and_text\lite\pyg `
  --group "0:stock,1:thread,2:text"
```

## 3. JSON → PyG (lite)

Long-running. Adjust workers / paths to your machine.

Primary folder only:

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
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

Or edit paths inside the post-export script and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/threads/post_thread_text_pyg_export.ps1
```

Writes `train.txt` / `val.txt` / `test.txt` under **`lite/`** (parent of `lite/pyg/`), and class weights using that same root so split lists and `rglob` of `*.pt` stay consistent.

`artifacts/class_weights/thread_text/source_train_alpha05.json` (`--num_classes 3`, `--alpha 0.5`).

### Dataset path layout

- **Recommended:** `--dataset_path D:\thread_and_text\lite` plus **`--pt_subdir pyg`** so split files can live in `lite\` while graphs stay in `lite\pyg\`.
- **Also supported:** `--dataset_path D:\thread_and_text\lite\pyg` only — `CADSynth` will look for `train.txt` in the **parent** folder (`lite\`) when the dataset folder is named `pyg` (see [`data/dataset.py`](../../data/dataset.py) `_resolve_dataset_split_list`).

## 5. Stage 1 training (same hyperparameters as 2-class thread run)

Only change: **`--num_classes 3`**, dataset path, class-weights path, and **`--run_name`**. Example:

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
conda run -n brep_mfr_pyg python segmentation.py train `
  --dataset_path D:\thread_and_text\lite `
  --pt_subdir pyg `
  --num_classes 3 `
  --drop_invalid_graphs `
  --class_weights_path artifacts/class_weights/thread_text/source_train_alpha05.json `
  --batch_size 1 `
  --accumulate_grad_batches 32 `
  --precision 16-mixed `
  --max_epochs 100 `
  --num_workers 0 `
  --log_every_n_steps 50 `
  --dropout 0.3 --attention_dropout 0.3 --act-dropout 0.3 `
  --d_model 512 --dim_node 256 --n_heads 32 --n_layers_encode 8 --warmup_freeze_epochs 3 `
  --run_name thread_text_lite_ce_weighted_exp1
```

Tune `--batch_size` / `--accumulate_grad_batches` / `--precision` / `--max_graph_nodes` for VRAM like the [thread README](README.md) OOM section.

## 6. Recover a trained lite checkpoint by introducing A1 + A3

The `lite` profile used above contains **no A1, A2, or A3 tensors**. A trained lite checkpoint is still reusable because the model shape is unchanged, but its A1/A3 attention-bias modules were never exercised.

The safe recovery path is:

1. Keep the existing `lite` dataset and checkpoint unchanged.
2. Reconvert the same repaired JSON files into a separate `no_a2` dataset (`A1+A3`, no dense A2).
3. Copy the original split lists unchanged so the experiment has identical train/val/test membership.
4. Validate labels/topology against the lite graphs.
5. Start a **new fine-tuning run with `--pre_train`**, a low backbone LR, a higher A1/A3 LR, and a five-epoch A1/A3 contribution ramp.

Do **not** use the old lite checkpoint with `--resume_from_checkpoint` for this transition. Exact resume restores the late optimizer/scheduler state. Use `--pre_train` once to create a fresh fine-tuning run; use `--resume_from_checkpoint` only to resume that new run after an interruption.

### 6.1 Finish and preserve the lite run

If lite training is currently in the middle of an epoch, let the epoch finish so Lightning updates `last.ckpt`. Keep both `best.ckpt` and `last.ckpt`; start from `best.ckpt` unless the final epochs clearly improved the desired validation metrics.

### 6.2 Build and validate the A1+A3 dataset

**Do not re-convert from JSON for this step.** The lite `.pt` graphs already have the geometry, labels, and `edge_index`. The fast path upgrades those files in place into a separate `no_a2` tree by attaching A1 (`spatial_pos`) + A3 (`edge_path`) only.

```powershell
cd C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG

# Recommended (~12 file workers, NumPy BFS, resume-safe). Aim: ~48k graphs in well under 2 hours.
powershell -ExecutionPolicy Bypass -File scripts/threads/prepare_a1_a3_finetune.ps1 `
  -LiteRoot Z:\thread_and_text\lite `
  -OutputRoot Z:\thread_and_text\no_a2 `
  -FileWorkers 12
```

What changed vs the old JSON converter:

| Old (slow) | New (fast) |
|---|---|
| Re-parse every JSON + rebuild UV tensors | Load existing `lite/pyg/*.pt` |
| Per-hop `torch` cell writes (~60s on N≈700) | NumPy all-pairs BFS (~1s on N≈700) |
| `--shortest_path_workers 8` spawned a process pool **per graph** (deadly on Windows) | One persistent pool of **file** workers; serial BFS inside each |
| ~15s/file → days for 40k | Target: minutes–under 2h for ~48k on a 12-core machine |

The prepare script defaults to this lite-upgrade path whenever `LiteRoot\pyg` exists. Pass `-FromJson` only if you truly need a from-scratch JSON rebuild (and keep `-ShortestPathWorkers 0`).

Direct Python equivalent:

```powershell
conda run -n brep_mfr_pyg python -u scripts/threads/upgrade_lite_pt_to_no_a2.py `
  --lite-root Z:\thread_and_text\lite `
  --output-root Z:\thread_and_text\no_a2 `
  --file-workers 12
```

Existing output `.pt` files are skipped (resume-safe). If the machine previously OOMed, open Task Manager and end leftover `python.exe` workers before restarting. If RAM is tight, drop to `-FileWorkers 6`.

For a quick validation smoke before scanning every graph, add `-ValidationMaxFiles 100`.

### 6.3 Start the new fine-tuning run

```powershell
powershell -ExecutionPolicy Bypass -File scripts/threads/train_a1_a3_from_lite.ps1 `
  -Checkpoint "C:\Users\RZA2\thread_project\BrepMFR\results\stage1\thread_text_lite_abc_jsons\best.ckpt" `
  -DatasetRoot Z:\thread_and_text\no_a2 `
  -MaxEpochs 30 `
  -MaxNodesForA3 768
```

The script preserves the original architecture and uses:

- pretrained backbone/classifier LR: `1e-4`;
- previously unused graph-attention-bias LR: `1e-3`;
- optimizer warmup: 1000 steps, applied before the first optimizer update;
- A1/A3 contribution: `0.1 → 1.0` over epochs 0–4;
- `--warmup_freeze_epochs 0` because freezing the encoder would also freeze the new A1/A3 modules;
- A3 cap: batches above 768 padded faces skip A3 before its dense tensor is collated, while A1 remains active.

The 768 cap protects memory; lower it if A3 still causes OOM. Set `-MaxNodesForA3 0` only when the GPU and host RAM can handle dense A3 for the largest graph.

### 6.4 Test the recovered model

```powershell
conda run -n brep_mfr_pyg python segmentation.py test `
  --dataset_path Z:\thread_and_text\no_a2 `
  --pt_subdir pyg `
  --num_classes 3 `
  --drop_invalid_graphs `
  --batch_size 4 `
  --num_workers 0 `
  --max_nodes_for_a3 768 `
  --checkpoint results/stage1/<new-run-name>/best.ckpt
```

Compare this result with the preserved lite checkpoint on the lite test set. Keep the A1+A3 model only if per-class precision/recall and macro metrics improve; adding structural bias is useful context, not a guaranteed improvement.

## Subgraph Training (k-hop neighborhoods) — Recommended for Severe Imbalance

The biggest lever against "text walls drowning thread signals" is to stop training on whole graphs.

Instead of feeding a 4000-face part, sample a handful of seeds (balanced across classes) and train only on their local 2-hop or 3-hop neighborhoods.

**This is fully opt-in and 100 % backward compatible.** Omit the flags and you get the exact old full-graph workflow.

### Quick start (Stage 1)

```powershell
conda run -n brep_mfr_pyg python segmentation.py train `
  --dataset_path D:\thread_and_text\merged_lite `
  --pt_subdir pyg `
  --num_classes 3 `
  --drop_invalid_graphs `
  --class_weights_path artifacts/class_weights/thread_text/source_train_alpha05.json `
  --batch_size 1 --accumulate_grad_batches 32 --precision 16-mixed `
  --max_epochs 100 --num_workers 0 `
  --loss_type focal `
  --subgraph_training --subgraph_k_hop 2 --subgraph_seeds_per_class "2,3,3" `
  --run_name thread_text_subgraph_k2_s233_$(Get-Date -Format 'yyyyMMdd_HHmmss')
```

What the flags mean:
- `--subgraph_training` — turn the feature on (default off = full graphs).
- `--subgraph_k_hop 2` — take faces reachable in ≤2 adjacency hops from each seed (sweet spot).
- `--subgraph_seeds_per_class "2,3,3"` — per original CAD part, draw at most 2 stock + 3 thread + 3 text seeds (if present). The model therefore sees a *balanced number of seeds*, not a balanced number of faces dictated by feature size.
- Validation stays on full graphs by default (good for comparable metrics). Add `--subgraph_on_val` only if you want to experiment.

Because each original part now contributes several small, class-balanced "views", the gradient sees far more thread signal per epoch and text no longer dominates by sheer face count.

You can go back to the previous behavior at any moment by deleting the three `--subgraph_*` flags from the command.

### How it interacts with everything else
- Class weights and Focal Loss still apply inside the subgraphs (they are just smaller).
- Random rotation is applied to the whole part first, then we crop — local geometry stays correctly oriented.
- The epoch counter is advanced automatically so the same part yields different random subgraphs on epoch N vs N+1.
- All checkpoints, LR scheduling on `per_class_accuracy`, TensorBoard extras, etc. continue to work.

## Related

- 2-class thread-only flow: [README.md](README.md)
- Generic label repair implementation: [`repair_json_face_labels.py`](repair_json_face_labels.py)
- **Combined corpus stats** (thread-only + thread+text PyG dirs): [`count_combined_label_distribution.py`](count_combined_label_distribution.py)


 python segmentation.py train `  --dataset_path Z:\thread_and_text\lite `  --pt_subdir pyg `  --num_classes 3 `  --drop_invalid_graphs `  --class_weights_path C:\Users\RZA2\thread_project\BrepMFR\artifacts\class_weights\thread_text\source_train_alpha05.json `  --batch_size 8 --accumulate_grad_batches 2 `  --precision 16-mixed `  --max_epochs 100 --warmup_freeze_epochs 3 `  --d_model 512 --dim_node 256 --n_heads 32 --n_layers_encode 8 `  --num_workers 4 --pin_memory `  --dropout 0.2 --attention_dropout 0.3 ` --loss_type ce `  --length_bucket_batching `  --run_name thread_text_new_macro_good_balance_ce_weighted_exp1



 python segmentation.py test `
  --dataset_path Z:\thread_and_text\lite `
  --pt_subdir pyg `
  --num_classes 3 `
  --drop_invalid_graphs `
  --batch_size 4 `
  --num_workers 0 `
  --checkpoint C:\Users\RZA2\thread_project\BrepMFR\results\stage1\thread_text_new_macro_good_balance_ce_weighted_exp1\last.ckpt





  SETUP NEW MACHINE:
  1) IF NEW IMAGE, NEW USER AND REGISTER IT
  2) SETUP THE THREADS FOLDER
  3) ADD THOSE NODES TO JENKINS
  4) .NET GO TO THE FOLDER AND RUN BOTH .EXES
  5) RUN THE SLDPRT FILDE IF NEEDED (CONTEXT IN CHAT)
  6) SETUP MACRO AND CLIS (MODIFY NEW MACRO PATHS)
  7) CREATE JSONS FROM STEP FIKES
  8) DELTE PREVIOUS FILES IN ROOT_JSON OR RENAME THE FODLER, AND THEN COLLECT ALL JSONS FORM VMS
  9) THEN FOLLOW THE TRAINNG PROCEDURE INT HE README_THREAD_TEXT.MD FILE
  10) Data Stats: 36K CADSYNTH 2KGEARS 10K ABC