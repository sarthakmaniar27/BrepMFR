# Thread + text (3-class) pipeline

SolidWorks-style face labels → BrepMFR Stage 1 with **`num_classes=3`**:

| Meaning | Raw `label` in JSON | After remap |
|---------|---------------------|-------------|
| Stock | `0`, `-1` | `0` |
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

## 4. Splits + class weights + recount

Edit paths inside the script if your dirs differ, then:

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
