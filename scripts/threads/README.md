# Thread identification dataset helpers

On Windows, use **`conda run -n brep_mfr_pyg python ...`** (or `conda activate brep_mfr_pyg`) so `torch_geometric` is available. Plain `python` may point to a base interpreter without PyG.

Use these after SolidWorks export so labels match BrepMFR Stage 1 (`--num_classes 2`).

**3-class thread + text** (stock / thread / text): see [README_thread_text.md](README_thread_text.md).

| Label in JSON | Meaning        | After repair        |
|---------------|----------------|---------------------|
| `-1`          | unknown/stock  | `0`                 |
| `0`           | stock          | `0`                 |
| `70`          | thread         | `1` (recommended)   |

## 1. Fix JSON labels

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
# Inspect only
conda run -n brep_mfr_pyg python scripts/threads/repair_thread_json_labels.py --json-dir D:/threads/json --dry-run

# Apply -1 to 0 and 70 to 1 without prompts
conda run -n brep_mfr_pyg python scripts/threads/repair_thread_json_labels.py --json-dir D:/threads/json --yes-minus-one --yes-remap-70
```

Then regenerate PyG (or run conversion once on clean labels).

**Why one-line JSON?** Earlier, `repair_thread_json_labels.py` used compact `json.dumps(..., separators=(",", ":"))` for speed and smaller files. It now writes **`indent=2`** so new repairs stay readable. To re-pretty **already** minified files (no label edits):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/prettify_thread_json.py --json-dir D:/threads/json
```

## 2. JSON → PyG + sidecar label JSON

```powershell
# Long-running (~hours for tens of thousands of JSON files).
conda run -n brep_mfr_pyg python scripts/inference/json_to_brepmfr_pyg_optimized.py `
  --json_dir D:/threads/json `
  --pt_out_dir D:/threads/pyg `
  --label_out_dir D:/threads/label `
  --spatial_pos_max 32 `
  --inference_profile no_a2 `
  --shortest_path_workers 8
```

When conversion finishes, run splits + class weights + label recount in one step:

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
powershell -ExecutionPolicy Bypass -File scripts/threads/post_pyg_export.ps1
```

If training fails with **empty `label_feature` (zero faces)** on a model, list bad checkpoints:

```powershell
conda run -n brep_mfr_pyg python scripts/threads/find_empty_face_graphs.py --scan-root D:/threads/lite/pyg
```

Then remove those stems from `train.txt` / `val.txt` / `test.txt` (or fix the JSON and re-export).

## 3. Class distribution

```powershell
conda run -n brep_mfr_pyg python scripts/threads/count_thread_label_distribution.py --json-dir D:/threads/json
conda run -n brep_mfr_pyg python scripts/threads/count_thread_label_distribution.py --pyg-dir D:/threads/pyg
```

## 4. Splits + Stage 1

`post_pyg_export.ps1` already runs `make_random_splits.py` and `compute_class_weights.py`. If you prefer manual steps:

```powershell
conda run -n brep_mfr_pyg python scripts/threads/make_random_splits.py --pyg-dir D:/threads/pyg --out-dir D:/threads/pyg

conda run -n brep_mfr_pyg python scripts/training/compute_class_weights.py `
  --dataset_path D:/threads/pyg --split train --num_classes 2 --alpha 0.5 `
  --num_workers 0 `
  --out artifacts/class_weights/thread/source_train_alpha05.json
```

## Stage 1 training command (run yourself)

Same backbone as the locked MFTR Stage 1 (`d_model=512`, `dim_node=256`, eight encoder layers), but **2 classes** and your **thread** dataset path. Uses `num_workers 0` on Windows.

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
conda run -n brep_mfr_pyg python segmentation.py train `
  --dataset_path D:/threads/pyg `
  --num_classes 2 `
  --drop_invalid_graphs `
  --class_weights_path artifacts/class_weights/thread/source_train_alpha05.json `
  --batch_size 32 `
  --max_epochs 100 `
  --num_workers 0 `
  --log_every_n_steps 50 `
  --dropout 0.3 --attention_dropout 0.3 --act-dropout 0.3 `
  --d_model 512 --dim_node 256 --n_heads 32 --n_layers_encode 8 --warmup_freeze_epochs 3 `
  --run_name thread_ce_weighted__manual
```

Optional: set `--run_name` to something unique; add `--resume_from_checkpoint ...` if resuming.

### CUDA out of memory (large graphs)

Self-attention cost scales with **max face count in the batch** (padding to the largest graph). With `--batch_size 32` and some very large SolidWorks B-reps, VRAM can exceed 80–96 GiB.

Try **in order**:

0. **Skip bad exports** (zero faces / empty `label_feature`): `--drop_invalid_graphs` (one-time scan at startup; avoids mid-epoch `ValueError`).
1. **Micro-batching** (same effective batch size): `--batch_size 1 --accumulate_grad_batches 32`
2. **Mixed precision**: `--precision 16-mixed` (or `bf16-mixed` on supported GPUs)
3. **Cap graph size** (drops huge models at dataset init): e.g. `--max_graph_nodes 768` (tune to your GPU), optionally with a slightly larger `--batch_size` once stable. This pass also drops invalid graphs (same as `--drop_invalid_graphs` alone).

Example combining (1) and (2):

```powershell
conda run -n brep_mfr_pyg python segmentation.py train `
  --dataset_path D:/threads/lite/pyg `
  --num_classes 2 `
  --drop_invalid_graphs `
  --class_weights_path artifacts/class_weights/thread/lite_source_train_alpha05.json `
  --batch_size 1 `
  --accumulate_grad_batches 32 `
  --precision 16-mixed `
  --max_epochs 100 `
  --num_workers 0 `
  --log_every_n_steps 50 `
  --dropout 0.3 --attention_dropout 0.3 --act-dropout 0.3 `
  --d_model 512 --dim_node 256 --n_heads 32 --n_layers_encode 8 --warmup_freeze_epochs 3 `
  --run_name thread_lite_ce_memsafe
```

## 5. PyG inference (test split)

Batch inference on ``test.txt`` with per-face CSVs plus ``metrics_test/`` (confusion matrix, per-class report):

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
conda run -n brep_mfr_pyg python scripts/threads/run_thread_pyg_inference.py `
  --checkpoint results/stage1/thread_lite_ce_weighted_exp1_memsafe/best.ckpt `
  --dataset_path D:/threads/lite `
  --split test `
  --batch_size 4 `
  --device cuda
```

Outputs:

- ``D:/threads/lite/inference_test/*.csv`` — one row per face (pred + GT when available)
- ``D:/threads/lite/metrics_test/`` — ``confusion_matrix.csv``, ``per_class.csv``, ``summary.md``

Arbitrary ``*.pt`` folder (no split list):

```powershell
conda run -n brep_mfr_pyg python scripts/threads/run_thread_pyg_inference.py `
  --checkpoint results/stage1/thread_lite_ce_weighted_exp1_memsafe/best.ckpt `
  --pyg_dir D:/threads/lite/pyg `
  --inference_dir D:/threads/lite/inference_manual
```
