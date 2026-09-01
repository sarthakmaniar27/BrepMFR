# Standalone scripts — recurring workflow guide

For the full operator loop (VBA JSON export → cleanup → infer → filter → Jenkins
allowlist ship), see [END_TO_END_PIPELINE.md](END_TO_END_PIPELINE.md).

Run all commands from the **repo root** (`BrepMFR_PyG`), unless a script is edited and run directly.

Shared matching key used by most cleanup scripts:

```text
JSON:  00000001_..._step_000_101.json
STEP:  00000001_..._step_000.step
Key:   00000001_..._step_000
```

Many JSONs (`_101`, `_102`, …) can share one STEP key. That is expected.

---

## Task set 1 — Clean duplicate JSONs in the JSON folder

**Goal:** In `C:\jsons`, keep one JSON per STEP key and remove SolidWorks temp `.SLDPRT` files.

| Script | Role |
|--------|------|
| `delete_duplicate_jsons.py` | Keep the lexicographically first JSON per key (usually `*_101.json`); delete extras + all `*.SLDPRT`. |

**Default paths (hardcoded in script):**

- JSON folder: `C:\jsons`

**Safety:** Open the script and check `DRY_RUN`:

- `DRY_RUN = True` → print what would be deleted; no deletes
- `DRY_RUN = False` → perform deletes

**Run:**

```powershell
python standalone_scripts/delete_duplicate_jsons.py
```

**What it does:**

1. Deletes leftover `*.SLDPRT` temps in the JSON folder
2. Groups `*.json` by `..._step_NNN`
3. Keeps the first name per group; deletes the rest (`*_102`, `*_103`, …)

**Optional check first (counts only, no deletes):**

```powershell
python standalone_scripts/count_unique_json_vs_steps.py
```

Compares unique keys in `C:\jsons` vs STEPs in `\\GR-SW65551\abc_steps`.

---

## Task set 2 — Delete STEPs that already have JSONs

**Goal:** From the STEP root folder, remove STEP files whose key already exists in the JSON folder (so you only keep STEPs still waiting for JSON export).

Pick the script that matches **your** JSON + STEP folders:

| Script | JSON folder | STEP folder | Notes |
|--------|-------------|-------------|-------|
| `delete_step_files.py` | `C:\jsons` | `\\GR-SW65551\abc_steps` | Main daily path after SolidWorks export into `C:\jsons` |
| `delete_step_files_from_abc_jsons.py` | `\\GR-SW26859\abc` | `\\GR-SW65551\abc_steps` | Same idea when JSONs live on the `abc` share |
| `check_and_delete_covered_steps.py` | `C:\jsons` (or `--json-dir`) | `C:\abc_steps_not_in_allowlist` (or `--step-dir`) | CLI + `--dry-run`; for the “not in allowlist” STEP copy |

### 2A — Default: `C:\jsons` ↔ `abc_steps`

**Safety:** In `delete_step_files.py`, set `DRY_RUN = True` for a preview, then `False` to delete.

```powershell
python standalone_scripts/delete_step_files.py
```

Deletes every `.step` / `.stp` in `abc_steps` whose key appears in `C:\jsons`.

### 2B — JSONs on the `abc` share

```powershell
python standalone_scripts/delete_step_files_from_abc_jsons.py
```

Same matching logic; paths are hardcoded to `\\GR-SW26859\abc` and `\\GR-SW65551\abc_steps`. Use `DRY_RUN` the same way.

### 2C — Coverage check + delete on a local STEP copy

```powershell
# Preview only
python standalone_scripts/check_and_delete_covered_steps.py --dry-run

# Delete covered STEPs
python standalone_scripts/check_and_delete_covered_steps.py

# Custom folders
python standalone_scripts/check_and_delete_covered_steps.py `
  --json-dir C:\jsons `
  --step-dir C:\abc_steps_not_in_allowlist `
  --dry-run
```

---

## Task set 3 — Run inference only on newly generated JSONs

**Goal:** Infer only JSONs that do not already have a predictions CSV (skip work already done).

### Option A — Lite ONNX, JSON → CSV in one shot (recommended for this loop)

| Script | Role |
|--------|------|
| `run_onnx_json_batch_inference.py` | Convert each top-level JSON → lite PyG → run lite ONNX → write CSV under `<json-dir>\inference\` |

**Key flag:** `--skip-existing` skips any JSON that already has `<stem>_predictions.csv` in the inference folder.

```powershell
conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_json_batch_inference.py `
  --json-dir C:\jsons `
  --skip-existing
```

Useful extras:

```powershell
# Smoke test on a few files
conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_json_batch_inference.py `
  --json-dir C:\jsons --max-files 5 --skip-existing --provider cpu

# Only re-scan existing CSVs (Stage-2 Thread/Text filter); no new inference
conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_json_batch_inference.py `
  --json-dir C:\jsons --stage2-only
```

Defaults:

- Model dir: `standalone_scripts/BrepMFR_lite_onnx_pyg_demo_v2`
- Output: `C:\jsons\inference\` (or `--inference-dir`)
- Stage-2 threshold for Thread/Text: `--confidence 0.80`

### Option B — A1+A3 / no_a2 model (newer Thread+Text ONNX)

This path needs **PyG `.pt` graphs** first, then ONNX. It does not take raw JSONs directly.

```powershell
# 1) Convert only the new JSONs into no_a2 graphs
conda run -n brep_mfr_pyg python scripts/inference/json_to_brepmfr_pyg_optimized.py `
  --json_dir C:\jsons `
  --output_dir <your_pyg_dataset_root> `
  --inference_profile no_a2

# 2) Run A1+A3 ONNX on the dataset (or a flat folder of .pt via --input)
conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_a1_a3_inference.py `
  --dataset-path <your_pyg_dataset_root> `
  --output-dir <csv_output_dir>
```

Default model package when using the migration export:

- `migration_to_c++/migration_to_c/no_a2_72k_epoch50_onnx/` (or `exported_a1_a3/`)

Pass `--onnx` / `--label-map` if you need a specific package (see script `--help`).

---

## Suggested daily loop (all three tasks)

Typical order after SolidWorks batch-exports new JSONs into `C:\jsons`:

```text
1) delete_duplicate_jsons.py          # clean multi-body / SLDPRT junk in C:\jsons
2) delete_step_files.py               # drop STEPs that already have JSONs
3) run_onnx_json_batch_inference.py --skip-existing   # infer only new JSONs
```

PowerShell (after setting `DRY_RUN = False` in the cleanup scripts when ready):

```powershell
python standalone_scripts/delete_duplicate_jsons.py
python standalone_scripts/delete_step_files.py
conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_json_batch_inference.py `
  --json-dir C:\jsons --skip-existing
```

---

## Related scripts (same folder, other workflows)

| Script | When to use |
|--------|-------------|
| `count_unique_json_vs_steps.py` | Report-only coverage of JSON keys vs `abc_steps` |
| `delete_jsons_on_allowlist.py` | Delete JSONs **not** on the no-Thread/Text allowlist (`--dry-run` supported) |
| `copy_steps_not_in_allowlist.py` | Copy STEPs for JSON keys outside the allowlist into a local folder |
| `match_allowlist_vs_jsons_folder.py` | Report allowlist vs JSON-folder key overlap |
| `filter_abc_steps_by_allowlist.py` / `Filter-AbcStepsByAllowlist.ps1` | Filter STEP trees by allowlist (Jenkins-friendly) |
| `export_step_allowlist_from_inference.py` | Build allowlist files from inference outputs |
| `run_onnx_step_inference.py` | Infer directly from STEP (no JSON); uses occwl/pythonocc |
| `run_onnx_pyg_inference.py` / `_v2.py` | Older lite / 3-class runners on `.pt` graphs |
| `BatchJsonExport.vba` / `BatchInference.vba` | SolidWorks-side batch export / inference helpers |

---

## Important notes

1. **Hardcoded paths:** `delete_duplicate_jsons.py`, `delete_step_files.py`, and `delete_step_files_from_abc_jsons.py` use paths inside the file. Edit those constants if your folders differ.
2. **Always dry-run deletes first** (`DRY_RUN = True` or `--dry-run`) on a large share.
3. **`--skip-existing` only looks at CSV presence**, not whether the JSON changed. Re-exporting a JSON with the same stem will be skipped until you remove its CSV.
4. Cleanup scripts are **not** on the Stage-1 training import path; they are ops utilities around export + inference.
