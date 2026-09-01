# ABC STEP → JSON → filter → Jenkins pipeline

End-to-end operator workflow for generating B-rep JSONs from ABC STEPs,
recovering from interrupted SolidWorks macros, running lite ONNX inference,
filtering out confident Thread/Text parts, then shipping allowlisted STEPs via
Jenkins.

Run Python commands from the **repo root** (`BrepMFR_PyG`) unless noted.

```text
ABC STEPs
  -> [1] SolidWorks VBA BatchJsonExport
  -> C:\jsons\*.json
  -> [2/3] cleanup if interrupted / always before next batch
  -> [4] lite ONNX inference (--skip-existing)
  -> [5] Stage-2 filter (no confident Thread/Text)
  -> allowlist of STEP keys
  -> [6] Jenkins / local filter: C:\abc_steps -> C:\abc_steps_filtered
```

Shared STEP key (used everywhere):

```text
JSON:  00000001_..._step_000_101.json
STEP:  00000001_..._step_000.step
Key:   00000001_..._step_000
```

---

## Folder map (defaults baked into the scripts)

| Path | Role |
|------|------|
| `C:\abc_steps` | Per-machine STEP pool (Jenkins source) |
| `C:\abc_steps_not_in_allowlist` | STEPs still needing JSON export (VBA input) |
| `C:\jsons` | Macro output: raw B-rep JSONs |
| `C:\jsons\batch_logs\` | VBA logs: `batch_log.txt`, `skip_list.txt`, `in_progress.txt` |
| `C:\jsons\Watchdog-StepOpen.ps1` | Copy of watchdog used by the macro |
| `C:\jsons\inference\` | Prediction CSVs + Stage-2 filter lists |
| `C:\abc_steps_filtered` | Jenkins / filter destination (allowlisted STEPs) |
| `\\GR-SW65551\abc_steps` | Shared STEP root used by some cleanup scripts |

---

## Step 1 — Generate new JSONs (SolidWorks VBA)

**Script:** `standalone_scripts/BatchJsonExport.vba`

**What it does:** Opens each `.step` in the input folder, exports B-rep JSON(s) to
`C:\jsons`, closes without saving. Uses an external watchdog because `LoadFile4`
cannot be cancelled from VBA.

| Setting in VBA | Default |
|----------------|---------|
| Input folder | `C:\abc_steps_not_in_allowlist\` (trailing `\` required) |
| Output folder | `C:\jsons` |
| Open timeout | `OPEN_TIMEOUT_SEC = 60` |
| Max STEP size | `MAX_STEP_BYTES = 80_000_000` (0 = no limit) |
| Watchdog | `C:\jsons\Watchdog-StepOpen.ps1` |

**Before first run:**

1. Copy `standalone_scripts/Watchdog-StepOpen.ps1` → `C:\jsons\Watchdog-StepOpen.ps1`
2. Confirm input folder exists and ends with `\`
3. Import / run `main` in SolidWorks

**Related (different job):** `BatchInference.vba` opens STEPs on
`\\GR-SW65551\abc_steps` and runs native inference command `100050`. That is
**not** the JSON-export path in this pipeline.

**Optional prep of VBA input folder** (STEPs whose JSON keys are outside the
current no-Thread/Text allowlist):

```powershell
python standalone_scripts/copy_steps_not_in_allowlist.py --dry-run
python standalone_scripts/copy_steps_not_in_allowlist.py
```

Defaults: JSON keys from `E:\jsons_from_all_machines` (or `--json-dir`), STEPs
from `\\GR-SW65551\abc_steps`, dest `C:\abc_steps_not_in_allowlist`.

---

## Step 2 — If the macro was interrupted

The macro may die mid-open (watchdog kill), crash, or be stopped manually.

**Automatic recovery already in the VBA:**

1. Before each open, writes the STEP name to `C:\jsons\batch_logs\in_progress.txt`
2. Starts `Watchdog-StepOpen.ps1`; if still in progress after timeout → append to
   `skip_list.txt` and kill SolidWorks
3. On next `main` start, `PromoteInProgressToSkip` moves any leftover
   `in_progress` name into `skip_list.txt` so that part is not retried

**What you do after an interrupt:**

1. Re-launch SolidWorks if it was killed
2. Run **Step 3** cleanup (duplicates + covered STEPs)
3. Re-run `BatchJsonExport.vba` — skipped / previously finished STEPs will not be
   re-exported once you have deleted covered STEPs from the input folder

Check logs:

```text
C:\jsons\batch_logs\batch_log.txt
C:\jsons\batch_logs\skip_list.txt
C:\jsons\batch_logs\in_progress.txt
```

---

## Step 3 — Cleaning (duplicates + delete covered STEPs)

Do this after every interrupted (or completed) export batch before starting a
new one.

### 3A — Clean duplicate / temp files in `C:\jsons`

**Script:** `delete_duplicate_jsons.py`

- Deletes leftover `*.SLDPRT` temps
- Keeps one JSON per STEP key (lexicographically first, usually `*_101.json`)
- Deletes extra body JSONs (`*_102`, …)

```powershell
# Edit DRY_RUN = True first for a preview, then False to delete
python standalone_scripts/delete_duplicate_jsons.py
```

### 3B — Delete STEPs that already have JSONs

So the STEP backlog only contains parts still waiting for export.

| Script | Use when |
|--------|----------|
| `delete_step_files.py` | JSON=`C:\jsons`, STEP=`\\GR-SW65551\abc_steps` |
| `check_and_delete_covered_steps.py` | JSON=`C:\jsons`, STEP=`C:\abc_steps_not_in_allowlist` (VBA input) — preferred for this pipeline |
| `delete_step_files_from_abc_jsons.py` | JSON=`\\GR-SW26859\abc`, STEP=`\\GR-SW65551\abc_steps` |

For the VBA input folder:

```powershell
python standalone_scripts/check_and_delete_covered_steps.py --dry-run
python standalone_scripts/check_and_delete_covered_steps.py `
  --json-dir C:\jsons `
  --step-dir C:\abc_steps_not_in_allowlist
```

Optional coverage report (shared `abc_steps`):

```powershell
python standalone_scripts/count_unique_json_vs_steps.py
```

**Safety:** Prefer `--dry-run` / `DRY_RUN = True` on the first pass.

---

## Step 4 — Inference on new JSONs only

When enough new JSONs exist in `C:\jsons`, run lite ONNX and **skip** anything
that already has a predictions CSV.

**Script:** `run_onnx_json_batch_inference.py`

```powershell
conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_json_batch_inference.py `
  --json-dir C:\jsons `
  --skip-existing
```

| Flag | Meaning |
|------|---------|
| `--skip-existing` | Skip Stage-1 if `<stem>_predictions.csv` already exists under `inference/` |
| `--max-files N` | Smoke test limit |
| `--provider cpu\|cuda\|auto` | ONNX Runtime provider |
| `--confidence 0.80` | Stage-2 Thread/Text threshold (also used in Step 5) |

**Outputs (Stage 1):**

- `C:\jsons\inference\<stem>_predictions.csv`
- `C:\jsons\inference\onnx_json_inference_summary.csv`

Model default: `standalone_scripts/BrepMFR_lite_onnx_pyg_demo_v2/brepmfr_lite.onnx`

---

## Step 5 — Filtering check on (new) inference results

Stage-2 runs **automatically** at the end of Step 4. To re-run filter only
(no new inference):

```powershell
conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_json_batch_inference.py `
  --json-dir C:\jsons `
  --stage2-only
```

**Rule:** A JSON is flagged as “clean” (keep / allowlist candidate) if **no face**
has `prob_Thread` or `prob_Text` above `--confidence` (default `0.80`).

**Outputs (Stage 2):**

| File | Contents |
|------|----------|
| `C:\jsons\inference\no_confident_thread_or_text.csv` | Flagged JSONs + max Thread/Text probs |
| `C:\jsons\inference\no_confident_thread_or_text.txt` | One JSON path per line |

These flagged JSONs are the filter result for this pipeline. Parts with confident
Thread/Text stay out of the allowlist.

Optional cross-check against an existing allowlist:

```powershell
python standalone_scripts/match_allowlist_vs_jsons_folder.py
```

---

## Step 6 — Build allowlist and send via Jenkins

Inference filter produces JSON paths. Downstream shipping uses a **STEP-key
allowlist** derived from those paths, then copies matching STEPs to a filtered
folder on each machine.

### 6A — Build allowlist from Stage-2 output

**Script:** `export_step_allowlist_from_inference.py`

```powershell
python standalone_scripts/export_step_allowlist_from_inference.py
```

Defaults:

- Reads: `C:\jsons\inference\no_confident_thread_or_text.txt` (and/or `.csv`)
- Writes: `C:\jsons\inference\allowed_step_keys.txt`
- Also refreshes repo copies: `standalone_scripts/allowed_step_keys.txt` and
  `allowed_step_keys_p1.txt` … `_p3.txt`

### 6B — Local dry-run / single-machine copy (optional)

**PowerShell:**

```powershell
.\standalone_scripts\Filter-AbcStepsByAllowlist.ps1 `
  -AllowlistPath 'C:\jsons\inference\allowed_step_keys.txt' `
  -SourceDir 'C:\abc_steps' `
  -DestDir 'C:\abc_steps_filtered' `
  -DryRun
```

**Python (same idea):**

```powershell
python standalone_scripts/filter_abc_steps_by_allowlist.py `
  --mode local `
  --allowlist C:\jsons\inference\allowed_step_keys.txt `
  --clear-dest `
  --dry-run

python standalone_scripts/filter_abc_steps_by_allowlist.py `
  --mode local `
  --allowlist C:\jsons\inference\allowed_step_keys.txt `
  --clear-dest
```

Result: allowlisted STEPs land in `C:\abc_steps_filtered`.

### 6C — Jenkins (multi-machine)

**Job file:** `standalone_scripts/Jenkinsfile.filter_abc_steps_no_thread_text`

| Setting | Value |
|---------|--------|
| Job intent | Filter-abc_steps-No-Thread-Text |
| Nodes | `walswkqa19383` … `walswkqa19441` (10 VMs) |
| Source | `C:\abc_steps` |
| Dest | `C:\abc_steps_filtered` |
| Allowlist | Embedded in the Jenkinsfile (agents cannot read the LP76 JSON share) |

**Refresh the Jenkinsfile after a new allowlist:**

```powershell
# Ensure standalone_scripts/allowed_step_keys.txt is up to date (from 6A)
python standalone_scripts/_gen_filter_jenkinsfile.py
```

That regenerates `Jenkinsfile.filter_abc_steps_no_thread_text` with the
allowlist embedded as `ALLOWLIST_P1/P2/P3`, then writeFile on each agent.

**On each agent the job:**

1. Writes `allowed_step_keys.txt` into the workspace
2. Clears existing `.step`/`.stp` in `C:\abc_steps_filtered`
3. Copies matching files from `C:\abc_steps` → `C:\abc_steps_filtered`
4. Reports per-node and total filtered counts

---

## Suggested full loop (checklist)

```text
[ ] 0. (Optional) copy_steps_not_in_allowlist.py  -> fill C:\abc_steps_not_in_allowlist
[ ] 1. BatchJsonExport.vba                         -> C:\jsons
[ ] 2. If interrupted: check batch_logs + restart SW
[ ] 3A. delete_duplicate_jsons.py                  -> clean C:\jsons
[ ] 3B. check_and_delete_covered_steps.py          -> shrink VBA STEP input
[ ] 4. run_onnx_json_batch_inference.py --skip-existing
[ ] 5. (Same command runs Stage-2; or --stage2-only to refilter)
[ ] 6A. export_step_allowlist_from_inference.py
[ ] 6B. Local dry-run filter (optional)
[ ] 6C. _gen_filter_jenkinsfile.py + run Jenkins job
```

PowerShell sketch after export + cleanup:

```powershell
conda run -n brep_mfr_pyg python standalone_scripts/run_onnx_json_batch_inference.py `
  --json-dir C:\jsons --skip-existing

python standalone_scripts/export_step_allowlist_from_inference.py
python standalone_scripts/_gen_filter_jenkinsfile.py
# Then run Jenkins job: Filter-abc_steps-No-Thread-Text
```

---

## What “filtered” means in this pipeline

1. **Filtered JSONs** = Stage-2 list: no face with Thread/Text prob above threshold  
   → `no_confident_thread_or_text.txt`
2. **Allowlist** = unique STEP keys extracted from that list  
   → `allowed_step_keys.txt`
3. **Shipped artifacts (Jenkins)** = STEPs whose keys are on the allowlist  
   → `C:\abc_steps_filtered` on each agent

The Jenkins stage copies **STEPs**, not the JSON files. The JSONs stay under
`C:\jsons` (and the flagged list under `inference/`). If you also need to copy
or prune JSON files by allowlist, use:

```powershell
# Report-only match
python standalone_scripts/match_allowlist_vs_jsons_folder.py

# Delete JSONs whose keys are NOT on the allowlist (keep the clean set)
python standalone_scripts/delete_jsons_on_allowlist.py --json-dir <dir> --dry-run
python standalone_scripts/delete_jsons_on_allowlist.py --json-dir <dir>
```

---

## Script index for this pipeline

| Step | Script |
|------|--------|
| 1 Export | `BatchJsonExport.vba` + `Watchdog-StepOpen.ps1` |
| 1 Prep | `copy_steps_not_in_allowlist.py` |
| 2 Interrupt | VBA `skip_list` / `in_progress` + watchdog |
| 3A Dupes | `delete_duplicate_jsons.py` |
| 3B Covered STEPs | `check_and_delete_covered_steps.py` (or `delete_step_files.py`) |
| 4 Infer | `run_onnx_json_batch_inference.py --skip-existing` |
| 5 Filter | same script Stage-2 / `--stage2-only` |
| 6 Allowlist | `export_step_allowlist_from_inference.py` |
| 6 Local ship | `Filter-AbcStepsByAllowlist.ps1` or `filter_abc_steps_by_allowlist.py` |
| 6 Jenkins | `_gen_filter_jenkinsfile.py` → `Jenkinsfile.filter_abc_steps_no_thread_text` |

For smaller task-set details (CLI flags, alternate folders), see
[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md).
