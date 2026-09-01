# Project Progress Tracking — ABC JSON Infer + Thread/Text Filter Pipeline

**Updated:** 2026-08-10  
**Focus:** Scripts that run ONNX inference on B-rep JSONs and keep only parts with **no** Thread/Text face probability ≥ 0.8, then ship allowlisted STEPs (local or Jenkins).

## Main system goals

1. Export B-rep JSONs from ABC STEPs (SolidWorks VBA).
2. Infer Stock / Thread / Text per face on those JSONs (lite ONNX).
3. **Filter:** keep JSONs where no face has `prob_Thread` or `prob_Text` above confidence **0.80**.
4. Convert that clean list into a STEP-key allowlist and copy matching STEPs to `C:\abc_steps_filtered` (local or multi-agent Jenkins).

Canonical docs (already in repo):

- [`standalone_scripts/END_TO_END_PIPELINE.md`](standalone_scripts/END_TO_END_PIPELINE.md) — full operator loop
- [`standalone_scripts/WORKFLOW_GUIDE.md`](standalone_scripts/WORKFLOW_GUIDE.md) — daily cleanup + infer task sets
- [`standalone_scripts/README.md`](standalone_scripts/README.md) — index of standalone utilities
- [`standalone_scripts/CONTINUOUS_PIPELINE.md`](standalone_scripts/CONTINUOUS_PIPELINE.md) — related continuous Stage-1/2 dedup (optional extension)

## Workflow (how files interact)

```text
ABC STEPs (C:\abc_steps / share)
  → [optional] copy_steps_not_in_allowlist.py → C:\abc_steps_not_in_allowlist
  → BatchJsonExport.vba (+ Watchdog-StepOpen.ps1) → C:\jsons\*.json
  → delete_duplicate_jsons.py (clean C:\jsons)
  → check_and_delete_covered_steps.py (shrink STEP backlog)
  → run_onnx_json_batch_inference.py
        Stage-1: JSON → lite PyG → ONNX → C:\jsons\inference\*_predictions.csv
        Stage-2: scan CSVs; flag JSONs with no Thread/Text ≥ 0.8
                 → no_confident_thread_or_text.txt/.csv
  → export_step_allowlist_from_inference.py → allowed_step_keys.txt
  → filter_abc_steps_by_allowlist.py / Filter-AbcStepsByAllowlist.ps1
        OR _gen_filter_jenkinsfile.py → Jenkinsfile.filter_abc_steps_no_thread_text
  → C:\abc_steps_filtered (clean STEPs only)
```

**Filter rule:** a JSON is “clean” iff for every face, `max(prob_Thread, prob_Text) ≤ 0.80` (default `--confidence 0.80`).

## File-by-file roles (this pipeline)

| File | Role |
|------|------|
| `standalone_scripts/run_onnx_json_batch_inference.py` | Core: Stage-1 ONNX on JSONs + Stage-2 Thread/Text confidence filter |
| `standalone_scripts/export_step_allowlist_from_inference.py` | Builds STEP-key allowlist from Stage-2 `no_confident_thread_or_text.*` |
| `standalone_scripts/filter_abc_steps_by_allowlist.py` | Copies allowlisted STEPs → `abc_steps_filtered` (local/Jenkins-friendly) |
| `standalone_scripts/Filter-AbcStepsByAllowlist.ps1` | PowerShell equivalent of the STEP copy |
| `standalone_scripts/_gen_filter_jenkinsfile.py` | Embeds allowlist into Jenkinsfile (agents can't read JSON share) |
| `standalone_scripts/Jenkinsfile.filter_abc_steps_no_thread_text` | Multi-VM Jenkins job: clear dest, copy matching STEPs |
| `standalone_scripts/Jenkinsfile.filter_jenkins_job` | Related/older Jenkins filter job variant |
| `standalone_scripts/BatchJsonExport.vba` | SolidWorks: STEP → B-rep JSON into `C:\jsons` |
| `standalone_scripts/Watchdog-StepOpen.ps1` | Kills hung STEP opens during VBA export |
| `standalone_scripts/delete_duplicate_jsons.py` | One JSON per STEP key; remove `.SLDPRT` temps |
| `standalone_scripts/check_and_delete_covered_steps.py` | Delete STEPs that already have JSONs (VBA input folder) |
| `standalone_scripts/delete_step_files.py` | Same idea for shared `abc_steps` vs `C:\jsons` |
| `standalone_scripts/delete_step_files_from_abc_jsons.py` | Same when JSONs live on `\\GR-SW26859\abc` |
| `standalone_scripts/copy_steps_not_in_allowlist.py` | Prep VBA input: STEPs whose keys are outside current allowlist |
| `standalone_scripts/match_allowlist_vs_jsons_folder.py` | Report-only allowlist ↔ JSON folder overlap |
| `standalone_scripts/delete_jsons_on_allowlist.py` | Prune JSONs **not** on allowlist (keep clean set) |
| `standalone_scripts/count_unique_json_vs_steps.py` | Coverage report JSON keys vs STEP share |
| `standalone_scripts/allowed_step_keys.txt` (+ `_p1`/`_p2`/`_p3`) | Materialized allowlist artifacts for Jenkins embedding |
| `standalone_scripts/BrepMFR_lite_onnx_pyg_demo_v2/` | Default lite ONNX package used by batch JSON inference |
| `standalone_scripts/END_TO_END_PIPELINE.md` | Authoritative end-to-end explanation |
| `standalone_scripts/WORKFLOW_GUIDE.md` | Recurring cleanup + infer loop |

### Related but alternate paths

| File | When it differs |
|------|-----------------|
| `standalone_scripts/run_onnx_a1_a3_inference.py` | A1+A3 / no_a2 ONNX on **PyG `.pt`**, not raw JSON |
| `scripts/inference/json_to_brepmfr_pyg_optimized.py` | JSON → PyG conversion for the A1+A3 path |
| `standalone_scripts/pipeline_dedup/*` + `CONTINUOUS_PIPELINE.md` | Continuous enqueue/distribute after filtering (dedup ledgers) |
| `standalone_scripts/BatchInference.vba` | Native SW inference command — **not** the JSON→ONNX filter path |

## Outputs that define “filtered”

1. `C:\jsons\inference\*_predictions.csv` — per-face probs  
2. `C:\jsons\inference\no_confident_thread_or_text.txt` — clean JSONs  
3. `C:\jsons\inference\allowed_step_keys.txt` — STEP keys to keep  
4. `C:\abc_steps_filtered` — shipped clean STEPs (Jenkins or local)
