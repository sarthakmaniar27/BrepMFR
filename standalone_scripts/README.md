# Standalone scripts

One-off utilities formerly under `migration_to_c/`: ONNX export/inference demos,
STEP/JSON housekeeping, allowlists, SolidWorks VBA, Jenkins filter pipeline, etc.

These are **not** part of the Stage‑1 training import path. Run from repo root, e.g.:

```powershell
python standalone_scripts/delete_jsons_on_allowlist.py --dry-run
python standalone_scripts/run_onnx_json_batch_inference.py --skip-existing
```

**Recurring task workflows** (clean duplicate JSONs → delete covered STEPs → infer only new JSONs):
see [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md).

**Full operator pipeline** (VBA export → interrupt cleanup → infer → Stage-2 filter →
allowlist → Jenkins `abc_steps_filtered`):
see [END_TO_END_PIPELINE.md](END_TO_END_PIPELINE.md).

**Continuous Stage-1 + Stage-2 (parallel, no duplicates, distribute-only Jenkins):**
see [CONTINUOUS_PIPELINE.md](CONTINUOUS_PIPELINE.md) and `pipeline_dedup/`.

Core training / split / remap tools stay under `scripts/threads/` and `scripts/inference/`.
