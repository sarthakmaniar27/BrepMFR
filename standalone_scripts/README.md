# Standalone scripts

One-off utilities formerly under `migration_to_c/`: ONNX export/inference demos,
STEP/JSON housekeeping, allowlists, SolidWorks VBA, Jenkins filter pipeline, etc.

These are **not** part of the Stage‑1 training import path. Run from repo root, e.g.:

```powershell
python standalone_scripts/delete_jsons_on_allowlist.py --dry-run
python standalone_scripts/run_onnx_json_batch_inference.py --skip-existing
```

Core training / split / remap tools stay under `scripts/threads/` and `scripts/inference/`.
