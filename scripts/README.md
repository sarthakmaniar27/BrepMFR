# Scripts index

Runnable utilities grouped by task. Prefer running from repo root (`cd BrepMFR_PyG`).

| Folder | Contents |
|--------|----------|
| `training/` | Class-weight computation for Stage 1 |
| `inference/` | PyG inference, STEP/single-graph paths, DGL→PyG conversion |
| `diagnostics/` | Audits, logit adjustment, paper table replication, TB helpers |
| `validation/` | Bin/label verification, dataset scans |
| `dataset_utils/` | Sorted CSV dumps and summarizers (large outputs stay under `sorted_dumps*`) |
| `monitoring/` | Stage 2 status / pollers |
| `smoke/` | Thin wiring checks (datasets, loaders) |

Heavy generated trees remain at `scripts/sorted_dumps/`, `scripts/sorted_dumps_full/`, `scripts/scan_reports/`.

All Python entrypoints use `bootstrap_path.py` at the repo root to put the project on `sys.path` regardless of nesting depth.

---

## A2 proximity ablation (omit dense A2 tensors)

**Ingest:** after bulk JSON is ready, regenerate PyG caches **without** storing `d2_distance` / `angle_distance` (no `face_pairs` read) into a **separate** tree so `output/bin` and `output/bin_skip_a2` never share the same rglob roots. Prefer `--inference_profile no_a2`; `--skip_a2` is a deprecated alias.

```powershell
conda activate brep_mfr_pyg
python scripts/inference/json_to_brepmfr_pyg.py --inference_profile no_a2 `
  --json_dir Z:/Experiment6/source_dataset/input `
  --pt_out_dir Z:/Experiment6_PyG/source_dataset/output/bin_skip_a2 `
  --label_out_dir Z:/Experiment6_PyG/source_dataset/output/label

python scripts/inference/json_to_brepmfr_pyg.py --inference_profile no_a2 `
  --json_dir Z:/Experiment6/target_dataset/input/json_new_labels_cadsynth_label_indices `
  --pt_out_dir Z:/Experiment6_PyG/target_dataset/output/bin_skip_a2 `
  --label_out_dir Z:/Experiment6_PyG/target_dataset/output/label
```

**Align triplets** (`output/bin`, `output/bin_skip_a2`, `output/label`) so every stem present in `bin/*.pt` has a matching `bin_skip_a2/*.pt` and `label/*.json`, and extra skip/label files not in `bin` are removed. Skip graphs are derived from the full `bin` graph by **removing** dense A2 tensors (non-A2 tensors match `bin`); labels are rewritten from `label_feature`.

```powershell
python scripts/inference/sync_triplet_outputs.py Z:/Experiment6_PyG/source_dataset Z:/Experiment6_PyG/target_dataset
python scripts/inference/sync_triplet_outputs.py Z:/Experiment6_PyG/source_dataset Z:/Experiment6_PyG/target_dataset --apply
```

**Train** with the same splits and hyperparameters as the full-A2 baseline, but constrain graph discovery so only `bin_skip_a2` is seen:

```powershell
python segmentation.py ... --dataset_path Z:/Experiment6_PyG/source_dataset `
  --pt_subdir output/bin_skip_a2 --run_name stage1_no_a2_proximity__...

python domain_adapt.py ... --source_path Z:/Experiment6_PyG/source_dataset `
  --target_path Z:/Experiment6_PyG/target_dataset `
  --pt_subdir output/bin_skip_a2 --run_name stage2_no_a2_proximity__...
```

**Eval / diagnostics:** pass `--pt_subdir output/bin_skip_a2` to `scripts/diagnostics/diagnose_stage1_target.py`, `logit_adjust_eval.py`, `paper_table3_eval.py`, and `stage2_logit_adjust_eval.py` whenever the dataset root would otherwise match graphs in multiple `bin*` folders.

**Sanity:**

```powershell
# Fast: triplet counts + random graphs omit A2 tensors
python scripts/smoke/smoke_skip_a2_training_ready.py Z:/Experiment6_PyG/source_dataset Z:/Experiment6_PyG/target_dataset
# Slow (rglob scan): CADSynth(train) resolves files and loads one graph
python scripts/smoke/smoke_skip_a2_training_ready.py ... --cad_synth_smoke
# Full torch.load() scan over train/val split stems (--pt_subdir must match training)
python scripts/diagnostics/verify_pt_loadable.py --dataset_path Z:/Experiment6_PyG/source_dataset --pt_subdir output/bin_skip_a2
python scripts/diagnostics/spot_check_skip_a2_pt.py --full_dir Z:/Experiment6_PyG/source_dataset/output/bin --skip_dir Z:/Experiment6_PyG/source_dataset/output/bin_skip_a2 --max_checks 12
python scripts/diagnostics/spot_check_skip_a2_pt.py --full_dir Z:/Experiment6_PyG/target_dataset/output/bin --skip_dir Z:/Experiment6_PyG/target_dataset/output/bin_skip_a2 --max_checks 12
```

**Full training runs** (mirror your baseline `--class_weights_path`, architecture flags, epochs; only `pt_subdir` + `run_name` distinguish the ablation):

```powershell
# Stage 1
python segmentation.py train --dataset_path Z:/Experiment6_PyG/source_dataset --dataset cadsynth ^
  --pt_subdir output/bin_skip_a2 --run_name stage1_no_a2_proximity__2026-05-10 --num_workers 0 ^
  --class_weights_path "<path\to\stage1_class_weights.json>"

# One-epoch Lightning smoke (~minutes): add --max_epochs 1 --limit_train_batches 4 --limit_val_batches 4

# Stage 2 (reuse your `--pre_train` Stage-1 ckpt; add IWDAN JSONs if used in baseline)
python domain_adapt.py train --source_path Z:/Experiment6_PyG/source_dataset ^
  --target_path Z:/Experiment6_PyG/target_dataset ^
  --pt_subdir output/bin_skip_a2 --run_name stage2_no_a2_proximity__2026-05-10 --num_workers 0 ^
  --pre_train Z:/path/to/stage1_no_a2_best.ckpt
```
