# Frozen baselines

This folder records **immutable references** so ablations stay comparable:

- Paths are workspace-relative (`BrepMFR_PyG` root).
- Large binaries (`.ckpt`) are **not committed**; the JSON pins the canonical path on disk.

## Full-A2 — Stage 1 weighted CE (frozen)

See `stage1_weighted_balanced_full_a2_locked.json`.

- **`pt_encoder` variant**: loads `output/bin` (default **`--pt_subdir output/bin`** — omit flag).
- **`zero-A2` ablation** uses **`--pt_subdir output/bin_skip_a2`** instead; checkpoints must not be mixed when comparing TensorBoard curves.
- Numeric parity with the locked run is spelled out under **`mirrored_train_args`** in the JSON (from that run's `hparams.yaml`, including **`max_epochs: 100`**, **`batch_size: 32`**, **`num_workers: 4`**). Match those and only change **`--run_name`** (and add **`--pt_subdir`** for the ablation).
