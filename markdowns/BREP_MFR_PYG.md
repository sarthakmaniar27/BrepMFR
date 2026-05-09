# BrepMFR PyG / Blackwell stack

This repository (`BrepMFR_PyG`) is a **PyTorch 2.7 + CUDA 12.8 + PyTorch Geometric** variant of BrepMFR. It removes **DGL** from the **training** path so you can run on **NVIDIA Blackwell (sm_120)** and use **Windows wheels** for PyG. The original **DGL `.bin`** graphs live under `Z:\Experiment6` (read-only); converted **`*.pt`** graphs and copied splits live under **`Z:\Experiment6_PyG`**.

---

## Why two conda environments?

| Environment | Purpose |
|-------------|---------|
| **`brep_mfr`** | **Legacy ingest / parity-only:** **`convert_dgl_bins_to_pyg.py`** loads existing `.bin` files; **`json_pyg_parity_vs_bin`** uses **`bin_to_pyg`** as ground truth vs JSON→PyG outputs. Prefer **`brep_mfr_pyg`** for new macro JSON → **`.pt`**. |
| **`brep_mfr_pyg`** | **Training, inference, and `json_to_brepmfr_pyg`** (SolidWorks JSON → **`.pt`**)—no DGL on this stack. |

Do **not** modify `Z:\Experiment6`; the legacy converter **reads** `.bin` and **writes** to `Z:\Experiment6_PyG`. Prefer **`scripts/inference/json_to_brepmfr_pyg.py`** for new solids so you skip DGL `.bin` entirely.

---

## Create the PyG training environment

From this repo root:

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
conda env create -f environment_pyg.yml -y
conda activate brep_mfr_pyg
```

The file uses **PyPI as the primary index** and **`--extra-index-url https://download.pytorch.org/whl/cu128`** so PyTorch cu128 wheels resolve **and** packages like `pytorch-lightning` install from PyPI. (Using `--index-url` alone for cu128 breaks Lightning install.)

### PyG binary extensions (required once per env)

```powershell
conda activate brep_mfr_pyg
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install torch-geometric==2.7.0
```

### Quick verify

```powershell
conda activate brep_mfr_pyg
python -c "import torch; import torch_geometric as pyg; print(torch.__version__, torch.cuda.is_available(), pyg.__version__)"
```

---

## SolidWorks JSON → PyG `.pt` (recommended; **`brep_mfr_pyg`**)

Use this path for **new macro JSON** instead of emitting DGL `.bin` then converting:

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
conda activate brep_mfr_pyg
python scripts/inference/json_to_brepmfr_pyg.py `
  --json_dir Z:/Experiment6/source_dataset/input `
  --pt_out_dir Z:/my_pyg_mirror/source_dataset/output/pt `
  --label_out_dir Z:/my_pyg_mirror/source_dataset/output/label
```

Semantics match **`BrepMFR/json_to_brepmfr_bin.py`** tensors + **`convert_dgl_bins_to_pyg.bin_to_pyg`** field naming.

### Parity and collator sanity (use **`Z:\Experiment_test`**)

When checking converter correctness:

1. **`mkdir`** / copy a **few** **`input\*.json`** and matching **`output\bin\*.bin`** pairs into **`Z:\Experiment_test\input_json`** and **`…\ref_bin`** (copy only—do **not** modify **`Experiment6`**).
2. Run **`conda activate brep_mfr`** (reference path imports **`dgl`**):

```powershell
python scripts/diagnostics/json_pyg_parity_vs_bin.py --root Z:/Experiment_test --patterns "*.json" `
  --write_log Z:/Experiment_test/parity_logs/parity.md
```

3. Build **`.pt`** under **`conda activate brep_mfr_pyg`** with **`json_to_brepmfr_pyg.py`** into **`Z:\Experiment_test\out_pyg`**, then:

```powershell
python scripts/diagnostics/json_pyg_collator_smoke.py --root Z:/Experiment_test/out_pyg --batch_size 2
```

---

## Convert DGL bins to PyG `.pt` (legacy env)

Dry-run (counts `.bin` files, no writes):

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
conda activate brep_mfr
python scripts/inference/convert_dgl_bins_to_pyg.py --src-root Z:/Experiment6 --dst-root Z:/Experiment6_PyG --dry-run
```

Full conversion (long-running; ~157k+ bins in a typical full tree):

```powershell
conda activate brep_mfr
python scripts/inference/convert_dgl_bins_to_pyg.py --src-root Z:/Experiment6 --dst-root Z:/Experiment6_PyG
```

The script:

- Writes **`*.pt`** (torch-serialized `torch_geometric.data.Data`) with the same relative paths as **`*.bin`**, under `Z:\Experiment6_PyG`.
- Copies **`output/**/*.json`** (skips macro `input` JSON noise).
- Copies **all `*.txt`** split lists (e.g. `train.txt`, `s_train.txt`, …) so paths stay self-contained.

**Dataset code** globs **`*[0-9].pt`** and uses stems listed in those `.txt` files—same as before, but with `.pt` instead of `.bin`.

---

## Code changes vs original BrepMFR (summary)

- **`data/dataset.py`:** Loads `torch.load(..., weights_only=False)`; no DGL in training path. `CADSynth` / `TransferDataset` enumerate `*.pt`.
- **`data/collator.py`:** Replaces `dgl.batch` with **`batch_edge_index(...)`**; batch dict key **`edge_index`** (shape `[2, total_edges]`) instead of **`graph`**.
- **`models/modules/brep_encoder.py`:** Passes `batch_data["edge_index"]` into `GraphAttnBias`.
- **`models/modules/layers/brep_encoder_layer.py`:** `_EdgeConv` and `GraphAttnBias` take **`edge_index`**; `src, dst = edge_index[0], edge_index[1]`.
- **`models/modules/utils/fairseq_shim.py`:** Replaces **fairseq** imports for PyTorch 2.7 compatibility (Lightning + training use this).
- **`models/modules/layers/feature_encoders.py` / `models/modules/utils/output.py`:** Unused DGL imports removed.
- **`segmentation.py` / `domain_adapt.py`:** **Lightning 2** `Trainer(...)` with `--max_epochs` / `--log_every_n_steps` (no `Trainer.add_argparse_args`).
- **`models/brepseg_model.py` / `models/transfer_model.py`:** `on_train_epoch_end` / `on_validation_epoch_end` / `on_test_epoch_end`; updated **`optimizer_step`** signature; `transfer_model` imports **`numpy`**.

Unused / legacy DGL helpers elsewhere in the repo (e.g. `stp_to_bin.py`, audits) were **not** all ported—use **`brep_mfr`** if you still need DGL-only scripts.

---

## Training paths and data roots

- Point **`--dataset_path`** (Stage 1) or **`--source_path` / `--target_path`** (Stage 2) at the dataset folders under **`Z:\Experiment6_PyG`** that contain the copied **`*.txt`** lists and **`output\*.pt`** files—the **mirror** of your old `Experiment6` layout.
- On Windows, **`--num_workers 0`** is the safest default unless you have verified multiprocessing.

**No training was started automatically** from this migration work; run training only when you explicitly intend to.

---

## Reusing Stage 1 weights (no re-train required)

You can **reuse the same Stage 1 checkpoint** (`best.ckpt` or equivalent) trained under the old DGL stack:

- **Learnable modules** (encoder, attention, classifier, etc.) are unchanged in **name and shape**; only **how the batch supplies topology** changed (`edge_index` vs DGL graph).
- **Stage 2** already uses `BrepSeg.load_from_checkpoint(args.pre_train)`; keep pointing **`--pre_train`** at that file.

Recommended **sanity check** after conversion: load the ckpt in `brep_mfr_pyg`, run **one batch** of validation or a short eval—expect **no** missing critical keys. If anything appears, it is usually fixable with `strict=False` or a one-off key remap (unexpected).

---

## `environment_pyg.yml` reference

- **Name:** `brep_mfr_pyg`
- **Python:** 3.10
- **pip:** `torch==2.7.0+cu128`, `torchvision`, `torchaudio`, `pytorch-lightning>=2.4,<2.6`, `torchmetrics`, `tensorboardX`, `scipy`, `prefetch_generator`, `prettytable`, `tqdm`, `numpy>=1.23,<2`, plus **`--extra-index-url`** for cu128 as documented in the YAML comments.

PyG wheels are **not** in that YAML; install them with the `pip install ... -f https://data.pyg.org/whl/torch-2.7.0+cu128.html` command above.

---

## Troubleshooting

| Issue | Likely cause |
|--------|----------------|
| `conda env create` fails on Lightning | `environment_pyg.yml` must use **`--extra-index-url`**, not **`--index-url`**, for the cu128 line (see file). |
| Conversion fails | Wrong env (need **`brep_mfr`** with DGL + torch compatible with how `.bin` was written). |
| Training cannot find samples | Split **`.txt`** stems must match **`*.pt`** under the **PyG** tree; lists must live next to the same relative layout as before. |
| CUDA / Blackwell | Driver + `torch 2.7.0+cu128` wheel must match; check `torch.cuda.get_device_capability()`. |

---

## Quick reference commands

```powershell
# Convert (legacy)
conda activate brep_mfr
cd C:\Users\D58\Desktop\BrepMFR_PyG
python scripts/inference/convert_dgl_bins_to_pyg.py --src-root Z:/Experiment6 --dst-root Z:/Experiment6_PyG

# Train (PyG env) — example only; adjust paths and do not run until you intend to
conda activate brep_mfr_pyg
cd C:\Users\D58\Desktop\BrepMFR_PyG
# python segmentation.py train --dataset_path <path_under_Experiment6_PyG> ...
# python domain_adapt.py train --source_path ... --target_path ... --pre_train <stage1.ckpt> ...
```

---

*Last updated to reflect the BrepMFR → PyG migration, dual-env workflow, `Z:\` layout, checkpoint reuse, and environment fix for PyPI + cu128.*
