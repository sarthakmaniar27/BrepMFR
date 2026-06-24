# PyTorch profiling for training (tutorial + BrepMFR_PyG playbook)

This guide explains **PyTorch Profiler**, how to view traces in **TensorBoard**, and how to use profiling to speed up **Stage 1** (`segmentation.py`) and **Stage 2** (`domain_adapt.py`) training in this repo—without guessing bottlenecks.

**Further reading**

- [PyTorch TensorBoard Profiler tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) (official)
- [Simple Ways to Speed Up Your PyTorch Model Training](https://towardsdatascience.com/simple-ways-to-speed-up-your-pytorch-model-training-9c9d4899313d/) (practical overview; profiler + allocator + DataLoader themes)

---

## 1. Why profile first?

Optimizations that help one codebase may hurt another. Before changing batch size, workers, precision, or kernels:

1. **Measure** where time goes (CPU vs GPU vs Python overhead).
2. **Measure** whether the GPU is idle waiting on data or saturated with kernels.
3. **Measure** memory residency and spikes (allocator churn vs real model size).

Profiling turns “training feels slow” into **actionable** hypotheses (DataLoader-bound vs encoder-bound vs fragmentation).

---

## 2. What PyTorch Profiler records

With typical settings you can capture:

| Signal | Use |
|--------|-----|
| **CPU op timings** | Python/PyTorch ops on host, DataLoader-related gaps |
| **CUDA kernel timings** | GPU work, overlap with CPU |
| **Memory timeline** | Peak usage, growth patterns; noisy allocator vs steady usage |

**Overhead trade-off:** enabling more activities (CPU + CUDA + memory + shapes) increases overhead and trace size. For kernel-timing fidelity you sometimes profile **CUDA-only** (lower overhead, less host context)—this repo exposes `--tb_profile_cuda_only` next to `--tb_profile`.

---

## 3. TensorBoard and trace export

Traces are written with:

`torch.profiler.tensorboard_trace_handler(log_dir)`

TensorBoard’s **PROFILE** tab loads these traces. Depending on your TensorBoard / PyTorch version, you may need the **`torch-tb-profiler`** plugin package—follow the [official profiler tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) for your stack.

**Important for this repo:** point TensorBoard at the **run logs directory** (same as scalars), not only `tensorboard/version_*`, so profiler plugins sit alongside Lightning logs. See [training_runs.md](training_runs.md).

---

## 4. Reading a trace (quick checklist)

When you open a captured step range:

1. **Locate the training step** — Lightning wraps your `training_step`; custom labels from `record_function` appear as nested ranges (see §5).
2. **Forward vs backward** — Backward often appears as distinct CUDA streams / threads vs forward; large gaps without kernels may be **host-side** or **sync**.
3. **GPU idle regions** — Low SM occupancy / utilization while CPU still busy → likely **input pipeline** (disk, workers, collator) or **single-threaded Python**.
4. **Memory view** — Repeated spikes or instability may indicate **variable tensor shapes** per batch (common with graph batches); consider allocator settings or shape bucketing after validating with traces.

---

## 5. Annotating code with `record_function`

Wrap coarse regions:

```python
with torch.profiler.record_function("my_stage"):
    ...
```

These labels show up in traces when the profiler is active. Cost when profiling is off is negligible for coarse spans.

In **BrepMFR_PyG**, Stage 1 (`BrepSeg`) and Stage 2 (`DomainAdapt`) use a **small fixed set** of labels for encoder, heads, domain loss (Stage 2), and loss computation—see [`models/brepseg_model.py`](../models/brepseg_model.py) and [`models/transfer_model.py`](../models/transfer_model.py).

| Stage | Step | `record_function` names |
|-------|------|-------------------------|
| 1 | train / val | `brep_encoder`, `pool_attn_classifier`, `loss` |
| 2 | train | `brep_encoder`, `pool_attn_classifier_st`, `loss_cls_entropy`, `domain_adv`, `loss_total` |
| 2 | val | `brep_encoder`, `pool_attn_classifier_st`, `val_loss_terms` |

---

## 6. What is already integrated here

[`callbacks/training_logging.py`](../callbacks/training_logging.py) defines **`build_pytorch_profiler`**, which attaches Lightning **`PyTorchProfiler`** when:

- **`segmentation.py train`** or **`domain_adapt.py train`** is run with **`--tb_profile`**.

Behavior:

- Writes traces under **`logs_path`** = `results/logs/stage{1,2}/<run_name>/`
- Uses a **schedule** (`wait`, `warmup`, `active`, `repeat`) so only part of training is profiled (controlled by `--tb_profile_*` CLI flags)
- Default **`record_shapes=True`**, **`profile_memory=True`**
- **`--tb_profile_cuda_only`** → CUDA activities only when a GPU is available (less overhead; less CPU detail)

---

## 7. How to run a profiling smoke job

From the repo root, use a **tiny** run so traces stay small:

**Stage 1**

```powershell
python segmentation.py train --dataset_path <DATASET> `
  --run_name profile_smoke_s1 `
  --max_epochs 1 --limit_train_batches 5 --limit_val_batches 2 `
  --tb_profile `
  --tb_profile_wait 1 --tb_profile_warmup 1 --tb_profile_active 3 --tb_profile_repeat 1
```

**Stage 2** (exercises `collator_st` / source+target)

```powershell
python domain_adapt.py train --source_path <SRC> --target_path <TGT> --pre_train <CKPT> `
  --run_name profile_smoke_s2 `
  --max_epochs 1 --limit_train_batches 5 --limit_val_batches 2 `
  --tb_profile
```

**TensorBoard**

```powershell
tensorboard --logdir results/logs/stage1/profile_smoke_s1/
tensorboard --logdir results/logs/stage2/profile_smoke_s2/
```

Optional: append **`--tb_profile_cuda_only`** for CUDA-focused traces (less overhead).

Open the **PROFILE** (or **PyTorch Profiler**) tab and select the captured trace.

---

## 7b. Full-length Stage 1 training **with** profiler (alpha / production recipe)

Use the **same** hyperparameters as your normal Stage 1 run, and add **`--tb_profile`** plus the schedule flags. Traces land next to TensorBoard under `results/logs/stage1/<run_name>/` (same `--logdir` you use for scalars).

```powershell
cd C:\Users\D58\Desktop\BrepMFR_PyG
conda activate brep_mfr_pyg

python segmentation.py train `
  --dataset_path Z:/Experiment6_PyG/source_dataset `
  --pt_subdir output/bin `
  --class_weights_path artifacts/class_weights/ablation/source_train_alpha100.json `
  --batch_size 32 --num_workers 4 --max_epochs 100 --log_every_n_steps 50 `
  --dropout 0.3 --attention_dropout 0.3 --act-dropout 0.3 `
  --d_model 512 --dim_node 256 --n_heads 32 --n_layers_encode 8 `
  --warmup_freeze_epochs 3 --num_classes 25 `
  --tb_full_graph --csv_log `
  --tb_profile `
  --tb_profile_wait 1 --tb_profile_warmup 1 --tb_profile_active 3 --tb_profile_repeat 1 `
  --run_name ce_alpha100_profile__2026-05-12_manual
```

**Schedule meaning:** each **`repeat`** cycle skips **`wait`** steps, runs **`warmup`** steps (lower fidelity), then records **`active`** steps. With **`repeat=1`** you typically get **one** captured window early in training (low steady overhead afterward). Increase **`--tb_profile_repeat`** or **`--tb_profile_active`** if you want more steps recorded (more disk + slower steps while capturing).

Optional: **`--tb_profile_cuda_only`** — CUDA-heavy timeline, less CPU/context.

**Restart note:** Stop any older Stage 1 job using the GPU (`Task Manager` → GPU / Python, or close the terminal running training) before starting this command.

---

## 7c. TensorBoard: where to look (scalars, graphs, profiler, memory)

### Setup

1. Env (same as training): `conda activate brep_mfr_pyg`.
2. Profiler UI plugin (once): `pip install torch-tb-profiler`
3. Start TensorBoard pointing at the **run logs folder** (parent of `tensorboard/`):

```powershell
tensorboard --logdir "C:\Users\D58\Desktop\BrepMFR_PyG\results\logs\stage1\<run_name>"
```

Use **`127.0.0.1:6006`** (or the URL TB prints). Add **`--bind_all`** only if you need LAN access.

### Tab checklist

| Tab | What it shows | What to look for |
|-----|----------------|------------------|
| **SCALARS** | Loss, LR, accuracy, etc. | Training health; compare to baseline curves. |
| **TIME SERIES** | Same scalars (TB 2.x layout) | Trends over epochs. |
| **GRAPHS** | Module graphs when **`--tb_full_graph`** used | Encoder / head topology (approximate shapes). |
| **PROFILE** / **PyTorch Profiler** | Timeline + operator table + kernel/memory views | **Main performance dashboard** — see below. |

### PROFILE tab — practical steps

1. Left sidebar: choose a **worker** / **run** if multiple traces exist.
2. Open the latest **trace** (often named like `*.pt.trace.json`).
3. **Time axis:** zoom into one **training step** (look for Lightning / `training_step`).
4. **Custom ranges:** search or scan for **`brep_encoder`**, **`pool_attn_classifier`**, **`loss`** ([§5](#5-annotating-code-with-record_function)).
5. **GPU streams:** check for **long empty gaps** on the GPU row while CPU is busy → often **data loading / collation / `.to(device)`**.
6. **Heavy kernels:** wide CUDA blocks under **`aten::`** / **`Memcpy`** — note **`Memcpy HtoD`** vs compute kernels.
7. **Memory view** (if available in your TB/plugin version): spikes per step vs stable footprint — variable batch shapes → allocator churn.

### Files on disk (no browser)

| File | Contents |
|------|----------|
| `results/logs/stage1/<run_name>/fit-pytorch_profiler.txt` | Text summary table (CPU/CUDA totals, top ops). |
| `results/logs/stage1/<run_name>/*.pt.trace.json` | Full trace — load in PROFILE tab or **`chrome://tracing`** → Load. |
| `results/logs/stage1/<run_name>/csv_metrics/version_*/metrics.csv` | Per-step/per-epoch CSV when **`--csv_log`**. |
| `results/logs/stage1/<run_name>/tensorboard/version_*/events.*` | Scalar / graph events. |

---

## 8. Project-specific caveats (graphs + Windows)

- **Variable graph sizes** — Each batch can imply different padded shapes; that can stress the **CUDA caching allocator** (oscillating reserved memory in traces). Environment knobs such as `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` are sometimes suggested for frequently changing allocation sizes—validate on your driver/stack.
- **DataLoader `num_workers`** — Higher workers can overlap loading but increase RAM and (on Windows) spawn cost. [`data/dataset.py`](../data/dataset.py) documents Windows pitfalls; use traces to see if increasing workers actually hides GPU idle gaps.
- **`torch.cuda.empty_cache()` in training steps** — Forces allocator interactions and can **hurt** throughput; if traces show odd stalls around allocation, revisit whether `empty_cache()` every step is necessary (separate change from profiling).
- **`CUDA_LAUNCH_BLOCKING`** — **`segmentation.py` no longer sets this by default.** With `CUDA_LAUNCH_BLOCKING=1`, CUDA APIs wait for kernels to finish, greatly reducing throughput while making errors easier to locate. Enable only via **`--cuda_launch_blocking`** when debugging sync / illegal access. **`domain_adapt.py`** behaves the same.
- **`--pin_memory` / `--dataloader_prefetch_factor`** — Wired through [`data/dataset.py`](../data/dataset.py). Use after checking TB PROFILE and/or [`scripts/profiling/throughput_smoke.py`](../scripts/profiling/throughput_smoke.py) so you trade host RAM vs GPU idle gaps with measurements (watch Windows page-file / Error 1455 with huge batches).

---

## 9. Turning traces into optimizations (evidence-driven)

| Trace suggests | Next lever (examples) |
|----------------|----------------------|
| GPU idle + busy CPU before kernels | `num_workers`, faster storage, lighter collator hot path |
| Continuous kernels, low throughput | batch size, occupancy, kernel fusion (harder in custom Graphormer code) |
| Memory jitter / fragmentation | allocator env, batch bucketing by node count, avoid per-step `empty_cache` |
| Encoder dominates | algorithmic changes need **correctness tests**—profile before swapping attention kernels |

Defer heavy bets (**FlashAttention**, **`torch.compile`**, **FSDP**) until a baseline trace proves where time goes—the encoder here is **not** a stock `nn.TransformerEncoder` stack.

**GPU memory visualization** (optional): PyTorch’s memory snapshots / [Understanding GPU Memory](https://pytorch.org/blog/understanding-gpu-memory-1/) help interpret allocator spikes—not the same UI as TensorBoard PROFILE but complementary.

---

## 10. Optimization changelog (fill in as you go)

Use this table to record **measured** effects after each change (manual is fine).

| Date | Change | Stage | Epoch time / step time | Notes |
|------|--------|-------|-------------------------|-------|
| 2026-05-06 | **Removed default `CUDA_LAUNCH_BLOCKING=1`** in `segmentation.py`; added **`--cuda_launch_blocking`**, **`--pin_memory`**, **`--dataloader_prefetch_factor`** (Stage 1 + 2 parity in `domain_adapt.py`). Added **`scripts/profiling/throughput_smoke.py`**. | S1/S2 infra | Expect large wall-clock improvement vs prior default when profiler off | Expect **overlap** visible in traces vs previous serial-kernel captures. Tune loader + rerun TB PROFILE before raising batch size / compiler flags. |
| 2026-05-12 | Profiler smoke (`--tb_profile`, wait/warmup/active = 1/1/3, repeat 1); **no throughput change yet** | S1 + S2 | ~5 train batches @ ~0.25–1 it/s with profiler on (informal tqdm); Lightning rollup CPU ≫ CUDA self-time in `fit-pytorch_profiler.txt` | GPU: RTX PRO 6000 Blackwell, driver 581.80. Logs: `results/logs/stage{1,2}/profile_smoke_*__2026-05-12/`. Interpretation: `results/diagnostics/profiling_2026-05-12/interpretation.md`. |
| | Baseline production (profiler off) | | | Fill wall-clock/step after next clean timed epoch |
| | e.g. num_workers sweep | | | Trace screenshot / TB run name |

---

## 11. Success criteria (for your team)

- You can capture a **PROFILE** trace for **Stage 1** and **Stage 2** using **`--tb_profile`**.
- You can name the dominant bucket (**data vs encoder vs heads vs domain loss**) from trace + `record_function` labels.
- Follow-up speedups are justified by **before/after** timing rows in §10.
