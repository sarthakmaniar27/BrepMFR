# Experiment tracking and hyperparameter optimization

This tutorial explains **experiment tracking** (recording what you ran and how well it did) and **hyperparameter optimization (HPO)** (searching systematically over training knobs). It then maps each common tool to **BrepMFR_PyG**: what is integrated today, where outputs go, and how to turn features on.

For checkpoint and folder conventions, see **[training_runs.md](training_runs.md)**.

---

## Two different jobs

| Goal | Question it answers | Typical tools |
|------|---------------------|----------------|
| **Experiment tracking** | What was the config? What were the metrics? Can I compare runs A vs B? | TensorBoard, CSV logs, Weights & Biases, MLflow |
| **Hyperparameter optimization** | Which learning rate, batch size, schedule, etc. gives the best validation metric? | Optuna, Ray Tune, W&B Sweeps |

Tracking can be **passive** (every train run logs scalars). HPO is **active** (many trials, each suggested parameters, objective = metric).

---

## TensorBoard

**What it is.** A local web UI backed by event files: **scalars** (loss, accuracy), optional **images** (e.g. confusion matrices), **graphs** (traced submodules), **histograms**, **text**, and **profiler** traces when enabled.

**What it is used for.** Monitoring one or many runs on your machine without signing up for a service; inspecting curves and debugging training.

**In this repo.**

- Every Stage 1 / Stage 2 train uses Lightning **TensorBoardLogger** via [`callbacks/training_logging.py`](../callbacks/training_logging.py).
- Logs live under `results/logs/stage{1,2}/<run_name>/tensorboard/version_*`.
- Optional flags on [`segmentation.py`](../segmentation.py) and [`domain_adapt.py`](../domain_adapt.py): `--tb_profile`, `--tb_full_graph` (see [training_runs.md](training_runs.md)).

**How to view.**

```powershell
cd <BrepMFR_PyG_repo_root>
tensorboard --logdir results/logs/stage1/<run_name>/
```

Use the **run folder** as `--logdir` when you also want the **PROFILE** tab for PyTorch profiler traces.

---

## CSV metrics (Lightning CSVLogger)

**What it is.** A flat **metrics table** per run (`metrics.csv` under a versioned subfolder).

**What it is used for.** Spreadsheet-friendly exports, simple plotting in Python/R/Excel, and scripting without parsing protobuf event files.

**In this repo.** Enable **`--csv_log`** on `segmentation.py train` or `domain_adapt.py train`. Outputs: `results/logs/stage{1,2}/<run_name>/csv_metrics/version_*/metrics.csv`.

---

## Weights & Biases (W&B)

**What it is.** A **hosted** experiment platform: runs, configs, metric curves, system stats, optional artifacts, team comparison and sweep UI.

**What it is used for.** Comparing many runs in the browser, sharing results, and (with Sweeps) driving HPO from the cloud UI.

**In this repo.**

- Opt-in: **`--use_wandb`** and optionally **`--wandb_project`** on [`segmentation.py`](../segmentation.py) and [`domain_adapt.py`](../domain_adapt.py).
- Implementation: [`build_loggers()` in `callbacks/training_logging.py`](../callbacks/training_logging.py) appends Lightning **`WandbLogger`** next to TensorBoard.
- Requires `pip install wandb` and a [wandb.ai](https://wandb.ai) account. Local run metadata often appears under `results/logs/.../wandb/` during training.

**Typical command.**

```powershell
python segmentation.py train --dataset_path <DATASET> --use_wandb --wandb_project brepmfr-pyg
```

If you did **not** pass `--use_wandb`, that training run **does not** appear in W&B.

---

## Optuna

**What it is.** An **HPO library**: you define an **objective** (return a number to minimize or maximize), a **study** holds trials, and Optuna suggests hyperparameters using samplers and optional **pruners** (early-stop bad trials).

**What it is used for.** Automating searches over discrete and continuous knobs (learning rate, epochs, dropout, etc.) with reproducible studies.

**In this repo (Stage 1 example only).**

- **`scripts/hpo/optuna_subprocess_stage1.py`** — runs **`segmentation.py train`** in a **subprocess** per trial with a suggested `--max_epochs`, fixed `--run_name` pattern `optuna_s1_t<N>`, then reads **`eval_loss`** back from TensorBoard using **`scripts/hpo/tb_metrics.py`**.
- **[`scripts/hpo/README_OPTUNA.md`](../scripts/hpo/README_OPTUNA.md)** — extending objectives; notes on **`PyTorchLightningPruningCallback`** (not wired into the main training scripts yet).

**Important.** A normal interactive **`python segmentation.py train ...`** is **not** an Optuna trial unless you started it from the Optuna driver (or your own study). Check **`--run_name`**: Optuna trials look like **`optuna_s1_t0`**, **`optuna_s1_t1`**, ….

**Example (from repo root).**

```powershell
python scripts/hpo/optuna_subprocess_stage1.py --dataset_path <DATASET_ROOT> --trials 12
```

Dependencies: `optuna` (listed in `environment_pyg.yml`).

---

## Ray Tune

**What it is.** A **distributed** tuning framework (often with Ray Train) suited to **many parallel trials** across GPUs or a cluster.

**What it is used for.** Large sweep throughput where subprocess loops are too slow.

**In this repo.** **Not integrated** into `segmentation.py` / `domain_adapt.py`. Orientation only: [`scripts/hpo/README_RAY.md`](../scripts/hpo/README_RAY.md).

---

## MLflow

**What it is.** An experiment-tracking stack that can run **self-hosted** or managed: runs, parameters, metrics, artifacts; popular in enterprises and air-gapped setups.

**What it is used for.** Same genre as W&B for tracking and registry; different deployment model.

**In this repo.** **No MLflow logger** is wired into Lightning training today. You could add a second logger similar to W&B if needed.

---

## Summary: BrepMFR_PyG

| Tool | Role | Integrated? | How / where |
|------|------|----------------|-------------|
| **TensorBoard** | Local dashboards | Yes (default) | Logs under `results/logs/stage*/<run>/tensorboard/` |
| **CSVLogger** | Tabular metrics | Yes | `--csv_log` → `csv_metrics/version_*/metrics.csv` |
| **W&B** | Hosted tracking | Yes (opt-in) | `--use_wandb` [--wandb_project …] |
| **PyTorch Profiler → TB** | Performance traces | Yes (opt-in) | `--tb_profile` (+ schedule flags) |
| **Optuna** | HPO | Partial | Stage 1 subprocess driver only; see `scripts/hpo/` |
| **Ray Tune** | Distributed HPO | No | README pointer only |
| **MLflow** | Tracking / registry | No | Conceptual only |

---

## Combining tools

- **Daily training:** TensorBoard (and optionally **`--csv_log`** or **`--use_wandb`**) on every run.
- **Hyperparameter search:** keep TensorBoard for each trial; use Optuna’s driver to orchestrate many subprocess runs and scrape **`eval_loss`** from TB (or extend the objective to read CSV).
- **Never assume Optuna or W&B ran:** verify **`--use_wandb`** for W&B and **`optuna_s1_t*`** / Optuna process for Optuna.

For profiler and graph options on TensorBoard, see the **TensorBoard** section in [training_runs.md](training_runs.md) and the dedicated [PyTorch profiling guide](pytorch_profiling_guide.md).
