# BrepMFR

Code for BrepMFR: Enhancing machining feature recognition in B-rep models through deep learning and domain adaptation.

![The network architecture of BrepMFR](docs/img/network_architecture.jpg)

## About BrepMFR

BrepMFR, a novel deep learning network designed for machining feature recognition on B-rep models within the CAD/CAM domain. The original B-rep model is converted into a graph representation for network-friendly input, where graph nodes and edges respectively correspond to faces and edges of the original model. Subsequently, we leverage a graph neural network based on the Transformer architecture and graph attention mechanism to encode both local geometric shape and global topological relationships, achieving high-level semantic extraction and prediction of machining feature categories. Furthermore, to enhance the performance of neural networks on real-world CAD models, we adopt a two-step training strategy within a novel transfer learning framework.

## Preparation

### Environment setup

```
git clone https://github.com/zhangshuming0668/BrepMFR.git
cd BrepMFR
conda env create -f environment.yml
conda activate brep_mfr
```

### Data preparation

Our synthetic CAD datasets have been publicly available on [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=931c088fd44f4d3e82891a5180f10d90)

## Training

For machining feature recognition, the network can be trained using:
```
python segmentation.py train --dataset_path /path/to/dataset --max_epochs 1000 --batch_size 64
```

The **checkpoints** go under **`results/stage1/<run_name>/`**, while **logs** (TensorBoard, optional CSV/W&B) go under **`results/logs/stage1/<run_name>/`** (see [docs/training_runs.md](docs/training_runs.md)). Monitor TensorBoard:

```
tensorboard --logdir results/logs/stage1/<run_name>/tensorboard
```

## Testing

The best checkpoints based on the smallest validation loss are saved in the results folder. The checkpoints can be used to test the model as follows:

```
python segmentation.py test --dataset_path /path/to/dataset --checkpoint ./results/stage1/<run_name>/best.ckpt --batch_size 64
```

## Repository layout (PyG fork — this workspace)

- **`segmentation.py`**, **`domain_adapt.py`**: Stage 1 and Stage 2 training entrypoints (run from repo root).
- **`models/`**, **`data/`**: Core Lightning models and dataloaders.
- **`artifacts/class_weights/`**: Version-controlled canonical JSON for **Stage 1 CE weights** (`stage1/`) and **Stage 2 IWDAN priors** (`stage2_iwdan/`). See [`artifacts/class_weights/README.md`](artifacts/class_weights/README.md).
- **`artifacts/baseline/`**: Immutable reference pointers for reproducibility (frozen **Full-A2** weighted Stage 1 and related artifacts). See [`artifacts/baseline/README.md`](artifacts/baseline/README.md).
- **`scripts/`**: Runnable utilities grouped by task ([`scripts/README.md`](scripts/README.md)); each script boots the repo root via **`bootstrap_path.py`**.
- **`tools/`**: Dataset and pipeline maintenance utilities ([`tools/README.md`](tools/README.md)) — STEP/JSON converters, bin audits, renaming, chunking.
- **`results/`**: Training outputs only (typically gitignored): **checkpoints** under `results/stage{1,2}/`, **logs** (TensorBoard, CSV, W&B) under `results/logs/stage{1,2}/`.

### Training run folders

- **`segmentation.py train`** → checkpoints: `results/stage1/<run_name>/`; logs: `results/logs/stage1/<run_name>/`.
- **`domain_adapt.py train`** → checkpoints: `results/stage2/<run_name>/`; logs: `results/logs/stage2/<run_name>/`.

Checkpoints stay top-level under the stage folder (no extra `MMDD/HHMMSS` nesting). Logs use Lightning `version_*` subfolders inside `tensorboard/` and `csv_metrics/`. Full convention and **canonical balanced Stage 1 / Stage 2** pointers are in **[docs/training_runs.md](docs/training_runs.md)**.

**Frozen Full-A2 baseline** (weighted CE Stage 1) is recorded under **`artifacts/baseline/`**.

**Not moved**: `results/diagnostics/` (eval outputs). Prefer **`artifacts/class_weights/`** over ad hoc JSON under `results/`.


