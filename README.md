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

The logs and checkpoints go under **`results/stage1/<run_name>/`** (see [docs/training_runs.md](docs/training_runs.md)), and can be monitored with Tensorboard:

```
tensorboard --logdir results/stage1/<run_name>/tensorboard
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
- **`scripts/`**: Runnable utilities grouped by task ([`scripts/README.md`](scripts/README.md)); each script boots the repo root via **`bootstrap_path.py`**.
- **`tools/`**: Dataset and pipeline maintenance utilities ([`tools/README.md`](tools/README.md)) — STEP/JSON converters, bin audits, renaming, chunking.
- **`results/`**: Training logs and checkpoints only (typically gitignored).

### Training run folders (`results/stage1`, `results/stage2`)

- **`segmentation.py train`** → `results/stage1/<run_name>/` (default auto `ce_weighted_balanced__YYYY-MM-DD_HHMMSS_mmm` unless you pass **`--run_name`**).
- **`domain_adapt.py train`** → `results/stage2/<run_name>/` (default **`transfer_iwdan_weighted__...`** if **`--iwdan`**, else **`transfer_dann__...`**).

Checkpoints and TensorBoard events live **directly under that run folder** (no extra `MMDD/HHMMSS` nesting). Full convention and **which folder is your canonical balanced Stage 1 / Stage 2** are in **[docs/training_runs.md](docs/training_runs.md)**.

**Not moved**: `results/diagnostics/` (eval outputs). Prefer **`artifacts/class_weights/`** over ad hoc JSON under `results/`.


