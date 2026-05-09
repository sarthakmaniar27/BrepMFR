# Class weight JSON artifacts

Canonical copies of **`scripts/training/compute_class_weights.py`** output live here so weights are **version-controlled** (unlike ephemeral runs under ignored `results/`).

## Layout

| Directory | Purpose | CLI consumption |
|-----------|---------|-----------------|
| `stage1/` | Per-class CE loss weights (`weights` key) plus `counts` | `segmentation.py --class_weights_path` |
| `stage2_iwdan/` | Class **priors** from the same JSON format (`counts`; ratios used for IW) | `domain_adapt.py --iwdan` with `--iwdan_source_priors` / `--iwdan_target_priors` |

Stage 2 files are logically separate from Stage 1: use **distinct filenames** so IWDAN prior JSONs are never confused with Stage 1 training weights JSONs.

Example frozen files:

- `stage1/source_train_alpha05.json` — Stage 1 CE weights
- `stage2_iwdan/source_train_priors.json`, `stage2_iwdan/target_train_priors.json` — IWDAN (same JSON schema; counts used as priors)


From repo root:

```powershell
python scripts/training/compute_class_weights.py `
  --dataset_path "Z:/path/to/source_dataset" `
  --split train --num_classes 25 --alpha 0.5 `
  --out "artifacts/class_weights/stage1/source_train_alpha05.json"
```

(Optional scratch output to `results/class_weights/` is fine; copy the JSON you intend to freeze into `artifacts/`.)
