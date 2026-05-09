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

Heavy generated trees remain at `scripts/sorted_dumps/`, `scripts/sorted_dumps_full/`, `scripts/scan_reports/`.

All Python entrypoints use `bootstrap_path.py` at the repo root to put the project on `sys.path` regardless of nesting depth.
