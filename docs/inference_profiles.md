# Inference profiles (JSON → PyG and optional Graphormer biases)

This document matches `scripts/inference/json_to_brepmfr_pyg.py` and the collator / `GraphAttnBias` behavior in `BrepMFR_PyG`.

## Profiles

| Profile | Stored `d2_distance` / `angle_distance` | Stored `spatial_pos` / `edge_path` / `attn_bias` | Collated `batch_data` | `GraphAttnBias` |
|--------|----------------------------------------|--------------------------------------------------|------------------------|-----------------|
| `full` | Yes (dense N×N×64) | Yes | All tensors | A1 + A2 + A3 (multi-hop) |
| `no_a2` | **Omitted** (`has_a2=False`) | Yes | `d2_distance`/`angle_distance` are `None` | A1 + A3; A2 skipped |
| `lite` | Omitted | **Omitted** (`has_a1=False`, `has_a3=False`) | `spatial_pos`/`edge_path`/`None`; `attn_bias` synthesized in collator | A2 skipped; A1/A3 skipped |

**Homogeneous batches:** `data/collator.py` requires every graph in a batch to agree on `has_a2`, `has_a1`, and `has_a3`. Do not mix `full` and `lite` graphs in one batch. Use `pt_subdir` (see `data/dataset.py`) so file lists point at one layout only.

**Legacy `.pt`:** Graphs without `has_a2` / `has_a1` / `has_a3` infer flags from which tensors exist (backward compatible with older caches that store dense all-zero A2).

**Numerical note:** Skipping the A2 MLP branch is **not** identical to feeding all-zero histograms through `NonLinear` + `BatchNorm1d`. Treat `no_a2` / `lite` as deployment modes; expect metric drift unless you re-tune.

## Optimized converter (`json_to_brepmfr_pyg_optimized.py`)

Same profiles and collator semantics; differences are compact dtypes (e.g. `uint8` `spatial_pos`), no stored `attn_bias`, optional **NPZ pre-BFS cache**, and a **direct NPZ→tensor** path on cache hits (no reconstructed `faces` / `edges` dicts).

- **NPZ cache:** `--use-npz-cache --cache-dir DIR` writes `<stem>.npz` after node/edge arrays and `final_src`/`final_dst` are built; on a hit, loads arrays and runs BFS + PyG assembly. Default hit path is **direct**; `--no-npz-direct` forces the slower dict-reconstruction route for parity checks.
- **Benchmarks:** `--bench-npz-cache` (requires `--json-dir`, `--pt-out-dir`, `--cache-dir`) runs JSON→NPZ+`.pt` then NPZ-direct→`.pt` per file and prints summed timing buckets (`io_load`, `tensor_prep`, `shortest_path_a1_a3`, `pyg_pack`, `npz_cache_write`, `torch_save`, `total_wall`). `--bench-scipy-bfs` compares SciPy CSR all-pairs hop counts vs **serial** prefix BFS (skips graphs with `N > --bench-scipy-max-n`, default 800, to avoid O(N³)-style wall time on huge meshes).

CLI flags mirror the base script plus `--profile`, `--cache-dir`, `--use-npz-cache`, `--rebuild-cache`, `--legacy-bfs`, `--selftest-bfs`, and the options above.

## CLI (`json_to_brepmfr_pyg.py`)

- `--inference_profile {full,no_a2,lite}` (default `full`).
- `--skip_a2`: deprecated alias for `no_a2` (overrides profile when set).
- `--max_edge_path_len` (default `16`): BFS / `edge_path` last dimension; align with training `multi_hop_max_dist`.
- `--float16_storage`: store `node_data`, `edge_data`, `face_area` as `float16`; collator promotes to `float32` before the encoder.
- `--shortest_path_workers N` (`N>1`, default `0`): parallelize **A1** BFS over source nodes for `full` / `no_a2` when `N≥64` and `cpu_count≥2`. Does not apply to `lite`. Large graphs on many-core machines see the largest win.

## JSON → `.pt` runtime

- **`no_a2` vs `full`:** Both still run all-pairs shortest paths and build `edge_path`; only A2 histogram tensors are skipped. Expect similar CPU time to `full` for the same JSON.
- **`lite`:** Skips shortest-path work entirely (fastest ingest) but the checkpoint must be used with the optional-bias encoder path (no A1/A3 bias terms).
- **I/O:** Reading huge JSON from a network drive (e.g. `Y:`) and writing `.pt` there dominates wall time on some setups; copying JSON to a local SSD first often helps more than micro-optimizations.

## GPU memory (`full` / `no_a2` with A3)

The multi-hop edge bias path (`GraphAttnBias`, A3) gathers edge features along a dense tensor of length **≈ N×N×`multi_hop_max_dist`** per graph. Large mechanical meshes (many thousands of faces) can request **hundreds of GiB** and fail with `CUDA out of memory` at `edge_feature[dim_0, edge_path]`.

- **Inference scripts** (`run_pyg_inference.py`, `export_uv_json_pred.py`, `step_infer_features.py`) default to **`--max_nodes_for_a3 768`**: A3 is skipped (with a one-time warning) when the padded node count exceeds that cap; A1 spatial bias still runs. Use **`--max_nodes_for_a3 0`** to disable the cap if you have enough GPU memory and need full A3 on huge graphs.
- **Training** leaves the cap unset (`None` = no limit), so behavior matches earlier code unless you add `max_nodes_for_a3` to your training args.
- **Alternative:** Regenerate graphs with **`inference_profile=lite`**, which omits `edge_path` / A1 so memory scales with edges, not N².

## TensorBoard tracing

`models/tb_graph_utils.batch_to_flat` requires a **dense** batch (all `TRACE_BATCH_KEYS` tensors present). Use full-graph data for `add_graph`; lite / no_a2 batches are not supported there by design.

## References

- Dense vs sparse graph transformer scaling trade-offs: [Scaling Graph Transformers (survey)](https://arxiv.org/html/2508.17175v1).
