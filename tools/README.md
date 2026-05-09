# Maintenance and dataset tools

Standalone utilities that are **not** part of the core training CLI (`segmentation.py`, `domain_adapt.py`). Run from repo root when they need filesystem paths relative to this project:

```powershell
cd C:\path\to\BrepMFR_PyG
python tools/bins/check_bins.py ...
```

## Layout

| Folder | Typical use |
|--------|--------------|
| `bins/` | DGL `.bin` inspection, validation, range scans |
| `labels/` | Label remapping, extraction, corrective passes |
| `chunking/` | Chunk create/combine/cleanup for bulk processing |
| `file_ops/` | Batch rename / reorder helpers for dataset files |
| `viz/` | DGL plotting and histogram scripts |
| `pipeline/` | JSON↔`.bin`, STEP↔graphs, UV extraction (`extract_uv_points.py`), `occwl` shims |

`tools/repo_root.py` exposes `REPO_ROOT` for tools that prepend the repository to `sys.path` after locating `bootstrap_path.py` / `segmentation.py` at the workspace root.
