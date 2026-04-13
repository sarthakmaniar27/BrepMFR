import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import dgl
from dgl.data.utils import load_graphs


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {path} | {e}")
        return None


def _tensor_to_nested_list(t: torch.Tensor) -> Any:
    """
    Convert a tensor to JSON-serializable nested Python lists.
    - moves to CPU
    - detaches
    - converts float/int safely
    """
    if not isinstance(t, torch.Tensor):
        return t
    return t.detach().cpu().tolist()


def _infer_uv_grid(uv_tensor: torch.Tensor) -> Tuple[List, Dict[str, Any]]:
    """
    Try to interpret the per-face uv_tensor as a 5x5 grid (typical BrepMFR UV grid).
    Handles common layouts:
      - [25, C] -> reshape to [5,5,C]
      - [5,5,C] -> use directly
      - [C, 25] or [C,5,5] -> attempt permute
      - fallback: return raw list
    Returns:
      (uv_grid_as_list, meta)
    """
    meta: Dict[str, Any] = {"original_shape": list(uv_tensor.shape)}

    t = uv_tensor
    if t.ndim == 3 and t.shape[0] == 5 and t.shape[1] == 5:
        meta["interpreted_as"] = "[5,5,C]"
        return _tensor_to_nested_list(t), meta

    if t.ndim == 2:
        # [25, C] -> [5,5,C]
        if t.shape[0] == 25:
            c = t.shape[1]
            meta["interpreted_as"] = "[25,C] -> [5,5,C]"
            return _tensor_to_nested_list(t.reshape(5, 5, c)), meta

        # [C, 25] -> transpose -> [25,C] -> [5,5,C]
        if t.shape[1] == 25:
            c = t.shape[0]
            meta["interpreted_as"] = "[C,25] -> [25,C] -> [5,5,C]"
            t2 = t.transpose(0, 1).contiguous()  # [25, C]
            return _tensor_to_nested_list(t2.reshape(5, 5, c)), meta

    if t.ndim == 3:
        # [C,5,5] -> [5,5,C]
        if t.shape[1] == 5 and t.shape[2] == 5:
            meta["interpreted_as"] = "[C,5,5] -> [5,5,C]"
            t2 = t.permute(1, 2, 0).contiguous()
            return _tensor_to_nested_list(t2), meta

        # [25,1,C] weird cases: squeeze and retry
        if 25 in t.shape:
            meta["interpreted_as"] = "squeezed_fallback"
            t2 = t.squeeze()
            if isinstance(t2, torch.Tensor) and t2.ndim >= 2:
                return _infer_uv_grid(t2)

    # Fallback: give raw data as-is
    meta["interpreted_as"] = "raw_fallback"
    return _tensor_to_nested_list(t), meta


def extract_uv_points_for_labeled_faces(
    bin_path: Path,
    label_json_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Extract UV grids for faces whose labels != 0.
    Assumes labels list aligns with face/node indices in the DGL graph.
    """
    label_data = _safe_read_json(label_json_path)
    if not label_data:
        return None

    labels = label_data.get("labels", None)
    if labels is None or not isinstance(labels, list):
        print(f"[ERROR] Missing/invalid 'labels' list in: {label_json_path}")
        return None

    try:
        graphs, _ = load_graphs(str(bin_path))
    except Exception as e:
        print(f"[ERROR] Failed to load BIN graph: {bin_path} | {e}")
        return None

    if not graphs:
        print(f"[ERROR] No graphs found in: {bin_path}")
        return None

    g = graphs[0]

    if "x" not in g.ndata:
        print(f"[ERROR] g.ndata['x'] not found in graph from: {bin_path}")
        return None

    x = g.ndata["x"]  # expected per-face uv grid tensor
    num_faces = g.num_nodes()

    # Validate alignment
    if len(labels) != num_faces:
        print(
            f"[WARN] Label count != num faces for {bin_path.name}: "
            f"labels={len(labels)} vs faces={num_faces}. "
            f"Will process min(len(labels), num_faces)."
        )

    n = min(len(labels), num_faces)

    faces_out: List[Dict[str, Any]] = []
    for face_idx in range(n):
        lab = labels[face_idx]
        if isinstance(lab, bool):  # avoid True/False behaving like ints
            lab = int(lab)

        if not isinstance(lab, (int, float)):
            continue

        if int(lab) == 0:
            continue

        uv_tensor = x[face_idx]
        uv_grid, meta = _infer_uv_grid(uv_tensor)

        faces_out.append(
            {
                "face_index": face_idx,
                "label": int(lab),
                "uv_grid": uv_grid,          # usually [5][5][C] after inference
                "uv_meta": meta,             # shapes + how it was interpreted
            }
        )

    return {
        "file": bin_path.stem,
        "bin_path": str(bin_path),
        "label_path": str(label_json_path),
        "num_faces_in_graph": int(num_faces),
        "num_labels_in_json": int(len(labels)),
        "num_labeled_faces": int(len(faces_out)),
        "faces": faces_out,
    }


def main():
    # Your paths (as requested)
    bin_dir = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\authors_data\bin")
    label_dir = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\authors_data\label")
    out_dir = Path(r"Z:\uv_json")

    out_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(bin_dir.glob("*.bin"))
    if not bin_files:
        print(f"[ERROR] No .bin files found in: {bin_dir}")
        return

    ok = 0
    skipped = 0

    for bin_path in bin_files:
        label_path = label_dir / f"{bin_path.stem}.json"
        if not label_path.exists():
            print(f"[SKIP] Missing label json for {bin_path.name}: {label_path}")
            skipped += 1
            continue

        result = extract_uv_points_for_labeled_faces(bin_path, label_path)
        if result is None:
            print(f"[SKIP] Failed processing: {bin_path.name}")
            skipped += 1
            continue

        out_path = out_dir / f"{bin_path.stem}.json"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"[OK] Wrote: {out_path} | labeled_faces={result['num_labeled_faces']}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] Failed writing output JSON: {out_path} | {e}")
            skipped += 1

    print(f"\nDONE. ok={ok}, skipped={skipped}, total_bins={len(bin_files)}")


if __name__ == "__main__":
    # Optional: make prints more readable if you debug interactively
    torch.set_printoptions(threshold=10_000, linewidth=200, precision=4, sci_mode=False)
    main()
