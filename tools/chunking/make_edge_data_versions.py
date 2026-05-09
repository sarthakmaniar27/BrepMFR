# make_edge_data_versions.py
# Creates 3 JSON variants that modify edge["pt"] (edge UV curve samples) as requested.
#
# Version 1 (v1_json): keep 7 channels, but replace channel[6] curvature with safe inverse: 1/curvature
# Version 2 (v2_json): keep 7 channels, replace channel[6] with wrapped dihedral angle edge["a"] (same for all 5 samples)
# Version 3 (v3_json): drop channel[6] entirely => edge["pt"] becomes 5x6 flattened (length 30)
#
# IMPORTANT:
# - Your current json_to_brepmfr_bin.py expects edge["pt"] to be 5x7 (length 35). If you use v3_json,
#   you must also update the converter to reshape edges with C=6.
#
# Usage:
#   python make_edge_data_versions.py
#   (paths are hardcoded to exactly what you provided)
#
# Optional:
#   python make_edge_data_versions.py --eps 1e-8 --clamp_inv_abs_max 1000

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple
from typing import Union

import numpy as np
from tqdm import tqdm


# ----------------------------
# HARD-CODED PATHS (as requested)
# ----------------------------

ROOT_DIR = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment3\target_dataset\input")
INPUT_DIR = ROOT_DIR / "json_new_labels"

OUT_BASE = ROOT_DIR / "edge_data_versions"
OUT_V1 = OUT_BASE / "v1_json"
OUT_V2 = OUT_BASE / "v2_json"
OUT_V3 = OUT_BASE / "v3_json"


# ----------------------------
# Helpers
# ----------------------------
def _wrap_to_pi(x: float) -> float:
    # matches converter: (a + pi) % (2*pi) - pi
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _ensure_dirs():
    OUT_V1.mkdir(parents=True, exist_ok=True)
    OUT_V2.mkdir(parents=True, exist_ok=True)
    OUT_V3.mkdir(parents=True, exist_ok=True)


def _as_edge_grid_5x7(pt_list: Any) -> Tuple[np.ndarray, bool]:
    """
    Returns (grid, ok) where grid is float32 shape (5,7).
    Accepts list-like; validates total length == 35.
    """
    if not isinstance(pt_list, (list, tuple)):
        return np.empty((0, 0), dtype=np.float32), False
    if len(pt_list) != 35:
        return np.empty((0, 0), dtype=np.float32), False
    arr = np.asarray(pt_list, dtype=np.float32)
    if arr.size != 35:
        return np.empty((0, 0), dtype=np.float32), False
    grid = arr.reshape(5, 7)
    return grid, True


def _flatten_list(arr: np.ndarray) -> list:
    # Keep JSON clean (python floats)
    return [float(x) for x in arr.reshape(-1).tolist()]


# ----------------------------
# Main transformation
# ----------------------------


def process_all_jsons(eps: float, clamp_inv_abs_max: Union[float, None]) -> None:
    # Your code here
    _ensure_dirs()

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No .json files found in: {INPUT_DIR}")
        return

    # counters
    n_files = 0
    n_written_v1 = 0
    n_written_v2 = 0
    n_written_v3 = 0

    bad_schema = 0
    bad_pt_shape = 0

    v1_inverted_count = 0
    v1_preserved_minus1_count = 0
    v1_zero_or_tiny_count = 0
    v1_inf_or_nan_fixed = 0
    v1_clamped = 0

    v2_set_count = 0

    v3_dropped_count = 0

    for jp in tqdm(json_files, desc="Building edge_data_versions", unit="file"):
        n_files += 1
        try:
            data: Dict[str, Any] = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            bad_schema += 1
            continue

        if not isinstance(data, dict) or "edges" not in data or not isinstance(data["edges"], list):
            bad_schema += 1
            continue

        edges = data["edges"]

        # Prepare deep-ish copies (cheap, safe): copy top dict + edges list + each edge dict
        d1 = dict(data)
        d2 = dict(data)
        d3 = dict(data)

        d1_edges = [dict(e) if isinstance(e, dict) else e for e in edges]
        d2_edges = [dict(e) if isinstance(e, dict) else e for e in edges]
        d3_edges = [dict(e) if isinstance(e, dict) else e for e in edges]

        d1["edges"] = d1_edges
        d2["edges"] = d2_edges
        d3["edges"] = d3_edges

        # Transform edges
        for i, (e1, e2, e3) in enumerate(zip(d1_edges, d2_edges, d3_edges)):
            if not isinstance(e1, dict) or not isinstance(e2, dict) or not isinstance(e3, dict):
                bad_schema += 1
                continue

            pt = e1.get("pt", None)
            grid, ok = _as_edge_grid_5x7(pt)
            if not ok:
                bad_pt_shape += 1
                continue

            # ----------------------------
            # Version 1: invert curvature safely in channel 6
            # ----------------------------
            curv = grid[:, 6].copy()

            # preserve -1 exactly (your generator uses -1 as padding sentinel in multiple places)
            is_minus1 = np.isclose(curv, -1.0)
            v1_preserved_minus1_count += int(is_minus1.sum())

            valid = ~is_minus1
            tiny = valid & (np.abs(curv) <= eps)
            v1_zero_or_tiny_count += int(tiny.sum())

            inv = curv.copy()
            inv[tiny] = 0.0
            inv_valid = valid & ~tiny

            # 1/x
            inv[inv_valid] = 1.0 / inv[inv_valid]
            v1_inverted_count += int(inv_valid.sum())

            # fix inf/nan
            bad = ~np.isfinite(inv)
            if bad.any():
                v1_inf_or_nan_fixed += int(bad.sum())
                inv[bad] = 0.0

            # optional clamp for safety (prevents re-exploding from small curvature)
            if clamp_inv_abs_max is not None and clamp_inv_abs_max > 0:
                too_big = np.abs(inv) > clamp_inv_abs_max
                if too_big.any():
                    v1_clamped += int(too_big.sum())
                    inv = np.clip(inv, -clamp_inv_abs_max, clamp_inv_abs_max)

            grid_v1 = grid.copy()
            grid_v1[:, 6] = inv
            e1["pt"] = _flatten_list(grid_v1)

            # ----------------------------
            # Version 2: set channel 6 to dihedral angle e["a"] (wrapped), repeated for all 5 samples
            # ----------------------------
            ang = e2.get("a", None)
            if ang is None:
                # If missing, keep original channel 6 unchanged (still write file)
                pass
            else:
                try:
                    ang_f = _wrap_to_pi(float(ang))
                    grid_v2 = grid.copy()
                    grid_v2[:, 6] = ang_f
                    e2["pt"] = _flatten_list(grid_v2)
                    v2_set_count += 1
                except Exception:
                    pass

            # ----------------------------
            # Version 3: drop channel 6 -> 5x6
            # ----------------------------
            grid_v3 = grid[:, :6].copy()
            e3["pt"] = _flatten_list(grid_v3)
            v3_dropped_count += 1

        # Write outputs (always write, even if some edges had bad pt shapes; those edges remain unmodified)
        out1 = OUT_V1 / jp.name
        out2 = OUT_V2 / jp.name
        out3 = OUT_V3 / jp.name

        out1.write_text(json.dumps(d1, indent=2), encoding="utf-8")
        out2.write_text(json.dumps(d2, indent=2), encoding="utf-8")
        out3.write_text(json.dumps(d3, indent=2), encoding="utf-8")
        n_written_v1 += 1
        n_written_v2 += 1
        n_written_v3 += 1

    # concise summary
    print("\n[SUMMARY]")
    print(f"Input dir           : {INPUT_DIR}")
    print(f"Files found         : {n_files}")
    print(f"Written v1/v2/v3    : {n_written_v1}/{n_written_v2}/{n_written_v3}")
    print(f"Bad schema files    : {bad_schema}")
    print(f"Bad edge pt shapes  : {bad_pt_shape}  (expected edge['pt'] length == 35)")

    print("\n[V1 stats] (channel6 := 1/curvature, preserving -1, tiny->0)")
    print(f"  inverted samples  : {v1_inverted_count}")
    print(f"  preserved -1      : {v1_preserved_minus1_count}")
    print(f"  tiny/zero -> 0    : {v1_zero_or_tiny_count}")
    print(f"  inf/nan fixed     : {v1_inf_or_nan_fixed}")
    if clamp_inv_abs_max is not None:
        print(f"  clamped |inv|>{clamp_inv_abs_max}: {v1_clamped}")

    print("\n[V2 stats] (channel6 := wrapped edge['a'])")
    print(f"  edges updated      : {v2_set_count}")

    print("\n[V3 stats] (drop channel6; pt becomes 5x6)")
    print(f"  edges updated      : {v3_dropped_count}")

    print("\n[OUTPUT DIRS]")
    print(f"  v1_json: {OUT_V1}")
    print(f"  v2_json: {OUT_V2}")
    print(f"  v3_json: {OUT_V3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-8, help="Treat |curvature|<=eps as zero for inversion")
    ap.add_argument(
        "--clamp_inv_abs_max",
        type=float,
        default=None,
        help="Optional: clamp |1/curvature| to this max abs value (e.g., 1000). Default: no clamp.",
    )
    args = ap.parse_args()
    process_all_jsons(eps=args.eps, clamp_inv_abs_max=args.clamp_inv_abs_max)


if __name__ == "__main__":
    main()