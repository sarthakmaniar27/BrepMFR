"""
Validate our regenerated CADSynth .bin files against the authors' originals.

Compares each file pair (ours has `_101` suffix stripped) on:

  1. File existence / pairing coverage.
  2. DGL graph topology: node count, edge count, edge connectivity (src/dst).
  3. Node features (ndata):
       x (uvgrid samples)   - shape/dtype match; distribution stats
       z (face_type)        - discrete, expect exact match
       y (face_area)        - continuous, expect close
       l (face_loop count)  - discrete, expect exact match
       a (face_adj count)   - discrete, expect exact match
       f (feature label)    - discrete, expect exact match
  4. Edge features (edata):
       x (uv samples on edges) - shape/dtype match
       t (edge_type)           - discrete, expect exact match
       l (edge_len)            - continuous, expect close
       a (edge_ang)            - continuous, expect close
       c (edge_conv)           - discrete, expect exact match
  5. Label JSON: face_label list equality (topology assumed node-aligned).

Usage:
    python scripts/validate_cadsynth_vs_authors.py [--n 300] [--seed 42]

Outputs a rich console report plus writes a JSON summary next to the script.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from dgl.data.utils import load_graphs

OURS_BIN = Path(r"Z:\Experiment6\source_dataset\output\bin")
OURS_LBL = Path(r"Z:\Experiment6\source_dataset\output\label")
AUTH_BIN = Path(r"Z:\authors_data\bin")
AUTH_LBL = Path(r"Z:\authors_data\label")

NDATA_FIELDS = ["x", "z", "y", "l", "a", "f"]
EDATA_FIELDS = ["x", "t", "l", "a", "c"]
DISCRETE_NDATA = {"z", "l", "a", "f"}
DISCRETE_EDATA = {"t", "c"}

ATOL = 1e-5
RTOL = 1e-4


def stem_of_ours(p: Path) -> str:
    # "00000007_101.bin" -> "00000007"
    name = p.stem
    if name.endswith("_101"):
        return name[:-4]
    return name


def build_index() -> Tuple[Dict[str, Path], Dict[str, Path], Dict[str, Path], Dict[str, Path]]:
    print("[index] scanning bin folders...", flush=True)
    t0 = time.time()
    ours_bin: Dict[str, Path] = {}
    for p in OURS_BIN.glob("*.bin"):
        ours_bin[stem_of_ours(p)] = p
    print(f"[index]   ours   bin: {len(ours_bin):,}  ({time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    auth_bin: Dict[str, Path] = {}
    for p in AUTH_BIN.glob("*.bin"):
        auth_bin[p.stem] = p
    print(f"[index]   auth   bin: {len(auth_bin):,}  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    ours_lbl: Dict[str, Path] = {}
    for p in OURS_LBL.glob("*.json"):
        name = p.stem
        if name.endswith("_101"):
            name = name[:-4]
        ours_lbl[name] = p
    print(f"[index]   ours label: {len(ours_lbl):,}  ({time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    auth_lbl: Dict[str, Path] = {}
    for p in AUTH_LBL.glob("*.json"):
        auth_lbl[p.stem] = p
    print(f"[index]   auth label: {len(auth_lbl):,}  ({time.time()-t0:.1f}s)", flush=True)
    return ours_bin, auth_bin, ours_lbl, auth_lbl


def tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    tt = t.detach().cpu()
    if tt.dtype.is_floating_point:
        arr = tt.float().numpy().reshape(-1)
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
    arr = tt.long().numpy().reshape(-1)
    if arr.size == 0:
        return {"n": 0}
    cnt = Counter(arr.tolist())
    top = cnt.most_common(5)
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "unique": len(cnt),
        "top5": top,
    }


def _as_long_1d(a: torch.Tensor) -> np.ndarray:
    return a.detach().cpu().long().numpy().reshape(-1)


def _as_float_nd(a: torch.Tensor) -> np.ndarray:
    return a.detach().cpu().float().numpy()


def compare_discrete(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    aa = _as_long_1d(a)
    bb = _as_long_1d(b)
    if aa.shape != bb.shape:
        return {"equal": False, "reason": "shape_mismatch", "a_shape": tuple(a.shape), "b_shape": tuple(b.shape)}
    eq = np.array_equal(aa, bb)
    if eq:
        return {"equal": True, "same_multiset": True, "n": int(aa.size)}
    # permutation-invariant: same multiset?
    same_multiset = Counter(aa.tolist()) == Counter(bb.tolist())
    diff_count = int(np.sum(aa != bb))
    return {
        "equal": False,
        "same_multiset": bool(same_multiset),
        "n": int(aa.size),
        "diff_count": diff_count,
        "diff_frac": diff_count / max(1, aa.size),
        "a_stats": tensor_stats(a),
        "b_stats": tensor_stats(b),
    }


def compare_continuous(a: torch.Tensor, b: torch.Tensor, atol: float = ATOL, rtol: float = RTOL) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {
            "close": False,
            "reason": "shape_mismatch",
            "a_shape": tuple(a.shape),
            "b_shape": tuple(b.shape),
        }
    aa = _as_float_nd(a).reshape(-1)
    bb = _as_float_nd(b).reshape(-1)
    if aa.size == 0:
        return {"close": True, "close_after_sort": True, "n": 0}
    allclose = np.allclose(aa, bb, atol=atol, rtol=rtol)
    delta = aa - bb
    abs_delta = np.abs(delta)
    # permutation-invariant: same sorted values?
    sorted_a = np.sort(aa)
    sorted_b = np.sort(bb)
    close_after_sort = np.allclose(sorted_a, sorted_b, atol=atol, rtol=rtol)
    sorted_delta = np.abs(sorted_a - sorted_b)
    return {
        "close": bool(allclose),
        "close_after_sort": bool(close_after_sort),
        "n": int(aa.size),
        "max_abs_err": float(abs_delta.max()),
        "mean_abs_err": float(abs_delta.mean()),
        "max_abs_err_after_sort": float(sorted_delta.max()),
        "mean_abs_err_after_sort": float(sorted_delta.mean()),
        "a_stats": tensor_stats(a),
        "b_stats": tensor_stats(b),
    }


def load_graph(p: Path):
    (gs, _lbls) = load_graphs(str(p))
    if len(gs) != 1:
        raise RuntimeError(f"Expected 1 graph in {p}, got {len(gs)}")
    return gs[0]


def try_node_alignment(g_o, g_a) -> Optional[np.ndarray]:
    """
    Try to find a permutation pi such that g_o.ndata[*][pi] == g_a.ndata[*]
    using a tuple of (face_type z, face_loop l, face_adj a, label_feature f,
    rounded face_area y). Returns pi as a numpy array (len N) or None if the
    tuple keys are not unique (cannot be aligned uniquely).
    """
    if g_o.num_nodes() != g_a.num_nodes():
        return None
    n = g_o.num_nodes()
    if n == 0:
        return np.array([], dtype=np.int64)

    def key_of(g, i):
        return (
            int(g.ndata["z"][i].item()) if "z" in g.ndata else -1,
            int(g.ndata["l"][i].item()) if "l" in g.ndata else -1,
            int(g.ndata["a"][i].item()) if "a" in g.ndata else -1,
            int(g.ndata["f"][i].item()) if "f" in g.ndata else -1,
            round(float(g.ndata["y"][i].item()), 4) if "y" in g.ndata else 0.0,
        )

    keys_o = [key_of(g_o, i) for i in range(n)]
    keys_a = [key_of(g_a, i) for i in range(n)]
    # build dict: key -> list of indices in auth
    idx_a: Dict[Tuple, List[int]] = defaultdict(list)
    for i, k in enumerate(keys_a):
        idx_a[k].append(i)
    pi = np.full(n, -1, dtype=np.int64)
    for i, k in enumerate(keys_o):
        pool = idx_a.get(k)
        if not pool:
            return None
        pi[i] = pool.pop(0)
    return pi


def compare_pair(stem: str, p_ours: Path, p_auth: Path, l_ours: Optional[Path], l_auth: Optional[Path]) -> Dict[str, Any]:
    r: Dict[str, Any] = {"stem": stem}
    try:
        g_o = load_graph(p_ours)
        g_a = load_graph(p_auth)
    except Exception as e:
        r["error"] = f"load_error: {e}"
        return r

    r["num_nodes_ours"] = int(g_o.num_nodes())
    r["num_nodes_auth"] = int(g_a.num_nodes())
    r["num_edges_ours"] = int(g_o.num_edges())
    r["num_edges_auth"] = int(g_a.num_edges())
    r["node_count_match"] = r["num_nodes_ours"] == r["num_nodes_auth"]
    r["edge_count_match"] = r["num_edges_ours"] == r["num_edges_auth"]

    # Attempt node-level permutation alignment using discrete invariants +
    # rounded face_area. Re-run comparison under the permutation if successful.
    pi = try_node_alignment(g_o, g_a)
    r["node_alignment_found"] = pi is not None

    # edge connectivity (treated as unordered multiset of (src,dst) pairs)
    s_o, d_o = g_o.edges()
    s_a, d_a = g_a.edges()
    edges_o = np.stack([s_o.numpy(), d_o.numpy()], axis=1)
    edges_a = np.stack([s_a.numpy(), d_a.numpy()], axis=1)
    # canonicalise by sorting each row then lexsort
    def canon(e):
        return np.array(sorted(map(tuple, e.tolist())))
    try:
        ec_o = canon(edges_o)
        ec_a = canon(edges_a)
        if ec_o.shape == ec_a.shape and np.array_equal(ec_o, ec_a):
            r["edge_topology_match"] = True
        else:
            r["edge_topology_match"] = False
            # try the undirected canonicalisation: sort each pair internally
            def canon_undir(e):
                return np.array(sorted(tuple(sorted(p)) for p in e.tolist()))
            try:
                ecu_o = canon_undir(edges_o)
                ecu_a = canon_undir(edges_a)
                r["edge_topology_match_undirected"] = bool(
                    ecu_o.shape == ecu_a.shape and np.array_equal(ecu_o, ecu_a)
                )
            except Exception:
                r["edge_topology_match_undirected"] = False
    except Exception as e:
        r["edge_topology_match"] = False
        r["edge_topology_error"] = str(e)

    # ndata
    r["ndata"] = {}
    for k in NDATA_FIELDS:
        if k not in g_o.ndata and k not in g_a.ndata:
            r["ndata"][k] = {"missing_both": True}
            continue
        if k not in g_o.ndata or k not in g_a.ndata:
            r["ndata"][k] = {
                "missing_ours": k not in g_o.ndata,
                "missing_auth": k not in g_a.ndata,
            }
            continue
        a = g_o.ndata[k]
        b = g_a.ndata[k]
        entry: Dict[str, Any] = {
            "shape_ours": tuple(a.shape),
            "shape_auth": tuple(b.shape),
            "dtype_ours": str(a.dtype),
            "dtype_auth": str(b.dtype),
        }
        if k in DISCRETE_NDATA:
            entry["cmp"] = compare_discrete(a, b)
        else:
            entry["cmp"] = compare_continuous(a, b)
        # permutation-aligned comparison (ours reindexed by pi)
        if pi is not None and a.shape[0] == pi.shape[0]:
            a_perm = a[torch.from_numpy(pi).long()]
            if k in DISCRETE_NDATA:
                entry["cmp_aligned"] = compare_discrete(a_perm, b)
            else:
                entry["cmp_aligned"] = compare_continuous(a_perm, b)
        r["ndata"][k] = entry

    # edata
    r["edata"] = {}
    for k in EDATA_FIELDS:
        if k not in g_o.edata and k not in g_a.edata:
            r["edata"][k] = {"missing_both": True}
            continue
        if k not in g_o.edata or k not in g_a.edata:
            r["edata"][k] = {
                "missing_ours": k not in g_o.edata,
                "missing_auth": k not in g_a.edata,
            }
            continue
        a = g_o.edata[k]
        b = g_a.edata[k]
        entry = {
            "shape_ours": tuple(a.shape),
            "shape_auth": tuple(b.shape),
            "dtype_ours": str(a.dtype),
            "dtype_auth": str(b.dtype),
        }
        if k in DISCRETE_EDATA:
            entry["cmp"] = compare_discrete(a, b)
        else:
            entry["cmp"] = compare_continuous(a, b)
        r["edata"][k] = entry

    # labels
    if l_ours is not None and l_auth is not None:
        try:
            with open(l_ours) as fh:
                j_o = json.load(fh)
            with open(l_auth) as fh:
                j_a = json.load(fh)
            r["label_json_ours_keys"] = list(j_o.keys()) if isinstance(j_o, dict) else None
            r["label_json_auth_keys"] = list(j_a.keys()) if isinstance(j_a, dict) else None

            def pick_labels(j):
                if isinstance(j, list):
                    return j
                if isinstance(j, dict):
                    for key in ("labels", "face_label", "face_labels"):
                        if key in j:
                            return j[key]
                return None

            lo = pick_labels(j_o)
            la = pick_labels(j_a)
            if lo is None or la is None:
                r["label_json_equal"] = None
                r["label_json_note"] = "no labels key recognized"
            else:
                r["label_ours_n"] = len(lo)
                r["label_auth_n"] = len(la)
                r["label_json_equal"] = bool(lo == la)
                r["label_multiset_equal"] = Counter(lo) == Counter(la)
                if not r["label_json_equal"] and len(lo) == len(la):
                    diffs = sum(1 for x, y in zip(lo, la) if x != y)
                    r["label_diff_count"] = diffs

                # Cross-check: does JSON labels == graph["f"] within each side?
                if "f" in g_o.ndata:
                    fo = g_o.ndata["f"].long().cpu().numpy().tolist()
                    r["labels_json_eq_graphF_ours"] = (list(lo) == list(fo))
                    r["labels_json_multiset_eq_graphF_ours"] = (Counter(lo) == Counter(fo))
                if "f" in g_a.ndata:
                    fa = g_a.ndata["f"].long().cpu().numpy().tolist()
                    r["labels_json_eq_graphF_auth"] = (list(la) == list(fa))
                    r["labels_json_multiset_eq_graphF_auth"] = (Counter(la) == Counter(fa))
        except Exception as e:
            r["label_json_error"] = str(e)

    # ------------------------------------------------------------------
    # Area-based node alignment (face_area is 98.5% identical across
    # ours and authors, so it is the most reliable alignment anchor).
    # When areas are unique, we can recover pi; then we test whether
    # label_feature and face_type agree AFTER alignment.
    # ------------------------------------------------------------------
    if "y" in g_o.ndata and "y" in g_a.ndata and g_o.num_nodes() == g_a.num_nodes():
        yo = g_o.ndata["y"].float().cpu().numpy()
        ya = g_a.ndata["y"].float().cpu().numpy()
        # Greedy nearest-area matching without replacement
        n = len(yo)
        used = np.zeros(n, dtype=bool)
        pi_y = np.full(n, -1, dtype=np.int64)
        max_diff = 0.0
        ambiguous = 0
        for i in range(n):
            best_j = -1
            best_d = np.inf
            second_d = np.inf
            for j in range(n):
                if used[j]:
                    continue
                d = abs(yo[i] - ya[j])
                if d < best_d:
                    second_d = best_d
                    best_d = d
                    best_j = j
                elif d < second_d:
                    second_d = d
            if best_j >= 0:
                used[best_j] = True
                pi_y[i] = best_j
                max_diff = max(max_diff, best_d)
                if second_d < best_d + 1e-4:
                    ambiguous += 1
        r["y_alignment_max_area_err"] = float(max_diff)
        r["y_alignment_ambiguous"] = int(ambiguous)
        # Apply pi_y to ours and compare f + z against authors
        if "f" in g_o.ndata and "f" in g_a.ndata:
            fo_aligned = g_o.ndata["f"].long().cpu().numpy()[pi_y]
            fa = g_a.ndata["f"].long().cpu().numpy()
            r["f_area_aligned_equal"] = bool(np.array_equal(fo_aligned, fa))
            r["f_area_aligned_diff"] = int(np.sum(fo_aligned != fa))
            # store pairs for cross-tab
            r["f_pairs_aligned"] = list(zip(fo_aligned.tolist(), fa.tolist()))
        if "z" in g_o.ndata and "z" in g_a.ndata:
            zo_aligned = g_o.ndata["z"].long().cpu().numpy()[pi_y]
            za = g_a.ndata["z"].long().cpu().numpy()
            r["z_area_aligned_equal"] = bool(np.array_equal(zo_aligned, za))
            r["z_area_aligned_diff"] = int(np.sum(zo_aligned != za))
            r["z_pairs_aligned"] = list(zip(zo_aligned.tolist(), za.tolist()))
        # JSON labels multiset vs graph["f"] already captured above, but also
        # compute: label_json ours aligned-by-pi_y vs authors json
        try:
            lo_list = pick_labels(j_o)
            la_list = pick_labels(j_a)
            if lo_list is not None and la_list is not None and len(lo_list) == n and len(la_list) == n:
                lo_arr = np.array(lo_list, dtype=np.int64)
                la_arr = np.array(la_list, dtype=np.int64)
                lo_aligned = lo_arr[pi_y]
                r["label_json_area_aligned_equal"] = bool(np.array_equal(lo_aligned, la_arr))
                r["label_json_area_aligned_diff"] = int(np.sum(lo_aligned != la_arr))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Edge-angle convention check: detect systematic range difference.
    # ------------------------------------------------------------------
    if "a" in g_o.edata and "a" in g_a.edata:
        ao = g_o.edata["a"].float().cpu().numpy().reshape(-1)
        aa = g_a.edata["a"].float().cpu().numpy().reshape(-1)
        r["edge_angle_ours_range"] = [float(ao.min()) if ao.size else 0.0, float(ao.max()) if ao.size else 0.0]
        r["edge_angle_auth_range"] = [float(aa.min()) if aa.size else 0.0, float(aa.max()) if aa.size else 0.0]
        # Test convention hypothesis: authors clip to [-pi/2, pi/2],
        # ours span a wider interval up to [-pi, pi]. Check if |ours| ever
        # exceeds pi/2+eps where authors never do.
        r["edge_angle_ours_exceeds_halfpi"] = int(np.sum(np.abs(ao) > (np.pi / 2.0 + 1e-3)))
        r["edge_angle_auth_exceeds_halfpi"] = int(np.sum(np.abs(aa) > (np.pi / 2.0 + 1e-3)))

    return r


def _empty_field_agg():
    return {
        "exact_match": 0,               # raw: same values in same order
        "close_match": 0,               # raw: allclose
        "multiset_match": 0,            # discrete: same set of values regardless of order
        "sort_close_match": 0,          # continuous: same sorted values within tol
        "aligned_exact": 0,             # after permutation alignment (discrete)
        "aligned_close": 0,             # after permutation alignment (continuous)
        "shape_mismatch": 0,
        "value_mismatch": 0,
        "max_abs_err": 0.0,
        "max_diff_frac": 0.0,
        "range_mismatch": 0,            # min/max ranges differ meaningfully
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {
        "total_checked": len(results),
        "load_errors": 0,
        "node_count_mismatch": 0,
        "edge_count_mismatch": 0,
        "edge_topology_mismatch": 0,
        "edge_topology_mismatch_even_undirected": 0,
        "node_alignment_found": 0,
        "ndata": {k: _empty_field_agg() for k in NDATA_FIELDS},
        "edata": {k: _empty_field_agg() for k in EDATA_FIELDS},
        "label_json_equal": 0,
        "label_json_multiset_equal": 0,
        "label_json_diff": 0,
        "label_json_missing": 0,
    }

    def bump_field(bucket: Dict[str, Any], k: str, cmp: Dict[str, Any], aligned: Optional[Dict[str, Any]], is_discrete: bool):
        if "shape_mismatch" in str(cmp.get("reason", "")):
            bucket[k]["shape_mismatch"] += 1
            return
        if is_discrete:
            if cmp.get("equal"):
                bucket[k]["exact_match"] += 1
            else:
                bucket[k]["value_mismatch"] += 1
                if cmp.get("same_multiset"):
                    bucket[k]["multiset_match"] += 1
                bucket[k]["max_diff_frac"] = max(bucket[k]["max_diff_frac"], float(cmp.get("diff_frac", 0.0)))
            if aligned is not None and aligned.get("equal"):
                bucket[k]["aligned_exact"] += 1
        else:
            if cmp.get("close"):
                bucket[k]["close_match"] += 1
            else:
                bucket[k]["value_mismatch"] += 1
                if cmp.get("close_after_sort"):
                    bucket[k]["sort_close_match"] += 1
                bucket[k]["max_abs_err"] = max(bucket[k]["max_abs_err"], float(cmp.get("max_abs_err", 0.0)))
                # range mismatch: max of |a_stats.min - b_stats.min| or |max-max|
                a_stats = cmp.get("a_stats") or {}
                b_stats = cmp.get("b_stats") or {}
                if "min" in a_stats and "min" in b_stats:
                    if abs(a_stats["min"] - b_stats["min"]) > 0.5 or abs(a_stats["max"] - b_stats["max"]) > 0.5:
                        bucket[k]["range_mismatch"] += 1
            if aligned is not None and aligned.get("close"):
                bucket[k]["aligned_close"] += 1

    for r in results:
        if "error" in r:
            agg["load_errors"] += 1
            continue
        if not r.get("node_count_match", True):
            agg["node_count_mismatch"] += 1
        if not r.get("edge_count_match", True):
            agg["edge_count_mismatch"] += 1
        if not r.get("edge_topology_match", True):
            agg["edge_topology_mismatch"] += 1
            if not r.get("edge_topology_match_undirected", False):
                agg["edge_topology_mismatch_even_undirected"] += 1
        if r.get("node_alignment_found"):
            agg["node_alignment_found"] += 1

        for k in NDATA_FIELDS:
            e = r.get("ndata", {}).get(k)
            if not e or "cmp" not in e:
                continue
            bump_field(agg["ndata"], k, e["cmp"], e.get("cmp_aligned"), k in DISCRETE_NDATA)

        for k in EDATA_FIELDS:
            e = r.get("edata", {}).get(k)
            if not e or "cmp" not in e:
                continue
            bump_field(agg["edata"], k, e["cmp"], None, k in DISCRETE_EDATA)

        lj = r.get("label_json_equal")
        if lj is True:
            agg["label_json_equal"] += 1
        elif lj is False:
            agg["label_json_diff"] += 1
        else:
            agg["label_json_missing"] += 1
        if r.get("label_multiset_equal"):
            agg["label_json_multiset_equal"] += 1

        # cross-consistency: JSON labels vs graph ndata["f"] within each side
        for side_key in ("ours", "auth"):
            if r.get(f"labels_json_eq_graphF_{side_key}") is True:
                agg.setdefault(f"json_eq_graphF_{side_key}", 0)
                agg[f"json_eq_graphF_{side_key}"] += 1
            if r.get(f"labels_json_multiset_eq_graphF_{side_key}") is True:
                agg.setdefault(f"json_multiset_eq_graphF_{side_key}", 0)
                agg[f"json_multiset_eq_graphF_{side_key}"] += 1

        # area-aligned agreement on label_feature + face_type + json labels
        for key, agg_key in (
            ("f_area_aligned_equal", "f_area_aligned_equal"),
            ("z_area_aligned_equal", "z_area_aligned_equal"),
            ("label_json_area_aligned_equal", "label_json_area_aligned_equal"),
        ):
            if r.get(key) is True:
                agg.setdefault(agg_key, 0)
                agg[agg_key] += 1

        # accumulate disagreement counts (for per-face %)
        for key in ("f_area_aligned_diff", "z_area_aligned_diff", "label_json_area_aligned_diff"):
            if key in r:
                agg.setdefault(f"total_{key}", 0)
                agg[f"total_{key}"] += int(r[key])

        # Confusion matrices
        for key in ("f_pairs_aligned", "z_pairs_aligned"):
            if key in r:
                cm_key = f"{key}_confmat"
                agg.setdefault(cm_key, defaultdict(int))
                for pair in r[key]:
                    agg[cm_key][pair] += 1

        # edge angle convention check
        if "edge_angle_ours_exceeds_halfpi" in r:
            agg.setdefault("edge_angle_ours_exceeds_halfpi_total", 0)
            agg["edge_angle_ours_exceeds_halfpi_total"] += int(r["edge_angle_ours_exceeds_halfpi"])
            agg.setdefault("edge_angle_auth_exceeds_halfpi_total", 0)
            agg["edge_angle_auth_exceeds_halfpi_total"] += int(r["edge_angle_auth_exceeds_halfpi"])
            agg.setdefault("files_with_ours_angles_beyond_halfpi", 0)
            if r["edge_angle_ours_exceeds_halfpi"] > 0:
                agg["files_with_ours_angles_beyond_halfpi"] += 1
            agg.setdefault("files_with_auth_angles_beyond_halfpi", 0)
            if r["edge_angle_auth_exceeds_halfpi"] > 0:
                agg["files_with_auth_angles_beyond_halfpi"] += 1

    return agg


def pretty_print_summary(agg: Dict[str, Any], n_pairs_available: int, extras: Dict[str, Any]) -> None:
    N = agg["total_checked"]
    pct = lambda x: f"{100.0*x/max(1,N):5.1f}%"
    print()
    print("=" * 120)
    print("  CADSYNTH BIN FILE VALIDATION: OURS vs AUTHORS")
    print("=" * 120)
    print(f"  Pairs available (matching stems): {n_pairs_available:,}")
    print(f"  Our bin files with stems missing in authors : {extras['ours_only']:,}")
    print(f"  Author bin files with stems missing in ours : {extras['auth_only']:,}")
    print(f"  Checked in this run               : {N:,}")
    print(f"  Load errors                       : {agg['load_errors']:,}")
    print()
    print("  Topology & alignment:")
    print(f"    Node count mismatches          : {agg['node_count_mismatch']:>7,}   {pct(agg['node_count_mismatch'])}")
    print(f"    Edge count mismatches          : {agg['edge_count_mismatch']:>7,}   {pct(agg['edge_count_mismatch'])}")
    print(f"    Edge topology mismatches (raw) : {agg['edge_topology_mismatch']:>7,}   {pct(agg['edge_topology_mismatch'])}")
    print(f"      ...even undirected           : {agg['edge_topology_mismatch_even_undirected']:>7,}   {pct(agg['edge_topology_mismatch_even_undirected'])}")
    print(f"    Node alignment pi found        : {agg['node_alignment_found']:>7,}   {pct(agg['node_alignment_found'])}")
    print()
    print("  Node features (ndata) - counts out of {:,} pairs checked:".format(N))
    header = f"    {'field':<8}{'disc':<6}{'exact':>9}{'close':>9}{'multiset':>10}{'sortCl':>8}{'alignedEx':>11}{'alignedCl':>11}{'shape!=':>9}{'value!=':>9}{'range!=':>9}{'maxErr':>12}"
    print(header)
    for k in NDATA_FIELDS:
        s = agg['ndata'][k]
        disc = "YES" if k in DISCRETE_NDATA else "no"
        print(f"    {k:<8}{disc:<6}{s['exact_match']:>9,}{s['close_match']:>9,}{s['multiset_match']:>10,}{s['sort_close_match']:>8,}{s['aligned_exact']:>11,}{s['aligned_close']:>11,}{s['shape_mismatch']:>9,}{s['value_mismatch']:>9,}{s['range_mismatch']:>9,}{s['max_abs_err']:>12.3g}")
    print()
    print("  Edge features (edata):")
    print(header)
    for k in EDATA_FIELDS:
        s = agg['edata'][k]
        disc = "YES" if k in DISCRETE_EDATA else "no"
        print(f"    {k:<8}{disc:<6}{s['exact_match']:>9,}{s['close_match']:>9,}{s['multiset_match']:>10,}{s['sort_close_match']:>8,}{s['aligned_exact']:>11,}{s['aligned_close']:>11,}{s['shape_mismatch']:>9,}{s['value_mismatch']:>9,}{s['range_mismatch']:>9,}{s['max_abs_err']:>12.3g}")
    print()
    print("  Label JSON comparison:")
    print(f"    equal (same order)      : {agg['label_json_equal']:>7,}   {pct(agg['label_json_equal'])}")
    print(f"    multiset equal          : {agg['label_json_multiset_equal']:>7,}   {pct(agg['label_json_multiset_equal'])}")
    print(f"    diff                    : {agg['label_json_diff']:>7,}   {pct(agg['label_json_diff'])}")
    print(f"    missing                 : {agg['label_json_missing']:>7,}   {pct(agg['label_json_missing'])}")
    print()
    print("  JSON vs graph['f'] consistency WITHIN each dataset:")
    print(f"    ours JSON == ours graph['f']          : {agg.get('json_eq_graphF_ours', 0):>7,}   {pct(agg.get('json_eq_graphF_ours', 0))}")
    print(f"    ours JSON multiset == graph['f'] mset : {agg.get('json_multiset_eq_graphF_ours', 0):>7,}   {pct(agg.get('json_multiset_eq_graphF_ours', 0))}")
    print(f"    auth JSON == auth graph['f']          : {agg.get('json_eq_graphF_auth', 0):>7,}   {pct(agg.get('json_eq_graphF_auth', 0))}")
    print(f"    auth JSON multiset == graph['f'] mset : {agg.get('json_multiset_eq_graphF_auth', 0):>7,}   {pct(agg.get('json_multiset_eq_graphF_auth', 0))}")
    print()
    print("  Area-anchored alignment (pi_y from greedy face_area matching):")
    print(f"    label_feature f == authors after pi_y : {agg.get('f_area_aligned_equal', 0):>7,}   {pct(agg.get('f_area_aligned_equal', 0))}")
    print(f"    face_type     z == authors after pi_y : {agg.get('z_area_aligned_equal', 0):>7,}   {pct(agg.get('z_area_aligned_equal', 0))}")
    print(f"    JSON labels     == authors after pi_y : {agg.get('label_json_area_aligned_equal', 0):>7,}   {pct(agg.get('label_json_area_aligned_equal', 0))}")
    print(f"    total disagreeing faces (f)           : {agg.get('total_f_area_aligned_diff', 0):>7,}")
    print(f"    total disagreeing faces (z)           : {agg.get('total_z_area_aligned_diff', 0):>7,}")
    print(f"    total disagreeing faces (json labels) : {agg.get('total_label_json_area_aligned_diff', 0):>7,}")
    print()
    # Confusion matrices for face labels and face types
    f_cm = agg.get("f_pairs_aligned_confmat")
    if f_cm:
        print()
        print("  Confusion matrix (label_feature 'f' after area alignment):")
        all_classes = sorted({c for pair in f_cm for c in pair})
        # Header
        hdr = "       " + "".join(f"{c:>6}" for c in all_classes) + "  <- authors"
        print(hdr)
        total = sum(f_cm.values())
        diag = sum(v for (o, a), v in f_cm.items() if o == a)
        for o in all_classes:
            row_total = sum(v for (oo, _), v in f_cm.items() if oo == o)
            cells = [f_cm.get((o, a), 0) for a in all_classes]
            marker = "  " + ("*" if any(c > 0 and i != all_classes.index(o) for i, c in enumerate(cells)) else "")
            print(f"   {o:>3}: " + "".join(f"{c:>6}" for c in cells) + f"   row_total={row_total}{marker}")
        print(f"   ours\\auth  (diagonal = agree = {diag:,} / {total:,} = {100.0*diag/max(1,total):.1f}%)")

    z_cm = agg.get("z_pairs_aligned_confmat")
    if z_cm:
        print()
        print("  Confusion matrix (face_type 'z' after area alignment):")
        all_classes = sorted({c for pair in z_cm for c in pair})
        hdr = "       " + "".join(f"{c:>6}" for c in all_classes) + "  <- authors"
        print(hdr)
        total = sum(z_cm.values())
        diag = sum(v for (o, a), v in z_cm.items() if o == a)
        for o in all_classes:
            row_total = sum(v for (oo, _), v in z_cm.items() if oo == o)
            cells = [z_cm.get((o, a), 0) for a in all_classes]
            print(f"   {o:>3}: " + "".join(f"{c:>6}" for c in cells) + f"   row_total={row_total}")
        print(f"   ours\\auth  (diagonal = agree = {diag:,} / {total:,} = {100.0*diag/max(1,total):.1f}%)")

    print()
    print("  Edge-angle convention check:")
    o_over = agg.get("edge_angle_ours_exceeds_halfpi_total", 0)
    a_over = agg.get("edge_angle_auth_exceeds_halfpi_total", 0)
    print(f"    # ours edges with |angle| > pi/2 (total across all checked files) : {o_over:,}")
    print(f"    # auth edges with |angle| > pi/2 (total across all checked files) : {a_over:,}")
    print(f"    files with ANY ours edge beyond [-pi/2,pi/2]   : {agg.get('files_with_ours_angles_beyond_halfpi', 0):>7,}   {pct(agg.get('files_with_ours_angles_beyond_halfpi', 0))}")
    print(f"    files with ANY auth edge beyond [-pi/2,pi/2]   : {agg.get('files_with_auth_angles_beyond_halfpi', 0):>7,}   {pct(agg.get('files_with_auth_angles_beyond_halfpi', 0))}")
    print("=" * 120)
    print()
    print("  Legend:")
    print("    exact      = same values in same order (strictest)")
    print("    close      = np.allclose with atol=1e-5, rtol=1e-4 (continuous only)")
    print("    multiset   = same bag of values ignoring order (discrete only)")
    print("    sortCl     = same sorted values within tolerance (continuous only, permutation-invariant)")
    print("    alignedEx  = exact match after remapping our nodes via inferred permutation pi (ndata only)")
    print("    alignedCl  = close match after remapping our nodes via inferred permutation pi (ndata only)")
    print("    range!=    = min/max of distribution differs by >0.5 (potential convention difference)")
    print("=" * 120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="Number of pairs to deep-compare")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategy", choices=["random", "head", "stratified"], default="stratified")
    ap.add_argument("--out", default=str(Path(__file__).parent / "cadsynth_validation_report.json"))
    ap.add_argument("--first-mismatch-examples", type=int, default=3, help="Save first N detailed mismatch reports for inspection")
    args = ap.parse_args()

    ours_bin, auth_bin, ours_lbl, auth_lbl = build_index()

    common = sorted(set(ours_bin) & set(auth_bin))
    ours_only = sorted(set(ours_bin) - set(auth_bin))
    auth_only = sorted(set(auth_bin) - set(ours_bin))
    print(f"[index] common pairs : {len(common):,}")
    print(f"[index] ours-only    : {len(ours_only):,}")
    print(f"[index] auth-only    : {len(auth_only):,}")
    if ours_only[:3]:
        print(f"[index]   sample ours-only stems: {ours_only[:3]}")
    if auth_only[:3]:
        print(f"[index]   sample auth-only stems: {auth_only[:3]}")

    rng = random.Random(args.seed)
    if args.strategy == "random":
        pick = rng.sample(common, min(args.n, len(common)))
    elif args.strategy == "head":
        pick = common[: args.n]
    else:
        # stratified: head + mid + tail + random inside
        n = min(args.n, len(common))
        head = common[: n // 4]
        tail = common[-(n // 4):]
        mid_start = len(common) // 2 - n // 8
        mid = common[mid_start: mid_start + n // 4]
        rest = [x for x in common if x not in set(head) | set(tail) | set(mid)]
        rand_pool = rng.sample(rest, n - len(head) - len(mid) - len(tail))
        pick = list(dict.fromkeys(head + mid + tail + rand_pool))[:n]

    print(f"[compare] deep-comparing {len(pick)} pairs (strategy={args.strategy})", flush=True)
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    bad_examples: List[Dict[str, Any]] = []
    for i, stem in enumerate(pick, 1):
        r = compare_pair(stem, ours_bin[stem], auth_bin[stem], ours_lbl.get(stem), auth_lbl.get(stem))
        results.append(r)
        is_bad = (
            "error" in r
            or not r.get("node_count_match", True)
            or not r.get("edge_count_match", True)
            or not r.get("edge_topology_match", True)
            or any(e.get("cmp", {}).get("equal") is False for e in r.get("ndata", {}).values() if isinstance(e, dict))
            or any(e.get("cmp", {}).get("equal") is False for e in r.get("edata", {}).values() if isinstance(e, dict))
            or r.get("label_json_equal") is False
        )
        if is_bad and len(bad_examples) < args.first_mismatch_examples:
            bad_examples.append(r)
        if i % 25 == 0 or i == len(pick):
            print(f"  [{i}/{len(pick)}]  elapsed={time.time()-t0:.1f}s", flush=True)

    agg = aggregate(results)
    extras = {"ours_only": len(ours_only), "auth_only": len(auth_only)}
    pretty_print_summary(agg, len(common), extras)

    # save full JSON (convert non-json-serializable items)
    agg_json = dict(agg)
    for k in list(agg_json.keys()):
        v = agg_json[k]
        if isinstance(v, defaultdict) or (isinstance(v, dict) and v and all(isinstance(kk, tuple) for kk in v.keys())):
            agg_json[k] = {f"{kk[0]}->{kk[1]}": vv for kk, vv in v.items()}
    # strip per-pair pair-lists to keep file small
    per_pair_json = []
    for r in results:
        rr = dict(r)
        rr.pop("f_pairs_aligned", None)
        rr.pop("z_pairs_aligned", None)
        per_pair_json.append(rr)
    out = {
        "config": {
            "n": args.n,
            "seed": args.seed,
            "strategy": args.strategy,
            "pairs_available": len(common),
            "ours_only": len(ours_only),
            "auth_only": len(auth_only),
        },
        "summary": agg_json,
        "per_pair": per_pair_json,
        "bad_examples": bad_examples,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[out] full report -> {args.out}")

    if bad_examples:
        print("\n" + "-" * 96)
        print(f"  First {len(bad_examples)} mismatch example(s) (detailed):")
        print("-" * 96)
        for b in bad_examples:
            print(json.dumps(b, indent=2, default=str))
            print("-" * 96)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
