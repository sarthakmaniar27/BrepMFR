import argparse
import sys
from typing import Dict, Any, Tuple, List

import torch
from dgl.data.utils import load_graphs


# -----------------------------
# Expected BrepMFR keys
# -----------------------------
REQUIRED_NDATA = ["x", "z", "y", "l", "a", "f"]
REQUIRED_EDATA = ["x", "t", "l", "a", "c"]
REQUIRED_META  = ["edges_path", "spatial_pos", "d2_distance", "angle_distance"]


def _as_int(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return x
    # sometimes saved as float by mistake
    return x.to(torch.int64)


def _basic_tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["dtype"] = str(t.dtype).replace("torch.", "")
    out["shape"] = list(t.shape)
    out["device"] = str(t.device)

    if t.numel() == 0:
        out["numel"] = 0
        return out

    out["numel"] = int(t.numel())

    if t.dtype.is_floating_point:
        out["nan"] = bool(torch.isnan(t).any().item())
        out["inf"] = bool(torch.isinf(t).any().item())
        out["min"] = float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).min().item())
        out["max"] = float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).max().item())
        out["mean"] = float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).mean().item())
    else:
        out["min"] = int(t.min().item())
        out["max"] = int(t.max().item())

    return out


def _check_present(graph, label: str, keys: List[str], kind: str) -> List[str]:
    missing = []
    store = graph.ndata if kind == "ndata" else graph.edata
    for k in keys:
        if k not in store:
            missing.append(f"[{label}] Missing {kind}['{k}']")
    return missing


def _check_present_meta(meta: Dict[str, torch.Tensor], label: str) -> List[str]:
    missing = []
    for k in REQUIRED_META:
        if k not in meta:
            missing.append(f"[{label}] Missing meta['{k}']")
    return missing


def _edges_bidirectional_ratio(src: torch.Tensor, dst: torch.Tensor) -> float:
    # ratio of edges that have reverse present
    # note: can be expensive for big graphs; but ok for sanity on one file.
    pairs = torch.stack([src, dst], dim=1)
    rev   = torch.stack([dst, src], dim=1)
    # hash pairs to 64-bit
    # key = u << 32 | v (assumes < 2^32 nodes)
    key_pairs = (pairs[:, 0].to(torch.int64) << 32) | pairs[:, 1].to(torch.int64)
    key_rev   = (rev[:, 0].to(torch.int64) << 32) | rev[:, 1].to(torch.int64)
    s = set(key_pairs.cpu().tolist())
    rev_hits = sum((k in s) for k in key_rev.cpu().tolist())
    return float(rev_hits) / float(len(key_rev)) if len(key_rev) else 1.0


def _validate_index_range(
    name: str,
    t: torch.Tensor,
    lo: int,
    hi: int,
    label: str,
    allow_zero: bool = True
) -> List[str]:
    """
    Checks t is integer-like and all values in [lo, hi] (inclusive).
    """
    errs = []
    ti = _as_int(t)

    if ti.numel() == 0:
        return errs

    tmin = int(ti.min().item())
    tmax = int(ti.max().item())

    if allow_zero and lo > 0:
        lo_eff = 0
    else:
        lo_eff = lo

    if tmin < lo_eff or tmax > hi:
        errs.append(
            f"[{label}] OUT-OF-RANGE: {name} min={tmin}, max={tmax}, expected in [{lo_eff}, {hi}]"
        )
    return errs


def inspect_one_bin(path: str, label: str) -> Tuple[Any, Dict[str, torch.Tensor], Dict[str, Any], List[str]]:
    graphs, meta = load_graphs(path)
    if len(graphs) != 1:
        raise RuntimeError(f"{path}: expected exactly 1 graph, got {len(graphs)}")

    g = graphs[0]
    errors: List[str] = []

    # Required key checks
    errors += _check_present(g, label, REQUIRED_NDATA, "ndata")
    errors += _check_present(g, label, REQUIRED_EDATA, "edata")
    errors += _check_present_meta(meta, label)

    # Collect stats for all tensors we care about
    stats: Dict[str, Any] = {}
    stats["num_nodes"] = int(g.num_nodes())
    stats["num_edges"] = int(g.num_edges())

    # Edge list sanity
    src, dst = g.edges()
    stats["bidirectional_ratio"] = _edges_bidirectional_ratio(src, dst)
    stats["src_min"] = int(src.min().item()) if src.numel() else None
    stats["src_max"] = int(src.max().item()) if src.numel() else None
    stats["dst_min"] = int(dst.min().item()) if dst.numel() else None
    stats["dst_max"] = int(dst.max().item()) if dst.numel() else None

    # Index bounds: src/dst must be in [0, num_nodes-1]
    if src.numel():
        if int(src.min().item()) < 0 or int(dst.min().item()) < 0:
            errors.append(f"[{label}] Negative node index found in edges()")
        if int(src.max().item()) >= g.num_nodes() or int(dst.max().item()) >= g.num_nodes():
            errors.append(
                f"[{label}] Edge node index exceeds num_nodes-1: "
                f"src_max={int(src.max().item())}, dst_max={int(dst.max().item())}, num_nodes={g.num_nodes()}"
            )

    # Tensor stats
    for k in REQUIRED_NDATA:
        stats[f"ndata.{k}"] = _basic_tensor_stats(g.ndata[k])

    for k in REQUIRED_EDATA:
        stats[f"edata.{k}"] = _basic_tensor_stats(g.edata[k])

    for k in REQUIRED_META:
        stats[f"meta.{k}"] = _basic_tensor_stats(meta[k])

    # Shape constraints that must hold
    N = g.num_nodes()
    E = g.num_edges()

    # Node geometry: [N, 5, 5, 7]
    if "x" in g.ndata:
        x = g.ndata["x"]
        if x.dim() != 4 or x.shape[0] != N or x.shape[-1] != 7:
            errors.append(f"[{label}] ndata['x'] shape expected [N,5,5,7]-like, got {list(x.shape)}")
    # Edge geometry: commonly [E, 5, 6] or [E, 5, 6?], but model uses rotate_uvgrid which expects [...,:3] and [...,3:6]
    if "x" in g.edata:
        ex = g.edata["x"]
        if ex.shape[0] != E:
            errors.append(f"[{label}] edata['x'] first dim must be E, got {list(ex.shape)}")

    # Scalar lengths
    for key in ["z", "y", "l", "a", "f"]:
        if key in g.ndata and g.ndata[key].shape[0] != N:
            errors.append(f"[{label}] ndata['{key}'] must have shape [N], got {list(g.ndata[key].shape)}")

    for key in ["t", "l", "a", "c"]:
        if key in g.edata and g.edata[key].shape[0] != E:
            errors.append(f"[{label}] edata['{key}'] must have shape [E], got {list(g.edata[key].shape)}")

    # Meta shapes
    if "spatial_pos" in meta:
        sp = meta["spatial_pos"]
        if sp.shape[0] != N or sp.shape[1] != N:
            errors.append(f"[{label}] meta['spatial_pos'] must be [N,N], got {list(sp.shape)}")
    if "edges_path" in meta:
        ep = meta["edges_path"]
        if ep.shape[0] != N or ep.shape[1] != N:
            errors.append(f"[{label}] meta['edges_path'] must be [N,N,max_dist], got {list(ep.shape)}")
    if "d2_distance" in meta:
        d2 = meta["d2_distance"]
        if d2.shape[0] != N or d2.shape[1] != N or (d2.dim() != 3 or d2.shape[2] != 64):
            errors.append(f"[{label}] meta['d2_distance'] must be [N,N,64], got {list(d2.shape)}")
    if "angle_distance" in meta:
        ad = meta["angle_distance"]
        if ad.shape[0] != N or ad.shape[1] != N or (ad.dim() != 3 or ad.shape[2] != 64):
            errors.append(f"[{label}] meta['angle_distance'] must be [N,N,64], got {list(ad.shape)}")

    # -----------------------------
    # Range checks (likely embedding indices)
    # NOTE: these limits are based on BrepMFR collator defaults:
    #   multi_hop_max_dist = 16
    #   spatial_pos_max    = 32
    #
    # If authors bins show higher maxima, update these.
    # -----------------------------
    if "z" in g.ndata:
        errors += _validate_index_range("ndata['z'] face_type", g.ndata["z"], 0, 7, label)
    if "f" in g.ndata:
        # dataset.py expects 0..24 for stage-1 (25 classes)
        errors += _validate_index_range("ndata['f'] label_feature", g.ndata["f"], 0, 24, label)
    if "t" in g.edata:
        errors += _validate_index_range("edata['t'] edge_type", g.edata["t"], 0, 7, label)
    if "c" in g.edata:
        # common: 0=concave, 1=convex, 2=flat (or similar)
        # if your data uses {-1,0,1} you must remap to 0..2
        errors += _validate_index_range("edata['c'] edge_convexity", g.edata["c"], 0, 2, label)

    if "spatial_pos" in meta:
        errors += _validate_index_range("meta['spatial_pos']", meta["spatial_pos"], 0, 32, label)

    if "edges_path" in meta:
        # graphormer-style multi-hop edge input uses hop distances up to multi_hop_max_dist (16)
        errors += _validate_index_range("meta['edges_path']", meta["edges_path"], 0, 16, label)

    # Float sanity
    for name, t in [
        ("ndata['x']", g.ndata.get("x", None)),
        ("edata['x']", g.edata.get("x", None)),
        ("meta['d2_distance']", meta.get("d2_distance", None)),
        ("meta['angle_distance']", meta.get("angle_distance", None)),
    ]:
        if t is None or t.numel() == 0:
            continue
        if t.dtype.is_floating_point:
            if torch.isnan(t).any() or torch.isinf(t).any():
                errors.append(f"[{label}] NaN/Inf found in {name}")

    return g, meta, stats, errors


def compare_stats(stats_a: Dict[str, Any], stats_b: Dict[str, Any], label_a: str, label_b: str) -> None:
    print("\n==================== HIGH-SIGNAL COMPARISON ====================")
    keys = [
        "num_nodes", "num_edges", "bidirectional_ratio",
        "ndata.x", "edata.x",
        "meta.edges_path", "meta.spatial_pos", "meta.d2_distance", "meta.angle_distance",
    ]

    def _shape_of(stats: Dict[str, Any], k: str):
        if k in ("num_nodes", "num_edges", "bidirectional_ratio"):
            return stats.get(k, None)
        return stats.get(k, {}).get("shape", None)

    for k in keys:
        ka = k.replace("ndata.", "ndata.").replace("edata.", "edata.").replace("meta.", "meta.")
        if k in ("ndata.x", "edata.x"):
            ka = k.replace(".", ".")
            a = stats_a.get(k.replace("ndata.x", "ndata.x").replace("edata.x", "edata.x"), None)

        a_val = _shape_of(stats_a, k)
        b_val = _shape_of(stats_b, k)
        print(f"{k:18s} | {label_a}: {a_val}  ||  {label_b}: {b_val}")


def main():
    authors_bin = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\authors_data\bin\00034236.bin"
    my_bin = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output\bin\00034236_101.bin"
    ap = argparse.ArgumentParser()
    ap.add_argument("--author_bin", default=authors_bin, required=False, help="Path to author .bin")
    ap.add_argument("--my_bin", default=my_bin, required=False, help="Path to your .bin")
    args = ap.parse_args()

    print("Loading and inspecting bins...\n")

    g_a, meta_a, stats_a, errs_a = inspect_one_bin(args.author_bin, "AUTHOR")
    g_m, meta_m, stats_m, errs_m = inspect_one_bin(args.my_bin, "MINE")

    # Print summary
    print("=============== AUTHOR STATS ===============")
    print(f"num_nodes={stats_a['num_nodes']}  num_edges={stats_a['num_edges']}  bidir_ratio={stats_a['bidirectional_ratio']:.3f}")
    print("===============  MY STATS   ===============")
    print(f"num_nodes={stats_m['num_nodes']}  num_edges={stats_m['num_edges']}  bidir_ratio={stats_m['bidirectional_ratio']:.3f}")

    # Print high-signal comparison
    compare_stats(stats_a, stats_m, "AUTHOR", "MINE")

    # Print errors
    if errs_a:
        print("\n==================== AUTHOR ERRORS ====================")
        for e in errs_a:
            print(e)

    if errs_m:
        print("\n==================== MY ERRORS (LIKELY CAUSE) ====================")
        for e in errs_m:
            print(e)

    # Strong hint if edges missing reverse direction
    if stats_m["bidirectional_ratio"] < 0.95 and stats_a["bidirectional_ratio"] >= 0.95:
        print("\n>>> NOTE: Your graph is not bidirectional like the author bin.")
        print(">>> BrepMFR CadSynth bins are typically stored with BOTH directions (u->v and v->u).")
        print(">>> Missing reverse edges changes E, edge features alignment, and multi-hop encodings.")

    # Strong hint if spatial_pos / edges_path exceed limits
    # (Even one out-of-range value can trigger CUDA device-side assert)
    print("\nDone.\n")
    if errs_m:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
