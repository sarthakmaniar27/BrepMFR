# scan_bin_ranges.py
# Scan all .bin files and report GLOBAL min/max/count (and approximate unique counts) for EVERY feature key.
# Also validates edges_path indices against per-graph num_edges (common cause of CUDA device-side assert).

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Set, List, Tuple

import torch
from dgl.data.utils import load_graphs
from tqdm import tqdm

# ---------------------------- 
# CONFIG (edit these only)
# ----------------------------
ROOT_DIR = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\target_dataset\output\bin"
# ROOT_DIR = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output\bin"
# ROOT_DIR = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment3\target_dataset\output\bin"
FILE_PATTERN = ".bin"          # scans recursively for files ending with this suffix

UNIQUE_ENABLED = True          # keep approximate uniques (cap)
UNIQUE_CAP = 200_000           # cap per key (keeps memory bounded)
PRINT_UNIQUE_SAMPLES = False   # set True if you want to print small samples too


# ----------------------------
# Helpers
# ----------------------------
def _is_int_tensor(t: torch.Tensor) -> bool:
    return t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _safe_min_max(t: torch.Tensor):
    # Works for CPU tensors. (Your scan runs on CPU.)
    if t.numel() == 0:
        return None, None
    return t.min().item(), t.max().item()


def _flatten_for_uniques(t: torch.Tensor) -> torch.Tensor:
    # For uniques we only consider integer tensors
    return t.reshape(-1)


@dataclass
class Stat:
    dtype: Optional[str] = None
    min_v: Optional[float] = None
    max_v: Optional[float] = None
    count: int = 0
    uniques: Set[Any] = field(default_factory=set)

    def update(self, t: torch.Tensor):
        if t is None:
            return
        if not torch.is_tensor(t):
            return
        if t.numel() == 0:
            return

        # dtype
        if self.dtype is None:
            self.dtype = str(t.dtype)

        # min/max
        mn, mx = _safe_min_max(t)
        if mn is None:
            return
        if self.min_v is None or mn < self.min_v:
            self.min_v = mn
        if self.max_v is None or mx > self.max_v:
            self.max_v = mx

        # count
        self.count += int(t.numel())

        # uniques (approx)
        if UNIQUE_ENABLED and _is_int_tensor(t) and len(self.uniques) < UNIQUE_CAP:
            flat = _flatten_for_uniques(t)
            # Only add as many as needed to hit cap
            room = UNIQUE_CAP - len(self.uniques)
            if room <= 0:
                return
            # To avoid huge overhead, sample if tensor is massive
            if flat.numel() > 5_000_000:
                # deterministic-ish sampling
                idx = torch.linspace(0, flat.numel() - 1, steps=min(room, 50_000)).long()
                vals = flat[idx].tolist()
            else:
                vals = flat.tolist()
            for v in vals[:room]:
                self.uniques.add(int(v))


def _fmt_num(x, width=12):
    if x is None:
        return " " * (width - 3) + "N/A"
    # int-like?
    if abs(x - int(x)) < 1e-9:
        return f"{int(x):>{width}d}"
    return f"{x:>{width}.6g}"


def _fmt_int(x, width=12):
    if x is None:
        return " " * (width - 3) + "N/A"
    return f"{int(x):>{width}d}"


def _walk_bin_files(root_dir: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(FILE_PATTERN):
                out.append(os.path.join(r, fn))
    return sorted(out)


# ----------------------------
# Main scan
# ----------------------------
def main():
    print(f"[INFO] ROOT_DIR: {ROOT_DIR}")
    print(f"[INFO] FILE_PATTERN: *{FILE_PATTERN}")
    print(f"[INFO] UNIQUE_ENABLED: {UNIQUE_ENABLED} (cap={UNIQUE_CAP})")

    files = _walk_bin_files(ROOT_DIR)
    print(f"[INFO] Found {len(files)} bin files\n")

    # Global stats for every key we encounter
    stats: Dict[str, Stat] = {}

    # Extra validation: edges_path indices must be < num_edges (per graph)
    edge_path_oob = 0
    edge_path_oob_samples: List[Tuple[str, int, int]] = []  # (file, max_edge_id, num_edges)

    failed = 0
    loaded_ok = 0

    pbar = tqdm(files, desc="Scanning .bin files", unit="file")

    for fp in pbar:
        try:
            graphs, aux = load_graphs(fp)
            g = graphs[0]

            # ----------------------------
            # 1) Node (face) tensors (as stored in bin)
            # ----------------------------
            for k, t in g.ndata.items():
                key = f"ndata:{k}"
                stats.setdefault(key, Stat()).update(t)

            # ----------------------------
            # 2) Edge tensors (as stored in bin)
            # ----------------------------
            for k, t in g.edata.items():
                key = f"edata:{k}"
                stats.setdefault(key, Stat()).update(t)

            # ----------------------------
            # 3) Aux tensors (as stored in label dict)
            # ----------------------------
            if isinstance(aux, dict):
                for k, t in aux.items():
                    key = f"aux:{k}"
                    stats.setdefault(key, Stat()).update(t)

            # ----------------------------
            # 4) Derived (NOT stored): node_degree (face_degree)
            # ----------------------------
            # In your dataset.py you compute dense_adj.sum(dim=1), but in_degrees() is equivalent for this graph.
            deg = g.in_degrees()  # int64
            stats.setdefault("derived:node_degree", Stat()).update(deg)

            # ----------------------------
            # 5) Validate edges_path indices vs num_edges
            # ----------------------------
            if isinstance(aux, dict) and "edges_path" in aux:
                ep = aux["edges_path"]
                if torch.is_tensor(ep) and ep.numel() > 0:
                    # Your generator fills with -1 for padding. Ignore negatives for this check.
                    ep_valid = ep[ep >= 0]
                    if ep_valid.numel() > 0:
                        mx = int(ep_valid.max().item())
                        E = int(g.num_edges())
                        if mx >= E:
                            edge_path_oob += 1
                            if len(edge_path_oob_samples) < 10:
                                edge_path_oob_samples.append((fp, mx, E))

            loaded_ok += 1

        except Exception:
            failed += 1

    print(f"\n[INFO] Completed. Loaded OK: {loaded_ok}, Failed: {failed}\n")

    # ----------------------------
    # Friendly alias mapping (what your model/code calls these)
    # ----------------------------
    alias = [
        ("face_type",   "ndata:z"),
        ("face_area",   "ndata:y"),
        ("face_loop",   "ndata:l"),
        ("face_adj",    "ndata:a"),
        ("label_face",  "ndata:f"),
        ("node_data",   "ndata:x"),
        ("edge_type",   "edata:t"),
        ("edge_len",    "edata:l"),
        ("edge_ang",    "edata:a"),
        ("edge_conv",   "edata:c"),
        ("edge_data",   "edata:x"),
        ("edge_path",   "aux:edges_path"),
        ("spatial_pos", "aux:spatial_pos"),
        ("d2_distance", "aux:d2_distance"),
        ("angle_dist",  "aux:angle_distance"),
        ("face_degree", "derived:node_degree"),
    ]

    print("Alias mapping (your code name -> actual stored key):")
    for a, k in alias:
        print(f"  - {a:<12} -> {k}")
    print("")

    # ----------------------------
    # Print stats table (all keys found)
    # ----------------------------
    # Sort keys to keep output stable and readable
    keys_sorted = sorted(stats.keys())

    header = (
        f"{'KEY':<22}"
        f"{'DTYPE':<12}"
        f"{'MIN':>12}"
        f"{'MAX':>12}"
        f"{'COUNT':>14}"
        f"{'UNIQUE~':>12}"
    )
    print(header)
    print("-" * len(header))

    for k in keys_sorted:
        s = stats[k]
        uniq_n = len(s.uniques) if UNIQUE_ENABLED else 0
        print(
            f"{k:<22}"
            f"{(s.dtype or 'N/A'):<12}"
            f"{_fmt_num(s.min_v)}"
            f"{_fmt_num(s.max_v)}"
            f"{_fmt_int(s.count, width=14)}"
            f"{_fmt_int(uniq_n)}"
        )

    # ----------------------------
    # Embedding suggestions (categorical int tensors)
    # ----------------------------
    print("\nEmbedding size suggestion (safe rule: num_embeddings >= max+1 for categorical int features):")
    def suggest(name: str, key: str):
        s = stats.get(key)
        if s is None or s.max_v is None:
            print(f"  - {name}: (not found)")
            return
        mx = int(s.max_v)
        print(f"  - {name}: max={mx} => num_embeddings >= {mx + 1}   ({key})")

    # These are the ones you embed/index as categorical in the model
    suggest("face_type", "ndata:z")
    suggest("face_loop", "ndata:l")
    suggest("edge_type", "edata:t")
    suggest("edge_conv", "edata:c")
    suggest("spatial_pos", "aux:spatial_pos")

    # edges_path is not an embedding lookup; it's an index into edge features per-graph, so we validate separately below.
    s_ep = stats.get("aux:edges_path")
    if s_ep is not None and s_ep.min_v is not None and s_ep.max_v is not None:
        print(f"  - edge_path (aux:edges_path): min={int(s_ep.min_v)} max={int(s_ep.max_v)}  (note: -1 is padding in your generator)")

    # ----------------------------
    # Critical validation summary
    # ----------------------------
    print("\nSanity checks:")
    print(f"  - edges_path out-of-bounds vs num_edges: {edge_path_oob} files")
    if edge_path_oob_samples:
        print("    Samples (file, max_edge_id_in_edges_path, num_edges):")
        for fp, mx, E in edge_path_oob_samples:
            print(f"      * {fp} | max_edge_id={mx} | num_edges={E}")

    if PRINT_UNIQUE_SAMPLES and UNIQUE_ENABLED:
        print("\nUnique samples (first ~20 values, per key):")
        for k in keys_sorted:
            s = stats[k]
            if len(s.uniques) > 0:
                sample = sorted(list(s.uniques))[:20]
                print(f"  - {k}: {sample}")


if __name__ == "__main__":
    main()
