#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SolidWorks macro JSON → ``torch_geometric.data.Data``, saved as ``.pt`` (+ optional label JSON).

**Migration parity with the old three-step toolchain**

1. ``BrepMFR/json_to_brepmfr_bin.py`` (JSON → DGL ``.bin`` + labels): face order, edges, flipping,
   scalar ``edge_ang`` wrap, ``spatial_pos`` / ``edge_path``, ``face_pairs`` → A2 tensors.
2. ``BrepMFR/append_angle_7th_channel.py``: wrap edge UV-grid channel 7 (index 6) to
   :math:`[-\\pi, \\pi)` in place on ``edata['x']``.
3. ``convert_dgl_bins_to_pyg`` / ``bin_to_pyg``: ``.bin`` → ``.pt``.

This script merges (1)+(2)+(3)'s numeric result in one PyTorch-only path: every ``.pt`` is written
like **post-``append_angle``** graphs. You do **not** run ``append_angle_7th_channel`` anymore.

**Inference profiles** (``--inference_profile``):

- ``full``: A1 shortest-path tensors, A2 from ``face_pairs``, A3 ``edge_path`` (no stored ``attn_bias``; collator fills zeros).
- ``no_a2``: Same as full but **omits** ``d2_distance`` / ``angle_distance`` (no dense A2 on disk).
  Collator / ``GraphAttnBias`` run in A2-disabled mode (not numerically identical to legacy
  zero-filled A2 because of BatchNorm paths).
- ``lite``: Omits A2, A1, A3, and **does not store** ``spatial_pos``, ``edge_path``, or ``attn_bias``
  (smallest ``.pt`` + fastest JSON ingest). Collator synthesizes ``attn_bias`` for attention.

**Speed:** ``no_a2`` and ``full`` still run **all-pairs shortest paths** (A1); omitting A2 does not
remove that cost. The serial BFS uses NumPy buffers (far faster than per-cell torch writes). Prefer
file-level parallelism via ``scripts/threads/upgrade_lite_pt_to_no_a2.py`` when upgrading an existing
lite corpus. ``--shortest_path_workers N`` (``N>1``) only helps single huge graphs (N≥512) and is
expensive to spawn repeatedly on Windows — keep it at 0 for bulk conversion.

``--skip_a2`` is kept as a **deprecated** alias for ``--inference_profile no_a2``.

Dataset loading (``CADSynth``, ``TransferDataset``): ``torch.load(...)`` unchanged. **No DGL**
required on this ingest path.

**Pairing discipline (targets):** under ``Experiment6/target_dataset/input`` multiple ``json_*\\``
trees can share one filename with different ``face[].label``; use the subtree that matches your
``.pt`` / training labels.

**Roots:** ``Z:\\Experiment6`` / ``Z:\\Experiment6_PyG`` are historical corpora; ``Z:\\Experiment_test``
holds writable copies for optional JSON-vs-``.bin`` parity (parity tooling may still ``import dgl``
to load old ``.bin`` files only).

Example:
  conda activate brep_mfr_pyg
  python scripts/inference/json_to_brepmfr_pyg_optimized.py \\
    --json_dir Z:/Experiment_test/input_json \\
    --abc_json_dir Z:/Experiment_test/abc_jsons \\
    --pt_out_dir Z:/Experiment_test/out_pyg \\
    --label_out_dir Z:/Experiment_test/out_label \\
    --inference_profile no_a2 --profile

**This file** (``*_optimized.py``) adds: optional ``orjson``, profiling, no stored ``attn_bias``,
compact ``spatial_pos`` (``uint8``) / ``edge_path`` (``int16`` when safe), optional NPZ pre-BFS cache
with a **direct NPZ→tensor path** (no reconstructed ``faces``/``edges`` dicts; use ``--use-npz-cache``),
prefix-path BFS bit-identical to the legacy backtracking implementation, and optional benchmarks
(``--bench-npz-cache``, ``--bench-scipy-bfs``). The original ``json_to_brepmfr_pyg.py`` is unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import traceback
from collections import deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch import FloatTensor
from tqdm import tqdm

try:
    import orjson as _orjson  # type: ignore[assignment]
except ImportError:
    _orjson = None

# Try to import compiled BFS kernel (built via scripts/inference/build_bfs.py).
try:
    from _bfs_kernel import all_pairs_bfs as _bfs_cython

    _HAS_CYTHON_BFS = True
except ImportError:
    _bfs_cython = None  # type: ignore[assignment,misc]
    _HAS_CYTHON_BFS = False

InferenceProfile = Literal["full", "no_a2", "lite"]


def _new_pyg_graph() -> Any:
    """Lazily import torch_geometric so ``--selftest-bfs`` can run without PyG installed."""
    from torch_geometric.data import Data

    return Data()


def load_json_fast(path: Path) -> dict:
    if _orjson is not None:
        return _orjson.loads(path.read_bytes())
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_pyg_tensor_sizes(pyg: Any, title: str = "PyG tensor sizes") -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    total_b = 0.0
    for key in sorted(pyg.keys()):
        value = pyg[key]
        if torch.is_tensor(value):
            nbytes = value.numel() * value.element_size()
            total_b += nbytes
            print(
                f"{key:22s} shape={str(tuple(value.shape)):32s} "
                f"dtype={str(value.dtype):14s} size_mb={nbytes / (1024 ** 2):.4f}"
            )
    print("-" * 80)
    print(f"Total tensor memory: {total_b / (1024 ** 2):.4f} MB")
    print("=" * 80)


def _reshape_face_uv(flat_uv: list, U: int = 5, V: int = 5, C: int = 7) -> np.ndarray:
    """Reshapes flat face list to (U, V, 7) grid: [x, y, z, nx, ny, nz, mask]."""
    arr = np.asarray(flat_uv, dtype=np.float32)
    return arr.reshape(U, V, C)


def _reshape_edge_pt(flat_pt: list, U: int = 5, C: int = 7) -> np.ndarray:
    """Reshapes flat edge list to (U, 7) grid: [x, y, z, tx, ty, tz, angle]."""
    arr = np.asarray(flat_pt, dtype=np.float32)
    return arr.reshape(U, C)


def _wrap_edge_uv_angle_ch7(edge_x: np.ndarray) -> None:
    """
    In-place wrap of channel index 6 (7th UV sample feature) on every directed edge arc.

    Equivalent to ``BrepMFR/append_angle_7th_channel.wrap_to_pi_tensor`` on ``edata['x'][:, :, 6]``.
    Applied unconditionally so JSON→``.pt`` matches the legacy workflow **after** that script.
    """
    pi = np.float32(math.pi)
    two_pi = np.float32(2.0 * math.pi)
    edge_x[:, :, 6] = (edge_x[:, :, 6] + pi) % two_pi - pi


def _directed_adj_with_edge_ids(src_nodes, dst_nodes, num_nodes: int) -> list[list[tuple[int, int]]]:
    """Adjacency with parallel edge arc ids (same order as ``final_src`` / ``final_dst``)."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for ei, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
        adj[u].append((v, ei))
    return adj


def _adj_to_csr(src_nodes, dst_nodes, num_nodes: int):
    """Directed arc lists → CSR ``(offsets, targets, edge_ids)`` for the Cython BFS kernel."""
    src = np.asarray(src_nodes, dtype=np.int32)
    dst = np.asarray(dst_nodes, dtype=np.int32)
    nnz = src.size
    if nnz == 0:
        offsets = np.zeros(num_nodes + 1, dtype=np.int32)
        empty = np.empty(0, dtype=np.int32)
        return offsets, empty, empty
    # Stable sort by source preserves the same neighbour iteration order as
    # _directed_adj_with_edge_ids, so BFS breaks ties identically.
    order = np.argsort(src, kind="stable").astype(np.int32)
    tgt = dst[order]
    counts = np.bincount(src, minlength=num_nodes)
    offsets = np.zeros(num_nodes + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    return offsets, tgt, order  # order[j] = original arc index = edge ID


def _shortest_paths_from_adj_serial_legacy(adj: list[list[tuple[int, int]]], num_nodes: int, max_dist: int):
    """
    Legacy A1 + A3: BFS then per-target backtrack (reference for bit-identical checks).
    """
    spatial_pos = torch.full((num_nodes, num_nodes), fill_value=10**9, dtype=torch.int32)
    edges_path = torch.full((num_nodes, num_nodes, max_dist), fill_value=-1, dtype=torch.int32)

    for s in range(num_nodes):
        spatial_pos[s, s] = 0
        dist = [-1] * num_nodes
        prev_node = [-1] * num_nodes
        prev_edge = [-1] * num_nodes
        q = deque([s])
        dist[s] = 0

        while q:
            u = q.popleft()
            for v, ei in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev_node[v] = u
                    prev_edge[v] = ei
                    q.append(v)

        for t in range(num_nodes):
            if dist[t] == -1:
                continue
            spatial_pos[s, t] = dist[t]
            if t == s:
                continue

            path_edges = []
            cur = t
            while cur != s and cur != -1:
                path_edges.append(prev_edge[cur])
                cur = prev_node[cur]
            path_edges.reverse()

            for k in range(min(len(path_edges), max_dist)):
                edges_path[s, t, k] = int(path_edges[k])

    return spatial_pos, edges_path


def _shortest_paths_from_adj_serial(adj: list[list[tuple[int, int]]], num_nodes: int, max_dist: int):
    """
    A1 + edge_path via parent-backtrack BFS (Graphormer-style).

    Instead of copying a ``max_dist``-element numpy row on every BFS edge
    traversal (the old prefix approach), this stores only ``parent_node`` /
    ``parent_edge`` during BFS (2 int writes per edge) and backtracks after
    BFS completes.  Same result, far fewer memory copies on large graphs.
    """
    spatial_pos = np.full((num_nodes, num_nodes), 10**9, dtype=np.int32)
    edges_path = np.full((num_nodes, num_nodes, max_dist), -1, dtype=np.int32)

    # Reusable buffers across all sources (no per-source allocation).
    parent_node = [0] * num_nodes
    parent_edge = [0] * num_nodes
    path_buf = [0] * num_nodes  # max path length = num_nodes - 1

    for s in range(num_nodes):
        dist = [-1] * num_nodes
        dist[s] = 0
        spatial_pos[s, s] = 0
        q = deque([s])

        # BFS: store parent pointers only (no row copies).
        while q:
            u = q.popleft()
            du = dist[u]
            for v, ei in adj[u]:
                if dist[v] != -1:
                    continue
                dist[v] = du + 1
                parent_node[v] = u
                parent_edge[v] = ei
                q.append(v)

        # Backtrack from each reachable target to reconstruct edge paths.
        for t in range(num_nodes):
            d = dist[t]
            if d <= 0:
                continue
            spatial_pos[s, t] = d

            # Walk t → s, collecting edges in reverse.
            cur = t
            plen = 0
            while cur != s:
                path_buf[plen] = parent_edge[cur]
                cur = parent_node[cur]
                plen += 1

            # Copy first min(plen, max_dist) edges, reversed, into result.
            k = plen if plen < max_dist else max_dist
            for j in range(k):
                edges_path[s, t, j] = path_buf[plen - 1 - j]

    return torch.from_numpy(spatial_pos), torch.from_numpy(edges_path)


# ProcessPool workers (spawn on Windows): adj + max_dist set once per pool.
_SP_ADJ_WORKER: list[list[tuple[int, int]]] | None = None
_SP_MAX_DIST_WORKER: int = 16


def _shortest_paths_worker_init(adj: list[list[tuple[int, int]]], max_dist: int) -> None:
    global _SP_ADJ_WORKER, _SP_MAX_DIST_WORKER
    _SP_ADJ_WORKER = adj
    _SP_MAX_DIST_WORKER = int(max_dist)


def _shortest_paths_worker_one_row_prefix(s: int) -> tuple[int, np.ndarray, np.ndarray]:
    """One source row with prefix-path BFS (numpy for ProcessPool IPC)."""
    adj = _SP_ADJ_WORKER
    max_dist = _SP_MAX_DIST_WORKER
    assert adj is not None
    num_nodes = len(adj)

    dist_row = np.full(num_nodes, 10**9, dtype=np.int32)
    path_block = np.full((num_nodes, max_dist), -1, dtype=np.int32)
    dist_row[s] = 0

    dist = [-1] * num_nodes
    path_pref = np.full((num_nodes, max_dist), -1, dtype=np.int32)
    q = deque([s])
    dist[s] = 0

    while q:
        u = q.popleft()
        du = dist[u]
        for v, ei in adj[u]:
            if dist[v] != -1:
                continue
            dist[v] = du + 1
            path_pref[v] = path_pref[u].copy()
            if du < max_dist:
                path_pref[v, du] = ei
            q.append(v)

    for t in range(num_nodes):
        if dist[t] == -1:
            continue
        dist_row[t] = dist[t]
        if t == s:
            continue
        path_block[t, :] = path_pref[t, :]

    return s, dist_row, path_block


def _shortest_paths_from_adj_parallel(
    adj: list[list[tuple[int, int]]],
    num_nodes: int,
    max_dist: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_workers = min(int(num_workers), num_nodes, os.cpu_count() or 1)
    spatial_pos = torch.full((num_nodes, num_nodes), fill_value=10**9, dtype=torch.int32)
    edges_path = torch.full((num_nodes, num_nodes, max_dist), fill_value=-1, dtype=torch.int32)
    chunksize = max(1, num_nodes // (n_workers * 8))

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_shortest_paths_worker_init,
        initargs=(adj, max_dist),
    ) as pool:
        for s, dist_row, path_block in pool.map(
            _shortest_paths_worker_one_row_prefix,
            range(num_nodes),
            chunksize=chunksize,
        ):
            spatial_pos[s] = torch.from_numpy(dist_row)
            edges_path[s] = torch.from_numpy(path_block)

    return spatial_pos, edges_path


def _compute_shortest_paths_edge_indices(
    src_nodes,
    dst_nodes,
    num_nodes: int,
    max_dist: int = 16,
    num_workers: int = 0,
    legacy_bfs: bool = False,
):
    """
    Computes A1 (Shortest path distance) and A3 (Chain of edge indices).
    Matches Graphormer-style encoding where edges_path stores the sequence of edge IDs.

    Prefer serial NumPy BFS for normal graphs. Per-source ``ProcessPool`` only helps on very
    large single graphs (N>=512) and is costly to spawn repeatedly on Windows — for corpus
    conversion, use file-level workers in ``upgrade_lite_pt_to_no_a2.py`` instead.
    """
    if legacy_bfs:
        adj = _directed_adj_with_edge_ids(src_nodes, dst_nodes, num_nodes)
        return _shortest_paths_from_adj_serial_legacy(adj, num_nodes, max_dist)
    # Compiled Cython kernel (fastest): CSR adjacency → C-level BFS + backtrack.
    if _HAS_CYTHON_BFS and num_workers <= 1:
        offsets, tgt, eid = _adj_to_csr(src_nodes, dst_nodes, num_nodes)
        sp, ep = _bfs_cython(offsets, tgt, eid, num_nodes, max_dist)
        return torch.from_numpy(sp), torch.from_numpy(ep)
    # Python fallback with parent-backtrack BFS.
    adj = _directed_adj_with_edge_ids(src_nodes, dst_nodes, num_nodes)
    use_parallel = (
        num_workers > 1
        and num_nodes >= 512
        and (os.cpu_count() or 1) >= 2
    )
    if use_parallel:
        return _shortest_paths_from_adj_parallel(adj, num_nodes, max_dist, num_workers)
    return _shortest_paths_from_adj_serial(adj, num_nodes, max_dist)


def _shortest_hops_scipy_csr(
    final_src: list[int] | np.ndarray,
    final_dst: list[int] | np.ndarray,
    num_nodes: int,
) -> torch.Tensor:
    """
    All-pairs **hop counts** on the **directed** graph where parallel arcs collapse to one unit edge.
    Matches unweighted BFS distances on the same multigraph for hop length (not edge ids).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    src = np.asarray(final_src, dtype=np.int64).ravel()
    dst = np.asarray(final_dst, dtype=np.int64).ravel()
    if src.size != dst.size:
        raise ValueError("final_src / final_dst length mismatch")
    packed = src * np.int64(num_nodes) + dst
    uniq = np.unique(packed)
    ui = uniq // np.int64(num_nodes)
    uj = uniq % np.int64(num_nodes)
    data = np.ones(uniq.size, dtype=np.float64)
    mat = csr_matrix((data, (ui.astype(np.int64), uj.astype(np.int64))), shape=(num_nodes, num_nodes))
    dist = shortest_path(mat, directed=True, unweighted=True)
    t = torch.as_tensor(dist, dtype=torch.float64)
    t = torch.where(torch.isfinite(t), t, torch.full_like(t, 1.0e9))
    return t.to(torch.int32)


def benchmark_scipy_vs_prefix_bfs_hops(
    final_src: list[int],
    final_dst: list[int],
    num_nodes: int,
    max_dist: int = 16,
) -> dict[str, float | bool]:
    """Time SciPy hop matrix vs prefix serial BFS; check distance matrix equality."""
    adj = _directed_adj_with_edge_ids(final_src, final_dst, num_nodes)
    t0 = time.perf_counter()
    sp_prefix, _ep = _shortest_paths_from_adj_serial(adj, num_nodes, max_dist)
    t_prefix = time.perf_counter() - t0
    t0 = time.perf_counter()
    sp_sci = _shortest_hops_scipy_csr(final_src, final_dst, num_nodes)
    t_scipy = time.perf_counter() - t0
    ok = torch.equal(sp_prefix, sp_sci)
    return {
        "prefix_serial_s": t_prefix,
        "scipy_csr_s": t_scipy,
        "distances_match": bool(ok),
        "speedup_scipy_vs_prefix": (t_prefix / t_scipy) if t_scipy > 0 else float("inf"),
    }


def _tensor_prep_keys_json() -> tuple[str, ...]:
    return (
        "json_faces_edges_access",
        "adjacency_directed_edges",
        "node_tensor_fill",
        "edge_tensor_fill",
        "wrap_edge_angle_ch7",
    )


def _collapse_tensor_prep_json_timing(timing: dict[str, float]) -> None:
    """Fold fine-grained JSON ingest timers into ``tensor_prep`` for reporting."""
    keys = _tensor_prep_keys_json()
    s = sum(float(timing.pop(k, 0.0)) for k in keys)
    timing["tensor_prep"] = timing.get("tensor_prep", 0.0) + s


def _build_a2_tensors(data: dict, face_id_to_node: dict, N: int) -> tuple:
    """
    Builds d2_distance and angle_distance tensors from face_pairs in JSON.

    Design A:
    - Exactly one JSON entry per unordered face pair
    - entry["a3"]   = histogram for face_pair[0] -> face_pair[1]
    - entry["a3_1"] = histogram for face_pair[1] -> face_pair[0]

    D2 is symmetric.
    A3 is asymmetric.
    """
    d2_distance = torch.zeros((N, N, 64), dtype=torch.float32)
    angle_distance = torch.zeros((N, N, 64), dtype=torch.float32)

    face_pairs = data.get("face_pairs", [])
    if not face_pairs:
        return d2_distance, angle_distance

    pair_lut = {}
    for entry in face_pairs:
        fi = int(entry["face_pair"][0])
        fj = int(entry["face_pair"][1])

        if fi == fj:
            continue

        key = (min(fi, fj), max(fi, fj))

        if key in pair_lut:
            raise ValueError(
                f"Duplicate unordered face pair found in JSON for faces {key}. "
                f"Design A requires exactly one entry per unordered pair."
            )

        if len(entry["d2"]) != 64:
            raise ValueError(f"d2 histogram for pair {fi, fj} does not have length 64")
        if len(entry["a3"]) != 64:
            raise ValueError(f"a3 histogram for pair {fi, fj} does not have length 64")
        if len(entry["a3_1"]) != 64:
            raise ValueError(f"a3_1 histogram for pair {fi, fj} does not have length 64")

        pair_lut[key] = entry

    node_to_face_id = {v: k for k, v in face_id_to_node.items()}

    for ni in range(N):
        for nj in range(N):
            if ni == nj:
                continue

            fi = node_to_face_id[ni]
            fj = node_to_face_id[nj]

            key = (min(fi, fj), max(fi, fj))
            entry = pair_lut.get(key)
            if entry is None:
                continue

            stored_f0 = int(entry["face_pair"][0])
            stored_f1 = int(entry["face_pair"][1])

            d2_distance[ni, nj] = torch.tensor(entry["d2"], dtype=torch.float32)

            if fi == stored_f0 and fj == stored_f1:
                angle_distance[ni, nj] = torch.tensor(entry["a3"], dtype=torch.float32)
            elif fi == stored_f1 and fj == stored_f0:
                angle_distance[ni, nj] = torch.tensor(entry["a3_1"], dtype=torch.float32)
            else:
                raise RuntimeError(
                    f"Inconsistent face pair mapping for query ({fi}, {fj}) "
                    f"against stored pair ({stored_f0}, {stored_f1})"
                )

    return d2_distance, angle_distance


def _write_label_json(label_out_dir: Path, file_stem: str, labels_list: list):
    label_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_out_dir / f"{file_stem}.json"
    payload = {"file_name": file_stem, "labels": labels_list}
    out_path.write_text(json.dumps(payload, indent=3), encoding="utf-8")


PYG_ATTRS_ORDER = (
    "node_data",
    "edge_data",
    "face_type",
    "face_area",
    "face_loop",
    "face_adj",
    "label_feature",
    "edge_type",
    "edge_len",
    "edge_ang",
    "edge_conv",
    "node_degree",
    "edge_path",
    "spatial_pos",
    "d2_distance",
    "angle_distance",
    "edge_index",
    "data_id",
)

NPZ_CACHE_VERSION = 1


def _directed_arcs_from_macro_dict(data: dict) -> tuple[list[int], list[int], int]:
    """Directed adjacency arc lists (same convention as ``tensors_from_brep_json_dict``)."""
    faces, edges = data["faces"], data["edges"]
    sorted_faces = sorted(faces, key=lambda x: int(x["id"]))
    face_id_to_node = {int(f["id"]): i for i, f in enumerate(sorted_faces)}
    N = len(sorted_faces)
    adj = defaultdict(list)
    for e in edges:
        f1, f2 = int(e["nf"][0]), int(e["nf"][1])
        if f1 in face_id_to_node and f2 in face_id_to_node:
            u, v = face_id_to_node[f1], face_id_to_node[f2]
            adj[u].append(v)
            adj[v].append(u)
    final_src: list[int] = []
    final_dst: list[int] = []
    for i in range(N):
        for neighbor in sorted(adj[i]):
            final_src.append(i)
            final_dst.append(neighbor)
    return final_src, final_dst, N


def tensors_from_brep_json_dict(
    data: dict,
    spatial_pos_max: int = 32,
    inference_profile: InferenceProfile = "full",
    max_edge_path_len: int = 16,
    float16_storage: bool = False,
    shortest_path_workers: int = 0,
    timing: Optional[dict[str, float]] = None,
    legacy_bfs: bool = False,
    npz_cache_path: Optional[Path] = None,
) -> tuple[Any, list[int]]:
    """
    Build PyG ``Data`` from decoded macro JSON (**same tensors as** ``json_to_brepmfr_bin`` **then**
    ``append_angle_7th_channel`` on ``edge_data[..., 6]``).

    ``inference_profile`` controls which dense pairwise tensors are stored (see module docstring).
    """
    def _mark(name: str, t0: float) -> float:
        if timing is not None:
            timing[name] = time.perf_counter() - t0
        return time.perf_counter()

    t0 = time.perf_counter()
    faces, edges = data["faces"], data["edges"]
    t0 = _mark("json_faces_edges_access", t0)

    sorted_faces = sorted(faces, key=lambda x: int(x["id"]))
    face_id_to_node = {int(f["id"]): i for i, f in enumerate(sorted_faces)}
    node_to_face_id = {v: k for k, v in face_id_to_node.items()}
    N = len(sorted_faces)

    adj = defaultdict(list)
    edge_lut = {}
    for e in edges:
        f1, f2 = int(e["nf"][0]), int(e["nf"][1])
        if f1 in face_id_to_node and f2 in face_id_to_node:
            u, v = face_id_to_node[f1], face_id_to_node[f2]
            adj[u].append(v)
            adj[v].append(u)
            edge_lut[frozenset([f1, f2])] = e

    final_src, final_dst = [], []
    for i in range(N):
        for neighbor in sorted(adj[i]):
            final_src.append(i)
            final_dst.append(neighbor)
    E = len(final_src)
    t0 = _mark("adjacency_directed_edges", t0)

    node_x = np.zeros((N, 5, 5, 7), dtype=np.float32)
    node_z = np.zeros(N, dtype=np.int32)
    node_l = np.zeros(N, dtype=np.int32)
    node_a = np.zeros(N, dtype=np.int32)
    node_f = np.zeros(N, dtype=np.int32)
    node_y = np.zeros(N, dtype=np.float32)

    labels_list = [0] * N

    for f in sorted_faces:
        ni = face_id_to_node[int(f["id"])]
        node_x[ni] = _reshape_face_uv(f["uv"])
        node_z[ni] = int(f["z"])
        node_y[ni] = float(f["y"])
        node_l[ni] = int(f["l"])
        node_a[ni] = int(f["a"])
        lbl = int(f.get("label", 0))
        node_f[ni] = lbl
        labels_list[ni] = lbl

    t0 = _mark("node_tensor_fill", t0)

    edge_x = np.zeros((E, 5, 7), dtype=np.float32)
    edge_t = np.zeros(E, dtype=np.int32)
    edge_c = np.zeros(E, dtype=np.int32)
    edge_l = np.zeros(E, dtype=np.float32)
    edge_a = np.zeros(E, dtype=np.float32)

    for i, (u_idx, v_idx) in enumerate(zip(final_src, final_dst)):
        u_fid, v_fid = node_to_face_id[u_idx], node_to_face_id[v_idx]
        eobj = edge_lut[frozenset([u_fid, v_fid])]
        raw_pts = _reshape_edge_pt(eobj["pt"])
        if u_fid == int(eobj["nf"][0]):
            edge_x[i] = raw_pts
        else:
            flipped = np.flip(raw_pts, axis=0).copy()
            flipped[:, 3:6] *= -1.0
            edge_x[i] = flipped
        edge_t[i] = int(eobj.get("t", 0))
        edge_l[i] = float(eobj.get("l", 0.0))
        edge_c[i] = int(eobj.get("c", 0))
        edge_a[i] = (float(eobj.get("a", 0.0)) + np.pi) % (2 * np.pi) - np.pi

    edge_nf0_np = np.array([node_to_face_id[u] for u in final_src], dtype=np.int32)
    edge_nf1_np = np.array([node_to_face_id[v] for v in final_dst], dtype=np.int32)
    face_ids_np = np.array([node_to_face_id[i] for i in range(N)], dtype=np.int32)

    t0 = _mark("edge_tensor_fill", t0)

    _wrap_edge_uv_angle_ch7(edge_x)
    t0 = _mark("wrap_edge_angle_ch7", t0)

    if inference_profile == "lite":
        spatial_pos_i = None
        edges_path_i = None
        d2_distance = None
        angle_distance = None
    else:
        spatial_pos, edges_path = _compute_shortest_paths_edge_indices(
            final_src,
            final_dst,
            N,
            max_dist=max_edge_path_len,
            num_workers=shortest_path_workers,
            legacy_bfs=legacy_bfs,
        )
        t0 = _mark("shortest_path_a1_a3", t0)
        if inference_profile == "full":
            d2_distance, angle_distance = _build_a2_tensors(data, face_id_to_node, N)
        else:
            d2_distance, angle_distance = None, None
        t0 = _mark("a2_histograms_full_only", t0)

        max_p = int(spatial_pos[spatial_pos < 10**8].max().item()) if N > 1 else 0
        edges_path = edges_path[:, :, :max_p]
        spatial_pos = spatial_pos.clamp(max=spatial_pos_max)

        edges_path_int = edges_path.to(torch.int32)
        if E <= 32767:
            edges_path_i = edges_path_int.to(torch.int16)
            edge_path_storage = "int16"
        else:
            edges_path_i = edges_path_int
            edge_path_storage = "int32"

        spatial_pos_i = spatial_pos.to(torch.uint8)

    # Match bin_to_pyg: out-degree equals row-sum of directed adjacency (one row per arc u->v)
    src_counts = torch.bincount(
        torch.tensor(final_src, dtype=torch.long),
        minlength=N,
    )
    node_degree = src_counts.view(-1)

    t_py = time.perf_counter()
    pyg = _new_pyg_graph()

    xt = torch.from_numpy(node_x)
    pyg.node_data = xt.type(FloatTensor)

    pyg.edge_data = torch.from_numpy(edge_x).type(FloatTensor)

    pyg.face_type = torch.from_numpy(node_z).type(torch.int)
    pyg.face_area = torch.from_numpy(node_y).type(torch.float)
    pyg.face_loop = torch.from_numpy(node_l).type(torch.int)
    pyg.face_adj = torch.from_numpy(node_a).type(torch.int)
    pyg.label_feature = torch.from_numpy(node_f).type(torch.int)

    pyg.edge_type = torch.from_numpy(edge_t).type(torch.int)
    pyg.edge_len = torch.from_numpy(edge_l).type(torch.float)
    pyg.edge_ang = torch.from_numpy(edge_a).type(torch.float)
    pyg.edge_conv = torch.from_numpy(edge_c).type(torch.int)

    pyg.node_degree = node_degree

    if inference_profile == "lite":
        pyg.has_a1 = False
        pyg.has_a2 = False
        pyg.has_a3 = False
        pyg.inference_profile = "lite"
    elif inference_profile == "no_a2":
        pyg.has_a1 = True
        pyg.has_a2 = False
        pyg.has_a3 = True
        pyg.inference_profile = "no_a2"
        pyg.has_stored_attn_bias = False
        pyg.edge_path = edges_path_i
        pyg.spatial_pos = spatial_pos_i
        pyg.edge_path_storage_dtype = edge_path_storage
        pyg.spatial_pos_storage_dtype = "uint8"
    else:
        pyg.has_a1 = True
        pyg.has_a2 = True
        pyg.has_a3 = True
        pyg.inference_profile = "full"
        pyg.has_stored_attn_bias = False
        pyg.edge_path = edges_path_i
        pyg.spatial_pos = spatial_pos_i
        pyg.edge_path_storage_dtype = edge_path_storage
        pyg.spatial_pos_storage_dtype = "uint8"
        pyg.d2_distance = d2_distance
        pyg.angle_distance = angle_distance

    if float16_storage:
        pyg.node_data = pyg.node_data.half()
        pyg.edge_data = pyg.edge_data.half()
        pyg.face_area = pyg.face_area.half()
        pyg.store_float16 = True
    else:
        pyg.store_float16 = False

    ei = torch.stack(
        [
            torch.tensor(final_src, dtype=torch.long),
            torch.tensor(final_dst, dtype=torch.long),
        ],
        dim=0,
    )
    pyg.edge_index = ei

    if timing is not None:
        timing["pyg_assign_edge_index"] = time.perf_counter() - t_py

    if npz_cache_path is not None:
        t_npz = time.perf_counter()
        pairs_bytes = json.dumps(data.get("face_pairs", [])).encode("utf-8")
        _save_npz_pre_bfs_cache(
            npz_cache_path,
            node_x=node_x,
            node_z=node_z,
            node_y=node_y,
            node_l=node_l,
            node_a=node_a,
            node_f=node_f,
            edge_x=edge_x,
            edge_t=edge_t,
            edge_c=edge_c,
            edge_l=edge_l,
            edge_a=edge_a,
            final_src=final_src,
            final_dst=final_dst,
            face_ids=face_ids_np,
            edge_nf0=edge_nf0_np,
            edge_nf1=edge_nf1_np,
            face_pairs_json_bytes=pairs_bytes,
        )
        if timing is not None:
            timing["npz_cache_write"] = time.perf_counter() - t_npz

    if timing is not None:
        _collapse_tensor_prep_json_timing(timing)
        timing["tensor_prep"] = float(timing.get("tensor_prep", 0.0)) + float(
            timing.pop("a2_histograms_full_only", 0.0)
        )
        timing["pyg_pack"] = float(timing.pop("pyg_assign_edge_index", 0.0))

    return pyg, labels_list


def tensors_from_npz_arrays(
    z: Any,
    *,
    spatial_pos_max: int = 32,
    inference_profile: InferenceProfile = "full",
    max_edge_path_len: int = 16,
    float16_storage: bool = False,
    shortest_path_workers: int = 0,
    timing: Optional[dict[str, float]] = None,
    legacy_bfs: bool = False,
) -> tuple[Any, list[int]]:
    """
    Build PyG tensors from an opened NPZ cache (**no** JSON-style ``faces`` / ``edges`` dicts).

    Arrays must match ``_save_npz_pre_bfs_cache`` (post ``_wrap_edge_uv_angle_ch7`` on ``edge_x``).
    """
    def _mark(name: str, t0: float) -> float:
        if timing is not None:
            timing[name] = time.perf_counter() - t0
        return time.perf_counter()

    t0 = time.perf_counter()
    ver = int(np.asarray(z["version"])[0])
    if ver != NPZ_CACHE_VERSION:
        raise ValueError(f"NPZ cache version mismatch: got {ver}, expected {NPZ_CACHE_VERSION}")

    node_x = np.ascontiguousarray(np.asarray(z["node_x"], dtype=np.float32))
    node_z = np.ascontiguousarray(np.asarray(z["node_z"], dtype=np.int32))
    node_y = np.ascontiguousarray(np.asarray(z["node_y"], dtype=np.float32))
    node_l = np.ascontiguousarray(np.asarray(z["node_l"], dtype=np.int32))
    node_a = np.ascontiguousarray(np.asarray(z["node_a"], dtype=np.int32))
    node_f = np.ascontiguousarray(np.asarray(z["node_f"], dtype=np.int32))
    edge_x = np.ascontiguousarray(np.asarray(z["edge_x"], dtype=np.float32))
    edge_t = np.ascontiguousarray(np.asarray(z["edge_t"], dtype=np.int32))
    edge_c = np.ascontiguousarray(np.asarray(z["edge_c"], dtype=np.int32))
    edge_l = np.ascontiguousarray(np.asarray(z["edge_l"], dtype=np.float32))
    edge_a = np.ascontiguousarray(np.asarray(z["edge_a"], dtype=np.float32))
    final_src = np.asarray(z["final_src"], dtype=np.int32).tolist()
    final_dst = np.asarray(z["final_dst"], dtype=np.int32).tolist()
    face_ids_np = np.ascontiguousarray(np.asarray(z["face_ids"], dtype=np.int32))
    fp = z["face_pairs_json"]
    pairs_bytes = fp.tobytes() if fp.size else b"[]"
    face_pairs = json.loads(pairs_bytes.decode("utf-8"))
    data_stub: dict = {"face_pairs": face_pairs}

    N = int(node_x.shape[0])
    E = len(final_src)
    labels_list = [int(node_f[i]) for i in range(N)]
    face_id_to_node = {int(face_ids_np[i]): i for i in range(N)}
    t0 = _mark("tensor_prep", t0)

    if inference_profile == "lite":
        spatial_pos_i = None
        edges_path_i = None
        d2_distance = None
        angle_distance = None
        edge_path_storage = "none"
    else:
        spatial_pos, edges_path = _compute_shortest_paths_edge_indices(
            final_src,
            final_dst,
            N,
            max_dist=max_edge_path_len,
            num_workers=shortest_path_workers,
            legacy_bfs=legacy_bfs,
        )
        t0 = _mark("shortest_path_a1_a3", t0)
        if inference_profile == "full":
            d2_distance, angle_distance = _build_a2_tensors(data_stub, face_id_to_node, N)
        else:
            d2_distance, angle_distance = None, None
        t0 = _mark("a2_histograms_full_only", t0)

        max_p = int(spatial_pos[spatial_pos < 10**8].max().item()) if N > 1 else 0
        edges_path = edges_path[:, :, :max_p]
        spatial_pos = spatial_pos.clamp(max=spatial_pos_max)

        edges_path_int = edges_path.to(torch.int32)
        if E <= 32767:
            edges_path_i = edges_path_int.to(torch.int16)
            edge_path_storage = "int16"
        else:
            edges_path_i = edges_path_int
            edge_path_storage = "int32"

        spatial_pos_i = spatial_pos.to(torch.uint8)

    src_counts = torch.bincount(
        torch.tensor(final_src, dtype=torch.long),
        minlength=N,
    )
    node_degree = src_counts.view(-1)

    t_py = time.perf_counter()
    pyg = _new_pyg_graph()

    xt = torch.from_numpy(node_x)
    pyg.node_data = xt.type(FloatTensor)

    pyg.edge_data = torch.from_numpy(edge_x).type(FloatTensor)

    pyg.face_type = torch.from_numpy(node_z).type(torch.int)
    pyg.face_area = torch.from_numpy(node_y).type(torch.float)
    pyg.face_loop = torch.from_numpy(node_l).type(torch.int)
    pyg.face_adj = torch.from_numpy(node_a).type(torch.int)
    pyg.label_feature = torch.from_numpy(node_f).type(torch.int)

    pyg.edge_type = torch.from_numpy(edge_t).type(torch.int)
    pyg.edge_len = torch.from_numpy(edge_l).type(torch.float)
    pyg.edge_ang = torch.from_numpy(edge_a).type(torch.float)
    pyg.edge_conv = torch.from_numpy(edge_c).type(torch.int)

    pyg.node_degree = node_degree

    if inference_profile == "lite":
        pyg.has_a1 = False
        pyg.has_a2 = False
        pyg.has_a3 = False
        pyg.inference_profile = "lite"
    elif inference_profile == "no_a2":
        pyg.has_a1 = True
        pyg.has_a2 = False
        pyg.has_a3 = True
        pyg.inference_profile = "no_a2"
        pyg.has_stored_attn_bias = False
        pyg.edge_path = edges_path_i
        pyg.spatial_pos = spatial_pos_i
        pyg.edge_path_storage_dtype = edge_path_storage
        pyg.spatial_pos_storage_dtype = "uint8"
    else:
        pyg.has_a1 = True
        pyg.has_a2 = True
        pyg.has_a3 = True
        pyg.inference_profile = "full"
        pyg.has_stored_attn_bias = False
        pyg.edge_path = edges_path_i
        pyg.spatial_pos = spatial_pos_i
        pyg.edge_path_storage_dtype = edge_path_storage
        pyg.spatial_pos_storage_dtype = "uint8"
        pyg.d2_distance = d2_distance
        pyg.angle_distance = angle_distance

    if float16_storage:
        pyg.node_data = pyg.node_data.half()
        pyg.edge_data = pyg.edge_data.half()
        pyg.face_area = pyg.face_area.half()
        pyg.store_float16 = True
    else:
        pyg.store_float16 = False

    ei = torch.stack(
        [
            torch.tensor(final_src, dtype=torch.long),
            torch.tensor(final_dst, dtype=torch.long),
        ],
        dim=0,
    )
    pyg.edge_index = ei

    if timing is not None:
        timing["pyg_assign_edge_index"] = time.perf_counter() - t_py
        timing["tensor_prep"] = float(timing.get("tensor_prep", 0.0)) + float(
            timing.pop("a2_histograms_full_only", 0.0)
        )
        timing["pyg_pack"] = float(timing.pop("pyg_assign_edge_index", 0.0))

    return pyg, labels_list


def _save_npz_pre_bfs_cache(
    path: Path,
    *,
    node_x: np.ndarray,
    node_z: np.ndarray,
    node_y: np.ndarray,
    node_l: np.ndarray,
    node_a: np.ndarray,
    node_f: np.ndarray,
    edge_x: np.ndarray,
    edge_t: np.ndarray,
    edge_c: np.ndarray,
    edge_l: np.ndarray,
    edge_a: np.ndarray,
    final_src: list[int],
    final_dst: list[int],
    face_ids: np.ndarray,
    edge_nf0: np.ndarray,
    edge_nf1: np.ndarray,
    face_pairs_json_bytes: bytes,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if face_pairs_json_bytes:
        fp = np.frombuffer(face_pairs_json_bytes, dtype=np.uint8)
    else:
        fp = np.zeros(0, dtype=np.uint8)
    np.savez(
        path,
        version=np.array([NPZ_CACHE_VERSION], dtype=np.int32),
        node_x=node_x,
        node_z=node_z,
        node_y=node_y,
        node_l=node_l,
        node_a=node_a,
        node_f=node_f,
        edge_x=edge_x,
        edge_t=edge_t,
        edge_c=edge_c,
        edge_l=edge_l,
        edge_a=edge_a,
        final_src=np.asarray(final_src, dtype=np.int32),
        final_dst=np.asarray(final_dst, dtype=np.int32),
        face_ids=np.asarray(face_ids, dtype=np.int32),
        edge_nf0=np.asarray(edge_nf0, dtype=np.int32),
        edge_nf1=np.asarray(edge_nf1, dtype=np.int32),
        face_pairs_json=fp,
    )


def _data_dict_from_npz_opened(z: Any) -> dict:
    ver = int(np.asarray(z["version"])[0])
    if ver != NPZ_CACHE_VERSION:
        raise ValueError(f"NPZ cache version mismatch: got {ver}, expected {NPZ_CACHE_VERSION}")
    node_x = np.asarray(z["node_x"], dtype=np.float32)
    N = node_x.shape[0]
    face_ids = np.asarray(z["face_ids"], dtype=np.int32)
    faces = []
    for i in range(N):
        faces.append(
            {
                "id": int(face_ids[i]),
                "uv": node_x[i].reshape(-1).tolist(),
                "z": int(np.asarray(z["node_z"])[i]),
                "y": float(np.asarray(z["node_y"])[i]),
                "l": int(np.asarray(z["node_l"])[i]),
                "a": int(np.asarray(z["node_a"])[i]),
                "label": int(np.asarray(z["node_f"])[i]),
            }
        )
    edge_x = np.asarray(z["edge_x"], dtype=np.float32)
    edge_t = np.asarray(z["edge_t"], dtype=np.int32)
    edge_c = np.asarray(z["edge_c"], dtype=np.int32)
    edge_l = np.asarray(z["edge_l"], dtype=np.float32)
    edge_a = np.asarray(z["edge_a"], dtype=np.float32)
    E = edge_x.shape[0]
    e_nf0 = np.asarray(z["edge_nf0"], dtype=np.int32)
    e_nf1 = np.asarray(z["edge_nf1"], dtype=np.int32)
    edges = []
    for i in range(E):
        edges.append(
            {
                "nf": [int(e_nf0[i]), int(e_nf1[i])],
                "pt": edge_x[i].reshape(-1).tolist(),
                "t": int(edge_t[i]),
                "c": int(edge_c[i]),
                "l": float(edge_l[i]),
                "a": float(edge_a[i]),
            }
        )
    fp = z["face_pairs_json"]
    pairs_bytes = fp.tobytes() if fp.size else b"[]"
    face_pairs = json.loads(pairs_bytes.decode("utf-8"))
    return {"faces": faces, "edges": edges, "face_pairs": face_pairs}


def _data_dict_from_npz_cache(path: Path) -> dict:
    z = np.load(path, mmap_mode="c", allow_pickle=False)
    try:
        return _data_dict_from_npz_opened(z)
    finally:
        z.close()


def run_bfs_selftest(max_n: int = 14, trials: int = 30) -> None:
    """Assert prefix BFS matches legacy backtracking on random undirected graphs."""
    import random

    rng = random.Random(0)
    for n in range(2, max_n + 1):
        for _ in range(trials):
            src, dst = [], []
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.35:
                        ei = len(src)
                        src.extend([u, v])
                        dst.extend([v, u])
            if not src:
                continue
            adj = _directed_adj_with_edge_ids(src, dst, n)
            K = 16
            a1, b1 = _shortest_paths_from_adj_serial_legacy(adj, n, K)
            a2, b2 = _shortest_paths_from_adj_serial(adj, n, K)
            assert torch.equal(a1, a2), f"spatial_pos mismatch n={n}"
            assert torch.equal(b1, b2), f"edge_path mismatch n={n}"
    print("run_bfs_selftest: OK (prefix vs legacy).")


def build_pyg_from_json_path(
    json_path: Path | str,
    spatial_pos_max: int = 32,
    skip_a2: bool = False,
    inference_profile: Optional[InferenceProfile] = None,
    max_edge_path_len: int = 16,
    float16_storage: bool = False,
    shortest_path_workers: int = 0,
    legacy_bfs: bool = False,
) -> Any:
    """Load one JSON path, set ``data_id`` from stem (fallback 0 like ``data.dgl_bin_to_pyg``).

    If ``skip_a2`` is True, it forces ``inference_profile`` to ``no_a2`` regardless of ``inference_profile``.
    """
    json_path = Path(json_path)
    data = load_json_fast(json_path)
    prof: InferenceProfile = "no_a2" if skip_a2 else (inference_profile or "full")
    pyg, _ = tensors_from_brep_json_dict(
        data,
        spatial_pos_max=spatial_pos_max,
        inference_profile=prof,
        max_edge_path_len=max_edge_path_len,
        float16_storage=float16_storage,
        shortest_path_workers=shortest_path_workers,
        legacy_bfs=legacy_bfs,
    )
    stem = json_path.stem
    try:
        pyg.data_id = int(stem.split("_")[-1])
    except ValueError:
        pyg.data_id = 0
    return pyg


def _atomic_torch_save(obj, path: Path) -> None:
    """Write ``.pt`` to a temp file in the destination directory, then replace.

    Direct ``torch.save`` into the final path can leave a truncated ZIP if a
    worker is killed mid-write. The lite converter previously skipped any
    existing ``.pt``, so those truncated files survived into A1/A3 upgrade.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dest_tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        torch.save(obj, dest_tmp)
        os.replace(dest_tmp, path)
    except BaseException:
        try:
            dest_tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def convert_one_json(
    json_path: Path,
    pt_out_dir: Path,
    label_out_dir: Optional[Path],
    spatial_pos_max: int = 32,
    inference_profile: InferenceProfile = "full",
    max_edge_path_len: int = 16,
    float16_storage: bool = False,
    shortest_path_workers: int = 0,
    *,
    profile: bool = False,
    legacy_bfs: bool = False,
    cache_dir: Optional[Path] = None,
    use_npz_cache: bool = False,
    rebuild_cache: bool = False,
    npz_direct: bool = True,
    out_timing: Optional[dict[str, float]] = None,
) -> Path:
    file_stem = json_path.stem
    cache_path = (cache_dir / f"{file_stem}.npz") if cache_dir else None
    timing: Optional[dict[str, float]] = {} if (profile or out_timing is not None) else None
    wall0 = time.perf_counter()

    from_cache = bool(
        use_npz_cache and cache_path is not None and cache_path.is_file() and not rebuild_cache
    )
    want_save_cache = bool(
        use_npz_cache and cache_path is not None and (rebuild_cache or not cache_path.is_file())
    )

    t_io = time.perf_counter()
    z: Any = None
    if from_cache:
        z = np.load(cache_path, mmap_mode="c", allow_pickle=False)
        if timing is not None:
            timing["io_load"] = time.perf_counter() - t_io
        try:
            if npz_direct:
                pyg, labels_list = tensors_from_npz_arrays(
                    z,
                    spatial_pos_max=spatial_pos_max,
                    inference_profile=inference_profile,
                    max_edge_path_len=max_edge_path_len,
                    float16_storage=float16_storage,
                    shortest_path_workers=shortest_path_workers,
                    timing=timing,
                    legacy_bfs=legacy_bfs,
                )
            else:
                data = _data_dict_from_npz_opened(z)
                pyg, labels_list = tensors_from_brep_json_dict(
                    data,
                    spatial_pos_max=spatial_pos_max,
                    inference_profile=inference_profile,
                    max_edge_path_len=max_edge_path_len,
                    float16_storage=float16_storage,
                    shortest_path_workers=shortest_path_workers,
                    timing=timing,
                    legacy_bfs=legacy_bfs,
                    npz_cache_path=None,
                )
        finally:
            if z is not None:
                z.close()
    else:
        data = load_json_fast(json_path)
        if timing is not None:
            timing["io_load"] = time.perf_counter() - t_io
        pyg, labels_list = tensors_from_brep_json_dict(
            data,
            spatial_pos_max=spatial_pos_max,
            inference_profile=inference_profile,
            max_edge_path_len=max_edge_path_len,
            float16_storage=float16_storage,
            shortest_path_workers=shortest_path_workers,
            timing=timing,
            legacy_bfs=legacy_bfs,
            npz_cache_path=cache_path if want_save_cache and not from_cache else None,
        )

    if profile and timing:
        print(f"[TIME] tensors_from breakdown ({file_stem}):")
        order = (
            "io_load",
            "tensor_prep",
            "shortest_path_a1_a3",
            "pyg_pack",
            "npz_cache_write",
        )
        for k in order:
            if k in timing:
                print(f"       {k}: {timing[k]:.4f}s")
        for k in sorted(timing.keys()):
            if k not in order and k not in ("torch_save", "total_wall"):
                print(f"       {k}: {timing[k]:.4f}s")

    try:
        pyg.data_id = int(file_stem.split("_")[-1])
    except ValueError:
        pyg.data_id = 0

    pt_out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = pt_out_dir / f"{file_stem}.pt"
    if profile:
        print_pyg_tensor_sizes(pyg, title=f"{file_stem} | profile={inference_profile}")
    t_save = time.perf_counter()
    _atomic_torch_save(pyg, out_pt)
    if timing is not None:
        timing["torch_save"] = time.perf_counter() - t_save
        timing["total_wall"] = time.perf_counter() - wall0

    if profile:
        print(f"[TIME] torch_save {timing['torch_save']:.4f}s")
        print(f"[TIME] total_wall {timing['total_wall']:.4f}s")

    if out_timing is not None and timing is not None:
        out_timing.clear()
        out_timing.update(timing)

    if label_out_dir is not None:
        _write_label_json(label_out_dir, file_stem, labels_list)

    return out_pt


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert SolidWorks JSON → PyG ``.pt`` (+ optional labels). "
            "Optimized replica: compact dtypes, optional NPZ pre-BFS cache, orjson when installed."
        )
    )
    parser.add_argument("--json_dir", type=str, default=None, help="Input folder containing per-model *.json")
    parser.add_argument(
        "--abc_json_dir",
        type=str,
        default=None,
        help="Optional second JSON folder (e.g. abc_jsons). Converted into the same --pt_out_dir / --label_out_dir.",
    )
    parser.add_argument("--pt_out_dir", type=str, default=None, help="Output folder for ``.pt`` graphs")
    parser.add_argument("--label_out_dir", type=str, default=None)
    parser.add_argument("--spatial_pos_max", type=int, default=32)
    parser.add_argument(
        "--inference_profile",
        type=str,
        choices=("full", "no_a2", "lite"),
        default="full",
        help="full: A1+A2+A3; no_a2: A1+A3 (no dense A2); lite: omit A1/A2/A3 (smallest .pt). attn_bias is never stored; collator synthesizes zeros.",
    )
    parser.add_argument(
        "--skip_a2",
        action="store_true",
        help="Deprecated: same as --inference_profile no_a2 (overrides profile when set).",
    )
    parser.add_argument(
        "--max_edge_path_len",
        type=int,
        default=16,
        help="Max hops stored in edge_path / shortest-path BFS cap (align with collator multi_hop_max_dist).",
    )
    parser.add_argument(
        "--float16_storage",
        action="store_true",
        help="Save node_data, edge_data, face_area as float16 (collator promotes to float32 before the encoder).",
    )
    parser.add_argument(
        "--shortest_path_workers",
        type=int,
        default=0,
        help="Parallel BFS sources for A1 (full/no_a2 only). 0=serial. Try 8 on large N; ignored for lite or N<64.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-stage timings, tensor byte sizes, and torch.save duration for each converted file.",
    )
    parser.add_argument(
        "--legacy-bfs",
        action="store_true",
        help="Use legacy serial backtracking BFS (debug / parity); disables prefix path in the serial path.",
    )
    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        type=str,
        default=None,
        help="Directory for optional ``<stem>.npz`` caches (pre-BFS arrays + topology). Used with --use-npz-cache.",
    )
    parser.add_argument(
        "--use-npz-cache",
        action="store_true",
        help="Read ``<stem>.npz`` from --cache-dir when present (skip JSON parse); write cache when missing or with --rebuild-cache.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore existing ``.npz`` and rebuild from JSON (requires --use-npz-cache and --cache-dir).",
    )
    parser.add_argument(
        "--selftest-bfs",
        action="store_true",
        help="Run prefix vs legacy BFS self-test on random graphs and exit (no conversion).",
    )
    parser.add_argument(
        "--no-npz-direct",
        action="store_true",
        help="On NPZ cache hit, rebuild macro dicts and use the JSON tensor path (slower).",
    )
    parser.add_argument(
        "--bench-npz-cache",
        action="store_true",
        help="Two-pass benchmark per JSON: (1) JSON→NPZ+.pt (2) NPZ-direct→.pt; print summed timings.",
    )
    parser.add_argument(
        "--bench-scipy-bfs",
        action="store_true",
        help="Time SciPy CSR hop matrix vs prefix serial BFS (hop counts only; must match).",
    )
    parser.add_argument(
        "--bench-scipy-limit",
        type=int,
        default=0,
        help="Max JSON files for --bench-scipy-bfs (0 = all).",
    )
    parser.add_argument(
        "--bench-scipy-max-n",
        type=int,
        default=800,
        help="Skip graphs with more than this many faces (serial prefix BFS is O(N²) per source). Use 0 for no cap.",
    )
    args = parser.parse_args()

    if args.selftest_bfs:
        run_bfs_selftest()
        return

    if args.bench_scipy_bfs:
        if not args.json_dir:
            parser.error("--bench-scipy-bfs requires --json-dir")
        json_dir = Path(args.json_dir)
        files = sorted(json_dir.glob("*.json"))
        lim = int(args.bench_scipy_limit)
        if lim > 0:
            files = files[:lim]
        print(
            "bench_scipy_bfs: SciPy CSR all-pairs hop counts vs prefix serial BFS "
            f"({len(files)} file(s), max_edge_path_len={args.max_edge_path_len})."
        )
        for jp in files:
            data = load_json_fast(jp)
            fs, fd, n = _directed_arcs_from_macro_dict(data)
            cap = int(args.bench_scipy_max_n)
            if cap > 0 and n > cap:
                print(f"  {jp.name}  N={n}  SKIP (N>{cap}; raise --bench-scipy-max-n or use a smaller JSON folder)")
                continue
            st = benchmark_scipy_vs_prefix_bfs_hops(fs, fd, n, max_dist=args.max_edge_path_len)
            ok = st["distances_match"]
            print(
                f"  {jp.name}  N={n}  prefix={st['prefix_serial_s']:.4f}s  scipy={st['scipy_csr_s']:.4f}s  "
                f"match={ok}  scipy/prefix={st['speedup_scipy_vs_prefix']:.3f}x"
            )
        return

    if args.bench_npz_cache:
        if not args.json_dir or not args.pt_out_dir or not args.cache_dir:
            parser.error("--bench-npz-cache requires --json-dir, --pt-out-dir, and --cache-dir")
        json_dir = Path(args.json_dir)
        pt_out_dir = Path(args.pt_out_dir)
        label_out_dir = Path(args.label_out_dir) if args.label_out_dir else None
        cache_dir = Path(args.cache_dir)
        profile_eff: InferenceProfile = "no_a2" if args.skip_a2 else args.inference_profile  # type: ignore[assignment]
        json_files = sorted(json_dir.glob("*.json"))
        _ = _new_pyg_graph()  # warmup lazy torch_geometric import so pyg_pack timings are comparable
        keys = [
            "io_load",
            "tensor_prep",
            "shortest_path_a1_a3",
            "pyg_pack",
            "npz_cache_write",
            "torch_save",
            "total_wall",
        ]
        sum1 = {k: 0.0 for k in keys}
        sum2 = {k: 0.0 for k in keys}
        n_ok = 0
        for jp in json_files:
            stem = jp.stem
            cpath = cache_dir / f"{stem}.npz"
            outp = pt_out_dir / f"{stem}.pt"
            if cpath.exists():
                cpath.unlink()
            if outp.exists():
                outp.unlink()
            t1: dict[str, float] = {}
            try:
                convert_one_json(
                    jp,
                    pt_out_dir,
                    label_out_dir,
                    spatial_pos_max=args.spatial_pos_max,
                    inference_profile=profile_eff,
                    max_edge_path_len=args.max_edge_path_len,
                    float16_storage=args.float16_storage,
                    shortest_path_workers=args.shortest_path_workers,
                    profile=False,
                    legacy_bfs=args.legacy_bfs,
                    cache_dir=cache_dir,
                    use_npz_cache=True,
                    rebuild_cache=True,
                    npz_direct=False,
                    out_timing=t1,
                )
            except Exception as e:
                print(f"[bench-npz-cache] run1 FAIL {jp.name}: {e}")
                traceback.print_exc()
                continue
            for k in keys:
                sum1[k] += float(t1.get(k, 0.0))
            if outp.exists():
                outp.unlink()
            t2: dict[str, float] = {}
            try:
                convert_one_json(
                    jp,
                    pt_out_dir,
                    label_out_dir,
                    spatial_pos_max=args.spatial_pos_max,
                    inference_profile=profile_eff,
                    max_edge_path_len=args.max_edge_path_len,
                    float16_storage=args.float16_storage,
                    shortest_path_workers=args.shortest_path_workers,
                    profile=False,
                    legacy_bfs=args.legacy_bfs,
                    cache_dir=cache_dir,
                    use_npz_cache=True,
                    rebuild_cache=False,
                    npz_direct=True,
                    out_timing=t2,
                )
            except Exception as e:
                print(f"[bench-npz-cache] run2 FAIL {jp.name}: {e}")
                traceback.print_exc()
                continue
            for k in keys:
                sum2[k] += float(t2.get(k, 0.0))
            n_ok += 1

        print("\n" + "=" * 80)
        print(f"bench-npz-cache: summed over {n_ok} JSON file(s)")
        print("=" * 80)
        print(f"{'metric':<28} {'run1 JSON+NPZ+PT':>18} {'run2 NPZ-direct+PT':>22} {'run2/run1':>12}")
        for k in keys:
            a, b = sum1[k], sum2[k]
            ratio = (b / a) if a > 1e-9 else float("nan")
            print(f"{k:<28} {a:>18.4f} {b:>22.4f} {ratio:>12.4f}")
        print("=" * 80)
        return

    if not args.json_dir or not args.pt_out_dir:
        parser.error("conversion requires --json_dir and --pt_out_dir (omit when using --selftest-bfs only)")

    if args.use_npz_cache and not args.cache_dir:
        parser.error("--use-npz-cache requires --cache-dir")
    if args.rebuild_cache and not args.use_npz_cache:
        parser.error("--rebuild-cache requires --use-npz-cache")

    profile: InferenceProfile = "no_a2" if args.skip_a2 else args.inference_profile  # type: ignore[assignment]

    json_dir = Path(args.json_dir)
    pt_out_dir = Path(args.pt_out_dir)
    label_out_dir = Path(args.label_out_dir) if args.label_out_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Primary folder + optional ABC / extra folder → same .pt / label outputs
    json_files: list[Path] = sorted(json_dir.glob("*.json"))
    abc_stems: list[str] = []
    if args.abc_json_dir:
        abc_dir = Path(args.abc_json_dir)
        if not abc_dir.is_dir():
            parser.error(f"--abc_json_dir is not a directory: {abc_dir}")
        abc_files = sorted(abc_dir.glob("*.json"))
        abc_stems = [p.stem for p in abc_files]
        primary_names = {p.name for p in json_files}
        overlap = [p.name for p in abc_files if p.name in primary_names]
        if overlap:
            print(
                f"[WARN] {len(overlap)} filename(s) exist in both --json_dir and --abc_json_dir; "
                "ABC copy will overwrite / skip the same stem. First overlap: "
                + ", ".join(overlap[:5])
            )
        json_files = sorted(json_files + abc_files, key=lambda p: p.name.lower())
        print(f"Primary JSON: {len(list(json_dir.glob('*.json')))} | ABC JSON: {len(abc_files)} | Combined: {len(json_files)}")

    ok = 0
    skipped = 0
    failed = 0
    consecutive_output_open_failures = 0
    conversion_times = []
    wall_start = time.time()

    for jp in tqdm(json_files, desc="Converting", unit="file"):
        file_stem = jp.stem
        pt_exists = (pt_out_dir / f"{file_stem}.pt").exists()
        label_exists = (label_out_dir / f"{file_stem}.json").exists() if label_out_dir else True
        if pt_exists and label_exists:
            skipped += 1
            continue
        t0 = time.perf_counter()
        try:
            convert_one_json(
                jp,
                pt_out_dir,
                label_out_dir,
                spatial_pos_max=args.spatial_pos_max,
                inference_profile=profile,
                max_edge_path_len=args.max_edge_path_len,
                float16_storage=args.float16_storage,
                shortest_path_workers=args.shortest_path_workers,
                profile=args.profile,
                legacy_bfs=args.legacy_bfs,
                cache_dir=cache_dir,
                use_npz_cache=args.use_npz_cache,
                rebuild_cache=args.rebuild_cache,
                npz_direct=not args.no_npz_direct,
            )
            ok += 1
            consecutive_output_open_failures = 0
        except Exception as e:
            print(f"\n[FAIL] {jp.name}: {e}")
            traceback.print_exc()
            failed += 1
            message = str(e).lower()
            if "cannot be opened" in message or "pytorchfilewriter" in message:
                consecutive_output_open_failures += 1
                if consecutive_output_open_failures >= 3:
                    try:
                        free_gb = shutil.disk_usage(pt_out_dir).free / (1024 ** 3)
                        free_text = f"{free_gb:.2f} GiB"
                    except OSError:
                        free_text = "unknown"
                    raise RuntimeError(
                        "Aborting after 3 consecutive output-open failures. "
                        f"Destination={pt_out_dir}; free space={free_text}. "
                        "Check disk space, directory existence, permissions, and concurrent writers."
                    ) from e
            else:
                consecutive_output_open_failures = 0
        conversion_times.append(time.perf_counter() - t0)

    wall_total = time.time() - wall_start
    print(
        f"\nDone. Converted: {ok} | Skipped (already exist): {skipped} | Failed: {failed} | Total: {len(json_files)}"
    )
    if conversion_times:
        avg_ms = (sum(conversion_times) / len(conversion_times)) * 1000
        min_ms = min(conversion_times) * 1000
        max_ms = max(conversion_times) * 1000
        print(f"Per-file conversion time — avg: {avg_ms:.1f} ms | min: {min_ms:.1f} ms | max: {max_ms:.1f} ms")
    print(f"Total wall-clock time: {wall_total:.1f} s ({wall_total/60:.2f} min)")

    # Manifest for make_random_splits.py --abc-json-dir (stems that came from ABC folder)
    if abc_stems:
        manifest = pt_out_dir.parent / "abc_stems.txt"
        manifest.write_text("\n".join(abc_stems) + ("\n" if abc_stems else ""), encoding="utf-8")
        print(f"Wrote ABC stem manifest: {manifest}  ({len(abc_stems):,} stems)")


if __name__ == "__main__":
    main()
