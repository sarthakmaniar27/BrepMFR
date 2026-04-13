# json_to_brepmfr_bin.py
# Converts your SolidWorks-produced JSON files into BrepMFR-compatible DGL .bin graphs.

import argparse
import json
import os
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import torch
import dgl
from dgl.data.utils import save_graphs


def _reshape_face_uv(flat_uv: list, U: int = 5, V: int = 5, C: int = 7) -> np.ndarray:
    arr = np.asarray(flat_uv, dtype=np.float32)
    expected = U * V * C
    if arr.size != expected:
        raise ValueError(f"Face uv length {arr.size} != expected {expected} ({U}*{V}*{C})")
    return arr.reshape(U, V, C)


def _reshape_edge_pt(flat_pt: list, U: int = 5, C: int = 6) -> np.ndarray:
    arr = np.asarray(flat_pt, dtype=np.float32)
    expected = U * C
    if arr.size != expected:
        raise ValueError(f"Edge pt length {arr.size} != expected {expected} ({U}*{C})")
    return arr.reshape(U, C)


def _build_face_id_map(faces: list) -> dict:
    """
    Faces can appear unsorted. We map face 'id' -> node_index [0..N-1].
    Node order is sorted by face id to stay stable.
    """
    face_ids = [int(f["id"]) for f in faces]
    uniq = sorted(set(face_ids))
    if len(uniq) != len(face_ids):
        # If duplicates exist, that's ambiguous for graph nodes.
        raise ValueError("Duplicate face ids found in faces[]. 'id' must be unique per face.")
    return {fid: i for i, fid in enumerate(uniq)}


def _edge_obj_lookup(edges: list):
    """
    Build lookup: (src_face_id, dst_face_id) -> edge_obj
    We assume edge['nf'] is ordered [src, dst] (directed).
    If your nf is undirected, we still fallback to reversed lookup.
    """
    lut = {}
    for e in edges:
        nf = e.get("nf", None)
        if not nf or len(nf) != 2:
            raise ValueError("Each edge must have 'nf': [src_face_id, dst_face_id]")
        s, d = int(nf[0]), int(nf[1])
        lut[(s, d)] = e
    return lut


def _compute_shortest_paths_edge_types(
    src_nodes: list,
    dst_nodes: list,
    edge_types: torch.Tensor,
    num_nodes: int,
    max_dist: int = 16,
    unreachable_dist: int = 10**9,
):
    """
    Returns:
      spatial_pos: [N, N] int64 (shortest path length; 0 on diag; large for unreachable)
      edges_path:  [N, N, max_dist] int64 (edge type along shortest path, padded with -1)
                  edges_path[i,j,k] = edge_type of k-th hop from i->j, else -1
    """
    # adjacency list storing (neighbor, edge_idx)
    adj = [[] for _ in range(num_nodes)]
    for ei, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
        adj[u].append((v, ei))

    spatial_pos = torch.full((num_nodes, num_nodes), fill_value=unreachable_dist, dtype=torch.long)
    edges_path = torch.full((num_nodes, num_nodes, max_dist), fill_value=-1, dtype=torch.long)

    for s in range(num_nodes):
        spatial_pos[s, s] = 0

        # BFS
        dist = [-1] * num_nodes
        prev_node = [-1] * num_nodes
        prev_edge = [-1] * num_nodes

        q = deque([s])
        dist[s] = 0

        while q:
            u = q.popleft()
            du = dist[u]
            if du >= max_dist:
                # We still want distances, but path tensor only stores up to max_dist hops.
                # Continue BFS anyway if you want full dist; here we can continue but it costs more.
                pass

            for v, ei in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    prev_node[v] = u
                    prev_edge[v] = ei
                    q.append(v)

        # fill tensors for this source
        for t in range(num_nodes):
            if dist[t] == -1:
                continue
            spatial_pos[s, t] = dist[t]

            # reconstruct path edges (backtrack)
            if t == s:
                continue
            path_edges = []
            cur = t
            while cur != s and cur != -1:
                ei = prev_edge[cur]
                if ei == -1:
                    break
                path_edges.append(ei)
                cur = prev_node[cur]
            path_edges.reverse()

            # write edge types into edges_path
            for k in range(min(len(path_edges), max_dist)):
                ei = path_edges[k]
                edges_path[s, t, k] = int(edge_types[ei].item())

    return spatial_pos, edges_path


def convert_one_json(json_path: Path, out_dir: Path, max_dist: int = 16, spatial_pos_max: int = 32):
    data = json.loads(json_path.read_text())

    faces = data["faces"]
    edges = data["edges"]
    conn = data["connectivity"]
    src_raw = conn["src"]
    dst_raw = conn["dest"]

    face_id_to_node = _build_face_id_map(faces)

    # Build node features arrays aligned by node index
    # Node index order is sorted(face_id). We'll write into arrays accordingly.
    N = len(faces)

    node_x = np.zeros((N, 5, 5, 7), dtype=np.float32)
    node_z = np.zeros((N,), dtype=np.int64)     # face type
    node_y = np.zeros((N,), dtype=np.float32)   # face area
    node_l = np.zeros((N,), dtype=np.int64)     # loops
    node_a = np.zeros((N,), dtype=np.int64)     # adjacent faces count
    node_f = np.zeros((N,), dtype=np.int64)     # label (if absent -> 0)

    for f in faces:
        fid = int(f["id"])
        ni = face_id_to_node[fid]

        node_x[ni] = _reshape_face_uv(f["uv"], 5, 5, 7)
        node_z[ni] = int(f["z"])
        node_y[ni] = float(f["y"])
        node_l[ni] = int(f["l"])
        node_a[ni] = int(f["a"])
        if "f" in f:
            node_f[ni] = int(f["f"])

    # Build edge list from connectivity; remap face ids to node indices
    src_nodes = [face_id_to_node[int(s)] for s in src_raw]
    dst_nodes = [face_id_to_node[int(d)] for d in dst_raw]
    E = len(src_nodes)

    # Lookup edge objects by (nf0,nf1)
    lut = _edge_obj_lookup(edges)

    edge_x = np.zeros((E, 5, 6), dtype=np.float32)
    edge_t = np.zeros((E,), dtype=np.int64)
    edge_l = np.zeros((E,), dtype=np.float32)
    edge_a = np.zeros((E,), dtype=np.float32)
    edge_c = np.zeros((E,), dtype=np.int64)

    # We need to assign each connectivity edge its corresponding edge record.
    # Prefer exact direction match on nf; otherwise fallback to reversed.
    for i, (s_fid, d_fid) in enumerate(zip(src_raw, dst_raw)):
        s_fid = int(s_fid)
        d_fid = int(d_fid)

        eobj = lut.get((s_fid, d_fid), None)
        if eobj is None:
            eobj = lut.get((d_fid, s_fid), None)
        if eobj is None:
            raise ValueError(
                f"No edge record found for connectivity ({s_fid}->{d_fid}). "
                "Ensure edges[].nf covers all directed pairs or at least undirected pairs."
            )

        edge_x[i] = _reshape_edge_pt(eobj["pt"], 5, 6)
        edge_t[i] = int(eobj["t"])
        edge_l[i] = float(eobj["l"])
        edge_a[i] = float(eobj["a"])
        edge_c[i] = int(eobj["c"])

    # Create DGL graph
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)), num_nodes=N)

    # Attach node features
    g.ndata["x"] = torch.from_numpy(node_x).float()
    g.ndata["z"] = torch.from_numpy(node_z).long()
    g.ndata["y"] = torch.from_numpy(node_y).float()
    g.ndata["l"] = torch.from_numpy(node_l).long()
    g.ndata["a"] = torch.from_numpy(node_a).long()
    g.ndata["f"] = torch.from_numpy(node_f).long()

    # Attach edge features
    g.edata["x"] = torch.from_numpy(edge_x).float()
    g.edata["t"] = torch.from_numpy(edge_t).long()
    g.edata["l"] = torch.from_numpy(edge_l).float()
    g.edata["a"] = torch.from_numpy(edge_a).float()
    g.edata["c"] = torch.from_numpy(edge_c).long()

    # ---- A1 / A3 tensors expected by dataset.py ----
    # spatial_pos: shortest path lengths
    # edges_path: shortest path edge-type sequences (A3-ish)
    spatial_pos, edges_path = _compute_shortest_paths_edge_types(
        src_nodes=src_nodes,
        dst_nodes=dst_nodes,
        edge_types=g.edata["t"],
        num_nodes=N,
        max_dist=max_dist,
    )

    # cap spatial_pos for masking behavior in collator (anything >= spatial_pos_max becomes -inf in attn_bias)
    spatial_pos = spatial_pos.clamp(max=spatial_pos_max)

    # A2 skipped: store zeros, but keep correct shapes
    d2_distance = torch.zeros((N, N, 64), dtype=torch.float32)
    angle_distance = torch.zeros((N, N, 64), dtype=torch.float32)

    # Save
    out_path = out_dir / (json_path.stem + ".bin")
    save_graphs(
        str(out_path),
        [g],
        {
            "edges_path": edges_path.long(),
            "spatial_pos": spatial_pos.long(),
            "d2_distance": d2_distance.float(),
            "angle_distance": angle_distance.float(),
        },
    )
    return out_path


def main():
    parser = argparse.ArgumentParser("Convert SolidWorks JSON -> BrepMFR DGL .bin graphs")
    parser.add_argument("--json_dir", type=str, required=True, help="Folder containing *.json files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for *.bin files")
    parser.add_argument("--max_dist", type=int, default=16, help="edges_path third dim (BrepMFR uses 16)")
    parser.add_argument("--spatial_pos_max", type=int, default=32, help="cap spatial_pos (BrepMFR uses 32)")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    ok = 0
    for jp in json_files:
        try:
            outp = convert_one_json(jp, out_dir, max_dist=args.max_dist, spatial_pos_max=args.spatial_pos_max)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {jp.name}: {e}")

    print(f"Converted {ok}/{len(json_files)} files into {out_dir}")


if __name__ == "__main__":
    main()

# python json_to_brepmfr_bin.py --json_dir /path/to/jsons --out_dir /path/to/bins
# json_to_brepmfr_bin.py
# Converts your SolidWorks-produced JSON files into BrepMFR-compatible DGL .bin graphs.

import argparse
import json
import os
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import torch
import dgl
from dgl.data.utils import save_graphs


def _reshape_face_uv(flat_uv: list, U: int = 5, V: int = 5, C: int = 7) -> np.ndarray:
    arr = np.asarray(flat_uv, dtype=np.float32)
    expected = U * V * C
    if arr.size != expected:
        raise ValueError(f"Face uv length {arr.size} != expected {expected} ({U}*{V}*{C})")
    return arr.reshape(U, V, C)


def _reshape_edge_pt(flat_pt: list, U: int = 5, C: int = 6) -> np.ndarray:
    arr = np.asarray(flat_pt, dtype=np.float32)
    expected = U * C
    if arr.size != expected:
        raise ValueError(f"Edge pt length {arr.size} != expected {expected} ({U}*{C})")
    return arr.reshape(U, C)


def _build_face_id_map(faces: list) -> dict:
    """
    Faces can appear unsorted. We map face 'id' -> node_index [0..N-1].
    Node order is sorted by face id to stay stable.
    """
    face_ids = [int(f["id"]) for f in faces]
    uniq = sorted(set(face_ids))
    if len(uniq) != len(face_ids):
        # If duplicates exist, that's ambiguous for graph nodes.
        raise ValueError("Duplicate face ids found in faces[]. 'id' must be unique per face.")
    return {fid: i for i, fid in enumerate(uniq)}


def _edge_obj_lookup(edges: list):
    """
    Build lookup: (src_face_id, dst_face_id) -> edge_obj
    We assume edge['nf'] is ordered [src, dst] (directed).
    If your nf is undirected, we still fallback to reversed lookup.
    """
    lut = {}
    for e in edges:
        nf = e.get("nf", None)
        if not nf or len(nf) != 2:
            raise ValueError("Each edge must have 'nf': [src_face_id, dst_face_id]")
        s, d = int(nf[0]), int(nf[1])
        lut[(s, d)] = e
    return lut


def _compute_shortest_paths_edge_types(
    src_nodes: list,
    dst_nodes: list,
    edge_types: torch.Tensor,
    num_nodes: int,
    max_dist: int = 16,
    unreachable_dist: int = 10**9,
):
    """
    Returns:
      spatial_pos: [N, N] int64 (shortest path length; 0 on diag; large for unreachable)
      edges_path:  [N, N, max_dist] int64 (edge type along shortest path, padded with -1)
                  edges_path[i,j,k] = edge_type of k-th hop from i->j, else -1
    """
    # adjacency list storing (neighbor, edge_idx)
    adj = [[] for _ in range(num_nodes)]
    for ei, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
        adj[u].append((v, ei))

    spatial_pos = torch.full((num_nodes, num_nodes), fill_value=unreachable_dist, dtype=torch.long)
    edges_path = torch.full((num_nodes, num_nodes, max_dist), fill_value=-1, dtype=torch.long)

    for s in range(num_nodes):
        spatial_pos[s, s] = 0

        # BFS
        dist = [-1] * num_nodes
        prev_node = [-1] * num_nodes
        prev_edge = [-1] * num_nodes

        q = deque([s])
        dist[s] = 0

        while q:
            u = q.popleft()
            du = dist[u]
            if du >= max_dist:
                # We still want distances, but path tensor only stores up to max_dist hops.
                # Continue BFS anyway if you want full dist; here we can continue but it costs more.
                pass

            for v, ei in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    prev_node[v] = u
                    prev_edge[v] = ei
                    q.append(v)

        # fill tensors for this source
        for t in range(num_nodes):
            if dist[t] == -1:
                continue
            spatial_pos[s, t] = dist[t]

            # reconstruct path edges (backtrack)
            if t == s:
                continue
            path_edges = []
            cur = t
            while cur != s and cur != -1:
                ei = prev_edge[cur]
                if ei == -1:
                    break
                path_edges.append(ei)
                cur = prev_node[cur]
            path_edges.reverse()

            # write edge types into edges_path
            for k in range(min(len(path_edges), max_dist)):
                ei = path_edges[k]
                edges_path[s, t, k] = int(edge_types[ei].item())

    return spatial_pos, edges_path


def convert_one_json(json_path: Path, out_dir: Path, max_dist: int = 16, spatial_pos_max: int = 32):
    data = json.loads(json_path.read_text())

    faces = data["faces"]
    edges = data["edges"]
    conn = data["connectivity"]
    src_raw = conn["src"]
    dst_raw = conn["dest"]

    face_id_to_node = _build_face_id_map(faces)

    # Build node features arrays aligned by node index
    # Node index order is sorted(face_id). We'll write into arrays accordingly.
    N = len(faces)

    node_x = np.zeros((N, 5, 5, 7), dtype=np.float32)
    node_z = np.zeros((N,), dtype=np.int64)     # face type
    node_y = np.zeros((N,), dtype=np.float32)   # face area
    node_l = np.zeros((N,), dtype=np.int64)     # loops
    node_a = np.zeros((N,), dtype=np.int64)     # adjacent faces count
    node_f = np.zeros((N,), dtype=np.int64)     # label (if absent -> 0)
    node_id = np.zeros((N,), dtype=np.int64)

    for f in faces:
        fid = int(f["id"])
        ni = face_id_to_node[fid]

        node_x[ni] = _reshape_face_uv(f["uv"], 5, 5, 7)
        node_z[ni] = int(f["z"])
        node_y[ni] = float(f["y"])
        node_l[ni] = int(f["l"])
        node_a[ni] = int(f["a"])
        node_id[ni] = int(f["id"])
        if "f" in f:
            node_f[ni] = int(f["f"])
        

    # Build edge list from connectivity; remap face ids to node indices
    src_nodes = [face_id_to_node[int(s)] for s in src_raw]
    dst_nodes = [face_id_to_node[int(d)] for d in dst_raw]
    E = len(src_nodes)

    # Lookup edge objects by (nf0,nf1)
    lut = _edge_obj_lookup(edges)

    edge_x = np.zeros((E, 5, 6), dtype=np.float32)
    edge_t = np.zeros((E,), dtype=np.int64)
    edge_l = np.zeros((E,), dtype=np.float32)
    edge_a = np.zeros((E,), dtype=np.float32)
    edge_c = np.zeros((E,), dtype=np.int64)
    edge_id = np.zeros((E,), dtype=np.int64)

    # We need to assign each connectivity edge its corresponding edge record.
    # Prefer exact direction match on nf; otherwise fallback to reversed.
    for i, (s_fid, d_fid) in enumerate(zip(src_raw, dst_raw)):
        s_fid = int(s_fid)
        d_fid = int(d_fid)

        eobj = lut.get((s_fid, d_fid), None)
        if eobj is None:
            eobj = lut.get((d_fid, s_fid), None)
        if eobj is None:
            raise ValueError(
                f"No edge record found for connectivity ({s_fid}->{d_fid}). "
                "Ensure edges[].nf covers all directed pairs or at least undirected pairs."
            )

        edge_x[i] = _reshape_edge_pt(eobj["pt"], 5, 6)
        edge_t[i] = int(eobj["t"])
        edge_l[i] = float(eobj["l"])
        edge_a[i] = float(eobj["a"])
        edge_c[i] = int(eobj["c"])
        edge_id[i] = int(eobj["id"])

    # Create DGL graph
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)), num_nodes=N)

    # Attach node features
    g.ndata["x"] = torch.from_numpy(node_x).float()
    g.ndata["z"] = torch.from_numpy(node_z).long()
    g.ndata["y"] = torch.from_numpy(node_y).float()
    g.ndata["l"] = torch.from_numpy(node_l).long()
    g.ndata["a"] = torch.from_numpy(node_a).long()
    g.ndata["f"] = torch.from_numpy(node_f).long()
    g.ndata["id"] = torch.from_numpy(node_id).long()

    # Attach edge features
    g.edata["x"] = torch.from_numpy(edge_x).float()
    g.edata["t"] = torch.from_numpy(edge_t).long()
    g.edata["l"] = torch.from_numpy(edge_l).float()
    g.edata["a"] = torch.from_numpy(edge_a).float()
    g.edata["c"] = torch.from_numpy(edge_c).long()
    g.edata["id"] = torch.from_numpy(edge_id).long()

    # ---- A1 / A3 tensors expected by dataset.py ----
    # spatial_pos: shortest path lengths
    # edges_path: shortest path edge-type sequences (A3-ish)
    spatial_pos, edges_path = _compute_shortest_paths_edge_types(
        src_nodes=src_nodes,
        dst_nodes=dst_nodes,
        edge_types=g.edata["t"],
        num_nodes=N,
        max_dist=max_dist,
    )

    # cap spatial_pos for masking behavior in collator (anything >= spatial_pos_max becomes -inf in attn_bias)
    spatial_pos = spatial_pos.clamp(max=spatial_pos_max)

    # A2 skipped: store zeros, but keep correct shapes
    d2_distance = torch.zeros((N, N, 64), dtype=torch.float32)
    angle_distance = torch.zeros((N, N, 64), dtype=torch.float32)

    # Save
    out_path = out_dir / (json_path.stem + ".bin")
    save_graphs(
        str(out_path),
        [g],
        {
            "edges_path": edges_path.long(),
            "spatial_pos": spatial_pos.long(),
            "d2_distance": d2_distance.float(),
            "angle_distance": angle_distance.float(),
        },
    )
    return out_path


def main():
    parser = argparse.ArgumentParser("Convert SolidWorks JSON -> BrepMFR DGL .bin graphs")
    parser.add_argument("--json_dir", type=str, required=True, help="Folder containing *.json files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for *.bin files")
    parser.add_argument("--max_dist", type=int, default=16, help="edges_path third dim (BrepMFR uses 16)")
    parser.add_argument("--spatial_pos_max", type=int, default=32, help="cap spatial_pos (BrepMFR uses 32)")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}")

    ok = 0
    for jp in json_files:
        try:
            outp = convert_one_json(jp, out_dir, max_dist=args.max_dist, spatial_pos_max=args.spatial_pos_max)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {jp.name}: {e}")

    print(f"Converted {ok}/{len(json_files)} files into {out_dir}")


if __name__ == "__main__":
    main()

# python json_to_brepmfr_bin.py --json_dir /path/to/jsons --out_dir /path/to/bins
