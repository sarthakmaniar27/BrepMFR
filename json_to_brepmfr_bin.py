# json_to_brepmfr_bin.py
# Converts SolidWorks-produced JSON files into BrepMFR-compatible DGL .bin graphs.
# ALSO exports a label JSON file per model in the requested format.

import argparse
import json
import time
import torch
import numpy as np
import dgl
from dgl.data.utils import save_graphs
from pathlib import Path
from collections import deque, defaultdict
import traceback
from typing import Optional
from tqdm import tqdm


def _reshape_face_uv(flat_uv: list, U: int = 5, V: int = 5, C: int = 7) -> np.ndarray:
    """Reshapes flat face list to (U, V, 7) grid: [x, y, z, nx, ny, nz, mask]."""
    arr = np.asarray(flat_uv, dtype=np.float32)
    return arr.reshape(U, V, C)


def _reshape_edge_pt(flat_pt: list, U: int = 5, C: int = 7) -> np.ndarray:
    """Reshapes flat edge list to (U, 7) grid: [x, y, z, tx, ty, tz, angle]."""
    arr = np.asarray(flat_pt, dtype=np.float32)
    return arr.reshape(U, C)


def _compute_shortest_paths_edge_indices(src_nodes, dst_nodes, num_nodes, max_dist=16):
    """
    Computes A1 (Shortest path distance) and A3 (Chain of edge indices).
    Matches Graphormer-style encoding where edges_path stores the sequence of edge IDs.
    """
    adj = [[] for _ in range(num_nodes)]
    for ei, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
        adj[u].append((v, ei))

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


def _build_a2_tensors(data: dict, face_id_to_node: dict, N: int) -> tuple:
    """
    Builds d2_distance and angle_distance tensors from the face_pairs field in JSON.

    face_pairs stores one entry per unordered pair {i, j} with i != j.
    The pair values are face IDs (matching face["id"]).
    Both (i,j) and (j,i) are filled symmetrically from the same entry.
    Diagonal entries (i==i) remain zero as per the paper (no self-spatial relation).

    Returns:
        d2_distance  : torch.FloatTensor shape (N, N, 64)
        angle_distance: torch.FloatTensor shape (N, N, 64)
    """
    d2_distance = torch.zeros((N, N, 64), dtype=torch.float32)
    angle_distance = torch.zeros((N, N, 64), dtype=torch.float32)

    face_pairs = data.get("face_pairs", [])
    if not face_pairs:
        # face_pairs key absent — return zeros (backward compat)
        return d2_distance, angle_distance

    # Build lookup: (face_id_a, face_id_b) -> entry
    # The JSON stores only one direction per pair — we handle both directions below.
    pair_lut = {}
    for entry in face_pairs:
        fi, fj = int(entry["face_pair"][0]), int(entry["face_pair"][1])
        pair_lut[(fi, fj)] = entry

    # node_to_face_id is the inverse of face_id_to_node
    node_to_face_id = {v: k for k, v in face_id_to_node.items()}

    for ni in range(N):
        for nj in range(N):
            if ni == nj:
                # Diagonal stays zero — paper explicitly zeros self-relations
                continue

            fi = node_to_face_id[ni]
            fj = node_to_face_id[nj]

            # Try both directions since JSON stores only one
            entry = pair_lut.get((fi, fj)) or pair_lut.get((fj, fi))
            if entry is None:
                # No pair data — leave as zero
                continue

            d2_distance[ni, nj] = torch.tensor(entry["d2"], dtype=torch.float32)
            angle_distance[ni, nj] = torch.tensor(entry["a3"], dtype=torch.float32)

    return d2_distance, angle_distance


def _write_label_json(label_out_dir: Path, file_stem: str, labels_list: list):
    """
    Writes a label file:
    {
      "file_name": "<stem>",
      "labels": [ ... per-face labels ... ]
    }
    """
    label_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_out_dir / f"{file_stem}.json"
    payload = {"file_name": file_stem, "labels": labels_list}
    out_path.write_text(json.dumps(payload, indent=3), encoding="utf-8")


def convert_one_json(json_path: Path, bin_out_dir: Path, label_out_dir: Optional[Path], spatial_pos_max: int = 32):
    """
    Core conversion logic for a single B-rep JSON.
    Ensures geometric alignment, bidirectional symmetry, and proximity encoding.

    New:
      - Reads face label from f["label"] and stores into g.ndata["f"]
      - Reads face_pairs for A2 proximity (d2_distance, angle_distance)
      - Optionally writes label json file to label_out_dir
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    faces, edges = data["faces"], data["edges"]

    # 1) Deterministic Face Mapping (sorted by face id)
    sorted_faces = sorted(faces, key=lambda x: int(x["id"]))
    face_id_to_node = {int(f["id"]): i for i, f in enumerate(sorted_faces)}
    node_to_face_id = {v: k for k, v in face_id_to_node.items()}
    N = len(sorted_faces)

    # 2) Build adjacency and edge lookup
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

    # 3) Node features
    node_x = np.zeros((N, 5, 5, 7), dtype=np.float32)
    node_z = np.zeros(N, dtype=np.int32)
    node_l = np.zeros(N, dtype=np.int32)
    node_a = np.zeros(N, dtype=np.int32)
    node_f = np.zeros(N, dtype=np.int32)  # labels
    node_y = np.zeros(N, dtype=np.float32)

    # IMPORTANT: use sorted_faces so labels align with node indices
    # Also collect labels list in the same order for writing label json
    labels_list = [0] * N

    for f in sorted_faces:
        ni = face_id_to_node[int(f["id"])]
        node_x[ni] = _reshape_face_uv(f["uv"])
        node_z[ni] = int(f["z"])
        node_y[ni] = float(f["y"])
        node_l[ni] = int(f["l"])
        node_a[ni] = int(f["a"])

        # Read "label" from face json (fallback to 0 if missing)
        lbl = int(f.get("label", 0))
        node_f[ni] = lbl
        labels_list[ni] = lbl

    # 4) Edge features with geometric flipping
    edge_x = np.zeros((E, 5, 7), dtype=np.float32)
    edge_t = np.zeros(E, dtype=np.int32)
    edge_c = np.zeros(E, dtype=np.int32)
    edge_l = np.zeros(E, dtype=np.float32)
    edge_a = np.zeros(E, dtype=np.float32)

    for i, (u_idx, v_idx) in enumerate(zip(final_src, final_dst)):
        u_fid, v_fid = node_to_face_id[u_idx], node_to_face_id[v_idx]
        eobj = edge_lut[frozenset([u_fid, v_fid])]

        raw_pts = _reshape_edge_pt(eobj["pt"])

        # Check direction: Is u the 'natural' first face?
        if u_fid == int(eobj["nf"][0]):
            edge_x[i] = raw_pts
        else:
            # FLIP: reverse points and negate tangent vectors
            flipped = np.flip(raw_pts, axis=0).copy()
            flipped[:, 3:6] *= -1.0
            edge_x[i] = flipped

        # Keep your existing behavior here
        edge_t[i] = int(eobj["t"])
        edge_l[i] = float(eobj["l"])
        edge_c[i] = int(eobj["c"])
        edge_a[i] = (float(eobj["a"]) + np.pi) % (2 * np.pi) - np.pi  # wrap to (-pi, pi]

    # 5) Graph + proximity tensors
    g = dgl.graph((torch.tensor(final_src), torch.tensor(final_dst)), num_nodes=N)

    g.ndata.update(
        {
            "x": torch.from_numpy(node_x),
            "z": torch.from_numpy(node_z).int(),
            "y": torch.from_numpy(node_y),
            "l": torch.from_numpy(node_l).int(),
            "a": torch.from_numpy(node_a).int(),
            "f": torch.from_numpy(node_f).int(),  # real labels
        }
    )

    g.edata.update(
        {
            "x": torch.from_numpy(edge_x),
            "t": torch.from_numpy(edge_t).int(),
            "l": torch.from_numpy(edge_l),
            "a": torch.from_numpy(edge_a),
            "c": torch.from_numpy(edge_c).int(),
        }
    )

    spatial_pos, edges_path = _compute_shortest_paths_edge_indices(final_src, final_dst, N)

    # A2 proximity: read real d2 and angle histograms from JSON
    d2_distance, angle_distance = _build_a2_tensors(data, face_id_to_node, N)

    # Final slicing and capping
    max_p = int(spatial_pos[spatial_pos < 10**8].max().item()) if N > 1 else 0
    edges_path = edges_path[:, :, :max_p]  # keeps your current behavior
    spatial_pos = spatial_pos.clamp(max=spatial_pos_max)

    # Output .bin
    bin_out_dir.mkdir(parents=True, exist_ok=True)
    file_stem = json_path.stem
    out_bin = bin_out_dir / f"{file_stem}.bin"
    save_graphs(
        str(out_bin),
        [g],
        {
            "edges_path": edges_path.int(),
            "spatial_pos": spatial_pos.int(),
            "d2_distance": d2_distance,
            "angle_distance": angle_distance,
        },
    )

    # Output label json
    if label_out_dir is not None:
        _write_label_json(label_out_dir, file_stem, labels_list)

    return out_bin


def main():
    parser = argparse.ArgumentParser("Convert SolidWorks JSON -> BrepMFR DGL .bin graphs (+ labels json)")
    parser.add_argument("--json_dir", type=str, required=True, help="Input folder containing per-model json files")
    parser.add_argument("--bin_out_dir", type=str, required=True, help="Output folder for .bin graphs")
    parser.add_argument("--label_out_dir", type=str, default=None, help="Output folder for labels .json (optional)")
    parser.add_argument("--spatial_pos_max", type=int, default=32)
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    bin_out_dir = Path(args.bin_out_dir)
    label_out_dir = Path(args.label_out_dir) if args.label_out_dir else None

    json_files = sorted(json_dir.glob("*.json"))
    ok = 0
    skipped = 0
    failed = 0

    conversion_times = []
    wall_start = time.time()

    for jp in tqdm(json_files, desc="Converting", unit="file"):
        file_stem = jp.stem

        # Skip if both .bin and label .json already exist
        bin_exists = (bin_out_dir / f"{file_stem}.bin").exists()
        label_exists = (label_out_dir / f"{file_stem}.json").exists() if label_out_dir else True
        if bin_exists and label_exists:
            skipped += 1
            continue

        t0 = time.perf_counter()
        try:
            convert_one_json(jp, bin_out_dir, label_out_dir, spatial_pos_max=args.spatial_pos_max)
            ok += 1
        except Exception as e:
            print(f"\n[FAIL] {jp.name}: {e}")
            traceback.print_exc()
            failed += 1
        conversion_times.append(time.perf_counter() - t0)

    wall_total = time.time() - wall_start

    print(f"\nDone. Converted: {ok} | Skipped (already exist): {skipped} | Failed: {failed} | Total: {len(json_files)}")
    if conversion_times:
        avg_ms = (sum(conversion_times) / len(conversion_times)) * 1000
        min_ms = min(conversion_times) * 1000
        max_ms = max(conversion_times) * 1000
        print(f"Per-file conversion time  — avg: {avg_ms:.1f} ms | min: {min_ms:.1f} ms | max: {max_ms:.1f} ms")
    print(f"Total wall-clock time: {wall_total:.1f} s  ({wall_total/60:.2f} min)")


if __name__ == "__main__":
    main()