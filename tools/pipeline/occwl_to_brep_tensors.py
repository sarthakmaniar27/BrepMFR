# -*- coding: utf-8 -*-
"""
Occwl face-adjacency graph → Torch tensors / PyG Data for BrepMFR **without DGL**.

Training pipeline still uses DGL .bin via stp_to_bin.build_graph + save_to_binary;
inference-only PyG stacks only need this module + PyG.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import FloatTensor

from occwl_pythonocc_patch import apply_pythonocc_occwl_compatibility

FACE_TYPE_MAP = {
    "plane": 0,
    "cylinder": 1,
    "cone": 2,
    "sphere": 3,
    "torus": 4,
    "bezier": 5,
    "bspline": 6,
    "nurbs": 7,
}

EDGE_TYPE_MAP = {
    "line": 0,
    "circle": 1,
    "ellipse": 2,
    "parabola": 3,
    "hyperbola": 4,
    "bezier": 5,
    "bspline": 6,
    "nurbs": 7,
}

CONVEXITY_MAP = {"concave": 0, "convex": 1, "smooth": 2}


def tensor_dict_from_face_adjacency(
    graph,
    curv_num_u_samples: int = 5,
    surf_num_u_samples: int = 5,
    surf_num_v_samples: int = 5,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    graph :
        Output of occwl.graph.face_adjacency(solid).
    """
    from occwl.uvgrid import ugrid, uvgrid

    graph_face_feat = []
    face_types = []
    face_areas = []
    face_loops = []
    face_adjs = []

    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]

        try:
            points = uvgrid(
                face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
            )
            normals = uvgrid(
                face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
            )
            visibility_status = uvgrid(
                face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
            )
            mask = np.logical_or(visibility_status == 0, visibility_status == 2)
            if mask.ndim > 2:
                mask = mask.squeeze()
            if mask.ndim == 2:
                mask = mask.astype(np.float32)[..., np.newaxis]
            else:
                mask = mask.reshape(surf_num_u_samples, surf_num_v_samples, 1).astype(np.float32)
            if points.ndim != 3:
                points = points.reshape(surf_num_u_samples, surf_num_v_samples, -1)
            if normals.ndim != 3:
                normals = normals.reshape(surf_num_u_samples, surf_num_v_samples, -1)
            face_feat = np.concatenate((points, normals, mask), axis=-1)
        except Exception as e:
            print(f"Error processing face {face_idx}: {str(e)}")
            face_feat = np.zeros((surf_num_u_samples, surf_num_v_samples, 7), dtype=np.float32)

        graph_face_feat.append(face_feat)

        try:
            surface_type = str(face.surface_type()).lower()
            face_type = FACE_TYPE_MAP.get(surface_type, 0)
        except Exception:
            face_type = 0
        face_types.append(face_type)

        try:
            face_areas.append(face.area())
        except Exception:
            face_areas.append(0.0)

        try:
            face_loops.append(face.number_of_loops())
        except Exception:
            face_loops.append(1)

        adj_count = 0
        for edge in graph.edges:
            if edge[0] == face_idx or edge[1] == face_idx:
                adj_count += 1
        face_adjs.append(adj_count)

    graph_face_feat = np.asarray(graph_face_feat)
    face_types = np.array(face_types)
    face_areas = np.array(face_areas, dtype=np.float32)
    face_loops = np.array(face_loops)
    face_adjs = np.array(face_adjs)

    graph_edge_feat = []
    edge_types = []
    edge_lengths = []
    edge_angles = []
    edge_convs = []

    for edge_idx in graph.edges:
        edge = graph.edges[edge_idx]["edge"]
        if not edge.has_curve():
            edge_types.append(0)
            edge_lengths.append(0.0)
            edge_angles.append(0.0)
            edge_convs.append(0)
            continue

        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        edge_feat_6ch = np.concatenate((points, tangents), axis=-1)
        one_channel = np.full((edge_feat_6ch.shape[0], 1), 1.5707963705062866, dtype=np.float32)
        edge_feat = np.concatenate((edge_feat_6ch, one_channel), axis=-1)
        graph_edge_feat.append(edge_feat)

        try:
            curve_type = str(edge.curve_type()).lower()
            edge_type = EDGE_TYPE_MAP.get(curve_type, 0)
        except Exception:
            edge_type = 0
        edge_types.append(edge_type)

        try:
            start_point = edge.start_point()
            end_point = edge.end_point()
            length = np.linalg.norm(np.array(end_point) - np.array(start_point))
            edge_lengths.append(length)
        except Exception:
            edge_lengths.append(0.0)

        try:
            if curv_num_u_samples >= 2:
                start_tangent = tangents[0]
                end_tangent = tangents[-1]
                start_tangent_norm = start_tangent / (np.linalg.norm(start_tangent) + 1e-10)
                end_tangent_norm = end_tangent / (np.linalg.norm(end_tangent) + 1e-10)
                angle = np.arccos(np.clip(np.dot(start_tangent_norm, end_tangent_norm), -1.0, 1.0))
                edge_angles.append(angle)
            else:
                edge_angles.append(0.0)
        except Exception:
            edge_angles.append(0.0)

        try:
            edge_convs.append(CONVEXITY_MAP["convex"])
        except Exception:
            edge_convs.append(1)

    edge_types = np.array(edge_types, dtype=np.int32)
    edge_lengths = np.array(edge_lengths, dtype=np.float32)
    edge_angles = np.array(edge_angles, dtype=np.float32)
    edge_convs = np.array(edge_convs, dtype=np.int32)

    if len(graph_edge_feat) == 0:
        graph_edge_feat = np.array([], dtype=np.float32)
    else:
        graph_edge_feat = np.asarray(graph_edge_feat)

    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    num_nodes = len(graph.nodes)

    edge_data_tensor: Optional[torch.Tensor]
    if len(graph_edge_feat) > 0:
        num_edges = len(graph_edge_feat)
        num_samples = curv_num_u_samples
        edge_data_tensor = torch.zeros((num_edges, num_samples, 7), dtype=torch.float)
        for i, feat in enumerate(graph_edge_feat):
            if feat.shape[-1] < 7:
                padding = np.zeros((feat.shape[0], 7 - feat.shape[-1]), dtype=np.float32)
                feat = np.concatenate((feat, padding), axis=-1)
            edge_data_tensor[i] = torch.from_numpy(feat)
    else:
        edge_data_tensor = None

    max_dist = 16
    edges_path = np.zeros((num_nodes, num_nodes, max_dist), dtype=np.int32)
    for i, edge in enumerate(edges):
        u, v = edge
        edges_path[u, v, 0] = i + 1
        edges_path[v, u, 0] = i + 1
    for i in range(num_nodes):
        edges_path[i, i, 0] = 0

    centroids = []
    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        try:
            centroids.append(face.mid_point())
        except Exception:
            centroids.append([0.0, 0.0, 0.0])

    spatial_pos = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            distance = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
            spatial_pos[i, j] = int(distance * 1000)

    ei = torch.tensor([src, dst], dtype=torch.long)

    return {
        "num_nodes": num_nodes,
        "node_data": torch.from_numpy(graph_face_feat).float(),
        "face_type": torch.from_numpy(face_types).int(),
        "face_area": torch.from_numpy(face_areas).float(),
        "face_loop": torch.from_numpy(face_loops).int(),
        "face_adj": torch.from_numpy(face_adjs).int(),
        "label_feature": torch.zeros(num_nodes, dtype=torch.int),
        "edge_data": edge_data_tensor,
        "edge_type": torch.from_numpy(edge_types).int(),
        "edge_len": torch.from_numpy(edge_lengths).float(),
        "edge_ang": torch.from_numpy(edge_angles).float(),
        "edge_conv": torch.from_numpy(edge_convs).int(),
        "edge_index": ei,
        "edges_path": torch.from_numpy(edges_path).int(),
        "spatial_pos": torch.from_numpy(spatial_pos).int(),
        "d2_distance": torch.zeros(num_nodes, num_nodes, 64, dtype=torch.float),
        "angle_distance": torch.zeros(num_nodes, num_nodes, 64, dtype=torch.float),
    }


def solid_to_pyg_data(
    solid,
    curv_num_u_samples: int = 5,
    surf_num_u_samples: int = 5,
    surf_num_v_samples: int = 5,
    data_id: int = 0,
):
    """``solid`` from STEP (see ``convert_stp_path_to_pyg``) → PyG ``Data`` (BrepMFR layout)."""
    apply_pythonocc_occwl_compatibility()
    from occwl.graph import face_adjacency
    from torch_geometric.data import Data as PYGGraph

    adj = face_adjacency(solid)
    t = tensor_dict_from_face_adjacency(
        adj, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples
    )
    n = int(t["num_nodes"])
    pyg = PYGGraph()
    pyg.node_data = t["node_data"].type(FloatTensor)
    pyg.face_type = t["face_type"].type(torch.int)
    pyg.face_area = t["face_area"].type(torch.float)
    pyg.face_loop = t["face_loop"].type(torch.int)
    pyg.face_adj = t["face_adj"].type(torch.int)
    pyg.label_feature = t["label_feature"].type(torch.int)
    pyg.edge_type = t["edge_type"].type(torch.int)
    pyg.edge_len = t["edge_len"].type(torch.float)
    pyg.edge_ang = t["edge_ang"].type(torch.float)
    pyg.edge_conv = t["edge_conv"].type(torch.int)
    if t["edge_data"] is not None:
        pyg.edge_data = t["edge_data"].type(FloatTensor)
    else:
        pyg.edge_data = torch.zeros(0, curv_num_u_samples, 7, dtype=torch.float)
    pyg.edge_index = t["edge_index"].long()
    pyg.attn_bias = torch.zeros([n + 1, n + 1], dtype=torch.float)
    pyg.edge_path = t["edges_path"]
    pyg.spatial_pos = t["spatial_pos"].type(torch.int)
    pyg.d2_distance = t["d2_distance"].type(torch.float)
    pyg.angle_distance = t["angle_distance"].type(torch.float)
    row, col = pyg.edge_index
    deg = torch.zeros(n, dtype=torch.long)
    if row.numel() > 0:
        deg.index_add_(0, row, torch.ones_like(row))
        deg.index_add_(0, col, torch.ones_like(col))
    pyg.node_degree = deg
    pyg.data_id = int(data_id)
    return pyg


def convert_stp_path_to_pyg(
    stp_file_path,
    curv_u_samples: int = 5,
    surf_u_samples: int = 5,
    surf_v_samples: int = 5,
):
    """Load STEP and return a single PyG ``Data`` (first solid), or ``None`` on failure."""
    apply_pythonocc_occwl_compatibility()
    from occwl.compound import Compound

    path = pathlib.Path(stp_file_path)
    solids = list(Compound.load_from_step(str(path)).solids())
    if not solids or len(solids) == 0:
        return None
    stem = path.stem
    try:
        data_id = int(stem.split("_")[-1])
    except ValueError:
        data_id = 0
    return solid_to_pyg_data(
        solids[0],
        curv_u_samples,
        surf_u_samples,
        surf_v_samples,
        data_id=data_id,
    )
