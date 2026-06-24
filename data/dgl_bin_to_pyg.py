# -*- coding: utf-8 -*-
"""
Load **legacy** DGL-serialized ``dgl.save_graphs`` `.bin` into PyG ``Data``.

**Not used on the migrated training/inference path** (JSON→``.pt`` via ``json_to_brepmfr_pyg`` keeps
everything in PyTorch / PyG). This module remains for **parity checks vs old ``.bin``** and the
optional ``convert_dgl_bins_to_pyg.py`` migrate script; ``dgl`` is imported **only inside**
``bin_to_pyg``.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import FloatTensor
from torch_geometric.data import Data as PYGGraph


def bin_to_pyg(bin_path: Path | str) -> PYGGraph:
    """``dgl.data.utils.save_graphs`` .bin → ``torch_geometric.data.Data`` (BrepMFR field names)."""
    from dgl.data.utils import load_graphs

    bin_path = Path(bin_path)
    graphfile = load_graphs(str(bin_path))
    graph = graphfile[0][0]
    pyg = PYGGraph()

    pyg.node_data = graph.ndata["x"].type(FloatTensor)
    pyg.edge_data = graph.edata["x"].type(FloatTensor)

    pyg.face_type = graph.ndata["z"].type(torch.int)
    pyg.face_area = graph.ndata["y"].type(torch.float)
    pyg.face_loop = graph.ndata["l"].type(torch.int)
    pyg.face_adj = graph.ndata["a"].type(torch.int)
    pyg.label_feature = graph.ndata["f"].type(torch.int)

    pyg.edge_type = graph.edata["t"].type(torch.int)
    pyg.edge_len = graph.edata["l"].type(torch.float)
    pyg.edge_ang = graph.edata["a"].type(torch.float)
    pyg.edge_conv = graph.edata["c"].type(torch.int)

    u, v = graph.edges()
    pyg.edge_index = torch.stack([u.long(), v.long()], dim=0)
    n_nodes = graph.num_nodes()
    # Match JSON→PyG enumeration: ``out-degree == bincount(src)`` (same as one row-sum of dense
    # adjacency for graphs built from SolidWorks topology without duplicate directed arcs).
    pyg.node_degree = torch.bincount(u.long(), minlength=n_nodes)

    pyg.attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float)

    pyg.edge_path = graphfile[1]["edges_path"]
    pyg.spatial_pos = graphfile[1]["spatial_pos"]
    pyg.d2_distance = graphfile[1]["d2_distance"]
    pyg.angle_distance = graphfile[1]["angle_distance"]

    pyg.has_a1 = True
    pyg.has_a2 = True
    pyg.has_a3 = True
    pyg.inference_profile = "full"

    stem = bin_path.stem
    try:
        pyg.data_id = int(stem.split("_")[-1])
    except ValueError:
        pyg.data_id = 0

    return pyg
