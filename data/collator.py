# -*- coding: utf-8 -*-
import torch
import sys
sys.path.append('..')
from models.modules.utils.macro import *


def batch_edge_index(edge_indices, num_nodes_per_graph):
    """Concatenate per-graph COO indices with node offsets (same layout as dgl.batch)."""
    if not edge_indices:
        return torch.zeros(2, 0, dtype=torch.long)
    offset = 0
    parts = []
    for ei, n in zip(edge_indices, num_nodes_per_graph):
        ei = ei.long()
        parts.append(ei + offset)
        offset += int(n)
    return torch.cat(parts, dim=1)


def _item_has_a2(item) -> bool:
    if hasattr(item, "has_a2"):
        return bool(item.has_a2)
    d2 = getattr(item, "d2_distance", None)
    ang = getattr(item, "angle_distance", None)
    return d2 is not None and ang is not None


def _item_has_a1(item) -> bool:
    if hasattr(item, "has_a1"):
        return bool(item.has_a1)
    return getattr(item, "spatial_pos", None) is not None


def _item_has_a3(item) -> bool:
    if hasattr(item, "has_a3"):
        return bool(item.has_a3)
    return getattr(item, "edge_path", None) is not None


def _ensure_attn_bias(item, n_nodes: int) -> torch.Tensor:
    ab = getattr(item, "attn_bias", None)
    if ab is None:
        return torch.zeros(n_nodes + 1, n_nodes + 1, dtype=torch.float32)
    return ab.float() if ab.dtype != torch.float32 else ab


def _require_homogeneous(name: str, flags: list[bool]) -> bool:
    if len(set(flags)) > 1:
        raise ValueError(
            f"collator: mixed {name} in batch ({sum(flags)} true / {len(flags) - sum(flags)} false). "
            "Use separate dataloaders or pt_subdir so each batch is homogeneous."
        )
    return flags[0]


def pad_mask_unsqueeze(x, padlen):  #x[num_nodes]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_ones([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_1d_unsqueeze(x, padlen):  #x[num_nodes]
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_face_unsqueeze(x, padlen):  #x[num_nodes]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):  # x[num_nodes, num_nodes]
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_d2_pos_unsqueeze(x, padlen): # x[num_nodes, num_nodes, 32]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 64], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_ang_pos_unsqueeze(x, padlen): # x[num_nodes, num_nodes, 32]
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, 64], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)
     
def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):  #x[num_nodes, num_nodes, max_dist]
    xlen1, xlen2, xlen3 = x.size() 
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = -1 * x.new_ones([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)


def _maybe_float32_features(node_data, edge_data, face_area):
    if node_data.dtype == torch.float16:
        node_data = node_data.float()
    if edge_data.dtype == torch.float16:
        edge_data = edge_data.float()
    if face_area.dtype == torch.float16:
        face_area = face_area.float()
    return node_data, edge_data, face_area


def collator(items, multi_hop_max_dist, spatial_pos_max):  #items({PYGGraph_1, PYGGraph_1_mian}, {PYGGraph_2, PYGGraph_2_mian}, ..., PYGGraph_batchsize)
    batch_has_a2 = _require_homogeneous("has_a2", [_item_has_a2(x) for x in items])
    batch_has_a1 = _require_homogeneous("has_a1", [_item_has_a1(x) for x in items])
    batch_has_a3 = _require_homogeneous("has_a3", [_item_has_a3(x) for x in items])
    if batch_has_a3 and not batch_has_a1:
        raise ValueError("collator: edge_path (A3) requires spatial_pos (A1) in this encoder.")

    rows = []
    for item in items:
        n = item.node_data.size(0)
        sp = getattr(item, "spatial_pos", None)
        d2 = getattr(item, "d2_distance", None) if batch_has_a2 else None
        ang = getattr(item, "angle_distance", None) if batch_has_a2 else None
        ep = getattr(item, "edge_path", None)
        ep_slice = ep[:, :, :multi_hop_max_dist] if ep is not None else None
        rows.append(
            (
                item.edge_index,
                item.node_data,
                item.face_area,
                item.face_type,
                item.face_loop,
                item.face_adj,
                item.edge_data,
                item.edge_type,
                item.edge_len,
                item.edge_ang,
                item.edge_conv,
                item.node_degree,
                _ensure_attn_bias(item, n),
                sp,
                d2,
                ang,
                ep_slice,
                item.label_feature,
                getattr(item, "data_id", 0),
            )
        )

    (
        edge_indices,
        node_datas,
        face_areas,
        face_types,
        face_loops,
        face_adjs,
        edge_datas,
        edge_types,
        edge_lens,
        edge_angs,
        edge_convs,
        node_degrees,
        attn_biases,
        spatial_poses,
        d2_distances,
        angle_distances,
        edge_paths,
        label_features,
        data_ids
    ) = zip(*rows)

    if batch_has_a1:
        for idx, _ in enumerate(attn_biases):
            sp = spatial_poses[idx]
            if sp is not None:
                attn_biases[idx][1:, 1:][sp >= spatial_pos_max] = float("-inf")
        
    max_node_num = max(i.size(0) for i in node_datas)
    max_edge_num = max(i.size(0) for i in edge_datas)
    if batch_has_a3:
        max_dist = max(i.size(-1) for i in edge_paths if i is not None)
        max_dist = max(max_dist, multi_hop_max_dist)
    else:
        max_dist = multi_hop_max_dist

    padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in node_datas]
    padding_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in padding_mask_list])
    
    edge_padding_mask_list = [torch.zeros([i.size(0)], dtype=torch.bool) for i in edge_datas]
    edge_padding_mask = torch.cat([pad_mask_unsqueeze(i, max_edge_num) for i in edge_padding_mask_list])
    
    node_data = torch.cat([i for i in node_datas])
    node_data, edge_data, face_area = _maybe_float32_features(node_data, torch.cat([i for i in edge_datas]), torch.cat([i for i in face_areas]))

    face_type = torch.cat([i for i in face_types])
    face_loop = torch.cat([i for i in face_loops])
    face_adj = torch.cat([i for i in face_adjs])
    
    edge_type = torch.cat([i for i in edge_types])
    edge_len = torch.cat([i for i in edge_lens])
    edge_ang = torch.cat([i for i in edge_angs])
    edge_conv = torch.cat([i for i in edge_convs])
    
    if batch_has_a3:
        edge_path = torch.cat(
            [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_paths]
        ).long()
    else:
        edge_path = None
    
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
   
    if batch_has_a1:
        # Graphs may store spatial_pos as uint8 on disk; nn.Embedding and masks expect int64.
        spatial_pos = torch.cat(
            [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
        ).long()
    else:
        spatial_pos = None

    if batch_has_a2:
        d2_distance = torch.cat(
            [pad_d2_pos_unsqueeze(i, max_node_num) for i in d2_distances]
        )
        angle_distance = torch.cat(
            [pad_ang_pos_unsqueeze(i, max_node_num) for i in angle_distances]
        )
    else:
        d2_distance = None
        angle_distance = None
    
    in_degree = torch.cat([i for i in node_degrees])

    num_nodes_list = [nd.size(0) for nd in node_datas]
    batched_edge_index = batch_edge_index(edge_indices, num_nodes_list)

    batched_label_feature = torch.cat([i for i in label_features])

    data_ids = torch.tensor([i for i in data_ids])

    batch_data = dict(
        padding_mask = padding_mask,
        edge_padding_mask = edge_padding_mask,
        edge_index=batched_edge_index,

        node_data = node_data,
        face_area = face_area,
        face_type = face_type,
        face_loop = face_loop,
        face_adj = face_adj,

        edge_data = edge_data,
        edge_type = edge_type,
        edge_len = edge_len,
        edge_ang = edge_ang,
        edge_conv = edge_conv,

        in_degree = in_degree,
        out_degree = in_degree,
        attn_bias = attn_bias,
        spatial_pos = spatial_pos,
        d2_distance = d2_distance,
        angle_distance = angle_distance,
        edge_path = edge_path,

        label_feature = batched_label_feature,
        id = data_ids
    )
    return batch_data


def collator_st(items, multi_hop_max_dist, spatial_pos_max):
    """Same packing as ``collator`` over ``[source_0,…,source_{B-1}, target_0,…, target_{B-1}]``."""
    flat = [pair["source_data"] for pair in items] + [pair["target_data"] for pair in items]
    return collator(flat, multi_hop_max_dist, spatial_pos_max)
