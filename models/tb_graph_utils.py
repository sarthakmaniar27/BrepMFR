# -*- coding: utf-8 -*-
"""Helpers for TensorBoard static graphs: bounded collator batches and trace wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from data.collator import collator

# Must match keys and order from ``collator.collator`` / ``collator.collator_st`` return dicts.
TRACE_BATCH_KEYS: Tuple[str, ...] = (
    "padding_mask",
    "edge_padding_mask",
    "edge_index",
    "node_data",
    "face_area",
    "face_type",
    "face_loop",
    "face_adj",
    "edge_data",
    "edge_type",
    "edge_len",
    "edge_ang",
    "edge_conv",
    "in_degree",
    "out_degree",
    "attn_bias",
    "spatial_pos",
    "d2_distance",
    "angle_distance",
    "edge_path",
    "label_feature",
    "id",
)


def batch_to_flat(batch: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
    """TensorBoard ``add_graph`` input: only keys that are tensors (full batches)."""
    missing = [k for k in TRACE_BATCH_KEYS if not isinstance(batch.get(k), torch.Tensor)]
    if missing:
        raise ValueError(
            "batch_to_flat: expected dense Graphormer batch for TB trace; "
            f"missing or non-tensor keys: {missing}. Use full-graph ``.pt`` for add_graph."
        )
    return tuple(batch[k] for k in TRACE_BATCH_KEYS)


def flat_to_batch(flat: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    return {k: flat[i] for i, k in enumerate(TRACE_BATCH_KEYS)}


class EncoderSegTraceWrapper(nn.Module):
    """Same tensor path as ``BrepSeg.validation_step`` (encoder + attention + classifier)."""

    def __init__(
        self,
        encoder: nn.Module,
        attention: nn.Module,
        classifier: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.classifier = classifier

    def forward(self, *flat: torch.Tensor) -> torch.Tensor:
        batch = flat_to_batch(flat)
        node_emb, graph_emb = self.encoder(batch, last_state_only=True)
        node_emb = node_emb[0].permute(1, 0, 2)
        node_emb = node_emb[:, 1:, :]
        padding_mask = batch["padding_mask"]
        node_pos = torch.where(padding_mask == False)  # noqa: E712
        node_z = node_emb[node_pos]
        padding_mask_ = ~padding_mask
        num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1)
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        z = self.attention([node_z, graph_z])
        return self.classifier(z)


class DomainGrlDiscTraceWrapper(nn.Module):
    """domain_adv.grl -> domain_adv.domain_discriminator (fixed concat features)."""

    def __init__(self, domain_adv: nn.Module):
        super().__init__()
        self.grl = domain_adv.grl
        self.discriminator = domain_adv.domain_discriminator

    def forward(self, f_cat: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self.grl(f_cat))


def try_build_trace_batch_from_dataset(
    dataset_root: Path | str,
    pt_subdir: Optional[str],
    *,
    multi_hop_max_dist: int = 16,
    spatial_pos_max: int = 32,
    max_nodes: int = 48,
    max_edges: int = 96,
    max_files_to_scan: int = 120,
    split_file: str = "train.txt",
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load one small graph from the dataset tree and collate with ``collator``.

    Returns ``(batch_dict, "")`` on success, or ``(None, reason)``.
    """
    from data.dataset import (  # noqa: PLC0415 — heavy (torch_geometric); training env only
        _load_pyg_sample,
        _resolve_dataset_split_list,
        _resolve_graph_pt_scan_root,
    )

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        return None, f"dataset_root not a directory: {root}"

    try:
        scan = _resolve_graph_pt_scan_root(root, pt_subdir)
        list_path = _resolve_dataset_split_list(root, split_file)
    except (FileNotFoundError, OSError) as exc:
        return None, f"resolve paths failed: {exc}"

    try:
        stems = {ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()}
    except OSError as exc:
        return None, f"read split list failed: {exc}"

    seen = 0
    for path in scan.rglob("*[0-9].pt"):
        if path.stem not in stems:
            continue
        seen += 1
        if seen > max_files_to_scan:
            break
        try:
            g = _load_pyg_sample(path)
        except Exception as exc:  # noqa: BLE001
            continue
        try:
            n_node = int(g.node_data.size(0))
            n_edge = int(g.edge_data.size(0))
        except Exception:
            continue
        if n_node > max_nodes or n_edge > max_edges:
            continue
        try:
            batch = collator([g], multi_hop_max_dist, spatial_pos_max)
        except Exception as exc:  # noqa: BLE001
            continue
        note = f"from_pt={path.name} nodes={n_node} edges={n_edge}"
        return batch, note

    return (
        None,
        f"no graph within caps (nodes<={max_nodes}, edges<={max_edges}) among first {max_files_to_scan} train-listed files under {scan}",
    )


def summarize_trace_batch(batch: Dict[str, Any]) -> str:
    pm = batch["padding_mask"]
    d2 = batch.get("d2_distance")
    d2_s = "None" if d2 is None else str(tuple(d2.shape))
    return (
        f"padding_mask shape={tuple(pm.shape)} "
        f"node_data shape={tuple(batch['node_data'].shape)} "
        f"edge_data shape={tuple(batch['edge_data'].shape)} "
        f"d2_distance={d2_s}"
    )


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device)
        else:
            out[k] = v
    return out
