"""
Subgraph training support for BrepMFR face classification.

Instead of feeding entire (often huge) B-Rep face graphs, we sample a small number
of "seed" faces (biased toward rare classes like thread), extract their k-hop
neighborhoods, and train on the induced union subgraph.

This dramatically reduces domination by massive text regions while preserving
local geometric + topological context that the BrepEncoder + Attention classifier
needs.

Key properties:
- Pure node-level classification (no global graph pooling for the label), so
  subgraphs are semantically valid training examples.
- All node/edge/dense A1/A2/A3 tensors are sliced consistently.
- Default behavior of the rest of the system is unchanged (opt-in only).

Typical usage (inside CADSynth or a wrapper):
    from .subgraph_sampler import sample_balanced_subgraph
    sub = sample_balanced_subgraph(full_pyg, k_hop=2, seeds_per_class=(2, 3, 3))
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch_geometric.data import Data as PYGGraph
from torch_geometric.utils import k_hop_subgraph


# --------------------------------------------------------------------------------------
# Low-level extraction
# --------------------------------------------------------------------------------------


def _get_has_flags(item: PYGGraph) -> Tuple[bool, bool, bool]:
    """Return (has_a1, has_a2, has_a3) using the same logic as the collator."""
    has_a1 = bool(getattr(item, "has_a1", getattr(item, "spatial_pos", None) is not None))
    has_a2 = bool(
        getattr(item, "has_a2", None)
        or (getattr(item, "d2_distance", None) is not None and getattr(item, "angle_distance", None) is not None)
    )
    has_a3 = bool(getattr(item, "has_a3", getattr(item, "edge_path", None) is not None))
    return has_a1, has_a2, has_a3


def _slice_node_tensor(t: Optional[torch.Tensor], idx: torch.Tensor) -> Optional[torch.Tensor]:
    if t is None:
        return None
    # Works for 1D [N] and 2D+ node features [N, ...]
    return t[idx]


def _slice_edge_tensor(t: Optional[torch.Tensor], edge_mask: torch.Tensor) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t[edge_mask]


def _slice_dense_square(t: Optional[torch.Tensor], idx: torch.Tensor) -> Optional[torch.Tensor]:
    """Slice an [N, N, ...] tensor (spatial_pos, d2_distance, angle_distance, edge_path, ...)."""
    if t is None:
        return None
    # idx shape [n_sub]
    return t[idx][:, idx]


def _slice_edge_path(t: Optional[torch.Tensor], idx: torch.Tensor) -> Optional[torch.Tensor]:
    """edge_path is [N, N, D]."""
    if t is None:
        return None
    return t[idx][:, idx, :]


def _make_fresh_attn_bias(n_nodes: int) -> torch.Tensor:
    """The stored attn_bias is typically a zero (N+1,N+1) placeholder; the real bias
    is injected inside GraphAttnBias from spatial/edge features. We create a correctly
    sized zero matrix for the subgraph.
    """
    return torch.zeros((n_nodes + 1, n_nodes + 1), dtype=torch.float32)


def _recompute_induced_degrees(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Compute out-degree in the *induced* subgraph (matches how node_degree is built originally)."""
    if n_nodes == 0:
        return torch.zeros(0, dtype=torch.long)
    src = edge_index[0]
    return torch.bincount(src, minlength=n_nodes)


def extract_k_hop_subgraph(
    pyg: PYGGraph,
    seed_nodes: Union[int, Sequence[int], torch.Tensor],
    k_hop: int = 2,
    *,
    relabel_nodes: bool = True,
    recompute_degree: bool = True,
) -> PYGGraph:
    """Extract the union of k-hop neighborhoods around the given seed node(s).

    All node-level, edge-level, and dense pairwise tensors (A1/A2/A3) are sliced
    to the selected node subset. A fresh zero attn_bias of the right size is created.

    Args:
        pyg: A full (or already partial) PyG Data with BrepMFR attributes.
        seed_nodes: Node index or list/tensor of seed indices (in original node space).
        k_hop: Number of hops (1 = immediate neighbors, 2 is a good default for local context).
        relabel_nodes: Passed to k_hop_subgraph (almost always True).
        recompute_degree: If True, node_degree becomes induced-subgraph degrees.
                          If False, we keep the original degrees of the selected nodes
                          (carries some global-size signal but is less "honest" locally).

    Returns:
        A new PyG Data representing the (usually much smaller) subgraph. It retains
        has_a1/has_a2/has_a3, inference_profile, data_id, and float16 hints from the parent.
    """
    if pyg.edge_index is None or pyg.edge_index.numel() == 0:
        # Degenerate graph (shouldn't happen after filtering); return as-is or tiny copy.
        return _shallow_copy_with_flags(pyg)

    edge_index = pyg.edge_index.long()
    num_nodes = int(pyg.node_data.size(0))

    # Normalize seeds to tensor on CPU for k_hop_subgraph
    if isinstance(seed_nodes, int):
        seed_nodes_t = torch.tensor([seed_nodes], dtype=torch.long)
    else:
        seed_nodes_t = torch.as_tensor(seed_nodes, dtype=torch.long).view(-1)

    # Remove any seeds that are out of range (defensive)
    seed_nodes_t = seed_nodes_t[(seed_nodes_t >= 0) & (seed_nodes_t < num_nodes)]
    if seed_nodes_t.numel() == 0:
        # Fallback: pick node 0 so we never return an empty object from here
        seed_nodes_t = torch.tensor([0], dtype=torch.long)

    # k_hop_subgraph returns (subset, edge_index, inv_map, edge_mask)
    subset, sub_edge_index, _, edge_mask = k_hop_subgraph(
        node_idx=seed_nodes_t,
        num_hops=int(k_hop),
        edge_index=edge_index,
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes,
        flow="source_to_target",
    )

    # subset is already sorted unique nodes in the union neighborhood
    subset = subset.long()
    n_sub = int(subset.size(0))

    # Build the new object
    sub = PYGGraph()

    # --- Node tensors ---
    sub.node_data = _slice_node_tensor(pyg.node_data, subset)
    sub.face_area = _slice_node_tensor(getattr(pyg, "face_area", None), subset)
    sub.face_type = _slice_node_tensor(getattr(pyg, "face_type", None), subset)
    sub.face_loop = _slice_node_tensor(getattr(pyg, "face_loop", None), subset)
    sub.face_adj = _slice_node_tensor(getattr(pyg, "face_adj", None), subset)
    sub.label_feature = _slice_node_tensor(pyg.label_feature, subset)
    sub.node_degree = (
        _recompute_induced_degrees(sub_edge_index, n_sub)
        if recompute_degree
        else _slice_node_tensor(getattr(pyg, "node_degree", None), subset)
    )

    # --- Edge tensors ---
    sub.edge_index = sub_edge_index  # already relabeled 0..n_sub-1
    sub.edge_data = _slice_edge_tensor(getattr(pyg, "edge_data", None), edge_mask)
    sub.edge_type = _slice_edge_tensor(getattr(pyg, "edge_type", None), edge_mask)
    sub.edge_len = _slice_edge_tensor(getattr(pyg, "edge_len", None), edge_mask)
    sub.edge_ang = _slice_edge_tensor(getattr(pyg, "edge_ang", None), edge_mask)
    sub.edge_conv = _slice_edge_tensor(getattr(pyg, "edge_conv", None), edge_mask)

    # --- Dense pairwise (A1 / A2 / A3) ---
    has_a1, has_a2, has_a3 = _get_has_flags(pyg)

    if has_a1:
        sp = getattr(pyg, "spatial_pos", None)
        sub.spatial_pos = _slice_dense_square(sp, subset) if sp is not None else None
    else:
        sub.spatial_pos = None

    if has_a2:
        d2 = getattr(pyg, "d2_distance", None)
        ang = getattr(pyg, "angle_distance", None)
        sub.d2_distance = _slice_dense_square(d2, subset) if d2 is not None else None
        sub.angle_distance = _slice_dense_square(ang, subset) if ang is not None else None
    else:
        sub.d2_distance = None
        sub.angle_distance = None

    if has_a3:
        ep = getattr(pyg, "edge_path", None)
        sub.edge_path = _slice_edge_path(ep, subset) if ep is not None else None
    else:
        sub.edge_path = None

    # Fresh attn_bias sized for the subgraph (the encoder layer will fill meaningful values)
    sub.attn_bias = _make_fresh_attn_bias(n_sub)

    # --- Bookkeeping flags (critical for collator homogeneity checks) ---
    sub.has_a1 = has_a1
    sub.has_a2 = has_a2
    sub.has_a3 = has_a3
    sub.inference_profile = getattr(pyg, "inference_profile", "subgraph")
    # Preserve data_id for traceability (logging, media, etc.)
    if hasattr(pyg, "data_id"):
        sub.data_id = getattr(pyg, "data_id")

    # Float16 storage hint (if present on parent, we keep tensors as-is; collator will cast)
    if hasattr(pyg, "store_float16"):
        sub.store_float16 = getattr(pyg, "store_float16")

    # Optional original node ids inside this subgraph (useful for debugging / visualization)
    # Store as a tensor so it survives collation.
    sub._subgraph_orig_nodes = subset  # private; not used by core pipeline

    return sub


def _shallow_copy_with_flags(pyg: PYGGraph) -> PYGGraph:
    """Minimal copy used only for degenerate empty graphs."""
    sub = PYGGraph()
    for k in ("node_data", "edge_data", "label_feature", "edge_index"):
        if hasattr(pyg, k):
            setattr(sub, k, getattr(pyg, k))
    has_a1, has_a2, has_a3 = _get_has_flags(pyg)
    sub.has_a1 = has_a1
    sub.has_a2 = has_a2
    sub.has_a3 = has_a3
    sub.attn_bias = _make_fresh_attn_bias(int(getattr(pyg, "node_data", torch.empty(0)).size(0)))
    if hasattr(pyg, "data_id"):
        sub.data_id = pyg.data_id
    if hasattr(pyg, "inference_profile"):
        sub.inference_profile = pyg.inference_profile
    return sub


# --------------------------------------------------------------------------------------
# Seed selection
# --------------------------------------------------------------------------------------


def group_nodes_by_label(label_feature: torch.Tensor, num_classes: Optional[int] = None) -> Dict[int, torch.Tensor]:
    """Return {class_label: tensor_of_node_indices}."""
    labels = label_feature.view(-1).long()
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
    groups: Dict[int, torch.Tensor] = {}
    for c in range(num_classes):
        pos = torch.nonzero(labels == c, as_tuple=False).view(-1)
        if pos.numel() > 0:
            groups[c] = pos
    # Also keep any out-of-range labels if they exist (shouldn't after filtering)
    uniq = torch.unique(labels)
    for c in uniq.tolist():
        if c not in groups:
            pos = torch.nonzero(labels == c, as_tuple=False).view(-1)
            if pos.numel() > 0:
                groups[c] = pos
    return groups


def sample_seeds_from_groups(
    groups: Dict[int, torch.Tensor],
    seeds_per_class: Sequence[int],
    *,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Sample up to seeds_per_class[c] nodes for each class c that is present.

    Returns a 1D tensor of unique seed node indices (original numbering).
    Order is arbitrary (we shuffle for variety).
    """
    if rng is None:
        rng = random.Random()

    seeds: List[int] = []
    for c, budget in enumerate(seeds_per_class):
        if budget <= 0:
            continue
        nodes = groups.get(c)
        if nodes is None or nodes.numel() == 0:
            continue
        nodes_list = nodes.tolist()
        rng.shuffle(nodes_list)
        take = min(budget, len(nodes_list))
        seeds.extend(nodes_list[:take])

    if not seeds:
        # Fallback: pick up to 4 arbitrary nodes so training doesn't see empty items
        # Prefer any nodes at all.
        for nodes in groups.values():
            nodes_list = nodes.tolist()
            rng.shuffle(nodes_list)
            seeds.extend(nodes_list[:4])
            break

    # unique + tensor
    uniq = sorted(set(seeds))
    return torch.tensor(uniq, dtype=torch.long)


def parse_seeds_per_class(spec: Union[str, Sequence[int], None], num_classes: int) -> List[int]:
    """Parse a CLI-friendly spec into a list of length num_classes.

    Accepted forms:
      - None or ""            -> all zeros (caller decides a default)
      - "3,3,2"               -> positional for classes 0,1,2 (rest 0)
      - "0:2,1:3,2:3"         -> explicit class:budget pairs
      - [2, 3, 3]             -> python sequence
    """
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return [0] * num_classes

    if isinstance(spec, (list, tuple)):
        out = list(spec) + [0] * max(0, num_classes - len(spec))
        return out[:num_classes]

    s = spec.strip()
    if "," in s and ":" not in s:
        # "3,3,2"
        parts = [int(x) for x in s.split(",")]
        out = parts + [0] * max(0, num_classes - len(parts))
        return out[:num_classes]

    # "0:2,1:3,2:3" or mixed
    out = [0] * num_classes
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            c_str, b_str = token.split(":", 1)
            c = int(c_str)
            b = int(b_str)
            if 0 <= c < num_classes:
                out[c] = b
        else:
            # bare number? treat as budget for class 0 (defensive)
            out[0] = int(token)
    return out


# --------------------------------------------------------------------------------------
# High-level helper used by dataset
# --------------------------------------------------------------------------------------


def sample_balanced_subgraph(
    pyg: PYGGraph,
    *,
    k_hop: int = 2,
    seeds_per_class: Sequence[int] = (2, 3, 3),
    num_classes: Optional[int] = None,
    rng: Optional[random.Random] = None,
    recompute_degree: bool = True,
) -> PYGGraph:
    """Convenience: pick balanced seeds by class then extract their k-hop union subgraph.

    This is the function you typically call from a Dataset.__getitem__ when subgraph
    training is enabled.
    """
    labels = getattr(pyg, "label_feature", None)
    if labels is None or labels.numel() == 0:
        return pyg  # nothing to do

    if num_classes is None:
        num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 3

    groups = group_nodes_by_label(labels, num_classes=num_classes)
    seeds = sample_seeds_from_groups(groups, seeds_per_class, rng=rng)
    if seeds.numel() == 0:
        return pyg

    return extract_k_hop_subgraph(
        pyg,
        seed_nodes=seeds,
        k_hop=k_hop,
        relabel_nodes=True,
        recompute_degree=recompute_degree,
    )


# --------------------------------------------------------------------------------------
# Small deterministic-per-item RNG helper (good variety across epochs without full shuffle)
# --------------------------------------------------------------------------------------


def make_rng_for_index(global_seed: int, epoch: int, index: int, split: str = "train") -> random.Random:
    """Create a reproducible RNG for a specific (epoch, index) so runs are deterministic
    when desired, while still providing different subgraphs across epochs.
    """
    # Mix a few ints into a stable seed. Not cryptographically strong; good enough.
    h = (global_seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    h ^= (epoch * 2654435761) & 0xFFFFFFFFFFFFFFFF
    h ^= (index * 2654435761) & 0xFFFFFFFFFFFFFFFF
    h ^= hash(split) & 0xFFFFFFFFFFFFFFFF
    return random.Random(h & 0x7FFFFFFF)