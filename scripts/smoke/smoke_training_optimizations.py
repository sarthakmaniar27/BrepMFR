# -*- coding: utf-8 -*-
"""Fast parity checks for the Stage-1 training performance paths."""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.length_bucket_batch_sampler import LengthBucketBatchSampler
from models.brepseg_model import (
    ClassificationLossFromLogits,
    CrossEntropyLoss,
    FocalLoss,
)
from models.modules.layers.multihead_attention import MultiheadAttention


def check_a3_fused_reduction() -> None:
    generator = torch.Generator().manual_seed(7)
    pairs, hops, heads = 23, 5, 4
    edge = torch.randn(pairs, hops, heads, generator=generator)
    weights = torch.randn(hops, heads, heads, generator=generator)
    reference = sum(edge[:, hop] @ weights[hop] for hop in range(hops))
    fused = edge.flatten(start_dim=-2) @ weights.reshape(hops * heads, heads)
    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-6)


def check_logits_losses() -> None:
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(37, 3, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 3, (37,), generator=generator)
    weights = torch.tensor([0.8, 1.1, 1.3], dtype=torch.float64)
    one_hot = F.one_hot(labels, 3)
    probabilities = logits.softmax(dim=-1)

    old_ce = CrossEntropyLoss(one_hot, probabilities, class_level_weight=weights)
    new_ce = ClassificationLossFromLogits(
        labels, logits, loss_type="ce", class_level_weight=weights
    )
    torch.testing.assert_close(new_ce, old_ce, rtol=1e-9, atol=1e-10)

    old_focal = FocalLoss(
        one_hot, probabilities, gamma=2.0, class_level_weight=weights
    )
    new_focal = ClassificationLossFromLogits(
        labels,
        logits,
        loss_type="focal",
        focal_gamma=2.0,
        class_level_weight=weights,
    )
    torch.testing.assert_close(new_focal, old_focal, rtol=1e-9, atol=1e-10)


def check_adaptive_sampler() -> None:
    counts = [40, 45, 50, 100, 110, 300, 320, 500, 769, 900]
    sampler = LengthBucketBatchSampler(
        [f"graph_{count}.pt" for count in counts],
        base_batch_size=8,
        node_counts=counts,
        node_sq_budget=400_000,
        a3_node_cap=768,
        shuffle=False,
    )
    batches = list(iter(sampler))
    assert sorted(index for batch in batches for index in batch) == list(range(len(counts)))
    for batch in batches:
        batch_counts = [counts[index] for index in batch]
        cost = len(batch) * max(batch_counts) ** 2
        assert cost <= 400_000 or len(batch) == 1
        assert all(count <= 768 for count in batch_counts) or all(
            count > 768 for count in batch_counts
        )


def check_sdpa_attention() -> None:
    torch.manual_seed(13)
    module = MultiheadAttention(
        embed_dim=16,
        num_heads=4,
        dropout=0.0,
        self_attention=True,
    ).eval()
    query = torch.randn(7, 3, 16)
    bias = torch.randn(3, 4, 7, 7) * 0.05
    padding = torch.tensor(
        [[False] * 7, [False] * 5 + [True] * 2, [False] * 6 + [True]]
    )
    with torch.no_grad():
        fused, _ = module(
            query, query, query, bias, key_padding_mask=padding, need_weights=False
        )
        reference, _ = module(
            query, query, query, bias, key_padding_mask=padding, need_weights=True
        )
    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-6)


def main() -> None:
    check_a3_fused_reduction()
    check_logits_losses()
    check_adaptive_sampler()
    check_sdpa_attention()
    print("Training optimization parity smoke: OK")


if __name__ == "__main__":
    main()