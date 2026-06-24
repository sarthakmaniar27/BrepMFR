#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collator + BrepEncoder forward smoke for full / no_a2 / lite graphs (one JSON each)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "test/json/00004121_101.json",
        help="SolidWorks-style BrepMFR JSON",
    )
    args = ap.parse_args()

    from data.collator import collator
    from models.modules.brep_encoder import BrepEncoder
    from scripts.inference import json_to_brepmfr_pyg as j2p

    jp = args.json.resolve()
    if not jp.is_file():
        raise SystemExit(f"Missing JSON: {jp}")

    enc = BrepEncoder(
        num_degree=128,
        num_spatial=64,
        num_edge_dis=64,
        edge_type="multi_hop",
        multi_hop_max_dist=16,
        num_encoder_layers=2,
        embedding_dim=128,
        ffn_embedding_dim=128,
        num_attention_heads=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        encoder_normalize_before=True,
        pre_layernorm=True,
        apply_params_init=False,
        activation_fn="gelu",
    ).eval()

    for profile in ("full", "no_a2", "lite"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        pyg, _ = j2p.tensors_from_brep_json_dict(
            data,
            spatial_pos_max=32,
            inference_profile=profile,  # type: ignore[arg-type]
            max_edge_path_len=16,
            float16_storage=False,
        )
        pyg.data_id = 0
        b = collator([pyg], multi_hop_max_dist=16, spatial_pos_max=32)
        with torch.no_grad():
            _, _ = enc(b, last_state_only=True)
        print(f"OK profile={profile} nodes={pyg.node_data.size(0)} edges={pyg.edge_data.size(0)}")

    print("smoke_inference_profiles_forward: all profiles passed.")


if __name__ == "__main__":
    main()
