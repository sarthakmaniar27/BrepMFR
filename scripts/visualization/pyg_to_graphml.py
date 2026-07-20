#!/usr/bin/env python3
"""Export BrepMFR PyTorch Geometric ``.pt`` graphs as GraphML files.

The exporter preserves graph topology and scalar face/edge attributes useful in
graph visualization tools. High-dimensional UV tensors (``node_data`` and
``edge_data``) are intentionally omitted because GraphML attributes must be
scalar values.

Example:
    conda run -n brep_mfr_pyg python scripts/visualization/pyg_to_graphml.py \
      --pyg_dir Z:/Demo/grab_cad_brepmfr_testing/pyg \
      --graphml_dir Z:/Demo/grab_cad_brepmfr_testing/graphml_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from tqdm import tqdm


NODE_ATTRIBUTES = (
    "face_type",
    "face_area",
    "face_loop",
    "face_adj",
    "label_feature",
    "node_degree",
)

EDGE_ATTRIBUTES = (
    "edge_type",
    "edge_len",
    "edge_ang",
    "edge_conv",
)


def _python_scalar(value: Any) -> bool | int | float | str:
    """Convert a tensor/numpy scalar into a GraphML-compatible Python value."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _indexed_scalar(graph: Any, name: str, index: int) -> bool | int | float | str | None:
    value = getattr(graph, name, None)
    if value is None or not hasattr(value, "__len__") or index >= len(value):
        return None
    item = value[index]
    if torch.is_tensor(item) and item.numel() != 1:
        return None
    return _python_scalar(item)


def pyg_to_networkx(graph: Any, source_name: str) -> nx.MultiDiGraph:
    """Convert one BrepMFR PyG object to a directed NetworkX multigraph."""
    if not hasattr(graph, "edge_index") or not hasattr(graph, "node_data"):
        raise ValueError("expected a BrepMFR PyG graph with edge_index and node_data")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(graph.edge_index.shape)}")

    num_nodes = int(graph.node_data.shape[0])
    num_edges = int(graph.edge_index.shape[1])
    result = nx.MultiDiGraph()
    result.graph.update(
        {
            "source_file": source_name,
            "num_faces": num_nodes,
            "num_directed_edges": num_edges,
            "inference_profile": str(getattr(graph, "inference_profile", "unknown")),
            "data_id": _python_scalar(getattr(graph, "data_id", "")),
        }
    )

    for node_index in range(num_nodes):
        attrs: dict[str, bool | int | float | str] = {"face_index": node_index}
        for name in NODE_ATTRIBUTES:
            value = _indexed_scalar(graph, name, node_index)
            if value is not None:
                attrs[name] = value
        result.add_node(node_index, **attrs)

    edge_index = graph.edge_index.detach().cpu()
    for edge_index_position in range(num_edges):
        source = int(edge_index[0, edge_index_position])
        target = int(edge_index[1, edge_index_position])
        attrs = {"edge_index": edge_index_position}
        for name in EDGE_ATTRIBUTES:
            value = _indexed_scalar(graph, name, edge_index_position)
            if value is not None:
                attrs[name] = value
        result.add_edge(source, target, key=edge_index_position, **attrs)

    return result


def export_graph(path: Path, output_path: Path) -> None:
    graph = torch.load(path, map_location="cpu", weights_only=False)
    networkx_graph = pyg_to_networkx(graph, path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(networkx_graph, output_path, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export BrepMFR PyG .pt graphs to directed GraphML files."
    )
    parser.add_argument("--pyg_dir", type=Path, required=True, help="Directory containing .pt files.")
    parser.add_argument(
        "--graphml_dir",
        type=Path,
        required=True,
        help="Directory in which to write one .graphml file per .pt file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing GraphML files instead of skipping them.",
    )
    args = parser.parse_args()

    pyg_dir = args.pyg_dir.expanduser().resolve()
    graphml_dir = args.graphml_dir.expanduser().resolve()
    if not pyg_dir.is_dir():
        raise SystemExit(f"PyG directory does not exist: {pyg_dir}")

    files = sorted(pyg_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files found in: {pyg_dir}")

    converted = 0
    skipped = 0
    failed = 0
    for path in tqdm(files, desc="Exporting GraphML", unit="graph"):
        output_path = graphml_dir / f"{path.stem}.graphml"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            export_graph(path, output_path)
            converted += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path.name}: {exc}")

    print(
        f"Done. Converted: {converted} | Skipped: {skipped} | "
        f"Failed: {failed} | Total: {len(files)}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
