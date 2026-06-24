#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Offline face-adjacency graph visualization for PyG ``.pt`` samples (CadSynth / MFCAD++).

Exports under ``--out_dir``:

- ``html/*.html`` — PyVis (optional subgraph when ``--max_nodes`` is set)
- ``images/*.png`` — matplotlib layout (**default**), optional ``.svg``; plus optional ``legend_classes.png``
- ``graphml/*.graphml`` — **full** directed face adjacency + attributes for Graphia / Gephi

All ``*.pt`` under ``--bin_dir`` or ``--dataset_root`` (+ ``--pt_subdir``) are collected — **no**
train/val/test split lists.

Examples::

  conda activate brep_mfr_pyg
  python scripts/visualization/visualize_pyg_face_graph.py ^
    --dataset_root Z:/Experiment6_PyG/source_dataset ^
    --pt_subdir output/bin ^
    --out_dir Z:/graph_visualization/cadsynth ^
    --write_legend ^
    --check_label_json Z:/Experiment6_PyG/source_dataset/output/label

  Faster batch (skip matplotlib PNG for each graph): add ``--no-png``.

Windows OpenMP: this script sets ``KMP_DUPLICATE_LIB_OK=TRUE`` by default (see ``os.environ.setdefault``).
Status messages and tqdm use **stderr** with flush; if your terminal still looks blank, try ``python -u ...``.
Pairwise tensors (``edge_path``, full ``attn_bias`` blocks) are not expanded per node in GraphML;
per-node row summaries and UV grid channel means are included instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

from cadsynth_face_labels import (
    FACE_LABEL_NAME,
    LABEL_HEX_COLORS,
    label_color,
    label_name,
)

LEGEND_FILENAME = "legend_classes.png"


def _log(msg: str) -> None:
    """stderr + flush so terminals show output during long discovery on network drives."""
    print(msg, file=sys.stderr, flush=True)


FACE_TYPE_NAME = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 6: "other"}
EDGE_TYPE_NAME = {0: "line", 1: "circle", 2: "ellipse", 5: "bspline_other"}
EDGE_CONV_NAME = {0: "smooth", 1: "convex", 2: "concave"}

_UV_FACE_KEYS = ("x", "y", "z", "nx", "ny", "nz", "mask")
_UV_EDGE_KEYS = ("x", "y", "z", "tx", "ty", "tz", "angle")


def _resolve_graph_pt_scan_root(root_dir: Path, pt_subdir: Optional[str]) -> Path:
    root_dir = Path(root_dir).resolve()
    if not pt_subdir:
        return root_dir
    sub = Path(pt_subdir)
    scan = root_dir / sub if not sub.is_absolute() else sub
    if not scan.is_dir():
        raise FileNotFoundError(
            f"--pt_subdir resolved to missing directory: {scan}\n"
            f"dataset root was: {root_dir}"
        )
    return scan


def _py_val(x: Any) -> Any:
    """GraphML-safe scalar or short string."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            x = x.item()
        else:
            return json.dumps(x.numpy().tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if not np.isfinite(v):
            return 0.0
        return v
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, str):
        return x[:5000]
    return str(x)[:5000]


def load_pyg(pt_path: Path) -> object:
    obj = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not hasattr(obj, "edge_index"):
        raise ValueError(f"{pt_path}: loaded object missing edge_index")
    if not hasattr(obj, "label_feature"):
        raise ValueError(f"{pt_path}: loaded object missing label_feature")
    return obj


def validate_pyg(obj: object, pt_path: Path) -> torch.Tensor:
    ei = obj.edge_index
    lf = obj.label_feature.view(-1).long()
    if ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError(f"{pt_path}: edge_index must be [2, E], got {tuple(ei.shape)}")
    n = int(lf.numel())
    if n == 0:
        return lf
    if ei.numel() > 0:
        mx = int(ei.max().item())
        if mx >= n:
            raise ValueError(f"{pt_path}: edge_index max node {mx} >= num_nodes {n}")
        mn = int(ei.min().item())
        if mn < 0:
            raise ValueError(f"{pt_path}: negative edge_index ({mn})")
    return lf


def maybe_check_label_json(
    pt_path: Path,
    label_dir: Path,
    lf: torch.Tensor,
    warnings_out: List[str],
) -> None:
    jp = label_dir / f"{pt_path.stem}.json"
    if not jp.is_file():
        return
    payload = json.loads(jp.read_text(encoding="utf-8"))
    arr = np.asarray(payload.get("labels"), dtype=np.int64)
    v = lf.detach().cpu().numpy().astype(np.int64)
    if arr.shape != v.shape:
        warnings_out.append(
            f"[label] shape mismatch {pt_path.name}: JSON {arr.shape} vs pt {v.shape}"
        )
        return
    mism = int(np.sum(arr != v))
    if mism:
        warnings_out.append(
            f"[label] {mism}/{arr.size} labels differ vs JSON: {pt_path.name}"
        )


def subsample_keep(num_nodes: int, max_nodes: Optional[int], seed: int) -> Set[int]:
    if max_nodes is None or num_nodes <= max_nodes:
        return set(range(num_nodes))
    import random

    rng = random.Random(seed)
    return set(rng.sample(range(num_nodes), max_nodes))


def build_undirected_simple_graph(ei: torch.Tensor, keep: Set[int]) -> object:
    import networkx as nx

    pairs = set()
    ecols = int(ei.shape[1])
    for k in range(ecols):
        u = int(ei[0, k])
        v = int(ei[1, k])
        if u == v or u not in keep or v not in keep:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs.add((a, b))
    g = nx.Graph()
    g.add_nodes_from(sorted(keep))
    g.add_edges_from(sorted(pairs))
    return g


def node_tooltip(data: object, face_idx: int, lf: torch.Tensor) -> str:
    lid = int(lf[face_idx].item())
    lines = [
        f"face_idx={face_idx}",
        f"class_id={lid}",
        f"class_name={label_name(lid)}",
    ]
    if hasattr(data, "face_area"):
        try:
            lines.append(f"area={float(data.face_area[face_idx]):.6g}")
        except Exception:
            pass
    if hasattr(data, "face_type"):
        try:
            lines.append(f"face_type={int(data.face_type[face_idx])}")
        except Exception:
            pass
    return "<br>".join(lines)


def export_pyvis_html(
    g: object,
    data: object,
    lf: torch.Tensor,
    out_html: Path,
    physics: bool,
) -> None:
    from pyvis.network import Network

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=False,
    )
    net.toggle_physics(physics)
    for n in g.nodes:
        ni = int(n)
        lid = int(lf[ni].item())
        net.add_node(
            ni,
            label=str(ni),
            title=node_tooltip(data, ni, lf),
            color=label_color(lid),
        )
    for u, v in g.edges:
        net.add_edge(int(u), int(v), color="#BBBBBB", width=1)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_html))


def export_matplotlib_figure(
    g: object,
    lf: torch.Tensor,
    out_path: Path,
    seed: int,
    fmt: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.spring_layout(g, seed=seed, iterations=50)
    nodes = list(g.nodes())
    colors = [label_color(int(lf[int(n)].item())) for n in nodes]

    plt.figure(figsize=(11, 11), dpi=120)
    nx.draw_networkx_edges(g, pos, width=0.6, edge_color="#CCCCCC", alpha=0.85)
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=nodes,
        node_color=colors,
        node_size=55,
        linewidths=0.4,
        edgecolors="#222222",
    )
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), format=fmt, bbox_inches="tight", facecolor="white")
    plt.close()


def write_class_legend_png(out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(9, 14))
    ax.axis("off")
    handles = [
        Patch(
            facecolor=LABEL_HEX_COLORS[i],
            edgecolor="#333333",
            linewidth=0.5,
            label=f"{i}: {FACE_LABEL_NAME[i]}",
        )
        for i in range(len(LABEL_HEX_COLORS))
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)


def _face_uv_channel_means(data: object, face_idx: int) -> dict[str, float]:
    out: dict[str, float] = {}
    if not hasattr(data, "node_data") or data.node_data is None:
        return out
    nd = data.node_data[face_idx].detach().cpu().float().numpy()
    if nd.ndim != 3 or nd.shape[-1] < 7:
        return out
    m = nd.mean(axis=(0, 1))
    for i, key in enumerate(_UV_FACE_KEYS):
        out[f"face_uv_mean_{key}"] = float(m[i]) if i < len(m) else 0.0
    return out


def _spatial_pos_row_stats(data: object, face_idx: int) -> dict[str, float]:
    out: dict[str, float] = {}
    if not hasattr(data, "spatial_pos") or data.spatial_pos is None:
        return out
    row = data.spatial_pos[face_idx].detach().cpu().float().numpy().ravel()
    row = row[np.isfinite(row)]
    row = row[row < 1e7]
    if row.size == 0:
        return out
    out["spatial_pos_row_mean"] = float(row.mean())
    out["spatial_pos_row_min"] = float(row.min())
    out["spatial_pos_row_max"] = float(row.max())
    return out


def _pair_tensor_row_mean(data: object, attr: str, face_idx: int) -> Optional[float]:
    if not hasattr(data, attr):
        return None
    t = getattr(data, attr)
    if t is None or not isinstance(t, torch.Tensor):
        return None
    if t.dim() < 2:
        return None
    row = t[face_idx].float().mean().item()
    return float(row) if np.isfinite(row) else None


def export_graphml_full(data: object, pt_path: Path, out_graphml: Path) -> None:
    """Directed arc-for-arc graph + scalar / summarized attributes (Graphia / Gephi)."""
    import networkx as nx

    lf = data.label_feature.view(-1).long()
    n = int(lf.numel())
    ei = data.edge_index.long()
    e_cols = int(ei.shape[1])

    G = nx.DiGraph()
    G.graph["source_pt"] = pt_path.name
    G.graph["num_faces"] = str(n)
    G.graph["num_directed_arcs"] = str(e_cols)
    G.graph["attr_note"] = (
        "Face/edge UV grids: channel means only (face_uv_mean_*, edge_uv_mean_*). "
        "Pairwise tensors summarized per-node (row_mean) or per-arc pair (uv_mean); "
        "raw edge_path / full attn_bias blocks are not inlined."
    )

    for i in range(n):
        lid = int(lf[i].item())
        attrs: dict[str, Any] = {
            "label_id": lid,
            "label_name": label_name(lid),
            "viz_color_hex": label_color(lid),
        }
        if hasattr(data, "face_type") and data.face_type is not None:
            z = int(data.face_type[i].item())
            attrs["face_type"] = z
            attrs["face_type_name"] = FACE_TYPE_NAME.get(z, f"unknown_{z}")
        if hasattr(data, "face_area") and data.face_area is not None:
            attrs["face_area"] = _py_val(data.face_area[i])
        if hasattr(data, "face_loop") and data.face_loop is not None:
            attrs["face_loop"] = int(data.face_loop[i].item())
        if hasattr(data, "face_adj") and data.face_adj is not None:
            attrs["face_adj"] = int(data.face_adj[i].item())
        if hasattr(data, "node_degree") and data.node_degree is not None:
            attrs["node_degree"] = int(data.node_degree[i].item())
        if hasattr(data, "data_id"):
            try:
                attrs["data_id"] = int(getattr(data, "data_id"))
            except Exception:
                attrs["data_id"] = str(getattr(data, "data_id"))
        attrs.update(_face_uv_channel_means(data, i))
        attrs.update(_spatial_pos_row_stats(data, i))
        for key in ("d2_distance", "angle_distance"):
            rm = _pair_tensor_row_mean(data, key, i)
            if rm is not None:
                attrs[f"{key}_row_mean"] = rm

        if hasattr(data, "attn_bias") and data.attn_bias is not None:
            ab = data.attn_bias
            if isinstance(ab, torch.Tensor) and ab.dim() == 2 and i < ab.shape[0]:
                lim = min(n, ab.shape[1])
                if lim > 0:
                    attrs["attn_bias_face_row_mean"] = float(
                        ab[i, :lim].float().mean().item()
                    )

        G.add_node(i, **{str(k): _py_val(v) for k, v in attrs.items()})

    ed = getattr(data, "edge_data", None)
    et = getattr(data, "edge_type", None)
    elen = getattr(data, "edge_len", None)
    eang = getattr(data, "edge_ang", None)
    ecnv = getattr(data, "edge_conv", None)

    for k in range(e_cols):
        u = int(ei[0, k])
        v = int(ei[1, k])
        eattrs: dict[str, Any] = {"arc_index": k, "src_face": u, "dst_face": v}
        if et is not None:
            t = int(et[k].item())
            eattrs["edge_type"] = t
            eattrs["edge_type_name"] = EDGE_TYPE_NAME.get(t, f"unknown_{t}")
        if elen is not None:
            eattrs["edge_len"] = _py_val(elen[k])
        if eang is not None:
            eattrs["edge_ang"] = _py_val(eang[k])
        if ecnv is not None:
            c = int(ecnv[k].item())
            eattrs["edge_conv"] = c
            eattrs["edge_conv_name"] = EDGE_CONV_NAME.get(c, f"unknown_{c}")

        if ed is not None and isinstance(ed, torch.Tensor) and k < ed.shape[0]:
            row = ed[k].detach().cpu().float().numpy()
            if row.ndim == 2 and row.shape[-1] >= 7:
                m = row.mean(axis=0)
                for j, key in enumerate(_UV_EDGE_KEYS):
                    eattrs[f"edge_uv_mean_{key}"] = float(m[j]) if j < len(m) else 0.0

        if hasattr(data, "spatial_pos") and data.spatial_pos is not None:
            spm = data.spatial_pos
            if isinstance(spm, torch.Tensor) and u < spm.shape[0] and v < spm.shape[1]:
                eattrs["spatial_pos_uv"] = int(spm[u, v].item())

        if hasattr(data, "edge_path") and data.edge_path is not None:
            ep = data.edge_path
            if isinstance(ep, torch.Tensor) and u < ep.shape[0] and v < ep.shape[1]:
                seq = ep[u, v].detach().cpu().long().numpy().ravel()
                seq = seq[seq >= 0]
                eattrs["shortest_path_num_hops"] = int(seq.size)
                if seq.size:
                    eattrs["shortest_path_first_edge_graph_idx"] = int(seq[0])

        for pair_attr in ("d2_distance", "angle_distance"):
            if not hasattr(data, pair_attr):
                continue
            t = getattr(data, pair_attr)
            if isinstance(t, torch.Tensor) and t.dim() >= 3:
                if u < t.shape[0] and v < t.shape[1]:
                    cell = t[u, v].float().mean().item()
                    if np.isfinite(cell):
                        eattrs[f"{pair_attr}_uv_mean"] = float(cell)

        G.add_edge(u, v, **{str(kk): _py_val(vv) for kk, vv in eattrs.items()})

    out_graphml.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(out_graphml))


def collect_pt_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for p in args.pt:
        paths.append(Path(p).resolve())
    if args.bin_dir:
        bd = Path(args.bin_dir).resolve()
        _log(f"Globbing under {bd} ({args.glob_pattern}) ...")
        for p in sorted(bd.glob(args.glob_pattern)):
            if p.is_file():
                paths.append(p.resolve())
    if args.dataset_root:
        root = Path(args.dataset_root).resolve()
        scan = _resolve_graph_pt_scan_root(root, args.pt_subdir or None)
        _log(f"Discovering *[0-9].pt under:\n  {scan}")
        _log(
            "This step can take several minutes on large or network folders "
            "(nothing else prints until discovery finishes)."
        )
        discovered: list[Path] = []
        n_seen = 0
        for p in scan.rglob("*[0-9].pt"):
            discovered.append(p.resolve())
            n_seen += 1
            if n_seen % 4000 == 0:
                _log(f"  ... {n_seen} paths discovered so far")
        _log(f"Discovery finished: {n_seen} path(s).")
        paths.extend(discovered)
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    uniq.sort(key=lambda x: str(x))
    return uniq


def output_paths_for_stem(
    pt_path: Path,
    base_out: Path,
    html: bool,
    png: bool,
    svg: bool,
    graphml: bool,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    stem = pt_path.stem
    h = base_out / "html" / f"{stem}.html" if html else None
    p = base_out / "images" / f"{stem}.png" if png else None
    s = base_out / "images" / f"{stem}.svg" if svg else None
    gm = base_out / "graphml" / f"{stem}.graphml" if graphml else None
    return h, p, s, gm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize B-rep face PyG graphs (.pt).")
    p.add_argument("--pt", action="append", default=[], help="Path to one .pt file (repeatable).")
    p.add_argument("--bin_dir", type=str, default=None, help="Directory; glob all graphs.")
    p.add_argument("--glob_pattern", type=str, default="*.pt", help="Used with --bin_dir.")
    p.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Collect every *[0-9].pt under --pt_subdir (no train/val/test lists).",
    )
    p.add_argument(
        "--pt_subdir",
        type=str,
        default="output/bin",
        help=(
            "Under dataset_root for discovery. Pass empty string to scan all of dataset_root."
        ),
    )
    p.add_argument("--out_dir", type=str, required=True, help="Gets html/, images/, graphml/.")
    p.add_argument("--max_nodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_graphs", type=int, default=None)
    p.add_argument("--html", dest="html", action="store_true", default=True)
    p.add_argument("--no_html", dest="html", action="store_false")
    p.add_argument(
        "--png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export per-graph PNG under images/ (default: on). Use --no-png to skip (faster).",
    )
    p.add_argument("--svg", action="store_true")
    p.add_argument("--graphml", dest="graphml", action="store_true", default=True)
    p.add_argument("--no_graphml", dest="graphml", action="store_false")
    p.add_argument("--physics", action="store_true", default=True)
    p.add_argument("--no_physics", dest="physics", action="store_false")
    p.add_argument(
        "--check_label_json",
        type=str,
        default=None,
        help="Directory of label JSON files (<stem>.json); mismatches summarized after export.",
    )
    p.add_argument(
        "--write_legend",
        action="store_true",
        help=f'Write "{LEGEND_FILENAME}" under out_dir/images/ once.',
    )
    p.epilog = (
        "Outputs: out_dir/html/, out_dir/images/ (PNG default), out_dir/graphml/. "
        "HTML/PNG/SVG use an induced subgraph when --max_nodes is set; "
        "GraphML always contains the full graph from the .pt file. "
        "PyVis: --no_physics for large subgraphs."
    )
    return p.parse_args()


_PBAR_FMT = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]"


def main() -> None:
    args = parse_args()
    _log("BrepMFR PyG graph visualization — starting.")
    out_dir = Path(args.out_dir).resolve()
    html_dir = out_dir / "html"
    images_dir = out_dir / "images"
    graphml_dir = out_dir / "graphml"
    html_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    graphml_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output root: {out_dir}")

    label_json_dir = (
        Path(args.check_label_json).resolve() if args.check_label_json else None
    )

    pts = collect_pt_paths(args)
    if not pts:
        _log(
            "No .pt files matched. Provide --pt, --bin_dir, and/or --dataset_root.",
        )
        sys.exit(1)
    if args.max_graphs is not None:
        pts = pts[: int(args.max_graphs)]

    _log(f"Unique graphs to process: {len(pts)}")

    if args.write_legend:
        legend_path = images_dir / LEGEND_FILENAME
        if not legend_path.is_file():
            write_class_legend_png(legend_path)
            _log(f"Wrote legend: {legend_path}")

    label_warnings: List[str] = []

    _log("Exporting (progress bar on stderr) ...")
    for pt_path in tqdm(
        pts,
        desc="Exporting",
        total=len(pts),
        bar_format=_PBAR_FMT,
        file=sys.stderr,
        mininterval=0.25,
        dynamic_ncols=True,
    ):
        data = load_pyg(pt_path)
        lf = validate_pyg(data, pt_path)
        n_full = int(lf.numel())

        if label_json_dir is not None:
            maybe_check_label_json(pt_path, label_json_dir, lf, label_warnings)

        if args.graphml:
            gm_path = graphml_dir / f"{pt_path.stem}.graphml"
            export_graphml_full(data, pt_path, gm_path)

        keep = subsample_keep(n_full, args.max_nodes, args.seed)
        ei = data.edge_index.long()
        g = build_undirected_simple_graph(ei, keep)

        h_out, png_out, svg_out, _ = output_paths_for_stem(
            pt_path,
            out_dir,
            args.html,
            args.png,
            args.svg,
            False,
        )

        if h_out:
            export_pyvis_html(g, data, lf, h_out, physics=args.physics)
        if png_out:
            export_matplotlib_figure(g, lf, png_out, seed=args.seed, fmt="png")
        if svg_out:
            export_matplotlib_figure(g, lf, svg_out, seed=args.seed, fmt="svg")

    if label_warnings:
        _log(f"\nLabel check: {len(label_warnings)} issue(s), first 15:")
        for line in label_warnings[:15]:
            _log(line)

    _log(f"Done. Outputs under: {out_dir}")


if __name__ == "__main__":
    main()
