# -*- coding: utf-8 -*-
"""Canonical CADSynth / BrepMFR 25-class face labels + fixed visualization palette."""

from __future__ import annotations

import os

# Same Windows OpenMP workaround as visualize_pyg_face_graph (matplotlib loads early here).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.colors as mcolors

NUM_CLASSES = 25

FACE_LABEL_NAME = {
    0: "Stock",
    1: "Rectangular through slot",
    2: "Triangular through slot",
    3: "Rectangular passage",
    4: "Triangular passage",
    5: "6-sided passage",
    6: "Rectangular through step",
    7: "2-sided through step",
    8: "Slanted through step",
    9: "Rectangular blind step",
    10: "Triangular blind step",
    11: "Rectangular blind slot",
    12: "Rectangular pocket",
    13: "Triangular pocket",
    14: "6-sided pocket",
    15: "Chamfer",
    16: "Circular through slot",
    17: "Through hole",
    18: "Circular blind step",
    19: "Horizontal circular end blind slot",
    20: "Vertical circular end blind slot",
    21: "Circular end pocket",
    22: "O-ring",
    23: "Blind hole",
    24: "Round",
}

IGNORE_LABEL_COLOR = "#4A4A4A"


def _build_label_hex_colors(num_classes: int = NUM_CLASSES) -> tuple[str, ...]:
    """Muted gray for Stock (0); turbo-sampled distinct hues for machining classes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = getattr(plt, "colormaps", None)
    cmap = cmap["turbo"] if cmap is not None else plt.cm.get_cmap("turbo")
    stock = "#B8B8B8"
    out: list[str] = [stock]
    denom = max(num_classes - 2, 1)
    for i in range(1, num_classes):
        t = 0.07 + 0.86 * ((i - 1) / denom)
        out.append(mcolors.to_hex(cmap(t)))
    return tuple(out)


LABEL_HEX_COLORS = _build_label_hex_colors()


def label_name(class_id: int) -> str:
    if class_id < 0 or class_id >= NUM_CLASSES:
        return f"out_of_range({class_id})"
    return FACE_LABEL_NAME[class_id]


def label_color(class_id: int) -> str:
    if class_id < 0 or class_id >= NUM_CLASSES:
        return IGNORE_LABEL_COLOR
    return LABEL_HEX_COLORS[class_id]
