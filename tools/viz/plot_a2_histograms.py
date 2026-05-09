# plot_a2_histograms.py
# Plots D2 and A3 proximity histograms from a BrepMFR .bin file.
# Displays one face pair per page. Navigate with keyboard arrows or buttons.
#
# Usage:
#   python plot_a2_histograms.py --bin_path path/to/model.bin
#   python plot_a2_histograms.py --bin_path path/to/model.bin --out_dir ./histograms

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from dgl.data.utils import load_graphs


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_bin(bin_path: str):
    graphs, labels = load_graphs(bin_path)
    g = graphs[0]
    d2 = labels.get("d2_distance")
    a3 = labels.get("angle_distance")
    return g, d2, a3


def print_model_info(g, d2, a3):
    N = g.num_nodes()
    E = g.num_edges()
    print()
    print("=" * 52)
    print(f"  Faces (nodes)           : {N}")
    print(f"  Edges                   : {E}")
    if d2 is not None and a3 is not None:
        nonzero = sum(
            1 for i in range(N) for j in range(N)
            if i != j and float(d2[i, j].sum()) > 0
        )
        print(f"  Non-zero A2 pairs       : {nonzero} / {N * (N - 1)}")
    else:
        print("  WARNING: d2_distance or angle_distance missing in bin.")
    print("=" * 52)
    print()
    return N


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def select_pairs(N: int, d2, a3):
    pairs_with_data = [
        (i, j)
        for i in range(N)
        for j in range(N)
        if i != j and d2 is not None and float(d2[i, j].sum()) > 0
    ]

    if not pairs_with_data:
        print("[WARNING] No non-zero A2 pairs found. Using all off-diagonal pairs.")
        pairs_with_data = [(i, j) for i in range(N) for j in range(N) if i != j]

    cap = len(pairs_with_data)
    print(f"Pairs with non-zero A2 data: {cap}")
    print(f"Enter the number of face pairs to inspect (1 - {min(cap, 100)}).")
    print("You will be able to navigate through them one at a time.\n")

    while True:
        try:
            n = int(input(f"How many pairs? (1-{min(cap, 100)}): ").strip())
            if 1 <= n <= min(cap, 100):
                break
            print(f"  Enter a value between 1 and {min(cap, 100)}.")
        except ValueError:
            print("  Invalid input.")

    selected = pairs_with_data[:n]
    print(f"\n{n} pair(s) selected.")
    print("Navigation: click Prev/Next buttons, or press LEFT/RIGHT arrow keys.\n")
    return selected


# ---------------------------------------------------------------------------
# Single-pair render
# ---------------------------------------------------------------------------

def draw_pair(fig, i: int, j: int, d2_hist: np.ndarray, a3_hist: np.ndarray,
              pair_idx: int, total: int, bin_path: str):
    """Clear the figure and draw histograms for one face pair."""
    fig.clf()
    fig.patch.set_facecolor("#fafafa")

    fig.suptitle(
        f"A2 Proximity Histograms  —  Face {i} \u2192 Face {j}"
        f"    [{pair_idx + 1} / {total}]\n"
        f"{os.path.basename(bin_path)}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        wspace=0.38,
        left=0.09,
        right=0.97,
        top=0.84,
        bottom=0.22,
    )

    bins = np.arange(64)

    # ---- D2 ----
    ax_d2 = fig.add_subplot(gs[0, 0])
    ax_d2.bar(bins, d2_hist, color="#4A90C4", edgecolor="none", width=0.82)
    ax_d2.set_xlim(-1, 64)
    ax_d2.set_ylim(0, max(float(d2_hist.max()) * 1.30, 0.002))
    ax_d2.set_xlabel("Bin index  (0 = closest,  63 = farthest)", fontsize=10)
    ax_d2.set_ylabel("Frequency", fontsize=10)
    ax_d2.set_title(f"(b)  Normalized D2 Distance\nFace {i} \u2192 Face {j}", fontsize=10, pad=6)
    ax_d2.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax_d2.tick_params(labelsize=9)
    ax_d2.spines["top"].set_visible(False)
    ax_d2.spines["right"].set_visible(False)
    ax_d2.annotate(
        f"sum={d2_hist.sum():.4f}   mean={d2_hist.mean():.5f}",
        xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=8, color="#555",
    )

    # ---- A3 ----
    ax_a3 = fig.add_subplot(gs[0, 1])
    ax_a3.bar(bins, a3_hist, color="#4A90C4", edgecolor="none", width=0.82)
    ax_a3.set_xlim(-1, 64)
    ax_a3.set_ylim(0, max(float(a3_hist.max()) * 1.30, 0.002))
    ax_a3.set_xlabel("Bin index  (0\u00b0 = aligned,  63 = 180\u00b0)", fontsize=10)
    ax_a3.set_ylabel("Frequency", fontsize=10)
    ax_a3.set_title(f"(c)  Normalized A3 Distance\nFace {i} \u2192 Face {j}", fontsize=10, pad=6)
    ax_a3.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax_a3.tick_params(labelsize=9)
    ax_a3.spines["top"].set_visible(False)
    ax_a3.spines["right"].set_visible(False)
    ax_a3.annotate(
        f"sum={a3_hist.sum():.4f}   mean={a3_hist.mean():.5f}",
        xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=8, color="#555",
    )

    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Navigator (buttons + keyboard)
# ---------------------------------------------------------------------------

class Navigator:
    def __init__(self, fig, selected, d2, a3, bin_path, out_dir):
        self.fig = fig
        self.selected = selected
        self.d2 = d2
        self.a3 = a3
        self.bin_path = bin_path
        self.out_dir = out_dir
        self.idx = 0
        self.total = len(selected)

        # Button axes — placed in the bottom strip (below gs bottom=0.22)
        ax_prev    = fig.add_axes([0.08, 0.06, 0.13, 0.06])
        ax_next    = fig.add_axes([0.23, 0.06, 0.13, 0.06])
        ax_save    = fig.add_axes([0.45, 0.06, 0.16, 0.06])
        ax_saveall = fig.add_axes([0.63, 0.06, 0.16, 0.06])

        self.btn_prev    = Button(ax_prev,    "\u25c4  Prev",    color="#e8e8e8", hovercolor="#d0d0d0")
        self.btn_next    = Button(ax_next,    "Next  \u25ba",    color="#e8e8e8", hovercolor="#d0d0d0")
        self.btn_save    = Button(ax_save,    "Save this",       color="#c8dff5", hovercolor="#a8c8f0")
        self.btn_saveall = Button(ax_saveall, "Save all",        color="#c8dff5", hovercolor="#a8c8f0")

        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.btn_save.on_clicked(self._save_current)
        self.btn_saveall.on_clicked(self._save_all)

        fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._render()

    def _render(self):
        i, j = self.selected[self.idx]
        d2h = self.d2[i, j].numpy().astype(np.float32) if self.d2 is not None else np.zeros(64)
        a3h = self.a3[i, j].numpy().astype(np.float32) if self.a3 is not None else np.zeros(64)
        draw_pair(self.fig, i, j, d2h, a3h, self.idx, self.total, self.bin_path)
        # draw_pair calls fig.clf() which removes button axes — re-add them
        self._restore_buttons()
        self.fig.canvas.draw_idle()

    def _restore_buttons(self):
        """Re-attach button axes after fig.clf()."""
        ax_prev    = self.fig.add_axes([0.08, 0.06, 0.13, 0.06])
        ax_next    = self.fig.add_axes([0.23, 0.06, 0.13, 0.06])
        ax_save    = self.fig.add_axes([0.45, 0.06, 0.16, 0.06])
        ax_saveall = self.fig.add_axes([0.63, 0.06, 0.16, 0.06])

        self.btn_prev    = Button(ax_prev,    "\u25c4  Prev",  color="#e8e8e8", hovercolor="#d0d0d0")
        self.btn_next    = Button(ax_next,    "Next  \u25ba",  color="#e8e8e8", hovercolor="#d0d0d0")
        self.btn_save    = Button(ax_save,    "Save this",     color="#c8dff5", hovercolor="#a8c8f0")
        self.btn_saveall = Button(ax_saveall, "Save all",      color="#c8dff5", hovercolor="#a8c8f0")

        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.btn_save.on_clicked(self._save_current)
        self.btn_saveall.on_clicked(self._save_all)

    def _prev(self, event=None):
        if self.idx > 0:
            self.idx -= 1
            self._render()

    def _next(self, event=None):
        if self.idx < self.total - 1:
            self.idx += 1
            self._render()

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self._next()
        elif event.key in ("left", "p"):
            self._prev()

    def _save_current(self, event=None):
        i, j = self.selected[self.idx]
        path = self._out_path(i, j)
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    def _save_all(self, event=None):
        if self.out_dir is None:
            print("[INFO] Provide --out_dir to enable Save all.")
            return
        original = self.idx
        for k in range(self.total):
            self.idx = k
            i, j = self.selected[k]
            d2h = self.d2[i, j].numpy().astype(np.float32) if self.d2 is not None else np.zeros(64)
            a3h = self.a3[i, j].numpy().astype(np.float32) if self.a3 is not None else np.zeros(64)
            # Draw off-screen without triggering interactive render loop
            tmp_fig, tmp_axes = plt.subplots(1, 2, figsize=(13, 5.5))
            tmp_fig.patch.set_facecolor("#fafafa")
            self._draw_to_axes(tmp_axes[0], tmp_axes[1], i, j, d2h, a3h, k)
            path = self._out_path(i, j)
            tmp_fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(tmp_fig)
            print(f"  Saved [{k + 1}/{self.total}]: {path}")
        self.idx = original
        self._render()
        print("Done.")

    def _draw_to_axes(self, ax_d2, ax_a3, i, j, d2h, a3h, pair_idx):
        bins = np.arange(64)
        for ax, hist, title, xlabel in [
            (ax_d2, d2h,
             f"(b) Normalized D2 Distance\nFace {i} \u2192 Face {j}",
             "Bin index  (0 = closest,  63 = farthest)"),
            (ax_a3, a3h,
             f"(c) Normalized A3 Distance\nFace {i} \u2192 Face {j}",
             "Bin index  (0\u00b0 = aligned,  63 = 180\u00b0)"),
        ]:
            ax.bar(bins, hist, color="#4A90C4", edgecolor="none", width=0.82)
            ax.set_xlim(-1, 64)
            ax.set_ylim(0, max(float(hist.max()) * 1.30, 0.002))
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(title, fontsize=10, pad=6)
            ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.annotate(
                f"sum={hist.sum():.4f}   mean={hist.mean():.5f}",
                xy=(0.98, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="#555",
            )
        basename = os.path.basename(self.bin_path)
        ax_d2.figure.suptitle(
            f"A2 Proximity  —  Face {i} \u2192 Face {j}"
            f"  [{pair_idx + 1}/{self.total}]  |  {basename}",
            fontsize=10, fontweight="bold",
        )

    def _out_path(self, i: int, j: int) -> str:
        stem = os.path.splitext(os.path.basename(self.bin_path))[0]
        name = f"{stem}_face{i}_to_face{j}.png"
        d = self.out_dir or "."
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect A2 proximity histograms (D2 + A3) from a BrepMFR .bin file."
    )
    parser.add_argument("--bin_path", type=str, required=True,
                        help="Path to a .bin graph file.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save figures. Required for 'Save all'.")
    args = parser.parse_args()

    print(f"\nLoading: {args.bin_path}")
    try:
        g, d2, a3 = load_bin(args.bin_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    N = print_model_info(g, d2, a3)

    if N < 2:
        print("[ERROR] Model has fewer than 2 faces.")
        sys.exit(1)

    if d2 is None or a3 is None:
        print("[ERROR] d2_distance or angle_distance not found in bin.")
        print("        Regenerate bins with A2 proximity computation enabled.")
        sys.exit(1)

    selected = select_pairs(N, d2, a3)

    fig = plt.figure(figsize=(13, 5.5))
    Navigator(fig, selected, d2, a3, args.bin_path, args.out_dir)
    plt.show()


if __name__ == "__main__":
    main()