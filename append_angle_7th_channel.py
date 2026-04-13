import math
from pathlib import Path
from typing import Optional, Tuple

import torch
from dgl.data.utils import load_graphs, save_graphs
from tqdm import tqdm


# -----------------------
# CONFIG
# -----------------------
BIN_DIR = Path(
    r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\target_dataset\output\bin"
)

EKEY_X = "x"
EKEY_A = "a"

N_UV = 5
VEC_D = 7
CHANNEL_IDX = 6
ATOL = 1e-5


# -----------------------
# Helpers
# -----------------------
def wrap_to_pi_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap values to [-pi, pi)
    """
    two_pi = 2.0 * math.pi
    return (x + math.pi) % two_pi - math.pi


def coerce_edge_x_shape(edge_x: torch.Tensor) -> Tuple[Optional[torch.Tensor], str, Optional[str]]:
    """
    Returns:
        reshaped_tensor, error_msg, original_layout

    Supported input layouts:
      - (E, 5, 7)
      - (E, 35)
      - (E, 1, 35)
      - (E, 35, 1)
    """
    if edge_x is None or not torch.is_tensor(edge_x):
        return None, "edata['x'] missing or not a tensor", None

    t = edge_x

    if t.ndim == 3 and t.shape[1] == N_UV and t.shape[2] == VEC_D:
        return t, "", "E_5_7"

    if t.ndim == 2 and t.shape[1] == N_UV * VEC_D:
        return t.reshape(t.shape[0], N_UV, VEC_D), "", "E_35"

    if t.ndim == 3 and t.shape[1] == 1 and t.shape[2] == N_UV * VEC_D:
        t2 = t.squeeze(1).reshape(t.shape[0], N_UV, VEC_D)
        return t2, "", "E_1_35"

    if t.ndim == 3 and t.shape[2] == 1 and t.shape[1] == N_UV * VEC_D:
        t2 = t.squeeze(2).reshape(t.shape[0], N_UV, VEC_D)
        return t2, "", "E_35_1"

    return None, f"Unexpected edata['x'] shape: {tuple(t.shape)}", None


def restore_edge_x_shape(edge_x_reshaped: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "E_5_7":
        return edge_x_reshaped
    if layout == "E_35":
        return edge_x_reshaped.reshape(edge_x_reshaped.shape[0], N_UV * VEC_D)
    if layout == "E_1_35":
        return edge_x_reshaped.reshape(edge_x_reshaped.shape[0], 1, N_UV * VEC_D)
    if layout == "E_35_1":
        return edge_x_reshaped.reshape(edge_x_reshaped.shape[0], N_UV * VEC_D, 1)
    raise ValueError(f"Unsupported layout: {layout}")


def coerce_edge_a_shape(edge_a: torch.Tensor) -> Tuple[Optional[torch.Tensor], str]:
    """
    Coerce edata['a'] to shape (E,)
    Supports:
      - (E,)
      - (E, 1)
    """
    if edge_a is None or not torch.is_tensor(edge_a):
        return None, "edata['a'] missing or not a tensor"

    t = edge_a

    if t.ndim == 1:
        return t, ""

    if t.ndim == 2 and t.shape[1] == 1:
        return t.squeeze(1), ""

    return None, f"Unexpected edata['a'] shape: {tuple(t.shape)}"


# -----------------------
# Main
# -----------------------
def main():
    if not BIN_DIR.exists():
        print(f"[ERROR] Folder does not exist: {BIN_DIR}")
        return

    bin_files = sorted(BIN_DIR.rglob("*.bin"))
    if not bin_files:
        print(f"[ERROR] No .bin files found in: {BIN_DIR}")
        return

    print(f"Found {len(bin_files)} bin files\n")

    files_updated = 0
    files_failed = 0
    total_graphs = 0
    total_edges = 0

    # verification stats
    graphs_with_missing_a = 0
    total_edges_compared_raw = 0
    total_edges_compared_wrapped = 0
    raw_match_edges = 0
    raw_mismatch_edges = 0
    wrapped_match_edges = 0
    wrapped_mismatch_edges = 0

    for fp in tqdm(bin_files, desc="Wrapping x[:,:,6] to [-pi, pi)", unit="file"):
        try:
            graphs, label_dict = load_graphs(str(fp))
            if not graphs:
                raise RuntimeError("No graphs found in bin")

            updated_graphs = []

            for g in graphs:
                total_graphs += 1

                if EKEY_X not in g.edata:
                    raise RuntimeError("Missing edata['x']")

                edge_x_raw = g.edata[EKEY_X]
                edge_x, err_x, layout = coerce_edge_x_shape(edge_x_raw)
                if edge_x is None:
                    raise RuntimeError(err_x)

                num_edges = int(g.num_edges())
                if edge_x.shape[0] != num_edges:
                    raise RuntimeError(
                        f"Mismatch in x edges: num_edges={num_edges}, x.shape[0]={edge_x.shape[0]}"
                    )

                total_edges += num_edges

                # Current 7th channel values from edata:x
                x_ch7 = edge_x[:, :, CHANNEL_IDX]          # (E, 5)
                x_ch7_first = x_ch7[:, 0]                  # (E,)

                # Optional verification against edata:a
                if EKEY_A in g.edata:
                    edge_a_raw = g.edata[EKEY_A]
                    edge_a, err_a = coerce_edge_a_shape(edge_a_raw)
                    if edge_a is None:
                        raise RuntimeError(err_a)

                    if edge_a.shape[0] != num_edges:
                        raise RuntimeError(
                            f"Mismatch in a edges: num_edges={num_edges}, a.shape[0]={edge_a.shape[0]}"
                        )

                    # Compare raw x channel7 vs raw edata:a
                    raw_close = torch.isclose(
                        x_ch7_first.to(edge_a.dtype), edge_a, atol=ATOL, rtol=0.0
                    )
                    total_edges_compared_raw += num_edges
                    raw_match_edges += int(raw_close.sum().item())
                    raw_mismatch_edges += int((~raw_close).sum().item())

                    # Compare wrapped x channel7 vs wrapped edata:a
                    x_wrapped_first = wrap_to_pi_tensor(x_ch7_first)
                    a_wrapped = wrap_to_pi_tensor(edge_a.to(x_wrapped_first.dtype))
                    wrapped_close = torch.isclose(
                        x_wrapped_first, a_wrapped, atol=ATOL, rtol=0.0
                    )
                    total_edges_compared_wrapped += num_edges
                    wrapped_match_edges += int(wrapped_close.sum().item())
                    wrapped_mismatch_edges += int((~wrapped_close).sum().item())
                else:
                    graphs_with_missing_a += 1

                # Wrap the EXISTING 7th channel values only
                x_ch7_wrapped = wrap_to_pi_tensor(x_ch7)

                edge_x_mod = edge_x.clone()
                edge_x_mod[:, :, CHANNEL_IDX] = x_ch7_wrapped

                g.edata[EKEY_X] = restore_edge_x_shape(edge_x_mod, layout)
                updated_graphs.append(g)

            save_graphs(str(fp), updated_graphs, labels=label_dict)
            files_updated += 1

        except Exception:
            files_failed += 1

    print("\n========== DONE ==========")
    print(f"Files updated                    : {files_updated}")
    print(f"Files failed                     : {files_failed}")
    print(f"Graphs processed                 : {total_graphs}")
    print(f"Total edges processed            : {total_edges}")
    print(f"Graphs missing edata['a']        : {graphs_with_missing_a}")

    if total_edges_compared_raw > 0:
        print("\n[Verification: x channel7 vs edata:a]")
        print(f"Raw comparison edges             : {total_edges_compared_raw}")
        print(f"Raw matches                      : {raw_match_edges}")
        print(f"Raw mismatches                   : {raw_mismatch_edges}")
        print(f"Wrapped comparison edges         : {total_edges_compared_wrapped}")
        print(f"Wrapped matches                  : {wrapped_match_edges}")
        print(f"Wrapped mismatches               : {wrapped_mismatch_edges}")

    print(f"\nUpdated in place at              : {BIN_DIR}")


if __name__ == "__main__":
    main()