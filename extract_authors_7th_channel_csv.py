import csv
from pathlib import Path
from typing import Optional, Tuple

import torch
from dgl.data.utils import load_graphs


# -----------------------
# CONFIG
# -----------------------
BIN_ROOT = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment4\source_dataset\output\bin_temp")
OUT_CSV = Path(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\temp\cadsynth_temp_edata_x_ch7.csv")

EKEY_X = "x"   # UV grid points of curve
EKEY_T = "t"   # edge type
EKEY_A = "a"   # dihedral edge angle

N_UV = 5
VEC_D = 7
CHANNEL_IDX = 6  # 7th channel in the 7D vector


def _find_bin_files(root: Path):
    return sorted(root.rglob("*.bin"))


def _coerce_edge_x_shape(edge_x: torch.Tensor) -> Tuple[Optional[torch.Tensor], str]:
    if edge_x is None or not torch.is_tensor(edge_x):
        return None, "edata['x'] missing"

    t = edge_x.detach().cpu()

    # (E, 5, 7)
    if t.ndim == 3 and t.shape[1] == N_UV and t.shape[2] == VEC_D:
        return t, ""

    # (E, 35)
    if t.ndim == 2 and t.shape[1] == N_UV * VEC_D:
        return t.reshape(t.shape[0], N_UV, VEC_D), ""

    # (E, 1, 35) or (E, 35, 1)
    if t.ndim == 3:
        if t.shape[1] == 1 and t.shape[2] == N_UV * VEC_D:
            t2 = t.squeeze(1)
            return t2.reshape(t2.shape[0], N_UV, VEC_D), ""
        if t.shape[2] == 1 and t.shape[1] == N_UV * VEC_D:
            t2 = t.squeeze(2)
            return t2.reshape(t2.shape[0], N_UV, VEC_D), ""

    return None, f"Unexpected edata['x'] shape: {tuple(t.shape)}"


def _coerce_1d_per_edge(tensor: torch.Tensor, key_name: str) -> Tuple[Optional[torch.Tensor], str]:
    """
    Coerce per-edge tensor to shape (E,).
    Supports:
      - (E,)
      - (E,1)
    """
    if tensor is None or not torch.is_tensor(tensor):
        return None, f"edata['{key_name}'] missing"

    t = tensor.detach().cpu()

    if t.ndim == 1:
        return t, ""
    if t.ndim == 2 and t.shape[1] == 1:
        return t.squeeze(1), ""

    return None, f"Unexpected edata['{key_name}'] shape: {tuple(t.shape)}"


def main():
    bin_files = _find_bin_files(BIN_ROOT)
    print(f"[INFO] Found {len(bin_files)} .bin files")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "num_faces",
        "num_edges",
        "edge_id",
        "edge_type",      # edata:t
        "dihedral_angle", # edata:a
        "ch7_u0",
        "ch7_u1",
        "ch7_u2",
        "ch7_u3",
        "ch7_u4",
    ]

    rows_written = 0

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fp in bin_files:
            try:
                graphs, _ = load_graphs(str(fp))
                if not graphs:
                    raise RuntimeError("No graphs found in bin")

                g = graphs[0]
                num_faces = int(g.num_nodes())
                num_edges = int(g.num_edges())

                # Required keys
                for k in (EKEY_X, EKEY_T, EKEY_A):
                    if k not in g.edata:
                        raise RuntimeError(f"Missing edata['{k}']")

                edge_x, err_x = _coerce_edge_x_shape(g.edata[EKEY_X])
                if edge_x is None:
                    raise RuntimeError(err_x)

                edge_t, err_t = _coerce_1d_per_edge(g.edata[EKEY_T], EKEY_T)
                if edge_t is None:
                    raise RuntimeError(err_t)

                edge_a, err_a = _coerce_1d_per_edge(g.edata[EKEY_A], EKEY_A)
                if edge_a is None:
                    raise RuntimeError(err_a)

                if num_edges != edge_x.shape[0]:
                    raise RuntimeError("Edge count mismatch for edata['x']")
                if num_edges != edge_t.shape[0]:
                    raise RuntimeError("Edge count mismatch for edata['t']")
                if num_edges != edge_a.shape[0]:
                    raise RuntimeError("Edge count mismatch for edata['a']")

                # Extract 7th channel values for all 5 UV points -> (E, 5)
                ch7 = edge_x[:, :, CHANNEL_IDX]

                # Write one row per edge
                for e in range(num_edges):
                    vals = ch7[e].tolist()
                    writer.writerow(
                        {
                            "filename": fp.name,
                            "num_faces": num_faces,
                            "num_edges": num_edges,
                            "edge_id": e,
                            "edge_type": int(edge_t[e].item()) if edge_t[e].numel() == 1 else edge_t[e].item(),
                            "dihedral_angle": float(edge_a[e].item()),
                            "ch7_u0": float(vals[0]),
                            "ch7_u1": float(vals[1]),
                            "ch7_u2": float(vals[2]),
                            "ch7_u3": float(vals[3]),
                            "ch7_u4": float(vals[4]),
                        }
                    )
                    rows_written += 1

            except Exception as e:
                print(f"[WARN] {fp.name} failed: {type(e).__name__}: {e}")

    print("========== DONE ==========")
    print(f"Rows written: {rows_written}")
    print(f"CSV saved at: {OUT_CSV}")


if __name__ == "__main__":
    main()