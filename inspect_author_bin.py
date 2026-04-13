import argparse
import numpy as np
import torch
from dgl.data.utils import load_graphs

def _stats(t: torch.Tensor):
    t = t.detach().cpu()
    finite = torch.isfinite(t) if t.is_floating_point() else torch.ones_like(t, dtype=torch.bool)
    n = t.numel()
    n_finite = int(finite.sum().item())
    if n_finite == 0:
        return {"shape": list(t.shape), "dtype": str(t.dtype), "min": None, "max": None, "finite": f"{n_finite}/{n}"}
    tf = t[finite]
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(tf.min().item()),
        "max": float(tf.max().item()),
        "finite": f"{n_finite}/{n}",
    }

def _print_tensor(name: str, t: torch.Tensor, indent="  "):
    s = _stats(t)
    print(f"{indent}{name}: shape={s['shape']} dtype={s['dtype']} finite={s['finite']} min={s['min']} max={s['max']}")

def _print_dict_tensors(title: str, d: dict, indent="  "):
    print(f"\n{title}")
    keys = sorted(list(d.keys()))
    if not keys:
        print(f"{indent}(none)")
        return
    for k in keys:
        v = d[k]
        if torch.is_tensor(v):
            _print_tensor(k, v, indent=indent)
        else:
            # aux dict sometimes contains non-tensors
            print(f"{indent}{k}: (non-tensor) type={type(v)}")

def inspect_edge_data_vs_angle(g):
    if "x" not in g.edata:
        print("\n[edge_data] edata['x'] not present.")
        return
    ex = g.edata["x"].detach().cpu()
    print("\n[edge_data = edata['x']]")
    _print_tensor("edata:x", ex, indent="  ")

    # Print per-channel stats if last dim is channels
    if ex.ndim == 3:
        E, S, C = ex.shape
        print(f"  Interpreting as [num_edges={E}, samples={S}, channels={C}]")
        # per-channel min/max (fast enough for single file)
        for c in range(C):
            ch = ex[:, :, c]
            s = _stats(ch)
            print(f"    channel[{c}]: min={s['min']} max={s['max']} finite={s['finite']}")

        # If channel 6 exists, test correlation with edge_ang (edata['a'])
        if C >= 7 and "a" in g.edata:
            ea = g.edata["a"].detach().cpu().float()  # [E]
            # compare ex[:,:,6] to ea repeated across samples
            ch6 = ex[:, :, 6].float()                 # [E,S]
            ea_rep = ea.view(-1, 1).expand(-1, S)     # [E,S]

            diff = (ch6 - ea_rep).abs()
            # finite-only diff
            mask = torch.isfinite(diff)
            if mask.any():
                max_diff = float(diff[mask].max().item())
                mean_diff = float(diff[mask].mean().item())
                # Also correlation on finite values
                v1 = ch6[mask].flatten()
                v2 = ea_rep[mask].flatten()
                # handle constant vectors
                corr = None
                if v1.numel() > 2 and float(v1.std().item()) > 0 and float(v2.std().item()) > 0:
                    corr = float(torch.corrcoef(torch.stack([v1, v2]))[0, 1].item())
                print("\n  [check] Does channel[6] behave like edge_ang (edata['a'])?")
                print(f"    max|channel6 - edge_ang| = {max_diff}")
                print(f"    mean|channel6 - edge_ang| = {mean_diff}")
                print(f"    corr(channel6, edge_ang) = {corr}")
            else:
                print("\n  [check] channel6/edge_ang diff has no finite values to compare.")
    else:
        print("  NOTE: edata['x'] is not 3D, so cannot interpret as (E, samples, channels).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bin",
        required=True,
        help=r'Path to a .bin file, e.g. "C:\...\authors_data\bin\00000000.bin"',
    )
    args = ap.parse_args()

    graphs, aux = load_graphs(args.bin)
    g = graphs[0]

    print(f"[FILE] {args.bin}")
    print(f"[GRAPH] num_nodes={g.num_nodes()} num_edges={g.num_edges()}")

    # List & stats for all stored tensors
    _print_dict_tensors("[NDATA keys]", g.ndata)
    _print_dict_tensors("[EDATA keys]", g.edata)
    if isinstance(aux, dict):
        _print_dict_tensors("[AUX keys]", aux)
    else:
        print("\n[AUX] (not a dict) type=", type(aux))

    # Focus: edge curve channels + whether channel6 == edge_ang
    inspect_edge_data_vs_angle(g)

    # Also inspect node_data shape/channels
    if "x" in g.ndata:
        nx = g.ndata["x"].detach().cpu()
        print("\n[node_data = ndata['x']]")
        _print_tensor("ndata:x", nx, indent="  ")
        if nx.ndim == 4:
            N, U, V, C = nx.shape
            print(f"  Interpreting as [num_faces={N}, U={U}, V={V}, channels={C}]")
            # per-channel min/max for face uv-grid (optional)
            for c in range(C):
                ch = nx[:, :, :, c]
                s = _stats(ch)
                print(f"    channel[{c}]: min={s['min']} max={s['max']} finite={s['finite']}")

if __name__ == "__main__":
    main()