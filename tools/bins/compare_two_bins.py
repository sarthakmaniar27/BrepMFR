import torch
from dgl.data.utils import load_graphs
import numpy as np


AUTHOR_BIN = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\authors_data\bin\00060708.bin"
YOUR_BIN   = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output\bin\00060708_101.bin"


def tensor_stats(name, t):
    if t is None:
        print(f"{name}: None")
        return
    
    if not torch.is_tensor(t):
        print(f"{name}: Not a tensor")
        return
    
    t = t.detach().cpu()
    
    print(f"\n{name}")
    print("-" * 60)
    print(f"dtype : {t.dtype}")
    print(f"shape : {tuple(t.shape)}")
    print(f"numel : {t.numel()}")

    if t.numel() == 0:
        return

    if t.is_floating_point():
        print(f"min   : {float(t.min())}")
        print(f"max   : {float(t.max())}")
        print(f"mean  : {float(t.mean())}")
        print(f"std   : {float(t.std())}")
        print(f"nan   : {int(torch.isnan(t).sum())}")
        print(f"inf   : {int(torch.isinf(t).sum())}")

        # Top 5 absolute values
        flat = t.reshape(-1)
        abs_flat = torch.abs(flat)
        idx = torch.topk(abs_flat, min(5, flat.numel())).indices
        print("top 5 |values|:", flat[idx].tolist())

    else:
        print(f"min   : {int(t.min())}")
        print(f"max   : {int(t.max())}")
        unique = torch.unique(t)
        print(f"unique_count: {len(unique)}")
        print("sample uniques:", unique[:10].tolist())


def print_curvature_info(g):
    if "x" not in g.edata:
        print("\nNo edata:x found")
        return
    
    edata_x = g.edata["x"].detach().cpu()
    
    if edata_x.dim() < 2:
        print("edata:x shape unexpected:", edata_x.shape)
        return
    
    # curvature channel assumed index 6 (based on your earlier finding)
    if edata_x.shape[-1] > 6:
        curvature = edata_x[..., 6]
        print("\n[Curvature Channel Stats]")
        print("-" * 60)
        print("shape:", curvature.shape)
        print("min :", float(curvature.min()))
        print("max :", float(curvature.max()))
        print("mean:", float(curvature.mean()))
        print("std :", float(curvature.std()))
        print(">1e5 :", int((curvature.abs() > 1e5).sum()))
        print(">1e6 :", int((curvature.abs() > 1e6).sum()))
        print(">1e7 :", int((curvature.abs() > 1e7).sum()))
    else:
        print("Curvature channel index 6 not present.")


def inspect_bin(path, title):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    graphs, aux = load_graphs(path)
    g = graphs[0]

    print("\nGraph Info")
    print("-" * 60)
    print("num_nodes:", g.num_nodes())
    print("num_edges:", g.num_edges())

    print("\n--- ndata ---")
    for k, v in g.ndata.items():
        tensor_stats(f"ndata:{k}", v)

    print("\n--- edata ---")
    for k, v in g.edata.items():
        tensor_stats(f"edata:{k}", v)

    print("\n--- aux ---")
    if isinstance(aux, dict):
        for k, v in aux.items():
            tensor_stats(f"aux:{k}", v)

    print_curvature_info(g)


def main():
    inspect_bin(AUTHOR_BIN, "AUTHOR BIN")
    inspect_bin(YOUR_BIN, "YOUR BIN")


if __name__ == "__main__":
    main()