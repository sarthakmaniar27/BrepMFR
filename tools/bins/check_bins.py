# import os, glob
# import torch
# from dgl.data.utils import load_graphs

# def check(root, max_files=200):
#     bin_dir = os.path.join(root, "bin")
#     files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))[:max_files]
#     print(root, "bins:", len(files))
#     for p in files:
#         gs, meta = load_graphs(p)
#         g = gs[0]
#         if "f" not in g.ndata:
#             raise RuntimeError(f"Missing ndata['f'] in {p}")
#         f = g.ndata["f"].long()
#         mn, mx = int(f.min()), int(f.max())
#         if mn < 0 or mx >= 25:
#             print("BAD LABEL RANGE", p, "min", mn, "max", mx, "unique_sample", torch.unique(f)[:20])
#             return
#     print("label ranges look OK in first", len(files), "files")

# check(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset")
# check(r"C:\Users\smr52\Desktop\MFCAD++\Experiment")


from dgl.data.utils import load_graphs
g, meta = load_graphs(r"C:\Users\smr52\Desktop\MFCAD++\Experiment\bin\5_101.bin")
sp = meta["spatial_pos"]
print(int(sp.max()), int(sp.min()))
