# import os
# import math
# from pathlib import Path

# import torch
# from dgl.data.utils import load_graphs
# from tqdm import tqdm

# # ----------------------------
# # CONFIG
# # ----------------------------
# ROOT_DIR = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output\bin"
# FILE_PATTERN = ".bin"

# CHANNELS = ["x", "y", "z", "tx", "ty", "tz", "curvature"]

# def walk_bin_files(root_dir: str):
#     out = []
#     for r, _, files in os.walk(root_dir):
#         for fn in files:
#             if fn.lower().endswith(FILE_PATTERN):
#                 out.append(os.path.join(r, fn))
#     return sorted(out)

# def unravel_3d(flat_idx: int, shape):
#     # shape = (E, U, C)
#     E, U, C = shape
#     e = flat_idx // (U * C)
#     rem = flat_idx % (U * C)
#     u = rem // C
#     c = rem % C
#     return int(e), int(u), int(c)

# def main():
#     files = walk_bin_files(ROOT_DIR)
#     print(f"[INFO] ROOT_DIR: {ROOT_DIR}")
#     print(f"[INFO] Found {len(files)} bin files")

#     best_abs = -1.0
#     best = None

#     failed = 0
#     max_error_print = 5

#     for fp in tqdm(files, desc="Scanning max(edata:x)", unit="file"):
#         try:
#             graphs, _ = load_graphs(fp)
#             g = graphs[0]

#             if "x" not in g.edata:
#                 continue

#             edge_x = g.edata["x"]
#             if edge_x.numel() == 0:
#                 continue

#             edge_x = edge_x.cpu()

#             abs_vals = edge_x.abs()
#             local_max = float(abs_vals.max().item())

#             if local_max > best_abs:
#                 flat_idx = int(abs_vals.reshape(-1).argmax().item())
#                 e_idx, u_idx, ch_idx = unravel_3d(flat_idx, abs_vals.shape)
#                 val = float(edge_x[e_idx, u_idx, ch_idx].item())

#                 best_abs = local_max
#                 best = (Path(fp).name, e_idx, u_idx, ch_idx, CHANNELS[ch_idx], val, math.isfinite(val))

#         except Exception as e:
#             failed += 1
#             if failed <= max_error_print:
#                 print(f"[ERROR] {Path(fp).name}: {e}")
#             elif failed == max_error_print + 1:
#                 print("[ERROR] ... further errors suppressed")

#     print("\n[RESULT]")
#     print(f"Failed files: {failed}")
#     if best is None:
#         print("No edata:x found in any bins.")
#         return

#     fname, e_idx, u_idx, ch_idx, ch_name, val, is_finite = best
#     print(f"max|edata:x| = {best_abs}")
#     print(f"file={fname} edge_idx={e_idx} sample_idx={u_idx} channel={ch_idx}({ch_name}) value={val} finite={is_finite}")

# if __name__ == "__main__":
#     main()


import os
from pathlib import Path
import torch
from dgl.data.utils import load_graphs
from tqdm import tqdm

ROOT_DIR = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output\bin"
THRESHOLDS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

def walk(root):
    for r,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".bin"):
                yield os.path.join(r,f)

counts = {t: 0 for t in THRESHOLDS}
total = 0

for fp in tqdm(list(walk(ROOT_DIR)), desc="Scanning curvature", unit="file"):
    g = load_graphs(fp)[0][0]
    ex = g.edata["x"].cpu()          # (E,5,7)
    curv = ex[..., 6].abs().reshape(-1)
    total += curv.numel()
    for t in THRESHOLDS:
        counts[t] += int((curv > t).sum().item())

print("total curvature samples:", total)
for t in THRESHOLDS:
    print(f"> {t:g}: {counts[t]}")