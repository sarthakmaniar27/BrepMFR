# import torch
# import dgl
# from dgl.data.utils import load_graphs

# def check_for_out_of_bounds(bin_path):
#     graphs, label_dict = load_graphs(bin_path)
#     g = graphs[0]
#     num_faces = g.num_nodes()
    
#     print(f"\n--- Checking Integrity for: {bin_path} ---")

#     # 1. Check A1: spatial_pos (Shortest Path Distance)
#     if 'spatial_pos' in label_dict:
#         sp = label_dict['spatial_pos']
#         # Most BrepMFR models use an embedding of size 32 or 64 for spatial_pos
#         # If your SolidWorks script found a distance of 100, it will crash here.
#         print(f"A1 (spatial_pos) max value: {sp.max().item()}")
#         if sp.max() >= 32: # Typical limit
#             print(f"!!! ALERT: spatial_pos ({sp.max()}) might be too high for the model's Embedding layer.")

#     # 2. Check A3: edges_path
#     if 'edges_path' in label_dict:
#         ep = label_dict['edges_path']
#         print(f"A3 (edges_path) max value: {ep.max().item()}")
#         # In A3, indices are usually Edge Types. 
#         # Check how many edge types your model expects (usually around 12-16).
#         if ep.max() > 20: 
#             print(f"!!! ALERT: edges_path value ({ep.max()}) seems very high for edge-type indices.")

#     # 3. Check A2: D2/Angle Distances (The Histograms)
#     if 'd2_distance' in label_dict:
#         d2 = label_dict['d2_distance']
#         # The model expects a last dimension of 64
#         print(f"A2 (d2_distance) shape: {list(d2.shape)}")
#         if d2.shape[-1] != 64:
#             print(f"!!! CRITICAL: Model expects 64 bins, but your data has {d2.shape[-1]}.")
        
#         # Check for NaNs (Common with custom SolidWorks calculations)
#         if torch.isnan(d2).any():
#             print("!!! CRITICAL: Found NaNs in d2_distance histograms.")

#     # 4. Check Node Labels (Machining Features)
#     if 'f' in g.ndata:
#         labels = g.ndata['f']
#         print(f"Label (f) range: [{labels.min()}, {labels.max()}]")
#         # If your number of features is 24, max label must be 23.
#         if labels.max() >= 24: 
#              print("!!! ALERT: Feature labels might exceed the classifier output size.")

# if __name__ == "__main__":
#     #path = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output\bin\00000002_101.bin"
#     path = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\authors_data\bin\00000002.bin"
#     check_for_out_of_bounds(path)

import dgl
from dgl.data.utils import load_graphs
graphs, label_dict = load_graphs(r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\authors_data\bin\00000002.bin")
g = graphs[0]

num_nodes = g.num_nodes()
d2_shape = label_dict['d2_distance'].shape[0]

if num_nodes != d2_shape:
    print(f"CRITICAL MISMATCH: Graph has {num_nodes} nodes, but A2 matrix has {d2_shape} rows!")