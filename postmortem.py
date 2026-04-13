# import dgl
# import torch
# from dgl.data.utils import load_graphs

# def post_mortem_authors_bin(bin_path):
#     print(f"\n{'='*20} POST-MORTEM: {bin_path} {'='*20}")
    
#     # 1. Load the raw list from DGL
#     graphs, label_dict = load_graphs(str(bin_path))
#     g = graphs[0]

#     # 2. Inspect Node Data (ndata)
#     print("\n[NODE DATA (ndata)]")
#     for key in g.ndata.keys():
#         tensor = g.ndata[key]
#         print(f"Key: '{key}' | Shape: {list(tensor.shape)} | Dtype: {tensor.dtype}")
#         # 'x' should be [num_faces, U, V, 7] (UV grid)
#         # 'f' is the label (Machining Feature ID)

#     # 3. Inspect Edge Data (edata)
#     print("\n[EDGE DATA (edata)]")
#     for key in g.edata.keys():
#         tensor = g.edata[key]
#         print(f"Key: '{key}' | Shape: {list(tensor.shape)} | Dtype: {tensor.dtype}")
#         # 'x' should be [num_edges, U, 10] (Edge points)

#     # 4. Inspect Global Metadata (The missing piece!)
#     print("\n[GLOBAL METADATA (label_dict)]")
#     if not label_dict:
#         print("ALERT: No global metadata found in this bin file!")
#     else:
#         for key in label_dict.keys():
#             tensor = label_dict[key]
#             print(f"Key: '{key}' | Shape: {list(tensor.shape)} | Dtype: {tensor.dtype}")

#     # 5. Check Topology
#     print("\n[TOPOLOGY]")
#     u, v = g.edges()
#     print(f"Total Nodes: {g.num_nodes()}")
#     print(f"Total Edges: {g.num_edges()}")
    
#     # Check if edges are mirrored (0->1 exists, does 1->0 exist?)
#     edge_pairs = set(zip(u.tolist(), v.tolist()))
#     mirrored = all((v_idx, u_idx) in edge_pairs for u_idx, v_idx in edge_pairs)
#     print(f"Is Graph Fully Bidirectional? {mirrored}")

# if __name__ == "__main__":
#     # Point this to the authors' file you uploaded
#     # AUTHORS_FILE = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output\bin\00060708_101.bin"
#     AUTHORS_FILE = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\source_dataset\output\bin\00098525_101.bin"
#     post_mortem_authors_bin(AUTHORS_FILE)


import torch
import dgl
from dgl.data.utils import load_graphs

# Set print options to see full matrices without truncation
torch.set_printoptions(threshold=10_000, linewidth=200, precision=4, sci_mode=False)

def run_detailed_post_mortem(bin_path):
    graphs, label_dict = load_graphs(bin_path)
    g = graphs[0]
    
    print(f"{'='*20} DETAILED POST-MORTEM: {bin_path} {'='*20}\n")

    # 1. FACE ATTRIBUTES (Node Data)
    # Mapping based on Component 4: Type (z), Area (y), Loops (l), Adjacent (a)
    print("### [FACE ATTRIBUTES (4 Scalars)]")
    if all(k in g.ndata for k in ['z', 'y', 'l', 'a']):
        for i in range(g.num_nodes()):
            print(f"Node {i}: Type(z)={g.ndata['z'][i].item()}, "
                  f"Area(y)={g.ndata['y'][i].item():.4f}, "
                  f"Loops(l)={g.ndata['l'][i].item()}, "
                  f"Adj(a)={g.ndata['a'][i].item()}")
    else:
        print("Missing some face attribute keys in ndata.")
    print("-" * 50)

    # 2. EDGE ATTRIBUTES (Edge Data)
    # Mapping based on Component 3: Type (t), Length (l), Convexity (c), Dihedral (a)
    print("\n### [EDGE ATTRIBUTES (4 Scalars)]")
    if all(k in g.edata for k in ['t', 'l', 'c', 'a']):
        src, dst = g.edges()
        for i in range(g.num_edges()):
            print(f"Edge {i} ({src[i]}->{dst[i]}): Type(t)={g.edata['t'][i].item()}, "
                  f"Length(l)={g.edata['l'][i].item():.4f}, "
                  f"Convexity(c)={g.edata['c'][i].item()}, "
                  f"Dihedral(a)={g.edata['a'][i].item():.4f}")
    else:
        print("Missing some edge attribute keys in edata.")
    print("-" * 50)

    # 3. A1 PROXIMITY: Distance Matrix (spatial_pos)
    print("\n### [A1 PROXIMITY: SPATIAL POSITION (Distance Matrix)]")
    if 'spatial_pos' in label_dict:
        print(label_dict['spatial_pos'])
    else:
        print("Key 'spatial_pos' not found in label_dict.")

    # 4. A3 PROXIMITY: Edges Path
    print("\n### [A3 PROXIMITY: EDGES PATH (Chain of edge types)]")
    if 'edges_path' in label_dict:
        # edges_path is [N, N, max_dist]
        print(f"Shape: {label_dict['edges_path'].shape}")
        print(label_dict['edges_path'])
    else:
        print("Key 'edges_path' not found in label_dict.")

    # 5. A2 PROXIMITY: Statistical Positional Relations
    print("\n### [A2 PROXIMITY: D2 & ANGLE DISTANCE]")
    if 'd2_distance' in label_dict and 'angle_distance' in label_dict:
        print(f"D2 Distance Shape: {label_dict['d2_distance'].shape}")
        print(f"Angle Distance Shape: {label_dict['angle_distance'].shape}")
        # Printing full 64-dim histograms is overwhelming; 
        # showing mean values per face-pair to check for data presence.
        print("\nMean D2 Histogram Values (per face pair):")
        #rint(label_dict['d2_distance'].mean(dim=-1))
        
        print("\nMean Angle Histogram Values (per face pair):")
       #print(label_dict['angle_distance'].mean(dim=-1))
    else:
        print("A2 metadata keys missing.")

    # 5. A2 PROXIMITY: Statistical Positional Relations
    print("\n### [A2 PROXIMITY: D2 & ANGLE DISTANCE]")
    if 'd2_distance' in label_dict and 'angle_distance' in label_dict:
        d2 = label_dict['d2_distance']
        a3 = label_dict['angle_distance']
        num_faces = d2.shape[0]
        
        print(f"D2 Distance Shape: {d2.shape}")
        print(f"Angle Distance Shape: {a3.shape}")

        # --- OPTION 1: Summary (Mean check) ---
        print("\nMean D2 Histogram Values (per face pair):")
        print(d2.mean(dim=-1))

        # --- OPTION 2: Full Histogram Printing ---
        PRINT_FULL_HISTOGRAMS = True  # Toggle this to False to hide details
        
        if PRINT_FULL_HISTOGRAMS:
            # Prevent PyTorch from truncating the 64 values with "..."
            torch.set_printoptions(threshold=1000, linewidth=160, precision=4, sci_mode=False)
            
            print("\n--- FULL 64-BIN HISTOGRAMS ---")
            for i in range(num_faces):
                for j in range(num_faces):
                    # We usually skip the diagonal (self-relations) as they are 0.0
                    if i == j: continue 
                    
                    # Only print pairs that actually have data (sum > 0)
                    if d2[i, j].sum() > 0:
                        print(f"\n[Face {i} -> Face {j}]")
                        print(f"  D2 Dist: {d2[i, j].tolist()}")
                        print(f"  A3 Angl: {a3[i, j].tolist()}")
                        
    else:
        print("A2 metadata keys missing.")

if __name__ == "__main__":
    # path = r"C:\Users\smr52\Desktop\Projects\Satish\Test\Bin\00098525_101.bin"
    # path = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\our_data\output\bin\00000002_101.bin"
    path = r"C:\Users\smr52\Desktop\Projects\Satish\BrepMFR\dataset\Experiment6\source_dataset\output\bin\00098525_101.bin"
    run_detailed_post_mortem(path)
    