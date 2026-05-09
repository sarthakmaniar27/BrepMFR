import dgl
import torch
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path

# --- PANDAS CONFIGURATION: PREVENT PRUNING ---
pd.set_option('display.max_rows', None)      # Show every single row
pd.set_option('display.max_columns', None)   # Show every single column
pd.set_option('display.width', 1000)         # Wide terminal output
pd.set_option('display.max_colwidth', None)  # Don't cut off long text

def visualize_brep_deep_inspect(bin_path):
    # 1. Load Graph
    graphs, _ = dgl.load_graphs(str(bin_path))
    g = graphs[0]
    
    # Feature Maps
    face_type_map = {0: "Plane", 1: "Cylinder", 2: "Cone", 3: "Sphere", 4: "Torus", 5: "Other"}
    edge_type_map = {0: "Line", 1: "Circle", 2: "Ellipse", 3: "Parabola", 4: "Hyperbola", 5: "Other"}
    conv_map = {0: "Concave", 1: "Convex", 2: "Flat"}

    print(f"\n{'#'*90}")
    print(f"### FULL AUDIT REPORT: {Path(bin_path).name}")
    print(f"{'#'*90}")

    # 2. COMPLETE FACE LIST
    face_types = [face_type_map.get(int(z), f"ID:{int(z)}") for z in g.ndata['z']]
    face_data = {
        "DGL_Idx": np.arange(g.num_nodes()),
        # "JSON_ID": g.ndata['id'].numpy() if 'id' in g.ndata else "N/A",
        "Type": face_types,
        "Area": g.ndata['y'].numpy(),
        "Loops": g.ndata['l'].numpy(),
        "Adj_Count": g.ndata['a'].numpy()
    }
    df_faces = pd.DataFrame(face_data)
    print("\n[COMPLETE FACE LIST]")
    print(df_faces.to_string(index=False))

    # 3. COMPLETE EDGE LIST
    u, v = g.edges()
    edge_types = [edge_type_map.get(int(t), f"ID:{int(t)}") for t in g.edata['t']]
    edge_convs = [conv_map.get(int(c), f"ID:{int(c)}") for c in g.edata['c']]
    
    edge_data = {
        "DGL_Idx": np.arange(g.num_edges()),
        # "JSON_ID": g.edata['id'].numpy() if 'id' in g.edata else "N/A",
        "Connection": [f"F{src} -> F{dst}" for src, dst in zip(u.numpy(), v.numpy())],
        "Type": edge_types,
        "Convexity": edge_convs,
        "Length": g.edata['l'].numpy()
    }
    df_edges = pd.DataFrame(edge_data)
    print("\n[COMPLETE EDGE LIST]")
    print(df_edges.to_string(index=False))

    # 4. STATISTICAL SUMMARY
    print(f"\n{'-'*30} GEOMETRIC SUMMARY {'-'*30}")
    print(f"Total Faces: {g.num_nodes()}")
    print(f"Total Edges: {g.num_edges()}")
    
    print("\nFace Type Distribution:")
    print(df_faces['Type'].value_counts().to_string())
    
    print("\nEdge Convexity Distribution:")
    print(df_edges['Convexity'].value_counts().to_string())
    
    print(f"\nTotal Area: {df_faces['Area'].sum():.4f}")
    print(f"Total Edge Length: {df_edges['Length'].sum():.4f}")
    print(f"{'#'*90}\n")

    # 5. 3D VISUALIZATION (Plotly)
    # [3D visualization logic remains same as previous version]
    points = g.ndata['x'][:, :, :, :3] 
    centroids = points.mean(dim=(1, 2)).numpy()
    
    edge_traces = []
    for i in range(g.num_edges()):
        p0, p1 = centroids[u[i]], centroids[v[i]]
        hover_text = (f"Edge {i}<br>"
                      f"{df_edges['Connection'][i]}<br>"
                      f"Conv: {df_edges['Convexity'][i]}")
        edge_traces.append(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode='lines', line=dict(color='rgba(150,150,150,0.6)', width=4),
            hoverinfo='text', hovertext=hover_text, showlegend=False
        ))

    node_trace = go.Scatter3d(
        x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
        mode='markers+text',
        marker=dict(size=10, color=g.ndata['z'].numpy(), colorscale='Viridis'),
        text=[f"F{i}" for i in range(g.num_nodes())],
        hoverinfo='text', name='Faces'
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(template="plotly_dark", title=f"BRep Audit: {Path(bin_path).name}")
    fig.write_html("full_audit_viz.html")

if __name__ == "__main__":
    MY_PATH = r"C:\Users\smr52\Desktop\Projects\Satish\Test\Bin\00098525_101.bin"
    visualize_brep_deep_inspect(MY_PATH)