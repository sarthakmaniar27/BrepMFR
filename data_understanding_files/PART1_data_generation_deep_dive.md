# Part 1: Data Generation — The Complete Deep Dive

> **Who is this for?** Someone with **zero experience** in CAD, SolidWorks, or geometric machine learning. Every concept is explained from scratch with analogies, diagrams, and line-by-line walkthroughs of the actual code in this repository.

---

## Table of Contents

1. [Why Do We Need Data Generation At All?](#1-why-do-we-need-data-generation-at-all)
2. [What Is a CAD Model and What Is a STEP File?](#2-what-is-a-cad-model-and-what-is-a-step-file)
3. [B-Rep: The Mathematical DNA of a 3D Shape](#3-b-rep-the-mathematical-dna-of-a-3d-shape)
4. [The 25 Machining Feature Classes](#4-the-25-machining-feature-classes)
5. [End-to-End Pipeline Overview](#5-end-to-end-pipeline-overview)
6. [Stage A: Sharding & Staging (Distributing Files Across VMs)](#6-stage-a-sharding--staging-distributing-files-across-vms)
7. [Stage B: Synthetic Geometry Augmentation (SolidWorks)](#7-stage-b-synthetic-geometry-augmentation-solidworks)
8. [Stage C: Graph Compilation — JSON to PyTorch Geometric `.pt`](#8-stage-c-graph-compilation--json-to-pytorch-geometric-pt)
9. [The `.pt` File: Every Tensor Explained](#9-the-pt-file-every-tensor-explained)
10. [The Three Inference Profiles: `full`, `no_a2`, `lite`](#10-the-three-inference-profiles-full-no_a2-lite)
11. [Dataset Splits and Split Lists](#11-dataset-splits-and-split-lists)
12. [Collation: How Individual Graphs Become a Batch](#12-collation-how-individual-graphs-become-a-batch)
13. [Infrastructure: Jenkins, VMs, and the "Session 0" Problem](#13-infrastructure-jenkins-vms-and-the-session-0-problem)
14. [Known Limitations and Gotchas](#14-known-limitations-and-gotchas)
15. [File-by-File Reference Map](#15-file-by-file-reference-map)

---

## 1. Why Do We Need Data Generation At All?

Machine learning models need **training data** — thousands (or millions) of labelled examples to learn from. In this project, the "examples" are 3D CAD models where every surface (face) has been annotated with its machining feature type (e.g., "this face is the wall of a through-hole").

The problem is:

| Challenge | Why It's Hard |
|-----------|---------------|
| **Real CAD labels are scarce** | A human engineer must manually inspect every face of every 3D model and assign a label. This is extremely tedious. |
| **Neural networks are data-hungry** | A Graph Transformer with millions of parameters needs hundreds of thousands of labelled graphs to generalise well. |
| **Variety is critical** | The model must see holes, slots, pockets, steps, chamfers, threads, and more — in many sizes, orientations, and combinations. |

The solution is **synthetic data generation**: we write scripts that _programmatically_ create 3D models with known labels. Because the script created the model, it already knows which face is which feature — no manual labelling needed.

This project goes even further: it takes **real STEP files** (which are simple shapes), opens them in SolidWorks, and **programmatically adds complex features** (threads, engraved text) on top of them. This gives us models that are more realistic than pure random generation.

---

## 2. What Is a CAD Model and What Is a STEP File?

### CAD (Computer-Aided Design)
CAD software (SolidWorks, AutoCAD, CATIA, Fusion 360, etc.) is what engineers use to design 3D objects digitally before they are manufactured. Think of it as the 3D equivalent of drawing blueprints, except on a computer.

### STEP File (.step / .stp)
STEP (Standard for the Exchange of Product Data) is a **universal file format** for 3D CAD models. It's an international standard (ISO 10303) that allows models created in one CAD tool to be opened in another.

Think of a STEP file like a PDF for 3D objects — it preserves the exact mathematical shape regardless of which software created it.

**In this project:** The starting point for data generation is a folder full of STEP files. Each STEP file contains one 3D part (e.g., a block with some holes, a bracket, a gear housing).

---

## 3. B-Rep: The Mathematical DNA of a 3D Shape

There are many ways a computer can represent a 3D shape:

| Method | How It Works | Precision | Example |
|--------|-------------|-----------|---------|
| **Voxels** | 3D pixels (tiny cubes on a grid) | Low (blocky, like Minecraft) | Medical imaging |
| **Mesh (STL)** | Millions of tiny triangles covering the surface | Medium (approximation) | 3D printing, video games |
| **B-Rep** | Exact mathematical surfaces & curves | **Perfect** (exact equations) | CAD/CAM (this project) |

### What Is B-Rep (Boundary Representation)?

B-Rep describes a 3D shape using the **boundaries** (surfaces) that enclose it. Imagine a cardboard box:

```
         Vertex (corner point)
            o────────────o
           /|            /|
          / |           / |
         o────────────o  |   ← Edge (line where two faces meet)
         |  |         |  |
         |  o─────────|──o
         | /          | /
         |/           |/
         o────────────o

    Each flat side = one Face (with a mathematical equation: "plane")
    Each seam      = one Edge (with a mathematical equation: "line")
    Each corner    = one Vertex (a 3D point: x, y, z)
```

In B-Rep:
- **Faces** are surfaces. A face has a mathematical surface equation. Simple faces are flat planes or cylinders. Complex faces can be freeform curved surfaces (NURBS).
- **Edges** are curves where two faces meet. An edge has a mathematical curve equation (straight line, circular arc, ellipse, spline, etc.).
- **Vertices** are points where edges meet. A vertex is just an (x, y, z) coordinate.

### Why B-Rep Matters for This Project

Because B-Rep preserves the **exact mathematical intent** of the design (a cylinder remains a perfect cylinder, not an approximation made of triangles), the AI can learn to recognise geometric features with high precision. A "through-hole" is always a perfect cylinder, not "roughly round".

### The Face as the Unit of Classification

Our AI's job is **face segmentation**: for every face in a B-Rep model, predict which machining feature class it belongs to. This is like image segmentation (labelling every pixel), except we label every face.

---

## 4. The 25 Machining Feature Classes

The model classifies each face into one of **25 categories**. These correspond to distinct machining operations or parts of operations:

| Class ID | Feature Name | Description |
|----------|-------------|-------------|
| 0 | **Stock** | The raw material block (no machining needed). **~58% of all faces!** |
| 1–4 | **Through Holes** (variants) | Holes drilled all the way through a part |
| 5–8 | **Blind Holes** (variants) | Holes drilled partway, with a flat or conical bottom |
| 9–12 | **Slots** (through/blind, various) | Long rectangular channels cut into the part |
| 13–16 | **Pockets** (through/blind, variants) | Rectangular cavities (like a slot but wider) |
| 17–20 | **Steps/Shoulders** | Step-like features where material is removed from an edge |
| 21–24 | **Chamfers, Rounds, Fillets** | Angled or rounded edge treatments |

> **Key insight:** Class 0 (Stock) dominates the dataset massively. In a typical part, the "raw block" faces vastly outnumber the machined feature faces. This creates a **class imbalance** problem addressed in training (Part 2).

---

## 5. End-to-End Pipeline Overview

The full data generation pipeline has three stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE A: SHARDING & STAGING                         │
│                                                                             │
│  [Master STEP Share]  ──(Hash Sharding)──▶  [VM 1 local STEPS]             │
│   (network drive)           ↓                [VM 2 local STEPS]             │
│   all .step files       stage_shard.py       [VM 3 local STEPS]             │
│                                              [VM N local STEPS]             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              STAGE B: SYNTHETIC GEOMETRY AUGMENTATION (SolidWorks)          │
│                                                                             │
│  For each .step file on this VM:                                            │
│    1. SwOrchestrator.Cli launches SolidWorks + VBA macro                    │
│    2. Macro opens STEP file in SolidWorks                                   │
│    3. Macro adds synthetic threads & engraved text to faces                 │
│    4. Macro saves augmented model as .SLDPRT                                │
│    5. Macro exports B-Rep geometry as BrepJson (.json)                      │
│    6. Orchestrator monitors heartbeats, restarts on crash                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              STAGE C: GRAPH COMPILATION (Python / PyTorch Geometric)        │
│                                                                             │
│  json_to_brepmfr_pyg.py converts each BrepJson file:                       │
│    1. Maps faces → graph nodes, touching boundaries → graph edges           │
│    2. Extracts UV grids, edge curves, surface types, areas, angles          │
│    3. Computes shortest paths between all face pairs (A1)                   │
│    4. Computes D2 distance & dihedral angle histograms (A2)                 │
│    5. Records edge-index chains along shortest paths (A3)                   │
│    6. Saves everything as a single .pt file (PyTorch Geometric Data)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

Let's examine each stage in excruciating detail.

---

## 6. Stage A: Sharding & Staging (Distributing Files Across VMs)

### The Problem
SolidWorks is a heavyweight Windows-only desktop application. Processing one STEP file can take seconds to minutes. To process tens of thousands of files in a reasonable time, we need **multiple machines working in parallel**.

### The Solution: Deterministic Hash Sharding

The script [`stage_shard.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/SwOrchestrator/stage_shard.py) distributes files across VMs without any coordination between them.

#### How It Works (Step by Step)

1. **Central storage:** All STEP files live on a shared network drive (e.g., `\\fileserver\steps_master`).

2. **Each VM knows its identity:** VM 0 out of 4 total, VM 1 out of 4, etc. These are passed as `--shard-index` and `--total-shards`.

3. **Hashing determines assignment:** For each filename:
   ```python
   def shard_of(name: str, total: int) -> int:
       h = hashlib.md5(name.lower().encode("utf-8")).hexdigest()
       return int(h, 16) % total
   ```
   - Take the filename (e.g., `"bracket_00123.step"`)
   - Convert to lowercase
   - Compute its MD5 hash (a 128-bit fingerprint)
   - Take `hash mod total_shards` → this gives a number 0, 1, 2, or 3
   - If that number matches this VM's shard index, this VM processes the file

4. **Local staging:** Assigned files are hard-linked (or copied if on different disk volumes) to a local directory (`C:\ThreadRecognition\STEPS`).

5. **Idempotent:** If a file already exists locally, it's skipped. This means you can restart the script without re-downloading anything.

#### Why MD5 Hash Sharding?

- **Deterministic:** The same filename always goes to the same VM, no matter how many times you run the script.
- **No coordination needed:** VMs don't need to talk to each other. Each independently computes which files belong to it.
- **Balanced:** MD5 hashes are uniformly distributed, so each VM gets roughly the same number of files.

#### The Critical Limitation

**Do NOT change `TOTAL_SHARDS` mid-batch!** Changing the number of VMs reassigns every file to a potentially different shard. A file previously on VM 0 might now belong to VM 2. This breaks the local cache and forces re-processing.

---

## 7. Stage B: Synthetic Geometry Augmentation (SolidWorks)

This is the most complex and infrastructure-heavy stage. It involves three components working together:

### 7.1 The SwOrchestrator (C# Application)

**File:** [`SwOrchestrator/`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/SwOrchestrator)

SolidWorks was never designed to run unattended on thousands of files. It will inevitably:
- Leak memory until the system runs out of RAM
- Freeze on corrupt or complex geometry
- Pop up modal dialog boxes that block execution

The **SwOrchestrator** is a custom C# wrapper that "babysits" SolidWorks:

#### Launch Mechanism
```
SwOrchestrator.Cli.exe → launches → SLDWORKS.exe /m ThreadCreationScript8.swp
```

The `/m` flag tells SolidWorks to immediately execute the specified VBA macro upon startup.

#### Heartbeat Monitoring
While the macro runs, it writes a "heartbeat" to a file (`heartbeat.txt`):
```
2026-05-19 17:33:14|var_start_3|C:\ThreadRecognition\STEPS\00000055.stp
```

The orchestrator reads this file every 5 seconds (configurable via `--poll-interval`).

- **Stall Timeout (default 900s = 15 min):** If the heartbeat hasn't been updated for this long, the orchestrator assumes SolidWorks is frozen.
- **Startup Grace (default 300s = 5 min):** Extra time allowed for the very first file (SolidWorks takes a while to boot).

#### Crash Recovery
When a stall or crash is detected:
1. Kill the SolidWorks process tree: `taskkill /F /IM sldworks.exe /T`
2. Record the problem file in `skip_files.txt`
3. Wait 8 seconds for RAM to clear (configurable via `--cooldown`)
4. Relaunch SolidWorks with the same macro
5. The macro's resume logic picks up where it left off

#### Two Flavours

| Variant | Purpose | Interface |
|---------|---------|-----------|
| `SwOrchestrator.Gui.exe` | Interactive monitoring on one VM | WPF window with progress bar |
| `SwOrchestrator.Cli.exe` | Automated use with Jenkins | Console output with exit codes |

The orchestrator's tunable settings:

| Setting | Default | Purpose |
|---------|---------|---------|
| `--stall-timeout` | 900s | Hang detection threshold |
| `--startup-grace` | 300s | Extra time for first file |
| `--poll-interval` | 5s | Heartbeat check frequency |
| `--cooldown` | 8s | Pause between kill and relaunch |
| `--crash-threshold` | 3 | Auto-blacklist after N crashes |
| `--max-restarts` | 10000 | Hard ceiling on restart loop |

### 7.2 The VBA Macro (`ThreadCreationScript8.bas`)

**File:** [`ThreadCreationScript8.bas`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/SwOrchestrator/ThreadCreationScript8.bas)

This is the heart of data generation. It runs **inside SolidWorks** (as a VBA macro compiled into a `.swp` binary) and performs the actual synthetic augmentation.

#### What the Macro Does (Step by Step)

**1. Load Skip List**
```vba
Private Sub LoadSkipList()
    ' Reads skip_files.txt into a dictionary
    ' Files listed here (written by the orchestrator after crashes) are bypassed
```

**2. Enumerate and Process Each STEP File**
```vba
Sub main()
    ProcessAllStepFilesInFolder sFolder, "*.step"
    ProcessAllStepFilesInFolder sFolder, "*.stp"
End Sub
```

**3. For Each STEP File, Create 6 Variations**
```vba
Private Sub ProcessStepFile(ByVal sStepPath As String)
    ' Skip if in blacklist
    ' Skip if all 6 variations already exist (resume-on-crash)
    For v = 1 To NUM_THREAD_VARIATIONS
        ' Skip this variation if its output .SLDPRT already exists
        ' Open the STEP file fresh each time
        ' Add synthetic features based on variation index
        ' Save as .SLDPRT
        ' Export B-Rep JSON
    Next v
End Sub
```

Each STEP file gets **6 variations** with different thread types:

| Variation | Thread Type | Right-Handed? | Thread Size | Fraction |
|-----------|------------|---------------|-------------|----------|
| 1 | inch die | Yes | #8-36 | 50% |
| 2 | inch die | No | #1.7500-5 | 36% |
| 3 | metric tap | Yes | M12x1.75 | 67% |
| 4 | metric tap | No | M36x2.0 | 45% |
| 5 | sp4xx bottle | Yes | SP410-L-6 | 23% |
| 6 | sp4xx bottle | No | SP425-L-12 | 75% |

**4. Find Candidate Edges for Threading**
```vba
vOut = swAppInternal.AITrainUtils(AI_CMD_THREAD_RIM_EDGES, inArgs)
```
This uses a **SolidWorks internal AI utility** to find circular "rim" edges on the model — edges that form the mouth of a cylindrical hole or boss, perfect for threading.

**5. Create Sweep Thread Features**
For each rim edge found:
- Get the cylinder radius and axial length
- Apply the thread profile from the variation table
- Use SolidWorks FeatureManager to create a Sweep Thread on the edge

**6. Export B-Rep as JSON**
```vba
swAppInternal.BaselineOutputCmd 100040, BREP_JSON_OUT
```
This SolidWorks internal command exports the **entire B-Rep topology and geometry** of the augmented model as a JSON file. The JSON contains:
- All faces with UV-grid samples, surface types, areas, labels
- All edges with curve samples, types, lengths, angles, convexity
- Face-pair pairwise distance and angle histograms (`face_pairs`)

**7. Write Heartbeats Throughout**
```vba
WriteHeartbeat sStepPath, "var_start_3"    ' Before each variation
WriteHeartbeat sStepPath, "var_done_3"     ' After each variation
WriteHeartbeat sStepPath, "file_done"      ' After all 6 variations
```

### 7.3 The BrepJson Format

The JSON output by SolidWorks contains three main sections:

#### `faces` Array
Each face object contains:
```json
{
  "id": 42,                          // Unique face ID
  "uv": [x1,y1,z1,nx1,ny1,nz1,m1, ...],  // Flattened 5×5×7 UV-grid
  "z": 2,                            // Surface type (0=plane, 1=cylinder, 2=cone, etc.)
  "y": 0.00345,                      // Face area in square meters
  "l": 3,                            // Number of boundary loops
  "a": 4,                            // Adjacency count
  "label": 7                         // Machining feature class (0-24)
}
```

**UV-Grid explained:** Imagine unfolding a face's surface into a flat rectangle and sampling a 5×5 grid of points on it. At each grid point, you record:
- `(x, y, z)`: the 3D position of that point
- `(nx, ny, nz)`: the surface normal vector at that point (which direction the surface "faces")
- `mask`: a trimming mask (is this grid point actually on the face, or outside its boundary?)

This gives the neural network a "fingerprint" of the face's shape.

#### `edges` Array
Each edge object contains:
```json
{
  "nf": [42, 55],                    // IDs of the two faces this edge separates
  "pt": [x1,y1,z1,tx1,ty1,tz1,a1, ...],  // Flattened 5×7 curve samples
  "t": 1,                            // Curve type (0=line, 1=circle, 2=ellipse, etc.)
  "l": 0.0523,                       // Edge length in meters
  "a": 1.5708,                       // Dihedral angle between the two adjacent faces
  "c": 1                             // Convexity (0=flat, 1=convex, 2=concave)
}
```

**Curve samples explained:** Sample 5 points along the edge curve. At each point, record:
- `(x, y, z)`: the 3D position
- `(tx, ty, tz)`: the tangent direction (which way the curve goes at that point)
- `angle`: a directed angle parameter

#### `face_pairs` Array (A2 Data)
For every pair of faces, a set of geometric descriptors:
```json
{
  "face_pair": [42, 55],
  "d2": [0.001, 0.003, ...],         // 64-bin D2 distance histogram
  "a3": [0.002, 0.005, ...],         // 64-bin dihedral angle histogram (face 42 → 55)
  "a3_1": [0.003, 0.004, ...]        // 64-bin dihedral angle histogram (face 55 → 42)
}
```

These are **shape distribution descriptors** (borrowed from 3D shape retrieval). They capture the geometric relationship between two faces, even if they don't touch.

---

## 8. Stage C: Graph Compilation — JSON to PyTorch Geometric `.pt`

**File:** [`scripts/inference/json_to_brepmfr_pyg.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/inference/json_to_brepmfr_pyg.py)

This Python script is the bridge between the SolidWorks world (CAD geometry in JSON) and the machine learning world (PyTorch tensors in a `.pt` graph file).

### 8.1 What Happens (Step by Step)

**Step 1: Parse JSON & Build Node/Edge Mappings**
```python
sorted_faces = sorted(faces, key=lambda x: int(x["id"]))
face_id_to_node = {int(f["id"]): i for i, f in enumerate(sorted_faces)}
```
- Sort faces by their CAD ID
- Assign consecutive node indices (0, 1, 2, ...) to each face
- This creates the mapping: **CAD face ID → graph node index**

**Step 2: Build Adjacency from Edges**
```python
for e in edges:
    f1, f2 = int(e["nf"][0]), int(e["nf"][1])
    u, v = face_id_to_node[f1], face_id_to_node[f2]
    adj[u].append(v)
    adj[v].append(u)  # Undirected: both directions
```
- Each edge in the JSON connects two faces
- In the graph, this becomes a **bidirectional** connection

**Step 3: Create Directed Edge List**
```python
for i in range(N):
    for neighbor in sorted(adj[i]):
        final_src.append(i)
        final_dst.append(neighbor)
```
- For each node, enumerate all its neighbors in sorted order
- This creates the COO (Coordinate) format edge index: `edge_index[0] = sources`, `edge_index[1] = destinations`

**Step 4: Extract Node Features (Face UV-Grids)**
```python
node_x[ni] = _reshape_face_uv(f["uv"])  # → shape (5, 5, 7)
```
- The flat UV array from JSON is reshaped into a 5×5 grid with 7 channels per point
- Also extract: face type (`z`), face area (`y`), face loops (`l`), face adjacency count (`a`), label (`label`)

**Step 5: Extract Edge Features (Curve Samples)**
```python
raw_pts = _reshape_edge_pt(eobj["pt"])  # → shape (5, 7)
```
- If the edge direction matches the JSON ordering, use as-is
- If reversed, **flip the sample order and negate tangent vectors** (because the curve direction is opposite)
- Apply angle normalisation: wrap channel 7 (angle) to `[-π, π)`:
  ```python
  edge_x[:, :, 6] = (edge_x[:, :, 6] + pi) % two_pi - pi
  ```

**Step 6: Compute A1 — Shortest Path Distances**
Using BFS from every node to every other node:
```python
spatial_pos, edges_path = _compute_shortest_paths_edge_indices(...)
```
- `spatial_pos[i][j]` = number of hops (edges) in the shortest graph path from node i to node j
- `edges_path[i][j][k]` = the index of the k-th edge along that shortest path
- This is an **O(N² + N·E)** computation — can be parallelised across source nodes with `--shortest_path_workers`

**Step 7: Compute A2 — Pairwise Shape Descriptors**
```python
d2_distance, angle_distance = _build_a2_tensors(data, face_id_to_node, N)
```
- For every pair of faces, look up the D2 distance histogram and angle histogram from the JSON's `face_pairs`
- Store as dense tensors of shape `[N, N, 64]`
- Note: D2 is symmetric, but the angle histogram is **asymmetric** (the angle from face A→B is different from B→A)

**Step 8: Assemble PyG Data Object**
```python
pyg = PYGGraph()
pyg.node_data = ...       # [N, 5, 5, 7] face UV-grids
pyg.edge_data = ...       # [E, 5, 7] edge curve samples
pyg.face_type = ...       # [N] int, surface type
pyg.face_area = ...       # [N] float, area
pyg.label_feature = ...   # [N] int, machining feature class
pyg.spatial_pos = ...     # [N, N] int, shortest path distance
pyg.edge_path = ...       # [N, N, max_dist] int, edge indices along path
pyg.d2_distance = ...     # [N, N, 64] float, D2 histograms
pyg.angle_distance = ...  # [N, N, 64] float, angle histograms
pyg.edge_index = ...      # [2, E] long, COO edge list
```

**Step 9: Save as `.pt`**
```python
torch.save(pyg, out_pt)
```

---

## 9. The `.pt` File: Every Tensor Explained

Each `.pt` file is a single PyTorch Geometric `Data` object. Here is **every attribute** it contains, what it means, its shape, and how the neural network uses it:

### Node (Face) Features

| Attribute | Shape | Type | Description |
|-----------|-------|------|-------------|
| `node_data` | `[N, 5, 5, 7]` | float32 | UV-grid samples on each face. 7 channels = `[x, y, z, nx, ny, nz, mask]`. Processed by `SurfaceEncoder` (2D CNN). |
| `face_type` | `[N]` | int | Surface type integer. 0=padding, 1=plane, 2=cylinder, 3=cone, 4=sphere, 5=torus, 6=bspline, 7=other. Fed to an `nn.Embedding(8, 32)`. |
| `face_area` | `[N]` | float | Area of each face in m². Fed to a `NonLinear(1, 32)` encoder. |
| `face_loop` | `[N]` | int | Number of boundary loops. A simple face has 1 loop; a face with a hole in it has 2. Fed to `nn.Embedding(256, 32)`. |
| `face_adj` | `[N]` | int | Adjacency count (how many edges touch this face). |
| `node_degree` | `[N]` | long | Out-degree of each node in the directed graph (same as number of directed arcs leaving this node). Fed to `nn.Embedding(128, 32)` as the "degree encoding" in Graphormer. |
| `label_feature` | `[N]` | int | Ground-truth machining feature class (0–24). Used as the training target. |

### Edge (Curve) Features

| Attribute | Shape | Type | Description |
|-----------|-------|------|-------------|
| `edge_data` | `[E, 5, 7]` | float32 | Curve samples along each directed edge arc. 7 channels = `[x, y, z, tx, ty, tz, angle]`. The angle channel has been wrapped to `[-π, π)`. Processed by `CurveEncoder` (1D CNN). |
| `edge_type` | `[E]` | int | Curve type. 0=padding, 1=line, 2=circle, 3=ellipse, 4=bspline, 5=other. Fed to `nn.Embedding(6, n_heads)`. |
| `edge_len` | `[E]` | float | Edge length in meters. Fed to `NonLinear(1, n_heads)`. |
| `edge_ang` | `[E]` | float | Scalar dihedral angle at this edge, wrapped to `[-π, π)`. Fed to `NonLinear(1, n_heads)`. |
| `edge_conv` | `[E]` | int | Convexity. 0=flat/unknown, 1=convex (peak), 2=concave (valley). Fed to `nn.Embedding(3, n_heads)`. |
| `edge_index` | `[2, E]` | long | COO edge list. `edge_index[0]` = source nodes, `edge_index[1]` = destination nodes. Standard PyG format. |

### Pairwise / Structural Tensors

| Attribute | Shape | Type | Description |
|-----------|-------|------|-------------|
| `spatial_pos` (A1) | `[N, N]` | int | Shortest path distance between every pair of faces in the face-adjacency graph. Fed to `nn.Embedding(64, n_heads)` → attention bias. |
| `edge_path` (A3) | `[N, N, max_dist]` | int | For each face pair, the sequence of edge indices along the shortest path. -1 = padding. Used by the multi-hop edge encoding in `GraphAttnBias`. |
| `d2_distance` (A2) | `[N, N, 64]` | float32 | D2 shape distribution histogram (symmetric). Fed to `NonLinear(64, n_heads)` → attention bias. |
| `angle_distance` (A2) | `[N, N, 64]` | float32 | Dihedral angle histogram (asymmetric). Fed to `NonLinear(64, n_heads)` → attention bias. |
| `attn_bias` | `[N+1, N+1]` | float32 | Base attention bias (all zeros for most profiles). The +1 is for the virtual global `[CLS]` node. |

### Metadata

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_id` | int | Numeric ID parsed from the filename stem. |
| `has_a1` | bool | Whether this graph contains A1 (spatial_pos). |
| `has_a2` | bool | Whether this graph contains A2 (d2/angle_distance). |
| `has_a3` | bool | Whether this graph contains A3 (edge_path). |
| `inference_profile` | str | `"full"`, `"no_a2"`, or `"lite"`. |
| `store_float16` | bool | Whether node_data/edge_data were stored in half precision. |

---

## 10. The Three Inference Profiles: `full`, `no_a2`, `lite`

Not all tensors are always needed. The conversion script supports three profiles to trade off **disk space** and **conversion speed** against model accuracy:

| Profile | A1 (spatial_pos) | A2 (d2/angle) | A3 (edge_path) | `.pt` Size | Use Case |
|---------|:-:|:-:|:-:|------------|----------|
| `full` | ✅ | ✅ | ✅ | Largest | Best accuracy. Full pairwise tensors. |
| `no_a2` | ✅ | ❌ | ✅ | Medium | Saves the dense `[N,N,64]` A2 tensors. Still computes BFS. |
| `lite` | ❌ | ❌ | ❌ | Smallest | No BFS, no pairwise tensors. Fastest conversion. Uses only local edge features. |

To convert with a specific profile:
```powershell
python scripts/inference/json_to_brepmfr_pyg.py \
  --json_dir Z:/input_json \
  --pt_out_dir Z:/out_pyg \
  --inference_profile no_a2
```

---

## 11. Dataset Splits and Split Lists

The model needs separate data for training, validation, and testing. Splits are controlled by **text files** listing the graph stems (filenames without extension):

### For Stage 1 (Source-only Training)

| File | Purpose |
|------|---------|
| `train.txt` | List of graph stems for training |
| `val.txt` | List of graph stems for validation |
| `test.txt` | List of graph stems for testing |

### For Stage 2 (Transfer Learning)

| File | Purpose |
|------|---------|
| `s_train.txt` | Source domain training stems |
| `s_val.txt` | Source domain validation stems |
| `s_test.txt` | Source domain test stems |
| `t_train.txt` | Target domain training stems |
| `t_val.txt` | Target domain validation stems |
| `t_test.txt` | Target domain test stems |

The dataset class [`CADSynth`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/dataset.py#L133) loads files by:
1. Reading the split list (e.g., `train.txt`)
2. Recursively scanning the dataset directory for `.pt` files
3. Keeping only `.pt` files whose stem appears in the split list

---

## 12. Collation: How Individual Graphs Become a Batch

**File:** [`data/collator.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/collator.py)

Individual graphs have different numbers of nodes and edges. To feed them to the GPU in a batch, they need to be padded to the same size. The collator does this:

### What Padding Means

If batch has 3 graphs with 10, 15, and 8 nodes respectively:
- All are padded to 15 nodes (the maximum in this batch)
- A `padding_mask` tensor marks which positions are real (`False`) vs. padded (`True`)
- Padded positions get zero features and `-inf` attention bias (so the Transformer ignores them)

### The `collator()` Function

```python
def collator(items, multi_hop_max_dist, spatial_pos_max):
    # 1. Check that all items have the same profile (all A2, or all no-A2, etc.)
    # 2. For each graph: extract all tensors into a row
    # 3. Find max_node_num and max_edge_num across the batch
    # 4. Pad each tensor to the max size
    # 5. Concatenate into batch tensors
    # 6. Return as a dictionary
```

### The `collator_st()` Function (Stage 2)

For domain adaptation, each batch contains **paired source and target graphs**:
```python
def collator_st(items, multi_hop_max_dist, spatial_pos_max):
    flat = [pair["source_data"] for pair in items] + [pair["target_data"] for pair in items]
    return collator(flat, multi_hop_max_dist, spatial_pos_max)
```

Source graphs come first in the batch, target graphs second. The model uses `chunk(2, dim=0)` to split them back apart.

---

## 13. Infrastructure: Jenkins, VMs, and the "Session 0" Problem

### Why Jenkins?

Jenkins is a CI/CD automation server that can schedule and monitor jobs across multiple machines. Here it orchestrates the data generation pipeline across a fleet of Windows VMs.

### The "Session 0" Problem — A Critical Gotcha

**Windows Session 0** is a special non-interactive session where Windows Services run. It has no visible desktop, no GUI capability.

**The Problem:** If the Jenkins agent is installed as a standard Windows Service, it runs in Session 0. When it tries to launch SolidWorks, SolidWorks needs a GUI → instant crash.

**The Solution (per VM):**
1. Create a dedicated build user account
2. Enable auto-login for that user (so the desktop session starts automatically on boot)
3. Use Windows Task Scheduler to start the Jenkins agent "At log on" — this runs it in the interactive desktop session (Session 1+)
4. SolidWorks can now open normally with full GUI access

### The Jenkins Pipeline Flow

```
Jenkins Controller
    │
    ├── VM 1 (agent: solidworks-vm1)
    │     ├── stage_shard.py --shard-index 0 --total-shards 4
    │     └── SwOrchestrator.Cli.exe --steps ... --macro ...
    │
    ├── VM 2 (agent: solidworks-vm2)
    │     ├── stage_shard.py --shard-index 1 --total-shards 4
    │     └── SwOrchestrator.Cli.exe --steps ... --macro ...
    │
    ├── VM 3 (agent: solidworks-vm3)
    │     ├── stage_shard.py --shard-index 2 --total-shards 4
    │     └── SwOrchestrator.Cli.exe --steps ... --macro ...
    │
    └── VM 4 (agent: solidworks-vm4)
          ├── stage_shard.py --shard-index 3 --total-shards 4
          └── SwOrchestrator.Cli.exe --steps ... --macro ...
```

---

## 14. Known Limitations and Gotchas

### Infrastructure Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| **Windows-only** | Can't use Linux servers, Docker, or cloud GPUs for data gen | Maintain dedicated Windows VMs |
| **Single-threaded** | SolidWorks processes one file at a time per VM | Scale horizontally with more VMs |
| **VBA macro is binary `.swp`** | Can't diff/review macro changes in Git | Keep `.bas` source alongside, manually re-import |
| **Shard remapping on resize** | Changing VM count invalidates local caches | Finish current batch before changing fleet size |
| **Silent dialog freezes** | SolidWorks pops up modal dialogs that block the macro | Suppress via registry keys; orchestrator uses stall timeout |
| **Network copy bottleneck** | Combining outputs from multiple VMs requires reliable network copy | Use `robocopy` with retry, verify checksums |

### Data Quality Gotchas

| Gotcha | What Happens | How To Detect |
|--------|-------------|---------------|
| **Empty graphs** | Some STEP files produce zero faces after conversion | Use `--drop_invalid_graphs` flag when training |
| **Out-of-range labels** | Macro labels outside 0–24 range | Dataset `load_one_graph()` prints warnings |
| **Mixed inference profiles in one batch** | Collator crashes if some `.pt` have A2 and some don't | Use `--pt_subdir` to isolate different profiles |
| **Extremely large graphs** | Memory explosion in attention (O(N²)) | Use `--max_graph_nodes` to cap graph size |

---

## 15. File-by-File Reference Map

### Data Generation Pipeline

| File | Location | Role |
|------|----------|------|
| [`stage_shard.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/SwOrchestrator/stage_shard.py) | SwOrchestrator/ | Hash sharding to distribute STEP files across VMs |
| [`ThreadCreationScript8.bas`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/SwOrchestrator/ThreadCreationScript8.bas) | SwOrchestrator/ | VBA macro source — thread augmentation & BrepJson export |
| [`threadplustextgen8.swp`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/threadplustextgen8.swp) | data_understanding_files/ | Compiled binary of the VBA macro (run by SolidWorks) |
| [`SwOrchestrator.sln`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/SwOrchestrator/SwOrchestrator.sln) | SwOrchestrator/ | C# solution for the orchestrator tool |
| [`Jenkinsfile`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/SwOrchestrator/Jenkinsfile) | SwOrchestrator/ | Declarative Jenkins pipeline for parallel processing |

### Graph Compilation

| File | Location | Role |
|------|----------|------|
| [`json_to_brepmfr_pyg.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/inference/json_to_brepmfr_pyg.py) | scripts/inference/ | JSON → `.pt` converter (the main graph compiler) |
| [`json_to_brepmfr_pyg_optimized.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/inference/json_to_brepmfr_pyg_optimized.py) | scripts/inference/ | Performance-optimised variant of the converter |
| [`convert_dgl_bins_to_pyg.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/inference/convert_dgl_bins_to_pyg.py) | scripts/inference/ | Legacy DGL `.bin` → PyG `.pt` converter |

### Data Loading

| File | Location | Role |
|------|----------|------|
| [`dataset.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/dataset.py) | data/ | `CADSynth` (Stage 1) and `TransferDataset` (Stage 2) classes |
| [`collator.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/collator.py) | data/ | Batch collation with padding, masking, and edge index offsetting |
| [`utils.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/utils.py) | data/ | Random rotation augmentation for training |

---

> **Next:** Read [Part 2: Model Training & Inference](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/PART2_model_training_inference_deep_dive.md) for the complete explanation of the neural network architecture, training stages, domain adaptation, and inference pipeline.
