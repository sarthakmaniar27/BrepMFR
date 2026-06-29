# BrepMFR Project Explanation & Context Guide

Welcome! This document is a comprehensive, start-to-finish explanation of this project. It is designed for someone who is completely new to the worlds of **CAD**, **SolidWorks**, **GNNs (Graph Neural Networks)**, and **Domain Adaptation**. 

We do not assume you know any advanced machine learning or mechanical engineering terminology. Every complex concept is explained with simple analogies.

---

## Table of Contents
1. [The Goal: What is Machining Feature Recognition?](#1-the-goal-what-is-machining-feature-recognition)
2. [CAD & B-Rep: How Computers Represent 3D Objects](#2-cad--b-rep-how-computers-represent-3d-objects)
3. [Graphs & Graph Neural Networks (GNNs): Modeling Shapes as Networks](#3-graphs--graph-neural-networks-gnns-modeling-shapes-as-networks)
4. [The BrepMFR Network Architecture](#4-the-brepmfr-network-architecture)
5. [Domain Adaptation: Bridging the Gap from Simulation to Reality](#5-domain-adaptation-bridging-the-gap-from-simulation-to-reality)
6. [Overcoming the Key Challenges (Class Imbalance & Label Shift)](#6-overcoming-the-key-challenges-class-imbalance--label-shift)
7. [Codebase Map: What Files Do What?](#7-codebase-map-what-files-do-what)
8. [Quick Start: Running Training & Inference](#8-quick-start-running-training--inference)

---

## 1. The Goal: What is Machining Feature Recognition?

Before looking at code, let's understand the real-world manufacturing problem this project solves.

### The Problem: Design vs. Manufacturing
* **CAD (Computer-Aided Design):** Engineers use software like **SolidWorks**, AutoCAD, or CATIA to design 3D models of physical parts (e.g., a car engine component, a bracket, or a gear).
* **CAM (Computer-Aided Manufacturing):** Factories use computer-controlled machines (like CNC mills, drills, and lathes) to cut and carve the designed parts out of solid blocks of metal or plastic.
* **The Gap:** A CAD model is just a 3D shape. It doesn't tell the CNC machine *how* to carve it. A human machinist has to look at the 3D model and manually identify features like:
  * *"Here is a 10mm round hole; we need to drill it."*
  * *"Here is a rectangular slot; we need to mill it."*
  * *"Here is a flat outer face; we don't need to touch it."*

This manual translation process is slow, expensive, and prone to errors.

### The Solution: Machining Feature Recognition (MFR)
**MFR** is the process of using AI to automatically inspect a 3D CAD model and identify these "machining features" (holes, slots, pockets, steps, etc.). 

If the AI can label every surface of the 3D model, CAM software can automatically write the instructions (G-code) for the CNC machines, fully automating the path from design to manufacturing.

```
+------------+        +---------------+        +----------------------+
|  3D CAD    |  --->  |   BrepMFR     |  --->  | Automatically labeled|
| Model File |        | (Our GNN AI)  |        |  Machining Features  |
+------------+        +---------------+        +----------------------+
```

---

## 2. CAD & B-Rep: How Computers Represent 3D Objects

In order to feed a 3D model into an AI, we need to understand how the file represents the shape. There are three main ways computers store 3D shapes:

1. **Voxels:** Think of this as 3D pixels (like Minecraft blocks). They are easy for AI to process but are blocky and lack precision.
2. **Meshes (STL files):** The shape is covered by millions of tiny triangles (like in video games or 3D printing). While detailed, it loses the underlying mathematical design (e.g., a cylinder becomes hundreds of flat triangles).
3. **B-Rep (Boundary Representation):** This is the gold standard for CAD. It represents 3D shapes using their **exact mathematical boundaries**. This is the representation used in this project.

### Understanding B-Rep (Boundary Representation)
Imagine a hollow cardboard box. B-Rep describes the box using three fundamental geometric elements:
* **Faces (Surfaces):** The flat sides of the box. In B-Rep, a face has a mathematical surface equation (e.g., a flat plane, a cylinder, a sphere, or a complex curved surface called a NURBS).
* **Edges (Curves):** The seams where two faces meet. An edge has a mathematical curve equation (e.g., a straight line or a circular arc).
* **Vertices (Points):** The corner points where edges meet.

```
       Vertex (Point)
          o-----------o
         /           /|
        /  Face     / |
       o-----------o  |  Edge (Line where two faces meet)
       | (Surface) |  o
       |           | /
       |           |/
       o-----------o
```

### Face Segmentation
In this project, our goal is **Face Segmentation**. This means we want to classify **every single face** in a B-Rep model into a category. 
For example:
* Face 1 is labeled **Stock** (class 0) — the raw metal block.
* Face 2, 3, and 4 are labeled **Through Hole** — a hole that goes all the way through.
* Face 5 and 6 are labeled **Pocket** — a cavity cut into the metal.

---

## 3. Graphs & Graph Neural Networks (GNNs): Modeling Shapes as Networks

An AI network cannot directly read a 3D STEP file (the standard format for B-Rep CAD models). We must convert it into a structure the AI understands. 

Since B-Rep models consist of faces touching each other along edges, they are naturally represented as **Graphs**.

### What is a Graph?
In math and computer science, a **Graph** is a network made of:
* **Nodes (Vertices):** The points/objects in the network.
* **Edges (Connections):** The lines connecting the nodes.

*(Note: Don't confuse "Graph Edges" with "B-Rep Edges". In our B-Rep Graph, the graph nodes correspond to CAD faces, and the graph edges correspond to CAD edges).*

### Mapping B-Rep to a Graph
Here is the core mapping concept:
1. Each **Face** in the CAD model becomes a **Node** in our graph.
2. If two faces touch each other along a CAD **Edge**, we draw a **Graph Edge** between their corresponding nodes.

#### Simple Example: A 3D Wedge
Imagine a 3D wedge shape. It has:
* Face A (top slope)
* Face B (vertical back)
* Face C (flat bottom)
* Face D & E (triangular sides)

If Face A touches Face B, Face C, Face D, and Face E, then in the graph:
* Node A will have connections (edges) to Nodes B, C, D, and E.

```
      CAD Model (Wedge)                        B-Rep Graph
         
            /|                                      [Face B]
           / |                                      /   \
  Face A  /  | Face B                              /     \
  (Slope)/   | (Back)                         [Face A]---[Face D]
        /____|                                 \     /
        Face C (Bottom)                         \   /
                                                [Face C]
```

### What is a Graph Neural Network (GNN)?
A GNN is a type of deep learning model designed to process graphs. 
* **Message Passing:** Standard neural networks look at each object in isolation. GNNs work by letting nodes "talk" to their neighbors. A node looks at its own features, pulls in the features of its connected neighbors, and updates its own understanding.
* **Why this works for CAD:** A face by itself (e.g., a flat plane) doesn't tell you much. But if a GNN knows that this flat plane is connected to four cylindrical faces that form a circular boundary, it can easily deduce: *"Aha! This flat plane is the bottom of a pocket!"*

### What is a Graph Transformer (Graphormer)?
Traditional GNNs only allow nodes to talk to their immediate physical neighbors. However, in CAD models, two faces might be geometrically related even if they don't touch (for example, two coaxial holes on opposite sides of a bracket).
* **Self-Attention (Transformer):** A Transformer allows *every* face in the model to talk to *every other* face, regardless of whether they touch.
* **Graphormer:** A specialized type of Graph Transformer that adds **structural biases** to the attention mechanism. It doesn't just let faces talk; it biases their conversation based on:
  * **Shortest Path Distance:** How many steps in the graph separate Face A and Face B?
  * **Spatial Distance:** How far apart are they in 3D space?
  * **Edge Features:** What kind of CAD edges connect them?

---

## 4. The BrepMFR Network Architecture

Let's look at how the network represents and processes the graph data. You can trace this in the code under [models/modules/brep_encoder.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/models/modules/brep_encoder.py).

### Step 1: Input Features
When a CAD model is converted to a graph, we attach features to the nodes (faces) and edges (curves):
* **Node (Face) Features:**
  * **UV-Grids:** Since faces can be curved, we sample a grid of points on each face. For each point on the grid, we record its 3D coordinate $(x,y,z)$ and its surface normal vector $(nx,ny,nz)$. This gives the model a sense of the face's curvature and shape.
  * **Face Area:** The size of the face.
  * **Face Type:** An integer representing the surface type (e.g., Cylinder, Plane, Cone, Torus).
  * **Face Loop:** Features describing the boundaries/loops of the face.
  * **Degree:** The number of neighboring faces it touches.
* **Edge Features:**
  * **Edge Curve Samples:** Sample points along the boundary curve.
  * **Edge Length:** How long the boundary curve is.
  * **Edge Type:** An integer representing the curve type (e.g., Line, Circle, Ellipse).
  * **Convexity/Concavity:** An edge can be *convex* (like the peak of a roof), *concave* (like a valley), or *flat*. This is crucial for recognizing features.

### Step 2: The Encoder (`BrepEncoder`)
The `BrepEncoder` processes these inputs in sequence:
1. **Node Feature Embedding (`GraphNodeFeature`):** Converts the face properties (surface types, areas, degrees, and UV points) into a mathematical vector of length 256 for each face.
2. **Attention Bias Construction (`GraphAttnBias`):** Uses the edge features, spatial distances, and shortest path lengths to construct a giant "bias matrix". This matrix tells the Transformer which faces are highly related.
3. **Global Virtual Node (`[CLS]` token):** We add a fake "virtual node" that connects to every face. This node aggregates global information about the entire CAD model.
4. **Graph Transformer Layers:** The 256-length face vectors are passed through multiple Transformer layers. In each layer, faces exchange information via self-attention, guided by the bias matrix.

### Step 3: Global & Local Fusion (`Attention` & `Classifier`)
* **Local Representation:** The output of the Transformer for each face ($z_{node}$).
* **Global Representation:** The output of the virtual node ($z_{graph}$), representing the whole CAD model.
* **Fusion:** An [Attention](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/models/brepseg_model.py#L72-L83) module dynamically blends $z_{node}$ and $z_{graph}$ for each face. This ensures that the classifier knows both the local shape of the face and the global context of the part.
* **Classification:** A Multi-Layer Perceptron (MLP) classifier processes this combined vector and outputs a probability distribution over the 25 classes (representing the machining feature types).

---

## 5. Domain Adaptation: Bridging the Gap from Simulation to Reality

In machine learning, models perform best when training data looks exactly like deployment data. In CAD and manufacturing, this is a major challenge.

### The Problem: Labeled Data is Scarce
* **Source Domain (CADSynth):** We can write scripts to programmatically generate hundreds of thousands of random CAD models (cylinders with holes, blocks with slots, etc.) and save them with perfect labels automatically. This is our **Source Dataset**.
* **Target Domain (MFCAD++):** Real-world designs drawn by human engineers. They are far more complex, use different shapes, and do not come with face labels. Labeling them manually is incredibly tedious. This is our **Target Dataset**.

If we train our model *only* on synthetic data, its performance will tank when we try to run it on real-world CAD files due to the **Domain Gap** (the difference in visual features and complexity).

### The Solution: Two-Stage Training
To solve this, we train in two steps:

```
[ Stage 1: Supervised Training ]
   Train on Labeled Synthetic Data (CADSynth) -> Learn general geometry
                      |
                      v
[ Stage 2: Domain Adaptation (DANN) ]
   Align Synthetic & Unlabeled Real Data -> Make features domain-invariant
```

### Unsupervised Domain Adaptation (Stage 2)
In Stage 2, we train using **unlabeled** real CAD models alongside our labeled synthetic models. We want the model to learn to extract features that look identical regardless of whether a shape is synthetic or real. 

We do this using a **DANN (Domain-Adversarial Neural Network)** framework.

#### The DANN Min-Max Game
We add a third component to our network: a **Domain Discriminator**. 
* **The Domain Discriminator:** Looks at the features produced by the encoder and tries to guess: *"Is this feature from a synthetic model (Domain 1) or a real model (Domain 0)?"*
* **The Encoder:** Tries to extract features that fool the discriminator, making synthetic and real features indistinguishable.
* **The Classifier:** Continues to learn how to categorize features using the labeled synthetic data.

#### The Gradient Reversal Layer (GRL)
How do we train the encoder to *fool* the discriminator using gradient descent? We use a GRL.
* **Forward Pass:** The GRL does nothing. It passes features from the encoder to the discriminator normally.
* **Backward Pass (Learning):** The GRL **multiplies the gradients by a negative number ($-\lambda$)**. 
  * If the discriminator learns that feature $A$ looks synthetic, it tries to update the discriminator weights to detect it better.
  * When those learning signals flow backward through the GRL, they get reversed. This forces the encoder to adjust its weights so feature $A$ looks *less* synthetic, actively fooling the discriminator!

```
                    +-----------------+
                    |  Input Graphs   |
                    +-----------------+
                             |
                             v
                    +-----------------+
                    | Feature Encoder | <----\
                    +-----------------+      |
                       /           \         | Reversed Gradients
                      /             \        | (GRL flips sign)
                     v               v       |
             +--------------+     +-----+----+----+
             |  Classifier  |     | GRL Layer     |
             +--------------+     +---------------+
             | Class Label  |     | Domain Disc.  |
             | (Hole, Slot) |     | (Src vs Tgt)  |
             +--------------+     +---------------+
```

---

## 6. Overcoming the Key Challenges (Class Imbalance & Label Shift)

Historically, this codebase suffered from a "Stage 2 Plateau" — domain adaptation wasn't helping, and sometimes it made target accuracy *worse*. The engineering team diagnosed and solved these problems.

### Challenge 1: Extreme Class Imbalance
In CAD models, a massive percentage of the surface area belongs to the raw block of metal (the "Stock", Class 0), while features like slot-bottoms or pocket-corners are tiny. 
* **The Data:** As seen in `source_train_alpha05.json`, out of ~2.18 million faces, **~1.25 million belong to Class 0 (Stock)**. That is nearly 58%!
* **The Bug:** Without correction, the model learns to predict Class 0 almost all the time because it's a cheap way to get 58% accuracy. The encoder becomes biased, and rare classes get ignored.
* **The Fix:** We calculate **inverse-frequency class weights** (giving higher importance to rare classes) and apply them to the Cross-Entropy loss during Stage 1 training. This prevents the model from ignoring rare shapes.

### Challenge 2: Label Shift
**Label Shift** occurs when the frequency of classes is completely different between the Source and Target datasets.
For example, our synthetic generator might generate a ton of pockets, but real-world target parts contain mostly holes.
* **The DANN Failure:** Plain DANN tries to force the feature distributions of source and target to overlap perfectly. If source has 80% pockets and target has 80% holes, DANN will force the encoder to map pocket features onto hole features. This is destructive and ruins accuracy!
* **The Fix: IWDAN (Importance-Weighted DANN):** 
  We compute the ratio of class frequencies between target and source ($w[c] = P_{target}(c) / P_{source}(c)$). 
  During Stage 2 training, we re-weight the synthetic source features inside the domain discriminator using this importance ratio. This makes the discriminator see a source distribution that mimics the target distribution, preventing destructive feature alignment.

### Challenge 3: Over-confident Software Predictions & Post-hoc Logit Adjustment
When models are trained on imbalanced data, their raw output probabilities (the "softmax" scores) become highly over-confident. 
* **The Fix (Logit Adjustment):** During evaluation on the target domain, we perform a mathematical shift on the model's raw outputs (logits) using target class priors:
  $$\text{Adjusted Logit} = \text{Logit} - \tau \log P_{source}(c) + \tau \log P_{target}(c)$$
  This shifts the model's predictions away from synthetic-dominant classes and toward target-dominant classes, boosting performance at evaluation time without retraining.

### Challenge 4: The GRL Ramp
If the Gradient Reversal Layer's multiplier ($\lambda$) increases too quickly, the discriminator will overpower the encoder early in training, causing the model to diverge. 
* **The Fix:** We implement a **slow, warm-start GRL ramp** tied to the actual steps per epoch. This allows the encoder to learn stable geometric features before being subjected to heavy domain-adversarial pressure.

---

## 7. Codebase Map: What Files Do What?

Here is where the important pieces of the puzzle live in this repository:

### Core Code Directories
* [data/](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/data/)
  * [dataset.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/data/dataset.py): Defines `CADSynth` (for loading source files) and `TransferDataset` (loads paired source/target samples for Stage 2).
  * [collator.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/data/collator.py): Packs individual graph files into batches. `collator_st` specifically formats Stage 2 batches by pairing source and target graphs.
* [models/](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/models/)
  * [brepseg_model.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/models/brepseg_model.py): Defines the `BrepSeg` model used in **Stage 1** training.
  * [transfer_model.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/models/transfer_model.py): Defines the `DomainAdapt` model used in **Stage 2** training (includes GRL, Domain Discriminator, and IWDAN).
  * [modules/brep_encoder.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/models/modules/brep_encoder.py): The main B-Rep Graphormer encoder module.
* [artifacts/class_weights/](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/artifacts/class_weights/)
  * `stage1/source_train_alpha05.json`: Precomputed weights to balance Stage 1 CE loss.
  * `stage2_iwdan/`: Prior files used to correct label shift in Stage 2.

### Entry Point Scripts (Root Directory)
* [segmentation.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/segmentation.py): The main runner script for training/testing **Stage 1**.
* [domain_adapt.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/domain_adapt.py): The main runner script for training/testing **Stage 2**.

### Scripts & Diagnostics
* [scripts/diagnostics/diagnose_stage1_target.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/scripts/diagnostics/diagnose_stage1_target.py): An audit script that scans a checkpoint on target data and categorizes errors (e.g. collapsed classes vs. domain shift vs. label inconsistencies).
* [scripts/inference/json_to_brepmfr_pyg.py](file:///c:/Users/RZA2/Desktop/thread_project/BrepMFR/scripts/inference/json_to_brepmfr_pyg.py): Converts raw CAD JSON files (emitted by SolidWorks macros) directly into PyTorch Geometric `.pt` graph files.

---

## 8. Quick Start: Running Training & Inference

Here are the commands you use to run these pipelines. Make sure you are in the correct environment!

### Setup Environment
This codebase is migrated to **PyTorch Geometric (PyG)**, removing the legacy DGL dependency for training.
```powershell
conda activate brep_mfr_pyg
```

### Stage 1: Supervised Training on Source
To train Stage 1 with balanced class weights (recommended to prevent Stock class dominance):
```powershell
python segmentation.py train `
  --dataset_path "Z:/Experiment6_PyG/source_dataset" `
  --class_weights_path "artifacts/class_weights/stage1/source_train_alpha05.json" `
  --max_epochs 1000 `
  --batch_size 64 `
  --num_workers 0
```
*(Note: on Windows, keeping `--num_workers 0` is the safest way to avoid memory/pagefile crashes).*

### Stage 2: Domain Adaptation (DANN + IWDAN)
To train Stage 2 by loading your balanced Stage 1 checkpoint and adapting it to the target domain:
```powershell
python domain_adapt.py train `
  --source_path "Z:/Experiment6_PyG/source_dataset" `
  --target_path "Z:/Experiment6_PyG/target_dataset" `
  --pre_train "results/stage1/ce_weighted_balanced__<run_name>/best.ckpt" `
  --iwdan `
  --iwdan_source_priors "artifacts/class_weights/stage2_iwdan/source_train_priors.json" `
  --iwdan_target_priors "artifacts/class_weights/stage2_iwdan/target_train_priors.json" `
  --max_epochs 1000 `
  --batch_size 64 `
  --num_workers 0
```

### Running Diagnostics & Logit Adjustment
To evaluate a trained Stage 2 checkpoint on the target test set and sweep over logit adjustment temperatures ($\tau$) for optimal performance:
```powershell
python scripts/diagnostics/stage2_logit_adjust_eval.py `
  --checkpoint "results/stage2/transfer_iwdan_weighted__<run_name>/best.ckpt" `
  --source_path "Z:/Experiment6_PyG/source_dataset" `
  --target_path "Z:/Experiment6_PyG/target_dataset" `
  --target_split test `
  --batch_size 32 `
  --num_workers 0 `
  --out_dir "results/diagnostics/stage2_logit_adjust_t_test"
```

---

*This guide was generated to help you onboard. If you'd like to dive deeper into a specific file or equations, just ask!*
