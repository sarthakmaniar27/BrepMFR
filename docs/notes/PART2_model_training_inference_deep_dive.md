# Part 2: Model Training & Inference — The Complete Deep Dive

> **Who is this for?** Someone with **zero experience** in Graph Neural Networks, Transformers, domain adaptation, or the Graphormer architecture. Every concept is explained from first principles with analogies, architecture diagrams, and line-by-line walkthroughs of the actual code in this repository.

---

## Table of Contents

1. [The Big Picture: What Does This Model Do?](#1-the-big-picture-what-does-this-model-do)
2. [Graphs and Graph Neural Networks (GNNs) — From Zero](#2-graphs-and-graph-neural-networks-gnns--from-zero)
3. [The Transformer and Self-Attention — Simplified](#3-the-transformer-and-self-attention--simplified)
4. [Graphormer: A Graph Transformer with Structural Bias](#4-graphormer-a-graph-transformer-with-structural-bias)
5. [The BrepMFR Network Architecture (Layer by Layer)](#5-the-brepmfr-network-architecture-layer-by-layer)
6. [Stage 1: Supervised Training on Source Data](#6-stage-1-supervised-training-on-source-data)
7. [Stage 2: Domain Adaptation (DANN + IWDAN)](#7-stage-2-domain-adaptation-dann--iwdan)
8. [The Gradient Reversal Layer (GRL) — How Fooling Works](#8-the-gradient-reversal-layer-grl--how-fooling-works)
9. [IWDAN: Handling Label Shift](#9-iwdan-handling-label-shift)
10. [Class Imbalance and Weighted Cross-Entropy](#10-class-imbalance-and-weighted-cross-entropy)
11. [Logit Adjustment: Post-Training Calibration](#11-logit-adjustment-post-training-calibration)
12. [Optimisers, Learning Rate Schedules, and Hyperparameters](#12-optimisers-learning-rate-schedules-and-hyperparameters)
13. [Training Commands — Step by Step](#13-training-commands--step-by-step)
14. [Inference Pipeline](#14-inference-pipeline)
15. [Metrics: Per-Face Accuracy, Per-Class Accuracy, IoU](#15-metrics-per-face-accuracy-per-class-accuracy-iou)
16. [Monitoring: TensorBoard, CSV Logs, and W&B](#16-monitoring-tensorboard-csv-logs-and-wb)
17. [Checkpoints and Results Directory Layout](#17-checkpoints-and-results-directory-layout)
18. [File-by-File Reference Map](#18-file-by-file-reference-map)

---

## 1. The Big Picture: What Does This Model Do?

The BrepMFR model takes a **3D CAD model** (represented as a graph of faces) and predicts a **machining feature class** for every face.

```
Input:                           Output:
┌─────────────────┐             ┌─────────────────────────────────────┐
│  B-Rep Graph    │             │  Face 0: Stock (class 0)            │
│  (.pt file)     │  ────────▶  │  Face 1: Through Hole (class 2)     │
│  N faces,       │   BrepMFR   │  Face 2: Through Hole (class 2)     │
│  E edges,       │   Network   │  Face 3: Pocket (class 14)          │
│  features       │             │  Face 4: Stock (class 0)            │
│  for each       │             │  Face 5: Slot (class 10)            │
└─────────────────┘             └─────────────────────────────────────┘
```

This is a **node classification** task on a graph: each node (face) gets assigned one of 25 classes.

---

## 2. Graphs and Graph Neural Networks (GNNs) — From Zero

### What Is a Graph?

In math/CS, a **graph** is a network of objects (**nodes**) connected by relationships (**edges**). Examples:

- **Social network:** People (nodes) connected by friendships (edges)
- **Road map:** Intersections (nodes) connected by roads (edges)
- **B-Rep CAD model:** Faces (nodes) connected by shared boundary curves (edges)

```
    [Face A] ─── [Face B]
        |   \       |
        |    \      |
    [Face C]  [Face D]
```

> **Don't confuse graph edges with CAD edges!** In our graph, the "nodes" correspond to CAD faces, and the "graph edges" correspond to CAD boundary curves where two faces meet.

### Why Use a Graph for CAD Models?

B-Rep models are **naturally graph-shaped**:
- Faces touch other faces along shared edges
- A face's identity (is it a hole? a slot?) depends on **what faces are around it**
- A flat plane by itself could be anything — but if it's surrounded by cylindrical faces in a circle, it's probably the bottom of a pocket

### What Does a GNN Do?

A GNN processes graphs by **message passing**:

1. Each node starts with its own features (e.g., surface shape, area, type)
2. In each "round", every node **collects messages** from its neighbours
3. Each node **updates its own features** based on what it received
4. After several rounds, each node's features encode information about its **entire local neighbourhood**

Think of it like a game of telephone at a dinner party: after several rounds of whispering, everyone knows about everyone else — but they know more about the people sitting close to them.

```
Round 0:    Each face knows only about itself
Round 1:    Each face also knows about its direct neighbours
Round 2:    Each face knows about neighbours-of-neighbours
Round N:    Each face has context about N-hop neighbourhood
```

### The Limitation of Standard GNNs

Standard GNNs only let nodes talk to **direct neighbours**. But in CAD:
- Two faces on opposite sides of a part might be geometrically related (e.g., coaxial holes)
- The model needs **global context** to make accurate predictions

This is where **Transformers** come in.

---

## 3. The Transformer and Self-Attention — Simplified

### The Core Idea

A **Transformer** allows **every element to attend to every other element** in the input. Unlike GNNs (which restrict communication to neighbours), Transformers have no restrictions — every face can "look at" every other face.

### Self-Attention in Plain English

Imagine a classroom where every student can ask every other student a question:

1. **Query:** "What do I want to know?" — each face produces a query vector
2. **Key:** "What can I tell others?" — each face produces a key vector
3. **Value:** "What information do I carry?" — each face produces a value vector
4. **Attention Score:** How relevant is face B's information to face A? Computed as the dot product of A's query with B's key
5. **Output:** Each face's new representation is a weighted average of all faces' values, where the weights are the attention scores

```
Face A's new features = (0.8 × Face B's value) + (0.1 × Face C's value) + (0.05 × Face D's value) + ...
                         ^--- high attention       ^--- low attention       ^--- very low
```

### Multi-Head Attention

Instead of having one attention pattern, we run **multiple "attention heads" in parallel** (32 in this project). Each head can specialise:
- Head 1 might focus on geometric similarity
- Head 2 might focus on topological distance
- Head 3 might focus on surface type matching

---

## 4. Graphormer: A Graph Transformer with Structural Bias

BrepMFR uses a **Graphormer** — a specialised Graph Transformer that adds **structural biases** to the attention mechanism. Without these biases, the Transformer would treat the faces as an unordered bag (losing the graph structure).

### Three Structural Biases (A1, A2, A3)

#### A1 — Spatial Position Encoding (Shortest Path Distance)

`spatial_pos[i][j]` = number of hops in the shortest path from face i to face j.

This is fed through `nn.Embedding(64, n_heads)` to produce a bias per attention head. Faces that are topologically close get higher attention.

```
spatial_pos = [[0, 1, 2, 3],    ← Face 0 is 0 hops from itself, 1 hop from Face 1, etc.
               [1, 0, 1, 2],
               [2, 1, 0, 1],
               [3, 2, 1, 0]]
```

#### A2 — D2 Distance & Angle Histograms

For every pair of faces, two 64-bin histograms capture:
- **D2 distance:** Distribution of Euclidean distances between random point pairs on the two faces (captures how far apart they are in 3D space)
- **Angle:** Distribution of dihedral angles between the two faces (captures angular relationship)

These are processed by `NonLinear(64, n_heads)` and added to the attention bias.

#### A3 — Multi-Hop Edge Encoding

For each pair of faces, the Graphormer looks at the **sequence of edges along the shortest path** between them. Each edge's features (curve shape, type, length, angle, convexity) are encoded and then aggregated over the path.

```
Path from Face A to Face D: Edge 1 → Edge 5 → Edge 3
                              ↓          ↓          ↓
                          encode    encode    encode
                              ↓          ↓          ↓
                           aggregate (distance-weighted sum)
                              ↓
                         attention bias for (A, D)
```

### The Virtual Global Node ([CLS] Token)

A special "virtual node" is added that connects to every real face. After the Transformer processes everything, this virtual node's output contains a **global summary** of the entire CAD model. This is used later to provide global context to each face's classification.

---

## 5. The BrepMFR Network Architecture (Layer by Layer)

Let's trace exactly what happens when a batch of graphs enters the network. All the code is in the [`models/`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/) directory.

### 5.1 Input Processing

The batch dictionary arrives from the collator with all tensors padded to the same size.

### 5.2 GraphNodeFeature — Embedding Each Face

**File:** [`brep_encoder_layer.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/layers/brep_encoder_layer.py), class `GraphNodeFeature` (line 180)

This module converts raw face features into a 256-dimensional vector per face:

```
node_data [N, 5, 5, 7]
    │
    ├──▶ SurfaceEncoder (2D CNN: 7→64→128→256, pool, fc→128) ──▶ [N, 128]
    │
face_area [N, 1]
    ├──▶ NonLinear(1, 32) ──▶ [N, 32]
    │
face_type [N]
    ├──▶ nn.Embedding(8, 32) ──▶ [N, 32]
    │
face_loop [N]
    ├──▶ nn.Embedding(256, 32) ──▶ [N, 32]
    │
node_degree [N]
    ├──▶ nn.Embedding(128, 32) ──▶ [N, 32]
    │
    └───── CONCATENATE ────────────▶ [N, 256]  (= hidden_dim)
```

Then, the virtual global node (graph token) is prepended:
```
graph_token [1, 256]  +  face_features [N, 256]  →  [N+1, 256]
```

### 5.3 GraphAttnBias — Building the Attention Bias Matrix

**File:** [`brep_encoder_layer.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/layers/brep_encoder_layer.py), class `GraphAttnBias` (line 320)

This module constructs the bias matrix that guides attention. Shape: `[batch, n_heads, N+1, N+1]`.

```
Start: attn_bias = zeros [batch, n_heads, N+1, N+1]

A1: spatial_pos [N, N] → nn.Embedding → [N, N, n_heads] → add to bias[:, :, 1:, 1:]
    + graph_token_virtual_distance (distance from virtual node to all)

A2: d2_distance [N, N, 64] → NonLinear(64, n_heads) → add to bias[:, :, 1:, 1:]
    angle_distance [N, N, 64] → NonLinear(64, n_heads) → add to bias[:, :, 1:, 1:]

A3: For each edge in edge_path:
    edge_data [E, 5, 7] → CurveEncoder (1D CNN → n_heads)
    edge_type → nn.Embedding(6, n_heads)
    edge_len → NonLinear(1, n_heads)
    edge_ang → NonLinear(1, n_heads)
    edge_conv → nn.Embedding(3, n_heads)
    All summed → edge_feat [E, n_heads]
    
    + NodeCat: incorporate endpoint node features into edges
    → _EdgeConv: edge_feat + projected(node_src + node_dst)
    
    Index edge features along shortest paths → [N, N, max_dist, n_heads]
    Multiply by distance-decay weights (edge_dis_encoder)
    Average over path length → [N, N, n_heads]
    Add to bias[:, :, 1:, 1:]
```

### 5.4 Transformer Layers — The Core Processing

**File:** [`brep_encoder.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/brep_encoder.py) (line 29)

The BrepEncoder stacks **8 Transformer layers** (configurable via `--n_layers_encode`). Each layer:

```
Input: x [N+1, batch, 256]

1. Pre-LayerNorm
2. Multi-Head Self-Attention (32 heads, with structural bias)
   - Q, K, V projections: x → [N+1, batch, 256]
   - Attention scores = Q·Kᵀ / √d_head + attn_bias
   - Output = softmax(scores) · V
3. Residual connection: x = x + attention_output
4. Pre-LayerNorm
5. Feed-Forward Network (256 → 512 → 256, GELU activation)
6. Residual connection: x = x + ffn_output

Output: x [N+1, batch, 256]
```

After all 8 layers, a `tanh` activation is applied:
```python
x = self.tanh(x)
graph_rep = x[0, :, :]  # Virtual node → global graph representation
```

### 5.5 Attention Fusion — Blending Local and Global

**File:** [`brepseg_model.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py), class `Attention` (line 72)

Each face now has:
- `node_z`: its own per-face embedding (256-dim) — **local** information
- `graph_z`: the global virtual node's embedding (256-dim) — **global** information

The Attention module dynamically blends these:

```python
class Attention(nn.Module):
    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)         # [N, 2, 256]
        weights = self.dense_weight(stacked)          # [N, 2, 1]
        weights = F.softmax(weights, dim=1)           # Learned blend ratio
        outputs = torch.sum(stacked * weights, dim=1) # [N, 256]
        return outputs
```

So each face gets a **learned weighted average** of its local and global features. Some faces might rely more on local context (e.g., a simple flat plane), while others need global context (e.g., a through-hole that spans the entire part).

### 5.6 Classifier — Making Predictions

**File:** [`brepseg_model.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py), class `NonLinearClassifier` (line 13)

A 4-layer MLP with batch normalisation and dropout:

```
z [N, 256]
    │
    ├──▶ Linear(256, 512) → BN → ReLU → Dropout(0.3)
    ├──▶ Linear(512, 512) → BN → ReLU → Dropout(0.3)
    ├──▶ Linear(512, 256) → BN → ReLU → Dropout(0.3)
    └──▶ Linear(256, 25) → Softmax
    
    Output: [N, 25] probability distribution over 25 classes
```

### Complete Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         BrepMFR Architecture                         │
│                                                                      │
│  Input: .pt graph                                                    │
│    │                                                                 │
│    ├── node_data ──▶ SurfaceEncoder (2D CNN) ─┐                     │
│    ├── face_area ──▶ NonLinear(1,32) ─────────┤                     │
│    ├── face_type ──▶ Embedding(8,32) ─────────┤                     │
│    ├── face_loop ──▶ Embedding(256,32) ───────┤                     │
│    ├── node_degree ─▶ Embedding(128,32) ──────┤                     │
│    │                                          │                     │
│    │                           ┌──────────────┘                     │
│    │                           ▼                                     │
│    │                  CONCAT → [N, 256]                              │
│    │                  + graph_token [1, 256]                          │
│    │                           │                                     │
│    │                           ▼                                     │
│    │              ┌────────────────────────┐                         │
│    │              │   × 8 Transformer      │                         │
│    │    attn ─▶   │   Layers (256-dim,     │                         │
│    │    bias      │    32 heads, GELU,     │                         │
│    │              │    pre-LayerNorm)      │                         │
│    │              └────────────────────────┘                         │
│    │                           │                                     │
│    │                    tanh activation                               │
│    │                     ╱           ╲                                │
│    │               graph_rep      node_emb                           │
│    │               (virtual       (per-face)                         │
│    │                node)                                            │
│    │                     ╲           ╱                                │
│    │                  Attention Fusion                                │
│    │                    (learned blend)                               │
│    │                           │                                     │
│    │                    z [N, 256]                                    │
│    │                           │                                     │
│    │                    Classifier MLP                                │
│    │              (256→512→512→256→25)                                │
│    │                           │                                     │
│    │                    Softmax                                       │
│    │                           │                                     │
│    └────────────────▶ predictions [N, 25]                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. Stage 1: Supervised Training on Source Data

**File:** [`segmentation.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/segmentation.py) + [`models/brepseg_model.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py)

### What Happens in Stage 1?

The model is trained on **labelled synthetic data** (source domain). Every face in the training set has a known ground-truth class. The model learns to predict these classes.

### The Training Loop (Per Step)

```python
def training_step(self, batch, batch_idx):
    # 1. Encode all faces through the BrepEncoder
    node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)
    
    # 2. Remove virtual node from embeddings (index 0)
    node_emb = node_emb[0].permute(1, 0, 2)[:, 1:, :]
    
    # 3. Extract real (non-padded) node embeddings
    node_pos = torch.where(padding_mask == False)
    node_z = node_emb[node_pos]
    
    # 4. Expand graph embedding to match per-node
    graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0)
    
    # 5. Fuse local + global
    z = self.attention([node_z, graph_z])
    
    # 6. Classify
    node_seg = self.classifier(z)  # [total_nodes, 25]
    
    # 7. Compute cross-entropy loss with optional class weights
    labels_onehot = F.one_hot(labels, 25)
    loss = CrossEntropyLoss(labels_onehot, node_seg, class_level_weight=cw)
```

### The Loss Function

The project uses a **custom Cross-Entropy loss** (not `torch.nn.CrossEntropyLoss`):

```python
def CrossEntropyLoss(label, predict_prob, class_level_weight=None, 
                     instance_level_weight=None, epsilon=1e-12):
    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)
```

Key difference from standard PyTorch CE: the classifier outputs **softmax probabilities** (not raw logits), and the loss operates on these probabilities directly.

---

## 7. Stage 2: Domain Adaptation (DANN + IWDAN)

**File:** [`domain_adapt.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/domain_adapt.py) + [`models/transfer_model.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/transfer_model.py)

### The Problem: Domain Gap

A model trained on synthetic data performs poorly on real-world data. The visual style, complexity, and feature distribution are different. This is the **domain gap**.

```
Source Domain (Synthetic)          Target Domain (Real-world)
┌─────────────────────┐           ┌─────────────────────┐
│  Simple shapes       │           │  Complex real parts  │
│  Generated by script │           │  Designed by humans  │
│  Perfect labels      │           │  No labels (or few)  │
│  Lots of pockets     │           │  Lots of holes       │
└─────────────────────┘           └─────────────────────┘
                  ↑  Domain gap  ↑
```

### The Solution: Two-Stage Training

```
Stage 1: Train on labelled source data → learn general geometry features
                    ↓
Stage 2: Adapt features to be domain-invariant using unlabelled target data
```

### Stage 2 Architecture

The `DomainAdapt` model adds a **Domain Discriminator** and **Gradient Reversal Layer (GRL)** on top of the pretrained Stage 1 model:

```
                    ┌─────────────────┐
     Source Graph ──┤                 │
                    │  BrepEncoder    │──▶ z_s (source features)
     Target Graph ──┤  (pretrained)   │──▶ z_t (target features)
                    └─────────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
            ┌──────────┐   ┌──────────────┐
            │Classifier│   │     GRL      │
            │  (MLP)   │   │ (reverses    │
            └────┬─────┘   │  gradients)  │
                 │         └──────┬───────┘
                 │                │
                 ▼                ▼
            L_label          ┌──────────────┐
            (source CE)      │   Domain     │
                             │Discriminator │
                             │   (MLP)      │
                             └──────┬───────┘
                                    │
                                    ▼
                               L_adv
                           (domain loss)
```

### Stage 2 Joint Loss

The total training loss is:

```
L_total = L_label + 0.3 × L_adv + 0.1 × L_entropy
```

Where:
- **L_label:** Cross-entropy on **source** data (supervised, using labels)
- **L_adv:** Domain adversarial loss (forces domain-invariant features)
- **L_entropy:** Entropy minimisation on **target** predictions (encourages confident predictions)

### How Each Batch Works

Each training batch in Stage 2 contains **both source and target graphs**. The `collator_st` concatenates source first, target second:

```python
# In TransferDataset.__getitem__:
sample = {"source_data": sample_s, "target_data": sample_t}

# In collator_st:
flat = [pair["source_data"] for pair in items] + [pair["target_data"] for pair in items]
```

In the training step, the model splits them apart:
```python
node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)
```

---

## 8. The Gradient Reversal Layer (GRL) — How Fooling Works

**File:** [`models/modules/domain_adv/grl.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/domain_adv/grl.py)

### The Concept (Analogy)

Imagine two artists (the Encoder and the Discriminator) playing a game:

- **Discriminator:** "I can tell the difference between your synthetic and real paintings!"
- **Encoder:** "I'll modify my technique until you can't tell them apart!"

The GRL makes this adversarial game possible within standard gradient descent.

### Forward Pass
The GRL does **nothing** — it passes features through unchanged:
```python
def forward(ctx, input, coeff):
    output = input * 1.0  # Identity
    return output
```

### Backward Pass
The GRL **flips and scales** the gradients:
```python
def backward(ctx, grad_output):
    return grad_output.neg() * ctx.coeff  # Negate!
```

### Why This Works

During backpropagation:
1. The **discriminator** gets normal gradients → it learns to better distinguish source vs. target
2. The **encoder** gets **reversed** gradients → it learns to produce features that **confuse** the discriminator

Over time, the encoder produces features where source and target look identical to the discriminator.

### The Warm-Start Ramp

If the GRL is turned on too aggressively, the discriminator overpowers the encoder early in training, causing instability. The `WarmStartGradientReverseLayer` implements a **slow sigmoid ramp**:

```python
coeff = 2.0 * (hi - lo) / (1.0 + exp(-alpha * i/N)) - (hi - lo) + lo
```

Where:
- `i` = current iteration number
- `N` = `max_iters` (controls how slowly the ramp rises)
- `alpha` = controls the steepness (default 1.0)
- `lo` = starting value (0.0)
- `hi` = ending value (1.0)

```
λ coefficient over training:

1.0 |                                _______________
    |                           ____/
    |                       ___/
    |                   ___/
    |               ___/
    |          ____/
0.0 |_________/
    └──────────────────────────────────────────────▶ training steps
         0                  N/2                   N
```

**Critical detail:** The default `max_iters=1000` from the original dalib library saturates λ within half an epoch on this dataset. This project overrides it to:
```python
max_iters = estimated_steps_per_epoch * max_epochs * grl_ramp_frac  (default 0.5)
```

This means λ reaches 1.0 around the **midpoint** of training, giving the encoder a real warm-up.

---

## 9. IWDAN: Handling Label Shift

**File:** [`models/transfer_model.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/transfer_model.py) (line 32)

### What Is Label Shift?

**Label shift** occurs when the class distributions are different between source and target:

```
Source (synthetic): 58% stock, 15% pockets, 10% holes, 17% other
Target (real):      40% stock, 5% pockets,  30% holes, 25% other
```

### Why Plain DANN Fails Under Label Shift

Plain DANN forces the overall feature distribution of source to match target. But if source has lots of pockets and target has lots of holes, DANN will force pocket features to overlap with hole features. This is **destructive** — it corrupts the learned representations.

**Theorem (Zhao et al., ICML 2019):** Under label shift, DANN can actually **hurt** target performance.

### The IWDAN Fix

IWDAN (Importance-Weighted DANN) re-weights source features in the discriminator:

```python
# Per-class importance weight
w[c] = P_target(c) / P_source(c)  # How much more common is class c in target?
```

For example:
- Holes: `w = 30%/10% = 3.0` → source hole features get 3× weight
- Pockets: `w = 5%/15% = 0.33` → source pocket features get ⅓ weight

This makes the discriminator "see" a source distribution that **mimics** the target distribution, preventing destructive alignment.

### In the Code

```python
# Loading priors from JSON files
src_priors = _load_priors_json(src_priors_path, num_classes)
tgt_priors = _load_priors_json(tgt_priors_path, num_classes)

# Computing weights
def _compute_iwdan_weights(src_priors, tgt_priors, clip_max=10.0):
    w = tgt_priors / src_priors  # Per-class ratio
    w = np.clip(w, 1.0 / clip_max, clip_max)  # Clip extremes
    norm = float((src_priors * w).sum())  # Normalise
    w = w / norm
    return w
```

During training, these weights replace the uniform `1.0` on source nodes:
```python
if self.iwdan_enabled:
    iw = self.iwdan_weights.to(device=z_s.device)
    weight_s[:num_node_s] = iw[label_s]  # Per-face weight based on its class
else:
    weight_s[:num_node_s] = 1.0  # Uniform (vanilla DANN)
```

---

## 10. Class Imbalance and Weighted Cross-Entropy

### The Problem

In the source training data, class distribution is heavily skewed:

```
Class 0 (Stock): ████████████████████████████████████████████████ 58%
Class 1:         ████ 5%
Class 2:         ███ 4%
...
Class 23:        █ 0.5%
Class 24:        █ 0.3%
```

Without correction, the model learns to predict "Stock" for everything — it gets 58% accuracy "for free"!

### The Fix: Inverse-Frequency Class Weights

**File:** [`artifacts/class_weights/stage1/`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/artifacts/class_weights)

The script [`scripts/training/compute_class_weights.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/training) computes weights using the **effective number of samples** formula:

```
weight[c] = (1 - α) / (1 - α^count[c])
```

Where `α = 0.5` (configurable). Rare classes get higher weights.

These weights are stored as JSON in `artifacts/class_weights/stage1/source_train_alpha05.json`:
```json
{
  "num_classes": 25,
  "method": "effective_number",
  "alpha": 0.5,
  "counts": [1250000, 110000, 95000, ...],
  "weights": [0.23, 1.45, 1.67, ...]
}
```

And loaded during training:
```python
# In BrepSeg.__init__:
if self.class_weights_path:
    with open(self.class_weights_path, "r") as f:
        cw = json.load(f)
    weights = torch.tensor(cw["weights"], dtype=torch.float32)
    self.register_buffer("class_weights", weights)
```

**Important:** Class weights are applied only to the **training loss**, not the validation loss. This ensures the LR scheduler sees a clean validation signal.

---

## 11. Logit Adjustment: Post-Training Calibration

Even after weighted training, the model's output probabilities can be over-confident (biased toward source-dominant classes). **Logit adjustment** is a post-hoc calibration applied during evaluation:

```
Adjusted_Logit[c] = Logit[c] - τ × log(P_source(c)) + τ × log(P_target(c))
```

Where:
- `τ` = temperature parameter (swept to find the best value)
- `P_source(c)` = class prior in source data
- `P_target(c)` = class prior in target data

This shifts predictions away from source-dominant classes toward target-dominant classes, **without retraining**.

The sweep is run by:
```powershell
python scripts/diagnostics/stage2_logit_adjust_eval.py \
  --checkpoint results/stage2/<run>/best.ckpt \
  --source_path Z:/source_dataset \
  --target_path Z:/target_dataset
```

---

## 12. Optimisers, Learning Rate Schedules, and Hyperparameters

### Stage 1 Optimiser

```python
# In BrepSeg.configure_optimizers():
optimizer = torch.optim.AdamW(
    self.parameters(),
    lr=0.002,           # Peak learning rate
    betas=(0.9, 0.999), # Momentum parameters
    eps=1e-8,
    weight_decay=0.01,  # L2 regularisation
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',          # Reduce when eval_loss stops decreasing
    factor=0.5,          # Halve the LR
    patience=5,          # Wait 5 epochs before reducing
    min_lr=1e-6,         # Don't go below this
    cooldown=2,          # After reducing, wait 2 epochs before checking again
)
```

**Learning Rate Warmup (Stage 1 only):**
```python
# Linear warmup from 0 → 0.002 over first 5000 steps
if self.trainer.global_step < 5000:
    lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_scale * 0.002
```

### Stage 2 Optimiser

Three separate parameter groups with **asymmetric learning rates**:

```python
# Encoder + classifier: lr=1e-4 (fine-tuning, so small LR)
# Domain discriminator: lr=1e-3 (10× higher, needs to learn from scratch)
optimizer = AdamW(self.brep_encoder.parameters(), lr=1e-4, betas=(0.99, 0.999))
optimizer.add_param_group({"params": self.classifier.parameters(), lr: 1e-4})
optimizer.add_param_group({"params": self.domain_adv.parameters(), lr: 1e-3})
```

Note `betas=(0.99, 0.999)` — the first-moment coefficient is **0.99**, not the more common 0.9. This matches the original paper's configuration exactly.

### Key Hyperparameters

| Parameter | Stage 1 | Stage 2 | Meaning |
|-----------|---------|---------|---------|
| `d_model` | 512 | 512 | FFN hidden dim in Transformer |
| `dim_node` | 256 | 256 | Node embedding dimension |
| `n_heads` | 32 | 32 | Number of attention heads |
| `n_layers_encode` | 8 | 8 | Number of Transformer layers |
| `dropout` | 0.3 | 0.3 | General dropout rate |
| `attention_dropout` | 0.3 | 0.3 | Attention weight dropout |
| `act_dropout` | 0.3 | 0.3 | Activation dropout |
| `batch_size` | 64 | 64 | Graphs per batch |
| `gradient_clip_val` | 1.0 | 1.0 | Max gradient norm |
| `max_epochs` | 1000 | 1000 | Training ceiling |

---

## 13. Training Commands — Step by Step

### Environment Setup
```powershell
conda activate brep_mfr_pyg
```

### Stage 1: Supervised Training
```powershell
python segmentation.py train `
  --dataset_path "Z:/Experiment6_PyG/source_dataset" `
  --class_weights_path "artifacts/class_weights/stage1/source_train_alpha05.json" `
  --max_epochs 1000 `
  --batch_size 64 `
  --num_workers 0
```

Key flags:
- `--class_weights_path`: Activates weighted CE loss to counteract class imbalance
- `--num_workers 0`: **Critical on Windows** — using workers > 0 causes memory/pagefile crashes
- `--drop_invalid_graphs`: Removes empty graphs at startup
- `--max_graph_nodes N`: Drops graphs with > N faces (prevents OOM in attention)
- `--precision "16-mixed"`: Mixed-precision training (reduces GPU memory)
- `--accumulate_grad_batches N`: Gradient accumulation (effective batch = N × batch_size)

### Stage 2: Domain Adaptation
```powershell
python domain_adapt.py train `
  --source_path "Z:/Experiment6_PyG/source_dataset" `
  --target_path "Z:/Experiment6_PyG/target_dataset" `
  --pre_train "results/stage1/<run_name>/best.ckpt" `
  --iwdan `
  --iwdan_source_priors "artifacts/class_weights/stage2_iwdan/source_train_priors.json" `
  --iwdan_target_priors "artifacts/class_weights/stage2_iwdan/target_train_priors.json" `
  --max_epochs 1000 `
  --batch_size 64 `
  --num_workers 0
```

Key flags:
- `--pre_train`: Path to the best Stage 1 checkpoint (weights are loaded into encoder, attention, and classifier)
- `--iwdan`: Enables importance-weighted domain adaptation
- `--grl_ramp_frac 0.5`: λ reaches 1.0 halfway through training
- `--estimated_steps_per_epoch 2444`: Used to compute GRL max_iters

### Testing
```powershell
# Stage 1 test
python segmentation.py test `
  --dataset_path "Z:/source_dataset" `
  --checkpoint "results/stage1/<run_name>/best.ckpt" `
  --batch_size 64

# Stage 2 test
python domain_adapt.py test `
  --source_path "Z:/source_dataset" `
  --target_path "Z:/target_dataset" `
  --checkpoint "results/stage2/<run_name>/best.ckpt" `
  --batch_size 64
```

---

## 14. Inference Pipeline

For running predictions on new, unseen STEP files, the end-to-end pipeline is:

```
New STEP file
    │
    ▼
SolidWorks macro → BrepJson (.json)
    │
    ▼
json_to_brepmfr_pyg.py → .pt graph file
    │
    ▼
run_pyg_inference.py or step_infer_features.py
    │
    ▼
Per-face predictions (class labels for every face)
```

**File:** [`scripts/inference/run_pyg_inference.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/inference/run_pyg_inference.py)

This script:
1. Loads a trained checkpoint (Stage 1 or Stage 2)
2. Loads a `.pt` graph file
3. Runs the graph through the model
4. Outputs per-face class predictions

**File:** [`scripts/inference/step_infer_features.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/inference/step_infer_features.py)

This script provides an end-to-end inference pipeline from a STEP file:
1. Opens the STEP file in SolidWorks (or reads a pre-exported JSON)
2. Converts to `.pt`
3. Runs inference
4. Maps predictions back to original face IDs

---

## 15. Metrics: Per-Face Accuracy, Per-Class Accuracy, IoU

### Per-Face Accuracy (Overall)
```python
per_face_comp = (preds_np == labels_np).astype(np.int64)
per_face_accuracy = np.mean(per_face_comp)
```
What fraction of **all faces** were correctly classified? This is dominated by class 0 (Stock).

### Per-Class Accuracy
```python
for i in range(num_classes):
    class_pos = np.where(labels_np == i)
    class_i_preds = preds_np[class_pos]
    class_i_label = labels_np[class_pos]
    per_class_acc = np.mean(class_i_preds == class_i_label)
```
Average accuracy across all classes — gives equal weight to rare classes.

### IoU (Intersection over Union)
```python
# For each class c:
# Intersection = faces where pred=c AND label=c
# Union = faces where pred=c OR label=c
# IoU[c] = Intersection / Union
```
The strictest metric. A class gets low IoU if the model over-predicts OR under-predicts it.

### Feature-Only Accuracy (Stage 2)
```python
feature_pos = np.where(label_t_np > 0)  # Exclude Stock (class 0)
feature_pred = pred_t_np[feature_pos]
feature_label = label_t_np[feature_pos]
per_face_accuracy_feature = np.mean(feature_pred == feature_label)
```
Accuracy only on **machined features** (ignoring Stock). This is the most meaningful metric because Stock is trivially easy.

---

## 16. Monitoring: TensorBoard, CSV Logs, and W&B

### TensorBoard (Default)
```powershell
tensorboard --logdir results/logs/stage1/<run_name>/
```

Logged metrics include:
- `train_loss`, `eval_loss`
- `per_face_accuracy`, `per_class_accuracy`, `IoU`
- `current_lr` (learning rate)
- `val/max_pred_prob_batch0` (histogram of prediction confidence)
- `grl_lambda` (Stage 2: GRL coefficient value)
- `train_acc_s`, `train_acc_t` (Stage 2: source and target accuracy)
- Confusion matrices (as images, every 5 epochs)

### CSV Logging (Optional)
```powershell
python segmentation.py train ... --csv_log
```
Writes CSV files under `results/logs/stage1/<run_name>/csv_metrics/`.

### Weights & Biases (Optional)
```powershell
python segmentation.py train ... --use_wandb --wandb_project "brepmfr-pyg"
```

---

## 17. Checkpoints and Results Directory Layout

```
results/
├── stage1/
│   └── <run_name>/
│       ├── best.ckpt          ← Best model (lowest eval_loss)
│       ├── last.ckpt          ← Latest model
│       └── best-v1.ckpt      ← (save_top_k=10 keeps top 10 checkpoints)
│
├── stage2/
│   └── <run_name>/
│       ├── best.ckpt
│       └── last.ckpt
│
├── logs/
│   ├── stage1/
│   │   └── <run_name>/
│   │       ├── tensorboard/version_0/events.out.tfevents.*
│   │       └── csv_metrics/version_0/metrics.csv
│   └── stage2/
│       └── <run_name>/
│           └── tensorboard/...
│
└── diagnostics/               ← Evaluation outputs
```

### Run Name Convention

Default auto-generated names:
- Stage 1: `ce_weighted_balanced__2026-05-10_143022_041`
- Stage 2: `transfer_iwdan_weighted__2026-05-11_091500_123`

Format: `<strategy>__<date>_<time>_<milliseconds>`

Override with `--run_name`:
```powershell
python segmentation.py train --run_name "ablation_no_a2__experiment1"
```

### Checkpoint Monitoring

Stage 1 monitors `eval_loss` (validation loss) — lower is better.

Stage 2 monitors `eval_loss = 1 / target_accuracy` — so lowest eval_loss = highest target accuracy:
```python
target_acc = float(np.mean(per_face_comp_val))
eval_loss = 1.0 / (target_acc + 1e-9)
```

---

## 18. File-by-File Reference Map

### Entry Points

| File | Purpose |
|------|---------|
| [`segmentation.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/segmentation.py) | Stage 1 CLI: `train` / `test`. Parses args, creates dataset, trainer, model. |
| [`domain_adapt.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/domain_adapt.py) | Stage 2 CLI: `train` / `test`. Same structure, uses `TransferDataset` + `DomainAdapt`. |

### Models

| File | Purpose |
|------|---------|
| [`models/brepseg_model.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py) | `BrepSeg` Lightning module: encoder + attention + classifier + CE loss. Stage 1. |
| [`models/transfer_model.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/transfer_model.py) | `DomainAdapt` Lightning module: adds GRL + discriminator + IWDAN. Stage 2. |

### Encoder & Layers

| File | Purpose |
|------|---------|
| [`models/modules/brep_encoder.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/brep_encoder.py) | `BrepEncoder`: orchestrates node embedding + attention bias + Transformer stack. |
| [`models/modules/layers/brep_encoder_layer.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/layers/brep_encoder_layer.py) | `GraphNodeFeature`, `GraphAttnBias`, `GraphEncoderLayer` (single Transformer layer). |
| [`models/modules/layers/feature_encoders.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/layers/feature_encoders.py) | `SurfaceEncoder` (2D CNN for face UV-grids), `CurveEncoder` (1D CNN for edge curves). |
| [`models/modules/layers/multihead_attention.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/layers/multihead_attention.py) | Custom multi-head attention with `attn_bias` support. |

### Domain Adversarial

| File | Purpose |
|------|---------|
| [`models/modules/domain_adv/grl.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/domain_adv/grl.py) | `WarmStartGradientReverseLayer`: sigmoid-ramped gradient reversal. |
| [`models/modules/domain_adv/dann.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/domain_adv/dann.py) | `DomainAdversarialLoss`: BCE between domain predictions and labels. |
| [`models/modules/domain_adv/domain_discriminator.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/modules/domain_adv/domain_discriminator.py) | `DomainDiscriminator`: 3-layer MLP (in→512→512→1 with BatchNorm). |

### Data Loading

| File | Purpose |
|------|---------|
| [`data/dataset.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/dataset.py) | `CADSynth` (Stage 1 dataset), `TransferDataset` (Stage 2 paired loader). |
| [`data/collator.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/collator.py) | `collator()` pads + batches graphs. `collator_st()` for source+target pairs. |
| [`data/utils.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/utils.py) | `get_random_rotation()`, `rotate_uvgrid()` — data augmentation. |

### Callbacks & Logging

| File | Purpose |
|------|---------|
| [`callbacks/training_logging.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/callbacks/training_logging.py) | `build_loggers()`, `build_train_callbacks()`, `build_pytorch_profiler()`. |
| [`models/tensorboard_media.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/tensorboard_media.py) | TensorBoard confusion matrix images, histograms, and media logging. |

### Artifacts

| File | Purpose |
|------|---------|
| [`artifacts/class_weights/stage1/`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/artifacts/class_weights) | Precomputed CE class weights (effective number method, α=0.5). |
| [`artifacts/class_weights/stage2_iwdan/`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/artifacts/class_weights) | Source and target class priors for IWDAN importance weighting. |
| [`artifacts/baseline/`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/artifacts/baseline) | Frozen reference pointers for reproducibility. |

### Inference Scripts

| File | Purpose |
|------|---------|
| [`scripts/inference/run_pyg_inference.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/inference/run_pyg_inference.py) | Run inference on `.pt` files with a trained checkpoint. |
| [`scripts/inference/step_infer_features.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/inference/step_infer_features.py) | End-to-end inference from STEP file → predictions. |
| [`scripts/inference/export_uv_json_pred.py`](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/scripts/inference/export_uv_json_pred.py) | Export predictions back to JSON with UV visualisation data. |

---

> **Previous:** Read [Part 1: Data Generation](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data_understanding_files/PART1_data_generation_deep_dive.md) for the complete explanation of the data pipeline from STEP files to `.pt` graph files.
