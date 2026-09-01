# Stage 2: Domain Adaptation — Full Pipeline Walkthrough & Dry Run

> This document traces the **entire Stage 2 pipeline** from command-line entry through data loading, model initialization, forward pass, loss computation, gradient reversal, and backpropagation — with concrete tensor shapes and values at every step.

---

## Table of Contents

1. [What Stage 2 Is and Why It Exists](#1-what-stage-2-is-and-why-it-exists)
2. [Entry Point: `domain_adapt.py`](#2-entry-point-domain_adaptpy)
3. [Data Loading: `TransferDataset` and `collator_st`](#3-data-loading-transferdataset-and-collator_st)
4. [Model Initialization: `DomainAdapt`](#4-model-initialization-domainadapt)
5. [Forward Pass Dry Run: `training_step`](#5-forward-pass-dry-run-training_step)
6. [Loss Computation: Three Objectives](#6-loss-computation-three-objectives)
7. [Gradient Reversal Layer: The Adversarial Trick](#7-gradient-reversal-layer-the-adversarial-trick)
8. [Optimizer and LR Schedule](#8-optimizer-and-lr-schedule)
9. [Validation and Checkpointing](#9-validation-and-checkpointing)
10. [Complete Shape Trace Table](#10-complete-shape-trace-table)

---

## 1. What Stage 2 Is and Why It Exists

Stage 1 trains `BrepSeg` on **labeled synthetic data** (CADSynth). It achieves ~99.96% accuracy on synthetic test sets, but drops to 61-82% when applied to **real-world CAD models** from different distributions.

Stage 2 uses **Domain Adversarial Neural Networks (DANN)** to align the feature distributions of synthetic (source) and real (target) data. The core idea is adversarial: a discriminator tries to tell source features from target features, while the encoder tries to fool the discriminator by producing domain-invariant features.

**Paper Eq. 17 (the combined objective)**:

```
L = L_label + α · L_entropy + β · L_adv

where α = 0.1, β = 0.3
```

- `L_label`: Cross-entropy on labeled source data (same as Stage 1)
- `L_entropy`: Entropy loss on unlabeled target predictions (encourages confident predictions)
- `L_adv`: Domain adversarial loss (aligns source/target feature distributions)

---

## 2. Entry Point: `domain_adapt.py`

### Command line

```bash
python domain_adapt.py train \
    --source_path /data/CADSynth \
    --target_path /data/RealCAD \
    --pre_train results/stage1/best.ckpt \
    --batch_size 32 \
    --max_epochs 200 \
    --num_classes 25
```

### Key differences from Stage 1

| Aspect | Stage 1 (`segmentation.py`) | Stage 2 (`domain_adapt.py`) |
|--------|----------------------------|----------------------------|
| Model | `BrepSeg` | `DomainAdapt` |
| Dataset | `CADSynth` (single source) | `TransferDataset` (source + target) |
| Collator | `collator` | `collator_st` |
| Checkpoint monitor | `eval_loss` (min) | `per_face_accuracy_target` (max) |
| Pre-trained weights | Optional | **Required** (`--pre_train`) |
| Gradient clipping | Not set | `gradient_clip_val=1.0` |

---

## 3. Data Loading: `TransferDataset` and `collator_st`

### `TransferDataset.__getitem__` returns a **pair**

```python
def __getitem__(self, idx):
    sample_s = self.load_one_graph(fn_s)    # source PYGGraph (labeled)
    sample_t = self.load_one_graph(fn_t)    # target PYGGraph (labeled or unlabeled)
    return {"source_data": sample_s, "target_data": sample_t}
```

Each item is a dict with two PYGGraphs. The dataset length is `max(len(source), len(target))` — if one set is smaller, it wraps around with random sampling.

### `collator_st` merges source + target into a single batch

```python
items_source = [extract fields from item["source_data"] for item in items]  # 32 source graphs
items_target = [extract fields from item["target_data"] for item in items]  # 32 target graphs
items = items_source + items_target  # 64 graphs total, source first
```

Then it runs the **exact same padding/stacking logic** as `collator`. The key: **source graphs are always the first half of the batch, target graphs the second half.**

### Dry run: batch_size=32

```
32 source graphs + 32 target graphs = 64 graphs in the batch

Suppose:
  Source graphs: 8-20 faces each, 14-35 edges each
  Target graphs: 6-25 faces each, 10-40 edges each
  max_node = 25 (from a target graph)
  max_edge = 40 (from a target graph)
  total_source_nodes = 450  (sum of all 32 source graph node counts)
  total_target_nodes = 520  (sum of all 32 target graph node counts)
  total_nodes = 970
  total_edges = 1580
```

### Output `batch_data` dict shapes

```
padding_mask:      [64, 25]           ← first 32 rows = source, last 32 = target
edge_padding_mask: [64, 40]
node_data:         [970, 5, 5, 7]     ← 970 total nodes (flat-concatenated)
face_type:         [970]
...
edge_data:         [1580, 5, 7]       ← 1580 total edges
...
attn_bias:         [64, 26, 26]       ← 25+1 Vnode
spatial_pos:       [64, 25, 25]
d2_distance:       [64, 25, 25, 64]
angle_distance:    [64, 25, 25, 64]
edge_path:         [64, 25, 25, 16]
label_feature:     [970]              ← first 450 = source labels, last 520 = target labels
id:                [64]
graph:             DGL BatchedGraph   ← 64 subgraphs merged
```

---

## 4. Model Initialization: `DomainAdapt`

### Loading the pretrained Stage 1 model

```python
class DomainAdapt(pl.LightningModule):
    def __init__(self, args):
        pre_trained_model = BrepSeg.load_from_checkpoint(args.pre_train)
        self.brep_encoder = pre_trained_model.brep_encoder   # already trained
        self.attention = pre_trained_model.attention          # already trained
        self.classifier = pre_trained_model.classifier        # already trained
```

The entire `BrepSeg` model (encoder + attention + classifier) is loaded from the Stage 1 checkpoint. All three components are reused and continue to be fine-tuned.

### New components for domain adaptation

```python
        grl = WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1.,
            max_iters=200 * 1400,   # 280,000 total training steps
            auto_step=False
        )
        domain_discri = DomainDiscriminator(256, hidden_size=512)
        self.domain_adv = DomainAdversarialLoss(domain_discri, grl=grl)
```

### Architecture summary

```
DomainAdapt
├── brep_encoder          (from Stage 1 checkpoint)
│   ├── graph_node_feature
│   ├── graph_attn_bias
│   └── GraphEncoderLayer × 8
├── attention             (from Stage 1 — inter-graph attention)
├── classifier            (from Stage 1 — NonLinearClassifier)
│   ├── Linear(256→512) → BN → ReLU → Dropout
│   ├── Linear(512→512) → BN → ReLU → Dropout
│   ├── Linear(512→256) → BN → ReLU → Dropout
│   └── Linear(256→25) → Softmax
└── domain_adv            (NEW)
    ├── grl: WarmStartGradientReverseLayer
    └── domain_discriminator: DomainDiscriminator
        ├── Linear(256→512) → BN → ReLU
        ├── Linear(512→512) → BN → ReLU
        └── Linear(512→1) → Sigmoid
```

---

## 5. Forward Pass Dry Run: `training_step`

Using our example batch: 64 graphs (32 source + 32 target), max_node=25, total_source_nodes=450, total_target_nodes=520.

### Step 1: BrepEncoder — Process all 64 graphs together

```python
node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)
```

The encoder doesn't know about source/target — it sees 64 graphs as one batch.

```
node_emb = [inner_states[-1]]
  inner_states[-1]: [26, 64, 256]     ← (max_node+1, batch, dim) after 8 Transformer layers + tanh
graph_emb: [64, 256]                  ← Vnode representations
```

### Step 2: Extract node embeddings and remove Vnode

```python
node_emb = node_emb[0].permute(1, 0, 2)   # [26, 64, 256] → [64, 26, 256]
node_emb = node_emb[:, 1:, :]             # remove Vnode → [64, 25, 256]
```

### Step 3: Split source and target via `chunk(2)`

```python
node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
# node_emb_s: [32, 25, 256]   ← source graphs (first half)
# node_emb_t: [32, 25, 256]   ← target graphs (second half)

padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)
# padding_mask_s: [32, 25]
# padding_mask_t: [32, 25]
```

### Step 4: Flatten padded → real nodes (source)

```python
node_pos_s = torch.where(padding_mask_s == False)   # 450 real source node positions
node_z_s = node_emb_s[node_pos_s]                   # [450, 256]
```

Same for target:
```python
node_pos_t = torch.where(padding_mask_t == False)   # 520 real target node positions
node_z_t = node_emb_t[node_pos_t]                   # [520, 256]
```

### Step 5: Inter-graph attention (source)

```python
graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)
# graph_emb_s: [32, 256]   ← 32 source graph embeddings
# graph_emb_t: [32, 256]   ← 32 target graph embeddings

# Repeat each graph embedding for its number of real nodes
num_nodes_per_graph_s = torch.sum((~padding_mask_s).long(), dim=-1)  # [32], e.g. [12, 8, 15, ...]
graph_z_s = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0)
# graph_z_s: [450, 256] — each row is the graph embedding of the graph that node belongs to
```

For example, if graph 0 has 12 nodes and graph 1 has 8 nodes:
```
graph_z_s[0]  = graph_emb_s[0]   ← graph 0's embedding
graph_z_s[1]  = graph_emb_s[0]
...
graph_z_s[11] = graph_emb_s[0]   ← still graph 0
graph_z_s[12] = graph_emb_s[1]   ← graph 1's embedding starts
...
graph_z_s[19] = graph_emb_s[1]
...
```

The `Attention` module (inter-graph attention) combines them:

```python
z_s = self.attention([node_z_s, graph_z_s])  # [450, 256]
```

Inside `Attention.forward`:
```python
stacked = torch.stack([node_z_s, graph_z_s], dim=1)   # [450, 2, 256]
weights = self.dense_weight(stacked)                    # [450, 2, 1]
weights = F.softmax(weights, dim=1)                     # [450, 2, 1] — sums to 1
outputs = torch.sum(stacked * weights, dim=1)           # [450, 256]
```

This learns a per-node weighted combination of the node's own features and its graph-level context. For example:

```
For a node where local info is more important:
  weights = [[0.7], [0.3]]  →  z = 0.7 * node_z + 0.3 * graph_z

For a node where global context matters more:
  weights = [[0.4], [0.6]]  →  z = 0.4 * node_z + 0.6 * graph_z
```

Same for target:
```python
z_t = self.attention([node_z_t, graph_z_t])  # [520, 256]
```

### Step 6: Classification

```python
node_seg_s = self.classifier(z_s)   # [450, 25]  (25 class probabilities per source node)
node_seg_t = self.classifier(z_t)   # [520, 25]  (25 class probabilities per target node)
```

Inside `NonLinearClassifier`:
```
[450, 256] → Linear(256→512) → BN → ReLU → Dropout
           → Linear(512→512) → BN → ReLU → Dropout
           → Linear(512→256) → BN → ReLU → Dropout
           → Linear(256→25) → Softmax
           → [450, 25]
```

Each row sums to 1.0 (softmax output).

---

## 6. Loss Computation: Three Objectives

### Loss 1: `L_label` — Cross-entropy on source data (Paper Eq. 11)

```python
num_node_s = node_seg_s.size(0)     # 450
label_s = batch["label_feature"][:num_node_s].long()  # [450]
label_s_onehot = F.one_hot(label_s, 25)               # [450, 25]
loss_s = CrossEntropyLoss(label_s_onehot, node_seg_s)  # scalar
```

For a source node with ground truth label 3 ("through_slot") and predicted probabilities:

```
label_onehot = [0, 0, 0, 1, 0, ..., 0]  (25D)
predict_prob = [0.01, 0.02, 0.05, 0.82, 0.03, ..., 0.01]

CE = -1 * log(0.82) = 0.198   ← low loss, good prediction
```

For a misclassified node:
```
label_onehot = [0, 0, 0, 1, 0, ..., 0]
predict_prob = [0.01, 0.45, 0.05, 0.12, 0.03, ..., 0.01]

CE = -1 * log(0.12) = 2.12    ← high loss, bad prediction
```

`loss_s` is averaged over all 450 source nodes.

### Loss 2: `L_entropy` — Entropy loss on target data (Paper Eq. 15)

```python
loss_t = EntropyLoss(node_seg_t)   # scalar
```

This encourages the model to make **confident** predictions on target data, even without labels:

```python
entropy = -predict_prob * log(predict_prob)
# Summed over all classes, averaged over all target nodes
```

For a confident prediction (entropy is low):
```
predict_prob = [0.01, 0.02, 0.93, 0.01, ..., 0.01]
entropy = -(0.01·log(0.01) + 0.02·log(0.02) + 0.93·log(0.93) + ...) = 0.42
```

For an uncertain prediction (entropy is high):
```
predict_prob = [0.04, 0.04, 0.04, 0.04, ..., 0.04]   (uniform across 25 classes)
entropy = -(25 × 0.04·log(0.04)) = 3.22
```

The loss pushes target predictions toward confident (low-entropy) distributions.

### Loss 3: `L_adv` — Domain adversarial loss (Paper Eq. 12-13)

This is the most complex loss. It involves padding, the GRL, and the domain discriminator.

#### Step A: Pad source and target features to equal length

```python
max_num_node = max(num_node_s, num_node_t)  # max(450, 520) = 520

pad_z_s = nn.ZeroPad2d(padding=(0, 0, 0, 520 - 450))
z_s_ = pad_z_s(z_s)                        # [450, 256] → [520, 256]  (70 zero rows appended)

pad_z_t = nn.ZeroPad2d(padding=(0, 0, 0, 520 - 520))
z_t_ = pad_z_t(z_t)                        # [520, 256] → [520, 256]  (no padding needed)

# Weight masks: 1.0 for real nodes, 0.0 for padding
weight_s = [1, 1, ..., 1, 0, 0, ..., 0]    # first 450 = 1.0, last 70 = 0.0
weight_t = [1, 1, ..., 1]                   # all 520 = 1.0
```

#### Step B: Pass through GRL and discriminator

```python
loss_adv = self.domain_adv(z_s_, z_t_, weight_s, weight_t)
```

Inside `DomainAdversarialLoss.forward`:

```python
f = self.grl(torch.cat((z_s_, z_t_), dim=0))   # [1040, 256]
# GRL: forward = identity, backward = -λ · gradient
# At iter 0: λ ≈ 0.0 (no reversal yet)
# At iter 140,000 (halfway): λ ≈ 0.73
# At iter 280,000 (end): λ → 1.0

d = self.domain_discriminator(f)   # [1040, 1]  (sigmoid output: 0-1)
d_s, d_t = d.chunk(2, dim=0)      # [520, 1], [520, 1]

d_label_s = torch.ones(520, 1)    # source = 1
d_label_t = torch.zeros(520, 1)   # target = 0

loss = 0.5 * (BCE(d_s, 1, weight_s) + BCE(d_t, 0, weight_t))
```

The discriminator tries to output 1 for source features and 0 for target features. The BCE loss:

```
For a source node where d_s = 0.9 (correctly identified as source):
  BCE = -1 · log(0.9) = 0.105  ← low loss for discriminator

For a source node where d_s = 0.3 (fooled by encoder):
  BCE = -1 · log(0.3) = 1.204  ← high loss for discriminator (good for encoder!)
```

The weights ensure padding nodes (the 70 zero-padded rows in z_s_) don't contribute to the loss.

#### Step C: Compute domain accuracy (for monitoring)

```python
domain_acc = 0.5 * (binary_accuracy(d_s, 1) + binary_accuracy(d_t, 0))
# If acc ≈ 50%, the encoder has successfully confused the discriminator
# If acc ≈ 100%, the discriminator can still easily tell them apart
```

### Combined loss

```python
loss = loss_s + 0.3 * loss_adv + 0.1 * loss_t
```

With example values:
```
loss_s   = 0.45   (source CE loss — gets lower as training proceeds)
loss_adv = 0.68   (adversarial loss — should hover around 0.69 = -log(0.5) when balanced)
loss_t   = 1.80   (target entropy — gets lower as predictions become confident)

loss = 0.45 + 0.3 × 0.68 + 0.1 × 1.80
     = 0.45 + 0.204 + 0.18
     = 0.834
```

---

## 7. Gradient Reversal Layer: The Adversarial Trick

The GRL is the key mechanism that makes adversarial training work without alternating min/max steps.

### Forward pass: identity

```python
# GradientReverseFunction.forward:
output = input * 1.0    # just copies the input
# z_s_ and z_t_ pass through unchanged
```

### Backward pass: negate gradients

```python
# GradientReverseFunction.backward:
return grad_output.neg() * coeff    # FLIP the gradient sign, scale by λ
```

### What this means for training

Consider the gradient flow for `L_adv`:

```
L_adv
  ↓ gradient: ∂L/∂d (wants discriminator to improve)
DomainDiscriminator
  ↓ gradient: ∂L/∂f (wants features to be MORE distinguishable)
GRL
  ↓ gradient: -λ · ∂L/∂f (REVERSED! wants features to be LESS distinguishable)
BrepEncoder
```

**The discriminator** receives normal gradients → it gets better at distinguishing domains.

**The encoder** receives reversed gradients → it learns to produce features that confuse the discriminator.

This is the min-max game from the paper: the encoder and discriminator are trained simultaneously with a single optimizer, but with opposing objectives.

### Warm-start schedule for λ

```python
λ = 2(hi - lo) / (1 + exp(-α · i / N)) - (hi - lo) + lo
```

With `α=1, lo=0, hi=1, N=280,000`:

```
Step 0:         λ = 0.000   ← no reversal (let encoder settle first)
Step 28,000:    λ = 0.095   ← gentle reversal begins
Step 140,000:   λ = 0.462   ← moderate reversal
Step 280,000:   λ = 0.731   ← strong reversal
Step → ∞:       λ → 1.000   ← full reversal
```

The warm start prevents the adversarial signal from destabilizing the encoder before it has adapted to the new task.

### GRL step is called manually

```python
# In training_step, AFTER computing loss_adv:
self.domain_adv.grl.step()   # increment iter_num by 1
```

This is only called during training, not validation.

---

## 8. Optimizer and LR Schedule

### Four parameter groups with different learning rates

```python
optimizer = AdamW([
    {"params": brep_encoder,  "lr": 0.0001},   # conservative — already pretrained
    {"params": attention,     "lr": 0.0001},   # conservative — already pretrained
    {"params": classifier,    "lr": 0.0001},   # conservative — already pretrained
    {"params": domain_adv,    "lr": 0.001},    # 10× higher — new, needs fast learning
])
```

The domain discriminator gets a 10× higher learning rate because it's initialized from scratch and needs to catch up to the pretrained encoder.

### LR warmup (first 5000 steps)

```python
def optimizer_step(self, ...):
    if self.trainer.global_step < 5000:
        lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
        # Step 0:    lr_scale = 0.0002
        # Step 2500: lr_scale = 0.5
        # Step 5000: lr_scale = 1.0
        for pg, base_lr in zip(optimizer.param_groups, [0.0001, 0.0001, 0.0001, 0.001]):
            pg["lr"] = lr_scale * base_lr
```

### ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",                  # maximize per_face_accuracy_target
    factor=0.5,                  # halve LR when plateau detected
    patience=15,                 # wait 15 epochs before reducing
    min_lr=1e-5,                 # don't let LR go below 0.00001
    cooldown=5,                  # wait 5 epochs after reduction before checking again
)
```

Monitored metric: `per_face_accuracy_target` — the face classification accuracy on target domain data.

---

## 9. Validation and Checkpointing

### Validation follows the same forward pass

The `validation_step` runs the same encoder → split → attention → classifier flow, but:
- No GRL step (λ is not incremented)
- Domain adversarial loss is still computed for monitoring
- Predictions are accumulated across the epoch for aggregate metrics

### Epoch-end metrics

```python
def validation_epoch_end(self, ...):
    # Aggregate all predictions across batches
    per_face_accuracy_source   # accuracy on source domain faces
    per_face_accuracy_target   # accuracy on target domain faces ← CHECKPOINT METRIC
    per_face_accuracy_target_feature  # accuracy on target faces excluding "stock" (label 0)
    per_class_accuracy         # average accuracy per machining feature class
```

### Checkpointing

```python
ModelCheckpoint(
    monitor="per_face_accuracy_target",   # save when target accuracy improves
    mode="max",
    save_top_k=10,
    save_last=True,
)
```

---

## 10. Complete Shape Trace Table

Using batch_size=32, 32 source + 32 target graphs, max_node=25, total_source_nodes=450, total_target_nodes=520.

### Data Loading

| Step | Tensor | Shape | Description |
|------|--------|-------|-------------|
| TransferDataset | `__getitem__` returns | `{"source_data": PYGGraph, "target_data": PYGGraph}` | One pair per index |
| collator_st | `items` | 64 tuples | 32 source + 32 target concatenated |
| Batch output | `padding_mask` | `[64, 25]` | First 32 rows = source, last 32 = target |
| | `node_data` | `[970, 5, 5, 7]` | 450 source + 520 target nodes flat |
| | `label_feature` | `[970]` | 450 source labels + 520 target labels flat |
| | `attn_bias` | `[64, 26, 26]` | 25 nodes + 1 Vnode |
| | `spatial_pos` | `[64, 25, 25]` | |
| | `edge_path` | `[64, 25, 25, 16]` | |

### BrepEncoder (all 64 graphs together)

| Step | Tensor | Shape | Description |
|------|--------|-------|-------------|
| GraphNodeFeature | `x` | `[64, 26, 256]` | Padded + Vnode |
| | `x_0` | `[970, 256]` | Flat node features |
| GraphAttnBias | `attn_bias` | `[64, 32, 26, 26]` | Per-head bias matrix |
| Transformer × 8 | `inner_states[-1]` | `[26, 64, 256]` | After 8 layers + tanh |
| Vnode extract | `graph_emb` | `[64, 256]` | Graph-level embeddings |

### Source-Target Split

| Step | Tensor | Shape | Description |
|------|--------|-------|-------------|
| Permute + strip Vnode | `node_emb` | `[64, 25, 256]` | |
| chunk source | `node_emb_s` | `[32, 25, 256]` | Source half |
| chunk target | `node_emb_t` | `[32, 25, 256]` | Target half |
| Flatten source | `node_z_s` | `[450, 256]` | Real source nodes |
| Flatten target | `node_z_t` | `[520, 256]` | Real target nodes |
| Graph embed split | `graph_emb_s` | `[32, 256]` | Source graph embeddings |
| | `graph_emb_t` | `[32, 256]` | Target graph embeddings |
| Repeat interleave | `graph_z_s` | `[450, 256]` | Graph emb repeated per node |
| | `graph_z_t` | `[520, 256]` | |

### Inter-graph Attention

| Step | Tensor | Shape | Description |
|------|--------|-------|-------------|
| Stack | `[node_z_s, graph_z_s]` | `[450, 2, 256]` | Node + graph features |
| Weights | `weights` | `[450, 2, 1]` | Softmax attention weights |
| Output | `z_s` | `[450, 256]` | Weighted combination |
| | `z_t` | `[520, 256]` | |

### Classification

| Step | Tensor | Shape | Description |
|------|--------|-------|-------------|
| Classifier(z_s) | `node_seg_s` | `[450, 25]` | Source class probabilities |
| Classifier(z_t) | `node_seg_t` | `[520, 25]` | Target class probabilities |

### Loss Computation

| Step | Tensor | Shape | Description |
|------|--------|-------|-------------|
| Source labels | `label_s` | `[450]` | From `label_feature[:450]` |
| One-hot | `label_s_onehot` | `[450, 25]` | |
| `loss_s` | scalar | | CE on source |
| `loss_t` | scalar | | Entropy on target |
| Pad z_s | `z_s_` | `[520, 256]` | 70 zero rows added |
| Pad z_t | `z_t_` | `[520, 256]` | No padding needed |
| `weight_s` | `[520]` | | First 450 = 1.0, rest = 0.0 |
| `weight_t` | `[520]` | | All 1.0 |

### Domain Adversarial Loss

| Step | Tensor | Shape | Description |
|------|--------|-------|-------------|
| Concatenate | `cat(z_s_, z_t_)` | `[1040, 256]` | Source + target features |
| GRL | `f` | `[1040, 256]` | Identity forward, -λ·grad backward |
| Discriminator | `d` | `[1040, 1]` | Domain prediction (sigmoid) |
| Split | `d_s` | `[520, 1]` | Source domain predictions |
| | `d_t` | `[520, 1]` | Target domain predictions |
| Labels | `d_label_s` | `[520, 1]` | All 1.0 (source) |
| | `d_label_t` | `[520, 1]` | All 0.0 (target) |
| `loss_adv` | scalar | | Weighted BCE |

### Combined Loss and Backprop

```
loss = loss_s + 0.3 * loss_adv + 0.1 * loss_t

Gradient flow:
  loss_s    → classifier → attention → encoder     (normal gradients)
  loss_t    → classifier → attention → encoder     (normal gradients)
  loss_adv  → discriminator                         (normal gradients: improve discriminator)
  loss_adv  → GRL → attention → encoder             (REVERSED gradients: confuse discriminator)
```

---

## Appendix: Stage 1 vs Stage 2 Side-by-Side

```
STAGE 1 (Supervised)                    STAGE 2 (Domain Adaptation)
═══════════════════                     ═══════════════════════════

Dataset: CADSynth only                  Dataset: Source + Target paired
  ↓                                       ↓
collator → batch_data                   collator_st → batch_data (2× batch size)
  ↓                                       ↓
BrepEncoder(batch)                      BrepEncoder(batch)  ← same encoder
  ↓                                       ↓
node_emb, graph_emb                     node_emb, graph_emb
  ↓                                       ↓
Flatten → Attention                     chunk(2) → source / target
  ↓                                       ↓                    ↓
Classifier → [N, 25]                    Flatten → Attention    Flatten → Attention
  ↓                                       ↓                    ↓
CE Loss(labels, preds)                  Classifier → seg_s    Classifier → seg_t
  ↓                                       ↓                    ↓
loss = L_label                          CE Loss(labels_s)    Entropy Loss
                                          ↓                    ↓
                                        Pad to equal length → GRL → Discriminator
                                          ↓
                                        loss = L_label + 0.3·L_adv + 0.1·L_entropy
```
