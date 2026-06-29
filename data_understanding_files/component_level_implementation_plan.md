# Class Imbalance Strategy — Deep Analysis & Implementation Plan

## What Our Focal Loss Experiment Proved

The Focal Loss run (γ=2.0 + class weights α=0.5) gave us critical evidence:

| Metric | Baseline CE | Focal Loss γ=2.0 | Takeaway |
|--------|------------|-------------------|----------|
| Thread recall | ~80% | **93.9% peak** | ✅ Thread CAN be learned better |
| Text recall | ~99% | **48%** collapsed | ❌ Over-corrected |
| per_face_accuracy | ~95% | **52%** collapsed | ❌ Training destabilised |
| LR | Healthy | **Decayed to 0** | ❌ Infrastructure bug |

**Key insight:** The model has the capacity to learn threads well. The problem is purely about how we weight the training signal.

---

## Critical Infrastructure Bug Found

> [!CAUTION]
> Before trying ANY new loss strategy, this bug must be fixed. It caused the LR collapse in the Focal Loss run and will sabotage every future experiment.

### The Bug: `eval_loss` monitors text-dominated CE loss

Three interconnected problems:

**1. `eval_loss` is unweighted CE** — [brepseg_model.py L413](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py#L413):
```python
loss = CrossEntropyLoss(labels_onehot, node_seg)  # No class weights!
```
Even when training uses Focal Loss, validation loss is computed with plain CE. Since 85% of validation faces are text, `eval_loss` is dominated by text accuracy.

**2. LR scheduler monitors `eval_loss`** — [brepseg_model.py L685](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py#L685):
```python
"monitor": "eval_loss",  # ← text-dominated metric
```
When Focal Loss improves thread accuracy but hurts text, `eval_loss` worsens → LR gets halved repeatedly → LR reaches 0 → training dies.

**3. Model checkpoint also monitors `eval_loss`** — [segmentation.py L456](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/segmentation.py#L456):
```python
checkpoint_callback = ModelCheckpoint(
    monitor="eval_loss",  # ← saves best TEXT model, not best OVERALL model
    ...
)
```
The saved "best" checkpoint may be the one with highest text accuracy, not the one with best balanced performance.

### The Fix

Change both monitors from `eval_loss` to `per_class_accuracy` (which we just added — it equally weights all 3 classes). Change mode from `min` to `max`.

---

## Evaluating Each Proposed Approach

### Approaches Ranked by Impact × Feasibility

| Rank | Approach | Impact | Effort | Risk | Verdict |
|------|---------|--------|--------|------|---------|
| **🥇 1** | **Fix infrastructure bug** (monitor metric) | Critical | 30 min | None | **DO FIRST — everything else fails without this** |
| **🥈 2** | **Component-balanced loss** | Very High | 3-4 hrs | Low | **Best single change — directly attacks root cause** |
| **🥉 3** | **Graph-level balanced sampling** | High | 2-3 hrs | Low | **Compounds with #2** |
| 4 | **Focal Loss γ=1.0 without class weights** | Medium | 5 min | Low | Already implemented, just change CLI args |
| 5 | **Hierarchical heads** (stock-vs-feature → thread-vs-text) | High | 1-2 days | Medium | Architecture change, good but not first |
| 6 | **Subgraph/cropped training** | High | 1-2 weeks | High | Major pipeline rewrite, defer |
| 7 | **Area-aware loss weighting** | Medium | 2-3 hrs | Low | Can layer on top of #2 |
| 8 | **Dataset-level rebalancing** | Very High | 1-2 weeks | Low | Requires SolidWorks macro changes, long-term |
| 9 | **Post-processing connected components** | Medium | 1 day | None | Inference-only, doesn't help training |
| 10 | **Text sub-labels** (cap/wall/boundary) | Medium | 1 week | Medium | Requires re-generation |

### Why Component-Balanced Loss Is the Best Next Step

The AI analysis nailed the core insight. Here's the math of why it matters:

**Current situation (per-face loss):**
```
One "SOLIDWORKS" text operation: 600 faces × 1 vote each = 600 training votes
One M10 thread operation:         12 faces × 1 vote each =  12 training votes
Ratio: 50:1 per feature instance
```

**With component-balanced loss:**
```
One "SOLIDWORKS" text operation: 600 faces × (1/√600) weight = ~24.5 effective votes
One M10 thread operation:         12 faces × (1/√12) weight  = ~3.5 effective votes
Ratio: 7:1 per feature instance (much better!)
```

This is fundamentally different from class weighting because it operates at the **feature instance level**, not the class level. A graph with 3 text features and 2 thread features would give them roughly equal importance, regardless of how many faces each generates.

**Feasibility check:** The `.pt` files contain `face_adj` (face adjacency info) and `label_feature` (per-face labels). From these, we can compute same-label connected components in the collator or at loss time using `edge_index`.

---

## Proposed Implementation Plan

### Phase 1: Fix Infrastructure (Must-Do Before Any Training)

> [!IMPORTANT]
> This is a 30-minute fix that prevents the LR collapse bug from sabotaging all future experiments.

#### [MODIFY] [brepseg_model.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py)

1. **Validation loss should use the same loss function as training** (line 413):
   - If training uses Focal Loss, validation should too
   - This ensures `eval_loss` reflects the actual training objective

2. **LR scheduler monitor** → change from `eval_loss` to `per_class_accuracy`, mode `min` → `max` (line 685)

3. **Add an early stopping patience** (optional safety net)

#### [MODIFY] [segmentation.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/segmentation.py)

1. **Checkpoint monitor** → change from `eval_loss` to `per_class_accuracy`, add `mode="max"` (line 456)

---

### Phase 2: Component-Balanced Loss Weighting

#### [MODIFY] [brepseg_model.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/models/brepseg_model.py)

Add a helper function that computes per-face instance weights based on same-label connected components:

```python
def compute_component_weights(labels, edge_index, num_nodes, method="sqrt"):
    """Compute per-face weights that normalize by connected component size.
    
    A 600-face text component gets weight 1/sqrt(600) per face,
    while a 12-face thread component gets weight 1/sqrt(12) per face.
    This makes each FEATURE INSTANCE equally important, regardless of
    how many faces it generates.
    """
    # Build same-label adjacency: only connect faces with identical labels
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst])
    filtered_edges = edge_index[:, same_label]
    
    # Find connected components via union-find (on CPU, fast for graph sizes)
    # ... returns component_id per face
    
    # Weight = 1 / sqrt(component_size)  or  1 / component_size
    component_sizes = ...  # count faces per component
    if method == "sqrt":
        weights = 1.0 / torch.sqrt(component_sizes[component_ids].float())
    else:
        weights = 1.0 / component_sizes[component_ids].float()
    
    return weights  # [num_faces]
```

Modify `training_step` to compute instance-level weights and pass them to the loss:

```python
# In training_step, after labels are extracted:
instance_weights = compute_component_weights(
    labels, batch["edge_index_flat"], ...
)
loss = FocalLoss(labels_onehot, node_seg, gamma=self.focal_gamma,
                 class_level_weight=cw,
                 instance_level_weight=instance_weights)
```

> [!NOTE]
> The `edge_index` in the batch dict is currently not present (the collator batches it as `edge_index` but the encoder uses `attn_bias`/`spatial_pos` instead). We need to verify the batched edge index is passed through to the model's `training_step`.

#### [MODIFY] [collator.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/collator.py)

Ensure the batched `edge_index` (already computed on line 261) is accessible in the batch dict (it already is on line 270 — just need to verify it's passed to the model).

---

### Phase 3: Graph-Level Balanced Sampling

#### [MODIFY] [dataset.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/data/dataset.py)

Add a `BalancedGraphSampler` that oversamples thread-bearing graphs:

```python
class BalancedGraphSampler:
    """Ensures each training batch contains thread-bearing graphs.
    
    Precomputes per-graph class presence at dataset init.
    Sampling weight: 1.0 + 5.0 * has_thread + 1.0 * has_text
    Thread-bearing graphs get 6× sampling probability.
    """
```

#### [MODIFY] [segmentation.py](file:///c:/Users/RZA2/Desktop/BrepMFR_PyG/BrepMFR_PyG/segmentation.py)

Add `--balanced_sampling` CLI flag that activates the balanced sampler instead of random shuffle.

---

## What I Would NOT Do (and Why)

| Approach | Why Skip (for now) |
|----------|-------------------|
| **Subgraph/cropped training** | Requires rewriting the entire data pipeline — collator, dataset, encoder input. Too invasive for a first improvement. |
| **Text sub-labels (cap/wall/boundary)** | Requires re-running SolidWorks data generation. Good idea long-term but blocks on the macro pipeline. |
| **Larger model / more capacity** | The model already learns text at 99%. More capacity would make text dominance worse, not better. |
| **Aggressive thread class weight** | Our Focal Loss experiment proved that over-amplifying thread creates thread false-positives and kills text. The component-balanced approach is strictly better. |
| **GraphSMOTE / synthetic node generation** | The literature papers mostly address node classification on citation/social graphs. Our B-Rep graphs have strict geometric constraints — you can't synthesize valid B-Rep faces via SMOTE. |

---

## Recommended Experiment Order

### Experiment 1: Infrastructure Fix + Focal γ=1.0 (No Class Weights)
**Purpose:** Establish a clean baseline with the LR collapse bug fixed.

```powershell
python segmentation.py train `
  --dataset_path "Z:\lite\pyg" `
  --num_classes 3 `
  --loss_type focal `
  --focal_gamma 1.0 `
  --batch_size 1 `
  --accumulate_grad_batches 64 `
  --precision "16-mixed" `
  --num_workers 0 `
  --drop_invalid_graphs `
  --run_name "exp1_focal_g1_infra_fix"
```

### Experiment 2: Component-Balanced Focal Loss
**Purpose:** Test the key hypothesis — does feature-instance-level balancing help?

```powershell
python segmentation.py train `
  --dataset_path "Z:\lite\pyg" `
  --num_classes 3 `
  --loss_type focal `
  --focal_gamma 1.0 `
  --component_balance sqrt `
  --batch_size 1 `
  --accumulate_grad_batches 64 `
  --precision "16-mixed" `
  --num_workers 0 `
  --drop_invalid_graphs `
  --run_name "exp2_component_balanced_focal"
```

### Experiment 3: Component-Balanced + Graph Sampling
**Purpose:** Full solution stack.

```powershell
python segmentation.py train `
  --dataset_path "Z:\lite\pyg" `
  --num_classes 3 `
  --loss_type focal `
  --focal_gamma 1.0 `
  --component_balance sqrt `
  --balanced_sampling `
  --batch_size 1 `
  --accumulate_grad_batches 64 `
  --precision "16-mixed" `
  --num_workers 0 `
  --drop_invalid_graphs `
  --run_name "exp3_full_balance_stack"
```

## Open Questions

1. **Should we change the checkpoint/LR monitor to `per_class_accuracy` (treats all 3 classes equally) or to something like `val_class_1_acc` (optimises directly for thread)?** Per-class accuracy is more balanced; thread-specific would be more targeted but risks overfitting.

2. **For component-balanced loss, should we use `1/sqrt(N)` or `1/N` normalisation?** `1/sqrt(N)` is gentler (the AI analysis recommends starting here); `1/N` fully equalises feature instances but may under-train text.

3. **What's the priority: run more experiments with the current data, or invest time in rebalancing the data generation pipeline?** Both are valid but very different time investments.
