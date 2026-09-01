# Are Target Labels Used for Training in Stage 2?

> **Short answer: No.** Target labels are never used to compute any training loss. They exist in the batch purely for monitoring accuracy during training.

---

## The Three Losses and What They Use

| Loss | Code | Formula | Uses target labels? | Purpose |
|------|------|---------|-------------------|---------|
| `loss_s` | `CrossEntropyLoss(label_s_onehot, node_seg_s)` | `-label · log(pred)` | No (source labels only) | Supervised classification |
| `loss_t` | `EntropyLoss(node_seg_t)` | `-pred · log(pred)` | **No** | Push target predictions toward confidence |
| `loss_adv` | `self.domain_adv(z_s_, z_t_, weight_s, weight_t)` | `BCE(discriminator(features))` | **No** | Align source/target feature distributions |

---

## Detailed Evidence from Code

### `loss_s` — Cross-entropy on source (uses source labels)

```python
label_s = batch["label_feature"][:num_node_s].long()
label_s_onehot = F.one_hot(label_s, self.num_classes)   # [450, 25]
loss_s = CrossEntropyLoss(label_s_onehot, node_seg_s)   # needs label + prediction
```

### `loss_t` — Entropy on target (NO labels at all)

```python
loss_t = EntropyLoss(node_seg_t)   # ← only takes predictions, no label argument
```

Compare the two function signatures:

```python
def CrossEntropyLoss(label, predict_prob, ...):     # needs label
    ce = -label * torch.log(predict_prob + epsilon)

def EntropyLoss(predict_prob, ...):                  # NO label parameter
    entropy = -predict_prob * torch.log(predict_prob + epsilon)
```

`EntropyLoss` computes self-information of the prediction distribution. It measures how uncertain the model is — low entropy means confident predictions, high entropy means uncertain. No ground truth is needed.

### `loss_adv` — Domain adversarial loss (NO labels at all)

```python
loss_adv = self.domain_adv(z_s_, z_t_, weight_s, weight_t)
```

Inside `DomainAdversarialLoss.forward`:

```python
d_label_s = torch.ones((f_s.size(0), 1))    # synthetic label: "source = 1"
d_label_t = torch.zeros((f_t.size(0), 1))   # synthetic label: "target = 0"
loss = 0.5 * (BCE(d_s, d_label_s, w_s) + BCE(d_t, d_label_t, w_t))
```

The only "labels" used here are synthetic domain labels (`1 = source`, `0 = target`) — not machining feature labels.

---

## Where Target Labels ARE Used (Monitoring Only)

```python
pred_t = torch.argmax(node_seg_t, dim=-1)
known_pos = torch.where(label_t < self.num_classes)
label_t_ = label_t[known_pos]
pred_t_ = pred_t[known_pos]
label_t_np = label_t_.long().detach().cpu().numpy()    # ← .detach() breaks gradient graph
pred_t_np = pred_t_.long().detach().cpu().numpy()
per_face_comp_t = (pred_t_np == label_t_np).astype(np.int)
self.log("train_acc_t", np.mean(per_face_comp_t), ...)  # ← logging only
```

Key observations:
1. **`.detach().cpu().numpy()`** — completely breaks the computation graph. No gradients can flow from this.
2. **`known_pos = torch.where(label_t < self.num_classes)`** — filters out unknown/open-set labels that exceed `num_classes`. This is needed because in open-set scenarios, some target faces may have labels the model wasn't trained to recognize.
3. The result is only used in `self.log(...)` — a TensorBoard metric for the researcher to watch.

---

## Why This Design?

This is the fundamental premise of **domain adaptation**: you have labeled data in the source domain but **unlabeled** (or only evaluation-labeled) data in the target domain.

The three training signals work together without target labels:

```
Source labels (supervised)
    ↓
loss_s: "Learn to classify machining features correctly on source data"
    +
loss_t: "Be confident about your predictions on target data" (no labels)
    +
loss_adv: "Make source and target features indistinguishable" (no labels)
    ↓
Result: The encoder produces domain-invariant features,
        so the classifier trained on source labels
        also works on target data.
```

The target labels in the batch are a convenience for the researcher — they let you track `per_face_accuracy_target` during training and use it as the checkpoint metric, without those labels ever influencing the model's weights.
