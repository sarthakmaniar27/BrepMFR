# Stage 2 Domain Adaptation Audit for BrepMFR

## Executive summary

Your data pipeline is now **very likely “conceptually correct”** for the BrepMFR feature contract (you’ve aligned normalization; you’ve confirmed face areas and edge lengths match the authors’ bins as multisets; and A3 asymmetry is now preserved). That shifts the remaining risk almost entirely to **Stage 2 training dynamics and code correctness**.

From the BrepMFR paper, Stage 2 is a **three-term joint objective** trained on **paired source+target batches**: a supervised source cross-entropy term, an unsupervised target entropy-minimization term, and a DANN-style adversarial domain discrimination term implemented via a **gradient reversal layer (GRL)**. The paper explicitly sets **α = 0.1** and **β = 0.3**. fileciteturn0file0

On the code side, the author’s released `transfer_model_og.py` contains at least two genuinely critical issues relative to the intended training dynamics:

- **Attention module is loaded but never optimized**, which prevents the inter-graph attention fusion from adapting during Stage 2. This is almost certainly a bug and can materially harm adaptation performance. fileciteturn0file1  
- **Validation “loss” is computed as 1/accuracy**, which is non-smooth, noisy, and can divide by zero; yet it is used as the learning-rate scheduler’s monitor. That is not paper-faithful and is unsafe. fileciteturn0file1

Your modified `transfer_model.py` fixes the first bug (adds attention params to the optimizer), fixes the second bug (uses a real combined validation objective), and additionally aligns optimizer hyperparameters with the paper’s AdamW settings and warmup concept. fileciteturn0file2turn0file0  
However, the modified code introduces **new risks** (hard-coded GRL schedule length, a Lightning `optimizer_step` override that may be version-fragile) and still contains **paper deviations** (classifier architecture and “softmax-before-CE” pattern).

The most likely remaining reasons Stage 2 may not improve target accuracy—assuming your data is correct—are:

- **Adversarial alignment implementation uncertainty** (we cannot verify `DomainAdversarialLoss` correctness because it is not in the uploaded code; your padding + weights strategy is only correct if that loss uses weights properly). fileciteturn0file2  
- **Entropy minimization destabilization** (entropy loss can sharpen wrong predictions early; ramping α is often needed even if the paper used a fixed α). fileciteturn0file0  
- **Batching/split invariants** (Stage 2 assumes strict ordering: first half of graphs are source, second half target; any collator/dataloader deviation silently breaks training). fileciteturn0file2turn0file4

What follows is a rigorous paper→code mapping, a full Stage 2 dry-run with tensor shapes, a training-dynamics audit, and a prioritized patch plan (diff-style) to make Stage 2 both paper-faithful and robust.

## Paper Stage 2 objective and intended training dynamics

### What the paper optimizes in Stage 2

The paper’s Stage 2 (domain adaptation stage) jointly trains the **pretrained** encoder and classifier together with an added **domain discriminator**, using labeled synthetic CAD graphs as source data and unlabeled real CAD graphs as target data. fileciteturn0file0

The stated optimization objective is:

\[
\min_{\theta_g,\theta_c,\theta_d}\;\; \mathcal{L}_{label} + \alpha\,\mathcal{L}_{entropy} + \beta\,\mathcal{L}_{adv}
\]
with **α = 0.1** and **β = 0.3** in their experiments. fileciteturn0file0

Key terms:

- **Label loss**: source-domain cross-entropy between node class probabilities and ground truth labels (their Eq. 11). fileciteturn0file0  
- **Entropy loss**: target-domain entropy minimization on predicted class distributions (their Eq. 15). fileciteturn0file0  
- **Adversarial loss**: domain discrimination loss where the discriminator tries to separate source vs target features, while the encoder is trained (via GRL) to make them indistinguishable (their Eq. 12–14). fileciteturn0file0

### GRL / adversarial dynamics the paper intends

The intended min-max game is:

- discriminator parameters minimize domain classification loss
- encoder parameters maximize that same loss

The paper uses a **Gradient Reversal Layer** that is identity in forward, but multiplies gradients by a negative constant on backward, allowing you to optimize a single “min” objective while still implementing the max-for-encoder behavior. fileciteturn0file0turn0search2

That is the same core mechanism as DANN (Ganin et al.), which is exactly what BrepMFR cites/uses conceptually. citeturn0search2

### Data flow and batching assumptions the paper implies

At Stage 2, each iteration provides:

- a **source graph** \( \mathcal{G}_s=(F_s,E_s) \) with labels per face node
- a **target graph** \( \mathcal{G}_t=(F_t,E_t) \) without labels (labels may exist only for evaluation) fileciteturn0file0

Both are passed through the same encoder to produce feature embeddings \(Z_s\) and \(Z_t\). The classifier outputs per-node class distributions; the discriminator outputs a domain prediction from the same embeddings. fileciteturn0file0

### Expected tensor shapes (paper-consistent)

Let:

- \(B\) = number of *source-target pairs* per batch (what your dataloader calls `batch_size`)
- total graphs in batch = \(2B\) (source graphs + target graphs)
- \(T\) = padded max number of face nodes per graph (within the batch)
- \(D\) = node embedding dim (paper uses 256; your code uses `args.dim_node=256`) fileciteturn0file2turn0file4
- \(K\) = number of classes

Then the encoder produces:

- node embeddings: typically \([2B, T, D]\) (sometimes with a global token giving \([2B, T{+}1, D]\))
- graph embeddings: \([2B, D]\)

and after masking out padded nodes:

- \(Z_s\): \([N_s, D]\), where \(N_s\) is total valid source nodes in the batch
- \(Z_t\): \([N_t, D]\)

Classifier outputs:

- \(P_s\): \([N_s, K]\)
- \(P_t\): \([N_t, K]\)

Losses:

- \(\mathcal{L}_{label}\) computed over \(N_s\)
- \(\mathcal{L}_{entropy}\) computed over \(N_t\)
- \(\mathcal{L}_{adv}\) computed over \(N_s\) and \(N_t\)

This is exactly how both code versions are structured at a high level. fileciteturn0file1turn0file2

## Code comparison: author original vs your modified, and deviations from the paper

This section has two parts:

- deviations from the paper that **exist in both** code versions
- deviations where your modified code differs from author code, with classification

### Deviations from the paper shared by both code versions

These are “paper vs both implementations” mismatches.

**Classifier architecture mismatch (benign to moderate risk)**  
The paper describes a 3-layer MLP head: FC(256→1024)→FC(1024→256)→FC(256→K) with BN and LeakyReLU. fileciteturn0file0  
Both `transfer_model_og.py` and `transfer_model.py` use a 4-layer head (256→512→512→256→K) with ReLU and Softmax in forward. fileciteturn0file1turn0file2  
Classification: **benign** if you treat the released code as ground truth; **moderate risk** if strict paper reproduction is required.

**Softmax-before-loss pattern (benign but not best-practice)**  
Both versions apply `softmax` in the classifier forward and then compute cross-entropy manually as `-y * log(p)`. fileciteturn0file1turn0file2  
That is mathematically consistent with the paper’s cross-entropy form (Eq. 11 uses probabilities) fileciteturn0file0, but it is less numerically stable than `nn.CrossEntropyLoss` on logits. PyTorch explicitly documents that `CrossEntropyLoss` expects **unnormalized logits** and internally applies log-softmax. citeturn2view1  
Classification: **benign** (works), but **recommended to change** for stability/gradients.

### Author original vs your modified: every meaningful deviation and classification

| Area | Author original (`transfer_model_og.py`) | Your modified (`transfer_model.py`) | Paper alignment | Classification |
|---|---|---|---|---|
| GRL behavior | Uses `DomainAdversarialLoss(domain_discri)` (GRL behavior hidden inside that module) fileciteturn0file1 | Injects `WarmStartGradientReverseLayer`, manual stepping, logs `grl_lambda` fileciteturn0file2 | Paper uses GRL but doesn’t specify warm-start | **Correct/beneficial**, but introduces schedule risk |
| Attention in optimizer | **Missing** attention params (attention never updated) fileciteturn0file1 | Adds attention params to optimizer param groups fileciteturn0file2 | Paper trains encoder+attention+classifier jointly | **Bug/critical** in original; **correct fix** in modified |
| Validation objective | `eval_loss = 1 / target_accuracy` fileciteturn0file1 | `eval_loss = Ls + β Ladv + α Lent` fileciteturn0file2 | Paper objective is loss sum | **Bug/critical** in original; **correct fix** in modified |
| Scheduler monitor | ReduceLROnPlateau(mode=min, monitor=`eval_loss`) where `eval_loss` is 1/acc fileciteturn0file1 | ReduceLROnPlateau(mode=max, monitor=`per_face_accuracy_target`) fileciteturn0file2turn0file4 | Paper says scheduler when loss stops decreasing fileciteturn0file0 | **Benign** but paper-deviating; recommend monitor a true loss |
| AdamW params | betas=(0.99,0.999), no weight_decay specified fileciteturn0file1 | betas=(0.9,0.999), eps=1e-8, weight_decay=0.01 fileciteturn0file2 | Paper’s exact AdamW params fileciteturn0file0 | **Correct/beneficial** |
| LR warmup | none | manual LR warmup in `optimizer_step` for 5000 steps fileciteturn0file2 | Paper uses warm-up ~50k steps fileciteturn0file0 | **Correct idea**, but **implementation risk** (Lightning hook fragility) |
| Label split assertions | none | asserts `num_node_s + num_node_t == total_labels` fileciteturn0file2 | Paper assumes correct split | **Correct/beneficial** (defensive) |
| Writing test outputs | writes `.txt` to a hard-coded server path fileciteturn0file1 | path-writing commented out (safer) fileciteturn0file2 | Not specified | **Benign / beneficial for portability** |

### Notable “paper deviations” that are likely hurting Stage 2 if still present elsewhere

These are not fully verifiable with the uploaded files, but they are high-yield suspects to audit next:

- **Batch ordering invariant**: Stage 2 assumes the batch is `[source graphs..., target graphs...]` so `.chunk(2, dim=0)` splits correctly. This must be guaranteed by `TransferDataset` and `collator_st`. fileciteturn0file2turn0file4  
- **Adversarial loss weighting correctness**: your code pads \(Z_s\) and \(Z_t\) to equal length and passes `weight_s`, `weight_t`. This is only correct if `DomainAdversarialLoss` truly ignores padded rows using those weights. fileciteturn0file2  

## End-to-end Stage 2 dry run

This is a dry-run of the actual execution path you are running:

`domain_adapt.py → dataloader → DomainAdapt.training_step / validation_step → optimizer/scheduler → checkpoint`

### Training entrypoint and high-level loop

`domain_adapt.py` constructs:

- `TransferDataset(... split="train")` and `TransferDataset(... split="val")`  
- dataloaders via `get_dataloader(batch_size=args.batch_size, ...)`  
- a Lightning `Trainer` with gradient clipping and a checkpoint callback that monitors `per_face_accuracy_target` in **max** mode. fileciteturn0file4  

Lightning training loop (automatic optimization):

1. fetch batch
2. call `training_step(batch, batch_idx)` → returns scalar loss
3. Lightning runs backward + optimizer step
4. after each val epoch, call validation loop, aggregate logged metrics
5. ReduceLROnPlateau uses the configured monitor to decide LR updates
6. `ModelCheckpoint` saves best checkpoints based on monitor metric fileciteturn0file4turn0file2

### Batch contract and where the source/target split happens

**Assumed collator output** (must be verified in your codebase):

- total graphs in batch: `2B`
- `batch["padding_mask"]`: shape `[2B, T]` with `False` for real nodes and `True` for padded positions (based on how you use it)
- `batch["label_feature"]`: shape `[Ns + Nt]`, flattened node labels in the same order that `node_pos_s/node_pos_t` will select nodes
- `batch["id"]`: shape `[2B]`, graph ids; first half should correspond to source, second half to target (your debug utilities already check this). fileciteturn0file2turn0file4  

**Source/target split occurs here**:

```python
node_emb = node_emb[0].permute(1, 0, 2)   # → [2B, T+1, D]
node_emb = node_emb[:, 1:, :]            # → [2B, T, D]
node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)   # each → [B, T, D]
padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # each → [B, T]
```

This split is the single most important batching invariant in Stage 2. fileciteturn0file2

### Encoder forward and embeddings

Let `D = args.dim_node = 256`, `K = args.num_classes`, `B = args.batch_size` (pairs), `T = max nodes in batch`.

1. Encoder forward:

- `node_emb, graph_emb = brep_encoder(batch, last_state_only=True)`
- expected: `node_emb[0]` is `[T+1, 2B, D]` (transformer uses a global token), and `graph_emb` is `[2B, D]` fileciteturn0file2

2. Remove global token and flatten valid nodes:

- `node_pos_s = where(padding_mask_s == False)` returns indices for real nodes  
- `node_z_s = node_emb_s[node_pos_s]` shape `[Ns, D]` where `Ns = sum_i n_i(source)`  
- similarly `node_z_t`: `[Nt, D]` fileciteturn0file2  

3. Graph embedding broadcast and attention fusion:

- `graph_emb_s`: `[B, D]` repeated to `[Ns, D]`  
- `z_s = attention([node_z_s, graph_z_s])` → `[Ns, D]`  
- `z_t` similarly `[Nt, D]` fileciteturn0file2  

### Loss branches and tensor shapes

#### Source supervised loss

- `node_seg_s = classifier(z_s)` → `[Ns, K]` (probabilities in current implementation) fileciteturn0file2  
- `label_s = batch["label_feature"][:Ns]` → `[Ns]`
- `loss_s = CrossEntropyLoss(one_hot(label_s), node_seg_s)`

#### Target entropy loss

- `node_seg_t = classifier(z_t)` → `[Nt, K]`  
- `loss_t = EntropyLoss(node_seg_t)`  
This matches the paper’s entropy minimization idea. fileciteturn0file0turn0file2  

#### Domain adversarial loss

Current implementation pads to equal length:

- `M = max(Ns, Nt)`
- `z_s_`: `[M, D]` (padded with rows of zeros)  
- `z_t_`: `[M, D]`  
- `weight_s`: `[M]` (1 for real rows, 0 for padded)  
- `weight_t`: `[M]`  
- `loss_adv = domain_adv(z_s_, z_t_, weight_s, weight_t)` fileciteturn0file2  

**Critical correctness condition**: `DomainAdversarialLoss` must use `weight_s/weight_t` to exclude padded rows from both the domain loss and domain accuracy; otherwise, padded zeros will leak into training.

In DANN-style systems, the GRL should be applied before the discriminator so gradients from `loss_adv` are reversed into the encoder’s features. citeturn0search2

#### Total loss

Your training loss matches the paper weights:

\[
\mathcal{L} = \mathcal{L}_{label} + 0.3\;\mathcal{L}_{adv} + 0.1\;\mathcal{L}_{entropy}
\]

and you explicitly label it “original paper’s values.” fileciteturn0file2turn0file0

### Backward, optimizer step, scheduler, checkpoint

- Lightning calls backward on the returned loss (automatic optimization). citeturn1view2  
- Optimizer step updates param groups; in your modified code, you also override `optimizer_step` to implement LR warmup for 5000 steps. fileciteturn0file2  
- ReduceLROnPlateau in Lightning requires a `"monitor"` metric when used; Lightning docs explicitly state this requirement. citeturn3search1turn3search5  
- In `domain_adapt.py`, `ModelCheckpoint` monitors `per_face_accuracy_target` and saves best models accordingly. fileciteturn0file4  

## Training dynamics audit and key failure modes

### Optimizer param groups and frozen/training modules

**Original author code problem**  
The original `configure_optimizers` updated encoder, classifier, and domain_adv but not attention. That means `self.attention` is effectively frozen throughout Stage 2 even though it is used to compute the very embeddings that all losses operate on. fileciteturn0file1  
This is exactly the kind of silent bug that prevents domain adaptation from doing the thing it is supposed to do.

**Your modified fix**  
You added attention to param groups, aligning with paper’s intent to jointly train the full encoder stack. fileciteturn0file2  

### Learning-rate / warmup alignment with paper

Paper implementation details specify AdamW parameters β₁=0.9, β₂=0.999, ε=1e−8, weight_decay=0.01, initial LR=0.001, and a warm-up stage (reported as “50,00 steps,” which is almost certainly 50,000). fileciteturn0file0

Your modified code matches β/ε/weight decay, uses warmup, but differs on:

- warmup length (5000 vs ~50k)
- LR for encoder/classifier (0.0001 vs 0.001)
- discriminator LR kept higher (0.001)

Those can be reasonable, but they are not strictly paper-faithful unless the paper explicitly used different LRs for different modules (not stated in the extracted text). fileciteturn0file0turn0file2  

### Scheduler correctness and metric choice

The paper says ReduceLROnPlateau is used “when the loss no longer decreases.” fileciteturn0file0  
Your modified code monitors target accuracy instead, and `domain_adapt.py` checkpoints on target accuracy as well. fileciteturn0file2turn0file4

This is not “wrong,” but it changes dynamics:

- accuracy is noisy (especially per-epoch when dataset is small)
- LR reductions keyed to accuracy plateaus can trigger too early/late
- loss components (especially adversarial) can improve while accuracy does not, and vice versa

Lightning docs also emphasize that ReduceLROnPlateau conditioning depends on the monitored metric being available at step time. citeturn3search1turn3search5

**Recommendation (paper-faithful + robust)**: monitor `eval_loss` (the true combined objective) for LR scheduling (mode=min), while checkpointing on target accuracy if you want the best-performing model artifact.

### GRL/discriminator behavior and stability

The paper’s GRL concept is standard DANN: identity forward, gradient reversal backward. fileciteturn0file0turn0search2

Your warm-start GRL is a stability improvement in many adversarial setups, but two implementation details can silently undermine it:

- `estimated_steps_per_epoch` is hard-coded in `__init__`, so if the real number of steps differs, the GRL schedule over/under-shoots and λ ramps too fast or too slow. fileciteturn0file2  
- The custom Lightning `optimizer_step` override can break across Lightning versions and can interact badly with AMP / gradient accumulation if your Trainer settings change. Lightning’s optimization docs note scheduler stepping rules differ under manual optimization, and advanced behaviors should be handled carefully. citeturn1view0turn2view2

### Loss scaling, numerical stability, and “softmax-before-CE”

Right now, classifier outputs probabilities and your CE is computed on `log(p+ε)`. This works, but PyTorch documents that `CrossEntropyLoss` expects unnormalized logits and internally applies log-softmax; applying softmax yourself tends to reduce gradient magnitude and can worsen numerical stability. citeturn2view1turn0search15

This becomes more important in Stage 2 because adversarial training already makes gradients “tug-of-war.” Any unnecessary gradient weakening can show up as “Stage 2 doesn’t move accuracy.”

### Entropy minimization risks

Entropy minimization is correct per paper (Eq. 15) fileciteturn0file0, but it is also a known destabilizer in unsupervised DA:

- early in training, target predictions are noisy
- minimizing entropy makes them confident even when wrong (confirmation bias)

If Stage 2 reduces target accuracy, this term is one of the first suspects. A common robustification is to ramp α from 0→0.1, or to apply entropy only above a confidence threshold.

### Adversarial padding/weighting correctness

The padding strategy is only correct if the weighting is implemented correctly *inside* `DomainAdversarialLoss`. Since that file isn’t in the uploaded set, you should treat this as an open critical item (see unit tests section).

A safer implementation avoids padding entirely by concatenating features and labels and computing BCE/CE directly on real samples only.

## Prioritized fixes with patch snippets

The fixes below are ordered by expected impact on Stage 2 accuracy and debugging value.

### Fix attention params in optimizer

You already implemented this; it is the correct fix and should not be reverted. fileciteturn0file2

If you want a clean diff-style patch (relative to the *original* author config):

```diff
--- a/transfer_model_og.py
+++ b/transfer_model.py
@@ def configure_optimizers(self):
- optimizer = torch.optim.AdamW(self.brep_encoder.parameters(), lr=0.0001, betas=(0.99, 0.999))
- optimizer.add_param_group({'params': self.classifier.parameters(), 'lr': 0.0001, 'betas': (0.99, 0.999)})
- optimizer.add_param_group({'params': self.domain_adv.parameters(), 'lr': 0.001, 'betas': (0.99, 0.999)})
+ optimizer = torch.optim.AdamW(self.brep_encoder.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
+ optimizer.add_param_group({'params': self.attention.parameters(), 'lr': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01})
+ optimizer.add_param_group({'params': self.classifier.parameters(), 'lr': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01})
+ optimizer.add_param_group({'params': self.domain_adv.parameters(), 'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01})
```

Paper optimizer hyperparams come directly from their implementation details. fileciteturn0file0turn0file2  

### Replace softmax+custom CE with logits + `nn.CrossEntropyLoss`

This is strongly recommended for stability and gradient strength. PyTorch documents that CE expects logits. citeturn2view1turn0search15

Patch concept (in both Stage 1 and Stage 2 code paths; shown here for this file):

```diff
--- a/transfer_model.py
+++ b/transfer_model.py
@@ class NonLinearClassifier(nn.Module):
-    def forward(self, inp):
+    def forward(self, inp):
         x = F.relu(self.bn1(self.linear1(inp)))
         x = self.dp1(x)
         x = F.relu(self.bn2(self.linear2(x)))
         x = self.dp2(x)
         x = F.relu(self.bn3(self.linear3(x)))
         x = self.dp3(x)
         x = self.linear4(x)
-        x = F.softmax(x, dim=-1)
-        return x
+        return x  # logits

@@ in DomainAdapt.training_step
- node_seg_s = self.classifier(z_s)  # probabilities
- label_s_onehot = F.one_hot(label_s, self.num_classes)
- loss_s = CrossEntropyLoss(label_s_onehot, node_seg_s)
+ logits_s = self.classifier(z_s)
+ loss_s = F.cross_entropy(logits_s, label_s)

- node_seg_t = self.classifier(z_t)
- loss_t = EntropyLoss(node_seg_t)
+ logits_t = self.classifier(z_t)
+ logp_t = F.log_softmax(logits_t, dim=-1)
+ p_t = logp_t.exp()
+ loss_t = -(p_t * logp_t).sum(dim=-1).mean()
```

This preserves argmax predictions (logits argmax == softmax argmax), but materially improves training numerical behavior.

### Fix validation loss/monitor to be paper-faithful

Your modified validation objective is correct; the remaining improvement is to make **LR scheduling** monitor a loss (mode=min), as the paper describes. fileciteturn0file0turn0file2

Suggested scheduler config:

```diff
- scheduler = ReduceLROnPlateau(optimizer, mode="max", ..., monitor="per_face_accuracy_target")
+ scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6)
...
- "monitor": "per_face_accuracy_target"
+ "monitor": "eval_loss"
```

Lightning explicitly requires the `"monitor"` field for ReduceLROnPlateau. citeturn3search1turn3search5

Keep checkpointing on `per_face_accuracy_target` if your goal is best target accuracy artifact; that’s an evaluation decision, not an optimization requirement. fileciteturn0file4

### Make α and β configurable from CLI

Paper uses α=0.1, β=0.3 fileciteturn0file0, but you will likely need to sweep for your SolidWorks-derived target domain.

Add to `domain_adapt.py`:

```diff
+ parser.add_argument("--alpha_entropy", type=float, default=0.1)
+ parser.add_argument("--beta_adv", type=float, default=0.3)
```

Then in `transfer_model.py`:

```diff
- loss = loss_s + 0.3 * loss_adv + 0.1 * loss_t
+ loss = loss_s + self.hparams.beta_adv * loss_adv + self.hparams.alpha_entropy * loss_t
```

### Remove hard-coded GRL schedule length and fragile `optimizer_step` override

Your warm-start GRL is a good idea, but make it robust:

- compute `max_iters` from Trainer at runtime (e.g., `self.trainer.estimated_stepping_batches`)  
- use a built-in warmup scheduler rather than overriding `optimizer_step`

Lightning’s docs emphasize that optimization hooks and scheduler stepping semantics differ across automatic/manual modes. citeturn1view0turn2view2

A robust pattern:

- remove `optimizer_step` override
- use `LambdaLR` for warmup (interval="step")
- use ReduceLROnPlateau on epoch for fine control

### Fix attention-bias double-add if present

This is **outside the uploaded files**, but based on your earlier code inspection you suspected attention bias may be added twice in the graph attention bias module. If that exists, it can distort attention logits.

Recommendation: audit the attention-bias construction; ensure base `attn_bias` is added exactly once, and masking semantics (`-inf`) are preserved.

### Add runtime assertions in dataset loader and ensure deterministic collator ordering

Again outside uploaded files, but required for Stage 2 correctness:

- assert `d2_distance.shape == (n,n,64)` and `angle_distance.shape == (n,n,64)`
- assert `edge_path.shape == (n,n,max_dist)` and values are in `[-1, num_edges-1]`
- assert A3 asymmetry for at least one pair per graph
- assert collator ordering: first B graphs are source, next B are target
- remove `drop_last=True` for validation/test loaders to make evaluation deterministic and complete

## Recommended training hyperparameters and diagnostics

### Hyperparameters to start with

Use the paper’s optimizer baseline as the starting point: AdamW β₁=0.9, β₂=0.999, ε=1e−8, weight_decay=0.01, ReduceLROnPlateau, warmup steps. fileciteturn0file0turn0file2

Then do a small sweep:

- **α (entropy)**: {0.0, 0.02, 0.05, 0.1}  
- **β (adv)**: {0.05, 0.1, 0.3, 0.5}  
- GRL schedule: fixed λ=1 vs warm-start ramp (if warm-start, make max_iters correct)

Also test a staged schedule:

- first N epochs: α=0 (no entropy), β ramp 0→0.3
- later epochs: α ramp 0→0.1

### Discriminator LR and update frequency

If domain discriminator accuracy stays near 1.0, it is overpowering the encoder; if it stays near 0.5 from the start with high loss, it may be too weak.

Lightning supports stepping optimizers at different frequencies (GAN-like schedules) in manual optimization. citeturn2view2  
Even with GRL, some setups benefit from:

- discriminator LR slightly higher than encoder (as you already do)
- optional extra discriminator steps early (2 D steps per 1 G step)

### Logging/visualizations to add (high value)

Minimum set for diagnosing DA:

- `loss_s`, `loss_t`, `loss_adv`, `total_loss` curves
- `domain_discriminator_accuracy` curve (should trend toward ~0.5 if alignment succeeds)
- GRL λ(t) curve if warm-start
- per-class target accuracy (already printed)
- confusion matrix per epoch (target)
- histogram of prediction confidences on target (entropy dynamics)
- checksums of A3 asymmetry rate in a random batch (ensure preprocessing invariants remain true)

## Unit tests and synthetic checks

These are small tests that catch the most common “Stage 2 silently wrong” failures.

### Collator ordering test

Create a fake batch with 2 source graphs and 2 target graphs with unique ids and verify:

- batch graph dimension is 4
- ids[0:2] are source, ids[2:4] are target
- `.chunk(2, dim=0)` partitions correctly

### Padding/label alignment test

Construct tiny graphs with known node counts, run collator, and assert:

- `sum(~padding_mask_s)` equals `num_node_s`
- `label_feature[:num_node_s]` equals the flattened source labels in node order
- your assert `num_node_s + num_node_t == total_labels` always holds (you already added this). fileciteturn0file2

### Adversarial loss padding/weights test

Feed synthetic `z_s` and `z_t` of different lengths and verify:

- padded rows do not affect loss value
- gradients for padded rows are exactly zero
- domain accuracy ignores padded rows

This validates the correctness of the “pad + weights” design.

### GRL gradient sign test

Toy network:

- feature extractor produces `z`
- discriminator predicts domain
- verify that enabling GRL flips the sign of gradients flowing into `z` relative to no-GRL

This test can be done with a single batch and deterministic seed.

### A3 asymmetry preservation test

For one graph:

- pick a pair (i,j) known to be asymmetric in JSON
- verify `angle_distance[i,j] != angle_distance[j,i]`
- after dataset→collator→batch→encoder input, verify these tensors are unchanged

This catches “accidentally symmetrized” bugs.

## Current vs recommended settings

### Optimizer/scheduler and loss config

| Setting | Current (modified) | Recommended (paper-faithful + robust) |
|---|---|---|
| AdamW betas/eps/wd | (0.9,0.999), eps 1e-8, wd 0.01 fileciteturn0file2turn0file0 | Keep |
| Encoder LR | 1e-4 fileciteturn0file2 | Sweep {1e-4, 3e-4, 1e-3} |
| Disc LR | 1e-3 fileciteturn0file2 | Start 1e-3; sweep {3e-4, 1e-3, 3e-3} |
| Warmup | custom `optimizer_step`, 5k steps fileciteturn0file2turn0file0 | Replace with scheduler-based warmup; align to ~50k if dataset scale matches paper |
| ReduceLROnPlateau | monitor target accuracy (mode=max) fileciteturn0file2turn0file4 | monitor `eval_loss` (mode=min) per paper; checkpoint on accuracy |
| Loss weights | α=0.1, β=0.3 hard-coded fileciteturn0file2turn0file0 | Make configurable; ramp α/β |
| CE formulation | softmax + manual CE fileciteturn0file2 | logits + `nn.CrossEntropyLoss` citeturn2view1 |

### Validation protocol

| Item | Current | Recommended |
|---|---|---|
| Val “loss” | true combined loss (good) fileciteturn0file2 | keep |
| Scheduler monitor | target accuracy | combined loss |
| drop_last (val/test) | unknown (depends on dataset code) | set `drop_last=False` for val/test |

## Mermaid diagrams

### Dataflow: Stage 2 forward and loss branches

```mermaid
flowchart LR
  A[Dataloader (TransferDataset)] --> B[collator_st: batch dict]
  B --> C[BrepEncoder f_g]
  C -->|node_emb, graph_emb| D[Split graphs: source/target chunk(2)]
  D --> E[Mask real nodes via padding_mask]
  E --> F[Repeat graph_emb per real node]
  F --> G[Inter-graph Attention fusion]
  G --> H[Node Classifier f_c]
  G --> I[GRL + Domain Discriminator f_d]

  H -->|Ps| J[L_label (source CE)]
  H -->|Pt| K[L_entropy (target entropy)]
  I --> L[L_adv (domain CE/BCE)]
  
  J --> M[Total loss]
  K --> M
  L --> M
```

### Training loop: Lightning execution

```mermaid
flowchart TD
  S[Trainer.fit] --> T[for each train batch]
  T --> U[training_step returns loss]
  U --> V[backward]
  V --> W[optimizer.step]
  W --> X[log train metrics]
  X --> T

  S --> Y[validation epoch]
  Y --> Z[validation_step computes losses/acc]
  Z --> AA[validation_epoch_end aggregates]
  AA --> AB[scheduler.step(monitor)]
  AB --> AC[ModelCheckpoint monitors metric]
  AC --> Y
```

## Prioritized fix timeline with effort/risk

| Priority | Fix | Effort | Risk | Why it matters |
|---|---|---:|---:|---|
| Short-term | Verify batching invariant (source first, target second) + add assertion | Low | Medium | If wrong, Stage 2 is fundamentally broken |
| Short-term | Confirm `DomainAdversarialLoss` correctly ignores padded rows using weights | Medium | High | Padding strategy can silently poison adversarial gradients |
| Short-term | Switch to logits + `nn.CrossEntropyLoss` + stable entropy computation | Medium | Medium | Stronger gradients, fewer numerical pathologies citeturn2view1 |
| Medium-term | Make α/β configurable + implement α/β ramps | Low | Low | Addresses entropy-minimization collapse risk |
| Medium-term | Replace custom warmup hook with scheduler-based warmup; compute steps dynamically | Medium | Medium | Removes Lightning-hook fragility; aligns with paper warmup fileciteturn0file0turn1view0 |
| Long-term | Consider switching adversarial loss to “no padding” concatenation formulation | Medium | Medium | Simplifies correctness and gradients |
| Long-term | Add torchmetrics-based confusion matrix and streaming accuracy (no giant lists) | Medium | Low | Improves speed and determinism |

---

### Open items (need confirmation from your full codebase)

These are necessary to complete a “whole-repo correctness proof,” but they are not in the uploaded file set:

- `TransferDataset` and `collator_st`: prove ordering/flattening invariants used by `.chunk(2)` and `label_feature` slicing
- `DomainAdversarialLoss`, `WarmStartGradientReverseLayer`, `DomainDiscriminator`: verify GRL placement, loss definition, and weight masking
- any graph attention bias module (A1/A2/A3 injection): confirm no double-add and correct masking

If you upload those modules (or paste them), I can extend this report into a full repository-level execution trace and pinpoint the exact failure mode that explains “Stage 2 doesn’t improve target accuracy.”