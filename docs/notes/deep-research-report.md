# Stage 2 Domain Adaptation Audit for BrepMFR

## What the paper actually specifies for Stage 2

The paper’s Stage 2 is a **domain adaptation** phase stacked on top of a **Stage 1 supervised pretraining** phase. The key idea is: pretrain the **B-rep Encoder** \(f_g(\mathcal{G};\theta_g)\) and **Node Classifier** \(f_c(z;\theta_c)\) on **labeled synthetic CAD** (source). Then, adapt to **unlabeled real CAD** (target) by aligning feature distributions using a **Domain Discriminator** \(f_d(z;\theta_d)\) trained adversarially via a **Gradient Reversal Layer (GRL)**. fileciteturn0file0

In Stage 2, the paper defines three losses:

- **Source supervised loss** \( \mathcal{L}_{label} \): cross-entropy computed using source labels (paper Eq. 11). fileciteturn0file0  
- **Target entropy loss** \( \mathcal{L}_{entropy} \): entropy minimization on *target predictions* (paper Eq. 15), since target is unlabeled. fileciteturn0file0  
- **Adversarial domain loss** \( \mathcal{L}_{adv} \): domain discrimination loss that trains \(f_d\) to separate source vs target, and trains \(f_g\) to confuse \(f_d\), implemented using a GRL (paper Eq. 12–13 discussion). fileciteturn0file0  

The combined Stage 2 objective is explicitly:

\[
\min_{\theta_g,\theta_c,\theta_d} \ \mathcal{L}_{label} + \alpha \mathcal{L}_{entropy} + \beta \mathcal{L}_{adv}
\]
with **\(\alpha=0.1\)** and **\(\beta=0.3\)** (paper Eq. 17). fileciteturn0file0

The paper also states important training hyperparameters (in “Implementation detail”):

- **AdamW** with \(\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}\), weight decay \(=0.01\)  
- **Initial lr = 0.001**
- A **warm-up stage of 50,000 steps**
- ReduceLROnPlateau triggered when the **loss** stops decreasing  
- Trained for up to **200 epochs**, batch size **64** (as reported in their setup) fileciteturn0file0

That is the baseline contract you should compare code against.

---

## Code-to-paper comparison: what’s right and what’s wrong

You asked for a strict “right vs wrong as per paper,” *comparing* the author’s original `transfer_model_og.py` and your modified `transfer_model.py`.

### What both code versions do correctly (aligned with the paper)

Both the original and your modified code implement the **core Stage 2 structure** correctly:

- Load a Stage 1 checkpoint and reuse:
  - encoder \(f_g\)
  - attention fusion module
  - node classifier \(f_c\) fileciteturn0file1 fileciteturn0file2
- Compute **source CE loss** on source nodes only fileciteturn0file1 fileciteturn0file2
- Compute **target entropy loss** using target predictions only (no target labels in the loss) fileciteturn0file1 fileciteturn0file2
- Compute an **adversarial domain loss** using a discriminator and GRL wrapper \( \mathcal{L}_{adv} \) fileciteturn0file1 fileciteturn0file2
- Combine losses with the same coefficients as the paper:
  - \( \alpha=0.1 \)
  - \( \beta=0.3 \) fileciteturn0file0 fileciteturn0file1 fileciteturn0file2

So at the **objective definition** level, the implementation is consistent with the paper.

### What was wrong in the author’s original Stage 2 code

These items are not “paper interpretation” issues; they are implementation problems in the author’s original Stage 2 training loop:

- **Validation loss was not a loss.**  
  The original `validation_step` sets `eval_loss = 1.0 / target_accuracy`, then monitors that value for LR scheduling/checkpointing. This is noisy, discontinuous, and can blow up if accuracy hits zero. It is not the paper’s Stage 2 objective and is not a proper validation loss. fileciteturn0file1

- **The attention module was not optimized.**  
  In the original `configure_optimizers()`, only the encoder, classifier, and discriminator were optimized; `self.attention` was missing from optimizer param groups. That means the fusion module in Eq. (10) is frozen during Stage 2, which is not implied by the paper. fileciteturn0file1

- **Hyperparameters drifted away from the stated setup.**  
  The paper’s AdamW params and learning rate differ from the author’s original Stage 2 code (betas, weight decay, lr). That mismatch is not automatically “wrong,” but it **is** inconsistent with the paper’s reported setup. fileciteturn0file0 fileciteturn0file1

### What your modified Stage 2 code fixes (and why it is more paper-consistent)

Your modified `transfer_model.py` directly addresses the two biggest correctness issues above:

- Validation now computes and logs a **true Stage 2 objective**  
  \( \mathcal{L}_{label} + 0.1\mathcal{L}_{entropy} + 0.3\mathcal{L}_{adv} \) (and logs the components). fileciteturn0file2  
  This is much closer to paper Eq. (17). fileciteturn0file0

- The optimizer now includes the **attention module parameters** (fusion module is trainable in Stage 2). fileciteturn0file2

- Your AdamW betas / eps / weight decay are aligned with the paper’s stated AdamW config (at least structurally). fileciteturn0file0 fileciteturn0file2

So from a “bug removal” standpoint, your modifications are directionally correct.

### What is still *not* paper-matching in your modified code

These are the most important remaining deviations from what the paper reports:

- **Learning rate magnitude**  
  Paper: lr starts at **0.001**.  
  Your Stage 2: base lr is **0.0001** for encoder/attention/classifier and **0.001** for discriminator. fileciteturn0file0 fileciteturn0file2  
  This can matter a lot: if your encoder lr is too small, *alignment won’t move the representation enough* to improve target accuracy.

- **Warm-up length**  
  Paper: warm-up is **50,000 steps**.  
  Your Stage 2 warm-up implementation ramps over **5,000 steps**. fileciteturn0file0 fileciteturn0file2  
  That is a 10× difference in schedule shape.

- **Scheduler monitoring**  
  Paper: ReduceLROnPlateau triggers when the **loss** stops decreasing. fileciteturn0file0  
  Your Stage 2 scheduler monitors **target accuracy** in `mode="max"`. fileciteturn0file2 fileciteturn0file4  
  This is not necessarily “wrong,” but it is explicitly different from “loss value no longer decreases.”

- **GRL warm-start schedule vs paper GRL description**  
  The paper describes GRL as \( \mathcal{D}(z)=z \) with reversed gradient \( \partial\mathcal{D}/\partial z = -I \) (i.e., plain reversal). fileciteturn0file0  
  Your code uses a **WarmStartGradientReverseLayer** whose reversal strength grows over training. fileciteturn0file2  
  Warm-start GRL can be beneficial, but it is a design change and must be scheduled correctly.

- **“Steps per epoch” constant is hard-coded**  
  Your GRL max-iteration schedule depends on `estimated_steps_per_epoch = 1400` (comment says “use 2800 for batch size 32”). fileciteturn0file2  
  If that estimate is wrong, your GRL schedule is wrong.

That last point is the biggest remaining **code-side correctness risk** even after your fixes.

---

## Dry run: Stage 2 execution workflow end-to-end

This is a procedural dry run of what happens when you run Stage 2 training, based on `domain_adapt.py` + your modified `transfer_model.py`.

### Entry point: `domain_adapt.py`

1. You run Stage 2 training like:
   - `python domain_adapt.py train --source_path ... --target_path ... --pre_train ...` fileciteturn0file4

2. The script configures checkpoints and TensorBoard:
   - checkpoint metric is `per_face_accuracy_target` (mode max) fileciteturn0file4

3. It instantiates:
   - `DomainAdapt(args)` (the LightningModule) fileciteturn0file4
   - `TransferDataset(... split="train")` with `random_rotate=True` fileciteturn0file4
   - `TransferDataset(... split="val")` with `random_rotate=False` fileciteturn0file4

4. `trainer.fit(model, train_loader, val_loader)` begins Lightning training epochs. fileciteturn0file4

### Model construction: `DomainAdapt.__init__`

5. It loads Stage 1 checkpoint:
   - `pre_trained_model = BrepSeg.load_from_checkpoint(args.pre_train)` fileciteturn0file2
   and reuses:
   - `self.brep_encoder`
   - `self.attention`
   - `self.classifier` fileciteturn0file2

6. It builds the domain adversarial branch:
   - creates `DomainDiscriminator(dim_node=256, hidden_size=512)`
   - wraps it with `DomainAdversarialLoss(..., grl=WarmStartGRL)` fileciteturn0file2

### Training step computation: `DomainAdapt.training_step`

Each batch contains **source graphs and target graphs**. The model assumes you collate batches such that:

- first half in batch dimension = source graphs
- second half = target graphs

7. Forward through encoder:
   - `node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)` fileciteturn0file2

8. Remove the global token and split into source/target:
   - reshape to `[B, T, C]`
   - drop token 0
   - `node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)` fileciteturn0file2

9. Gather only real nodes using padding masks:
   - `node_pos_s = where(padding_mask_s == False)`
   - flatten into `node_z_s` / `node_z_t` fileciteturn0file2

10. Expand graph embedding per node and fuse:
   - repeat graph embedding for each node
   - `z_s = attention([node_z_s, graph_z_s])`, same for target fileciteturn0file2

11. Classify:
   - `node_seg_s = classifier(z_s)`  
   - `node_seg_t = classifier(z_t)` fileciteturn0file2  
   (Your classifier outputs probabilities, because it includes a `softmax()` inside.)

12. Compute losses:
   - Source label CE: `loss_s = CrossEntropyLoss(one_hot(label_s), node_seg_s)` fileciteturn0file2  
   - Target entropy: `loss_t = EntropyLoss(node_seg_t)` fileciteturn0file2  
   - Adversarial loss: pad z_s/z_t to same length, build weights, call `loss_adv = domain_adv(z_s_, z_t_, weight_s, weight_t)` fileciteturn0file2  

13. Step the GRL schedule:
   - `self.domain_adv.grl.step()` called once per training step fileciteturn0file2

14. Combine and return:
   - `loss = loss_s + 0.3*loss_adv + 0.1*loss_t` fileciteturn0file2

Lightning backpropagates this returned loss.

### Validation loop: `validation_step` and `validation_epoch_end`

15. `validation_step` repeats the forward pass and computes:
   - `val_obj = loss_s + 0.3*loss_adv + 0.1*loss_t`
   - logs `eval_loss` and components fileciteturn0file2

16. `validation_epoch_end` computes:
   - `per_face_accuracy_target` over all collected target predictions in the epoch
   - logs it (this drives checkpointing and your scheduler monitor) fileciteturn0file2 fileciteturn0file4

---

## Where Stage 2 can still be failing even with “data side fixed”

Given your statement that the feature pipeline is now internally consistent, the most likely code-side reasons Stage 2 doesn’t improve target accuracy are *not* “obvious crashes.” They are subtle training-dynamics issues.

### GRL schedule probably mismatched to your real training step count

Your GRL warm-start depends on:

- `max_training_iters = args.max_epochs * estimated_steps_per_epoch`
- `estimated_steps_per_epoch` is **hard-coded** (1400) fileciteturn0file2  

If your actual steps/epoch is not 1400, your GRL schedule is wrong:

- If actual steps/epoch > 1400, lambda rises too fast early and stays saturated (aggressive alignment early can harm).
- If actual steps/epoch < 1400, lambda rises too slowly and alignment barely engages.

This is one of the highest-priority Stage 2 issues to correct because it directly controls the feature extractor gradients.

### Encoder learning rate may be too small to adapt

Paper trains with lr = 0.001. fileciteturn0file0  
Your encoder/attention/classifier are at lr = 0.0001. fileciteturn0file2  

That can cause a common failure mode:

- Domain discriminator learns quickly (lr 0.001)
- Encoder barely moves (lr 0.0001)
- Discriminator accuracy stays high
- Target accuracy does not improve because representation alignment is weak

Your logs already include discriminator accuracy (`train_transfer_acc`). If that stays far above ~0.5 late in training, this is happening. fileciteturn0file2

### Scheduler is driven by target accuracy, not objective loss

Paper says ReduceLROnPlateau reacts to loss no longer decreasing. fileciteturn0file0  
You monitor and schedule on `per_face_accuracy_target` directly. fileciteturn0file2 fileciteturn0file4  

Accuracy is much noisier than loss, and it is not smooth. This can cause:

- premature LR drops
- LR oscillation
- weaker convergence in adversarial training

### Warm-up length differs by 10×

Paper warm-up: 50,000 steps. fileciteturn0file0  
Your warm-up: 5,000 steps implemented in `optimizer_step`. fileciteturn0file2  

If your plan is *paper replication*, this difference matters.

### BatchNorm mixing source and target can hurt adaptation

Your classifier and several MLP blocks use BatchNorm. Both domains are in the same batch and BN statistics update jointly during training. fileciteturn0file1 fileciteturn0file2  

This can either help or harm depending on distribution gap. If the gap is large, BN can destabilize alignment. The paper doesn’t discuss BN domain handling, so this is a “practical engineering” risk rather than a paper mismatch.

---

## What to verify next in your codebase to explain low Stage 2 accuracy

You asked to “dive into every part… dataloader, collator, embeddings, padding, GRL, losses, training/val/test.”

You have already fixed several likely data-side issues, but for completeness, here is a **targeted verification checklist** that maps directly to the Stage 2 failure modes above.

### Confirm GRL schedule correctness empirically

Use TensorBoard to plot:

- `grl_lambda`
- `train_transfer_acc` (domain discriminator accuracy)
- `train_loss_transfer` (domain loss) fileciteturn0file2  

Expected healthy dynamics in many DANN-style systems:
- `grl_lambda`: rises gradually across training
- discriminator accuracy: trends toward ~0.5 as encoder alignment improves (not a rule, but a strong diagnostic)

If lambda saturates extremely early or extremely late relative to training, fix `max_iters`.

### Confirm adversarial loss ignores padded rows properly

Your code pads features to `max_num_node` and uses weights. fileciteturn0file2  
You must confirm inside `DomainAdversarialLoss` that:

- padded rows contribute ~0 gradient
- the loss is normalized by sum of weights (or equivalent)

If it is not, the discriminator may learn artifacts from padding.

This requires inspecting `models/modules/domain_adv/dann.py` and the discriminator implementation (those files were not included in the uploads you provided here).

### Confirm Stage 2 optimizer is actually updating what you think

Your modified optimizer includes all key modules now (encoder, attention, classifier, discriminator). fileciteturn0file2  

The next check is: confirm gradients are non-zero for:
- encoder layers
- attention fusion
- discriminator layers

A quick debugging step is printing gradient norms once per epoch.

---

## Bottom-line judgment

- Your modifications fixed two of the most serious Stage 2 engineering problems in the author’s original code: the **invalid validation loss** and the **missing attention optimizer parameters**. fileciteturn0file1 fileciteturn0file2  
- Your Stage 2 loss weights match the paper’s stated \(\alpha=0.1\), \(\beta=0.3\). fileciteturn0file0 fileciteturn0file2  
- The most likely remaining reason Stage 2 is not improving accuracy is **training dynamics**: GRL schedule calibration, encoder LR magnitude, and scheduler monitoring choice—each of which is currently not strictly aligned with the paper’s reported setup. fileciteturn0file0 fileciteturn0file2  

If you want me to complete the “dive into every file” audit exactly as you requested (collator ordering, dataset batching contract, brep_encoder feature formation, domain_adv loss normalization, etc.), upload these additional files and I will do the same paper-to-code correctness pass on them:

- `data/dataset.py`, `data/collator.py`
- `models/modules/domain_adv/dann.py`, `models/modules/domain_adv/domain_discriminator.py`, `models/modules/domain_adv/grl.py`
- `models/modules/brep_encoder.py` and encoder layer files

Right now, with the uploaded set, the strongest actionable suspect is the **hard-coded GRL max-iters / step schedule** and the **encoder LR being far below paper**—both are code-side issues entirely independent of your geometry kernel differences.