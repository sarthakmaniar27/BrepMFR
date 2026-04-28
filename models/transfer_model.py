# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import pathlib
import os

from .modules.brep_encoder import BrepEncoder
from .modules.utils.macro import *
from .modules.domain_adv.domain_discriminator import DomainDiscriminator
from .modules.domain_adv.dann import DomainAdversarialLoss
from .modules.domain_adv.grl import WarmStartGradientReverseLayer
from .brepseg_model import BrepSeg


class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(512, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=dropout)
        self.linear4 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dp3(x)
        x = self.linear4(x)
        x = F.softmax(x, dim=-1)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = self.dense_weight(stacked)
        weights = F.softmax(weights, dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)


def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)


class DomainAdapt(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = args.num_classes

        # Load pre-trained Stage 1 weights
        pre_trained_model = BrepSeg.load_from_checkpoint(args.pre_train)
        self.brep_encoder = pre_trained_model.brep_encoder
        self.attention = pre_trained_model.attention
        self.classifier = pre_trained_model.classifier

        # GRL schedule: lambda grows from 0 to ~0.96 over the full training run.
        # max_iters is set lazily on the first training step using
        # self.trainer.num_training_batches, which Lightning populates after the
        # DataLoader is created — it is not available here in __init__.
        # auto_step=False: iter_num is incremented manually in training_step only,
        # so validation forward passes do not advance the schedule.
        grl = WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1.,
            max_iters=1,   # placeholder — overwritten before first step in training_step
            auto_step=False,
        )
        domain_discri = DomainDiscriminator(args.dim_node, hidden_size=512)
        self.domain_adv = DomainAdversarialLoss(domain_discri, grl=grl)
        self._grl_configured = False  # flag: has max_iters been set yet?

        self.pred_s = []
        self.label_s = []
        self.pred_t = []
        self.label_t = []

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        self.brep_encoder.train()
        self.attention.train()
        self.classifier.train()
        self.domain_adv.train()

        # Configure GRL max_iters on the very first training step.
        # self.trainer.num_training_batches is set by Lightning after the DataLoader
        # is created, so it is guaranteed to be correct here.
        # self.trainer.max_epochs comes from the --max_epochs argument.
        if not self._grl_configured:
            steps_per_epoch = self.trainer.num_training_batches
            max_training_iters = self.trainer.max_epochs * steps_per_epoch
            self.domain_adv.grl.max_iters = max_training_iters
            self._grl_configured = True
            print(
                f"\n[GRL] max_iters set to {max_training_iters} "
                f"({self.trainer.max_epochs} epochs × {steps_per_epoch} batches/epoch)"
            )
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)

        # Split node embeddings into source and target halves -------------------
        # collator_st concatenates source graphs first, then target graphs.
        # chunk(2, dim=0) splits the batch dimension in half.
        node_emb = node_emb[0].permute(1, 0, 2)  # [batch*2, max_node+1, dim]
        node_emb = node_emb[:, 1:, :]            # [batch*2, max_node,   dim] — drop global virtual node
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)

        # Extract real (non-padded) node embeddings for source and target -------
        node_pos_s = torch.where(padding_mask_s == False)
        node_pos_t = torch.where(padding_mask_t == False)
        node_z_s = node_emb_s[node_pos_s]   # [total_source_nodes, dim]
        node_z_t = node_emb_t[node_pos_t]   # [total_target_nodes, dim]

        # Expand graph-level embedding to match per-node count ------------------
        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)

        num_nodes_per_graph_s = torch.sum(~padding_mask_s, dim=-1)  # [batch]
        graph_z_s = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
        z_s = self.attention([node_z_s, graph_z_s])

        num_nodes_per_graph_t = torch.sum(~padding_mask_t, dim=-1)  # [batch]
        graph_z_t = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
        z_t = self.attention([node_z_t, graph_z_t])

        # Node classification ---------------------------------------------------
        node_seg_s = self.classifier(z_s)  # [total_source_nodes, num_classes]
        node_seg_t = self.classifier(z_t)  # [total_target_nodes, num_classes]

        # Source supervised classification loss — L_label -----------------------
        num_node_s = node_seg_s.size(0)
        num_node_t = node_seg_t.size(0)

        assert num_node_s + num_node_t == batch["label_feature"].shape[0], (
            f"Label split mismatch: {num_node_s} + {num_node_t} "
            f"!= {batch['label_feature'].shape[0]}"
        )

        label_s = batch["label_feature"][:num_node_s].long()
        label_s_onehot = F.one_hot(label_s, self.num_classes)
        loss_s = CrossEntropyLoss(label_s_onehot, node_seg_s)
        self.log("train_loss_s", loss_s, on_step=False, on_epoch=True)

        # Target entropy minimisation loss — L_entropy --------------------------
        # Target labels are NOT used here. EntropyLoss operates on predictions only.
        label_t = batch["label_feature"][num_node_s:].long()  # kept for monitoring only
        loss_t = EntropyLoss(node_seg_t)
        self.log("train_loss_t", loss_t, on_step=False, on_epoch=True)

        # Domain adversarial loss — L_adv ---------------------------------------
        # Pad shorter side with zeros and use weight masks so padding nodes
        # do not contribute to the discriminator loss.
        max_num_node = max(num_node_s, num_node_t)
        pad_s = nn.ZeroPad2d((0, 0, 0, max_num_node - num_node_s))
        pad_t = nn.ZeroPad2d((0, 0, 0, max_num_node - num_node_t))
        z_s_ = pad_s(z_s)
        z_t_ = pad_t(z_t)
        weight_s = torch.zeros(max_num_node, device=z_s.device, dtype=z_s.dtype)
        weight_s[:num_node_s] = 1.0
        weight_t = torch.zeros(max_num_node, device=z_t.device, dtype=z_t.dtype)
        weight_t[:num_node_t] = 1.0
        loss_adv = self.domain_adv(z_s_, z_t_, weight_s, weight_t)

        # Advance GRL counter — training only, never during validation ----------
        self.domain_adv.grl.step()

        # Log GRL lambda for TensorBoard monitoring -----------------------------
        p = self.domain_adv.grl.iter_num / self.domain_adv.grl.max_iters
        lam = 2.0 / (1.0 + np.exp(-1.0 * p)) - 1.0
        self.log("grl_lambda", lam, on_step=False, on_epoch=True)

        domain_acc = self.domain_adv.domain_discriminator_accuracy
        self.log("train_loss_transfer", loss_adv, on_step=False, on_epoch=True)
        self.log("train_transfer_acc", domain_acc, on_step=False, on_epoch=True)

        # Per-face accuracy monitoring (source and target) ----------------------
        pred_s = torch.argmax(node_seg_s, dim=-1)
        pred_s_np = pred_s.long().detach().cpu().numpy()
        label_s_np = label_s.long().detach().cpu().numpy()
        per_face_comp_s = (pred_s_np == label_s_np).astype(np.int)
        self.log("train_acc_s", np.mean(per_face_comp_s), on_step=True, on_epoch=True)

        pred_t = torch.argmax(node_seg_t, dim=-1)
        known_pos = torch.where(label_t < self.num_classes)
        label_t_np = label_t[known_pos].long().detach().cpu().numpy()
        pred_t_np = pred_t[known_pos].long().detach().cpu().numpy()
        per_face_comp_t = (pred_t_np == label_t_np).astype(np.int)
        self.log("train_acc_t", np.mean(per_face_comp_t), on_step=True, on_epoch=True)

        # Joint loss — paper values: α=0.1, β=0.3 ------------------------------
        loss = loss_s + 0.3 * loss_adv + 0.1 * loss_t
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.attention.eval()
        self.classifier.eval()
        self.domain_adv.eval()

        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)

        node_emb = node_emb[0].permute(1, 0, 2)
        node_emb = node_emb[:, 1:, :]
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)

        node_pos_s = torch.where(padding_mask_s == False)
        node_pos_t = torch.where(padding_mask_t == False)
        node_z_s = node_emb_s[node_pos_s]
        node_z_t = node_emb_t[node_pos_t]

        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)

        num_nodes_per_graph_s = torch.sum(~padding_mask_s, dim=-1)
        graph_z_s = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(self.device)
        z_s = self.attention([node_z_s, graph_z_s])

        num_nodes_per_graph_t = torch.sum(~padding_mask_t, dim=-1)
        graph_z_t = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(self.device)
        z_t = self.attention([node_z_t, graph_z_t])

        node_seg_s = self.classifier(z_s)
        node_seg_t = self.classifier(z_t)

        num_node_s = node_seg_s.size(0)
        num_node_t = node_seg_t.size(0)

        assert num_node_s + num_node_t == batch["label_feature"].shape[0], (
            f"Label split mismatch: {num_node_s} + {num_node_t} "
            f"!= {batch['label_feature'].shape[0]}"
        )

        label_s = batch["label_feature"][:num_node_s].long()
        label_t = batch["label_feature"][num_node_s:num_node_s + num_node_t].long()

        label_s_onehot = F.one_hot(label_s, self.num_classes)
        loss_s = CrossEntropyLoss(label_s_onehot, node_seg_s)
        loss_t = EntropyLoss(node_seg_t)

        self.log("eval_loss_s", loss_s, on_step=False, on_epoch=True)
        self.log("eval_loss_t", loss_t, on_step=False, on_epoch=True)

        # Compute adversarial loss for monitoring only (GRL is not stepped here)
        max_num_node = max(num_node_s, num_node_t)
        pad_s = nn.ZeroPad2d((0, 0, 0, max_num_node - num_node_s))
        pad_t = nn.ZeroPad2d((0, 0, 0, max_num_node - num_node_t))
        z_s_ = pad_s(z_s)
        z_t_ = pad_t(z_t)
        weight_s = torch.zeros(max_num_node, device=z_s.device, dtype=z_s.dtype)
        weight_s[:num_node_s] = 1.0
        weight_t = torch.zeros(max_num_node, device=z_t.device, dtype=z_t.dtype)
        weight_t[:num_node_t] = 1.0
        loss_adv = self.domain_adv(z_s_, z_t_, weight_s, weight_t)
        self.log("eval_loss_transfer", loss_adv, on_step=False, on_epoch=True)

        # eval_loss = 1 / target_accuracy.
        # This converts the maximisation objective (accuracy) into a minimisation
        # objective so that ModelCheckpoint(monitor="eval_loss") with mode="min"
        # saves the checkpoint when target accuracy is highest.
        # The scheduler also monitors this signal with mode="min".
        pred_t_val = torch.argmax(node_seg_t, dim=-1)
        known_pos_val = torch.where(label_t < self.num_classes)
        label_t_known_val = label_t[known_pos_val].long().detach().cpu().numpy()
        pred_t_known_val = pred_t_val[known_pos_val].long().detach().cpu().numpy()
        per_face_comp_val = (pred_t_known_val == label_t_known_val).astype(np.int)
        target_acc = float(np.mean(per_face_comp_val))
        eval_loss = 1.0 / (target_acc + 1e-9)
        self.log("eval_loss", eval_loss, on_step=False, on_epoch=True)

        # Accumulate predictions for epoch-end accuracy computation -------------
        pred_s = torch.argmax(node_seg_s, dim=-1)
        pred_s_np = pred_s.long().detach().cpu().numpy()
        label_s_np = label_s.long().detach().cpu().numpy()
        for v in pred_s_np:
            self.pred_s.append(v)
        for v in label_s_np:
            self.label_s.append(v)

        pred_t = torch.argmax(node_seg_t, dim=-1)
        known_pos = torch.where(label_t < self.num_classes)
        label_t_np = label_t[known_pos].long().detach().cpu().numpy()
        pred_t_np = pred_t[known_pos].long().detach().cpu().numpy()
        for v in pred_t_np:
            self.pred_t.append(v)
        for v in label_t_np:
            self.label_t.append(v)

        return eval_loss

    def validation_epoch_end(self, val_step_outputs):
        # Source accuracy
        pred_s_np = np.array(self.pred_s)
        label_s_np = np.array(self.label_s)
        self.pred_s = []
        self.label_s = []
        per_face_comp_s = (pred_s_np == label_s_np).astype(np.int)
        self.log("per_face_accuracy_source", np.mean(per_face_comp_s))

        # Target overall accuracy
        pred_t_np = np.array(self.pred_t)
        label_t_np = np.array(self.label_t)
        self.pred_t = []
        self.label_t = []
        per_face_comp_t = (pred_t_np == label_t_np).astype(np.int)
        self.log("per_face_accuracy_target", np.mean(per_face_comp_t))

        # Target feature-only accuracy (label > 0, excludes stock)
        feature_pos = np.where(label_t_np > 0)
        feature_pred = pred_t_np[feature_pos]
        feature_label = label_t_np[feature_pos]
        per_face_comp_feature = (feature_pred == feature_label).astype(np.int)
        self.log("per_face_accuracy_target_feature", np.mean(per_face_comp_feature))

        # Per-class accuracy printed to console for monitoring
        print("num_classes: %s" % self.num_classes)
        per_class_acc = []
        for i in range(0, self.num_classes):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_pred = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp = (class_i_pred == class_i_label).astype(np.int)
                per_class_acc.append(np.mean(per_face_comp))
                print("class_%s_acc: %s" % (i + 1, np.mean(per_face_comp)))
        print("per_class_accuracy: %s" % np.mean(per_class_acc))

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.attention.eval()
        self.classifier.eval()
        self.domain_adv.eval()

        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)

        node_emb = node_emb[0].permute(1, 0, 2)
        node_emb = node_emb[:, 1:, :]
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)

        node_pos_s = torch.where(padding_mask_s == False)
        node_pos_t = torch.where(padding_mask_t == False)
        node_z_s = node_emb_s[node_pos_s]
        node_z_t = node_emb_t[node_pos_t]

        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)

        num_nodes_per_graph_s = torch.sum(~padding_mask_s, dim=-1)
        graph_z_s = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
        z_s = self.attention([node_z_s, graph_z_s])

        num_nodes_per_graph_t = torch.sum(~padding_mask_t, dim=-1)
        graph_z_t = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
        z_t = self.attention([node_z_t, graph_z_t])

        node_seg_s = self.classifier(z_s)
        node_seg_t = self.classifier(z_t)

        num_node_s = node_seg_s.size(0)
        pred_t = torch.argmax(F.softmax(node_seg_t, dim=-1), dim=-1)
        label_t = batch["label_feature"][num_node_s:].long()

        known_pos = torch.where(label_t < self.num_classes)
        label_t_np = label_t[known_pos].long().detach().cpu().numpy()
        pred_t_np = pred_t[known_pos].long().detach().cpu().numpy()

        for v in pred_t_np:
            self.pred_t.append(v)
        for v in label_t_np:
            self.label_t.append(v)

    def test_epoch_end(self, outputs):
        pred_t_np = np.array(self.pred_t)
        label_t_np = np.array(self.label_t)
        self.pred_t = []
        self.label_t = []

        per_face_comp_t = (pred_t_np == label_t_np).astype(np.int)
        self.log("per_face_accuracy_target", np.mean(per_face_comp_t))
        print("num_classes: %s" % self.num_classes)
        print("per_face_accuracy: %s" % np.mean(per_face_comp_t))

        # Feature-only accuracy (excludes stock label 0)
        feature_pos = np.where(label_t_np > 0)
        feature_pred = pred_t_np[feature_pos]
        feature_label = label_t_np[feature_pos]
        per_face_comp_feature = (feature_pred == feature_label).astype(np.int)
        self.log("per_face_accuracy_target_feature", np.mean(per_face_comp_feature))
        print("per_face_accuracy_feature: %s" % np.mean(per_face_comp_feature))

        # Per-class accuracy
        per_class_acc = []
        for i in range(0, self.num_classes):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp = (class_i_preds == class_i_label).astype(np.int)
                per_class_acc.append(np.mean(per_face_comp))
                print("class_%s_acc: %s" % (i + 1, np.mean(per_face_comp)))
        self.log("per_class_accuracy", np.mean(per_class_acc))
        print("per_class_accuracy: %s" % np.mean(per_class_acc))

        # IoU
        per_class_iou = []
        for i in range(0, self.num_classes):
            label_pos = np.where(label_t_np == i)
            pred_pos = np.where(pred_t_np == i)
            if len(pred_pos[0]) > 0 and len(label_pos[0]) > 0:
                class_i_preds = pred_t_np[label_pos]
                class_i_label = label_t_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int)
                Union = (class_i_preds != class_i_label).astype(np.int)
                class_i_preds_ = pred_t_np[pred_pos]
                class_i_label_ = label_t_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int)
                per_class_iou.append(
                    np.sum(Intersection) / (np.sum(Union) + np.sum(Intersection) + np.sum(Union_))
                )
        self.log("IoU", np.mean(per_class_iou))
        print("IoU: %s" % np.mean(per_class_iou))

    # -------------------------------------------------------------------------
    # Optimiser and scheduler
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        # Four param groups matching paper values (β1=0.9, β2=0.999, ε=1e-8, wd=0.01).
        # domain_adv uses a 10× higher LR so the discriminator learns faster than
        # the encoder — this is required for GRL training stability.
        optimizer = torch.optim.AdamW(
            self.brep_encoder.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        optimizer.add_param_group({
            "params": self.attention.parameters(),
            "lr": 0.0001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
        })
        optimizer.add_param_group({
            "params": self.classifier.parameters(),
            "lr": 0.0001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
        })
        optimizer.add_param_group({
            "params": self.domain_adv.parameters(),
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
        })

        # eval_loss = 1 / target_accuracy, so mode="min" is equivalent to
        # maximising target accuracy. patience=15 survives the natural oscillation
        # of the accuracy signal without firing prematurely.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=15,
            threshold=1e-4,
            threshold_mode="rel",
            min_lr=1e-5,
            cooldown=5,
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            # monitor="eval_loss" with mode="min" is correct because
            # eval_loss = 1 / target_accuracy — it decreases when accuracy rises.
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "eval_loss",
            },
        }

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu, using_native_amp, using_lbfgs,
    ):
        optimizer.step(closure=optimizer_closure)

        # Linear warmup for first 5000 steps — ramps LR from 0 to base value.
        # base_lrs must match param group order: encoder, attention, classifier, domain_adv.
        if self.trainer.global_step < 5000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
            base_lrs = [0.0001, 0.0001, 0.0001, 0.001]
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = lr_scale * base_lr
