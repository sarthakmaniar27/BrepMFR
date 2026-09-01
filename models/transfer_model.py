# -*- coding: utf-8 -*-
import json
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import pathlib
import os
import numpy as np

from .modules.brep_encoder import BrepEncoder
from .modules.utils.macro import *
from .modules.domain_adv.domain_discriminator import DomainDiscriminator
from .modules.domain_adv.dann import DomainAdversarialLoss
from .modules.domain_adv.grl import WarmStartGradientReverseLayer
from .brepseg_model import BrepSeg


def _load_priors_json(path: str, num_classes: int) -> np.ndarray:
    """Load class counts from a JSON file (scripts/training/compute_class_weights.py format)
    and return normalized class priors of shape [num_classes]."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    counts = np.asarray(data["counts"], dtype=np.float64)
    assert counts.shape[0] == num_classes, (
        f"Priors at {path} have {counts.shape[0]} classes, expected {num_classes}"
    )
    p = counts / max(1.0, counts.sum())
    return np.maximum(p, 1e-8)


def _compute_iwdan_weights(
    src_priors: np.ndarray,
    tgt_priors: np.ndarray,
    clip_max: float = 10.0,
) -> np.ndarray:
    """IWDAN per-class importance ratio w[c] = P_T(c)/P_S(c), normalized so
    that E_{y~P_S}[w(y)] = sum_c P_S(c) w(c) = 1 (preserves source mass).

    Tachet des Combes et al., NeurIPS 2020 — Importance-Weighted DANN under
    label shift. The clip prevents single rare-on-source classes from
    dominating the discriminator gradient.
    """
    w = tgt_priors / src_priors
    w = np.clip(w, 1.0 / clip_max, clip_max)
    norm = float((src_priors * w).sum())
    if norm > 0:
        w = w / norm
    return w.astype(np.float32)


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

        # Build the GRL with a schedule tied to ACTUAL training length, not the
        # dalib default of 1000 forwards (which saturates lambda<=>1.0 within
        # half an epoch on this dataset and gives the discriminator full power
        # before the encoder has any chance to adapt — a known cause of the
        # plateau under heavy label shift).
        #
        # max_iters defaults to roughly 0.5 * estimated_steps_per_epoch *
        # max_epochs, so lambda hits 1 around the midpoint of training and the
        # encoder gets a real warmup. Override with --grl_max_iters if needed.
        grl_max_iters = getattr(args, "grl_max_iters", 0)
        if grl_max_iters and grl_max_iters > 0:
            max_iters = int(grl_max_iters)
        else:
            est_steps_per_epoch = getattr(args, "estimated_steps_per_epoch", 2444)
            ramp_frac = getattr(args, "grl_ramp_frac", 0.5)
            max_iters = max(1, int(est_steps_per_epoch * args.max_epochs * ramp_frac))
        print(f"[Stage 2] GRL: alpha=1.0, lo=0.0, hi=1.0, "
              f"max_iters={max_iters} (auto_step=True)")

        grl = WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=1.0, max_iters=max_iters, auto_step=True,
        )
        domain_discri = DomainDiscriminator(args.dim_node, hidden_size=512)
        self.domain_adv = DomainAdversarialLoss(domain_discri, grl=grl)

        # IWDAN (Tachet des Combes et al. NeurIPS 2020): per-class importance
        # weights on the SOURCE side of the discriminator loss, so the
        # discriminator sees a re-weighted source distribution that has the
        # same class marginals as the target. This is the textbook fix for
        # DANN under label shift; without it, DANN can hurt target accuracy
        # (Zhao et al. ICML 2019 lower bound).
        self.iwdan_enabled = bool(getattr(args, "iwdan", False))
        if self.iwdan_enabled:
            src_priors_path = getattr(args, "iwdan_source_priors", None)
            tgt_priors_path = getattr(args, "iwdan_target_priors", None)
            assert src_priors_path and tgt_priors_path, (
                "IWDAN requires --iwdan_source_priors and --iwdan_target_priors"
            )
            src_priors = _load_priors_json(src_priors_path, self.num_classes)
            tgt_priors = _load_priors_json(tgt_priors_path, self.num_classes)
            iwdan_clip = float(getattr(args, "iwdan_clip", 10.0))
            iw = _compute_iwdan_weights(src_priors, tgt_priors, clip_max=iwdan_clip)
            self.register_buffer("iwdan_weights", torch.from_numpy(iw))
            print(f"[Stage 2] IWDAN ENABLED")
            print(f"  source priors from: {src_priors_path}")
            print(f"  target priors from: {tgt_priors_path}")
            print(f"  per-class importance w[c] = P_T(c)/P_S(c) (clipped to [{1.0/iwdan_clip:.3f}, {iwdan_clip}])")
            for c in range(self.num_classes):
                print(f"    class {c:2d}: src={100*src_priors[c]:6.3f}% "
                      f"tgt={100*tgt_priors[c]:6.3f}%  w={iw[c]:.4f}")
        else:
            print("[Stage 2] IWDAN disabled (vanilla DANN source weights).")

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

        with torch.profiler.record_function("brep_encoder"):
            node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)

        # Split node embeddings into source and target halves -------------------
        # collator_st concatenates source graphs first, then target graphs.
        # chunk(2, dim=0) splits the batch dimension in half.
        with torch.profiler.record_function("pool_attn_classifier_st"):
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

        with torch.profiler.record_function("loss_cls_entropy"):
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
        with torch.profiler.record_function("domain_adv"):
            max_num_node = max(num_node_s, num_node_t)
            pad_s = nn.ZeroPad2d((0, 0, 0, max_num_node - num_node_s))
            pad_t = nn.ZeroPad2d((0, 0, 0, max_num_node - num_node_t))
            z_s_ = pad_s(z_s)
            z_t_ = pad_t(z_t)
            weight_s = torch.zeros(max_num_node, device=z_s.device, dtype=z_s.dtype)
            if self.iwdan_enabled:
                # IWDAN: instead of all-ones on real source nodes, use per-class
                # importance weight P_T(y)/P_S(y). label_s is in [0, num_classes).
                iw = self.iwdan_weights.to(device=z_s.device, dtype=z_s.dtype)
                weight_s[:num_node_s] = iw[label_s]
            else:
                weight_s[:num_node_s] = 1.0
            weight_t = torch.zeros(max_num_node, device=z_t.device, dtype=z_t.dtype)
            weight_t[:num_node_t] = 1.0
            loss_adv = self.domain_adv(z_s_, z_t_, weight_s, weight_t)

            # Log GRL lambda each epoch for TensorBoard monitoring (matches authors'
            # default GRL: alpha=1, max_iters=1000, auto_step=True).
            grl = self.domain_adv.grl
            p = grl.iter_num / grl.max_iters
            lam = float(
                2.0 * (grl.hi - grl.lo) / (1.0 + np.exp(-grl.alpha * p))
                - (grl.hi - grl.lo) + grl.lo
            )
            self.log("grl_lambda", lam, on_step=False, on_epoch=True)

            domain_acc = self.domain_adv.domain_discriminator_accuracy
            self.log("train_loss_transfer", loss_adv, on_step=False, on_epoch=True)
            self.log("train_transfer_acc", domain_acc, on_step=False, on_epoch=True)

        # Per-face accuracy monitoring (source and target) ----------------------
        with torch.profiler.record_function("loss_total"):
            pred_s = torch.argmax(node_seg_s, dim=-1)
            pred_s_np = pred_s.long().detach().cpu().numpy()
            label_s_np = label_s.long().detach().cpu().numpy()
            per_face_comp_s = (pred_s_np == label_s_np).astype(np.int64)
            self.log("train_acc_s", np.mean(per_face_comp_s), on_step=True, on_epoch=True)

            pred_t = torch.argmax(node_seg_t, dim=-1)
            known_pos = torch.where(label_t < self.num_classes)
            label_t_np = label_t[known_pos].long().detach().cpu().numpy()
            pred_t_np = pred_t[known_pos].long().detach().cpu().numpy()
            per_face_comp_t = (pred_t_np == label_t_np).astype(np.int64)
            self.log("train_acc_t", np.mean(per_face_comp_t), on_step=True, on_epoch=True)

            # Joint loss — paper values: α=0.1, β=0.3 ------------------------------
            loss = loss_s + 0.3 * loss_adv + 0.1 * loss_t
            self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        current_lr = opt.param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.attention.eval()
        self.classifier.eval()
        self.domain_adv.eval()

        with torch.profiler.record_function("brep_encoder"):
            node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)

        with torch.profiler.record_function("pool_attn_classifier_st"):
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

        with torch.profiler.record_function("val_loss_terms"):
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
            pred_t_val = torch.argmax(node_seg_t, dim=-1)
            known_pos_val = torch.where(label_t < self.num_classes)
            label_t_known_val = label_t[known_pos_val].long().detach().cpu().numpy()
            pred_t_known_val = pred_t_val[known_pos_val].long().detach().cpu().numpy()
            per_face_comp_val = (pred_t_known_val == label_t_known_val).astype(np.int64)
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

        if (
            batch_idx == 0
            and self.trainer.current_epoch % 5 == 0
            and self.trainer.is_global_zero
        ):
            from models.tensorboard_media import tb_add_histogram

            mx = node_seg_t.max(dim=-1).values.detach().cpu()
            tb_add_histogram(
                self.trainer,
                "val/target_max_pred_prob_batch0",
                mx,
                int(self.trainer.global_step),
            )

        return eval_loss

    def on_validation_epoch_end(self):
        # Source accuracy
        pred_s_np = np.array(self.pred_s)
        label_s_np = np.array(self.label_s)
        self.pred_s = []
        self.label_s = []
        per_face_comp_s = (pred_s_np == label_s_np).astype(np.int64)
        self.log("per_face_accuracy_source", np.mean(per_face_comp_s))

        # Target overall accuracy
        pred_t_np = np.array(self.pred_t)
        label_t_np = np.array(self.label_t)
        self.pred_t = []
        self.label_t = []
        per_face_comp_t = (pred_t_np == label_t_np).astype(np.int64)
        self.log("per_face_accuracy_target", np.mean(per_face_comp_t))

        # Target feature-only accuracy (label > 0, excludes stock)
        feature_pos = np.where(label_t_np > 0)
        feature_pred = pred_t_np[feature_pos]
        feature_label = label_t_np[feature_pos]
        per_face_comp_feature = (feature_pred == feature_label).astype(np.int64)
        self.log("per_face_accuracy_target_feature", np.mean(per_face_comp_feature))

        # Per-class accuracy printed to console for monitoring
        print("num_classes: %s" % self.num_classes)
        per_class_acc = []
        for i in range(0, self.num_classes):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_pred = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp = (class_i_pred == class_i_label).astype(np.int64)
                per_class_acc.append(np.mean(per_face_comp))
                print("class_%s_acc: %s" % (i + 1, np.mean(per_face_comp)))
        print("per_class_accuracy: %s" % np.mean(per_class_acc))
        if len(per_class_acc) > 0:
            self.log("per_class_accuracy", float(np.mean(per_class_acc)))

        per_class_iou = []
        for i in range(0, self.num_classes):
            label_pos = np.where(label_t_np == i)[0]
            pred_pos = np.where(pred_t_np == i)[0]
            if len(pred_pos) > 0 and len(label_pos) > 0:
                class_i_preds = pred_t_np[label_pos]
                class_i_label = label_t_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int64)
                Union = (class_i_preds != class_i_label).astype(np.int64)
                class_i_preds_ = pred_t_np[pred_pos]
                class_i_label_ = label_t_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int64)
                per_class_iou.append(
                    float(
                        np.sum(Intersection)
                        / (
                            np.sum(Union)
                            + np.sum(Intersection)
                            + np.sum(Union_)
                            + 1e-9
                        )
                    )
                )
        if len(per_class_iou) > 0:
            self.log("IoU", float(np.mean(per_class_iou)))

        if self.trainer.is_global_zero:
            from models.tensorboard_media import log_segmentation_val_media

            log_segmentation_val_media(
                self.trainer,
                pred_s_np,
                label_s_np,
                self.num_classes,
                int(self.current_epoch),
                prefix="val/source",
            )
            log_segmentation_val_media(
                self.trainer,
                pred_t_np,
                label_t_np,
                self.num_classes,
                int(self.current_epoch),
                prefix="val/target",
            )
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
        pred_t = torch.argmax(node_seg_t, dim=-1)
        label_t = batch["label_feature"][num_node_s:].long()

        known_pos = torch.where(label_t < self.num_classes)
        label_t_np = label_t[known_pos].long().detach().cpu().numpy()
        pred_t_np = pred_t[known_pos].long().detach().cpu().numpy()

        for v in pred_t_np:
            self.pred_t.append(v)
        for v in label_t_np:
            self.label_t.append(v)

    def on_test_epoch_end(self):
        pred_t_np = np.array(self.pred_t)
        label_t_np = np.array(self.label_t)
        self.pred_t = []
        self.label_t = []

        per_face_comp_t = (pred_t_np == label_t_np).astype(np.int64)
        self.log("per_face_accuracy_target", np.mean(per_face_comp_t))
        print("num_classes: %s" % self.num_classes)
        print("per_face_accuracy: %s" % np.mean(per_face_comp_t))

        # Feature-only accuracy (excludes stock label 0)
        feature_pos = np.where(label_t_np > 0)
        feature_pred = pred_t_np[feature_pos]
        feature_label = label_t_np[feature_pos]
        per_face_comp_feature = (feature_pred == feature_label).astype(np.int64)
        self.log("per_face_accuracy_target_feature", np.mean(per_face_comp_feature))
        print("per_face_accuracy_feature: %s" % np.mean(per_face_comp_feature))

        # Per-class accuracy / precision / recall (one entry per training class)
        _default_names = {
            0: "Stock",
            1: "Thread",
            2: "Text",
            3: "Chamfer",
            4: "Fillet",
        }

        def _class_name(i: int) -> str:
            return _default_names.get(i, f"class_{i}")

        per_class_acc = []
        per_class_precision = []
        per_class_recall = []
        print("\nPer-class precision / recall / accuracy (target):")
        print(f"  {'class':<10} {'precision':>10} {'recall':>10} {'accuracy':>10} {'support':>10} {'pred_n':>10}")
        for i in range(0, self.num_classes):
            name = _class_name(i)
            label_pos = np.where(label_t_np == i)[0]
            pred_pos = np.where(pred_t_np == i)[0]
            support = int(len(label_pos))
            pred_n = int(len(pred_pos))
            tp = int(np.sum(pred_t_np[label_pos] == i)) if support > 0 else 0

            recall = float(tp / support) if support > 0 else 0.0
            precision = float(tp / pred_n) if pred_n > 0 else 0.0
            acc = recall

            per_class_acc.append(acc)
            per_class_precision.append(precision)
            per_class_recall.append(recall)

            self.log(f"test_class_{i}_acc", acc)
            self.log(f"test_class_{i}_precision", precision)
            self.log(f"test_class_{i}_recall", recall)
            if i in _default_names:
                self.log(f"test_{name}_precision", precision)
                self.log(f"test_{name}_recall", recall)
                self.log(f"test_{name}_acc", acc)

            print(
                f"  {name:<10} {precision:10.4f} {recall:10.4f} {acc:10.4f} "
                f"{support:10d} {pred_n:10d}"
            )

        if per_class_acc:
            macro_acc = float(np.mean(per_class_acc))
            macro_p = float(np.mean(per_class_precision))
            macro_r = float(np.mean(per_class_recall))
            self.log("per_class_accuracy", macro_acc)
            self.log("macro_precision", macro_p)
            self.log("macro_recall", macro_r)
            print("per_class_accuracy (macro): %s" % macro_acc)
            print("macro_precision: %s" % macro_p)
            print("macro_recall: %s" % macro_r)

        # IoU
        per_class_iou = []
        for i in range(0, self.num_classes):
            label_pos = np.where(label_t_np == i)
            pred_pos = np.where(pred_t_np == i)
            if len(pred_pos[0]) > 0 and len(label_pos[0]) > 0:
                class_i_preds = pred_t_np[label_pos]
                class_i_label = label_t_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int64)
                Union = (class_i_preds != class_i_label).astype(np.int64)
                class_i_preds_ = pred_t_np[pred_pos]
                class_i_label_ = label_t_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int64)
                per_class_iou.append(
                    np.sum(Intersection) / (np.sum(Union) + np.sum(Intersection) + np.sum(Union_))
                )
        self.log("IoU", np.mean(per_class_iou))
        print("IoU: %s" % np.mean(per_class_iou))

    # -------------------------------------------------------------------------
    # Optimiser and scheduler
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        # Match authors' published configuration EXACTLY:
        #   - 3 param groups (encoder, classifier, domain_adv) — attention is NOT optimized
        #   - betas=(0.99, 0.999) — note 0.99 first-moment, not the more common 0.9
        #   - encoder/classifier LR 1e-4; discriminator LR 1e-3 (10x asymmetric, intentional)
        #   - ReduceLROnPlateau patience=5, cooldown=2, min_lr=1e-6
        #   - NO Stage 2 warmup (Stage 1 has warmup, Stage 2 does not)
        # Source: https://github.com/zhangshuming0668/BrepMFR/blob/main/models/transfer_model.py
        optimizer = torch.optim.AdamW(
            self.brep_encoder.parameters(), lr=1e-4, betas=(0.99, 0.999),
        )
        optimizer.add_param_group({
            "params": self.classifier.parameters(),
            "lr": 1e-4,
            "betas": (0.99, 0.999),
        })
        optimizer.add_param_group({
            "params": self.domain_adv.parameters(),
            "lr": 1e-3,
            "betas": (0.99, 0.999),
        })

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            threshold_mode="rel",
            min_lr=1e-6,
            cooldown=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "eval_loss",
            },
        }
