# -*- coding: utf-8 -*-
import json
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import pathlib
import os

from .modules.brep_encoder import BrepEncoder
from .modules.utils.macro import *

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

    def forward_logits(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dp3(x)
        return self.linear4(x)

    def forward(self, inp):
        # Keep the public/inference contract (probabilities), while training uses
        # forward_logits + fused log-softmax losses below.
        return F.softmax(self.forward_logits(inp), dim=-1)


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


def FocalLoss(label, predict_prob, gamma=2.0, class_level_weight=None, epsilon=1e-12):
    """Focal Loss (Lin et al., ICCV 2017) for softmax probability outputs.

    Dynamically down-weights easy examples via a (1 - p_t)^gamma modulating
    factor.  When gamma=0 this reduces to standard cross-entropy.  gamma=2.0
    is the paper's recommended default.

    Works with the existing class_level_weight system: the per-class weight
    scales the focal-modulated loss for each class, giving both the
    easy-example suppression of Focal Loss AND the class-frequency correction
    of weighted CE.

    Args:
        label:              One-hot encoded labels [N, C]
        predict_prob:       Softmax probabilities   [N, C]
        gamma:              Focusing parameter (0 = CE, 2.0 recommended)
        class_level_weight: Per-class weight tensor [C] (optional)
        epsilon:            Numerical stability constant
    """
    N, C = label.size()
    assert predict_prob.size() == (N, C), 'fatal error: dimension mismatch!'

    # p_t = probability assigned to the TRUE class for each sample
    p_t = (label * predict_prob).sum(dim=-1)           # [N]
    focal_weight = (1.0 - p_t) ** gamma                # [N]  high when wrong, ~0 when right
    ce = -torch.log(p_t + epsilon)                      # [N]
    loss = focal_weight * ce                            # [N]

    if class_level_weight is not None:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        # Per-sample class weight: pick the weight of the true class
        true_class = label.argmax(dim=-1)                   # [N]
        cw = class_level_weight.squeeze(0)[true_class]      # [N]
        loss = loss * cw

    return loss.sum() / float(N)


def ClassificationLossFromLogits(
    labels,
    logits,
    *,
    loss_type="ce",
    focal_gamma=2.0,
    class_level_weight=None,
):
    """Numerically stable equivalent of the legacy softmax + one-hot losses."""
    if loss_type == "focal":
        log_p = F.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - log_pt.exp()) ** focal_gamma) * log_pt
        if class_level_weight is not None:
            loss = loss * class_level_weight[labels]
        return loss.sum() / labels.numel()
    return F.cross_entropy(
        logits,
        labels,
        weight=class_level_weight,
        reduction="sum",
    ) / labels.numel()


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = self.dense_weight(stacked)
        weights = F.softmax(weights, dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


class BrepSeg(pl.LightningModule):
    # def __init__(self, args):
    #     super().__init__()
    #     self.save_hyperparameters()
    #     self.num_classes = args.num_classes

    #     self.brep_encoder = BrepEncoder(
    #         # < for graphormer
    #         num_degree=128,  # number of in degree types in the graph
    #         num_spatial=64,  # number of spatial types in the graph
    #         num_edge_dis=64,  # number of edge dis types in the graph
    #         edge_type="multi_hop",  # edge type in the graph "multi_hop"
    #         multi_hop_max_dist=16,  # max distance of multi-hop edges
    #         # >
    #         num_encoder_layers=args.n_layers_encode,  # num encoder layers
    #         embedding_dim=args.dim_node,  # encoder embedding dimension
    #         ffn_embedding_dim=args.d_model,  # encoder embedding dimension for FFN
    #         num_attention_heads=args.n_heads,  # num encoder attention heads
    #         dropout=args.dropout,  # dropout probability
    #         attention_dropout=args.attention_dropout,  # dropout probability for"attention weights"
    #         activation_dropout=args.act_dropout,  # dropout probability after"activation in FFN"
    #         layerdrop=0.1,
    #         encoder_normalize_before=True,  # apply layernorm before each encoder block
    #         pre_layernorm=True,
    #         # apply layernorm before self-attention and ffn. Without this, post layernorm will used
    #         apply_params_init=True,  # use custom param initialization for Graphormer
    #         activation_fn="gelu",  # activation function to use
    #     )

    #     self.attention = Attention(args.dim_node)

    #     self.classifier = NonLinearClassifier(args.dim_node, args.num_classes, args.dropout)

    #     self.pred = []
    #     self.label = []



    #    # Optional: freeze encoder for first few epochs when fine-tuning 
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = args.num_classes

        self.brep_encoder = BrepEncoder(
            # < for graphormer
            num_degree=128,  # number of in degree types in the graph
            num_spatial=64,  # number of spatial types in the graph
            num_edge_dis=64,  # number of edge dis types in the graph
            edge_type="multi_hop",  # edge type in the graph "multi_hop"
            multi_hop_max_dist=16,  # max distance of multi-hop edges
            # >
            num_encoder_layers=args.n_layers_encode,  # num encoder layers
            embedding_dim=args.dim_node,  # encoder embedding dimension
            ffn_embedding_dim=args.d_model,  # encoder embedding dimension for FFN
            num_attention_heads=args.n_heads,  # num encoder attention heads
            dropout=args.dropout,  # dropout probability
            attention_dropout=args.attention_dropout,  # dropout probability for "attention weights"
            activation_dropout=args.act_dropout,  # dropout probability after "activation in FFN"
            layerdrop=0.1,
            encoder_normalize_before=True,  # apply layernorm before each encoder block
            pre_layernorm=True,
            # apply layernorm before self-attention and ffn. Without this, post layernorm will used
            apply_params_init=True,  # use custom param initialization for Graphormer
            activation_fn="gelu",  # activation function to use
            max_nodes_for_a3=getattr(args, "max_nodes_for_a3", None),
        )

        self.attention = Attention(args.dim_node)
        self.classifier = NonLinearClassifier(args.dim_node, args.num_classes, args.dropout)

        self.pred = []
        self.label = []
        # Validation metrics stay on-device and synchronize once per epoch,
        # instead of copying predictions to NumPy after every batch.
        self.register_buffer(
            "_val_confusion",
            torch.zeros(self.num_classes, self.num_classes, dtype=torch.long),
            persistent=False,
        )

        # Optional: freeze encoder for first few epochs when fine-tuning
        self.warmup_freeze_epochs = getattr(args, "warmup_freeze_epochs", 0)
        self.learning_rate = float(getattr(args, "learning_rate", 0.002))
        raw_a1_a3_lr = getattr(args, "a1_a3_learning_rate", None)
        self.a1_a3_learning_rate = (
            self.learning_rate if raw_a1_a3_lr is None else float(raw_a1_a3_lr)
        )
        self.optimizer_warmup_steps = int(getattr(args, "optimizer_warmup_steps", 5000))
        self.a1_a3_ramp_epochs = int(getattr(args, "a1_a3_ramp_epochs", 0))
        self.a1_a3_start_scale = float(getattr(args, "a1_a3_start_scale", 0.1))
        self.check_val_every_n_epoch = max(
            1, int(getattr(args, "check_val_every_n_epoch", 1))
        )
        self.fused_adamw = bool(getattr(args, "fused_adamw", False))
        self.batchnorm_finetune_mode = str(
            getattr(args, "batchnorm_finetune_mode", "update")
        )
        if self.batchnorm_finetune_mode not in {
            "update",
            "freeze_stats",
            "freeze_all",
        }:
            raise ValueError(
                "batchnorm_finetune_mode must be one of "
                "{'update', 'freeze_stats', 'freeze_all'}, got "
                f"{self.batchnorm_finetune_mode!r}"
            )
        self._frozen_batchnorm_reference = None

        # ---------------------------------------------------------
        # Loss function selection
        # ---------------------------------------------------------
        self.loss_type = getattr(args, "loss_type", "ce")
        self.focal_gamma = float(getattr(args, "focal_gamma", 2.0))
        if self.loss_type == "focal":
            print(f"\nUsing Focal Loss (gamma={self.focal_gamma})")
        else:
            print(f"\nUsing Cross-Entropy Loss")

        # ---------------------------------------------------------
        # Class-balanced loss weights
        # ---------------------------------------------------------
        # Stage 1 source training data is dominated by class 0 (~58% stock),
        # which produces an over-confident classifier and label-shift on target.
        # If --class_weights_path is provided, we load per-class weights from
        # that JSON (computed by scripts/training/compute_class_weights.py) and
        # apply them to BOTH the training-step and validation-step loss so
        # eval_loss reflects the true training objective (weighted CE or Focal
        # Loss). The LR scheduler monitors per_class_accuracy (macro-averaged)
        # instead of eval_loss to avoid the majority-class-dominated signal.
        self.class_weights_path = getattr(args, "class_weights_path", None)
        self.reuse_checkpoint_class_weights = bool(
            getattr(args, "reuse_checkpoint_class_weights", False)
        )
        if self.reuse_checkpoint_class_weights and self.class_weights_path:
            raise ValueError(
                "Use either reuse_checkpoint_class_weights or "
                "class_weights_path, not both"
            )
        if self.class_weights_path and not pathlib.Path(
            self.class_weights_path
        ).expanduser().is_file():
            print(
                f"\nclass_weights_path not found ({self.class_weights_path}); "
                f"using placeholder weights (checkpoint state_dict restores buffers)."
            )
            self.class_weights_path = None
        if self.reuse_checkpoint_class_weights:
            # A placeholder with the correct shape is registered now. The
            # embedded checkpoint buffer is copied during selective pre-load
            # and restored again by Lightning's checkpoint loader.
            weights = torch.ones(self.num_classes, dtype=torch.float32)
            self.use_class_weights = True
            print("\nReusing class weights embedded in the checkpoint.")
        elif self.class_weights_path:
            with open(self.class_weights_path, "r", encoding="utf-8") as f:
                cw = json.load(f)
            assert cw["num_classes"] == self.num_classes, (
                f"class_weights JSON num_classes={cw['num_classes']} != "
                f"model num_classes={self.num_classes}"
            )
            weights = torch.tensor(cw["weights"], dtype=torch.float32)
            self.use_class_weights = True
            print(f"\nLoaded class weights from: {self.class_weights_path}")
            print(f"  method={cw['method']} alpha={cw['alpha']} "
                  f"min={weights.min():.4f} max={weights.max():.4f} "
                  f"mean={weights.mean():.4f}")
        else:
            weights = torch.ones(self.num_classes, dtype=torch.float32)
            self.use_class_weights = False
        # Register as buffer so it moves to GPU and round-trips through
        # checkpoints. Even when unused (all 1.0) this keeps the model state
        # shape consistent.
        self.register_buffer("class_weights", weights)

        # ---------------------------------------------------------
        # Load pretrained Stage-1 checkpoint selectively
        # - load brep_encoder
        # - load attention
        # - load only classifier weights whose shapes match
        # ---------------------------------------------------------
        if getattr(args, "pre_train", None):
            print(f"\nLoading pretrained checkpoint from: {args.pre_train}")
            ckpt = torch.load(args.pre_train, map_location="cpu", weights_only=False)

            # Lightning checkpoints usually store model weights in "state_dict"
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

            # -------------------------
            # 1) Load brep_encoder
            # -------------------------
            encoder_state = {
                k.replace("brep_encoder.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("brep_encoder.")
            }

            enc_msg = self.brep_encoder.load_state_dict(encoder_state, strict=False)
            print("Loaded brep_encoder")
            print("  missing keys    :", enc_msg.missing_keys)
            print("  unexpected keys :", enc_msg.unexpected_keys)

            # -------------------------
            # 2) Load attention
            # -------------------------
            attention_state = {
                k.replace("attention.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("attention.")
            }

            att_msg = self.attention.load_state_dict(attention_state, strict=False)
            print("Loaded attention")
            print("  missing keys    :", att_msg.missing_keys)
            print("  unexpected keys :", att_msg.unexpected_keys)

            # -------------------------
            # 3) Partially load classifier
            #    only matching shapes
            # -------------------------
            classifier_state = {
                k.replace("classifier.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("classifier.")
            }

            current_classifier_state = self.classifier.state_dict()
            filtered_classifier_state = {}
            skipped_classifier_keys = []

            for k, v in classifier_state.items():
                if k in current_classifier_state and current_classifier_state[k].shape == v.shape:
                    filtered_classifier_state[k] = v
                else:
                    skipped_classifier_keys.append(k)

            current_classifier_state.update(filtered_classifier_state)
            cls_msg = self.classifier.load_state_dict(current_classifier_state, strict=False)

            print("Partially loaded classifier")
            print("  loaded keys     :", list(filtered_classifier_state.keys()))
            print("  skipped keys    :", skipped_classifier_keys)
            print("  missing keys    :", cls_msg.missing_keys)
            print("  unexpected keys :", cls_msg.unexpected_keys)

            if self.reuse_checkpoint_class_weights:
                embedded_weights = state_dict.get("class_weights")
                if embedded_weights is None:
                    raise ValueError(
                        "reuse_checkpoint_class_weights was requested, but the "
                        "pretrained checkpoint has no class_weights buffer"
                    )
                if embedded_weights.shape != self.class_weights.shape:
                    raise ValueError(
                        "Checkpoint class_weights shape mismatch: "
                        f"{tuple(embedded_weights.shape)} vs "
                        f"{tuple(self.class_weights.shape)}"
                    )
                self.class_weights.copy_(embedded_weights)
                print(
                    "Reused checkpoint class weights: "
                    f"{self.class_weights.detach().cpu().tolist()}"
                )

            # -------------------------
            # 4) Optional encoder freeze
            # -------------------------
            if self.warmup_freeze_epochs > 0:
                print(f"Freezing brep_encoder for first {self.warmup_freeze_epochs} epoch(s)")
                for p in self.brep_encoder.parameters():
                    p.requires_grad = False

        # A lite checkpoint contains the A1/A3 modules but they never received
        # gradients. Start their additive attention contribution gently, then
        # increase it at each epoch boundary.
        initial_a1_a3_scale = (
            self.a1_a3_start_scale if self.a1_a3_ramp_epochs > 0 else 1.0
        )
        self.brep_encoder.graph_attn_bias.set_a1_a3_scale(initial_a1_a3_scale)
        if self.a1_a3_ramp_epochs > 0:
            print(
                "A1/A3 gradual activation: "
                f"start_scale={initial_a1_a3_scale:.3f}, "
                f"ramp_epochs={self.a1_a3_ramp_epochs}"
            )
        self._configure_batchnorm_finetune()

    def _batchnorm_modules(self):
        batchnorm_base = nn.modules.batchnorm._BatchNorm
        return [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, batchnorm_base)
        ]

    def _configure_batchnorm_finetune(self):
        if self.batchnorm_finetune_mode == "update":
            return
        modules = self._batchnorm_modules()
        frozen_affine_tensors = 0
        if self.batchnorm_finetune_mode == "freeze_all":
            for _, module in modules:
                for parameter in (module.weight, module.bias):
                    if parameter is not None:
                        parameter.requires_grad = False
                        frozen_affine_tensors += 1
        self._enforce_batchnorm_finetune()
        print(
            "\nBatchNorm fine-tune policy: "
            f"mode={self.batchnorm_finetune_mode}, "
            f"modules={len(modules)}, "
            f"frozen_affine_tensors={frozen_affine_tensors}",
            flush=True,
        )

    def _enforce_batchnorm_finetune(self):
        if self.batchnorm_finetune_mode == "update":
            return
        # eval() on only the BatchNorm modules preserves their pretrained
        # running statistics while the rest of the network remains in train
        # mode (Dropout, Graphormer layers, etc.).
        for _, module in self._batchnorm_modules():
            module.eval()

    def _capture_batchnorm_reference(self):
        if self.batchnorm_finetune_mode == "update":
            self._frozen_batchnorm_reference = None
            return
        reference = {}
        for name, module in self._batchnorm_modules():
            if module.running_mean is not None:
                reference[f"{name}.running_mean"] = (
                    module.running_mean.detach().cpu().clone()
                )
            if module.running_var is not None:
                reference[f"{name}.running_var"] = (
                    module.running_var.detach().cpu().clone()
                )
            if module.num_batches_tracked is not None:
                reference[f"{name}.num_batches_tracked"] = (
                    module.num_batches_tracked.detach().cpu().clone()
                )
        self._frozen_batchnorm_reference = reference

    def _assert_batchnorm_reference_unchanged(self):
        reference = self._frozen_batchnorm_reference
        if reference is None:
            return
        current = dict(self.named_buffers())
        changed = [
            name
            for name, expected in reference.items()
            if name not in current
            or not torch.equal(current[name].detach().cpu(), expected)
        ]
        if changed:
            raise RuntimeError(
                "Frozen BatchNorm running statistics changed during training; "
                f"first changed buffers: {changed[:10]}"
            )

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode:
            self._enforce_batchnorm_finetune()
        return result

    def on_train_start(self):
        # This hook runs after Lightning restores an exact-resume checkpoint,
        # so the reference always reflects the weights actually being trained.
        self._enforce_batchnorm_finetune()
        self._capture_batchnorm_reference()

    def _a1_a3_scale_for_epoch(self, epoch: int) -> float:
        if self.a1_a3_ramp_epochs <= 0:
            return 1.0
        if self.a1_a3_ramp_epochs == 1:
            return 1.0
        progress = min(1.0, max(0.0, float(epoch) / float(self.a1_a3_ramp_epochs - 1)))
        return self.a1_a3_start_scale + (1.0 - self.a1_a3_start_scale) * progress

    def on_load_checkpoint(self, checkpoint):
        """Validate compatibility and support older A1/A3 checkpoints."""
        key = "brep_encoder.graph_attn_bias.a1_a3_scale"
        state_dict = checkpoint.get("state_dict", {})
        if (
            self.reuse_checkpoint_class_weights
            and "class_weights" not in state_dict
        ):
            raise ValueError(
                "reuse_checkpoint_class_weights was requested, but the "
                "checkpoint has no class_weights buffer"
            )
        if key not in state_dict:
            state_dict[key] = self.brep_encoder.graph_attn_bias.a1_a3_scale.detach().clone()

    # Gradually unfreeze encoder after warmup_freeze_epochs
    def on_train_epoch_start(self):
        self._enforce_batchnorm_finetune()
        a1_a3_scale = self._a1_a3_scale_for_epoch(int(self.current_epoch))
        self.brep_encoder.graph_attn_bias.set_a1_a3_scale(a1_a3_scale)
        self.log(
            "a1_a3_scale",
            a1_a3_scale,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        if self.a1_a3_ramp_epochs > 0:
            print(
                f"A1/A3 scale at epoch {self.current_epoch}: {a1_a3_scale:.3f}",
                flush=True,
            )
        if self.warmup_freeze_epochs > 0 and self.current_epoch == self.warmup_freeze_epochs:
            print(f"Unfreezing brep_encoder at epoch {self.current_epoch}")
            for p in self.brep_encoder.parameters():
                p.requires_grad = True

        # If subgraph training is active, advance the epoch so the *same* CAD file
        # yields different random k-hop neighborhoods on subsequent epochs.
        ds = getattr(self, "_train_dataset_for_subgraph", None)
        if ds is not None and hasattr(ds, "subgraph_epoch"):
            ds.subgraph_epoch = int(self.current_epoch)

    def training_step(self, batch, batch_idx):

        # brep encoder----------------------------------------------------------------------------------
        with torch.profiler.record_function("brep_encoder"):
            node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)

        # node classifier--------------------------------------------------------------------------------
        with torch.profiler.record_function("pool_attn_classifier"):
            node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node dim=0
            node_emb = node_emb[:, 1:, :]            # node_emb [batch_size, max_node_num, dim] without global node
            padding_mask = batch["padding_mask"]     # [batch_size, max_node_num]
            node_pos = torch.where(~padding_mask)  # [(batch_size, node_index)]
            node_z = node_emb[node_pos]  # [total_nodes, dim_z]
            # node_pos[0] is already the graph id for every flattened face.
            graph_z = graph_emb[node_pos[0]]
            z = self.attention([node_z, graph_z])
            node_logits = self.classifier.forward_logits(z)  # [total_nodes, num_classes]

        # loss-------------------------------------------------------------------------------------------
        with torch.profiler.record_function("loss"):
            labels = batch["label_feature"].long()
            cw = self.class_weights if self.use_class_weights else None
            loss = ClassificationLossFromLogits(
                labels,
                node_logits,
                loss_type=self.loss_type,
                focal_gamma=self.focal_gamma,
                class_level_weight=cw,
            )
        bs = int(labels.shape[0])
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        return loss

    def on_train_epoch_end(self):
        self._assert_batchnorm_reference_unchanged()
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        current_lr = opt.param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True)
        if len(opt.param_groups) > 1:
            self.log(
                "a1_a3_lr",
                opt.param_groups[1]["lr"],
                on_step=False,
                on_epoch=True,
            )


    def validation_step(self, batch, batch_idx):

        with torch.profiler.record_function("brep_encoder"):
            node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # logits [total_nodes, num_classes]

        with torch.profiler.record_function("pool_attn_classifier"):
            node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node dim=0
            node_emb = node_emb[:, 1:, :]            # node_emb [batch_size, max_node_num, dim] without global node
            padding_mask = batch["padding_mask"]     # [batch_size, max_node_num]
            node_pos = torch.where(~padding_mask)  # [(batch_size, node_index)]
            node_z = node_emb[node_pos]  # [total_nodes, dim]
            graph_z = graph_emb[node_pos[0]]
            z = self.attention([node_z, graph_z])
            node_logits = self.classifier.forward_logits(z)  # [total_nodes, num_classes]

        with torch.profiler.record_function("loss"):
            labels = batch["label_feature"].long()  # labels [total_nodes]
            cw = self.class_weights if self.use_class_weights else None
            loss = ClassificationLossFromLogits(
                labels,
                node_logits,
                loss_type=self.loss_type,
                focal_gamma=self.focal_gamma,
                class_level_weight=cw,
            )
            self.log(
                "eval_loss",
                loss,
                on_step=False,
                on_epoch=True,
                batch_size=int(labels.shape[0]),
            )

        preds = torch.argmax(node_logits, dim=-1)  # preds [total_nodes]
        valid = (labels >= 0) & (labels < self.num_classes)
        encoded = labels[valid] * self.num_classes + preds[valid]
        self._val_confusion.add_(
            torch.bincount(
                encoded,
                minlength=self.num_classes * self.num_classes,
            ).reshape(self.num_classes, self.num_classes)
        )

        if (
            batch_idx == 0
            and self.trainer.current_epoch % 5 == 0
            and self.trainer.is_global_zero
        ):
            from models.tensorboard_media import tb_add_histogram

            mx = node_logits.softmax(dim=-1).max(dim=-1).values.detach().cpu()
            tb_add_histogram(
                self.trainer,
                "val/max_pred_prob_batch0",
                mx,
                int(self.trainer.global_step),
            )

        return loss

    def on_validation_epoch_start(self):
        self._val_confusion.zero_()

    def on_validation_epoch_end(self):
        cm = self._val_confusion.detach().cpu().numpy().astype(np.float64)
        total = float(cm.sum())
        if total <= 0:
            return

        rows = cm.sum(axis=1)
        cols = cm.sum(axis=0)
        diag = np.diag(cm)
        self.log("per_face_accuracy", float(diag.sum() / total))

        per_class_acc = np.divide(
            diag,
            rows,
            out=np.zeros_like(diag, dtype=np.float64),
            where=rows > 0,
        )
        for i, acc in enumerate(per_class_acc):
            self.log(f"val_class_{i}_acc", float(acc))
        self.log("per_class_accuracy", float(per_class_acc.mean()))

        feature_total = float(rows[1:].sum())
        if feature_total > 0:
            self.log(
                "per_face_accuracy_feature",
                float(diag[1:].sum() / feature_total),
            )

        per_class_iou = []
        for i in range(self.num_classes):
            union = rows[i] + cols[i] - diag[i]
            # Preserve the legacy definition: only classes present in both the
            # labels and predictions contribute to mean IoU.
            if rows[i] > 0 and cols[i] > 0 and union > 0:
                per_class_iou.append(float(diag[i] / union))
        if per_class_iou:
            self.log("IoU", float(np.mean(per_class_iou)))

        if self.trainer.is_global_zero:
            from models.tensorboard_media import log_segmentation_val_confusion

            log_segmentation_val_confusion(
                self.trainer,
                cm,
                int(self.current_epoch),
                prefix="val",
            )

    def test_step(self, batch, batch_idx):

        # brep encoder----------------------------------------------------------------------------------
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # logits [total_nodes, num_classes]

        # node classifier-------------------------------------------------------------------------------
        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node dim=0
        node_emb = node_emb[:, 1:, :]  # node_emb [batch_size, max_node_num, dim] without global node
        padding_mask = batch["padding_mask"]  # [batch_size, max_node_num]
        node_pos = torch.where(~padding_mask)  # [(batch_size, node_index)]
        node_z = node_emb[node_pos]  # [total_nodes, dim]
        graph_z = graph_emb[node_pos[0]]
        z = self.attention([node_z, graph_z])
        node_logits = self.classifier.forward_logits(z)  # [total_nodes, num_classes]

        preds = torch.argmax(node_logits, dim=-1)  # preds [total_nodes]
        labels = batch["label_feature"].long()  # labels [total_nodes]
        known_pos = torch.where(labels < self.num_classes)
        labels_ = labels[known_pos]
        preds_ = preds[known_pos]
        labels_np = labels_.long().detach().cpu().numpy()
        preds_np = preds_.long().detach().cpu().numpy()

        for i in range(len(preds_np)): self.pred.append(preds_np[i])
        for i in range(len(labels_np)): self.label.append(labels_np[i])

        # 将结果转为txt文件----------------------------------------------------------------------------
        # n_graph, max_n_node = batch["padding_mask"].size()[:2]
        # node_pos = torch.where(batch["padding_mask"] == False)
        # face_feature = -1 * torch.ones([n_graph, max_n_node], device=self.device, dtype=torch.long)
        # face_feature[node_pos] = preds[:]
        # out_face_feature = face_feature.long().detach().cpu().numpy()  # [n_graph, max_n_node]
        # for i in range(n_graph):
        #     # 计算每个graph的实际n_node
        #     end_index = max_n_node - np.sum((out_face_feature[i][:] == -1).astype(np.int64))
        #     # masked出实际face feature
        #     pred_feature = out_face_feature[i][:end_index + 1]  # (n_node)

        #     output_path = pathlib.Path("/home/zhang/datasets_segmentation/2_val")
        #     file_name = "feature_" + str(batch["id"][i].long().detach().cpu().numpy()) + ".txt"
        #     file_path = os.path.join(output_path, file_name)
        #     feature_file = open(file_path, mode="a")
        #     for j in range(end_index):
        #         feature_file.write(str(pred_feature[j]))
        #         feature_file.write("\n")
        #     feature_file.close()

    def on_test_epoch_end(self):
        print("num_classes: %s" % self.num_classes)
        preds_np = np.array(self.pred)
        labels_np = np.array(self.label)
        self.pred = []
        self.label = []

        # Friendly names for common thread(+text/+chamfer/+fillet) setups; else class_i
        _default_names = {
            0: "Stock",
            1: "Thread",
            2: "Text",
            3: "Chamfer",
            4: "Fillet",
        }

        def _class_name(i: int) -> str:
            return _default_names.get(i, f"class_{i}")

        per_face_comp = (preds_np == labels_np).astype(np.int64)
        self.log("per_face_accuracy", np.mean(per_face_comp))
        print("per_face_accuracy: %s" % np.mean(per_face_comp))

        # Per-class accuracy / precision / recall (one entry per training class)
        per_class_acc = []
        per_class_precision = []
        per_class_recall = []
        print("\nPer-class precision / recall / accuracy:")
        print(f"  {'class':<10} {'precision':>10} {'recall':>10} {'accuracy':>10} {'support':>10} {'pred_n':>10}")
        for i in range(0, self.num_classes):
            name = _class_name(i)
            label_pos = np.where(labels_np == i)[0]
            pred_pos = np.where(preds_np == i)[0]
            support = int(len(label_pos))
            pred_n = int(len(pred_pos))
            tp = int(np.sum(preds_np[label_pos] == i)) if support > 0 else 0

            # Recall = TP / (TP+FN) = TP / support  (same as per-class accuracy)
            recall = float(tp / support) if support > 0 else 0.0
            # Precision = TP / (TP+FP) = TP / pred_n
            precision = float(tp / pred_n) if pred_n > 0 else 0.0
            acc = recall  # face-level class accuracy ≡ recall

            per_class_acc.append(acc)
            per_class_precision.append(precision)
            per_class_recall.append(recall)

            self.log(f"test_class_{i}_acc", acc)
            self.log(f"test_class_{i}_precision", precision)
            self.log(f"test_class_{i}_recall", recall)
            # Named aliases for Stock / Thread / Text / Chamfer / Fillet when applicable
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

        # IoU---------------------------------------------------------------------------------------
        per_class_iou = []
        for i in range (0, self.num_classes):
            label_pos = np.where(labels_np == i)
            pred_pos = np.where(preds_np == i)
            if len(pred_pos[0]) > 0 and len(label_pos[0]) > 0:
                class_i_preds = preds_np[label_pos]
                class_i_label = labels_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int64)
                Union = (class_i_preds != class_i_label).astype(np.int64)
                class_i_preds_ = preds_np[pred_pos]
                class_i_label_ = labels_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int64)
                per_class_iou.append(np.sum(Intersection) / (np.sum(Union) + np.sum(Intersection) + np.sum(Union_)))
        self.log("IoU", np.mean(per_class_iou))
        print("IoU: %s" % np.mean(per_class_iou))

        # confusion_matrix---------------------------------------------------------------------------
        # output_path = pathlib.Path("/home/zhang/datasets_segmentation/confusion_matrix.txt")
        # result_file = open(output_path, mode="a")
        # for i in range(0, self.num_classes):
        #     class_pos = np.where(labels_np == i)
        #     if len(class_pos[0]) > 0:
        #         class_i_preds = preds_np[class_pos]
        #         for j in range(0, self.num_classes):
        #             per_face_comp = (class_i_preds == j).astype(np.int64)
        #             acc_class_i = np.mean(per_face_comp)
        #             result_file.write(str(acc_class_i))
        #             if(j < self.num_classes-1):
        #                 result_file.write(" ")
        #         result_file.write("\n")
        # result_file.close()

    # original configure_optimizers() function

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=0.002, betas=(0.99, 0.999))

    #     # Learning Strategies
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
    #                                                            threshold=0.0001, threshold_mode='rel',
    #                                                            min_lr=0.000001, cooldown=2, verbose=False)

    #     return {"optimizer": optimizer,
    #             "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "eval_loss"}
    #             }

    # code matching paper's values    # def configure_optimizers(self):

    # def configure_optimizers(self):
    #         optimizer = torch.optim.AdamW(
    #             self.parameters(),
    #             lr=0.001,
    #             betas=(0.9, 0.999),
    #             eps=1e-8,
    #             weight_decay=0.01,
    #         )

    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer,
    #             mode="min",
    #             factor=0.5,
    #             patience=5,
    #             threshold=1e-4,
    #             threshold_mode="rel",
    #             min_lr=1e-6,
    #             cooldown=2,
    #             verbose=False,
    #         )

    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "eval_loss"},
    #         }

    def configure_optimizers(self):
        a1_a3_params = []
        base_params = []
        for name, parameter in self.named_parameters():
            if name.startswith("brep_encoder.graph_attn_bias."):
                a1_a3_params.append(parameter)
            else:
                base_params.append(parameter)

        parameter_groups = [
            {
                "params": base_params,
                "lr": self.learning_rate,
                "target_lr": self.learning_rate,
                "name": "pretrained_backbone",
            },
            {
                "params": a1_a3_params,
                "lr": self.a1_a3_learning_rate,
                "target_lr": self.a1_a3_learning_rate,
                "name": "a1_a3_bias",
            },
        ]
        optimizer_kwargs = dict(
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        if self.fused_adamw and torch.cuda.is_available():
            optimizer_kwargs["fused"] = True
        try:
            optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
        except (TypeError, RuntimeError) as exc:
            if "fused" not in optimizer_kwargs:
                raise
            print(f"Fused AdamW unavailable ({exc}); using foreach/default AdamW.", flush=True)
            optimizer_kwargs.pop("fused")
            optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
        print(
            "Optimizer learning rates: "
            f"backbone={self.learning_rate:g}, "
            f"A1/A3={self.a1_a3_learning_rate:g}, "
            f"warmup_steps={self.optimizer_warmup_steps}, fused={optimizer_kwargs.get('fused', False)}",
            flush=True,
        )

        # Monitor per_class_accuracy (macro-averaged, computed in
        # on_validation_epoch_end) instead of eval_loss.  On imbalanced data
        # eval_loss is dominated by the majority class, so a scheduler that
        # watches it will shrink the LR whenever Focal Loss / class weights
        # shift probability mass toward minority classes — even when macro
        # accuracy is improving.  per_class_accuracy treats every class
        # equally, so the scheduler now reacts to real segmentation progress.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            threshold=1e-4,
            threshold_mode='rel',
            min_lr=1e-6,
            cooldown=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": self.check_val_every_n_epoch,
                "monitor": "per_class_accuracy",
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None, **kwargs):
        # Set warmup rates before optimizer.step so the first update is not
        # accidentally taken at the full target LR. Each parameter group keeps
        # its own target, which is important for lite -> A1/A3 fine-tuning.
        if (
            self.optimizer_warmup_steps > 0
            and self.trainer.global_step < self.optimizer_warmup_steps
        ):
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1) / float(self.optimizer_warmup_steps),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * float(pg.get("target_lr", self.learning_rate))
        optimizer.step(closure=optimizer_closure)