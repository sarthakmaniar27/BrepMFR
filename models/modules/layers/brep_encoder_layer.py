from typing import Callable, Optional
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.fairseq_shim import (
    FairseqDropout,
    LayerNorm,
    quant_noise,
    get_activation_fn,
)
from .multihead_attention import MultiheadAttention
from .feature_encoders import SurfaceEncoder, CurveEncoder

class GraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        pre_layernorm: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.pre_layernorm = pre_layernorm

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class NonLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(output_dim)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = F.relu(self.bn2(self.linear2(x)))
        return x


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, num_heads, num_degree, hidden_dim, n_layers):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # node_feature encode
        self.surf_encoder = SurfaceEncoder(
            in_channels=7, output_dims=int(0.5 * hidden_dim)
        )
        self.face_area_encoder = NonLinear(1, int(0.125 * hidden_dim))
        self.face_type_encoder = nn.Embedding(8, int(0.125 * hidden_dim), padding_idx=0)
        self.face_loop_encoder = nn.Embedding(256, int(0.125 * hidden_dim), padding_idx=0)
        self.degree_encoder = nn.Embedding(num_degree, int(0.125 * hidden_dim), padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, x, face_area, face_type, face_loop, face_degree, padding_mask):
        # x [total_node_num, U_grid, V_grid, pnt_feature]
        # padding_mask [batch_size, max_node_num] 记录每个graph的实际长度，空位记为True
        n_graph, n_node = padding_mask.size()[:2]
        node_pos = torch.where(~padding_mask)

        x = x.permute(0, 3, 1, 2)
        x_ = self.surf_encoder(x)  # [total_nodes, n_hidden]
        face_area_ = self.face_area_encoder(face_area.unsqueeze(dim=1))  # [total_nodes, n_hidden]

        # Embeddings have fixed vocab sizes; macro / CAD data can exceed training ranges (CUDA assert).
        ft_idx = face_type.long().clamp(0, self.face_type_encoder.num_embeddings - 1)
        fl_idx = face_loop.long().clamp(0, self.face_loop_encoder.num_embeddings - 1)
        fd_idx = face_degree.long().clamp(0, self.degree_encoder.num_embeddings - 1)

        face_type_ = self.face_type_encoder(ft_idx)
        face_loop_ = self.face_loop_encoder(fl_idx)
        face_degree_ = self.degree_encoder(fd_idx)

        node_feature = torch.cat((x_, face_area_, face_type_, face_loop_, face_degree_), dim=-1)

        face_feature = torch.zeros([n_graph, n_node, self.hidden_dim], device=x.device, dtype=x.dtype)
        face_feature[node_pos] = node_feature[:]  # 空节点用0.0填充

        # 增加一个全局虚拟节点 [n_graph, 1, n_hidden]
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, face_feature], dim=1)  # [n_graph, max_node_num+1, n_hidden]
        return graph_node_feature, node_feature



class _MLP(nn.Module):
    """"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class _EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 2 from the paper where the edge features are
        updated using the node features at the endpoints.

        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, edge_index, nfeat, efeat):
        src, dst = edge_index[0], edge_index[1]
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        h = F.leaky_relu(self.batchnorm(h), inplace= True)
        return h


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """
    def __init__(
            self,
            dim_node,
            num_heads,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            n_layers,
            max_nodes_for_a3: Optional[int] = None,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        # A3 gathers O(n^2 * max_dist) edge indices; None = no cap (training / large GPUs).
        self.max_nodes_for_a3 = max_nodes_for_a3
        # Multiplier for the A1 shortest-path and A3 multi-hop edge biases.
        # It is a buffer so the exact scale used by a validation checkpoint is
        # restored for inference.  Lite-trained checkpoints predate this buffer;
        # BrepSeg.on_load_checkpoint supplies a backward-compatible default.
        self.register_buffer("a1_a3_scale", torch.tensor(1.0, dtype=torch.float32))

        # spatial_feature encode
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        # D2 A3 encoder
        # self.d2_pos_encoder = nn.Linear(64, num_heads, bias=False)
        # self.ang_pos_encoder = nn.Linear(64, num_heads, bias=False)
        self.d2_pos_encoder = NonLinear(64, num_heads)
        self.ang_pos_encoder = NonLinear(64, num_heads)

        # edge_feature encode
        self.curv_encoder = CurveEncoder(in_channels=7, output_dims=num_heads)
        self.edge_type_encoder = nn.Embedding(6, num_heads, padding_idx=0)
        self.edge_len_encoder = NonLinear(1, num_heads)
        self.edge_ang_encoder = NonLinear(1, num_heads)
        self.edge_conv_encoder = nn.Embedding(3, num_heads, padding_idx=0)

        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
            self.node_cat = _EdgeConv(
                edge_feats=num_heads,
                out_feats=num_heads,
                node_feats=dim_node,
            )

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def set_a1_a3_scale(self, scale: float) -> None:
        """Set the shared A1/A3 contribution multiplier without changing model shape."""
        value = float(scale)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"A1/A3 scale must be finite and in [0, 1], got {scale!r}")
        self.a1_a3_scale.fill_(value)

    def forward(self, attn_bias, spatial_pos, d2_distance, ang_distance, edge_data, edge_type, edge_len, edge_ang, edge_conv, edge_path, edge_padding_mask, edge_index, node_feat):
        n_graph = attn_bias.size(0)
        n_node = attn_bias.size(1) - 1

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # [n_graph, n_head, n_node+1, n_node+1] 描述每一头注意力下各节点之间的关系矩阵

        use_a1 = spatial_pos is not None
        use_a2 = d2_distance is not None and ang_distance is not None
        use_a3 = self.edge_type == "multi_hop" and edge_path is not None and use_a1
        run_a3 = use_a3
        if (
            run_a3
            and self.max_nodes_for_a3 is not None
            and n_node > self.max_nodes_for_a3
        ):
            if not getattr(self, "_warned_skip_a3_nodes", False):
                warnings.warn(
                    f"GraphAttnBias: skipping multi-hop edge bias (A3) because n_node={n_node} "
                    f"exceeds max_nodes_for_a3={self.max_nodes_for_a3} "
                    f"(dense gather would be ~O(n^2×{self.multi_hop_max_dist}) elements). "
                    f"Spatial bias (A1) still applies. Use --max_nodes_for_a3 0 to disable this cap, "
                    f"or regenerate graphs with inference_profile=lite (no edge_path).",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_skip_a3_nodes = True
            run_a3 = False

        # ---- DEBUG: spatial_pos sanity ----
        # if not hasattr(self, "_dbg_done"):
        #     self._dbg_done = 0

        # if self._dbg_done < 5:  # print only first few batches
        #     with torch.no_grad():
        #         sp = spatial_pos
        #         # Ensure on CPU for printing if needed
        #         sp_min = int(sp.min().item()) if sp.numel() > 0 else None
        #         sp_max = int(sp.max().item()) if sp.numel() > 0 else None
        #         num_spatial = int(self.spatial_pos_encoder.num_embeddings)
        #         pad_idx = self.spatial_pos_encoder.padding_idx

        #         print("\n[DEBUG][GraphAttnBias] spatial_pos stats:")
        #         print(f"  spatial_pos shape: {tuple(sp.shape)} dtype={sp.dtype} device={sp.device}")
        #         print(f"  spatial_pos min/max: {sp_min} / {sp_max}")
        #         print(f"  spatial_pos_encoder.num_embeddings: {num_spatial} (valid idx: 0..{num_spatial-1})")
        #         print(f"  spatial_pos_encoder.padding_idx: {pad_idx}")

        #         bad_hi = (sp >= num_spatial)
        #         bad_lo = (sp < 0)
        #         if bad_hi.any() or bad_lo.any():
        #             n_bad = int((bad_hi | bad_lo).sum().item())
        #             print(f"  !!! FOUND OOB spatial_pos indices: {n_bad}")
        #             # show a few offending positions/values
        #             bad_idx = torch.nonzero(bad_hi | bad_lo)[:10]
        #             print(f"  first bad indices (up to 10): {bad_idx.tolist()}")
        #             print(f"  first bad values: {sp[bad_idx[:,0], bad_idx[:,1], bad_idx[:,2]].tolist()}")
        #             # hard fail early with clear message (better than CUDA assert)
        #             raise RuntimeError(
        #                 f"OOB spatial_pos indices detected. max={sp_max}, "
        #                 f"but spatial_pos_encoder.num_embeddings={num_spatial}"
        #             )

        #     self._dbg_done += 1
        # ---- DEBUG END ----

        if use_a1:
            # spatial_pos must be in [0, num_embeddings-1]
            num_spatial = self.spatial_pos_encoder.num_embeddings
            spatial_pos = spatial_pos.clone()
            spatial_pos[spatial_pos < 0] = 0
            spatial_pos = spatial_pos.clamp(0, num_spatial - 1)

            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos)
            spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2)
            # a1_a3_scale is a float32 buffer; cast under AMP so half activations stay half.
            a1_a3_scale = self.a1_a3_scale.to(dtype=spatial_pos_bias.dtype)
            spatial_pos_bias = spatial_pos_bias * a1_a3_scale
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

            t = (
                self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
                * a1_a3_scale
            )
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        if use_a2:
            d2_distance = d2_distance.reshape(-1, 64)
            d2_pos_bias = self.d2_pos_encoder(d2_distance)
            d2_pos_bias = d2_pos_bias.reshape(n_graph, n_node, n_node, self.num_heads)
            d2_pos_bias = d2_pos_bias.permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + d2_pos_bias.to(
                dtype=graph_attn_bias.dtype
            )

            ang_distance = ang_distance.reshape(-1, 64)
            ang_pos_bias = self.ang_pos_encoder(ang_distance)
            ang_pos_bias = ang_pos_bias.reshape(n_graph, n_node, n_node, self.num_heads)
            ang_pos_bias = ang_pos_bias.permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + ang_pos_bias.to(
                dtype=graph_attn_bias.dtype
            )

        # edge_feature 边编码------------------------------------------------------------------------------------------------------------
        if run_a3:
            spatial_pos_ = spatial_pos.clone()  # Record the distance between any two nodes [batch_size, max_node_num, max_node_num]. The distance from a node to itself is recorded as 1.
            spatial_pos_[spatial_pos_ == 0] = 1  # Set the padding to 1. Empty spaces (which can be considered virtual nodes) are uniformly set to 1, and the distance from a node to itself is also recorded as 1.
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)  # After adjustment, the distance between any two directly connected nodes is also 1.
            spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)

            # Reduce edge_input
            max_dist = self.multi_hop_max_dist
            edge_pos = torch.where(~edge_padding_mask)  # edge_padding_mask [batch_size, max_edges_num]

            # Adjust the dimensions and perform curv_encode.
            edge_data = edge_data.permute(0, 2, 1)
            edge_data_ = self.curv_encoder(edge_data)  # [total_edges, n_head]


            # ---- DEBUG: edge_type / edge_conv sanity ----
            # if not hasattr(self, "_dbg_edge_meta"):
            #     self._dbg_edge_meta = 0

            # if self._dbg_edge_meta < 3:
            #     with torch.no_grad():
            #         et = edge_type
            #         ec = edge_conv

            #         et_min = int(et.min().item()) if et.numel() else None
            #         et_max = int(et.max().item()) if et.numel() else None
            #         ec_min = int(ec.min().item()) if ec.numel() else None
            #         ec_max = int(ec.max().item()) if ec.numel() else None

            #         et_vocab = int(self.edge_type_encoder.num_embeddings)  # should be 6
            #         ec_vocab = int(self.edge_conv_encoder.num_embeddings)  # should be 3

            #         print("\n[DEBUG][GraphAttnBias] edge_type/edge_conv stats:", flush=True)
            #         print(f"  edge_type shape={tuple(et.shape)} dtype={et.dtype} min/max={et_min}/{et_max} vocab={et_vocab} (valid 0..{et_vocab-1})", flush=True)
            #         print(f"  edge_conv shape={tuple(ec.shape)} dtype={ec.dtype} min/max={ec_min}/{ec_max} vocab={ec_vocab} (valid 0..{ec_vocab-1})", flush=True)

            #         bad_et = (et < 0) | (et >= et_vocab)
            #         bad_ec = (ec < 0) | (ec >= ec_vocab)

            #         if bad_et.any():
            #             n = int(bad_et.sum().item())
            #             vals = et[bad_et][:10].tolist()
            #             print(f"  !!! OOB edge_type count={n}, first_vals={vals}", flush=True)
            #             raise RuntimeError(f"OOB edge_type detected: min/max={et_min}/{et_max}, vocab={et_vocab}")

            #         if bad_ec.any():
            #             n = int(bad_ec.sum().item())
            #             vals = ec[bad_ec][:10].tolist()
            #             print(f"  !!! OOB edge_conv count={n}, first_vals={vals}", flush=True)
            #             raise RuntimeError(f"OOB edge_conv detected: min/max={ec_min}/{ec_max}, vocab={ec_vocab}")

            #     self._dbg_edge_meta += 1
            # ---- DEBUG END ----

            
            # debug: print edge_type and edge_conv stats and check for out-of-range values before encoding

            # edge_type = edge_type.to(torch.long)
            # nt = self.edge_type_encoder.num_embeddings
            # et_min, et_max = int(edge_type.min()), int(edge_type.max())
            # print("[DBG] edge_type min/max:", et_min, et_max, "num_embeddings:", nt)
            # assert 0 <= et_min and et_max < nt



            edge_type_ = self.edge_type_encoder(
                edge_type.long().clamp(0, self.edge_type_encoder.num_embeddings - 1)
            )
            edge_len_ = self.edge_len_encoder(edge_len.unsqueeze(dim=1))
            edge_ang_ = self.edge_ang_encoder(edge_ang.unsqueeze(dim=1))

            # edge_conv is already in the form of [0, 1, 2] in the converter, where 0 means no edge, 1 means convex edge, and 2 means concave edge. 
            # We directly encode it with an embedding layer. The padding index is set to 0, which will be ignored in the attention bias. 
            
            # edge_conv = edge_conv.to(torch.long)
            # ec_vocab = int(self.edge_conv_encoder.num_embeddings)
            # ec_min, ec_max = int(edge_conv.min()), int(edge_conv.max())
            # print("[DBG] edge_conv min/max:", ec_min, ec_max, "num_embeddings:", ec_vocab)
            # assert 0 <= ec_min and ec_max < ec_vocab


            edge_conv_ = self.edge_conv_encoder(
                edge_conv.long().clamp(0, self.edge_conv_encoder.num_embeddings - 1)
            )
            edge_feat = edge_data_ + edge_type_ + edge_len_ + edge_ang_ + edge_conv_

            # add node_feature to edge_feature
            edge_feat_ = self.node_cat(edge_index, node_feat, edge_feat)  # [total_edges, n_head]

            # Edge input expansion [total_edges, n_head]->[n_graph, max_node_num, max_node_num, max_dist, n_head]
            # new_zeros + explicit cast: under AMP, BatchNorm/MLP paths can disagree on float16 vs float32.
            n_edge = edge_padding_mask.size(1)
            edge_feature = edge_feat_.new_zeros(
                (n_graph, n_edge + 1, edge_feat_.size(-1))
            )
            edge_feature[edge_pos] = edge_feat_.to(dtype=edge_feature.dtype)

            edge_path = edge_path.reshape(n_graph, n_node * n_node, max_dist)


            
            # edge_feature has shape [n_graph, n_edge+1, ...]
            # the last index (n_edge) is reserved as the all-zero padding row
            pad_idx = edge_padding_mask.size(1)  # == n_edge in the batch
            edge_path = edge_path.to(torch.long)
            edge_path = torch.where(edge_path < 0, torch.full_like(edge_path, pad_idx), edge_path)
            edge_path = edge_path.clamp(0, pad_idx)


            dim_0 = torch.arange(n_graph, device=edge_path.device).reshape(n_graph, 1, 1)

            # ---- DEBUG: edge_path sanity ----
            # if self._dbg_done < 5:
            #     with torch.no_grad():
            #         ep = edge_path
            #         ep_min = int(ep.min().item()) if ep.numel() > 0 else None
            #         ep_max = int(ep.max().item()) if ep.numel() > 0 else None
            #         ef_dim1 = int(edge_feature.size(1))  # valid: 0..ef_dim1-1

            #         print("\n[DEBUG][GraphAttnBias] edge_path stats:")
            #         print(f"  edge_path shape: {tuple(ep.shape)} dtype={ep.dtype} device={ep.device}")
            #         print(f"  edge_path min/max: {ep_min} / {ep_max}")
            #         print(f"  edge_feature.size(1): {ef_dim1} (valid idx: 0..{ef_dim1-1})")

            #         # Note: -1 is used as padding in your converter; this indexing path does NOT support -1 safely.
            #         bad_hi = (ep >= ef_dim1)
            #         bad_lo = (ep < 0)
            #         if bad_hi.any() or bad_lo.any():
            #             n_bad = int((bad_hi | bad_lo).sum().item())
            #             print(f"  !!! FOUND OOB edge_path indices: {n_bad}")
            #             bad_idx = torch.nonzero(bad_hi | bad_lo)[:10]
            #             print(f"  first bad indices: {bad_idx.tolist()}")
            #             print(f"  first bad values: {ep.flatten()[bad_idx[:,0]].tolist() if ep.dim()==2 else 'see tensor'}")
            #             raise RuntimeError(
            #                 f"OOB edge_path indices detected. max={ep_max}, min={ep_min}, "
            #                 f"edge_feature.size(1)={ef_dim1}"
            #             )

            # ---- DEBUG END ----


            # debug: print edge_path stats and check for out-of-range values before indexing into edge_feature
            # ef = edge_feature.size(1)
            # ep_min, ep_max = int(edge_path.min()), int(edge_path.max())
            # print("[DBG] edge_path min/max:", ep_min, ep_max, "edge_feature.size(1):", ef)
            # assert 0 <= ep_min and ep_max < ef



            # Algebraically fuse the per-hop head transforms and hop reduction:
            #   sum_d(edge[d] @ W[d]) == cat(edge[d]) @ cat(W[d]).
            # The old bmm materialized another [D, B*N*N, H] tensor (several GB
            # at the adaptive batch budget). This produces only [B*N*N, H].
            edge_bias = edge_feature[dim_0, edge_path]
            edge_dis_weight = self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads
            )[:max_dist].reshape(max_dist * self.num_heads, self.num_heads)
            edge_bias = torch.matmul(
                edge_bias.flatten(start_dim=-2), edge_dis_weight
            ).reshape(n_graph, n_node, n_node, self.num_heads)
            edge_bias = edge_bias / spatial_pos_.to(edge_bias.dtype).unsqueeze(-1)
            edge_bias = edge_bias.permute(0, 3, 1, 2)
            edge_bias = edge_bias * self.a1_a3_scale.to(dtype=edge_bias.dtype)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_bias.to(
                dtype=graph_attn_bias.dtype
            )
        # edge_feature 边编码------------------------------------------------------------------------------------------------------------

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
        return graph_attn_bias