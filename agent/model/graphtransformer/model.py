# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from .modules import init_graphormer_params, GraphormerGraphEncoder

logger = logging.getLogger(__name__)


class GraphormerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_nodes = args.max_nodes
        self.graph_encoder = GraphormerGraphEncoder(
            # < for graphormer
            num_node_type=args.num_node_type,
            num_node_attr=args.num_node_attr,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.num_encoder_layers,
            embedding_dim=args.embedding_dim,
            ffn_embedding_dim=args.ffn_embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
        )

        self.masked_lm_pooler = nn.Linear(
            args.embedding_dim, args.embedding_dim
        )

        self.lm_head_transform_weight = nn.Linear(
            args.embedding_dim, args.embedding_dim
        )
        if args.activation_fn == 'gelu':
            self.activation_fn = F.gelu
        elif args.activation_fn == 'relu':
            self.activation_fn = F.relu
        self.layer_norm = LayerNorm(args.embedding_dim)

        self.tanh = nn.Tanh()
        self.lm_out1 = nn.Linear(
            args.embedding_dim, args.embedding_dim
        )
        self.lm_out2 = nn.Linear(
            args.embedding_dim, args.num_classes
        )

        self.apply(init_graphormer_params)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )

        nodes_rep = inner_states[-1].transpose(0, 1)
        nodes_rep = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(nodes_rep)))
        scores = self.lm_out2(nodes_rep)  # [B, N, NUM_CLASS]
        attention_mask = 1 - batched_data['x'].eq(0).long()
        if attention_mask is not None:
            # 1 means effective, 0 means not effective
            adder = (1.0 - attention_mask) * -10000.0
            # add to keep softmax ZERO
            scores += adder.unsqueeze(2)
        weights = torch.softmax(scores, dim=1)
        # weights: [batch_size, seq_length, num_of_class]
        weighted_x = weights * scores
        # weights_x: [batch_size, seq_length, num_of_class]
        logits = torch.sum(weighted_x, dim=1)
        # graph_rep = self.lm_out2(self.tanh(self.lm_out1(graph_rep))) # [B, NUM_CLASS]

        return logits

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict
