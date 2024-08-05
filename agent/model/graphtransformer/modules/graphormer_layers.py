# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, num_heads, num_node_type, num_node_attr, num_in_degree, num_out_degree, hidden_dim, n_layers
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_node_type = num_node_type

        # 1 for graph token
        self.node_type_encoder = nn.Embedding(num_node_type, hidden_dim, padding_idx=0)
        self.node_attr_encoder = nn.Embedding(num_node_attr, hidden_dim, padding_idx=0)
        self.target_encoder = nn.Embedding(2, hidden_dim, padding_idx = 0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x, node_attr, target_attr, in_degree, out_degree = (
            batched_data["x"],
            batched_data["node_attr"],
            batched_data["target_attr"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()

        # node feauture + graph token
        node_feature = self.node_type_encoder(x)  # [B, N, H]

        # if self.flag and perturb is not None:
        #     node_feature += perturb

        node_feature = (
            node_feature
            + self.node_attr_encoder(node_attr)
            + self.target_encoder(target_attr)
            # + self.in_degree_encoder(in_degree)
            # + self.out_degree_encoder(out_degree)
        ) # [B, N, H]

        return node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads,
        num_node_type,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x, attn_edge_type = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
            batched_data["attn_edge_type"],
        )
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        if self.edge_type == "multi_hop":
            edge_input = batched_data["edge_input"]

        n_graph, n_node = x.size()
        graph_attn_bias = attn_bias.clone() # [B, N, N]
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [B, H, N, N]
        # print(attn_bias)
        # spatial pos
        # [B, N, N, H] -> [B, H, N, N]
        # spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        # graph_attn_bias = graph_attn_bias + spatial_pos_bias

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [B, N, N, H] -> [B, H, N, N]
            edge_input = self.edge_encoder(attn_edge_type).permute(0, 3, 1, 2)
        
        graph_attn_bias = graph_attn_bias + edge_input
        
        return graph_attn_bias
