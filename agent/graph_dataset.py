#!/usr/bin/env python
# coding: utf-8

import algos
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(-10000.0)
        new_x[:xlen, :xlen] = x
        # new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def onehop_collate_fn(batch, zipped=False):
    if zipped:
        x, node_attr, target_attr, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, edge_input, label = batch
    else:
        x, node_attr, target_attr, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, edge_input, label = zip(
            *batch)
    max_node_num = max(i.size(0) for i in x)
    # max_dist = max(i.size(-2) for i in edge_input)

    x = pad_sequence(x, batch_first=True, padding_value=0)
    node_attr = pad_sequence(node_attr, batch_first=True, padding_value=0)
    target_attr = pad_sequence(target_attr, batch_first=True, padding_value=0)

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num) for i in attn_bias]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_type]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_pos]
    )

    in_degree = pad_sequence(in_degree, batch_first=True, padding_value=0)
    out_degree = pad_sequence(out_degree, batch_first=True, padding_value=0)

    if label is not None:
        label = torch.cat(label)

    return {
        'x': x,
        'node_attr': node_attr,
        'target_attr': target_attr,
        'attn_bias': attn_bias,
        'attn_edge_type': attn_edge_type,
        'spatial_pos': spatial_pos,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'label': label,
    }


def __preprocess_item(item, node_type_vocab, node_attr_vocab, edge_attr_vocab, spatial_pos_max):
    node, node_type, node_attr, edge_index, edge_attr, target_node = item['node'], item['node_type'], item['node_attr'], \
    item['edge_index'], item['edge_attr'], item['target_node']
    N = len(node)

    # node feature 
    x = torch.LongTensor([1] + [node_type_vocab[_] for _ in node_type])
    node_attr = torch.LongTensor([0] + [node_attr_vocab[_] for _ in node_attr])
    target_index = [0] + [node.index(_) for _ in target_node if _ in node]
    target_attr = torch.zeros(N + 1).long()
    target_attr[target_index] = 1

    edge_attr = torch.LongTensor([edge_attr_vocab[_] for _ in edge_attr])
    edge_index = torch.LongTensor(edge_index)
    # node adj matrix [N, N] bool
    adj = torch.zeros([N + 1, N + 1], dtype=torch.bool)
    adj[0, :] = True
    adj[:, 0] = True
    adj[edge_index[0, :] + 1, edge_index[1, :] + 1] = True
    for i in range(N + 1):
        adj[i, i] = True

        # edge feature here
    attn_edge_type = torch.zeros([N + 1, N + 1], dtype=torch.long)
    attn_edge_type[0, :] = 1
    attn_edge_type[:, 0] = 1
    attn_edge_type[edge_index[0, :] + 1, edge_index[1, :] + 1] = edge_attr
    for i in range(N + 1):
        attn_edge_type[i, i] = 2

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist.astype(np.int32), path.astype(np.int32),
                                      attn_edge_type.numpy().astype(np.int32))
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
    attn_bias[spatial_pos > spatial_pos_max] = -10000.0

    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = in_degree
    # combine
    return {
        'x': x,
        'node_attr': node_attr,
        'target_attr': target_attr,
        'attn_bias': attn_bias,
        'attn_edge_type': attn_edge_type,
        'spatial_pos': spatial_pos,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'edge_input': torch.from_numpy(edge_input).long()
    }
