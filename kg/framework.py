import copy
import math
import types
import inspect
from decorator import decorator
from collections import defaultdict

import torch
from torch import nn, autograd
from torch.nn import functional as F

from torchdrug import core, data, layers, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug import utils
from torch_scatter import scatter_add
import os
import pickle
import numpy as np

from extension import sparse
from . import model as model_
from util import sparse_tensor_index_select, sparse_tensor_flatten


def cached(model, debug=False):

    @decorator
    def wrapper(forward, self, *args, **kwargs):

        def equal(x, y):
            if isinstance(x, nn.Parameter):
                x = x.data
            if isinstance(y, nn.Parameter):
                y = y.data
            if type(x) != type(y):
                return False
            if isinstance(x, torch.Tensor):
                return x.shape == y.shape and (x == y).all()
            elif isinstance(x, data.Graph):
                if x.num_node != y.num_node or x.num_edge != y.num_edge or x.num_relation != y.num_relation:
                    return False
                edge_feature = getattr(x, "edge_feature", torch.tensor(0, device=x.device))
                y_edge_feature = getattr(y, "edge_feature", torch.tensor(0, device=y.device))
                if edge_feature.shape != y_edge_feature.shape:
                    return False
                return (x.edge_list == y.edge_list).all() and (x.edge_weight == y.edge_weight).all() \
                       and (edge_feature == y_edge_feature).all()
            else:
                return x == y

        if self.training:
            return forward(self, *args, **kwargs)

        func = inspect.signature(forward)
        func = func.bind(self, *args, **kwargs)
        func.apply_defaults()
        arguments = func.arguments.copy()
        arguments.pop(next(iter(arguments.keys())))

        if hasattr(self, "_forward_cache"):
            hit = True
            message = []
            for k, v in arguments.items():
                if not equal(self._forward_cache[k], v):
                    hit = False
                    message.append("%s: miss" % k)
                    break
                message.append("%s: hit" % k)
            if debug:
                print("[cache] %s" % ", ".join(message))
        else:
            hit = False
            if debug:
                print("[cache] cold start")
        if hit:
            return self._forward_cache["result"]
        else:
            self._forward_cache = {}

        for k, v in arguments.items():
            if isinstance(v, torch.Tensor) or isinstance(v, data.Graph):
                v = v.detach()
            self._forward_cache[k] = v
        result = forward(self, *args, **kwargs)
        self._forward_cache["result"] = result
        return result

    model = copy.copy(model)
    model.forward = types.MethodType(wrapper(model.forward.__func__), model)
    return model


# TODO: faster implemenation of remove
@torch.no_grad()
def remove(graph, edges, ratio=1):
    if ratio == 0:
        return graph
    in_edge = torch.zeros(graph.num_edge, dtype=torch.bool, device=graph.device)
    for edges_ in edges.split(2048):
        if edges.shape[-1] == 3:
            in_edge_ = (graph.edge_list.unsqueeze(0) == edges_.unsqueeze(1)).all(dim=-1).any(dim=0)
        elif edges.shape[-1] == 2:
            in_edge_ = (graph.edge_list[:, :2].unsqueeze(0) == edges_.unsqueeze(1)).all(dim=-1).any(dim=0)
        elif edges.shape[-1] == 1:
            in_edge_ = (graph.edge_list[:, :2].unsqueeze(0) == edges_.unsqueeze(1)).any(dim=-1).any(dim=0)
        else:
            raise ValueError
        in_edge = in_edge | in_edge_
    if ratio < 1:
        in_edge = in_edge & (torch.rand(len(in_edge), device=graph.device) < ratio)
    graph = graph.edge_mask(~in_edge)
    return graph


@torch.no_grad()
def remove_fast(graph, edges, ratio=1):
    # use the fact that `edges` are not many
    if ratio == 0:
        return graph

    in_edge = torch.zeros(graph.num_edge, dtype=torch.bool, device=graph.device)
    if edges.shape[-1] == 3:
        h_index, t_index, r_index = edges.t()
        h_index_set, h_inverse = torch.unique(h_index, return_inverse=True)
        t_index_set, t_inverse = torch.unique(t_index, return_inverse=True)
        r_index_set, r_inverse = torch.unique(r_index, return_inverse=True)
        h_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=graph.device)
        t_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=graph.device)
        r_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=graph.device)
        h_index2id[h_index_set] = torch.arange(len(h_index_set), device=graph.device)
        t_index2id[t_index_set] = torch.arange(len(t_index_set), device=graph.device)
        r_index2id[r_index_set] = torch.arange(len(r_index_set), device=graph.device)
        query_htr_index = h_index2id[h_index] * len(t_index_set) * len(r_index_set) + \
                          t_index2id[t_index] * len(r_index_set) + r_index2id[r_index]

        h_index, t_index, r_index = graph.edge_list.t()
        valid = (h_index2id[h_index] >= 0) & (t_index2id[t_index] >= 0) & (r_index2id[r_index] >= 0)
        h_index = h_index[valid]
        t_index = t_index[valid]
        r_index = r_index[valid]
        htr_index = h_index2id[h_index] * len(t_index_set) * len(r_index_set) + \
                    t_index2id[t_index] * len(r_index_set) + r_index2id[r_index]
        in_edge[valid] = (htr_index.unsqueeze(0) == query_htr_index.unsqueeze(-1)).any(dim=0)

    elif edges.shape[-1] == 2:
        h_index, t_index = edges.t()
        h_index_set, h_inverse = torch.unique(h_index, return_inverse=True)
        t_index_set, t_inverse = torch.unique(t_index, return_inverse=True)
        h_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=graph.device)
        t_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=graph.device)
        h_index2id[h_index_set] = torch.arange(len(h_index_set), device=graph.device)
        t_index2id[t_index_set] = torch.arange(len(t_index_set), device=graph.device)
        query_ht_index = h_index2id[h_index] * len(t_index_set) + t_index

        h_index, t_index = graph.edge_list.t()[:2]
        valid = (h_index2id[h_index] >= 0) & (t_index2id[t_index] >= 0)
        h_index = h_index[valid]
        t_index = t_index[valid]
        ht_index = h_index2id[h_index] * len(t_index_set) + t_index
        in_edge[valid] = (ht_index.unsqueeze(0) == query_ht_index.unsqueeze(-1)).any(dim=0)

    elif edges.shape[-1] == 1:
        node_index, = edges.t()
        node_index_set, node_inverse = torch.unique(node_index, return_inverse=True)
        node_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=graph.device)
        node_index2id[node_index_set] = torch.arange(len(node_index_set), device=graph.device)

        h_index, t_index = graph.edge_list.t()[:2]
        in_edge = (node_index2id[h_index] >= 0) | (node_index2id[t_index] >= 0)

    else:
        raise ValueError

    if ratio < 1:
        in_edge = in_edge & (torch.rand(len(in_edge), device=graph.device) < ratio)
    graph = graph.edge_mask(~in_edge)
    return graph


@R.register("framework.BellmanFordKDDCup")
class BellmanFordKDDCup(nn.Module, core.Configurable):

    def __init__(self, gnn_model, score_model, flip_edge=False, remove_one_hop=0, remove_two_hop=0, reverse=False):
        super(BellmanFordKDDCup, self).__init__()
        self.gnn_model = cached(gnn_model)
        self.score_model = score_model
        self.flip_edge = flip_edge
        self.remove_one_hop = remove_one_hop
        self.remove_two_hop = remove_two_hop
        self.reverse = reverse

    def remove_target_edges(self, graph, h_index, t_index, r_index):
        edges = torch.stack((h_index, t_index, r_index), dim=-1).flatten(0, -2)
        graph = remove_fast(graph, edges)

        edges1 = torch.stack((h_index, t_index), dim=-1).flatten(0, -2)
        edges2 = torch.stack((t_index, h_index), dim=-1).flatten(0, -2)
        edges = torch.cat([edges1, edges2])
        graph = remove_fast(graph, edges, ratio=self.remove_one_hop)

        edges = torch.cat((h_index, t_index)).flatten().unsqueeze(-1)
        graph = remove_fast(graph, edges, ratio=self.remove_two_hop)
        return graph

    def get_undirected(self, graph):
        edge_list = graph.edge_list[:, [1, 0, 2]]
        if graph.num_relation > 1: # knowledge graph
            edge_list[:, 2] += graph.num_relation
            num_relation = graph.num_relation * 2
        else:
            num_relation = graph.num_relation
        edge_list = torch.cat((graph.edge_list, edge_list))
        edge_weight = graph.edge_weight.repeat(2)
        data_dict, meta_dict = graph.data_by_meta(include="node")
        return type(graph)(edge_list, edge_weight, num_node=graph.num_node, num_relation=num_relation,
                           meta_dict=meta_dict, **data_dict)

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        relation_weight = self.score_model.relation.weight

        if all_loss is not None:
            graph = self.remove_target_edges(graph, h_index, t_index, r_index)
        if self.flip_edge:
            graph = self.get_undirected(graph)

        hr_index = h_index * graph.num_relation + r_index
        tr_index = t_index * graph.num_relation + r_index
        hr_index_set, hr_inverse = torch.unique(hr_index, return_inverse=True)
        tr_index_set, tr_inverse = torch.unique(tr_index, return_inverse=True)
        hr_index_set_h = hr_index_set // graph.num_relation
        hr_index_set_r = hr_index_set % graph.num_relation
        tr_index_set_t = tr_index_set // graph.num_relation
        tr_index_set_r = tr_index_set % graph.num_relation

        h_index_set, h_inverse = torch.unique(h_index, return_inverse=True)
        t_index_set, t_inverse = torch.unique(t_index, return_inverse=True)

        # compatible with inductive
        h_attention = functional.one_hot(h_index_set, graph.num_node)
        t_attention = functional.one_hot(t_index_set, graph.num_node)

        if self.reverse: # t -> h
            t_attention = functional.one_hot(tr_index_set_t, graph.num_node)
            t_value = torch.zeros(graph.num_node, t_attention.shape[0], relation_weight.shape[-1], device=self.device)
            index = tr_index_set_t.unsqueeze(0).unsqueeze(-1).expand(-1, -1, relation_weight.shape[-1])
            t_value = t_value.scatter_add(0, index, relation_weight[tr_index_set_r].unsqueeze(0))

            oneway = self.gnn_model(graph, t_value, all_loss, metric)["node_feature"]
            feature = torch.einsum("vtd, hv -> htd", oneway, h_attention)

            feature = feature[h_inverse, tr_inverse]
        else: # h -> t
            h_attention = functional.one_hot(hr_index_set_h, graph.num_node)
            h_value = torch.zeros(graph.num_node, h_attention.shape[0], relation_weight.shape[-1], device=self.device)
            index = hr_index_set_h.unsqueeze(0).unsqueeze(-1).expand(-1, -1, relation_weight.shape[-1])
            h_value = h_value.scatter_add(0, index, relation_weight[hr_index_set_r].unsqueeze(0))

            oneway = self.gnn_model(graph, h_value)["node_feature"]
            feature = torch.einsum("vhd, tv -> htd", oneway, t_attention)

            feature = feature[hr_inverse, t_inverse]

        # additional features
        feature = torch.cat([feature, relation_weight[r_index]], dim=-1)
        score = self.score_model.forward_feature(feature)
        return score


@R.register("framework.BatchedBellmanFordKDDCup")
class BatchedBellmanFordKDDCup(nn.Module, core.Configurable):

    def __init__(self, gnn_model, score_model, flip_edge=False, remove_one_hop=0, remove_two_hop=0, reverse=False,
                 entity_embedding_dim=None, relation_embedding_dim=None, head_feature_dim=None, tail_feature_dim=None,
                 use_graph_feature=None, use_query_relation=False,
                 use_degree_feature=False, learnable_tail=False,
                 learnable_zero=False, tied_relation_weight=True, activation="relu",
                 random_relation_corruption=0, random_entity_corruption=0):
        super(BatchedBellmanFordKDDCup, self).__init__()
        self.gnn_model = cached(gnn_model)
        self.score_model = score_model
        self.flip_edge = flip_edge
        self.remove_one_hop = remove_one_hop
        self.remove_two_hop = remove_two_hop
        self.reverse = reverse
        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.head_feature_dim = head_feature_dim
        self.tail_feature_dim = tail_feature_dim
        self.use_graph_feature = use_graph_feature
        self.use_degree_feature = use_degree_feature
        self.use_query_relation = use_query_relation
        self.learnable_tail = learnable_tail
        self.learnable_zero = learnable_zero
        self.tied_relation_weight = tied_relation_weight
        self.random_relation_corruption = random_relation_corruption
        self.random_entity_corruption = random_entity_corruption

        if entity_embedding_dim:
            self.h_embedding_linear = nn.Linear(entity_embedding_dim, gnn_model.output_dim)
            self.t_embedding_linear = nn.Linear(entity_embedding_dim, gnn_model.output_dim)
        if relation_embedding_dim:
            self.r_embedding_linear = nn.Linear(relation_embedding_dim, gnn_model.input_dim)
        if head_feature_dim:
            self.h_feature_linear = nn.Linear(head_feature_dim, gnn_model.input_dim)
        if tail_feature_dim:
            self.t_feature_linear = nn.Linear(tail_feature_dim, gnn_model.output_dim)
        if learnable_tail:
            self.query_tail = nn.Embedding(self.score_model.relation.num_embeddings, self.score_model.relation.embedding_dim)
        if learnable_zero:
            self.query_zero = nn.Embedding(self.score_model.relation.num_embeddings, self.score_model.relation.embedding_dim)
        if not tied_relation_weight:
            self.query_new = nn.Embedding(self.score_model.relation.num_embeddings, self.score_model.relation.embedding_dim)
        if use_degree_feature:
            self.degree_linear = nn.Linear(4, gnn_model.output_dim)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def remove_target_edges(self, graph, h_index, t_index, r_index):
        edges = torch.stack((h_index, t_index, r_index), dim=-1).flatten(0, -2)
        graph = remove_fast(graph, edges)

        edges1 = torch.stack((h_index, t_index), dim=-1).flatten(0, -2)
        edges2 = torch.stack((t_index, h_index), dim=-1).flatten(0, -2)
        edges = torch.cat([edges1, edges2])
        graph = remove_fast(graph, edges, ratio=self.remove_one_hop)

        edges = torch.cat((h_index, t_index)).flatten().unsqueeze(-1)
        graph = remove_fast(graph, edges, ratio=self.remove_two_hop)
        return graph

    def corrupt(self, graph, relation_ratio, entity_ratio):
        if relation_ratio == 0 and entity_ratio == 0:
            return graph
        
        assert len(graph) == 1, "Not Implemented for batch_size > 1"

        edge_list = graph.edge_list.clone()

        if relation_ratio > 0:
            index = (torch.rand(graph.num_edge, device=self.device) < relation_ratio).nonzero().squeeze(-1)
            num_sample = len(index)
            sample_index = torch.randperm(graph.num_edge, device=self.device)[:num_sample]
            edge_list[index, 2] = graph.edge_list[sample_index, 2]

        if entity_ratio > 0:
            rand = torch.rand(graph.num_edge, device=self.device)
            head_index = (rand < entity_ratio / 2).nonzero().squeeze(-1)
            tail_index = ((rand >= entity_ratio / 2) & (rand < entity_ratio)).nonzero().squeeze(-1)
            num_sample = len(head_index) + len(tail_index)
            sample_index = torch.randperm(graph.num_edge, device=self.device)[:num_sample]
            head_sample_index = sample_index[:len(head_index)]
            tail_sample_index = sample_index[len(head_index):]
            edge_list[head_index, 0] = graph.edge_list[head_sample_index, 0]
            edge_list[tail_index, 1] = graph.edge_list[tail_sample_index, 1]

        data_dict, meta_dict = graph.data_mask(include=("node", "graph"))

        return type(graph)(edge_list, graph.edge_weight,
                           num_nodes=graph.num_nodes, num_edges=graph.num_edges, num_relation=graph.num_relation,
                           offsets=graph._offsets, meta_dict=meta_dict, **data_dict)

    def get_undirected(self, graph):
        assert isinstance(graph, data.PackedGraph)

        edge_list = graph.edge_list[:, [1, 0, 2]]
        edge_list[:, 2] += graph.num_relation
        num_relation = graph.num_relation * 2
        edge_list = torch.stack([graph.edge_list, edge_list], dim=1).flatten(0, 1)
        offsets = graph._offsets.repeat(2, 1).t().flatten()
        edge_weight = graph.edge_weight.repeat(2)

        data_dict, meta_dict = graph.data_mask(include=("node", "graph"))

        return type(graph)(edge_list, edge_weight,
                           num_nodes=graph.num_nodes, num_edges=graph.num_edges * 2, num_relation=num_relation,
                           offsets=offsets, meta_dict=meta_dict, **data_dict)

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        relation_weight = self.score_model.relation.weight

        if all_loss is not None:
            graph = self.remove_target_edges(graph, h_index, t_index, r_index)
            graph = self.corrupt(graph, self.random_relation_corruption, self.random_entity_corruption)
        if self.flip_edge:
            graph = self.get_undirected(graph)

        hr_index = h_index * graph.num_relation + r_index
        tr_index = t_index * graph.num_relation + r_index
        hr_index_set, hr_inverse = torch.unique(hr_index, return_inverse=True)
        tr_index_set, tr_inverse = torch.unique(tr_index, return_inverse=True)
        hr_index_set_h = hr_index_set // graph.num_relation
        hr_index_set_r = hr_index_set % graph.num_relation
        tr_index_set_t = tr_index_set // graph.num_relation
        tr_index_set_r = tr_index_set % graph.num_relation

        if self.reverse: # t -> h
            if self.learnable_zero:
                t_value = self.query_zero.weight[tr_index_set_r]
                t_value = t_value[graph.node2graph]
            else:
                t_value = torch.zeros(graph.num_node, relation_weight.shape[-1], device=self.device)
            if self.learnable_tail:
                t_value[h_index] = self.query_tail.weight[r_index]
            t_value[tr_index_set_t] = relation_weight[tr_index_set_r]
            if self.head_feature_dim:
                t_feature = self.h_feature_linear(graph.t_feature)
                t_feature = self.activation(t_feature)
                t_value[tr_index_set_t] += t_feature

            if self.use_query_relation:
                with graph.graph():
                    graph.query_relation = relation_weight[tr_index_set_r]
            output = self.gnn_model(graph, t_value, all_loss, metric)
            oneway = output["node_feature"]
            feature = oneway[h_index]
        else: # h -> t
            if self.learnable_zero:
                h_value = self.query_zero.weight[hr_index_set_r]
                h_value = h_value[graph.node2graph]
            else:
                h_value = torch.zeros(graph.num_node, relation_weight.shape[-1], device=self.device)
            if self.learnable_tail:
                h_value[t_index] = self.query_tail.weight[r_index]
            h_value[hr_index_set_h] = relation_weight[hr_index_set_r]
            if self.head_feature_dim:
                h_feature = self.h_feature_linear(graph.h_feature)
                h_feature = self.activation(h_feature)
                h_value[hr_index_set_h] += h_feature

            if self.use_query_relation:
                with graph.graph():
                    graph.query = relation_weight[hr_index_set_r]
            output = self.gnn_model(graph, h_value, all_loss, metric)
            oneway = output["node_feature"]
            feature = oneway[t_index]

        # additional features
        if self.tied_relation_weight:
            relation_weight = self.score_model.relation.weight
        else:
            relation_weight = self.query_new.weight

        if self.head_feature_dim:
            if self.reverse:
                h_feature = graph.t_feature
            else:
                h_feature = graph.h_feature
            h_feature = self.h_feature_linear(h_feature)
            h_feature = self.activation(h_feature)
            features = [feature, relation_weight[r_index] + h_feature]
        else:
            features = [feature, relation_weight[r_index]]
        if self.entity_embedding_dim:
            h_embedding = self.h_embedding_linear(graph.h_embedding)
            h_embedding = self.activation(h_embedding)
            features.append(h_embedding)
            t_embedding = self.t_embedding_linear(graph.t_embedding)
            t_embedding = self.activation(t_embedding)
            features.append(t_embedding)
        if self.relation_embedding_dim:
            r_embedding = self.r_embedding_linear(graph.r_embedding)
            r_embedding = self.activation(r_embedding)
            features.append(r_embedding)
        if self.tail_feature_dim:
            if self.reverse:
                t_feature = graph.h_feature
            else:
                t_feature = graph.t_feature
            t_feature = self.t_feature_linear(t_feature)
            t_feature = self.activation(t_feature)
            features.append(t_feature)
        if self.use_graph_feature:
            features.append(output["graph_feature"].unsqueeze(1).expand_as(feature))
        if self.use_degree_feature:
            degree = torch.stack([graph.full_degree_in[h_index], graph.full_degree_out[h_index],
                                  graph.full_degree_in[t_index], graph.full_degree_out[t_index]], dim=-1)
            degree = (degree + 1).log()
            degree_feature = self.degree_linear(degree)
            features.append(degree_feature)
        feature = torch.cat(features, dim=-1)
        score = self.score_model.forward_feature(feature)
        return score


@R.register("framework.FeatureMLP")
class FeatureMLP(nn.Module, core.Configurable):

    def __init__(self, num_entity, num_relation, input_dim, hidden_dims, short_cut=False, batch_norm=False,
                 activation="relu", dropout=0, learnable_relation=False,
                 separate_head_tail=False, use_degree_feature=False,
                 score_func="TransE", max_score=10):
        super(FeatureMLP, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.learnable_relation = learnable_relation
        self.separate_head_tail = separate_head_tail
        self.use_degree_feature = use_degree_feature
        self.score_func = score_func
        self.max_score = max_score
        assert score_func in ["TransE", "DistMult", "SimplE", "RotatE", "Linear", "MLP"]

        num_feature = 2 if use_degree_feature else 1
        self.h_encoder = layers.MLP(input_dim * num_feature, hidden_dims, short_cut, batch_norm, activation, dropout)
        self.r_encoder = layers.MLP(input_dim, hidden_dims, activation, dropout)
        if learnable_relation:
            self.relation = nn.Embedding(num_relation, hidden_dims[-1])
        if separate_head_tail:
            self.t_encoder = layers.MLP(input_dim * num_feature, hidden_dims, short_cut, batch_norm, activation, dropout)
        else:
            self.t_encoder = self.h_encoder
        if use_degree_feature:
            self.degree_encoder = layers.MLP(4, input_dim)
        if score_func == "Linear":
            self.score_model = nn.Linear(hidden_dims[-1] * 3, 1)
        elif score_func == "MLP":
            self.score_model = layers.MLP(hidden_dims[-1] * 3, [hidden_dims[-1], 1], short_cut, batch_norm, activation,
                                          dropout)

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        h = graph.h_embedding
        t = graph.t_embedding
        r = graph.r_embedding
        if self.use_degree_feature:
            degree = torch.stack([graph.full_degree_in, graph.full_degree_out], dim=-1)
            degree = (degree + 1).log()
            degree = torch.cat([degree, 1 / degree.clamp(min=1e-2)], dim=-1)
            d = self.degree_encoder(degree)
            h = torch.cat([h, d[h_index]], dim=-1)
            t = torch.cat([t, d[t_index]], dim=-1)
        h = self.h_encoder(h)
        t = self.t_encoder(t)
        r = self.r_encoder(r)
        if self.learnable_relation:
            r = r + self.relation.weight[r_index]

        if self.score_func == "TransE":
            x = (h + r - t).abs().sum(dim=-1)
            score = self.max_score - x
        elif self.score_func == "DistMult":
            score = (h * r * t).sum(dim=-1)
        elif self.score_func == "SimplE":
            t_flipped = torch.cat(t.chunk(2, dim=-1)[::-1], dim=-1)
            score = (h * r * t_flipped).sum(dim=-1)
        elif self.score_func == "RotatE":
            h_re, h_im = h.chunk(2, dim=-1)
            r = r.chunk(2, dim=-1)[0]
            r_re, r_im = torch.cos(r), torch.sin(r)
            t_re, t_im = t.chunk(2, dim=-1)

            x_re = h_re * r_re - h_im * r_im - t_re
            x_im = h_re * r_im + h_im * r_re - t_im
            x = torch.stack([x_re, x_im], dim=-1)
            x = x.norm(p=2, dim=-1).sum(dim=-1)
            score = self.max_score - x
        elif self.score_func in ["Linear", "MLP"]:
            x = torch.cat([h, r, t], dim=-1)
            score = self.score_model(x).squeeze(-1)
        else:
            raise ValueError

        return score
