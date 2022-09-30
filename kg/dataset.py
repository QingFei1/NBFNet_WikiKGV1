import os
import re
import csv
import logging
from collections import defaultdict, deque

from tqdm import tqdm

import torch
from torch._six import container_abcs
from torch.utils import data as torch_data

import numpy as np

from torchdrug import core, data, utils
from torchdrug.utils import comm
from torchdrug.core import Registry as R

from extension import sparse

from ogb import lsc, linkproppred

logger = logging.getLogger(__name__)


path = os.path.dirname(__file__)

dataset_ext = utils.load_extension(
        "dataset_ext", [os.path.join(path, "../extension/dataset.cpp")], extra_cflags=["-g", "-Ofast"])


@R.register("datasets.WikiKG90M")
class WikiKG90M(data.KnowledgeGraphDataset):

    def __init__(self, path, num_hop=0, num_neighbor=0):
        path = os.path.expanduser(path)
        self.dataset = lsc.WikiKG90MDataset(root=path)

        triplets = self.dataset.train_hrt[:, [0, 2, 1]]
        self.load_triplet(triplets)
        self.valid_dict = self.dataset.valid_dict["h,r->t"]
        self.test_dict = self.dataset.test_dict["h,r->t"]

        # save memory
        del self.dataset

        self.num_hop = num_hop
        self.num_neighbor = num_neighbor

        if self.num_hop:
            self.build_index()

    def build_index(self):
        graph = self.graph

        node_in, node_out = graph.edge_list.t()[:2]
        self.order_in = node_in.argsort()
        self.order_out = node_out.argsort()
        self.degree_in = torch.bincount(node_in, minlength=graph.num_node)
        self.degree_out = torch.bincount(node_out, minlength=graph.num_node)
        self.cum_degree_in = self.degree_in.cumsum(0)
        self.cum_degree_out = self.degree_out.cumsum(0)

    def split(self):
        train_set = self
        valid_set = WikiKG90MTest(self.valid_dict)
        test_set = WikiKG90MTest(self.test_dict)

        return train_set, valid_set, test_set

    def get_n_hop_graph(self, nodes):
        queue = deque(nodes)
        distance = dict.fromkeys(nodes, 0)
        edge_index = []

        while queue:
            u = queue.popleft()
            if distance[u] == self.num_hop:
                continue

            if self.degree_in[u] > 0:
                start = torch.randint(self.degree_in[u], (1,)).item()
                for i in range(start, start + min(self.num_neighbor, self.degree_in[u])):
                    index = self.cum_degree_in[u].item() - self.degree_in[u].item() + i % self.degree_in[u].item()
                    index = self.order_in[index].item()
                    edge_index.append(index)
                    _u, v, r = self.graph.edge_list[index].tolist()
                    assert u == _u
                    if v not in distance:
                        distance[v] = distance[u] + 1
                        queue.append(v)

            if self.degree_out[u] > 0:
                start = torch.randint(self.degree_out[u], (1,)).item()
                for i in range(start, start + min(self.num_neighbor, self.degree_out[u])):
                    index = self.cum_degree_out[u].item() - self.degree_out[u].item() + i % self.degree_out[u].item()
                    index = self.order_out[index].item()
                    edge_index.append(index)
                    v, _u, r = self.graph.edge_list[index].tolist()
                    assert u == _u
                    if v not in distance:
                        distance[v] = distance[u] + 1
                        queue.append(v)

        edge_index = list(set(edge_index))
        mapping = {u: i for i, u in enumerate(distance.keys())}
        edge_list = self.graph.edge_list[edge_index]
        for i in range(len(edge_list)):
            for j in range(2):
                edge_list[i, j] = mapping[edge_list[i, j].item()]
        for i in range(len(nodes)):
            nodes[i] = mapping[nodes[i]]

        graph = data.Graph(edge_list, num_node=len(mapping), num_relation=self.graph.num_relation)
        return graph, nodes

    def __getitem__(self, index):
        if self.num_hop == 0:
            return super(WikiKG90M, self).__getitem__(index)

        triplet = self.graph.edge_list[index]
        h_index, t_index, r_index = triplet.tolist()
        nodes = [h_index, t_index]
        graph, nodes = self.get_n_hop_graph(nodes)
        h_index, t_index = nodes
        return {
            "triplet": [h_index, t_index, r_index],
            "graph": graph,
        }


@R.register("datasets.WikiKG90MTest")
class WikiKG90MTest(torch_data.Dataset):

    def __init__(self, test_dict):
        self.hr = test_dict["hr"]
        self.t_candidate = test_dict["t_candidate"]
        self.num_candidate = self.t_candidate.shape[-1]
        if "t_correct_index" in test_dict:
            self.t_correct_index = test_dict["t_correct_index"]
        else:
            self.t_correct_index = None

    def __getitem__(self, index):
        hr = self.hr[index].expand(self.num_candidate, -1)
        t = self.t_candidate[index].unsqueeze(-1)
        triplet = torch.cat([hr, t], dim=-1)[:, [0, 2, 1]]
        item = {"triplet": triplet}
        if self.t_correct_index is not None:
            item["target"] = torch.tensor(self.t_correct_index[index])
        return item

    def __len__(self):
        return len(self.hr)


@R.register("datasets.WikiKG90MNumpy")
class WikiKG90MNumpy(torch_data.Dataset, core.Configurable):

    def __init__(self, path, feature_file=None, train_relation=None,
                 num_hop=2, num_neighbor=100, num_head_neighbor=None, num_neighbor_expand=None,
                 degree_block_threshold=None,
                 uniform_negative_ratio=0, add_super_node=False, negative_hops=None,
                 init_distance_t=0, edge_one_more_hop=False, prune_isolated=0,
                 num_negative=0, strict_negative=False, use_entity_embedding=False, embedding_in_memory=False,
                 use_relation_embedding=False, shuffle=True, use_cpp_implementation=False, use_degree_info=False,
                 use_edge_weight=False, log_edge_weight=False):
        path = os.path.expanduser(path)
        self.dataset = lsc.WikiKG90MDataset(root=path)

        # safe to use int32
        edge_list = self.dataset.train_hrt[:, [0, 2, 1]].astype(np.int32)
        # random shuffle to reduce the bias in the original order
        if shuffle: # takes 2~3 minutes
            edge_list = np.random.permutation(edge_list)
        self.edge_list = edge_list
        self.valid_dict = self.dataset.valid_dict["h,r->t"]
        self.test_dict = self.dataset.test_dict["h,r->t"]
        if feature_file:
            feature_file = os.path.expanduser(feature_file)
            self.node_feature = np.load(feature_file)
        else:
            self.node_feature = None

        self.num_entity = self.edge_list[:, :2].max() + 1
        self.num_edge = len(self.edge_list)
        self.num_relation = self.edge_list[:, 2].max() + 1
        if add_super_node:
            self.num_relation += 1

        if use_entity_embedding:
            if embedding_in_memory:
                logger.warning("[%d] load entity features into RAM" % comm.get_rank())
                self.entity_embedding = self.dataset.all_entity_feat
                logger.warning("[%d] load entity features done" % comm.get_rank())
            else:
                self.entity_embedding = self.dataset.entity_feat
        if use_relation_embedding:
            if embedding_in_memory:
                self.relation_embedding = self.dataset.all_relation_feat
            else:
                self.relation_embedding = self.dataset.relation_feat

        # save memory
        del self.dataset

        self.train_relation = train_relation
        if train_relation is not None:
            self.train_indexes = np.isin(self.edge_list[:, 2], train_relation).nonzero()[0]
            if comm.get_rank() == 0:
                logger.warning("%d train relations are specified" % len(train_relation))
                logger.warning("training set is reduced to %d samples" % len(self.train_indexes))

        self.num_hop = num_hop
        if not isinstance(num_neighbor, container_abcs.Sequence):
            num_neighbor = [num_neighbor] * num_hop
        if num_head_neighbor is None:
            num_head_neighbor = num_neighbor
        if not isinstance(num_head_neighbor, container_abcs.Sequence):
            num_head_neighbor = [num_head_neighbor] * num_hop
        if num_neighbor_expand is not None and not isinstance(num_neighbor_expand, container_abcs.Sequence):
            num_neighbor_expand = [num_neighbor_expand] * (num_hop + 1)
        if edge_one_more_hop:
            if len(num_neighbor) <= num_hop:
                assert len(num_neighbor) == num_hop
                num_neighbor.append(num_neighbor[-1])
        self.num_neighbor = num_neighbor
        self.num_head_neighbor = num_head_neighbor
        self.num_neighbor_expand = num_neighbor_expand
        self.degree_block_threshold = degree_block_threshold
        self.negative_hops = negative_hops
        self.uniform_negative_ratio = uniform_negative_ratio
        self.add_super_node = add_super_node
        self.init_distance_t = init_distance_t
        self.edge_one_more_hop = edge_one_more_hop
        self.prune_isolated = prune_isolated
        self.num_negative = num_negative
        self.strict_negative = strict_negative
        self.use_entity_embedding = use_entity_embedding
        self.use_relation_embedding = use_relation_embedding
        self.use_cpp_implementation = use_cpp_implementation
        self.use_degree_info = use_degree_info
        self.use_edge_weight = use_edge_weight
        self.log_edge_weight = log_edge_weight
        if prune_isolated:
            assert num_neighbor_expand is None and not edge_one_more_hop
        if use_cpp_implementation:
            assert not edge_one_more_hop and init_distance_t == 0
            assert num_neighbor_expand is None
        assert init_distance_t >= 0

        self.build_index()

    def split(self):
        train_set = self
        valid_set = WikiKG90MTestNumpy(self, self.valid_dict)
        test_set = WikiKG90MTestNumpy(self, self.test_dict)

        return train_set, valid_set, test_set

    def shuffle(self):
        perm = np.random.permutation(self.num_edge)
        inv_perm = perm.argsort().astype(np.int32)
        self.edge_list = self.edge_list[perm]
        self.order_in = inv_perm[self.order_in]
        self.order_out = inv_perm[self.order_out]

    def build_index(self):
        node_in, node_out = self.edge_list.transpose()[:2]
        # safe to use int32
        self.order_in = node_in.argsort().astype(np.int32)
        self.order_out = node_out.argsort().astype(np.int32)
        self.degree_in = np.bincount(node_in, minlength=self.num_entity)
        self.degree_out = np.bincount(node_out, minlength=self.num_entity)
        self.offset_in = self.degree_in.cumsum(0) - self.degree_in
        self.offset_out = self.degree_out.cumsum(0) - self.degree_out

        if self.strict_negative:
            self.hr2t = defaultdict(list)
            h, t, r = self.edge_list.transpose()
            hr = h.astype(np.int64) * self.num_relation + r
            order = hr.argsort().astype(np.int32)
            h = h[order]
            r = r[order]
            hr = hr[order]
            t = t[order]
            last = 0
            logger.warning("[%d] begin strict negative table" % comm.get_rank())
            for i in tqdm(range(self.num_edge)):
                if hr[i] != hr[last]:
                    self.hr2t[(h[last], r[last])] = set(t[last: i])
                last = i
            self.hr2t[(h[last], r[last])] = set(t[last:])
            logger.warning("[%d] end strict negative table" % comm.get_rank())

    def get_n_hop_graph_cpp(self, nodes):
        edge_list, nodes = dataset_ext.get_n_hop_graph(nodes, self.edge_list, self.order_in, self.order_out,
                                                       self.degree_in, self.degree_out, self.offset_in, self.offset_out,
                                                       self.num_hop, self.num_neighbor)
        graph = data.Graph(edge_list, num_node=edge_list[:, :2].max() + 1, num_relation=self.num_relation)
        return graph, nodes

    def get_n_hop_graph_py(self, nodes, return_distance=False):
        if self.num_hop == 0:
            mapping = {u: i for i, u in enumerate(nodes)}
            inv_mapping = torch.tensor(nodes, dtype=torch.long).flatten()
            for i in range(len(nodes)):
                nodes[i] = mapping[nodes[i]]

            graph = data.Graph([], num_node=len(mapping), num_relation=self.num_relation)
            with graph.graph():
                graph.inv_mapping = inv_mapping
            return graph, nodes

        queue = deque(nodes)
        distance = dict.fromkeys(nodes[1:], self.init_distance_t)
        distance[nodes[0]] = 0
        source = dict.fromkeys(nodes[1:], 0b10)
        source[nodes[0]] = 0b01
        edge_index = []

        while queue:
            u = queue.popleft()
            if distance[u] == self.num_hop and not self.edge_one_more_hop:
                continue

            if source[u] & 0b01:
                num_neighbor = self.num_head_neighbor[distance[u]]
            else:
                num_neighbor = self.num_neighbor[distance[u]]

            expand = self.degree_in[u] > 0
            if self.degree_block_threshold is not None:
                expand &= self.degree_in[u] <= self.degree_block_threshold
            if expand:
                start = np.random.randint(0, self.degree_in[u])
                offset = self.offset_in[u]
                degree = self.degree_in[u]
                # TODO: replace with np.random.choice
                for i in range(start, start + min(num_neighbor, degree)):
                    index = offset + i % degree
                    index = self.order_in[index]
                    v = self.edge_list[index, 1]
                    assert u == self.edge_list[index, 0]
                    if v not in distance:
                        if distance[u] == self.num_hop:
                            # edge one more hop case
                            continue
                        distance[v] = distance[u] + 1
                        queue.append(v)
                    if v not in source:
                        source[v] = source[u]
                    else:
                        source[v] |= source[u]
                    edge_index.append(index)

            expand = self.degree_out[u] > 0
            if self.degree_block_threshold is not None:
                expand &= self.degree_out[u] <= self.degree_block_threshold
            if expand:
                start = np.random.randint(self.degree_out[u])
                offset = self.offset_out[u]
                degree = self.degree_out[u]
                # TODO: replace with np.random.choice
                for i in range(start, start + min(num_neighbor, degree)):
                    index = offset + i % degree
                    index = self.order_out[index]
                    v = self.edge_list[index, 0]
                    assert u == self.edge_list[index, 1]
                    if v not in distance:
                        if distance[u] == self.num_hop:
                            # edge one more hop case
                            continue
                        distance[v] = distance[u] + 1
                        queue.append(v)
                    if v not in source:
                        source[v] = source[u]
                    else:
                        source[v] |= source[u]
                    edge_index.append(index)

        if self.num_neighbor_expand:
            for u in distance:
                num_neighbor_expand = self.num_neighbor_expand[distance[u]]
                if self.degree_in[u] > 0:
                    start = np.random.randint(0, self.degree_in[u])
                    offset = self.offset_in[u]
                    degree = self.degree_in[u]
                    for i in range(start, start + min(num_neighbor_expand, degree)):
                        index = offset + i % degree
                        index = self.order_in[index]
                        v = self.edge_list[index, 1]
                        assert u == self.edge_list[index, 0]
                        if v in distance:
                            edge_index.append(index)
                if self.degree_out[u] > 0:
                    start = np.random.randint(0, self.degree_out[u])
                    degree = self.degree_out[u]
                    offset = self.offset_out[u]
                    for i in range(start, start + min(num_neighbor_expand, degree)):
                        index = offset + i % degree
                        index = self.order_out[index]
                        v = self.edge_list[index, 0]
                        assert u == self.edge_list[index, 1]
                        if v in distance:
                            edge_index.append(index)

        if return_distance:
            return distance

        if self.prune_isolated > 0:
            # backtrace BFS edges
            for index in edge_index[::-1]:
                u, v = self.edge_list[index, :2]
                # don't know the direction
                # just propagate in two directions
                source[u] |= source[v]
                source[v] |= source[u]

            node_index = []
            for k, v in source.items():
                # keep node within self.prune_isolated - 1 radius
                if v == 0b11 or distance[k] < self.prune_isolated:
                    node_index.append(k)
            node_index = list(set(node_index))

            new_edge_index = []
            for index in edge_index:
                u, v = self.edge_list[index, :2]
                if u in node_index and v in node_index:
                    new_edge_index.append(index)
            edge_index = new_edge_index
        else:
            node_index = list(distance.keys())

        edge_index = list(set(edge_index))
        mapping = {u: i for i, u in enumerate(node_index)}
        # inv_mapping = torch.tensor(nodes, dtype=torch.long).flatten()
        edge_list = self.edge_list[edge_index] # this is a copy, not in-place
        if self.use_edge_weight:
            node_in, node_out = edge_list.transpose()[:2]
            if self.log_edge_weight:
                edge_weight = 2 / (np.log(self.degree_in[node_in] + 1) + np.log(self.degree_out[node_out] + 1))
            else:
                edge_weight = 1 / np.sqrt((self.degree_in[node_in] + 1) * (self.degree_out[node_out] + 1))
            # make mean = 1
            edge_weight /= edge_weight.mean()
        for i in range(len(edge_list)):
            for j in range(2):
                edge_list[i, j] = mapping[edge_list[i, j]]
        if self.use_degree_info:
            degree_in = self.degree_in[node_index]
            degree_out = self.degree_out[node_index]
        for i in range(len(nodes)):
            nodes[i] = mapping[nodes[i]]

        num_node = len(mapping)
        if self.add_super_node:
            num_node += 1
            super_node = len(mapping)
            super_edge = np.empty((len(nodes), 3), dtype=np.int32)
            super_edge[:, 2] = self.num_relation - 1
            # distinguish the edge directions for head and tails
            super_edge[0, 0] = nodes[0]
            super_edge[0, 1] = super_node
            super_edge[1:, 0] = super_node
            super_edge[1:, 1] = nodes[1:]
            edge_list = np.concatenate([edge_list, super_edge], axis=0)

        if self.use_edge_weight:
            graph = data.Graph(edge_list, edge_weight, num_node=num_node, num_relation=self.num_relation)
        else:
            graph = data.Graph(edge_list, num_node=num_node, num_relation=self.num_relation)
        # with graph.graph():
        #     graph.inv_mapping = inv_mapping
        if self.use_degree_info:
            with graph.node():
                graph.full_degree_in = torch.tensor(degree_in, dtype=torch.float)
                graph.full_degree_out = torch.tensor(degree_out, dtype=torch.float)
        return graph, nodes

    def get_n_hop_graph(self, nodes):
        if self.use_cpp_implementation:
            return self.get_n_hop_graph_cpp(nodes)
        else:
            return self.get_n_hop_graph_py(nodes)

    def get_negative_t(self, h_index, r_index):
        neg_t_index = []

        for n in range(self.num_negative):
            if self.negative_hops is None or n < self.uniform_negative_ratio * self.num_negative:
                index = np.random.randint(0, self.num_entity)
                if self.strict_negative:
                    while index in self.hr2t[(h_index, r_index)]:
                        index = np.random.randint(0, self.num_entity)
            else:
                assert not self.strict_negative
                num_hop = np.random.choice(self.negative_hops)
                # random walk
                u = h_index
                for t in range(num_hop):
                    i = np.random.randint(self.degree_in[u] + self.degree_out[u])
                    if i < self.degree_in[u]:
                        edge_index = self.offset_in[u] + i
                        edge_index = self.order_in[edge_index]
                        assert u == self.edge_list[edge_index, 0]
                        u = self.edge_list[edge_index, 1]
                    else:
                        i = i - self.degree_in[u]
                        edge_index = self.offset_out[u] + i
                        edge_index = self.order_out[edge_index]
                        assert u == self.edge_list[edge_index, 1]
                        u = self.edge_list[edge_index, 0]
                index = u
            neg_t_index.append(index)

        return neg_t_index

    def __getitem__(self, index):
        if self.train_relation:
            h_index, t_index, r_index = self.edge_list[self.train_indexes[index]]
        else:
            h_index, t_index, r_index = self.edge_list[index]
        h_degree = self.degree_in[h_index] + self.degree_out[h_index]
        t_degree = self.degree_in[t_index] + self.degree_out[t_index]
        sample_weight = 1 / np.sqrt(h_degree * t_degree)
        if self.num_negative > 0:
            neg_t_index = self.get_negative_t(h_index, r_index)
            nodes = [h_index, t_index] + neg_t_index
            if self.use_entity_embedding:
                h_embedding = self.entity_embedding[h_index]
                t_embedding = self.entity_embedding[[t_index] + neg_t_index]
                # h_embedding = np.tile([h_embedding], (self.num_negative + 1, 1))
            if self.node_feature is not None:
                h_feature = self.node_feature[h_index]
                t_feature = self.node_feature[[t_index] + neg_t_index]
                # h_feature = np.tile([h_feature], (self.num_negative + 1, 1))
            graph, nodes = self.get_n_hop_graph(nodes)
            h_index, t_index = nodes[:2]
            neg_t_index = nodes[2:]
            h_index = np.tile([h_index], self.num_negative + 1)
            t_index = np.concatenate([[t_index], neg_t_index])
            r_index = np.tile([r_index], self.num_negative + 1)
        else:
            nodes = [h_index, t_index]
            if self.use_entity_embedding:
                h_embedding = self.entity_embedding[h_index]
                t_embedding = self.entity_embedding[t_index]
            graph, nodes = self.get_n_hop_graph(nodes)
            h_index, t_index = nodes

        triplet = np.stack([h_index, t_index, r_index], axis=-1)

        if self.use_entity_embedding:
            with graph.graph():
                graph.h_embedding = torch.tensor(h_embedding, dtype=torch.float)
                graph.t_embedding = torch.tensor(t_embedding, dtype=torch.float)
        if self.use_relation_embedding:
            with graph.graph():
                graph.r_embedding = torch.tensor(self.relation_embedding[r_index], dtype=torch.float)
        if self.node_feature is not None:
            with graph.graph():
                graph.h_feature = torch.tensor(h_feature, dtype=torch.float)
                graph.t_feature = torch.tensor(t_feature, dtype=torch.float)

        item = {
            "triplet": torch.tensor(triplet, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight),
            "graph": graph,
        }
        return item

    def __len__(self):
        if self.train_relation:
            return len(self.train_indexes)
        else:
            return self.num_edge


@R.register("datasets.WikiKG90MTestNumpy")
class WikiKG90MTestNumpy(torch_data.Dataset):

    def __init__(self, train_set, test_dict):
        self.train_set = train_set
        self.hr = test_dict["hr"]
        self.t_candidate = test_dict["t_candidate"]
        self.num_candidate = self.t_candidate.shape[-1]
        if "t_correct_index" in test_dict:
            self.t_correct_index = test_dict["t_correct_index"]
        else:
            self.t_correct_index = None

    def __getitem__(self, index):
        h_index, r_index = self.hr[index]
        t_index = self.t_candidate[index]

        nodes = [h_index] + t_index.tolist()
        if self.train_set.use_entity_embedding:
            h_embedding = self.train_set.entity_embedding[h_index]
            t_embedding = self.train_set.entity_embedding[t_index]
            h_embedding = np.tile([h_embedding], (self.num_candidate, 1))
        if self.train_set.node_feature is not None:
            h_feature = self.train_set.node_feature[h_index]
            t_feature = self.train_set.node_feature[t_index]
            # h_feature = np.tile([h_feature], (self.num_candidate, 1))
        graph, nodes = self.train_set.get_n_hop_graph(nodes)
        h_index = nodes[0]
        t_index = nodes[1:]

        triplet = np.empty((self.num_candidate, 3), dtype=np.int32)
        triplet[:, 0] = h_index
        triplet[:, 1] = t_index
        triplet[:, 2] = r_index

        if self.train_set.use_entity_embedding:
            with graph.graph():
                graph.h_embedding = torch.tensor(h_embedding, dtype=torch.float)
                graph.t_embedding = torch.tensor(t_embedding, dtype=torch.float)
        if self.train_set.use_relation_embedding:
            r_embedding = self.train_set.relation_embedding[r_index]
            r_embedding = np.tile([r_embedding], (self.num_candidate, 1))
            with graph.graph():
                graph.r_embedding = torch.tensor(r_embedding, dtype=torch.float)
        if self.train_set.node_feature is not None:
            with graph.graph():
                graph.h_feature = torch.tensor(h_feature, dtype=torch.float)
                graph.t_feature = torch.tensor(t_feature, dtype=torch.float)

        item = {
            "triplet": torch.tensor(triplet, dtype=torch.long),
            "graph": graph,
        }
        if self.t_correct_index is not None:
            item["target"] = torch.tensor(self.t_correct_index[index])
        return item

    def __len__(self):
        return len(self.hr)


class InductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):

    def load_inductive_tsvs(self, train_files, test_files, verbose=0):
        assert len(train_files) == len(test_files) == 2
        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        train_entity_vocab, inv_train_entity_vocab = self._standarize_vocab(None, inv_train_entity_vocab)
        test_entity_vocab, inv_test_entity_vocab = self._standarize_vocab(None, inv_test_entity_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

        self.train_graph = data.Graph(triplets[:num_samples[0]],
                                      num_node=len(train_entity_vocab), num_relation=len(relation_vocab))
        self.valid_graph = self.train_graph
        self.test_graph = data.Graph(triplets[sum(num_samples[:2]): sum(num_samples[:3])],
                                     num_node=len(test_entity_vocab), num_relation=len(relation_vocab))
        self.graph = self.train_graph
        self.triplets = torch.tensor(triplets[:sum(num_samples[:2])] + triplets[sum(num_samples[:3]):])
        self.num_samples = num_samples[:2] + num_samples[3:]
        self.train_entity_vocab = train_entity_vocab
        self.test_entity_vocab = test_entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_train_entity_vocab = inv_train_entity_vocab
        self.inv_test_entity_vocab = inv_test_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab

    def __getitem__(self, index):
        return self.triplets[index]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.FB15k237Inductive")
class FB15k237Inductive(InductiveKnowledgeGraphDataset):

    train_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
    ]

    test_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        train_files = []
        for url in self.train_urls:
            url = url % version
            save_file = "fb15k237_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            train_files.append(txt_file)
        test_files = []
        for url in self.test_urls:
            url = url % version
            save_file = "fb15k237_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            test_files.append(txt_file)

        self.load_inductive_tsvs(train_files, test_files, verbose=verbose)


@R.register("datasets.WN18RRInductive")
class WN18RRInductive(InductiveKnowledgeGraphDataset):

    train_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
    ]

    test_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        train_files = []
        for url in self.train_urls:
            url = url % version
            save_file = "wn18rr_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            train_files.append(txt_file)
        test_files = []
        for url in self.test_urls:
            url = url % version
            save_file = "wn18rr_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            test_files.append(txt_file)

        self.load_inductive_tsvs(train_files, test_files, verbose=verbose)