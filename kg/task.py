import os
import math
import logging

import numpy as np

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import core, tasks
from torchdrug.core import Registry as R
from torchdrug import utils
from torchdrug.layers import functional
from torchdrug.utils import comm

from extension import sparse

from ogb import lsc


logger = logging.getLogger(__name__)


def debug_memory(message):
    import psutil
    p = psutil.Process()
    rank = comm.get_rank()
    torch.cuda.reset_peak_memory_stats(rank)
    logger.warning("[%d] %s, CPU: %g GiB, GPU: %g GiB" %
                   (rank, message, p.memory_info().rss / 1e9, torch.cuda.max_memory_allocated(rank) / 1e9))


@R.register("task.KnowledgeGraphEmbeddingKDDCup")
class KnowledgeGraphEmbeddingKDDCup(tasks.KnowledgeGraphEmbedding, core.Configurable):

    eps = 1e-8

    def __init__(self, model, criterion="bce",
                 metric=("mr", "mrr", "hits@1", "hits@10", "hits@100", "mrr (top-10)", "mrr (lsc)"),
                 num_negative=128, negative_weight=1, margin=6, adversarial_temperature=0, topk_negative=0,
                 strict_negative=False, num_hop=0, num_neighbor=0, distributed_storage=False, batch_down_sample=False,
                 adj_update_frequency=1000, spmm_batch_size=None, use_cpu_sample=False, use_sample_weight=False,
                 label_smoothing=0, test_relation=None, softplus_scale=1):
        super(KnowledgeGraphEmbeddingKDDCup, self).__init__(model, criterion, metric, num_negative, margin,
                                                            adversarial_temperature, True)
        self.strict_negative = strict_negative
        self.num_hop = num_hop
        self.num_neighbor = num_neighbor
        self.negative_weight = negative_weight
        self.topk_negative = topk_negative
        self.distributed_storage = distributed_storage
        self.batch_down_sample = batch_down_sample
        self.adj_update_frequency = adj_update_frequency
        self.spmm_batch_size = spmm_batch_size
        self.use_cpu_sample = use_cpu_sample
        self.use_sample_weight = use_sample_weight
        self.label_smoothing = label_smoothing
        self.softplus_scale = softplus_scale
        if test_relation is not None:
            test_relation = torch.tensor(test_relation, dtype=torch.long)
            self.register_buffer("test_relation", test_relation)
            if comm.get_rank() == 0:
                logger.warning("%d test relations are specified" % len(test_relation))
        else:
            self.test_relation = None

        self.batch_id = 0

    def preprocess(self, train_set, valid_set, test_set):
        self.num_entity = train_set.num_entity
        self.num_relation = train_set.num_relation

        # degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
        # degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
        # for h, t, r in train_set:
        #     degree_hr[h, r] += 1
        #     degree_tr[t, r] += 1
        #
        # self.register_buffer("degree_hr", degree_hr)
        # self.register_buffer("degree_tr", degree_tr)

        if not self.use_cpu_sample:
            if self.distributed_storage:
                assert self.num_hop > 0
                graph = train_set.graph
                mask = torch.arange(comm.get_rank(), graph.num_edge, comm.get_world_size())
                g = graph.edge_mask(mask)
                debug_memory("before register")
                self.register_buffer("dist_graph", g)
            else:
                self.register_buffer("train_graph", train_set.graph)

        if isinstance(test_set, torch_data.Subset):
            assert isinstance(test_set.indices, range)
            self.test_range = test_set.indices
        else:
            self.test_range = range(len(test_set))

    def forward(self, batch, all_loss=None, metric=None):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        pos_pred = pred[:, 0]
        ranking = torch.sum(pos_pred.unsqueeze(-1) <= pred, dim=-1).clamp(min=1)
        metric["mrr"] = (1 / ranking.float()).mean()

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                target = torch.ones_like(pred) * self.label_smoothing
                target[:, 0] = 1 - self.label_smoothing
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

                if self.topk_negative > 0:
                    pos_loss, neg_loss = loss.split([1, loss.shape[-1] - 1], dim=-1)
                    pos_pred, neg_pred = pred.split([1, loss.shape[-1] - 1], dim=-1)
                    neg_loss, topk_index = neg_loss.topk(self.topk_negative, dim=-1)
                    neg_pred = neg_pred.gather(-1, topk_index)
                    loss = torch.cat([pos_loss, neg_loss], dim=-1)
                    pred = torch.cat([pos_pred, neg_pred], dim=-1)

                neg_weight = torch.ones_like(pred)
                if self.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = self.negative_weight * F.softmax(pred[:, 1:] / self.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = self.negative_weight / (neg_weight.shape[1] - 1)
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            elif criterion == "ce":
                target = torch.zeros(len(pred), dtype=torch.long, device=self.device)
                loss = F.cross_entropy(pred, target, reduction="none")
            elif criterion == "ranking":
                positive = pred[:, :1]
                negative = pred[:, 1:]
                target = torch.ones_like(negative)
                loss = F.margin_ranking_loss(positive, negative, target, margin=self.margin, reduction="none")
                loss = loss.mean(dim=-1)
            elif criterion == "softplus":
                positive = pred[:, :1]
                negative = pred[:, 1:]
                loss = F.softplus((negative - positive) * self.softplus_scale)
                loss = loss.mean(dim=-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            # sample_weight = self.degree_hr[pos_h_index, pos_r_index] * self.degree_tr[pos_t_index, pos_r_index]
            # sample_weight = 1 / sample_weight.float().sqrt()
            # loss = (loss * sample_weight).sum() / sample_weight.sum()
            if self.use_sample_weight:
                sample_weight = batch["sample_weight"]
                loss = (loss * sample_weight).sum() / sample_weight.sum()
            else:
                loss = loss.mean()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        batch_size = len(batch)

        if all_loss is None:
            # test
            if self.use_cpu_sample:
                graph = batch["graph"]
                h_index, t_index, r_index = batch["triplet"].unbind(-1)
                if self.test_relation is not None:
                    is_valid = (r_index[:, 0].unsqueeze(-1) == self.test_relation.unsqueeze(0)).any(dim=-1)
                    graph = graph.subbatch(is_valid)
                    h_index = h_index[is_valid]
                    t_index = t_index[is_valid]
                    r_index = r_index[is_valid]

                offset = (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)
                h_index = h_index + offset
                t_index = t_index  + offset
            else:
                raise NotImplementedError

            if len(graph) > 0:
                pred = self.model(graph, h_index, t_index, r_index)
                # in case of GPU OOM
                pred = pred.cpu()
            else:
                pred = torch.zeros(h_index.shape, dtype=torch.float)
        else:
            # train
            if self.use_cpu_sample:
                graph = batch["graph"]
                offset = graph.num_cum_nodes - graph.num_nodes
                pos_h_index, pos_t_index, pos_r_index = batch["triplet"].unbind(-1)
                if pos_h_index.ndim == 2:
                    offset = offset.unsqueeze(-1)
                pos_h_index = pos_h_index + offset
                pos_t_index = pos_t_index + offset
                metric["#sampled node (per triplet)"] = graph.num_node / len(graph)
                metric["#sampled edge (per triplet)"] = graph.num_edge / len(graph)
            else:
                pos_h_index, pos_t_index, pos_r_index = batch.transpose(0, -1)
                nodes = torch.stack([pos_h_index, pos_t_index], dim=-1)
                if self.distributed_storage:
                    if self.batch_down_sample:
                        graph, new_nodes = self.get_batch_n_hop_graph_distributed(self.dist_graph, nodes)
                        pos_h_index, pos_t_index = new_nodes.t()
                        assert (pos_h_index >= 0).all()
                        assert (pos_t_index >= 0).all()
                        metric["#sampled node (per triplet)"] = graph.num_node / len(graph)
                        metric["#sampled edge (per triplet)"] = graph.num_edge / len(graph)
                    else:
                        nodes = comm.cat(nodes).flatten()
                        graph, mapping = self.get_n_hop_graph_distributed(self.dist_graph, nodes)
                        pos_h_index = mapping[pos_h_index]
                        pos_t_index = mapping[pos_t_index]
                        assert (pos_h_index >= 0).all()
                        assert (pos_t_index >= 0).all()
                        metric["#sampled node (per batch)"] = graph.num_node.float()
                        metric["#sampled edge (per batch)"] = graph.num_edge.float()
                else:
                    if self.batch_down_sample:
                        graph, new_nodes = self.get_batch_n_hop_graph(self.train_graph, nodes)
                        pos_h_index, pos_t_index = new_nodes.t()
                        assert (pos_h_index >= 0).all()
                        assert (pos_t_index >= 0).all()
                        metric["#sampled node (per triplet)"] = graph.num_node / len(graph)
                        metric["#sampled edge (per triplet)"] = graph.num_edge / len(graph)
                    else:
                        raise NotImplementedError

            if pos_h_index.ndim == 1:
                if self.strict_negative:
                    if self.batch_down_sample:
                        neg_index = self.batch_strict_negative_sampling(pos_h_index, pos_t_index, pos_r_index, graph)
                    else:
                        neg_index = self.strict_negative_sampling(pos_h_index, pos_t_index, pos_r_index, graph)
                else:
                    neg_index = torch.randint(graph.num_node, (batch_size, self.num_negative), device=self.device)
                h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
                r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)

                t_index[:, 1:] = neg_index
            else:
                # negative samples are already provided by the dataloader
                h_index = pos_h_index
                t_index = pos_t_index
                r_index = pos_r_index

            pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
            # if self.batch_id % 10 == 0:
            #     debug_memory("training")

            self.batch_id += 1

        return pred

    def target(self, batch):
        # in case of GPU OOM
        if "target" in batch:
            if self.test_relation is not None:
                r_index = batch["triplet"][:, 0, -1]
                is_valid = (r_index.unsqueeze(-1) == self.test_relation.unsqueeze(0)).any(dim=-1)
                return batch["target"][is_valid].cpu()
            else:
                return batch["target"].cpu()
        else:
            # dummy
            return -torch.ones(1, dtype=torch.long)

    def save_test_submission(self, input_dict):
        assert 'h,r->t' in input_dict
        assert 't_pred_top10' in input_dict['h,r->t']

        t_pred_top10 = input_dict['h,r->t']['t_pred_top10']

        # assert t_pred_top10.shape == (1359303, 10)
        assert (0 <= t_pred_top10).all() and (t_pred_top10 < 1001).all()

        if isinstance(t_pred_top10, torch.Tensor):
            t_pred_top10 = t_pred_top10.cpu().numpy()
        t_pred_top10 = t_pred_top10.astype(np.int16)

        filename = "t_pred_wikikg90m_%d_%d" % (self.test_range.start, self.test_range.stop)
        test_range = np.array([self.test_range.start, self.test_range.stop])
        np.savez_compressed(filename, t_pred_top10=t_pred_top10, test_range=test_range)

    def evaluate(self, pred, target):
        if target[0] < 0: # test set, no target
            if comm.get_rank() == 0:
                logger.warning("save submission file")
            input_dict = {}
            input_dict["h,r->t"] = {"t_pred_top10": pred.topk(k=10, dim=-1)[1]}
            self.save_test_submission(input_dict)
            return

        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        # don't need to add 1 because pred[target] == pos_pred
        # clamp for safety
        ranking = torch.sum(pos_pred <= pred, dim=-1).clamp(min=1)

        metric = {}
        for _metric in self.metric:
            if _metric == "mr":
                score = ranking.float().mean()
            elif _metric == "mrr":
                score = (1 / ranking.float()).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (ranking <= threshold).float().mean()
            elif _metric == "mrr (top-10)":
                zeros = torch.zeros(ranking.shape, device=ranking.device)
                score = torch.where(ranking > 10, zeros, 1 / ranking.float())
                score = score.mean()
            elif _metric == "mrr (lsc)":
                evaluator = lsc.WikiKG90MEvaluator()
                input_dict = {}
                input_dict["h,r->t"] = {"t_pred_top10": pred.topk(k=10, dim=-1)[1], "t_correct_index": target}
                score = evaluator.eval(input_dict)["mrr"]
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric

    @torch.no_grad()
    def strict_negative_sampling(self, pos_h_index, pos_t_index, pos_r_index, graph):
        batch_size = len(pos_h_index)

        h_index_set, h_inverse = torch.unique(pos_h_index, return_inverse=True)
        h_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=self.device)
        h_index2id[h_index_set] = torch.arange(len(h_index_set), device=self.device)

        r_index_set, r_inverse = torch.unique(pos_r_index, return_inverse=True)
        r_index2id = -torch.ones(graph.num_relation, dtype=torch.long, device=self.device)
        r_index2id[r_index_set] = torch.arange(len(r_index_set), device=self.device)

        t_index_set, t_inverse = torch.unique(pos_t_index, return_inverse=True)
        t_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=self.device)
        t_index2id[t_index_set] = torch.arange(len(t_index_set), device=self.device)

        # compact pos indexes
        pos_h_index_compact = h_index2id[pos_h_index]
        pos_r_index_compact = r_index2id[pos_r_index]
        pos_t_index_compact = t_index2id[pos_t_index]

        hr_index_set, hr_inverse = torch.unique(pos_h_index_compact * len(r_index_set) + pos_r_index_compact, return_inverse=True)

        hr_index2id = -torch.ones(len(h_index_set) * len(r_index_set), dtype=torch.long, device=self.device)
        hr_index2id[hr_index_set] = torch.arange(len(hr_index_set), device=self.device)

        h_index, t_index, r_index = graph.edge_list.t()
        valid = (h_index2id[h_index] >= 0) & (r_index2id[r_index] >= 0)
        h_index = h_index[valid]
        r_index = r_index[valid]
        t_index = t_index[valid]

        h_index_compact = h_index2id[h_index]
        r_index_compact = r_index2id[r_index]
        hr_index_compact = h_index_compact * len(r_index_set) + r_index_compact

        valid = hr_index2id[hr_index_compact] >= 0
        hr_index_compact = hr_index_compact[valid]
        t_index = t_index[valid]

        t_mask_set = torch.ones(len(hr_index_set), graph.num_node, dtype=torch.bool, device=self.device)
        t_mask_set[hr_index2id[hr_index_compact], t_index] = 0
        t_mask = t_mask_set[hr_inverse]

        num_neg_t = t_mask.sum(dim=-1)
        num_cum_neg_t = num_neg_t.cumsum(0)

        neg_t_index = t_mask.nonzero()[:, 1]

        index = (torch.rand(batch_size, self.num_negative, device=self.device) * num_neg_t.unsqueeze(-1)).long()
        index = index + (num_cum_neg_t - num_neg_t).unsqueeze(-1)
        neg_index = neg_t_index[index]

        return neg_index

    @torch.no_grad()
    def batch_strict_negative_sampling(self, pos_h_index, pos_t_index, pos_r_index, graph):
        batch_size = len(pos_h_index)

        h_index_set, h_inverse = torch.unique(pos_h_index, return_inverse=True)
        h_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=self.device)
        h_index2id[h_index_set] = torch.arange(len(h_index_set), device=self.device)

        r_index_set, r_inverse = torch.unique(pos_r_index, return_inverse=True)
        r_index2id = -torch.ones(graph.num_relation, dtype=torch.long, device=self.device)
        r_index2id[r_index_set] = torch.arange(len(r_index_set), device=self.device)

        t_index_set, t_inverse = torch.unique(pos_t_index, return_inverse=True)
        t_index2id = -torch.ones(graph.num_node, dtype=torch.long, device=self.device)
        t_index2id[t_index_set] = torch.arange(len(t_index_set), device=self.device)

        # compact pos indexes
        pos_h_index_compact = h_index2id[pos_h_index]
        pos_r_index_compact = r_index2id[pos_r_index]
        pos_t_index_compact = t_index2id[pos_t_index]

        hr_index_set, hr_inverse = torch.unique(pos_h_index_compact * len(r_index_set) + pos_r_index_compact,
                                                return_inverse=True)

        hr_index2id = -torch.ones(len(h_index_set) * len(r_index_set), dtype=torch.long, device=self.device)
        hr_index2id[hr_index_set] = torch.arange(len(hr_index_set), device=self.device)

        h_index, t_index, r_index = graph.edge_list.t()
        valid = (h_index2id[h_index] >= 0) & (r_index2id[r_index] >= 0)
        h_index = h_index[valid]
        r_index = r_index[valid]
        t_index = t_index[valid]

        h_index_compact = h_index2id[h_index]
        r_index_compact = r_index2id[r_index]
        hr_index_compact = h_index_compact * len(r_index_set) + r_index_compact

        valid = hr_index2id[hr_index_compact] >= 0
        hr_index_compact = hr_index_compact[valid]
        t_index = t_index[valid]

        hr_index_set_h_compact = hr_index_set // len(r_index_set)
        hr_index_set_h = h_index_set[hr_index_set_h_compact]
        graph_index = graph.node2graph[hr_index_set_h]
        start = graph.num_cum_nodes[graph_index] - graph.num_nodes[graph_index]
        end = graph.num_cum_nodes[graph_index]
        start = start.unsqueeze(-1)
        end = end.unsqueeze(-1)

        t_mask_set = torch.zeros(len(hr_index_set), graph.num_node + 1, dtype=torch.long, device=self.device)
        t_mask_set.scatter_(1, start, torch.ones_like(start))
        t_mask_set.scatter_(1, end, -torch.ones_like(end))
        t_mask_set = t_mask_set.cumsum(-1).bool()
        t_mask_set[hr_index2id[hr_index_compact], t_index] = 0
        t_mask = t_mask_set[hr_inverse]

        num_neg_t = t_mask.sum(dim=-1)
        num_cum_neg_t = num_neg_t.cumsum(0)

        neg_t_index = t_mask.nonzero()[:, 1]

        index = (torch.rand(batch_size, self.num_negative, device=self.device) * num_neg_t.unsqueeze(-1)).long()
        index = index + (num_cum_neg_t - num_neg_t).unsqueeze(-1)
        neg_index = neg_t_index[index]

        graph_index = torch.arange(batch_size, device=self.device)
        assert (graph.node2graph[neg_index] == graph_index.unsqueeze(-1)).all()

        return neg_index

    @torch.no_grad()
    def get_n_hop_graph(self, graph, center_nodes):
        # TODO: This is only for debug. Not guaranteed to be correct.
        if self.num_hop == 0:
            return graph

        node_in, node_out = graph.edge_list.t()[:2]
        edge_weight = graph.edge_weight
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node))
        # adj: 9.9 GB
        # graph: 14 GB
        # CPU->GPU: 3s
        adjacency = adjacency.coalesce().cuda(self.device)
        print("cuda memory: %g GiB" % (torch.cuda.max_memory_allocated() / 1e9))

        for t in range(5):
            # CPU: 16s
            # GPU: 1.75s
            logger.warning("<<<< start")
            visited = torch.zeros(graph.num_node, 1, device=self.device)
            visited[center_nodes] = 1

            for i in range(self.num_hop):
                new_visited1 = sparse.generalized_spmm(adjacency, visited, addition="max")
                # in-place transpose
                indices = adjacency.indices()
                indices[:] = indices[[1, 0]]
                new_visited2 = sparse.generalized_spmm(adjacency, visited, addition="max")
                # in-place transpose
                indices = adjacency.indices()
                indices[:] = indices[[1, 0]]
                visited = torch.cat([visited, new_visited1, new_visited2], dim=-1).max(dim=-1)[0]
                visited = (visited > 0.5).float()

            visited = visited.squeeze(-1) > 0.5
            logger.warning("end >>>>")

            # CPU: 6s
            graph_ = graph.subgraph(visited)

        return graph_

    @torch.no_grad()
    def down_sample_adjacency_one_side(self, graph, node_x, num_neighbor):
        torch.cuda.synchronize(self.device)
        logger.warning("[%d] begin down sample" % comm.get_rank())
        size = torch.bincount(node_x, minlength=graph.num_node)

        # sort (randomly order same elements)
        perm = torch.randperm(graph.num_edge, device=self.device)
        order = node_x[perm].argsort() # argsort is very slow
        order = perm[order]

        starts = size.cumsum(0) - size
        ends = starts + size.clamp(max=num_neighbor)
        mask = functional.multi_slice_mask(starts, ends, graph.num_edge)
        edge_mask = order[mask]

        assert (torch.bincount(node_x[edge_mask], minlength=graph.num_node) == size.clamp(max=num_neighbor)).all()

        adjacency = utils.sparse_coo_tensor(graph.edge_list[edge_mask, :2].t(), graph.edge_weight[edge_mask],
                                            (graph.num_node, graph.num_node))

        torch.cuda.synchronize(self.device)
        logger.warning("[%d] end down sample" % comm.get_rank())
        return adjacency

    @torch.no_grad()
    def down_sample_adjacency(self, graph, transposed=False):
        node_in, node_out = graph.edge_list.t()[:2]
        if self.distributed_storage:
            num_neighbor = self.num_neighbor // comm.get_world_size()
        else:
            num_neighbor = self.num_neighbor
        adj_in = self.down_sample_adjacency_one_side(graph, node_in, num_neighbor)
        debug_memory("after adjacency")
        adj_out = self.down_sample_adjacency_one_side(graph, node_out, num_neighbor)
        debug_memory("after inv adjacency")
        if transposed:
            adj = adj_in.t() + adj_out
        else:
            adj = adj_in + adj_out.t()
        adj = adj.coalesce()
        return adj

    @torch.no_grad()
    def get_n_hop_graph_distributed(self, graph, nodes):
        assert self.num_hop > 0

        if not hasattr(self, "adjacency_t") or self.batch_id % self.adj_update_frequency == 0:
            debug_memory("start down sample adjacency")
            self.adjacency_t = self.down_sample_adjacency(graph, transposed=True)
            debug_memory("end down sample adjacency")
        adjacency_t = self.adjacency_t

        visited = torch.zeros(graph.num_node, dtype=torch.bool, device=self.device)
        visited[nodes] = 1

        for i in range(self.num_hop):
            new_visited = adjacency_t @ visited.unsqueeze(-1).float()
            new_visited = new_visited.squeeze(-1) > 0.5
            visited = torch.max(visited, new_visited)
            # inter-GPU sync
            visited = comm.stack(visited).max(dim=0)[0]

        new_graph = graph.subgraph(visited)
        # inter-GPU sync
        edge_list = comm.cat(new_graph.edge_list)
        new_graph = type(new_graph)(edge_list, num_node=new_graph.num_node, num_relation=new_graph.num_relation)

        mapping = -torch.ones(graph.num_node, dtype=torch.long, device=self.device)
        mapping[visited] = torch.arange(visited.sum(), device=self.device)

        return new_graph, mapping

    def spmm_binary(self, x, y):
        # y is binary matrix
        # prune zero rows
        is_non_zero = y.any(dim=-1)
        indices = x._indices()
        values = x._values()
        mask = is_non_zero[indices[1]]
        x_pruned = utils.sparse_coo_tensor(indices[:, mask], values[mask], x.shape)
        result = x_pruned @ y.float()
        return result

    @torch.no_grad()
    def get_batch_n_hop_graph_distributed(self, graph, nodes):
        # nodes: single GPU's batch h_index & t_index
        assert self.num_hop > 0

        if not hasattr(self, "adjacency_t") or self.batch_id % self.adj_update_frequency == 0:
            debug_memory("start down sample adjacency")
            self.adjacency_t = self.down_sample_adjacency(graph, transposed=True)
            debug_memory("end down sample adjacency")
        adjacency_t = self.adjacency_t

        batch_size = nodes.shape[0]
        spmm_batch_size = self.spmm_batch_size // comm.get_world_size()

        new_graphs = []
        new_nodes = []
        for start in range(0, batch_size, spmm_batch_size):
            end = min(start + spmm_batch_size, batch_size)
            actual_batch_size = end - start
            visited = torch.zeros(actual_batch_size, graph.num_node, dtype=torch.bool, device=self.device)
            sample_nodes = nodes[start: end]
            ones = torch.ones(sample_nodes.shape, dtype=torch.bool, device=self.device)
            visited.scatter_add_(1, sample_nodes, ones)

            # compute all GPUs batch together
            actual_batch_sizes = comm.cat(torch.tensor([actual_batch_size], device=self.device))
            sample_nodes = comm.cat(sample_nodes)
            visited = comm.cat(visited)

            for i in range(self.num_hop):
                new_visited = self.spmm_binary(adjacency_t, visited.t()) > 0.5
                new_visited = new_visited.t()
                visited = visited | new_visited
                # inter-GPU sync
                # stack causes GPU OOM
                # visited = comm.stack(visited).max(dim=0)[0]
                visited = comm.reduce(visited)

            spmm_new_graphs = []
            spmm_new_nodes = []
            # TODO: faster implementation of subgraphs, surpassing this loop
            for i in range(len(visited)):
                sample_visited = visited[i]
                sample_node = sample_nodes[i]
                new_graph = graph.subgraph(sample_visited)
                # inter-GPU sync
                edge_list = comm.cat(new_graph.edge_list)
                new_graph = type(new_graph)(edge_list, num_node=new_graph.num_node, num_relation=new_graph.num_relation)

                mapping = sample_visited.cumsum(0) - 1
                mapping[~sample_visited] = -1
                new_node = mapping[sample_node]
                spmm_new_graphs.append(new_graph)
                spmm_new_nodes.append(new_node)

            # get the batch of this GPU
            start = actual_batch_sizes[:comm.get_rank()].sum()
            new_graphs += spmm_new_graphs[start: start + actual_batch_size]
            new_nodes += spmm_new_nodes[start: start + actual_batch_size]

        new_graph = type(graph).pack(new_graphs)
        new_nodes = torch.stack(new_nodes)
        new_nodes = new_nodes + (new_graph.num_cum_nodes - new_graph.num_nodes).unsqueeze(-1)

        return new_graph, new_nodes

    @torch.no_grad()
    def get_batch_n_hop_graph(self, graph, nodes):
        assert self.num_hop > 0

        if not hasattr(self, "adjacency_t") or self.batch_id % self.adj_update_frequency == 0:
            debug_memory("start down sample adjacency")
            self.adjacency_t = self.down_sample_adjacency(graph, transposed=True)
            debug_memory("end down sample adjacency")
        adjacency_t = self.adjacency_t

        batch_size = nodes.shape[0]
        visited = torch.zeros(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        ones = torch.ones(nodes.shape, dtype=torch.bool, device=self.device)
        visited.scatter_add_(1, nodes, ones)

        debug_memory("begin spmm")
        for i in range(self.num_hop):
            new_visited = []
            # split to prevent GPU OOM
            for v in visited.split(self.spmm_batch_size):
                new_visited.append(self.spmm_binary(adjacency_t @ v.t()) > 0.5)
            new_visited = torch.cat(new_visited, dim=-1)
            new_visited = new_visited.t()
            visited = torch.max(visited, new_visited)
        debug_memory("end spmm")

        new_graphs = []
        new_nodes = []
        # TODO: faster implementation of subgraphs, surpassing this loop
        for i in range(batch_size):
            sample_visited = visited[i]
            sample_node = nodes[i]
            new_graph = graph.subgraph(sample_visited)
            mapping = sample_visited.cumsum(0) - 1
            mapping[~sample_visited] = -1
            new_node = mapping[sample_node]
            new_graphs.append(new_graph)
            new_nodes.append(new_node)
        new_graph = type(graph).pack(new_graphs)
        new_nodes = torch.stack(new_nodes)
        new_nodes = new_nodes + (new_graph.num_cum_nodes - new_graph.num_nodes).unsqueeze(-1)

        return new_graph, new_nodes