import math
import warnings

import torch
from torch._six import container_abcs
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min

from torchdrug import core, data, layers, models, utils
from torchdrug.utils import comm
from torchdrug.core import Registry as R
from torchdrug.layers import functional

from extension import sparse


def edge_dropout(graph, p=0):
    perm = torch.randperm(graph.num_edge, device=graph.device)[:int((1 - p) * graph.num_edge)]
    return graph.edge_mask(perm)


@R.register("model.MLPScoreKDDCup")
class MLPScoreKDDCup(layers.MLP, core.Configurable):

    def __init__(self, num_entity, num_relation, embedding_dim, hidden_dims, activation="relu", dropout=0,
                 num_feature=3):
        super(MLPScoreKDDCup, self).__init__(embedding_dim * num_feature, hidden_dims, activation, dropout)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.num_feature = num_feature

        self.relation = nn.Embedding(num_relation, embedding_dim)
        self.entity_norm = nn.LayerNorm(embedding_dim)
        self.relation_norm = nn.LayerNorm(embedding_dim)

        nn.init.kaiming_uniform_(self.relation.weight, a=math.sqrt(5), mode="fan_in")

    def forward(self, head, tail, relation, h_index=None, t_index=None, r_index=None, all_loss=None, metric=None):
        head = self.entity_norm(head)
        tail = self.entity_norm(tail)
        relation = self.relation_norm(relation)
        if h_index is not None:
            h = head[h_index]
            r = relation[r_index]
            t = tail[t_index]
        else:
            h = head
            r = relation
            t = tail
        x = torch.cat([h, r, t], dim=-1)
        score = super(MLPScoreKDDCup, self).forward(x)
        score = score.squeeze(-1)
        return score

    def forward_feature(self, feature):
        score = super(MLPScoreKDDCup, self).forward(feature)
        score = score.squeeze(-1)
        return score


class BatchedSemiringGraphConvKDDCup(nn.Module):

    def __init__(self, input_dim, output_dim, num_relation, layer_norm=False, batch_norm=False, instance_norm=False,
                 pna_aggregation=False, message_func="DistMult", aggregate_func="sum", aggregate_scale=None,
                 num_mlp_layer=1, activation="relu", group_size=4, degree_feature=False, squeeze_and_excitation=False,
                 msg_layer_norm=False, log_log_scale=False, query_dependent=False,
                 eps=1):
        super(BatchedSemiringGraphConvKDDCup, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mlp_layer = num_mlp_layer
        self.group_size = group_size
        self.degree_feature = degree_feature
        self.squeeze_and_excitation = squeeze_and_excitation
        self.log_log_scale = log_log_scale
        self.query_dependent = query_dependent
        self.eps = eps

        if pna_aggregation:
            if pna_aggregation == "bilinear":
                if log_log_scale:
                    add_model_dim = output_dim * 26
                else:
                    add_model_dim = output_dim * 16
            elif pna_aggregation == True or pna_aggregation == "no-bias":
                if log_log_scale:
                    add_model_dim = output_dim * 21
                else:
                    add_model_dim = output_dim * 13
            else:
                raise ValueError
        else:
            add_model_dim = output_dim
        if degree_feature:
            add_model_dim += 2
        self.add_model = layers.MLP(add_model_dim, [output_dim] * num_mlp_layer, activation=activation)
        if pna_aggregation == "no-bias":
            assert num_mlp_layer == 1
            self.add_model = nn.Linear(add_model_dim, output_dim, bias=False)
        else:
            self.add_model = layers.MLP(add_model_dim, [output_dim] * num_mlp_layer, activation=activation)

        if query_dependent:
            assert message_func == "DistMult"
            hidden_dims = [num_relation * input_dim]
            if isinstance(query_dependent, int):
                hidden_dims = [input_dim] * (query_dependent - 1) + hidden_dims
            self.query2relation = layers.MLP(input_dim, hidden_dims, activation=activation)
        if message_func == "group-MLP":
            self.relation = nn.Embedding(num_relation, input_dim * group_size)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)
        if message_func == "DistMult+":
            self.relation_bias = nn.Embedding(num_relation, input_dim)

        if squeeze_and_excitation:
            if pna_aggregation:
                self.excitation_layer = layers.MLP(output_dim * 12, [output_dim] * 2, activation=activation)
            else:
                self.excitation_layer = layers.MLP(output_dim, [output_dim] * 2, activation=activation)
        self.pna_aggregation = pna_aggregation
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.aggregate_scale = aggregate_scale
        assert message_func in ["TransE", "DistMult", "DistMult+", "DistMult-sigmoid", "ComplEx", "RotatE", "SimplE", "group-MLP"]
        assert aggregate_func in ["sum", "mean", "max", "min", "std", "bilinear-mean", "bilinear-sum"]
        assert aggregate_scale in [None, "log", "log^-1"]

        if message_func == "RotatE":
            max_score = 9
            nn.init.uniform_(self.relation.weight, -max_score * 2 / input_dim, max_score * 2 / input_dim)
            pi = torch.acos(torch.zeros(1)).item() * 2
            self.relation_scale = pi * input_dim / max_score / 2

        assert layer_norm + instance_norm + batch_norm <= 1
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if msg_layer_norm:
            self.msg_layer_norm = nn.LayerNorm(output_dim)
        else:
            self.msg_layer_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if instance_norm:
            self.instance_norm = layers.InstanceNorm(output_dim, affine=True)
        else:
            self.instance_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def message(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        x = input[node_in]

        if self.message_func == "SimplE":
            relation = self.relation.weight.chunk(2)[0]
            relation = torch.cat([relation, relation.flip(-1)], dim=0)
        elif self.message_func == "DistMult-sigmoid":
            relation = F.sigmoid(self.relation.weight)
        elif self.query_dependent:
            relation = self.query2relation(graph.query).view(graph.batch_size, -1, self.input_dim)
        else:
            relation = self.relation.weight

        if relation.ndim == 2:
            edge_feature = relation[graph.edge_list[:, 2]]
        elif relation.ndim == 3:
            edge_feature = relation[graph.edge2graph, graph.edge_list[:, 2]]

        if self.message_func == "TransE":
            message = x + edge_feature
        elif self.message_func in ["DistMult", "DistMult-sigmoid", "SimplE"]:
            message = x * edge_feature
        elif self.message_func == "DistMult+":
            message = x * edge_feature + self.relation_bias.weight[graph.edge_list[:, 2]]
        elif self.message_func == "ComplEx":
            x_re, x_im = x.chunk(2, dim=-1)
            r_re, r_im = edge_feature.chunk(2, dim=-1)
            message_re = x_re * r_re - x_im * r_im
            message_im = x_re * r_im + x_im * r_re
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "RotatE":
            edge_feature = edge_feature.chunk(2, dim=-1)[0] * self.relation_scale
            x_re, x_im = x.chunk(2, dim=-1)
            r_re, r_im = torch.cos(edge_feature), torch.sin(edge_feature)

            message_re = x_re * r_re - x_im * r_im
            message_im = x_re * r_im + x_im * r_re
            message = torch.cat([message_re, message_im], dim=-1)
        elif self.message_func == "group-MLP":
            num_edge = graph.num_edge
            edge_feature = edge_feature.view(num_edge * self.input_dim // self.group_size, self.group_size, self.group_size)
            x = x.view(num_edge * self.input_dim // self.group_size, 1, self.group_size)
            message = torch.bmm(x, edge_feature).view(num_edge, self.input_dim)
        else:
            raise ValueError

        if self.msg_layer_norm:
            message = self.msg_layer_norm(message)
        return message

    def aggregate(self, graph, message, boundary):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        message = message * edge_weight

        if self.aggregate_func == "sum":
            update = scatter_add(message, node_out, dim=0, dim_size=graph.num_node) + boundary
        elif self.aggregate_func == "mean":
            sum = scatter_add(message, node_out, dim=0, dim_size=graph.num_node) + boundary
            degree_out = graph.degree_out.unsqueeze(-1) + 1
            if message.ndim == 3:
                degree_out = degree_out.unsqueeze(-1)
            update = sum / degree_out
        elif self.aggregate_func == "max":
            update = scatter_max(message, node_out, dim=0, dim_size=graph.num_node)[0]
            update = torch.max(update, boundary)
        elif self.aggregate_func == "min":
            update = scatter_min(message, node_out, dim=0, dim_size=graph.num_node)[0]
            update = torch.min(update, boundary)
        elif self.aggregate_func == "std":
            sum = scatter_add(message, node_out, dim=0, dim_size=graph.num_node) + boundary
            sq_sum = scatter_add(message ** 2, node_out, dim=0, dim_size=graph.num_node) + boundary ** 2
            degree_out = graph.degree_out.unsqueeze(-1) + 1
            if message.ndim == 3:
                degree_out = degree_out.unsqueeze(-1)
            update = (sq_sum / degree_out - (sum / degree_out) ** 2).clamp(min=1e-6).sqrt()
        elif self.aggregate_func == "bilinear-mean":
            sum = scatter_add(message, node_out, dim=0, dim_size=graph.num_node) + boundary
            sq_sum = scatter_add(message ** 2, node_out, dim=0, dim_size=graph.num_node) + boundary ** 2
            degree_out = graph.degree_out.unsqueeze(-1) + 1
            if message.ndim == 3:
                degree_out = degree_out.unsqueeze(-1)
            update = (sum ** 2 - sq_sum) / degree_out / (degree_out - 1).clamp(min=1)
        elif self.aggregate_func == "bilinear-sum":
            sum = scatter_add(message, node_out, dim=0, dim_size=graph.num_node) + boundary
            sq_sum = scatter_add(message ** 2, node_out, dim=0, dim_size=graph.num_node) + boundary ** 2
            update = (sum ** 2 - sq_sum) / 2

        degree_out = graph.degree_out.unsqueeze(-1) + self.eps
        if message.ndim == 3:
            degree_out = degree_out.unsqueeze(-1)
        if self.aggregate_scale == "log":
            update = update * degree_out.log()
        elif self.aggregate_scale == "log^-1":
            update = update / degree_out.log().clamp(min=1e-2)

        return update

    def pna_aggregate(self, graph, message, boundary):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        if message.ndim == 3:
            edge_weight = edge_weight.unsqueeze(-1)
        message = message * edge_weight

        # message augmentation
        message = torch.cat([message, boundary])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=self.device)])

        mean = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)
        sq_mean = scatter_mean(message ** 2, node_out, dim=0, dim_size=graph.num_node)
        min = scatter_min(message, node_out, dim=0, dim_size=graph.num_node)[0]
        max = scatter_max(message, node_out, dim=0, dim_size=graph.num_node)[0]

        degree_out = graph.degree_out.unsqueeze(-1) + self.eps
        if message.ndim == 3:
            degree_out = degree_out.unsqueeze(-1)
        std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
        features = torch.cat([mean, min, max, std], dim=-1)

        scale = degree_out.log() # >= 0
        if self.log_log_scale:
            log_scale = (scale + 1).log() # >= 0
            log_scale = log_scale / log_scale.mean()
            scale = scale / scale.mean()
            scales = [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2), log_scale, 1 / log_scale.clamp(min=1e-2)]
        else:
            scale = scale / scale.mean()
            scales = [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)]
        scales = torch.cat(scales, dim=-1)

        update = features.unsqueeze(-1) * scales.unsqueeze(-2)
        update = update.flatten(-2)
        if self.degree_feature:
            degree = torch.stack([graph.full_degree_in, graph.full_degree_out], dim=-1)
            update = torch.cat([update, (degree + 1).log()], dim=-1)

        return update

    def bilinear_pna_aggregate(self, graph, message, boundary):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        if message.ndim == 3:
            edge_weight = edge_weight.unsqueeze(-1)
        message = message * edge_weight

        # message augmentation
        message = torch.cat([message, boundary])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=self.device)])

        mean = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)
        sq_mean = scatter_mean(message ** 2, node_out, dim=0, dim_size=graph.num_node)
        min = scatter_min(message, node_out, dim=0, dim_size=graph.num_node)[0]
        max = scatter_max(message, node_out, dim=0, dim_size=graph.num_node)[0]

        degree_out = graph.degree_out.unsqueeze(-1) + self.eps
        if message.ndim == 3:
            degree_out = degree_out.unsqueeze(-1)
        bilinear_mean = (mean ** 2 * degree_out - sq_mean) / (degree_out - 1).clamp(min=1)
        features = torch.cat([mean, sq_mean, min, max, bilinear_mean], dim=-1)

        scale = degree_out.log() # >= 0
        if self.log_log_scale:
            log_scale = (scale + 1).log() # >= 0
            log_scale = log_scale / log_scale.mean()
            scale = scale / scale.mean()
            scales = [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2), log_scale, 1 / log_scale.clamp(min=1e-2)]
        else:
            scale = scale / scale.mean()
            scales = [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)]
        scales = torch.cat(scales, dim=-1)

        update = features.unsqueeze(-1) * scales.unsqueeze(-2)
        update = update.flatten(-2)
        if self.degree_feature:
            degree = torch.stack([graph.full_degree_in, graph.full_degree_out], dim=-1)
            update = torch.cat([update, (degree + 1).log()], dim=-1)

        return update

    def combine(self, graph, input, update):
        if self.pna_aggregation:
            output = self.add_model(torch.cat([update, input], dim=-1))
        else:
            output = update + self.add_model(update)

        if self.squeeze_and_excitation:
            if self.pna_aggregation:
                mean = scatter_mean(output, graph.node2graph, dim=0, dim_size=graph.batch_size)
                sq_mean = scatter_mean(output ** 2, graph.node2graph, dim=0, dim_size=graph.batch_size)
                min = scatter_min(output, graph.node2graph, dim=0, dim_size=graph.batch_size)[0]
                max = scatter_max(output, graph.node2graph, dim=0, dim_size=graph.batch_size)[0]

                num_nodes = graph.num_nodes.float()
                if output.ndim == 3:
                    num_nodes = num_nodes.unsqueeze(-1)
                std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
                features = torch.cat([mean, min, max, std], dim=-1)

                scale = num_nodes.log()
                scale = scale / scale.mean()
                scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)

                feature = features.unsqueeze(-1) * scales.unsqueeze(-2)
                feature = feature.flatten(-2)
            else:
                feature = scatter_mean(output, graph.node2graph, dim=0, dim_size=graph.batch_size)
            gate = self.excitation_layer(feature)
            gate = torch.sigmoid(gate)
            output = output * gate[graph.node2graph]
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.instance_norm:
            output = self.instance_norm(graph, output)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, graph, input, boundary):
        message = self.message(graph, input)
        if self.pna_aggregation == "bilinear":
            update = self.bilinear_pna_aggregate(graph, message, boundary)
        elif self.pna_aggregation == True or self.pna_aggregation == "no-bias":
            update = self.pna_aggregate(graph, message, boundary)
        else:
            update = self.aggregate(graph, message, boundary)
        output = self.combine(graph, input, update)
        return output, message, self.relation.weight


@R.register("model.BatchedSemiringGCNKDDCup")
class BatchedSemiringGraphConvolutionalNetworkKDDCup(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, layer_norm=False, batch_norm=False, instance_norm=False,
                 pna_aggregation=False, message_func="DistMult", aggregate_func="sum", aggregate_scale=None,
                 num_mlp_layer=1, activation="relu", readout="sum", short_cut=False, concat_hidden=False,
                 share_weight=False, dropout_edge=0, message_l2_norm_weight=0, relation_l2_norm_weight=0, group_size=4,
                 stochastic_depth=None, short_cut_weight=1, degree_feature=False, squeeze_and_excitation=False,
                 msg_layer_norm=False, log_log_scale=False, query_dependent=False, eps=1):
        nn.Module.__init__(self)

        if not isinstance(hidden_dims, container_abcs.Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.dropout_edge = dropout_edge
        self.message_l2_norm_weight = message_l2_norm_weight
        self.relation_l2_norm_weight = relation_l2_norm_weight
        self.stochastic_depth = stochastic_depth
        self.short_cut_weight = short_cut_weight

        if share_weight:
            layer = BatchedSemiringGraphConvKDDCup(self.dims[0], self.dims[1], num_relation, layer_norm, batch_norm,
                                                   instance_norm, pna_aggregation, message_func, aggregate_func,
                                                   aggregate_scale, num_mlp_layer, activation,
                                                   group_size, degree_feature, squeeze_and_excitation, msg_layer_norm,
                                                   log_log_scale, query_dependent, eps)
            self.layers = nn.ModuleList([layer] * len(hidden_dims))
        else:
            self.layers = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layers.append(
                    BatchedSemiringGraphConvKDDCup(self.dims[i], self.dims[i + 1], num_relation, layer_norm, batch_norm,
                                                   instance_norm, pna_aggregation, message_func, aggregate_func,
                                                   aggregate_scale, num_mlp_layer, activation,
                                                   group_size, degree_feature, squeeze_and_excitation, msg_layer_norm,
                                                   log_log_scale, query_dependent, eps)
                )
        self.relation = nn.Embedding(num_relation, input_dim)

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        if self.dropout_edge > 0 and self.training:
            graph = edge_dropout(graph, self.dropout_edge)
            input = input / (1 - self.dropout_edge)

        hiddens = []
        message_norms = []
        relation_norms = []
        layer_input = input
        boundary = input
        for i, layer in enumerate(self.layers):
            if self.stochastic_depth is not None:
                prob = 1 - (1 - self.stochastic_depth) * i / (len(self.layers) - 1)
                if self.training:
                    if torch.rand(1) > prob:
                        # skip this layer
                        continue

            hidden, message, relation = layer(graph, layer_input, boundary)

            if self.stochastic_depth is not None:
                if not self.training:
                    hidden = hidden * prob

            if self.short_cut:
                hidden = hidden + layer_input * self.short_cut_weight
            hiddens.append(hidden)
            layer_input = hidden

            message_norm = message.norm(dim=-1) ** 2
            relation_norm = relation.norm(dim=-1) ** 2
            # per graph mean
            message_norm = scatter_mean(message_norm, graph.edge2graph, dim=0, dim_size=graph.batch_size)
            message_norms.append(message_norm.mean())
            relation_norms.append(relation_norm.mean())

        if all_loss is not None:
            # TODO: sum or mean?
            if self.message_l2_norm_weight > 0:
                message_norm = torch.stack(message_norms).sum()
                all_loss += self.message_l2_norm_weight * message_norm
                metric["message squared l2 norm"] = message_norm
            if self.relation_l2_norm_weight > 0:
                relation_norm = torch.stack(relation_norms).sum()
                all_loss += self.relation_l2_norm_weight * relation_norm
                metric["relation squared l2 norm"] = relation_norm

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
