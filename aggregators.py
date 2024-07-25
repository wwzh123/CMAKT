from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(nn.Module):
    def __init__(self, batch_size, seq_len, dim, dropout, act, name):
        super(Aggregator, self).__init__()
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dim = dim

    def forward(self, self_vectors, neighbor_vectors, question_embeddings):
        outputs = self._call(self_vectors, neighbor_vectors, question_embeddings)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, question_embeddings):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass


class SumAggregator(Aggregator):
    def __init__(self, batch_size, seq_len, dim, dropout=0., act=nn.ReLU(), name=None):
        super(SumAggregator, self).__init__(batch_size, seq_len, dim, dropout, act, name)

        self.weights = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_normal_(self.weights)
        self.bias = nn.Parameter(torch.zeros(dim))

    def _call(self, self_vectors, neighbor_vectors, question_embeddings):
        # [batch_size,seq_len, -1, dim]
        neighbors_agg = neighbor_vectors.mean(dim=-2)
        output = (self_vectors + neighbors_agg).view(-1, self.dim)

        # [-1, dim]
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = torch.matmul(output, self.weights) + self.bias

        # [batch_size,seq_len, -1, dim]
        output = output.view(self.batch_size, self.seq_len, -1, self.dim)

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, seq_len, dim, dropout=0., act=torch.nn.ReLU, name=None):
        super(ConcatAggregator, self).__init__(batch_size, seq_len, dim, dropout, act, name)

        self.weights = torch.nn.Parameter(torch.empty(dim * 2, dim))
        torch.nn.init.xavier_normal_(self.weights)
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def _call(self, self_vectors, neighbor_vectors, question_embeddings):
        # [batch_size,seq_len, -1, dim]
        neighbors_agg = torch.mean(neighbor_vectors, dim=-2)

        # [batch_size,seq_len, -1, dim * 2]
        output = torch.cat([self_vectors, neighbors_agg], dim=-1)

        # [-1, dim * 2]
        output = output.view(-1, self.dim * 2)
        output = F.dropout(output, p=self.dropout, training=self.training)

        # [-1, dim]
        output = torch.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = output.view(self.batch_size, self.seq_len, -1, self.dim)

        return self.act(output)
