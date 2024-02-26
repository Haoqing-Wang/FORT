import torch
import torch.nn as nn
import numpy as np


class ProtoNet(nn.Module):
    def __init__(self, encoder, n_way, n_support):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1
        self.name = 'ProtoNet'

    def forward(self, x):
        x = x.reshape(-1, *x.size()[2:])
        z_all, _ = self.encoder(x)  # (100, 2048)
        z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        z_proto = z_support.mean(1)
        z_query = z_query.reshape(self.n_way * self.n_query, -1)
        scores = -euclidean_dist(z_query, z_proto)
        return scores


def euclidean_dist(x, y):  # x:(n, d)  y:(m, d)
    y = y.transpose(0, 1)  # (d, m)
    xx = (x * x).sum(1, keepdim=True)  # (n, 1)
    yy = (y * y).sum(0, keepdim=True)  # (1, m)
    xy = x @ y  # (n, m)
    return xx - 2. * xy + yy
