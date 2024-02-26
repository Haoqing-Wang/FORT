import torch
import torch.nn as nn
import numpy as np
from qpth.qp import QPFunction


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()  # (n_batch, m, depth) or (m, depth)
    index = indices.reshape(indices.size()+torch.Size([1]))  # (n_batch, m, 1) or (m, 1)
    if len(indices.size()) < 2:
        encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    else:
        encoded_indicies = encoded_indicies.scatter_(2, index, 1)
    return encoded_indicies


def kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(-1)
    matrix2_flatten = matrix2.reshape(-1)
    return torch.mm(matrix1_flatten.unsqueeze(1), matrix2_flatten.unsqueeze(0)).reshape(list(matrix1.size())+list(matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0)*matrix2.size(0), matrix1.size(1)*matrix2.size(1))


class MetaOptNet(nn.Module):
    def __init__(self, encoder, n_way, n_support):
        super(MetaOptNet, self).__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1
        self.name = 'MetaOptNet'

    def classifier(self, x, C_reg=0.1, maxIter=15):  # (n_way*(n_support+n_query), d)
        num_query, num_support, d = self.n_way * self.n_query, self.n_way * self.n_support, x.size(1)
        x = x.reshape(self.n_way, -1, d)  # (n_way, n_support+n_query, d)
        support = x[:, :self.n_support].reshape(-1, d)  # (num_support, d)
        query = x[:, self.n_support:].reshape(-1, d)  # (num_query, d)
        support_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()  # (num_support)

        kernel_matrix = torch.mm(support, support.transpose(0, 1))  # (num_support, num_support)
        id_matrix = torch.eye(self.n_way).cuda()  # (n_way, n_way)
        block_kernel_matrix = kronecker(kernel_matrix, id_matrix)  # (num_support*n_way, num_support*n_way)

        block_kernel_matrix += 1.0 * torch.eye(self.n_way * num_support).cuda()
        support_labels_one_hot = one_hot(support_labels, self.n_way).flatten()  # (num_support, n_way)

        G = block_kernel_matrix  # (num_support*n_way, num_support*n_way)
        e = -1. * support_labels_one_hot  # (num_support, n_way)
        C = torch.eye(self.n_way * num_support).cuda()  # (num_support*n_way, num_support*n_way)
        h = C_reg * support_labels_one_hot  # (num_support, n_way)
        A = kronecker(torch.eye(num_support).cuda(), torch.ones(1, self.n_way).cuda())  # (num_support, num_support*n_way)
        b = torch.zeros(num_support).cuda()  # (num_support, num_support)
        # G, e, C, h, A, b = [x.float() for x in [G, e, C, h, A, b]]
        qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())  # (1, num_support*n_way)

        compatibility = torch.mm(support, query.transpose(0, 1))  # (num_support, num_query)
        compatibility = compatibility.unsqueeze(2).expand(num_support, num_query, self.n_way)  # (num_support, num_query, n_way)
        qp_sol = qp_sol.reshape(num_support, self.n_way)  # (num_support, n_way)
        logits = qp_sol.unsqueeze(1).expand(num_support, num_query, self.n_way)  # (num_support, num_query, n_way)
        logits = logits * compatibility  # (num_support, num_query, n_way)
        logits = torch.sum(logits, 0)  # (num_query, n_way)
        return logits

    def forward(self, x):
        x = x.reshape(-1, *x.size()[2:])
        z, _ = self.encoder(x)  # (n_way*(n_support+n_query), d)
        scores = self.classifier(z)
        return scores
