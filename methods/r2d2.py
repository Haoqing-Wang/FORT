import torch
import torch.nn as nn
import numpy as np


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


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse. Hence, we are solving AX=I.
    Parameters:
      b_mat:  a (n, n) Tensor.
    Returns: a (n, n) Tensor.
    """
    eyes = b_mat.new_ones(b_mat.size(-1)).diag().cuda()  # (n, n)
    b_inv, _ = torch.solve(eyes, b_mat)
    return b_inv


class R2D2(nn.Module):
    def __init__(self, encoder, n_way, n_support):
        super(R2D2, self).__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1
        self.name = 'R2D2'

    def classifier(self, x):  # (n_way*(n_support+n_query), d)
        num_query, num_support, d, lam = self.n_way * self.n_query, self.n_way * self.n_support, x.size(1), 50.
        x = x.reshape(self.n_way, -1, d)  # (n_way, n_support+n_query, d)
        support = x[:, :self.n_support].reshape(-1, d)  # (num_support, d)
        query = x[:, self.n_support:].reshape(-1, d)  # (num_query, d)
        support_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()  # (num_support)

        support_labels_one_hot = one_hot(support_labels, self.n_way)  # (num_support, n_way)
        id_matrix = torch.eye(num_support).cuda()  # (num_support, num_support)

        ridge_sol = torch.mm(support, support.transpose(0, 1)) + lam * id_matrix  # (num_support, num_support)
        ridge_sol = binv(ridge_sol)  # (num_support, num_support)
        ridge_sol = torch.mm(support.transpose(0, 1), ridge_sol)  # (d, num_support)
        ridge_sol = torch.mm(ridge_sol, support_labels_one_hot)  # (d, n_way)

        logits = torch.mm(query, ridge_sol)  # (num_query, n_way)
        return logits

    def forward(self, x):
        x = x.reshape(-1, *x.size()[2:])
        z, _ = self.encoder(x)  # (n_way*(n_support+n_query), d)
        scores = self.classifier(z)
        return scores