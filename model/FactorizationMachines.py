import torch
from torch import nn
import numpy as np


class FactorizationMachineModel(torch.nn.Module):
    """
    Credit: github.com/rixwew \n
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embedding_dim):
        super().__init__()
        self.fm = nn.FactorizationMachine(reduce_sum=True)

        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        self.linear = nn.Embedding(sum(field_dims), embedding_dim)
        self.lin_bias = nn.Parameter(torch.zeros((1,)))
        self.offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        

    def forward(self, x):
        lx = x + x.new_tensor(self.offset).unsqueeze(0)
        linear_res = torch.sum(self.linear(lx), dim=1) + self.lin_bias

        fx = self.embedding(x)
        square_of_sum = torch.sum(fx, dim=1) ** 2
        sum_of_square = torch.sum(fx ** 2, dim=1)
        fm_res = torch.sum((square_of_sum - sum_of_square), dim=1, keepdim=True)

        res = linear_res + 0.5 * fm_res

        return torch.sigmoid(res.squeeze(1))
