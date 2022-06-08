import torch
from torch import nn
import numpy as np


class FactorizationMachineModel(torch.nn.Module):

    def __init__(self, field_dims, embedding_dim):
        super().__init__()

        self.V = nn.Parameter(torch.randn(sum(field_dims), embedding_dim),requires_grad=True)
        self.lin = nn.Linear(sum(field_dims), 1)
        

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out
