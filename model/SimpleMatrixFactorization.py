from pyexpat import model
import torch
import numpy as np
from torchvision import models

class ModelMatrixFactorization(torch.nn.Module):

    def __init__(self, num_users, num_items, n_factors=20):
        super().__init__()

        self.vvg16 = models.vgg16(pretrained=True)

        self.user_factors = torch.nn.Embedding(num_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(num_items, n_factors, sparse=True)
        self.user_biases = torch.nn.Embedding(num_users, 1)
        self.item_biases = torch.nn.Embedding(num_items,1)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.item_factors.weight)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)

    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
        return pred.squeeze()