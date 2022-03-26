from turtle import forward
import numpy as np
from torch import nn
import torch


class ModelMLP(nn.Module):

    def __init__(self, num_users, num_items, dump: bool = False):
        super(ModelMLP, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=20, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )

        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=10)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=10)

    def forward(self, user_input, product_input):
        user_embedded = self.user_embedding(user_input)
        product_embedded = self.item_embedding(product_input)
        vector = torch.cat([user_embedded, product_embedded], dim=-1)
        score =  self.stack(vector)
        return torch.mul(score, 5)

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predict_labels = self(user_input, item_input)
        score = torch.mul(predict_labels, 5)
        return nn.MSELoss(score, labels)

class ModelMatrixFactorization(torch.nn.Module):

    def __init__(self, num_users, num_items, n_factors=20):
        super().__init__()
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

    