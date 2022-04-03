import numpy as np
from torch import nn
import torch


class ModelMLP(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=20, dump: bool = False):
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

        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

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



    