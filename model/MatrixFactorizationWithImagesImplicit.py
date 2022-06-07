import torch
import numpy as np
from model.CNNModels import EfficentNetB4Model


class MatrixFactorizationWithImagesImplicit(torch.nn.Module):

    def __init__(self, num_users, num_items, n_factors=100):
        super().__init__()

        self.image_feature_extractor = EfficentNetB4Model(num_of_latents=n_factors)

        self.user_factors = torch.nn.Embedding(num_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(num_items, n_factors, sparse=True)
        self.user_biases = torch.nn.Embedding(num_users, 1)
        self.item_biases = torch.nn.Embedding(num_items,1)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.item_factors.weight)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)



    def forward(self, image, user, item):
        image_factors = self.image_feature_extractor(image)
        item_factors = self.item_factors(item)

        item_v = 0.5*(image_factors + item_factors)
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (self.user_factors(user) * item_v).sum(1, keepdim=True)
        return torch.sigmoid(pred.squeeze(dim=1))