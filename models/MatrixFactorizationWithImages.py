import torch
import numpy as np
from torchvision import models
from models.CNNModels import vgg16Model
from models.CNNModels import EfficentNetB4Model
from models.MixerModel import mixer_l16, resmlp_12

class MatrixFactorizationWithImages(torch.nn.Module):

    def __init__(self, num_users, num_items, n_factors=100, feature_extractor=None):
        super().__init__()

        self.image_feature_extractor = feature_extractor

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
        return pred.squeeze(dim=1)

class MatrixFactorizationWithImages_split(torch.nn.Module):

    def __init__(self, num_users, num_items, n_factors=100, feature_extractor=None, item_factors=10):
        super().__init__()

        print(num_users)
        self.image_feature_extractor = feature_extractor

        self.user_factors = torch.nn.Embedding(num_users, n_factors)
        self.item_factors = torch.nn.Embedding(num_items, item_factors)
        self.user_biases = torch.nn.Embedding(num_users, 1)
        self.item_biases = torch.nn.Embedding(num_items,1)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.item_factors.weight)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)

    def forward(self, image, user, item):
        image_factors = self.image_feature_extractor(image)
        item_factors = self.item_factors(item)
        item_v = torch.cat((image_factors, item_factors), 1)
        user_embedding = self.user_factors(user)
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (user_embedding * item_v).sum(1, keepdim=True)
        return pred.squeeze(dim=1)

class MatrixFactorizationOnlyImages(torch.nn.Module):

    def __init__(self, num_users, num_items, n_factors=100, feature_extractor=None):
        super().__init__()

        self.image_feature_extractor = feature_extractor
        self.user_factors = torch.nn.Embedding(num_users, n_factors, sparse=True)
        self.user_biases = torch.nn.Embedding(num_users, 1)
        self.item_biases = torch.nn.Embedding(num_items,1)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)

    def forward(self, image, user, item):
        image_factors = self.image_feature_extractor(image)
        #item_factors = self.item_factors(item)
        #item_v = torch.cat((image_factors, item_factors), 1)
        user_embedding = self.user_factors(user)
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (user_embedding * image_factors).sum(1, keepdim=True)
        return pred.squeeze(dim=1)

def get_MF_with_images_vgg16(num_users, num_items, n_factors=100):
    return MatrixFactorizationWithImages(num_users=num_users, num_items=num_items, n_factors=n_factors,            
                                            feature_extractor=vgg16Model(num_of_latents=n_factors))
    
def get_MF_with_images_EfficentNetB4(num_users, num_items, n_factors=100):
    return MatrixFactorizationWithImages(num_users=num_users, num_items=num_items, n_factors=n_factors,            
                                            feature_extractor=EfficentNetB4Model(num_of_latents=n_factors))

def get_MF_with_images_Mixerl16(num_users, num_items, n_factors=100):
    return MatrixFactorizationWithImages(num_users=num_users, num_items=num_items, n_factors=n_factors,            
                                            feature_extractor=mixer_l16(num_classes=n_factors))

def get_MF_with_images_Mixer12_split(num_users, num_items, n_factors=100, item_factors=10):
    return MatrixFactorizationWithImages_split(num_users=num_users, num_items=num_items, n_factors=n_factors,            
                                            feature_extractor=resmlp_12(num_classes=n_factors-item_factors), item_factors=item_factors)

def get_MF_with_images_Efficent_split(num_users, num_items, n_factors=100, item_factors=10):
    return MatrixFactorizationWithImages_split(num_users=num_users, num_items=num_items, n_factors=n_factors,            
                                            feature_extractor=EfficentNetB4Model(num_of_latents=n_factors-item_factors), item_factors=item_factors)

def get_MF_only_images_Mixer12(num_users, num_items, n_factors=100):
    return MatrixFactorizationOnlyImages(num_users=num_users, num_items=num_items, n_factors=n_factors,            
                                            feature_extractor=resmlp_12(num_classes=100))
