from pickletools import optimize
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset.amazon_csj_dataset import AmazonCSJDatasetWithIMGHD, AmazonCSJDatasetWithIMG, AmazonCSJDataset
from models.MatrixFactorizationWithImages import get_MF_with_images_Mixer12_split, get_MF_with_images_Efficent_split, get_MF_with_images_vgg16,  get_MF_with_images_EfficentNetB4, get_MF_with_images_Mixerl16, get_MF_only_images_Mixer12
from models.SimpleMatrixFactorization import ModelMatrixFactorization
from dataset.amazon_dataset_utils import *
from trainer import Trainer

def get_trainer_imageHD(model_fn, timm_model=False, image_transform=None, path_to_csv='/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv', img_path=None , epochs=4, batch_size=512, lr=0.01, weight_decay=0.0, optimizer_fn=torch.optim.SGD, loss_fn=torch.nn.MSELoss, n_features=100, item_feature=10):
    #df = pd.read_csv(path_to_csv)
    #train_data = df[df['rank_latest'] != 1]
    #test_data = df[df['rank_latest'] == 1]
    train_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv')
    test_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv')
    num_users = train_data['userID'].nunique()
    num_items = train_data['productID'].nunique()
    print(num_users)
    print(test_data['reviewerID'].nunique())


    model = torch.nn.DataParallel(model_fn(num_items=num_items, num_users=num_users, n_factors=n_features, item_feature=item_feature))
    #model = torch.load('tmp_entire_model_imp.pth')

    if timm_model:
        image_transform = T.Compose([T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=None), T.CenterCrop(size=(224, 224)), T.ToTensor(), T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))])

    train_data = AmazonCSJDatasetWithIMGHD(path=None, df=train_data, prev_image_transform=image_transform, img_path=img_path)
    test_data = AmazonCSJDatasetWithIMGHD(path=None, df=test_data, prev_image_transform=image_transform, img_path=img_path)
    loss_fn = torch.nn.MSELoss()
    optimizer = optimizer_fn(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    return Trainer(model, train_data, test_data, loss_fn, optimizer, batch_size=batch_size, epochs=epochs)

def get_trainer_simple(path_to_csv='/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv', img_path=None , epochs=4, batch_size=512, lr=0.1, weight_decay=0.0, optimizer_fn=torch.optim.SGD, loss_fn=torch.nn.MSELoss):
    
    df = pd.read_csv(path_to_csv)
    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]
    #train_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv')
    #test_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv')
    num_users = df['userID'].nunique()
    num_items = df['productID'].nunique()


    model = torch.nn.DataParallel(ModelMatrixFactorization(num_items=num_items, num_users=num_users, n_factors=100))
    #model = torch.load('tmp_entire_model_imp.pth')
    
    train_data = AmazonCSJDataset(path=None, df=train_data)
    test_data = AmazonCSJDataset(path=None, df=test_data)
    loss_fn = torch.nn.MSELoss()
    optimizer = optimizer_fn(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    return Trainer(model, train_data, test_data, loss_fn, optimizer, batch_size=batch_size, epochs=epochs)


def get_trainer_vgg16_HD():
   return get_trainer_imageHD(get_MF_with_images_vgg16)

def get_trainer_mixer_HD(): 
    return get_trainer_imageHD(get_MF_with_images_Mixerl16, timm_model=True)

def get_trainer_efficent_HD():
    return get_trainer_imageHD(get_MF_with_images_EfficentNetB4)

def get_trainer_mixer_HD_split():
     return get_trainer_imageHD(get_MF_with_images_Mixer12_split, timm_model=True)

def get_trainer_efficent_HD_split():
     return get_trainer_imageHD(get_MF_with_images_Efficent_split, timm_model=True)

def get_trainer_mixer_HD_only():
     return get_trainer_imageHD(get_MF_only_images_Mixer12, timm_model=True)

if __name__ == '__main__':
    #trainer = get_trainer_vgg16_HD()
    trainer = get_trainer_efficent_HD_split()
    #trainer = get_trainer_mixer_HD_split()
    trainer.train_test()
    
