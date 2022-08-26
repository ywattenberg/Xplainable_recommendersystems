from pickletools import optimize
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.amazon_csj_dataset import AmazonCSJDatasetWithIMGHD, AmazonCSJDatasetWithIMG
from models.MatrixFactorizationWithImages import get_MF_with_images_Mixer12_split, get_MF_with_images_Efficent_split, get_MF_with_images_vgg16,  get_MF_with_images_EfficentNetB4, get_MF_with_images_Mixerl16, get_MF_only_images_Mixer12
from dataset.amazon_dataset_utils import *
from trainer import Trainer

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def get_trainer_imageHD(model_fn, timm_model=False, image_transform=None):
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]
    #train_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv')
    #test_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv')
    num_users = df['userID'].nunique()
    num_items = df['productID'].nunique()
    print(num_users)
    print(train_data['reviewerID'].nunique())


    model = torch.nn.DataParallel(model_fn(num_items=num_items, num_users=num_users))
    #model = torch.load('tmp_entire_model_imp.pth')

    if timm_model:
        config = resolve_data_config({}, model=model)
        image_transform = create_transform(**config)

    train_data = AmazonCSJDatasetWithIMGHD(path=None, df=train_data, prev_image_transform=image_transform)
    test_data = AmazonCSJDatasetWithIMGHD(path=None, df=test_data, prev_image_transform=image_transform)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    return Trainer(model, train_data, test_data, loss_fn, optimizer, batch_size=250, epochs=4)


def get_trainer_vgg16_HD():
   return get_trainer_imageHD(get_MF_with_images_vgg16)

def get_trainer_mixer_HD(): 
    return get_trainer_imageHD(get_MF_with_images_Mixerl16, timm_model=True)

def get_trainer_efficent_HD():
    return get_trainer_imageHD(get_MF_with_images_EfficentNetB4)

def get_trainer_mixer_HD_split():
     return get_trainer_imageHD(get_MF_with_images_Mixer12_split, timm_model=True)

def get_trainer_efficent_HD_split():
     return get_trainer_imageHD(get_MF_with_images_Efficent_split)

def get_trainer_mixer_HD_only():
     return get_trainer_imageHD(get_MF_only_images_Mixer12, timm_model=True)

if __name__ == '__main__':
    #trainer = get_trainer_vgg16_HD()
    trainer = get_trainer_efficent_HD_split()
    #trainer = get_trainer_mixer_HD_split()
    trainer.train_test()
    
