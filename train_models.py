from pickletools import optimize
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.amazon_csj_dataset import AmazonCSJDatasetWithIMGHD, AmazonCSJDatasetWithIMG
from models.MatrixFactorizationWithImages import get_MF_with_images_vgg16, get_MF_with_images_EfficentNetB4, get_MF_with_images_Mixerl16
from dataset.amazon_dataset_utils import *
from trainer import Trainer

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def get_trainer_imageHD(model_fn, timm_model=False, image_transform=None):
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]
    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()



    model = torch.nn.DataParallel(model_fn(num_items=num_items, num_users=num_users))
    
    if timm_model:
        config = resolve_data_config({}, model=model)
        image_transform = create_transform(**config)

    train_data = AmazonCSJDatasetWithIMGHD(path=None, df=train_data, image_transform=image_transform)
    test_data = AmazonCSJDatasetWithIMGHD(path=None, df=test_data, image_transform=image_transform)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    return Trainer(model, train_data, test_data, loss_fn, optimizer, batch_size=200, epochs=4)


def get_trainer_vgg16_HD():
   return get_trainer_imageHD(get_MF_with_images_vgg16)

def get_trainer_mixer_HD(): 
    return get_trainer_imageHD(get_MF_with_images_Mixerl16, timm_model=True)

def get_trainer_efficent_HD():
    return get_trainer_imageHD(get_MF_with_images_EfficentNetB4)

if __name__ == '__main__':
    #trainer = get_trainer_vgg16_HD()
    #trainer = get_trainer_efficent_HD()
    trainer = get_trainer_mixer_HD()
    trainer.train_test()
    
