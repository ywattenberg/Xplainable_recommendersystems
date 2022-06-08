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


def get_trainer_imageHD(model_fn):
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]
    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()

    train_data = AmazonCSJDatasetWithIMGHD(path=None, df=train_data)
    test_data = AmazonCSJDatasetWithIMGHD(path=None, df=test_data)

    model = model_fn(num_items=num_items, num_users=num_users)
    
    loss_fn = torch.nn.MSELoss()
    optimize = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-8)
    
    return Trainer(model, loss_fn, optimize, train_data, test_data, batch_size=80, epochs=4)


def get_trainer_vgg16_HD():
   return get_trainer_imageHD(get_MF_with_images_vgg16)

def get_trainer_mixer_HD():
    return get_trainer_imageHD(get_MF_with_images_Mixerl16)

def get_trainer_efficent_HD():
    return get_trainer_imageHD(get_MF_with_images_EfficentNetB4)

if __name__ == '__main__':
    #trainer = get_trainer_vgg16_HD()
    #trainer = get_trainer_efficent_HD()
    trainer = get_trainer_mixer_HD()
    trainer.train_test()
    