from pickletools import optimize
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset.amazon_csj_dataset import AmazonCSJDatasetWithIMGHD, AmazonCSJDatasetWithIMG
from models.MatrixFactorizationWithImages import get_MF_with_images_Mixer12_split, get_MF_with_images_Efficent_split, get_MF_with_images_vgg16,  get_MF_with_images_EfficentNetB4, get_MF_with_images_Mixerl16, get_MF_only_images_Mixer12
from dataset.amazon_dataset_utils import *
from trainer import Trainer
from train_models import get_trainer_imageHD

def loop():
    model_fn = get_MF_with_images_Mixer12_split
    path_to_csv = '/mnt/ds3lab-scratch/ywattenberg/data/compact_fashion_ImgHD.csv'
    batch_size_df = pd.DataFrame(columns=['batch_size', 'loss', 'time'])  
    for i in range(9):
        trainer = get_trainer_imageHD(model_fn, path_to_csv=path_to_csv, batch_size=2 ** (i+1), timm_model=True, img_path='/mnt/ds3lab-scratch/ywattenberg/data/fashion_imagesHD/')
        start = time.time()
        trainer.train_loop()
        total = time.time() - start
        batch_size_df = pd.concat()
        batch_size_df.append({'batch_size': 2 ** (i+1), 'loss': trainer.current_test_loss, 'time': total}, ignore_index=True)
    batch_size_df.to_csv('batch_size_mixer_split.csv')



if __name__ == '__main__':
    loop()
