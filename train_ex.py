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
from train_models import get_trainer_imageHD, get_trainer_simple

def loop():
    #model_fn = get_MF_with_images_Mixer12_split
    path_to_csv = '/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv'
    batch_size_df = pd.DataFrame(columns=['item feature', 'loss', 'time'])  
    for i in [10, 50, 70]:
        trainer = get_trainer_imageHD(model_fn=get_MF_with_images_Efficent_split, path_to_csv=path_to_csv, batch_size=512, lr=0.01, epochs=2, item_factors=i, timm_model = True)
        start = time.time()
        trainer.train_loop()
        loss = trainer.test_loop()
        total = time.time() - start
        batch_size_df.loc[len(batch_size_df)] = [i, loss, total]
        batch_size_df.to_csv('split_test.csv')



if __name__ == '__main__':
    loop()
