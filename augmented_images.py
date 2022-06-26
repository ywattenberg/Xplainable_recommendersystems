import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from random import randint

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from dataset.amazon_dataset_utils import transform, imageHD_transform
from PIL import Image
import math


def color_dist(p1, p2):
    r1, g1, b1 = p1
    r2, g2, b2 = p2
    return math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    #num_users = df['reviewerID'].nunique()
    #num_items = df['asin'].nunique()
    
    model = torch.load('/mnt/ds3lab-scratch/ywattenberg/models/tmp_entire_model_imp.pth').to(device)

    model = model.module

    #train_data = df[df['rank_latest'] != 1]
    #test_data = df[df['rank_latest'] == 1]

    train_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv') 
    test_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv')
    
    image_transform = create_transform(**resolve_data_config({}, model=model))

    length = len(test_data)
    for i in range(10):
        index = randint(0, length)
        user_input = test_data.iloc[index].userID
        product_input = test_data.iloc[index].productID
        img_input = Image.open(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', f'{test_data.iloc[index].asin}.jpg'))

        user_input_t = transform(user_input).unsqueeze(dim=0)
        product_input_t = transform(product_input).unsqueeze(dim=0)
        img_input_t = image_transform(img_input).unsqueeze(dim=0).to(device)
        
        pred = model(img_input_t, user_input_t, product_input_t)
        change = np.zeros([2,14,14], dtype=np.float32)
        print(pred)

        with torch.no_grad():
            for x in range(14):
                for y in range(14):
                    tmp = img_input_t.clone()
                    tmp[0, :, x*16:  x*16 + 16,  y*16: y*16 + 16] = 0.0
                    pred_tmp = model(tmp, user_input_t, product_input_t)
                    change[0,x,y] = pred_tmp.cpu().numpy() - pred.cpu().numpy()
                
                    tmp[0, :, x*16:  x*16 + 16,  y*16: y*16 + 16] = 1.0
                    pred_tmp = model(tmp, user_input_t, product_input_t)
                    change[1,x,y] = pred_tmp.cpu().numpy() - pred.cpu().numpy()


        diff = np.abs(change)
        diff_w = diff[0,:,:]
        diff_b = diff[1,:,:]
        
        max_w = np.max(diff_w)
        max_b = np.max(diff_b)
        print(max_w)
        print(max_b)
        attr_w = np.zeros([224,224], dtype=np.float32)
        attr_b = np.zeros([224,224], dtype=np.float32)
        for x in range(14):
            for y in range(14):
                attr_w[x*16:  x*16 + 16,  y*16: y*16 + 16] = diff_w[x,y]/float(max_w)
                attr_b[x*16:  x*16 + 16,  y*16: y*16 + 16] = diff_b[x,y]/float(max_b)

        fig = plt.figure(figsize=(10,15))
        fig.add_subplot(2,2,1)
        plt.imshow(attr_b)
        fig.add_subplot(2,2,2)
        plt.imshow(attr_b)
        plt.imshow(img_input_t.squeeze().cpu().detach().permute(1, 2, 0).numpy(), alpha=0.3)
        fig.add_subplot(2,2,3)
        plt.imshow(attr_w)
        fig.add_subplot(2,2,4)
        plt.imshow(attr_w)
        plt.imshow(img_input_t.squeeze().cpu().detach().permute(1, 2, 0).numpy(), alpha=0.3)
        fig.savefig(f'test_img/{i}.jpg')
        plt.close(fig)
    

if __name__ == '__main__':
    main()
