import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from random import randint
import torchvision.transforms as T
import math
from PIL import Image

import sys
sys.path.append('../Xplainable_recommendersystems')

from dataset.amazon_dataset_utils import transform, imageHD_transform


def color_dist(p1, p2):
    r1, g1, b1 = p1
    r2, g2, b2 = p2
    return math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    #num_users = df['reviewerID'].nunique()
    #num_items = df['asin'].nunique()
    
    model = torch.load('/mnt/ds3lab-scratch/ywattenberg/models/entire_model_mixer_split.pth').to(device)

    model = model.module

    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]

    #train_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv') 
    #test_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv')
    
    # image_transform = imageHD_transform 
    image_transform = T.Compose([T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=None), T.CenterCrop(size=(224, 224)), T.ToTensor(), T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))])

    length = len(test_data)
    for i in range(100):
        index = randint(0, length)
        user_input = test_data.iloc[index].userID
        product_input = test_data.iloc[index].productID
        img_input = Image.open(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', f'{test_data.iloc[index].asin}.jpg'))

        user_input_t = transform(user_input).unsqueeze(dim=0)
        product_input_t = transform(product_input).unsqueeze(dim=0)
        img_input_t = image_transform(img_input).unsqueeze(dim=0).to(device)
        
        pred = model(img_input_t, user_input_t, product_input_t)
        change = np.zeros([2,14,14], dtype=np.float32)
        mean = img_input_t.mean()

        with torch.no_grad():
            for x in range(8):
                for y in range(8):
                    tmp = img_input_t.clone()
                    tmp[0, :, x*28:  x*28 + 28,  y*28: y*28 + 28] = 0.0
                    #tmp[0, :, x*16:  x*16 + 16,  y*16: y*16 + 16] = 0.0
                    pred_tmp = model(tmp, user_input_t, product_input_t)
                    change[0,x,y] = pred_tmp.cpu().numpy() - pred.cpu().numpy()

                    tmp[0, :, x*28:  x*28 + 28,  y*28: y*28 + 28] = mean
                    #tmp[0, :, x*16:  x*16 + 16,  y*16: y*16 + 16] = 1.0
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
        for x in range(7):
            for y in range(7):
                attr_w[x*32:  x*32 + 32,  y*32: y*32 + 32] = diff_w[x,y]/float(max_w)
                attr_b[x*32:  x*32 + 32,  y*32: y*32 + 32] = diff_b[x,y]/float(max_b)
        
        # for x in range(14):
        #     for y in range(14):
        #         attr_w[x*16:  x*16 + 16,  y*16: y*16 + 16] = diff_w[x,y]/float(max_w)
        #         attr_b[x*16:  x*16 + 16,  y*16: y*16 + 16] = diff_b[x,y]/float(max_b)

        fig = plt.figure(figsize=(10,15))
        fig.add_subplot(2,2,1)
        plt.imshow(attr_b)
        fig.add_subplot(2,2,2)
        plt.imshow(attr_b)
        plt.imshow(img_input_t.squeeze().cpu().detach().permute(1, 2, 0).numpy(), alpha=0.2)
        fig.add_subplot(2,2,3)
        plt.imshow(attr_w)
        fig.add_subplot(2,2,4)
        plt.imshow(attr_w)
        plt.imshow(img_input_t.squeeze().cpu().detach().permute(1, 2, 0).numpy(), alpha=0.2)
        fig.savefig(f'test_img/{i}.jpg')
        plt.close(fig)
    
def colour_change(model, img_input, user_input, product_in):
    image_transform = T.Compose([T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=None), T.CenterCrop(size=(224, 224)), T.ToTensor(), T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))])
    img_input = image_transform(img_input).unsqueeze(dim=0).to('cuda')
    user_input = transform(user_input).unsqueeze(dim=0)
    product_input = transform(product_input).unsqueeze(dim=0)

    with torch.no_grad():
        pred = model(img_input, user_input, product_input)
    red = img_input.clone()
    blue = img_input.clone()
    green = img_input.clone()
    for x in range(224):
        for y in range(224):
            pixel = img_input[0, :, x, y].cpu().numpy()
            if color_dist(pixel, (1,1,1)) > 0.1:
                if red[0, 0, x, y] > 0.8:
                    red[0, 0, x, y] = 1.0
                else:
                    red[0, 0, x, y] = pixel[0] + 0.2
                
                if green[0, 1, x, y] > 0.8:
                    green[0, 1, x, y] = 1.0
                else:
                    green[0, 1, x, y] = pixel[1] + 0.2
                
                if blue[0, 2, x, y] > 0.8:
                    blue[0, 2, x, y] = 1.0
                else:
                    blue[0, 2, x, y] = pixel[2] + 0.2

    with torch.no_grad():
        pred_red = model(red, user_input, product_input)
        pred_blue = model(blue, user_input, product_input)
        pred_green = model(green, user_input, product_input)
    
    diff = torch.abs(pred_blue - pred)
    diff += torch.abs(pred_red - pred)
    diff += torch.abs(pred_green - pred)
    
    return diff


def gen_explanation(model, img_input, user_input, product_input, tmm_model=False):
    
    image_transform = T.Compose([T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=None), T.CenterCrop(size=(224, 224)), T.ToTensor(), T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))])
    img_input = image_transform(img_input).unsqueeze(dim=0).to('cuda')
    user_input = transform(user_input).unsqueeze(dim=0)
    product_input = transform(product_input).unsqueeze(dim=0)

    with torch.no_grad():
        pred = model(img_input, user_input, product_input)
        change = np.zeros([2,14,14], dtype=np.float32)
        mean = img_input.mean()

        for x in range(8):
            for y in range(8):
                tmp = img_input.clone()
                tmp[0, :, x*28:  x*28 + 28,  y*28: y*28 + 28] = 1.0
                #tmp[0, :, x*16:  x*16 + 16,  y*16: y*16 + 16] = 0.0
                pred_tmp = model(tmp, user_input, product_input)
                change[0,x,y] = pred_tmp.cpu().numpy() - pred.cpu().numpy()

                tmp[0, :, x*28:  x*28 + 28,  y*28: y*28 + 28] = mean
                #tmp[0, :, x*16:  x*16 + 16,  y*16: y*16 + 16] = 1.0
                pred_tmp = model(tmp, user_input, product_input)
                change[1,x,y] = pred_tmp.cpu().numpy() - pred.cpu().numpy()

    diff = np.abs(change)
    diff_w = diff[0,:,:]
    diff_b = diff[1,:,:]
    
    max_w = np.max(diff_w)
    max_b = np.max(diff_b)
    attr_w = np.zeros([224,224], dtype=np.float32)
    attr_b = np.zeros([224,224], dtype=np.float32)
    for x in range(7):
        for y in range(7):
            attr_w[x*32:  x*32 + 32,  y*32: y*32 + 32] = diff_w[x,y]/float(max_w)
            attr_b[x*32:  x*32 + 32,  y*32: y*32 + 32] = diff_b[x,y]/float(max_b)
    
    return attr_w, attr_b

if __name__ == '__main__':
    main()
