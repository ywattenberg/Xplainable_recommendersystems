import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from PIL import Image
from random import randint
from dataset.amazon_dataset_utils import parse
from dataset.amazon_dataset_utils import transform, imageHD_transform
from model.MatrixFactorizationWithImages import MatrixFactorizationWithImages


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()
    
    model = torch.nn.DataParallel(MatrixFactorizationWithImages(num_items=num_items, num_users=num_users).to(device=device))  
    model.load_state_dict(torch.load('/mnt/ds3lab-scratch/ywattenberg/models/model_weights_imgHD.pth', map_location=device))

    model = model.module

    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]

    white_base_img = torch.ones([1,3,500,500], dtype=torch.float32, requires_grad=True).to(device)
    black_base_img = torch.zeros([1,3,500,500], dtype=torch.float32, requires_grad=True).to(device)

    length = len(test_data)
    for i in range(10):
        index = randint(0, length)
        user_input = test_data.iloc[index].userID
        product_input = test_data.iloc[index].productID
        img_input = Image.open(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', f'{test_data.iloc[index].asin}.jpg'))

        rating = test_data.iloc[index].overall

        user_input_t = transform(user_input).unsqueeze(dim=0)
        product_input_t = transform(product_input).unsqueeze(dim=0)
        img_input_t = imageHD_transform(img_input).unsqueeze(dim=0)

        prediction_n = model(img_input_t, user_input_t, product_input_t)
        prediction_w = model(white_base_img, user_input_t, product_input_t)
        prediction_b = model(black_base_img, user_input_t, product_input_t)
        
        print(prediction_n)
        print(prediction_w)
        print(prediction_b)
        
