import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from random import randint
from model.MatrixFactorizationWithImages import MatrixFactorizationWithImages
from dataset.amazon_dataset_utils import transform, imageHD_transform
from PIL import Image
import math


def color_dist(p1, p2):
    r1, g1, b1 = p1
    r2, g2, b2 = p2
    return math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()
    
    model = torch.nn.DataParallel(MatrixFactorizationWithImages(num_items=num_items, num_users=num_users).to(device=device))  
    model.load_state_dict(torch.load('/mnt/ds3lab-scratch/ywattenberg/models/model_weights_imgHD.pth', map_location=device))

    model = model.module

    #train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]

    length = len(test_data)
    for i in range(10):
        index = randint(0, length)
        user_input = test_data.iloc[index].userID
        product_input = test_data.iloc[index].productID
        img_input = Image.open(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', f'{test_data.iloc[index].asin}.jpg'))

        user_input_t = transform(user_input).unsqueeze(dim=0)
        product_input_t = transform(product_input).unsqueeze(dim=0)
        img_input_t = imageHD_transform(img_input).unsqueeze(dim=0)

        pred = model(img_input_t, user_input_t, product_input_t)
        print(pred)
        fig = plt.figure(figsize=(10,15))
        fig.add_subplot(2, 1, 1)
        plt.imshow(img_input)
        plt.title(pred.item())


        for x in range(img_input.width):
            for y in range(img_input.height):
                if color_dist(img_input.getpixel((x,y)), (255, 255 ,255)) < 30:
                    r, g, b = img_input.getpixel((x,y))
                    r = r+150 if r < 155 else 255
                    img_input.putpixel((x,y), (0, 0, 0))
        
        user_input_t = transform(user_input).unsqueeze(dim=0)
        product_input_t = transform(product_input).unsqueeze(dim=0)
        img_input_t = imageHD_transform(img_input).unsqueeze(dim=0)

        fig.add_subplot(2, 1, 2)
        plt.imshow(img_input)
        plt.title(model(img_input_t, user_input_t, product_input_t).item())

        fig.savefig(f'test_img/{i}.jpg')
    

if __name__ == '__main__':
    main()