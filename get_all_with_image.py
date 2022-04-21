import os
import pandas as pd
import numpy as np
from PIL import Image

def filter_b_and_w(path):
    df = pd.read_csv(path)
    product_IDs = df['asin']
    one_channel = []

    for asin in product_IDs:
        img = Image.open(os.path.join('data/images', f'{asin}.jpg'))
        if img.mode == 'L':
            print(np.shape(img))
            one_channel.append(asin)

    with open('BW_img.txt', 'a') as file:
        for asin in one_channel:
            file.write(f'{asin}\n')     

    return one_channel  

def filter_img():
    have_img = set()
    not_jpg = list()
    for file in os.listdir('./data/imagesHD'):
        split = file.split('.')
        if split[1] != 'jpg':
            not_jpg.append(file)
        have_img.add(split[0])

    with open('have_img.txt', 'a') as file:
        for element in have_img:
            file.write(element)
            file.write('\n')
    return have_img

def main():
    #filter_img()
    # bw_imgs = filter_b_and_w('data/compact_CSJ_with_img.csv')
    # df = df[~df['asin'].isin(bw_imgs)]
    # df.to_csv('./data/compact_CSJ_with_img_no_BW.csv', index=False)
    bw_images = set()
    with open('BW_img.txt', 'r') as file:
        for l in file:
            bw_images.add(l)

    img = []
    with open('have_img.txt', 'r') as file:
        for l in file:
            if l not in bw_images:
                img.append(l)
    
    #os.remove('have_img.txt')
    with open('have_img_no_bw.txt', 'a') as file:
        for element in img:
            file.write(element)



    
    
