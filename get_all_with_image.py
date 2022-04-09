import os
import pandas as pd
import numpy as np
from PIL import Image

def filter_b_and_w(path):
    df = pd.read_csv(path)
    product_IDs = df['asin']
    one_channel = []

    for asin in product_IDs:
        img = Image.open(os.path.join('data/images', asin))
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
    for file in os.listdir('./data/images'):
        split = file.split('.')
        if split[1] != 'jpg':
            not_jpg.append(file)
        have_img.add(split[0])

    with open('not_jpg.txt', 'a') as file_jpg:
        for file in not_jpg:
            file_jpg.write(file)
    return have_img

def main():

    bw_imgs = filter_b_and_w('./data/compact_CSJ.csv')
    df = df[~df['asin'].isin(bw_imgs)]
    df.to_csv('./data/compact_CSJ_with_img_no_BW.csv', index=False)


if __name__ == '__main__':
    main()
    #filter_b_and_w('data/compact_CSJ_with_img.csv')
    
    
    
