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
    print(len(one_channel))

def main():
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
    
    df = pd.read_csv('./data/compact_CSJ.csv')
    df = df[df['asin'].isin(have_img)]
    print(len(have_img))
    print(df.asin.nunique())
    df.to_csv('./data/compact_CSJ_with_img.csv', index=False)


if __name__ == '__main__':
    #main()
    filter_b_and_w('data/compact_CSJ_with_img.csv')
    
    
    
