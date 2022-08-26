import os
import pandas as pd
import numpy as np
from PIL import Image

import sys
sys.path.append('../.')
from dataset.amazon_dataset_utils import prepare_dataset 



def filter_img(path):
    have_img = list()
    not_jpg = list()
    for file in os.listdir(path):
        split = file.split('.')
        if split[1] != 'jpg':
            not_jpg.append(file)
        try:
            img = Image.open(os.path.join(path, file))
            if img.mode != 'L':
                have_img.append(split[0])
        except:
            print(f'Error on img: {file}')

    with open('not_jpg.txt', 'a') as file:
        for element in not_jpg:
            file.write(element)
            file.write('\n')

    with open('have_img.txt', 'a') as file:
        for element in have_img:
            file.write(element)
            file.write('\n')
    return have_img

def create_csv(df, path, have_img):
    if have_img == None:
        have_img = set()
        with open('have_img.txt', 'r') as file:
            for line in file:
                have_img.add(line.replace('\n', ''))
    df = df[df['asin'].isin(have_img)]
    df.to_csv(path)

def main():
    df = prepare_dataset('/mnt/ds3lab-scratch/ywattenberg/data/AMAZON_FASHION.json') 
    have = filter_img('/mnt/ds3lab-scratch/ywattenberg/data/fashion_imagesHD')
    #df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_fashion.csv')
    create_csv(df, '/mnt/ds3lab-scratch/ywattenberg/data/compact_fashion_ImgHD.csv', have)
    # bw_images = set()
    # with open('BW_img.txt', 'r') as file:
    #     for l in file:
    #         bw_images.add(l)

    # img = []
    # with open('have_img.txt', 'r') as file:
    #     for l in file:
    #         if l not in bw_images:
    #             img.append(l)
    
    # #os.remove('have_img.txt')
    # with open('have_img_no_bw.txt', 'a') as file:
    #     for element in img:
    #         file.write(element)

if __name__ == '__main__':
    main()

    
    
