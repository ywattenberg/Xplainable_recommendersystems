import os
import pandas as pd
import numpy as np
from PIL import Image

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

def main():
    filter_img('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD')

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

    
    
