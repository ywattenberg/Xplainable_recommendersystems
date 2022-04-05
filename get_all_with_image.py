import os
import pandas as pd

def main():
    have_img = set()
    not_jpg = list()
    for file in os.listdir('./data/images'):
        split = file.split('.')
        if split[1] != 'jpg':
            not_jpg.append(file)
        have_img.add(split[0])

    with open('not_jpg.txt', 'a') as not_jpg:
        for file in not_jpg:
            not_jpg.write(file)
    
    df = pd.read_csv('./data/compact_CSJ.csv')
    df[df.asin.isin(have_img)]
    print(len(df))
    df.to_csv('./data/compact_CSJ_with_img.csv', index=False)

    
    
    
