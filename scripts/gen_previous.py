import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os

from dataset.amazon_dataset_utils import *

def main():
    test_df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv')
    train_df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv')
    test_users = set(test_df.reviewerID.unique())
    train_products = set(train_df.asin.unique())
    test_user_idx = dict(zip(test_users, np.zeros(len(test_users), dtype=int)))

    reviews = pd.DataFrame(columns=['reviewerID', 'asin', 'overall', 'reviewText'])
    
    asins = set()
    with open('/mnt/ds3lab-scratch/ywattenberg/data/asins.txt', 'r') as file:
        for line in file:
            asins.add(line.strip())

    selected_users = test_df.loc[(test_df['asin'].isin(asins))&(test_df['reviewerID'].isin(test_users))].reviewerID.unique()
    np.random.shuffle(selected_users)
    selected_users = set(selected_users[:1200])

    i = 0
    for line in parse('/mnt/ds3lab-scratch/ywattenberg/data/Clothing_Shoes_and_Jewelry_5.json'):
        if line['asin'] in train_products and line['reviewerID'] in selected_users and test_user_idx[line['reviewerID']] < 10:
            if 'reviewText' in line.keys():
                i += 1
                reviews.loc[i] = [line['reviewerID'], line['asin'], line['overall'], line['reviewText']]
                test_user_idx[line['reviewerID']] += 1
                print(i)
        if i > 10000:
            break
    reviews.to_csv('/mnt/ds3lab-scratch/ywattenberg/data/reviews_examples.csv', index=False)

            
if __name__ == '__main__':
    main()
