import pandas as pd
import numpy as np
from dataset.amazon_dataset_utils import *


df = getDF('/mnt/ds3lab-scratch/ywattenberg/data/Clothing_Shoes_and_Jewelry_5.json')
df = df[['overall', 'reviewerID', 'asin', 'reviewText']]

asins = set()
with open('/mnt/ds3lab-scratch/ywattenberg/data/asins.txt', 'r') as file:
    for line in file:
        asins.add(line.strip())

test_df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv')
test_users = set(test_df.reviewerID.unique())

df = df[(df['asin'].isin(asins))&(df['reviewerID'].isin(test_users))]
print(len(df))
df.to_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test_with_review.csv', index=False)
