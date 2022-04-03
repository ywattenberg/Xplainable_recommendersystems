import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from dataset.amazon_dataset_utils import *

class AmazonCSJDatasetFM(Dataset):
    def __init__(self, path, transform_user, transfrom_product, label_transform, df=None):
        if(path != None):
            df = getDF(path)
            df = df[['overall', 'reviewerID', 'asin', 'unixReviewTime']]
            self.df, self.num_users, self.num_products, self.user_ids, self.product_ids = encode_df(df)
            self.df.to_csv('data/compact_CSJ.csv', index=False)
        else:
            self.df = df
        self.transform_user = transform_user
        self.transform_product = transfrom_product
        self.label_transform = label_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        userID = self.df.userID.iloc[index]
        productID = self.df.productID.iloc[index]
        label = self.df.overall.iloc[index]
        return torch.cat((self.transform_user(userID), self.transform_product(productID)), 0), self.label_transform(label)