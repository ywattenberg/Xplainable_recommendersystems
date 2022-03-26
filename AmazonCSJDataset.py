import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class AmazonCSJDataset(Dataset):
    def __init__(self, path, transform, label_transform, df=None):
        if(path != None):
            df = self.getDF(path)
            df = df[['overall', 'reviewerID', 'asin', 'unixReviewTime']]
            self.df, self.num_users, self.num_products, self.user_ids, self.product_ids = self.encode_df(df)
            self.df.to_csv('data\\compact_CSJ.csv', index=False)
        else:
            self.df = df
        self.transform = transform
        self.label_transform = label_transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        reviewer = self.df.reviewerID.iloc[index]
        asin = self.df.asin.iloc[index]
        label = self.df.overall.iloc[index]
        return self.transform(reviewer), self.transform(asin), self.label_transform(label)

    def parse(self, path):
        with open(path, 'rb') as file:
            for l in file:
                yield json.loads(l)

    def getDF(self, path):
        i = 0
        df = {}
        for d in self.parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def encode_cols(self, column):
        keys = column.unique()
        key_to_id = {key:idx for idx,key in enumerate(keys)}
        return key_to_id, np.array(key_to_id[x] for x in column), len(keys)

    def encode_df(self, df):
        product_ids, df['asin'], num_products = self.encode_cols(df['asin'])
        user_ids, df['reviewerID'], num_users = self.encode_cols(df['reviewerID'])
        return df, num_users, num_products, user_ids, product_ids