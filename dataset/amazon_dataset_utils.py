import json
import torch
import pandas as pd 
import numpy as np
from torchvision import transforms


def parse(path):
    with open(path, 'rb') as file:
        for l in file:
            yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def encode_cols(column):
    keys = column.unique()
    key_to_id = {key:idx for idx,key in enumerate(keys)}
    return key_to_id, np.array(key_to_id[x] for x in column), len(keys)

def encode_df(df):
    product_ids, df['productID'], num_products = encode_cols(df['asin'])
    user_ids, df['userID'], num_users = encode_cols(df['reviewerID'])
    return df

def prepare_dataset(path):
    df = getDF(path)
    df = df[['overall', 'reviewerID', 'asin', 'unixReviewTime']]
    df = encode_df(df)
    df['rank_latest'] = df.groupby(['reviewerID'])['unixReviewTime'].rank(method='first', ascending=False)
    df.to_csv('data/compact_CSJ.csv', index=False)
    return df

def label_transform(z):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.tensor(z, dtype=torch.float32).to(device)

def transform(z):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tmp = torch.tensor(z).to(device)
    #tmp.requires_grad_()
    return tmp

def image_transform(img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.Resize(500), transforms.CenterCrop(500), transforms.ToTensor()])
    return transform(img).to(device)
