
from venv import create
import pandas as pd
import numpy as np
import torch

def main():
    df = pd.read_csv('data\\compat_CSJ.csv', low_memory=False)
    df, num_users, num_products, user_ids, product_ids = encode_df(df)
    print(f'Number of users {num_users}')
    print(f'Number of products {num_products}')
    print(df.head())


def encode_cols(column):
    keys = column.unique()
    key_to_id = {key:idx for idx,key in enumerate(keys)}
    return key_to_id, np.array(key_to_id[x] for x in column), len(keys)

def encode_df(df):
    product_ids, df['asin'], num_products = encode_cols(df['asin'])
    user_ids, df['reviewerID'], num_users = encode_cols(df['reviewerID'])
    return df, num_users, num_products, user_ids, product_ids

def predict(df, emb_user, emb_product):
    df['prediction'] = np.sum(np.multiply(emb_product[df['product_id']], emb_user[df['user_id']]), axis=1)
    return df

def cost(df, emb_user, emb_product):
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_product.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_product), emb_user.shape[0], emb_product.shape[0], 'prediction')
    return np.sum((Y-predicted).power(2))/df.shape[0]

def create_embeddings(n, K):
    return 11*np.random.random((n, K)) / K

def create_sparse_matrix(df, rows, cols, column_name='rating'):
    pass

if __name__ == '__main__':
    main()
