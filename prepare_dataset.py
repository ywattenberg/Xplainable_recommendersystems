import pandas as pd
import json
import numpy as np
    
def main():
    df = getDF('data/Clothing_Shoes_and_Jewelry_5.json')
    df = df[['overall', 'reviewerID', 'asin', 'unixReviewTime']]
    df, num_users, num_products, user_ids, product_ids = encode_df(df)
    df['rank_latest'] = df.groupby(['reviewerID'])['unixReviewTime'].rank(method='first', ascending=False)
    df.to_csv('data\\compact_CSJ.csv', index=False)

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
    product_ids, df['asin'], num_products = encode_cols(df['asin'])
    user_ids, df['reviewerID'], num_users = encode_cols(df['reviewerID'])
    return df, num_users, num_products, user_ids, product_ids

if __name__ == '__main__':
    main()
    
