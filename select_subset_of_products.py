import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    #df = pd.read_csv('data/compact_CSJ_imgHD_subset.csv')
    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]
    print(df['asin'].nunique())
    print(len(train_data))
    print(len(test_data))

    df_group = train_data.groupby(['asin'])

    group_size = df_group.size().sort_values(ascending=False)
    top_asin = group_size.iloc[:3_000]

    train_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    train_data = train_data[train_data['asin'].isin(top_asin.keys())]
    train_users = train_data['reviewerID']

    test_data = test_data[test_data['reviewerID'].isin(train_users)]
    test_data = test_data[test_data['asin'].isin(top_asin.keys())]
    
    print(len(train_data))
    print(len(test_data))
    train_data.to_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv', index=False)
    test_data.to_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv', index=False)





if __name__ == '__main__':
    main()
