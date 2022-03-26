from webbrowser import get
import pandas as pd
import json
from AmazonCSJDataset import AmazonCSJDataset
    
def main():
    ds = AmazonCSJDataset('data\\Clothing_Shoes_and_Jewelry_5.json')
    #df = pd.read_csv('data\\simple_cols.csv', low_memory=False)
    #df = df[['overall', 'reviewerID', 'asin', 'unixReviewTime', 'latest_rating']]
    #df.to_csv('compact_CSJ.csv')

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

if __name__ == '__main__':
    main()
    
