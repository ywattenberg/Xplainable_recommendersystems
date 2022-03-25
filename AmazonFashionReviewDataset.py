from ast import parse
import json
from tkinter.messagebox import NO
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class AmazonFashionReviewDataset(Dataset):
    def __init__(self, file, df = None, transform=None, target_transform=None):
        if(file != None):
            self.rating = pd.read_csv(file)
        else:
            self.rating = df

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        row = self.rating.iloc[idx]
        print(int(hash(row.loc['reviewerID'])) )
        return self.transform(int(hash(row.loc['reviewerID']))) , self.transform(row.loc['asin']) , self.target_transform(float(row.loc['overall']))

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