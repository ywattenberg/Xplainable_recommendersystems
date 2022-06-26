from torch.utils.data import Dataset
from dataset.amazon_dataset_utils import *
from PIL import Image
import numpy as np
import os


class AmazonCSJDataset(Dataset):
    def __init__(self, path, transform=transform, label_transform=label_transform, df=None):
        if(path != None):
            df = getDF(path)
            df = df[['overall', 'reviewerID', 'asin', 'unixReviewTime']]
            self.df, self.num_users, self.num_products, self.user_ids, self.product_ids = encode_df(df)
            self.df.to_csv('data/compact_CSJ_with_img.csv', index=False)
        else:
            self.df = df
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        userID = self.df.userID.iloc[index]
        productID = self.df.productID.iloc[index]
        label = self.df.overall.iloc[index]

        asin = self.df.asin.iloc[index]

        return self.transform(userID), self.transform(productID), self.label_transform(label)

class AmazonCSJDatasetWithIMG(Dataset):
    def __init__(self, path, transform=transform, label_transform=label_transform, image_transform=image_transform, df=None):
        if(path != None):
            df = getDF(path)
            df = df[['overall', 'reviewerID', 'asin', 'unixReviewTime']]
            self.df, self.num_users, self.num_products, self.user_ids, self.product_ids = encode_df(df)
            self.df.to_csv('data/compact_CSJ_with_img.csv', index=False)
        else:
            self.df = df
        self.transform = transform
        self.label_transform = label_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        userID = self.df.userID.iloc[index]
        productID = self.df.productID.iloc[index]
        label = self.df.overall.iloc[index]

        asin = self.df.asin.iloc[index]
        image = Image.open(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/images/', f'{asin}.jpg'))
        #image = np.transpose(image, (2,0,1))


        return self.transform(userID), self.transform(productID), self.image_transform(image), self.label_transform(label)

class AmazonCSJDatasetWithIMGHD(Dataset):
    def __init__(self, path, Atransform=transform, Alabel_transform=label_transform, Aimage_transform=imageHD_transform, df=None):
        if(path != None):
            df = getDF(path)
            df = df[['overall', 'reviewerID', 'asin', 'unixReviewTime']]
            self.df, self.num_users, self.num_products, self.user_ids, self.product_ids = encode_df(df)
            self.df.to_csv('data/compact_CSJ_with_img.csv', index=False)
        else:
            self.df = df
        self.transform = Atransform
        self.label_transform = Alabel_transform
        self.image_transform = Aimage_transform

        if Aimage_transform is None:
            self.image_transform = imageHD_transform
        if Atransform is None:
            self.transform = transform
        if Alabel_transform is None:
            self.label_transform = label_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        userID = self.df.userID.iloc[index]
        productID = self.df.productID.iloc[index]
        label = self.df.overall.iloc[index]

        asin = self.df.asin.iloc[index]
        image = Image.open(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', f'{asin}.jpg'))
        #image = np.transpose(image, (2,0,1))


        return self.image_transform(image).to('cuda'), self.transform(userID), self.transform(productID),  self.label_transform(label)
