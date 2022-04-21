from lib2to3.pgen2.pgen import DFAState
from operator import index
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.amazon_csj_dataset import AmazonCSJDatasetWithIMG
from model.SimpleMatrixFactorization import ModelMatrixFactorization
from model.MatrixFactorizationWithImages import MatrixFactorizationWithImages
from dataset.amazon_csj_dataset import AmazonCSJDataset
from dataset.amazon_dataset_utils import *


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (user_input, item_input, img_input, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(img_input, user_input, item_input)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(user_input)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for user_input, item_input, img_input, y in dataloader:
            pred = model(img_input, user_input, item_input)
            test_loss += loss_fn(pred, y).item()
            correct += (pred - y).abs().type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error \n Accuracy: {(100 * correct):>1f}%, Avg loss: {test_loss:>8f}')


def main():
    learning_rate = 0.1
    momentum = 0.9
    decay = 1e-8
    batch_size = 32
    epochs = 20

    #df = prepare_dataset('data/Clothing_Shoes_and_Jewelry_5.json')
    df = pd.read_csv('data/compact_CSJ_with_imgHD_no_BW.csv')
    df = encode_df(df)
    df['rank_latest'] = df.groupby(['reviewerID'])['unixReviewTime'].rank(method='first', ascending=False)
    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]
    df.to_csv('data/compact_CSJ_with_imgHD_no_BW.csv', index=False)

    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()

    train_data = AmazonCSJDatasetWithIMG(path=None, df=train_data)
    test_data = AmazonCSJDatasetWithIMG(path=None, df=test_data)

    print(len(train_data))
    print(len(test_data))

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')

    model = MatrixFactorizationWithImages(num_items=num_items, num_users=num_users).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    try:
        #model.load_state_dict(torch.load('model_weights.pth', map_location=device))
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")
        torch.save(model.state_dict(), 'model_weights_img.pth')
    except KeyboardInterrupt:
        print('Abort...')
        safe = input('Safe model [y]es/[n]o: ')
        if safe == 'y' or safe == 'Y':
            torch.save(model.state_dict(), 'model_weights_img.pth')
        else: 
            print('Not saving...')


if __name__ == '__main__':
    main()
