import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.amazon_csj_dataset import AmazonCSJDatasetWithIMGHD
from model.MatrixFactorizationWithImagesImplicit import MatrixFactorizationWithImagesImplicit
from dataset.amazon_dataset_utils import *


def label_transform(z):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if z > 3:
        z = 1
    else:
        z = 0
    return torch.tensor(z, dtype=torch.float32).to(device)


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
        
        if batch % 1000 == 0:
             torch.save(model, 'tmp_entire_model_imp.pth')




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
    learning_rate = 0.01
    momentum = 0.9
    decay = 1e-8
    batch_size = 80 
    epochs = 4 

    #df = prepare_dataset('data/Clothing_Shoes_and_Jewelry_5.json')
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    #df = encode_df(df)
    #df['rank_latest'] = df.groupby(['reviewerID'])['unixReviewTime'].rank(method='first', ascending=False)
    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]
    #df.to_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv', index=False)

    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()

    train_data = AmazonCSJDatasetWithIMGHD(path=None, df=train_data, label_transform=label_transform)
    test_data = AmazonCSJDatasetWithIMGHD(path=None, df=test_data, label_transform=label_transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')

    model = MatrixFactorizationWithImagesImplicit(num_items=num_items, num_users=num_users, n_factors=100).to(device)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    try:
        model = torch.nn.DataParallel(model)
        #model.load_state_dict(torch.load('model_weights_imgHD.pth', map_location=device))
        #model = torch.load('entire_model.pth')
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")
        torch.save(model.state_dict(), 'model_weights_imgHD_imp.pth')
        torch.save(model, 'entire_model_imp.pth')
    except KeyboardInterrupt:
        print('Abort...')
        safe = input('Safe model [y]es/[n]o: ')
        if safe == 'y' or safe == 'Y':
            torch.save(model.state_dict(), 'model_weights_imgHD_imp.pth')
            torch.save(model, 'entire_model_imp.pth')
        else: 
            print('Not saving...')


if __name__ == '__main__':
    main()
