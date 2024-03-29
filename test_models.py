import numpy as np
import pandas as pd 
import torch
from torch.utils.data import DataLoader
from dataset.amazon_csj_dataset import AmazonCSJDatasetWithIMG, AmazonCSJDatasetWithIMGHD
from dataset.amazon_csj_dataset import AmazonCSJDataset
from model.SimpleMatrixFactorization import ModelMatrixFactorization
from model.MatrixFactorizationWithImages import MatrixFactorizationWithImages 

def test_loop_img(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (user_input, item_input, img_input, y) in enumerate(dataloader):
            pred = model(img_input, user_input, item_input)
            test_loss += loss_fn(pred, y).item()
            correct += (pred - y).abs().type(torch.float).sum().item()
            if batch % 100 == 0:
                print(f'[{batch * len(user_input):>5d}/{size:>5d}]')

    test_loss /= num_batches
    correct /= size
    print(f'Test Error \n Accuracy: {(100 * correct):>1f}%, Avg loss: {test_loss:>8f}')

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (user_input, item_input, y) in enumerate(dataloader):
            pred = model(user_input, item_input)
            test_loss += loss_fn(pred, y).item()
            correct += (pred - y).abs().type(torch.float).sum().item()
            if batch % 100 == 0:
                print(f'[{batch * len(user_input):>5d}/{size:>5d}]')


    test_loss /= num_batches
    correct /= size
    print(f'Test Error \n Accuracy: {(100 * correct):>1f}%, Avg loss: {test_loss:>8f}')

def test_simple(device):
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ.csv')
    test_data = df[df['rank_latest'] == 1]
    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()
    test_data = AmazonCSJDataset(path=None, df=test_data)
    test_dataloader = DataLoader(test_data, batch_size=1024, shuffle=True)
    model = ModelMatrixFactorization(num_items=num_items, num_users=num_users, n_factors=100).to(device=device)
    model.load_state_dict(torch.load('model_weights_simple.pth', map_location=device))
    test_loop(test_dataloader, model, torch.nn.MSELoss())

def test_img(device):
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_with_img_no_BW.csv')
    test_data_img = df[df['rank_latest'] == 1]
    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()
    test_data_img = AmazonCSJDatasetWithIMG(path=None, df=test_data_img)
    test_img_dataloader = DataLoader(test_data_img, batch_size=32, shuffle=True)
    model_img = MatrixFactorizationWithImages(num_items=num_items, num_users=num_users).to(device=device)
    model_img.load_state_dict(torch.load('model_weights_img.pth', map_location=device))
    test_loop_img(test_img_dataloader, model_img, torch.nn.MSELoss())

def test_imgHD(device):
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()
    test_data = df[df['rank_latest'] == 1]
    test_data = test_data.iloc[:100_000]
    test_data = AmazonCSJDatasetWithIMGHD(path=None, df=test_data)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True) 
    model = torch.nn.DataParallel(MatrixFactorizationWithImages(num_items=num_items, num_users=num_users).to(device=device))  
    model.load_state_dict(torch.load('model_weights_imgHD.pth', map_location=device))
    #model = torch.load('entire_model.pth')
    test_loop_img(test_dataloader, model, torch.nn.MSELoss())

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    test_imgHD(device)
    
if __name__ == '__main__':
    main()
