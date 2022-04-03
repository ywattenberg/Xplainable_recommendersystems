from numpy import dtype
import torch
from dataset.amazon_csj_dataset import AmazonCSJDataset
from model.FactorizationMachines import FactorizationMachineModel
from dataset.amazon_csj_dataset import AmazonCSJDataset

from torch.utils.data import DataLoader
import pandas as pd
from dataset.amazon_dataset_utils import prepare_dataset


def label_transform(z):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.tensor(z, dtype=torch.float32).to(device)


def transform(z):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tmp = torch.tensor(z).to(device)
    #tmp.requires_grad_()
    return tmp


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for user_input, item_input, y in dataloader:
            pred = model(user_input, item_input)
            test_loss += loss_fn(pred, y).item()
            correct += (pred - y).abs().type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error \n Accuracy: {(100 * correct):>1f}%, Avg loss: {test_loss:>8f}')


def main():
    learning_rate = 0.1
    momentum = 0.9
    decay = 1e-8
    batch_size = 256
    epochs = 20

    #df = prepare_dataset('data/Clothing_Shoes_and_Jewelry_5.json')
    df = pd.read_csv('data/compact_CSJ.csv')
    # df['rank_latest'] = df.groupby(['reviewerID'])['unixReviewTime'].rank(method='first', ascending=False)
    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]

    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()
    
    user_trans = lambda t :(torch.nn.functional.one_hot(transform(t), num_users)).float()
    item_trans = lambda t : (torch.nn.functional.one_hot(transform(t), num_items)).float()

    train_data = AmazonCSJDatasetFM(path=None, df=train_data, transform_user=user_trans, transfrom_product=item_trans, label_transform=label_transform)
    test_data = AmazonCSJDatasetFM(path=None, df=test_data, transform_user=transform, transfrom_product=transform, label_transform=label_transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')

    model = FactorizationMachineModel([num_items, num_users], embedding_dim=20).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    try:
        #model.load_state_dict(torch.load('model_weights.pth', map_location=device))
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")
        torch.save(model.state_dict(), 'model_weights.pth')
    except KeyboardInterrupt:
        print('Abort...')
        safe = input('Safe model [y]es/[n]o: ')
        if safe == 'y' or safe == 'Y':
            torch.save(model.state_dict(), 'model_weights.pth')
        else: 
            print('Not saving...')


if __name__ == '__main__':
    main()
