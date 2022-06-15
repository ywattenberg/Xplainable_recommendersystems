import numpy as np
import pandas as pd
import torch
import warnings
import time
from datetime import datetime
from torch.utils.data import DataLoader, random_split




class Trainer():
    def __init__(self, model, train_data, test_data, loss_fn, optimizer, split_test:float=None, device=None, batch_size=32, epochs=10, shuffle=True, name=None):
        if model == None or train_data == None:
            raise Exception("Model and train_data must be specified")

        if test_data == None:
            if split_test == None:
                raise Exception("Must specify either test_data or split_train")
            elif split_test != None:
                #warnings.warn("Test data is not specified, will split train data into train and validation")
                train_len = int(len(test_data*(1-split_test)))
                test_len = len(test_data) - train_len
                train_data, test_data = random_split(train_data, (train_len, test_len))

        if optimizer == None:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-8)
        else:    
            self.optimizer = optimizer

        if device == None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if name == None:
            self.name = datetime.now().strftime("%Y-%m-%d_%H")
        else:
            self.name = name

        self.batch_size = batch_size
        self.num_epochs = epochs
        self.model = model.to(self.device)
        self.loss_fn = loss_fn

        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=shuffle)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=shuffle)

    def train_loop(self):
        size = len(self.train_dataloader.dataset)
        time_at_start = time.time()*1000
        for batch, (*input, y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            pred = self.model(*input)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(input[0])
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
                run_time = time.time()*1000 - time_at_start
                print(f'time running: {run_time}, time per elem: {run_time/(current+1)}')
            
            if batch % 1000 == 0:
                print('saving model...')
                torch.save(self.model, 'tmp_entire_model_imp.pth')

    def test_loop(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch, (*input, y) in enumerate(self.test_dataloader):
                pred = self.model(*input)
                loss = self.loss_fn(pred, y)
                test_loss += loss.item()
                correct += (pred - y).abs().type(torch.float).sum().item()

                if batch % 100 == 0:
                    print(f'Current testloss: {test_loss / (batch+1):>8f}')

        test_loss /= num_batches
        correct /= size
        print(f'Test Error \n Accuracy: {(100 * correct):>1f}%, Avg loss: {test_loss:>8f}')


    def train_test(self):
        try:
            for t in range(self.num_epochs):
                    print(f"Epoch {t + 1}\n-------------------------------")
                    self.train_loop()
                    self.test_loop()
            print("Done!")
            torch.save(self.model.state_dict(), f'model_weights_{self.name}.pth')
            torch.save(self.model, f'entire_model_{self.name}.pth')
        except KeyboardInterrupt:
            print('Abort...')
            safe = input('Safe model [y]es/[n]o: ')
            if safe == 'y' or safe == 'Y':
                torch.save(self.model.state_dict(), f'model_weights_{self.name}.pth')
                torch.save(self.model, f'entire_model_{self.name}.pth')
            else: 
                print('Not saving...')
