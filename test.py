import numpy as np
from dataset.amazon_dataset_utils import parse
import json
import torch

if __name__ == '__main__':
    for i in range (0, 100):
        t = torch.rand([1,3,500,500], dtype=torch.float32, requires_grad=True)
        torch.save(t , f'IG_base_tensor/base_tensor_{i}.pt')
        
