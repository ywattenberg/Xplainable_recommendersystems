import torch
import torch.nn as nn
import numpy as np

from timm.models import mixer_l16_224_in21k 


def mixer_l16(num_classes=1000, **kwargs):
    mixer = mixer_l16_224_in21k(True)
    for param in mixer.parameters():
        param.requires_grad = False
    
    mixer.head = nn.Sequential(nn.Linear(mixer.head.in_features, num_classes))

    for param in mixer.head.parameters():
         param.requires_grad = True
    
    return mixer