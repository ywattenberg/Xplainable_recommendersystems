import torch
import torch.nn as nn
import numpy as np

from timm.models import mixer_l16_224_in21k 


def mixer_l16(num_classes=1000, **kwargs):
    mixer = mixer_l16_224_in21k(True)
    print(mixer)
    return mixer
    # for param in mixer.parameters():
    #     param.requires_grad = False
    
    # for param in mixer.fc.parameters():
    #     param.requires_grad = True