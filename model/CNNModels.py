import torch
import torch.nn as nn
import numpy as np
from torchvision import models

class vgg16Model(torch.nn.Module):
    def __init__(self, num_of_latents) -> None:
        super().__init__()
        
        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.features.parameters():
            param.requires_grad = False

        num_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, num_of_latents)])
        vgg16.classifier = nn.Sequential(*features)
        self.vgg16 = vgg16

    def forward(self, image):
        return self.vgg16(image)

    
    