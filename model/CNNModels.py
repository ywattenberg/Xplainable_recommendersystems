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


class EfficentNetB4Model(torch.nn.Module):
    def __init__(self, num_of_latents) -> None:
        super().__init__()
        
        effb4 = models.efficientnet_b4(pretrained=True)

        for param in effb4.features.parameters():
            param.requires_grad = False

        num_features = effb4.classifier[-1].in_features
        effb4.eval()
        effb4.classifier = nn.Sequential(nn.Linear(num_features, num_of_latents))
        self.effb4 = effb4

    def forward(self, image):
        return self.effb4(image)

    