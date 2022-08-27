from torchvision import models, transforms
from PIL import Image
from captum.attr import IntegratedGradients
import torch
import numpy as np

def main():
    device = 'cuda'
    model = models.efficientnet_b3(pretrained=True)
    model.to(device).eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = Image.open('cat.jpg')
    image = transform(image)
    image = image.to(device)
    baseline = torch.zeros(image.shape).to(device).unsqueeze(dim=0)
    print(baseline.shape)
    attr = IntegratedGradients(model).attribute(image, baselines=baseline, internal_batch_size=1)

    #print(output)
    #ans = output.squeeze().detach().numpy().tolist()
    #print(ans.index(max(ans)))
    to_img = transforms.ToPILImage()
    to_img(attr.to('cpu')).save('C:/Users/User/Downloads/car_attr.jpg')



if __name__ == '__main__':
    main()
