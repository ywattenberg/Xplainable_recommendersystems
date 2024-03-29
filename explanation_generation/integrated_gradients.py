import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from random import randint
from captum.attr import IntegratedGradients
from PIL import Image
import cv2
from torchvision import transforms as T

import sys
sys.path.append('../Xplainable_recommendersystems')

from explanation_generation.test_opencv import simple_filter
from dataset.amazon_dataset_utils import transform, imageHD_transform

def calculate_IG(model, image, baseline, user_in, product_in, image_transform=None, tmm_model=False, 
                steps:int=200, device=None, transform_baseline=False, transform_in=True):
    if image_transform is None:
        if not tmm_model:
            image_transform = imageHD_transform
        elif tmm_model:
            image_transform = T.Compose([T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=None), T.CenterCrop(size=(224, 224)), T.ToTensor(), T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))])
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if transform_in:
        user_in = transform(user_in)
        product_in = transform(product_in)
        user_in = user_in.to(device)
        user_in = user_in.unsqueeze(0)
        product_in = product_in.to(device)
        product_in = product_in.unsqueeze(0)

    if transform_baseline:
        baseline = image_transform(baseline)
        baseline = baseline.to(device)
        baseline = baseline.unsqueeze(0)

    image = image_transform(image)
    image = image.unsqueeze(dim=0)
    image = image.to(device)
    image = image.requires_grad_(True)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(image, baselines=baseline, additional_forward_args=(user_in,  
                                product_in), n_steps=steps, method='gausslegendre', internal_batch_size=32)
    
    return attributions

def get_IG_attributions(model, image, user_in, product_in, image_transform=None, tmm_model=False, 
                        device=None):
    white_base_img = Image.fromarray(np.ones([224,224,3], dtype=np.uint8))
    black_base_img = Image.fromarray(np.zeros([224,224,3], dtype=np.uint8))

    base_tensors = []
    to_image = T.ToPILImage()
    for i in range(15):
        base_tensors.append(to_image(torch.load(f'IG_base_tensor/base_tensor_{i}.pt').squeeze().to(device)))
    
    attributions = []
    attributions.append(calculate_IG(model, image, white_base_img, user_in, product_in, image_transform=image_transform,    
                                    tmm_model=tmm_model, device=device, transform_baseline=True))
    attributions.append(calculate_IG(model, image, black_base_img, user_in, product_in, image_transform=image_transform,    
                                    tmm_model=tmm_model, device=device, transform_baseline=True))

    for base_tensor in base_tensors:
        attributions.append(calculate_IG(model, image, base_tensor, user_in, product_in, image_transform=image_transform,    
                                    tmm_model=tmm_model, device=device, transform_baseline=True))
    return attributions


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]
    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()

    #train_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv') 

    model = torch.load('/home/ywattenberg/Xplainable_recommendersystems/tmp_entire_model_imp.pth').to(device)
    model = model.module
    print(model)

    #test_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv')
    image_transform = T.Compose([T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=None), T.CenterCrop(size=(224, 224)), T.ToTensor(), T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))])
    length = len(test_data)
    

    for i in range(20):
        while True:
            index = randint(0, length)
            user_input = test_data.iloc[index].userID
            product_input = test_data.iloc[index].productID
            img_input = Image.open(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', 
                                                f'{test_data.iloc[index].asin}.jpg'))
            tmp_img = cv2.imread(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', 
                                            f'{test_data.iloc[index].asin}.jpg'))
            rating = test_data.iloc[index].overall
            if rating > 3 and len(df[df['userID'] == user_input]) > 10 and not simple_filter(tmp_img):
                break
        attributions = get_IG_attributions(model, img_input, user_input, product_input, device=device, tmm_model=True)
        
        prediction = model(image_transform(img_input).unsqueeze(0).to('cuda'), transform(user_input).unsqueeze(0).to('cuda'), transform(product_input).unsqueeze(0).to('cuda'))
        fig = plot_attributions(image_transform(img_input), attributions, user_input, rating, prediction.item() , f'Plot {i}')
        fig.savefig(f'IG/{i}.png')
        plt.close(fig)
        print('done with IG')


def aggregate_attributions(attribution_mask_w, attribution_mask_b,  attribution_mask_rand):
    attribution_mask_b = attribution_mask_b.squeeze().cpu().detach().abs().sum(dim=0).numpy()
    attribution_mask_w = attribution_mask_w.squeeze().cpu().detach().abs().sum(dim=0).numpy()
    attribution_mask_rand = attribution_mask_rand.squeeze().cpu().detach().abs().sum(dim=0).numpy()

    agg_b = np.zeros(attribution_mask_b.shape, dtype=np.float32)
    agg_w = np.zeros(attribution_mask_w.shape, dtype=np.float32)
    agg_rand = np.zeros(attribution_mask_rand.shape, dtype=np.float32)

    side_length = 28
    num_of_quads = int(attribution_mask_b.shape[0]/side_length)

    for x in range(num_of_quads):
        for y in range(num_of_quads):
            tmp = np.sum(attribution_mask_b[x*side_length:  x*side_length + side_length,  y*side_length: y*side_length + side_length])
            agg_b[x*side_length:  x*side_length + side_length,  y*side_length: y*side_length + side_length] = tmp


    for x in range(num_of_quads):
        for y in range(num_of_quads):
            tmp = np.sum(attribution_mask_w[x*side_length:  x*side_length + side_length,  y*side_length: y*side_length + side_length])
            agg_w[x*side_length:  x*side_length + side_length,  y*side_length: y*side_length + side_length] = tmp


    for x in range(num_of_quads):
        for y in range(num_of_quads):
            tmp = np.sum(attribution_mask_rand[x*side_length:  x*side_length + side_length,  y*side_length: y*side_length + side_length])
            agg_rand[x*side_length:  x*side_length + side_length,  y*side_length: y*side_length + side_length] = tmp
    
    return agg_b, agg_w, agg_rand


def plot_attributions(image, attributions, user_input, rating, prediction,suptitle, alpha=0.3):
    image = image.squeeze().cpu().detach()

    attribution_mask_w, attribution_mask_b,  attribution_mask_rand = aggregate_attributions(attributions[0], attributions[1],  torch.mean(torch.stack(attributions[2:]), dim=0))
    
    fig = plt.figure(figsize=(10,15))

    fig.add_subplot(4, 2, 1)
    plt.imshow(np.zeros([500,500]))
    plt.title(f'User {user_input}, rated: {rating}, {prediction}')
    
    fig.add_subplot(4, 2, 2)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Image')

    fig.add_subplot(4, 2, 3)
    plt.imshow(attribution_mask_b)
    plt.title('Attribution Mask (Black)')

    fig.add_subplot(4, 2, 4)
    plt.imshow(attribution_mask_b)
    plt.imshow(image.permute(1, 2, 0), alpha=alpha)
    plt.title('Overlay (Black)')

    fig.add_subplot(4, 2, 5)
    plt.imshow(attribution_mask_w)
    plt.title('Attribution Mask (White)')

    fig.add_subplot(4, 2, 6)
    plt.imshow(attribution_mask_w)
    plt.imshow(image.permute(1, 2, 0), alpha=alpha)
    plt.title('Overlay (White)')

    fig.add_subplot(4, 2, 7)
    plt.imshow(attribution_mask_rand)
    plt.title('Attribution Mask (Random)')

    fig.add_subplot(4, 2, 8)
    plt.imshow(attribution_mask_rand)
    plt.imshow(image.permute(1, 2, 0), alpha=alpha)
    plt.title('Overlay (Random)')
    plt.tight_layout()

    fig.suptitle(suptitle)
    return fig

def attributions_w_b_r(attributions):
    attribution_mask_b = attributions[0].squeeze().cpu().detach().abs().sum(dim=0).numpy()
    attribution_mask_w = attributions[1].squeeze().cpu().detach().abs().sum(dim=0).numpy()
    attribution_mask_rand = torch.mean(torch.stack(attributions[2:]), dim=0).squeeze().cpu().detach().abs().sum(dim=0).numpy()
    return attribution_mask_w, attribution_mask_b, attribution_mask_rand

if __name__ == '__main__':
    main()

