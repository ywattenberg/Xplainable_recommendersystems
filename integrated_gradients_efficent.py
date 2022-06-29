import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
from random import randint
from captum.attr import IntegratedGradients
from PIL import Image
from torchvision import transforms as T
from test_opencv import simple_filter
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from dataset.amazon_dataset_utils import transform, imageHD_transform

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    #num_users = df['reviewerID'].nunique()
    #num_items = df['asin'].nunique()
    
    model = torch.load('/mnt/ds3lab-scratch/ywattenberg/models/efficent_b4_10f_small_data.pth').to(device)

    model = model.module
    print(model)
    #model = torch.load('entire_model.pth')
    #model = model.module
    #model = MatrixFactorizationWithImages(num_items=num_items, num_users=num_users).to(device)
    #model.load_state_dict(torch.load('model_weights_imgHD.pth', map_location=device).module.state_dict())
    
    #train_data = df[df['rank_latest'] != 1]
    #test_data = df[df['rank_latest'] == 1]
    train_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_train.csv') 
    test_data = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD_subset_test.csv')
    image_transform = imageHD_transform #create_transform(**resolve_data_config({}, model=model))


    # white_base_img = image_transform(torch.ones([3,224,224], dtype=torch.float32, requires_grad=True)).to(device)
    # black_base_img = image_transform(torch.zeros([3,224,224], dtype=torch.float32, requires_grad=True)).to(device)
    white_base_img = image_transform(Image.fromarray(np.ones([500,500,3], dtype=np.uint8))).to(device).unsqueeze(0)
    black_base_img = image_transform(Image.fromarray(np.zeros([500,500,3], dtype=np.uint8))).to(device).unsqueeze(0)


    ig = IntegratedGradients(model)

    #dataset = AmazonCSJDatasetWithIMGHD(path=None, df=test_data)
    length = len(test_data)

    base_tensors = []
    for i in range(15):
        base_tensors.append(torch.load(f'IG_base_tensor/base_tensor_{i}.pt').to(device))

    for i in range(20):
        print('start')
        while True:
            index = randint(0, length)
            user_input = test_data.iloc[index].userID
            product_input = test_data.iloc[index].productID
            img_input = Image.open(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', f'{test_data.iloc[index].asin}.jpg'))
            tmp_img = cv2.imread(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', f'{test_data.iloc[index].asin}.jpg'))
            rating = test_data.iloc[index].overall
            if rating > 3 and len(df[df['userID'] == user_input]) > 10 and not simple_filter(tmp_img):
                break
        
        print('got ex')
        
        user_input_t = transform(user_input).unsqueeze(dim=0).to(device)
        product_input_t = transform(product_input).unsqueeze(dim=0).to(device)
        img_input_t = image_transform(img_input).unsqueeze(dim=0).to(device)

        img_attr_b, delta_b = ig.attribute((img_input_t), baselines=(black_base_img), additional_forward_args=(user_input_t, product_input_t), 
                                                n_steps=100, method='gausslegendre', return_convergence_delta=True, internal_batch_size=8)
        img_attr_w, delta_w = ig.attribute((img_input_t), baselines=(white_base_img), additional_forward_args=(user_input_t, product_input_t), 
                                                n_steps=100, method='gausslegendre', return_convergence_delta=True, internal_batch_size=8)

        img_attr_rand = []
        for tensor in base_tensors:
            tensor = T.ToPILImage()(tensor.squeeze(dim=0))
            img_attr_rand.append(ig.attribute((img_input_t), baselines=image_transform(tensor).to(device).unsqueeze(0), additional_forward_args=(user_input_t, product_input_t), 
                                                n_steps=100, method='gausslegendre', internal_batch_size=8))

        prediction = model(img_input_t, user_input_t, product_input_t)
        img_attr_avg = torch.mean(torch.stack(img_attr_rand), dim=0)
        fig = plot_attributions(img_input_t, img_attr_b, img_attr_w, img_attr_avg, user_input, rating, prediction.item() , f'Plot {i}')
        fig.savefig(f'IG/{i}.png')
        plt.close(fig)
        print('done with IG')
        prev_liked = df[df['userID'] == user_input]
        print(len(prev_liked))
        j = 0
        fig = plt.figure(figsize=(10,15))
        for i, line in prev_liked.iterrows():
            if j >= 6:
                break
            if line.overall > 3 and line.asin != test_data.iloc[index].asin:
                j += 1
                print('in')
                image = Image.open(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', f'{line.asin}.jpg'))
                fig.add_subplot(3, 2, j)
                #plt.plot([1,2,3,4,10], [54,56,8,84,54])
                plt.imshow(image)
                plt.title(f'rating {line.overall}')
        plt.tight_layout()        
        fig.savefig(f'IG/{i}_e.png')
        plt.close(fig)



def plot_attributions(image, attribution_mask_b, attribution_mask_w,  attribution_mask_rand, user_input, rating, prediction,suptitle, alpha=0.3):
    image = image.squeeze().cpu().detach()

    attribution_mask_b = attribution_mask_b.squeeze().cpu().detach().abs().sum(dim=0)
    attribution_mask_w = attribution_mask_w.squeeze().cpu().detach().abs().sum(dim=0)
    attribution_mask_rand = attribution_mask_rand.squeeze().cpu().detach().abs().sum(dim=0)

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

def calc_IG(model, base_img, product_img, user_in, product_in, steps):
    alphas = torch.linspace(start=0.0, end=1.0, steps=steps+1)

def interpolate_img(baseline, image, alphas):
    alphas_x = alphas[:, None, None, None]
    baseline_x = baseline.unsqueeze(dim=0)
    image_x = image.unsqueeze(dim=0)
    delta = image_x - baseline_x
    return baseline_x + alphas_x * delta

if __name__ == '__main__':
    main()

