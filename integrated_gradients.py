from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients
from model.MatrixFactorizationWithImages import MatrixFactorizationWithImages
from dataset.amazon_csj_dataset import AmazonCSJDatasetWithIMG 




def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv('data/compact_CSJ_with_img_no_BW.csv')
    num_users = df['reviewerID'].nunique()
    num_items = df['asin'].nunique()

    model = MatrixFactorizationWithImages(num_items=num_items, num_users=num_users).to(device)
    model.load_state_dict(torch.load('model_weights_img.pth', map_location=device))

    
    test_data = df[df['rank_latest'] == 1]

    white_base_img = torch.ones([1,3,50,50], dtype=torch.float32, requires_grad=True).to(device)
    black_base_img = torch.zeros([1,3,50,50], dtype=torch.float32, requires_grad=True).to(device)

    ig = IntegratedGradients(model)

    dataset = AmazonCSJDatasetWithIMG(path=None, df=test_data)
    length = len(dataset)

    for i in range(20):
        user_input, product_input, img_input, rating = dataset[randint(0, length)]

        user_input = user_input.unsqueeze(dim=0)
        product_input = product_input.unsqueeze(dim=0)
        img_input = img_input.unsqueeze(dim=0)

        img_attr_b, delta_b = ig.attribute((img_input), baselines=(black_base_img), additional_forward_args=(user_input, product_input), n_steps=200, method='gausslegendre',return_convergence_delta=True)

        img_attr_w, delta_w = ig.attribute((img_input), baselines=(white_base_img), additional_forward_args=(user_input, product_input), n_steps=200, method='gausslegendre',return_convergence_delta=True)

        plot_attributions(img_input, img_attr_b, img_attr_w, f'Plot {i}').savefig(f'IG/{i}.png')
    



def plot_attributions(image, attribution_mask_b, attribution_mask_w,  suptitle, alpha=0.4):
    image = image.squeeze().cpu().detach()
    attribution_mask_b = attribution_mask_b.squeeze().cpu().detach().abs().sum(dim=0)
    attribution_mask_w = attribution_mask_w.squeeze().cpu().detach().abs().sum(dim=0)
    
    fig = plt.figure(figsize=(10,15))

    fig.add_subplot(3, 2, 1)
    plt.imshow(np.zeros([50,50]))
    plt.title('Empty')
    
    fig.add_subplot(3, 2, 2)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Image')

    fig.add_subplot(3, 2, 3)
    plt.imshow(attribution_mask_b.permute(1, 2, 0))
    plt.title('Attribution Mask (Black)')

    fig.add_subplot(3, 2, 4)
    plt.imshow(attribution_mask_b.permute(1, 2, 0))
    plt.imshow(image.permute(1, 2, 0), alpha=alpha)
    plt.title('Overlay (Black)')

    fig.add_subplot(3, 2, 5)
    plt.imshow(attribution_mask_w.permute(1, 2, 0))
    plt.title('Attribution Mask (White)')

    fig.add_subplot(3, 2, 6)
    plt.imshow(attribution_mask_w.permute(1, 2, 0))
    plt.imshow(image.permute(1, 2, 0), alpha=alpha)
    plt.title('Overlay (White)')

    plt.tight_layout()

    fig.suptitle(suptitle)
    return fig

if __name__ == '__main__':
    main()

