from statistics import mode
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

    user_input, product_input, img_input, rating = dataset[0]

    user_input = user_input.unsqueeze(dim=0)
    product_input = product_input.unsqueeze(dim=0)
    img_input = img_input.unsqueeze(dim=0)

    img_attr_b, delta_b = ig.attribute((img_input), baselines=(black_base_img), additional_forward_args=(user_input, product_input), n_steps=200, method='gausslegendre',return_convergence_delta=True)

    img_attr_w, delta_w = ig.attribute((img_input), baselines=(white_base_img), additional_forward_args=(user_input, product_input), n_steps=200, method='gausslegendre',return_convergence_delta=True)

    plot_attributions(img_input, black_base_img, img_attr_b).savefig('test_IG.png')
    



def plot_attributions(image, baseline, attribution_mask, alpha=0.4):
    image = image.squeeze().cpu().detach()
    baseline = baseline.squeeze().cpu().detach()
    attribution_mask = attribution_mask.squeeze().cpu().detach()
    
    fig = plt.figure(figsize=(10,10))

    fig.add_subplot(2, 2, 1)
    plt.imshow(baseline.permute(1, 2, 0))
    plt.title('Baseline')
    
    fig.add_subplot(2, 2, 2)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Image')

    fig.add_subplot(2, 2, 3)
    plt.imshow(attribution_mask.permute(1, 2, 0))
    plt.title('Attribution Mask')

    fig.add_subplot(2, 2, 4)
    plt.imshow(attribution_mask.permute(1, 2, 0))
    plt.imshow(image.permute(1, 2, 0), alpha=alpha)
    plt.title('Overlay')

    plt.tight_layout()
    return fig
if __name__ == '__main__':
    main()

