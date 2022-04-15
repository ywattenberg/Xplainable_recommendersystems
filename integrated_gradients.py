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

    white_base_img = torch.ones([3,50,50], dtype=torch.float32, requires_grad=True)
    black_base_img = torch.zeros([3,50,50], dtype=torch.float32, requires_grad=True)

    ig = IntegratedGradients(model)

    dataset = AmazonCSJDatasetWithIMG(path=None, df=test_data)

    user_input, product_input, img_input, rating = dataset[0]

    (user_attr, product_attr, img_attr), delta = ig.attribute((user_input, product_input, img_input), baselines=(user_input, product_input, black_base_img), n_steps=100, return_convergence_delta=True)
    print(img_attr)

def plot_attributions(image, baseline, attribution_mask):
    # convert attribution array into displayable image
    

    # display images
    fig = plt.figure(figsize=(10,10))

    fig.add_subplot(2, 2, 1)
    plt.imshow(baseline)
    plt.axes('off')
    plt.title('Baseline')
    
    fig.add_subplot(2, 2, 2)
    plt.imshow(image)
    plt.axes('off')
    plt.title('Image')

    fig.add_subplot(2, 2, 3)
    plt.imshow(attribution_mask)
    plt.axes('off')
    plt.title('Attribution Mask')

    fig.add_subplot(2, 2, 4)
    plt.imshow()
    plt.axes('off')
    plt.title('Baseline')
if __name__ == '__main__':
    main()

