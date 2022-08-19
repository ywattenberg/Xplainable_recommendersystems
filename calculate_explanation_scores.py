import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2

from explanation_generation.integrated_gradients import aggregate_attributions, get_IG_attributions

def main():
    df_test = pd.read_csv('data/compact_CSJ_imgHD_subset_test.csv')
    chosen = [0, 1, 15, 18, 20, 22, 23, 25, 28, 31, 34, 36, 39, 40, 44, 45, 46, 47, 52, 53, 69, 70, 72, 73, 74, 75, 76, 78, 80, 84, 86, 87, 88, 90, 91, 96, 97, 100, 111, 113, 116, 117, 118, 119, 120, 122, 125, 128, 130, 133, 134, 136, 137, 139, 145, 150, 153, 155, 167, 170, 198, 212, 214, 217, 218, 222, 227, 233, 245, 249, 252, 263, 267, 268, 275, 277, 279, 294, 296, 321, 322, 328, 337, 338, 341, 348, 379, 380, 386, 390, 405, 436, 440, 454, 458, 460, 466, 467, 469, 474, 478, 480, 483, 485, 489, 501, 517, 529, 557, 559, 562, 563, 570, 572, 576, 579, 586, 593, 625, 645, 649, 665, 701, 705, 720, 721, 722, 723, 725, 728, 740]

    reviews = pd.read_csv('data/reviews_examples.csv')
    users = set(reviews.reviewerID.unique())

    model = torch.load('/mnt/ds3lab-scratch/ywattenberg/models/mixer_14_10f_small_data.pth').to('cuda')
    model = model.module
    model.eval()

    df_test_filter = df_test[(df_test.reviewerID.isin(users))]

    annotations = pd.read_csv('', index_col='Unnamed: 0')

    total_attribution_inside = []

    for idx, id in enumerate(chosen):
        overall, reviewerID, asin, _, productID, userID, _ = df_test_filter.iloc[id]

        
        
        if idx not in annotation.index: 
            continue
        
        annotation = annotations.iloc[idx]

        image = cv2.imread('data/images/' + asin + '.jpg')
        attributions = get_IG_attributions(model, image, productID, userID, tmm_model=True, device='cuda')
        agg_attributions = aggregate_attributions(attributions)

        # Calculate total attributed
        tmp = np.array()
        if annotation.bbox_0 != '[0 0 0]':
            bbox = bbox_to_arr(annotation.bbox_0)
            tmp = calc_attributions(bbox, attributions)
        if annotation.bbox_1 != '[0 0 0]':
            bbox = bbox_to_arr(annotation.bbox_1)
            tmp = np.add(tmp, calc_attributions(bbox, attributions))
        total_attribution_inside.append(tmp)

        # Calculate top aggregated attributed
        num_rects = 8
        side_length = 28

        agg_attributions_df = pd.DataFrame(columns=['x','y','w', 'b', 'r'])
        for x in range(num_rects):
            for y in range(num_rects):
                tmp_w = agg_attributions[x*side_length,  y*side_length][0]
                tmp_b = agg_attributions[x*side_length,  y*side_length][1]
                tmp_r = agg_attributions[x*side_length,  y*side_length][2]
                tmp_df = pd.DataFrame([x, y, ])
                agg_attributions_df = pd.concat([agg_attributions_df, tmp_df, tmp_w, tmp_b, tmp_r], axis=1)

        agg_attributions_df.sort_values(by=['w'], ascending=False, inplace=True)


def get_top_n_score(agg_attributions_df, n, col, x, y, w, h):
    agg_attributions_df.sort_values(by=[col], ascending=False, inplace=True)
    tmp_df = agg_attributions_df.iloc[:n]
    return tmp_df.apply(lambda row: overlap(row.x, row.y, row.w, row.h, x, y, w, h), axis=1)



def overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    if x1 == x1+w1 or y1 == y1+h1 or x2 == x2+w2 or y2 == y2+h2:
        return False
     
    # If one rectangle is on left side of other
    if x1 > x2+w2 or x2 > x1+w1:
        return False
 
    # If one rectangle is above other
    if y1+h1 > y2 or y2+h2 > y1:
        return False
 
    return True



def bbox_to_arr(bbox):
    x, y, w, h = bbox[1:-1].split(' ')
    return [int(x), int(y), int(w), int(h)]

        
def calc_attributions(bbox, attributions):
    x, y, w, h = bbox
    att_w = 0
    att_b = 0
    att_r = 0

    for i in range(w):
        for j in range(h):
            att_w += attributions[x+i, y+j][0]
            att_b += attributions[x+i, y+j][1]
            att_r += attributions[x+i, y+j][2]
    return np.array((att_w, att_b, att_r))
        


        





