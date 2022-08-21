import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2

from explanation_generation.integrated_gradients import aggregate_attributions, get_IG_attributions

def main():

    model = torch.load('/mnt/ds3lab-scratch/ywattenberg/models/entire_model_mixer_split.pth').to('cuda')
    model = model.module
    model.eval()

    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    annotations = pd.read_csv('annoatations/annotations_1-65_Piri.csv', index_col='Unnamed: 0')

    total_attribution_inside = []

    for row in annotations.iterrows():
        reviewerID = row['reviewerID'].reviewerID
        asin = row['asin'].asin
        
        userID = get_userID(reviewerID, df)
        productID = get_ProductID(asin, df)

        image = cv2.imread('data/images/' + asin + '.jpg')
        attributions = get_IG_attributions(model, image, productID, userID, tmm_model=True, device='cuda')
        agg_attributions = aggregate_attributions(attributions)

        # Calculate total attributed
        bboxes = []
        tmp = np.array()
        if row['bbox_0'] != '[0 0 0]':
            bbox = bbox_to_arr(row['bbox_0'])
            tmp = calc_attributions(bbox, attributions)
            bboxes.append(bbox)
        if row['bbox_1'] != '[0 0 0]':
            bbox = bbox_to_arr(row['bbox_1'])
            tmp = np.add(tmp, calc_attributions(bbox, attributions))
            bboxes.append(bbox)
        total_attribution_inside.append(tmp)

        # Calculate top aggregated attributed
        num_rects = 8
        side_length = 28

        agg_attributions_df = pd.DataFrame(columns=['x','y','side_length','w', 'b', 'r'])
        for x in range(num_rects):
            for y in range(num_rects):
                tmp_w = agg_attributions[x*side_length,  y*side_length][0]
                tmp_b = agg_attributions[x*side_length,  y*side_length][1]
                tmp_r = agg_attributions[x*side_length,  y*side_length][2]
                tmp_df = pd.DataFrame([x, y, side_length, tmp_w, tmp_b, tmp_r], columns=['x','y','side_length','w', 'b', 'r'])
                agg_attributions_df = pd.concat([agg_attributions_df, tmp_df], axis=1)
        
        print(agg_attributions_df)
        for bbox in bboxes:
            print(get_top_n_score(agg_attributions_df, 5, 'w', bbox[0], bbox[1], bbox[2], bbox[3]))


def get_userID(reviewerID, df):
    return df.loc[df.reviewerID == reviewerID].userID.values[0]

def get_ProductID(asin, df):
    return df.loc[df.asin == asin].productID.values[0]

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
        


        





