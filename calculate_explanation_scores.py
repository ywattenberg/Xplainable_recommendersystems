import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

from explanation_generation.integrated_gradients import aggregate_attributions, get_IG_attributions, attributions_w_b_r

def main():

    model = torch.load('/mnt/ds3lab-scratch/ywattenberg/models/entire_model_mixer_split.pth').to('cuda')
    model = model.module
    model.eval()

    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    annotations = pd.read_csv('annotations/annotations_1-65_Piri.csv', index_col='Unnamed: 0')

    total_attribution_inside = []
    total_score_w_b_r = [0.0,0.0,0.0]

    for index, row in annotations.iterrows():
        print(row.asin)
        reviewerID = row.reviewerID
        asin = row.asin
        
        userID = get_userID(reviewerID, df)
        productID = get_ProductID(asin, df)

        image = Image.open('/mnt/ds3lab-scratch/ywattenberg/data/images/' + asin + '.jpg')
        attributions = get_IG_attributions(model, image, productID, userID, tmm_model=True, device='cuda')
        agg_attributions = aggregate_attributions(attributions[0], attributions[1],  torch.mean(torch.stack(attributions[2:]), dim=0))

        # Calculate total attributed
        bboxes = []
        tmp = np.array([])
        if row.bbox_0 != '[0 0 0]':
            bbox = bbox_to_arr(row.bbox_0)
            tmp = calc_attributions(bbox, attributions)
            bboxes.append(bbox)
        if row.bbox_1 != '[0 0 0]':
            bbox = bbox_to_arr(row.bbox_1)
            tmp = np.add(tmp, calc_attributions(bbox, attributions))
            bboxes.append(bbox)
        total_attribution_inside.append(tmp)

        # Calculate top aggregated attributed
        num_rects = 8
        side_length = 28

        agg_attributions_df = pd.DataFrame(columns=['x','y','side_length','w', 'b', 'r'])
        for x in range(num_rects):
            for y in range(num_rects):
                tmp_w = agg_attributions[0][x*side_length,  y*side_length]
                tmp_b = agg_attributions[1][x*side_length,  y*side_length]
                tmp_r = agg_attributions[2][x*side_length,  y*side_length]
                #tmp_df = pd.DataFrame([x, y, side_length, tmp_w, tmp_b, tmp_r], columns=['x','y','side_length','w', 'b', 'r'])
                #agg_attributions_df = pd.concat([agg_attributions_df, tmp_df], axis=1)
                agg_attributions_df.loc[len(agg_attributions_df)] = [x*side_length, y*side_length, side_length, tmp_w, tmp_b, tmp_r]
        
        rects_in = set()
        tmp_score = 0.0
        for bbox in bboxes:
            score_df = get_top_n_score(agg_attributions_df, 10, 'w', bbox[0], bbox[1], bbox[2], bbox[3])
            score_df = score_df[~score_df.index.isin(rects_in)]
            tmp_score += len(score_df[score_df==True])
            rects_in.update(score_df[score_df==True].index.values)
        total_score_w_b_r[0] += tmp_score/10.0

        tmp_score = 0.0
        for bbox in bboxes:
            score_df = get_top_n_score(agg_attributions_df, 10, 'b', bbox[0], bbox[1], bbox[2], bbox[3]) # Number of rectangles that overlap with the bounding box
            score_df = score_df[~score_df.index.isin(rects_in)]
            tmp_score += len(score_df[score_df==True])
            rects_in.update(score_df[score_df==True].index.values)
        total_score_w_b_r[1] += tmp_score/10.0

        tmp_score = 0.0
        for bbox in bboxes:
            score_df = get_top_n_score(agg_attributions_df, 10, 'r', bbox[0], bbox[1], bbox[2], bbox[3])
            score_df = score_df[~score_df.index.isin(rects_in)]
            tmp_score += len(score_df[score_df==True])
            rects_in.update(score_df[score_df==True].index.values)
        total_score_w_b_r[2] += tmp_score/10.0
    print(total_score_w_b_r)
    




def get_userID(reviewerID, df):
    return df.loc[df.reviewerID == reviewerID].userID.values[0]

def get_ProductID(asin, df):
    return df.loc[df.asin == asin].productID.values[0]

def get_top_n_score(agg_attributions_df, n, col, x, y, w, h):
    agg_attributions_df.sort_values(by=[col], ascending=False, inplace=True)
    tmp_df = agg_attributions_df.iloc[:n]
    return tmp_df.apply(lambda row: overlap(row.x, row.y, row.side_length, row.side_length, x, y, w, h), axis=1)

def overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    if x1+w1 < x2 or x2+w2 < x1 or y1+h1 < y2 or y2+h2 < y1:
        return False
    return True

def bbox_to_arr(bbox):
    x, y, w, h = [str for str in bbox[1:-1].split(' ') if str]
    return [int(x), int(y), int(w), int(h)]

def calc_attributions(bbox, attributions):
    x, y, w, h = bbox
    attributions = attributions_w_b_r(attributions)
    att_w = 0
    att_b = 0
    att_r = 0
    for i in range(w):
        for j in range(h):
            att_w += attributions[0][ x+i, y+j]
            att_b += attributions[1][ x+i, y+j]
            att_r += attributions[2][ x+i, y+j]
    return np.array((att_w, att_b, att_r))
        

if __name__ == '__main__':
    main()
        





