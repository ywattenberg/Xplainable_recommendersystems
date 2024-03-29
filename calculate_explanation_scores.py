import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

from explanation_generation.integrated_gradients import aggregate_attributions, get_IG_attributions, attributions_w_b_r
from explanation_generation.augmented_images import gen_explanation, colour_change

def main():
    
    #model = torch.load('entire_model_2022-08-28_17.pth').to('cuda')
    model = torch.load('/mnt/ds3lab-scratch/ywattenberg/models/entire_model_vgg_add.pth').to('cuda')
    model = model.module
    model.eval()

    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    annotations = pd.read_csv('annotations/annotations_1.csv', index_col='Unnamed: 0')
    total_attribution_inside = [0.0,0.0,0.0]
    total_score_w_b_r = [0.0,0.0,0.0]
    total_score_counter = [0,0]

    for index, row in annotations.iterrows():
        print(index)
        reviewerID = row.reviewerID
        asin = row.asin

        userID = get_userID(reviewerID, df)
        productID = get_ProductID(asin, df)
        image = Image.open('/mnt/ds3lab-scratch/ywattenberg/data/images/' + asin + '.jpg')
        attributions = get_IG_attributions(model, image, userID, productID, tmm_model=True, device='cuda')
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
                

        total_attribution_inside[0] += tmp[0]/np.sum(agg_attributions_df.w.values)
        total_attribution_inside[1] += tmp[1]/np.sum(agg_attributions_df.b.values)
        total_attribution_inside[2] += tmp[2]/np.sum(agg_attributions_df.r.values)

        rects_in = set()
        tmp_score = 0.0
        top_k = 5
        for bbox in bboxes:
            score_df = get_top_n_score(agg_attributions_df, top_k, 'w', bbox[0], bbox[1], bbox[2], bbox[3])
            score_df = score_df[~score_df.index.isin(rects_in)]
            tmp_score += len(score_df[score_df==True])
            rects_in.update(score_df[score_df==True].index.values)
        total_score_w_b_r[0] += tmp_score

        rects_in = set()
        tmp_score = 0.0
        for bbox in bboxes:
            score_df = get_top_n_score(agg_attributions_df, top_k, 'b', bbox[0], bbox[1], bbox[2], bbox[3]) # Number of rectangles that overlap with the bounding box
            score_df = score_df[~score_df.index.isin(rects_in)]
            tmp_score += len(score_df[score_df==True])
            rects_in.update(score_df[score_df==True].index.values)
        total_score_w_b_r[1] += tmp_score

        rects_in = set()
        tmp_score = 0.0
        for bbox in bboxes:
            score_df = get_top_n_score(agg_attributions_df, top_k, 'r', bbox[0], bbox[1], bbox[2], bbox[3])
            score_df = score_df[~score_df.index.isin(rects_in)]
            tmp_score += len(score_df[score_df==True])
            rects_in.update(score_df[score_df==True].index.values)
        total_score_w_b_r[2] += tmp_score

        attributions = gen_explanation(model, image, userID, productID, tmm_model=True)
        agg_attributions_df = pd.DataFrame(columns=['x','y','side_length','w', 'm'])
        for x in range(num_rects):
            for y in range(num_rects):
                tmp_w = attributions[0][x*side_length,  y*side_length]
                tmp_b = attributions[1][x*side_length,  y*side_length]
                #tmp_df = pd.DataFrame([x, y, side_length, tmp_w, tmp_b, tmp_r], columns=['x','y','side_length','w', 'b', 'r'])
                #agg_attributions_df = pd.concat([agg_attributions_df, tmp_df], axis=1)
                agg_attributions_df.loc[len(agg_attributions_df)] = [x*side_length, y*side_length, side_length, tmp_w, tmp_b]
                rects_in = set()
        tmp_score = 0.0

        for bbox in bboxes:
            score_df = get_top_n_score(agg_attributions_df, top_k, 'w', bbox[0], bbox[1], bbox[2], bbox[3])
            score_df = score_df[~score_df.index.isin(rects_in)]
            tmp_score += len(score_df[score_df==True])
            rects_in.update(score_df[score_df==True].index.values)
        total_score_counter[0] += tmp_score

        rects_in = set()
        tmp_score = 0.0
        for bbox in bboxes:
            score_df = get_top_n_score(agg_attributions_df, top_k, 'm', bbox[0], bbox[1], bbox[2], bbox[3]) # Number of rectangles that overlap with the bounding box
            score_df = score_df[~score_df.index.isin(rects_in)]
            tmp_score += len(score_df[score_df==True])
            rects_in.update(score_df[score_df==True].index.values)
        total_score_counter[1] += tmp_score
        

    print(f' white: {total_score_w_b_r[0]/float(len(annotations))}, black: {total_score_w_b_r[1]/float(len(annotations))}, random: {total_score_w_b_r[2]/float(len(annotations))}')
    print(total_attribution_inside)
    print(f' counter white: {total_score_counter[0]/float(len(annotations))}, counter black: {total_score_counter[1]/float(len(annotations))}')


def color():
    #model = torch.load('entire_model_2022-08-28_17.pth').to('cuda')
    model = torch.load('/mnt/ds3lab-scratch/ywattenberg/models/entire_model_vgg_add.pth').to('cuda')
    model = model.module
    model.eval()

    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')
    annotations = pd.read_csv('annotations/annotations_1.csv', index_col='Unnamed: 0')

    chn = []
    chn_color = []
    for index, row in annotations.iterrows():
        print(index)
        reviewerID = row.reviewerID
        asin = row.asin
        color_imp = row.colour

        userID = get_userID(reviewerID, df)
        productID = get_ProductID(asin, df)
        image = Image.open('/mnt/ds3lab-scratch/ywattenberg/data/images/' + asin + '.jpg')
        
        if color_imp == 1:
            chn_color.append(color(model, image, userID, productID))
        else:
            chn.append(color(model, image, userID, productID))
    print(chn)
    print(chn_color)
        
    



def get_color_attribution(model, image, userID, productID, tmm_model=False):
    if tmm_model:
        attributions = gen_explanation(model, image, userID, productID, tmm_model=True)
        return attributions[0], attributions[1]
    else:
        attributions = gen_explanation(model, image, userID, productID, tmm_model=False)
        return attributions[0], attributions[1]

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
    color()
        





