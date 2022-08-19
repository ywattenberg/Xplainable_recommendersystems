import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from PIL import Image
import math
import cv2

def color_dist(p1, p2):
    r1, g1, b1 = p1
    r2, g2, b2 = p2
    return math.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)

def color_to_triplets(a):
    return (a[0], a[1], a[2])

def filter_img(img):
    height, width, _ = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5,5), 0)
    edge = cv2.Canny(gray_img, 50, 150)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(edge,kernel,iterations = 3)
    contours = sorted(cv2.findContours(dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-2:-1]
    
    mask = np.zeros((height, width), np.uint8)
    mask = cv2.drawContours(mask, contours,-1, 255, -1)
    return mask, cv2.bitwise_and(img, img, mask=mask)

def simple_filter(img):
    height, width, _ = img.shape
    color_buffer = [color_to_triplets(img[0, 0])]
    for x in range(height):
        for y in range(2):
            pixel = color_to_triplets(img[x, y])
            pixel_2 = color_to_triplets(img[x, width-y-1])
            for bgc in color_buffer:
                if color_dist(pixel, bgc) < 250:
                    break
                color_buffer.append(pixel)
                break
            if len(color_buffer) > 5:

                return True

    for y in range(width):
        for x in range(2):
            pixel = color_to_triplets(img[x, y])
            pixel_2 = color_to_triplets(img[x, width-1-y])
            for bgc in color_buffer:
                if color_dist(pixel, bgc) < 250:
                    break
                color_buffer.append(pixel)
                break
            if len(color_buffer) > 5:
                return True
    return False

def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    while (i < total_contours ):
        area = cv2.contourArea(contours[i])
        if(area > largest_area):
            largest_area = area
            largest_contour_index = i
        i+=1
            
    return largest_area, largest_contour_index

def segmentation(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #mean_bg_color = [0,0,0]
    # for x in range(10):
    #     for y in range(10):
    #         pixel = hsv_img[x, y]
    #         mean_bg_color = mean_bg_color + pixel
    # mean_bg_color = mean_bg_color / 100

    low = [0, 0, 230]
    high = [5, 5, 255]

    curr_mask = cv2.inRange(hsv_img, np.array(low), np.array(high))
    hsv_img[curr_mask > 0] = [0, 0, 0]

    gray_img = cv2.split(hsv_img)[2]
    gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)
    ret, threshold = cv2.threshold(gray_img, 90, 255, 0)

    contours, hierarchy =  cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    merged = []
    for c in contours:
        for point in c:
            merged.append(point)
    merged = np.array(merged)
    hull = cv2.convexHull(merged, False)
    #cv2.fillPoly(img, hull, (0,0,255))
    cv2.drawContours(img, [hull], -1, (0, 0, 255), 3)
    return img

def get_data():
    df = pd.read_csv('/mnt/ds3lab-scratch/ywattenberg/data/compact_CSJ_imgHD.csv')

    #train_data = df[df['rank_latest'] != 1]
    test_data = df[df['rank_latest'] == 1]

    length = len(test_data)
    i = 0

    while i < 100:
        index = randint(0, length)
        user_input = test_data.iloc[index].userID
        asin = test_data.iloc[index].asin
        product_input = test_data.iloc[index].productID
        img_input = cv2.imread(os.path.join('/mnt/ds3lab-scratch/ywattenberg/data/imagesHD/', f'{test_data.iloc[index].asin}.jpg'))
        
        to_clean = simple_filter(img_input)
        if to_clean:
            continue

        i += 1
        #out_img = test(img_input)
        cv2.imwrite(f'cv_false/{asin}.jpg', img_input)
        #cv2.imwrite(f'cv_false/{asin}_mask.jpg', mask)
        #cv2.imwrite(f'cv_false/{asin}_out.jpg', out_img)


def seg_data():
    for file in os.listdir('cv_false'):
        if 'jpg' in file: 
            img = cv2.imread(os.path.join('cv_false', file))
            img = segmentation(img)
            cv2.imwrite(f'cv_false/seg/{file}', img)
def main():
    seg_data()


if __name__ == '__main__':
    #main()
    for image in os.listdir('cv_false'):
        if 'jpg' in image:
            img = cv2.imread(os.path.join('cv_false/', image))
            asin = image.split('.')[0]
            mask = cv2.imread(os.path.join('cv_false/cloth_seg_out', f'{asin}.png'))
            mask2 = cv2.imread(os.path.join('cv_false/seg', image))
            fig = plt.figure(figsize=(10,15))

        fig.add_subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Image')
        
        fig.add_subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        plt.title('Mask')

        fig.add_subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
        plt.title('OpenCV Mask')

        fig.savefig(f'cv_false/both/{image}')
        plt.close(fig)
                