import numpy as np
from dataset.amazon_dataset_utils import parse
import json

if __name__ == '__main__':
    path = 'data/meta_Clothing_Shoes_and_Jewelry.json'
    wanted = set(['asin', 'imageURLHighRes', 'imageURL'])
    with open('data/compact_meta.json', 'a') as file: 
        for line in parse(path):
            result = {}
            for key in line.keys():
                if key in wanted:
                    result[key] = line[key]
            if len(result) > 1:
                file.write(json.dumps(result, indent=0).replace('\n', ''))
                file.write('\n')