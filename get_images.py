from asyncore import write
import requests
import shutil
from dataset.amazon_dataset_utils import parse
import os

def main(path):
    json_list = parse(path)
    for element in json_list:
        asin = element['asin']
        if 'imageURL' in element.keys():
            urls = element['imageURL']
            ending = urls[0].split('.')[-1] 
            r = requests.get(urls[0], stream=True)
            if r.status_code == 200:
                r.raw.decode_content = True
                with open(f'./data/images/{asin}.{ending}', 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                print(f'got image for {asin}')
            else:
                with open('failed.txt', 'a') as file:
                    file.write(f'{asin}; {urls}\n')
        else:
            # with open('data/no_image.txt', 'a') as file:
            #     file.write(f'{asin}\n')
            pass

def getLines(path):
    lines = []
    with open(path, 'r') as file:
        for l in file:
            lines.append(l.split('\n')[0])
    return lines

if __name__ == '__main__':
    #main('data/meta_Clothing_Shoes_and_Jewelry.json')
    files = [f for f in os.listdir('./data/images') if os.path.isfile(os.path.join('./data/images', f))]
    with open('downloaded.txt', 'w') as file:
        for item in files:
            file.writelines([item])