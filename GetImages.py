from asyncore import write
import requests
import shutil
from AmazonDatasetUtils import parse

def main(path):
    json_list = parse(path)
    for element in json_list:
        asin = element['asin']
        if 'imageURL' in element.keys():
            pass
            # urls = element['imageURL']
            # ending = urls[0].split('.')[-1] 
            # r = requests.get(urls[0], stream=True)
            # if r.status_code == 200:
            #     r.raw.decode_content = True
            #     with open((f'./data/images/{asin}.{ending}'), 'wb') as f:
            #         shutil.copyfileobj(r.raw, f)       
        else:
            with open('data/no_image.txt', 'a') as file:
                file.write(f'{asin}\n')

if __name__ == '__main__':
    main('data/meta_Clothing_Shoes_and_Jewelry.json')