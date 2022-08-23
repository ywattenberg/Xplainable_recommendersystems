import requests
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.append('../.')
from dataset.amazon_dataset_utils import parse


def main():
    path = '/mnt/ds3lab-scratch/ywattenberg/data/meta_AMAZON_FASHION.json'

    pool = ThreadPoolExecutor(16)
    futures = []
    counter = 0
    lines = []
    for line in parse(path):
        counter += 1
        lines.append(line)
        if counter == 100:
            counter = 0
            futures.append(pool.submit(download, lines))
            lines = []
    futures.append(pool.submit(download, lines))

    for task in futures:
        task.result()

def download(json_list):
    # got = set([])
    # with open('downloaded.txt', 'r') as file:
    #     for line in file.readlines():
    #         asin = line.split('.')[0]
    #         got.add(asin)
    for element in json_list:
        asin = element['asin']
        if 'imageURLHighRes' in element.keys():
            urls = element['imageURLHighRes']
            ending = urls[0].split('.')[-1] 
            try:
                r = requests.get(urls[0], stream=True)
                if r.status_code == 200:
                    r.raw.decode_content = True
                    with open(f'/mnt/ds3lab-scratch/ywattenberg/data/fashio_imagesHD/{asin}.{ending}', 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                    logger.info(f'GOT {asin}')
                else:
                    logger.info(f'FAILED {asin}; {urls}')
            except:
                logger.info(f'FAILED {asin}; {urls}')
        else:
            logger.info(f'NO {asin}')


if __name__ == '__main__':
    logpath = "./get_images.log"
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler(logpath)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    main()
