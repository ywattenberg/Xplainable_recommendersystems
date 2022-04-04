import requests
import shutil
from dataset.amazon_dataset_utils import parse
from concurrent.futures import ThreadPoolExecutor
import logging

def main():
    path = 'data/meta_Clothing_Shoes_and_Jewelry.json'
    needed = set([])
    with open('needed.txt', 'r') as file:
        for line in file.readlines():
            needed.add(line.replace('\n', ''))

    pool = ThreadPoolExecutor(16)
    futures = []
    counter = 0
    lines = []
    for line in parse(path):
        counter += 1
        lines.append(line)
        if counter == 100:
            counter = 0
            futures.append(pool.submit(download, lines, needed))
            lines = []
    futures.append(pool.submit(download, lines, needed))

    for task in futures:
        task.join()

def download(json_list, needed):
    # got = set([])
    # with open('downloaded.txt', 'r') as file:
    #     for line in file.readlines():
    #         asin = line.split('.')[0]
    #         got.add(asin)
    for element in json_list:
        asin = element['asin']
        if asin in needed:
            if 'imageURL' in element.keys():
                urls = element['imageURL']
                ending = urls[0].split('.')[-1] 
                try:
                    r = requests.get(urls[0], stream=True)
                    if r.status_code == 200:
                        r.raw.decode_content = True
                        with open(f'./data/images/{asin}.{ending}', 'wb') as f:
                            shutil.copyfileobj(r.raw, f)
                        logger.info(f'GOT {asin}\n')
                    else:
                        logger.info(f'FAILED {asin}; {urls}\n')
                except:
                    logger.info(f'FAILED {asin}; {urls}\n')
            else:
                logger.info(f'NO {asin}\n')


if __name__ == '__main__':
    logpath = "./get_images.log"
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler(logpath)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    main()
