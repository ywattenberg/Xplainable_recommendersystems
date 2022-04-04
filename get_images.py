from asyncore import write
from operator import ne
import requests
import shutil
from dataset.amazon_dataset_utils import parse
import os
import pandas as pd
import threading

def main(thread_num):
    path = 'data/meta_Clothing_Shoes_and_Jewelry.json'
    needed = set([])
    with open('needed.txt', 'r') as file:
        for line in file.readlines():
            needed.add(line.replace('\n', ''))

    # got = set([])
    # with open('downloaded.txt', 'r') as file:
    #     for line in file.readlines():
    #         asin = line.split('.')[0]
    #         got.add(asin)

    needed = needed

    json_list = parse(path)
    json_list = json_list[thread_num*167816:(thread_num+1)*167816]
    with open(f'no_image{thread_num}.txt', 'a') as noimage_file:
        with open(f'failed{thread_num}.txt', 'a') as failed_file:
            with open(f'downloaded{thread_num}.txt', 'a') as succ_file:
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
                                    succ_file.write(f'{asin}\n')
                                else:
                                    failed_file.write(f'{asin}; {urls}\n')
                            except:
                                failed_file.write(f'{asin}; {urls}\n')
                        else:
                            noimage_file.write(f'{asin}\n')


if __name__ == '__main__':
    threads = []

    for idx in range(16):
        threads.append(threading.Thread(target=main, args=(idx,)))
        threads[-1].start()

    for thread in threads:
        thread.join()
