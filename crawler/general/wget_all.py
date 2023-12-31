import argparse
import os

from tqdm import tqdm
import json

def get_parser():
    parser = argparse.ArgumentParser(description='Rename files in a dir')
    parser.add_argument('-p', '--path', type=str, help='path', default=None)
    parser.add_argument('-l', '--list', type=str, help='list file', default='urls.json')

    return parser

from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    args = get_parser().parse_args()
    path = args.path
    list_file = args.list

    # change to the path
    if path is not None:
        os.chdir(path)
    else:
        print('No path specified.')
        return
    
    with open(list_file, 'r') as f:
        urls = json.load(f)
    
    os.system('export http_proxy=http://127.0.0.1:7890')
    os.system('export https_proxy=http://127.0.0.1:7890')
    # for url in tqdm(urls):
    #     cmd = f'wget -c {url}'
    #     if os.system(cmd) != 0:
    #         tqdm.write(f'Failed to download {url}')

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(os.system, f'wget -c {url} > /dev/null 2>&1') for url in urls]
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result() != 0:
                tqdm.write(f'Failed to download')
                
if __name__ == '__main__':
    main()