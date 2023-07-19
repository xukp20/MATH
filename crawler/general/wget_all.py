import argparse
import os

from tqdm import tqdm
import json

def get_parser():
    parser = argparse.ArgumentParser(description='Rename files in a dir')
    parser.add_argument('-p', '--path', type=str, help='path', default=None)
    parser.add_argument('-l', '--list', type=str, help='list file', default='urls.json')

    return parser

def main():
    args = get_parser().parse_args()
    path = args.path
    list_file = args.list

    with open(list_file, 'r') as f:
        urls = json.load(f)

    # change to the path
    if path is not None:
        os.chdir(path)
    else:
        print('No path specified.')
        return
    
    for url in tqdm(urls):
        cmd = f'wget -c {url}'
        if os.system(cmd) != 0:
            tqdm.write(f'Failed to download {url}')

if __name__ == '__main__':
    main()