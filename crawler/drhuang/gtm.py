"""
    Crawler for GTM textbooks
"""

# from dr.huang.com
# index page
BASE_URL = "http://drhuang.com"
INDEX_PATH = "/science/mathematics/book/gtm/"
INDEX_URL = BASE_URL + INDEX_PATH

import argparse, re, os, requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='/data/xukp/drhuang/gtm', help='the directory to store the gtm books')
    parser.add_argument('-i', '--index_file', type=str, default='index.json', help='the index file of gtm , name to url')
    return parser


def get_index():
    """
    get the index of gtm books
    """
    response = requests.get(INDEX_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')
    index = {}
    for link in links:
        # get the ones which href start with /science/mathematics/book/gtm/
        # use the a's text as the file name
        if link['href'].startswith('/science/mathematics/book/gtm/'):
            index[link.text] = BASE_URL + link['href']
    return index


def get_book(url, output_dir, file_name):
    """
        Save pdf as file_name in output_dir
    """
    response = requests.get(url)
    with open(os.path.join(output_dir, file_name), 'wb') as f:
        f.write(response.content)
    

if __name__ == "__main__":
    args = get_parser().parse_args()
    index = get_index()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # save the index
    with open(os.path.join(args.output_dir, args.index_file), 'w') as f:
        json.dump(index, f, indent=4)
    
    for name, url in tqdm(index.items()):
        if not os.path.exists(os.path.join(args.output_dir, name)):
            get_book(url, args.output_dir, name)
