"""
    openmathdep crawler
"""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from tqdm import tqdm
import requests

MAX_THREADS = 1
INDEX_URL = 'https://download.tuxfamily.org/openmathdep/'

PROXIES = {'https': 'http://127.0.0.1:7890', 'http': 'http://127.0.0.1:7890'}

### Tool
# use backoff to help recover from exception
def backoff_hook(_details):
    tqdm.write("Retrying...")

import backoff
@backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=300, on_backoff=backoff_hook)
def get_page(url, params=None, **kwargs):
    """
        Get the page content
    """
    if params is None:
        return requests.get(url, proxies=PROXIES, **kwargs)
    else:
        return requests.get(url, params=params, proxies=PROXIES, **kwargs)
    
def get_parser():
    parser = argparse.ArgumentParser(description='OpenMathDep crawler')
    parser.add_argument('-p', '--path', type=str, help='path', default='/data/xukp/openmathdep')
    
    return parser

def get_dirs():
    # get the index page and parse the dirs
    r = get_page(INDEX_URL)
    dirs = []
    soup = BeautifulSoup(r.content, 'html.parser')

    # find all the <tr> except Parent Directory and README.html
    for tr in soup.find_all('tr'):
        if tr.find('a') is not None:
            if tr.find('a').text != 'Parent Directory' and tr.find('a').text != 'README.html':
                dirs.append(tr.find('a')['href'])
    return dirs

def get_file_urls(dir):
    dir_url = INDEX_URL + dir
    # same as get_dirs, get all except Parent Directory
    r = get_page(dir_url)
    urls = []
    soup = BeautifulSoup(r.content, 'html.parser')
    for tr in soup.find_all('tr'):
        if tr.find('a') is not None:
            if tr.find('a').text != 'Parent Directory':
                urls.append(dir_url + tr.find('a')['href'])
    return urls

def download_book(url, quiet=False):
    tries = 0
    while tries < 3:
        cmd = f'wget -c {url} > /dev/null 2>&1' if quiet else f'wget -c {url}'
        # if success, break
        if os.system(cmd) == 0:
            return True
        tries += 1
    tqdm.write("Unable to get from {}".format(url))
    return False


def download_book_multi_threaded(urls, quiet):
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(download_book, url, quiet) for url in urls]

        count = 0
        # Process the results as they complete
        for future in tqdm(as_completed(futures), desc='Downloading books', total=len(futures)):
            success = future.result()
            if success:
                count += 1
        print(f'Downloaded {count} books out of {len(futures)} tries.')


if __name__ == '__main__':
    args = get_parser().parse_args()
    path = args.path

    # change to the path
    if path is not None:
        os.chdir(path)
    else:
        print('No path specified.')
        exit(0)

    if os.path.exists('urls.json'):
        with open('urls.json', 'r') as f:
            urls = json.load(f)
        download_book_multi_threaded(urls, False)
        exit(0)
        
    # get dirs
    dirs = get_dirs()
    # get file urls
    urls = []
    for dir in tqdm(dirs):
        urls += get_file_urls(dir)

    # save urls
    with open('urls.json', 'w', encoding='utf-8') as f:
        json.dump(urls, f, indent=4)

    # download
    download_book_multi_threaded(urls, False)
