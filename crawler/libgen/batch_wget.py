"""
    To download batch of books from an index.json in a dir by wget
"""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
MAX_THREADS = 4

def get_parser():
    parser = argparse.ArgumentParser(description='Download batch of books from an index.json in a dir by wget')
    parser.add_argument('-i', '--input', type=str, help='input dir', default='/data/xukp/libgen/gre math')
    parser.add_argument('-f', '--index_file', type=str, help='index file', default='index.json')
    
    return parser

def download_book(book):
    urls = book['urls']
    for url in urls:
        cmd = f'wget -c {url}'
        # if success, break
        if os.system(cmd) == 0:
            return True, book['id']
    tqdm.write(f'Failed to download book {book["book_title"]}')
    return False, book['id']


def download_book_multi_threaded(index):
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(download_book, book) for book in index if book['downloaded'] == False]

        count = 0
        # Process the results as they complete
        for future in tqdm(as_completed(futures), desc='Downloading books', total=len(futures)):
            success, id = future.result()
            if success:
                count += 1
                for book in index:
                    if book['id'] == id:
                        book['downloaded'] = True
                        break
        
        print(f'Downloaded {count} books out of {len(futures)} tries.')
    # set index
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4)

if __name__ == '__main__':
    args = get_parser().parse_args()
    input_dir = args.input
    index_file = args.index_file

    with open(os.path.join(input_dir, index_file), 'r') as f:
        index = json.load(f)
    
    # move to input dir
    os.chdir(input_dir)
    download_book_multi_threaded(index)
    
            

