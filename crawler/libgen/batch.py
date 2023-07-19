"""
    To download a batch of dirs from keyword json
"""

from thread_base import crawl_libgen_multi_threaded
import json
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Download a batch of dirs from keyword json')
    parser.add_argument('-k', '--keyword_json', type=str, help='keyword json file', default='./keywords.json')
    parser.add_argument('-b', '--base_dir', type=str, help='base output dir', default='./data/xukp/libgen')
    parser.add_argument('-d', '--download', action='store_true', help='download')
    parser.add_argument('-c', '--cover', action='store_true', help='download cover')
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    keyword_json = args.keyword_json
    base_dir = args.base_dir
    download = args.download
    cover = args.cover

    with open(keyword_json, 'r') as f:
        keywords = json.load(f)
    
    for keyword in keywords:
        print(f'Keyword: {keyword}')
        output_dir = os.path.join(base_dir, keyword.replace(' ', '_'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        crawl_libgen_multi_threaded(keyword, output_dir, download=download, cover=cover)
