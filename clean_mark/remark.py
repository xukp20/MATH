"""
    Load the mark from index and recheck the ones that are marked not math
"""

import argparse
import json
import os
from tqdm import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Load the mark from index and recheck the ones that are marked not math')
    parser.add_argument('-i', '--input', type=str, help='input index file', default='index.json')
    parser.add_argument('-o', '--output', type=str, help='output index file', default='recheck_index.json')

    return parser.parse_args()

def main():
    args = arg_parse()
    index_file = args.input
    output_file = args.output

    with open(index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)

    # get all file names
    # keep not rm ones or no rm tag ones
    for book in tqdm(index, desc='Rechecking'):
        if book['clean'].get('rm', False):
            continue
        
        if book['mark']['math'] == 'no' or book['mark']['math'] == 'unknown':
            print(book['path'])
            math = input('Is it math? (y/n)')
            if math == 'y':
                book['mark']['math'] = 'yes'
            else:
                book['mark']['math'] = 'no'
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
