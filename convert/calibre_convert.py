"""
    Use the Calibre command line tool to convert from epub or mobi.. to pdf
"""

import argparse 
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
MAX_THREADS = 4
OUT_FORMAT = '.pdf'

def get_parser():
    parser = argparse.ArgumentParser(description='Convert batch of books from an index.json in a dir by Calibre')
    parser.add_argument('-i', '--input', type=str, help='input dir', default='/data/xukp/libgen/gre_math')
    parser.add_argument('-f', '--format', type=str, help='original format, None for all', default=None)
    parser.add_argument('-q', '--quiet', action='store_true', help='set Calibre output to quiet')
    parser.add_argument('-d', '--duplicate', type=str, help='duplicate strategy', default='rename', choices=['overwrite', 'skip', 'rename'])

    return parser


def convert_book_group(exist, group, quiet=False, duplicate='rename', dir='.'):
    # rename
    outputs = group.copy()
    if duplicate == 'rename':
        if exist:
            for i in range(len(outputs)):
                outputs[i] = outputs[i][:outputs[i].rfind('.')] + '_' + str(i+1) + OUT_FORMAT
        else:
            outputs[0] = outputs[0][:outputs[0].rfind('.')] + OUT_FORMAT
            for i in range(1, len(outputs)):
                outputs[i] = outputs[i][:outputs[i].rfind('.')] + '_' + str(i) + OUT_FORMAT

    # convert
    count = 0
    for i in range(len(group)):
        cmd = f'ebook-convert "{os.path.join(dir, group[i])}" "{os.path.join(dir, outputs[i])}"' + (' > /dev/null 2>&1' if quiet else '')
        if os.system(cmd) != 0:
            tqdm.write(f'Failed to convert book\n {group[i]}\n')
        else:
            tqdm.write(f'Converted book\n {group[i]}\n to\n {outputs[i]}\n')
            count += 1
    
    return count


def main():
    args = get_parser().parse_args()
    input_dir = args.input
    original_format = args.format
    quiet = args.quiet
    duplicate = args.duplicate

    # get all files
    files = []
    # only in the base path
    for file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file)):
            if file.endswith(OUT_FORMAT):
                continue

            if original_format is None or file.endswith(original_format):
                files.append(file)
    

    # group the files with the same name
    file_groups = {}
    for file in files:
        name = file[:file.rfind('.')]
        if name not in file_groups:
            file_groups[name] = []
        file_groups[name].append(file)

    # convert
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        for name, group in file_groups.items():
            # check if pdf exists
            exist = os.path.exists(os.path.join(input_dir, name + OUT_FORMAT))
            futures.append(executor.submit(convert_book_group, exist, group, quiet, duplicate, input_dir))

        count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc='Converting book groups'):
            count += future.result()
        
        print(f'Converted {count} books out of {len(files)}')
        

if __name__ == '__main__':
    main()