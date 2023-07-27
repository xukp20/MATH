"""
    Count the pdf pages and total size
"""

import os
import json
import PyPDF2
from tqdm import tqdm

def count_files(index_file, base_path, args):
    total_pages = 0
    total_size_bytes = 0

    math_counts = {}
    field_counts = {}
    level_counts = {}
    type_counts = {}

    categories = {}
    PREFIX = ['libgen', 'openstax_textbooks', 'other_books']    # remove 'openmathdep'

    with open(index_file, 'r') as f:
        index_data = json.load(f)

    for item in tqdm(index_data):
        if not args.all:
            if item.get('clean', {}).get('rm', False) is True:
                continue
            
            if args.math:
                if item.get('mark', {}).get('math') != 'yes':
                    continue

        relative_path = item.get('path')
        full_path = os.path.join(base_path, relative_path)

        # get categories
        if relative_path.startswith('openmathdep'):
            continue
        else:
            category = relative_path.split('/')[:-1].join('/')
            if category not in categories:
                categories[category] = {'total_pages': 0, 'total_size_mb': 0}

        if os.path.exists(full_path):
            num_pages, file_size_bytes = count_pages_and_size_pdf(full_path)
            if num_pages == -1:
                continue
            total_pages += num_pages
            total_size_bytes += file_size_bytes
            categories[category]['total_pages'] += num_pages
            categories[category]['total_size_mb'] += bytes_to_mb(file_size_bytes)

        else:
            print(f"File not found: {full_path}")

    total_size_mb = bytes_to_mb(total_size_bytes)

    print(f"Total Pages: {total_pages}")
    print(f"Total Size (MB): {total_size_mb:.2f} MB")

    print("\nSummary:")
    print("Math Counts:\n", math_counts)
    print("Field Counts\n:", field_counts)
    print("Level Counts\n:", level_counts)
    print("Type Counts:\n", type_counts)

    print("\nCategories:")
    print(categories)

    # sum the libgen
    libgen_total_pages = 0
    libgen_total_size_mb = 0
    for category in categories:
        if category.startswith('libgen'):
            libgen_total_pages += categories[category]['total_pages']
            libgen_total_size_mb += categories[category]['total_size_mb']
        
    print(f"Libgen Total Pages: {libgen_total_pages}")
    print(f"Libgen Total Size (MB): {libgen_total_size_mb:.2f} MB")

def count_pages_and_size_pdf(pdf_file):
    num_pages = 0
    file_size_bytes = os.path.getsize(pdf_file)

    with open(pdf_file, 'rb') as f:
        try:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
        except:
            tqdm.write("Error reading pdf")
            num_pages = -1

    return num_pages, file_size_bytes

def bytes_to_mb(size_in_bytes):
    return size_in_bytes / (1024 * 1024)

def count_field_values(counts_dict, item, field):
    value = item.get('mark', {}).get(field)
    if value:
        counts_dict[value] = counts_dict.get(value, 0) + 1

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='Count the pdf pages and total size')
    parser.add_argument('-b', '--base_path', type=str, help='base path', default='/data/xukp')
    parser.add_argument('-i', '--input', type=str, help='input index file', default='index.json')
    parser.add_argument('-m', '--math', action='store_true', help='count math')
    parser.add_argument('-a', '--all', action='store_true', help='count all')
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    index_file = args.input
    base_path = args.base_path

    count_files(index_file, base_path, args)