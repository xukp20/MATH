"""
    Count the pdf pages and total size
"""

import os
import json
import PyPDF2
from tqdm import tqdm

def count_files(index_file, base_path):
    total_pages = 0
    total_size_bytes = 0

    math_counts = {}
    field_counts = {}
    level_counts = {}
    type_counts = {}

    with open(index_file, 'r') as f:
        index_data = json.load(f)

    for item in tqdm(index_data):
        if item.get('clean', {}).get('rm', False) is True:
            continue
            
        if item.get('mark', {}).get('math') != 'yes':
            continue

        relative_path = item.get('path')
        full_path = os.path.join(base_path, relative_path)

        if os.path.exists(full_path):
            num_pages, file_size_bytes = count_pages_and_size_pdf(full_path)
            if num_pages == -1:
                continue
            total_pages += num_pages
            total_size_bytes += file_size_bytes

            # Count different values in each field
            count_field_values(math_counts, item, 'math')
            count_field_values(field_counts, item, 'field')
            count_field_values(level_counts, item, 'level')
            count_field_values(type_counts, item, 'type')
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
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    index_file = args.input
    base_path = args.base_path

    count_files(index_file, base_path)