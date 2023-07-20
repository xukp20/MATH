"""
    Detect broken pdf
"""

import PyPDF2
import argparse
import json
from tqdm import tqdm

def is_pdf_broken(pdf_file):
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            return num_pages > 0  # If the PDF has at least one page, it is not considered broken.
    except Exception as e:
        # An exception occurred while trying to read the PDF, indicating it is broken.
        print(f"Error reading PDF file: {e}")
        return True

def get_parser():
    parser = argparse.ArgumentParser(description='Mark the duplicate books in an index.json')
    parser.add_argument('-i', '--input', type=str, help='input index file', default='index.json')
    parser.add_argument('-o', '--output', type=str, help='output index file', default='index.json')
    
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    index_file = args.input
    output_file = args.output

    with open(index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)
    # get all file names
    # keep not rm ones or no rm tag ones
    file_names = [book['path'] for book in index if not book['clean'].get('rm', False)]

    # find broken pdfs
    broken_pdfs = []
    for file in tqdm(file_names):
        if is_pdf_broken(file):
            broken_pdfs.append(file)
    # mark the broken books
    for book in index:
        if book['path'] in broken_pdfs:
            book['clean']['broken'] = True
            book['clean']['rm'] = True
        else:
            book['clean']['broken'] = False

    # set index
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4, ensure_ascii=False)
        