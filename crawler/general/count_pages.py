import os
import PyPDF2
from tqdm import tqdm

def count_pages_in_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        try:
            reader = PyPDF2.PdfReader(file)
            return len(reader.pages)
        except:
            print("find broken pdf: {}".format((pdf_file)))
            return 0

def count_pages_in_directory(directory_path):
    total_pages = 0
    for root, dirs, files in os.walk(directory_path):
        for filename in tqdm(files, desc=root):
            if filename.lower().endswith('.pdf'):
                pdf_file_path = os.path.join(root, filename)
                total_pages += count_pages_in_pdf(pdf_file_path)
    return total_pages


import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='Count pages in PDF files in a dir')
    parser.add_argument('-p', '--path', type=str, help='path', default=None)

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    directory_path = args.path
    if directory_path is None:
        print('No path specified.')
        exit(0)
        
    total_pages = count_pages_in_directory(directory_path)
    print("Total pages of all PDF files in the directory (including subdirectories):", total_pages)