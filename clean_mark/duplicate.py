"""
    To mark the duplicate books in an index.json
"""

import os
import difflib
import json
import argparse

DIFF_THRESHOLD = 0.9
def are_filenames_similar(file1, file2):
    return difflib.SequenceMatcher(None, file1, file2).ratio() > 0.9

def find_duplicate_groups(file_names):
    groups = []
    while file_names:
        current_file = file_names.pop(0)
        group = [current_file]

        # Compare with the remaining file names
        for file in file_names[:]:
            if are_filenames_similar(os.path.basename(current_file), os.path.basename(file)):
                group.append(file)
                file_names.remove(file)

        # Only save the group if it has more than one file
        if len(group) > 1:
            groups.append(group)

    return groups


def recheck(groups):
    # manual check
    removes = []
    for group in groups:
        print("Group: {}".format([os.path.basename(file) for file in group]))
        answer = input("Are these files duplicate? (y/n)")
        if answer.lower() == 'y':
            print("Choose keep ones:")
            for i, file in enumerate(group):
                print("{}: {}".format(i, file))
            # can keep more than one
            keep_ones = input("Input the index of the files to keep (space-separated):")
            keep_ones = keep_ones.split()
            keep_ones = [int(i) for i in keep_ones]
            for i, file in enumerate(group):
                if i not in keep_ones:
                    removes.append(file)
    return removes


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

    # find duplicate groups
    groups = find_duplicate_groups(file_names)
    # manual check
    removes = recheck(groups)
    # mark the duplicate books
    for book in index:
        if book['path'] in removes:
            book['clean']['rm'] = True
        else:
            book['clean']['rm'] = False

    # save index
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4, ensure_ascii=False)

    


