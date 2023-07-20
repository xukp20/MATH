import os
import argparse
import json

def scan_pdf_files(base_path, include_subdirs):
    pdf_files = []
    for subdir in include_subdirs:
        subdir_path = os.path.join(base_path, subdir)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                if file_name.lower().endswith('.pdf'):
                    relative_path = os.path.relpath(os.path.join(subdir_path, file_name), base_path)
                    pdf_files.append(relative_path)
    return pdf_files

def main():
    parser = argparse.ArgumentParser(description="Scan PDF files in specified subdirectories.")
    parser.add_argument("-b", "--base_path", type=str, help="Base directory path.", default="/data/xukp")
    parser.add_argument("-i", "--include", nargs='+', default=[], help="Subdirectories to include in scanning (space-separated).")
    parser.add_argument("-f", "--include_file", type=str, help="File containing subdirectories to include in scanning (json format).", default=None)
    parser.add_argument("-o", "--output", type=str, help="Output file name.", default="index.json")
    args = parser.parse_args()

    base_path = args.base_path
    include_subdirs = args.include
    include_file = args.include_file
    if include_file is not None:
        with open(include_file, "r", encoding="utf-8") as json_file:
            include_subdirs += json.load(json_file)

    pdf_files_list = scan_pdf_files(base_path, include_subdirs)

    output_file = args.output
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump([{"path": pdf_file, "clean": {}, "mark": {}} for pdf_file in pdf_files_list], json_file, indent=4, ensure_ascii=False)
        

    print("PDF files list has been saved to {}".format(output_file))

if __name__ == "__main__":
    main()