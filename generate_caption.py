"""
    To generate captions for asy files by GPTs
"""

import argparse
import os
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='Generate captions for asy files by GPTs')
    parser.add_argument('--asy_dir', type=str, default='./asy', help='the directory of asy files')
    parser.add_argument('--output_dir', type=str, default='./caption', help='the directory of asy captions')
    parser.add_argument('--set', type=str, default='test', help='the set of asy files', choices=['train', 'test'])
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='the model to generate captions', choices=['gpt-3.5-turbo', 'gpt-4', 'text-davinci-003'])
    parser.add_argument('--temperature', type=float, default=1, help='the temperature of GPT')
    parser.add_argument('--max_tokens', type=int, default=1000, help='the max tokens of GPT')
    parser.add_argument('--top_p', type=float, default=1, help='the top p of GPT')

    return parser

# 2023/7/8 version of the instruction
ASY_INSTRUCTION = """You are given a asy command of latex, please describe the elements inside the corresponding picture it produces in a short paragraph. 
1. The paragraph should be list of <type of the element>: <position>
2. treat items like lines, dots, arrows as an element. 
3. try to produce high level description. For example, four lines may be concluded as a square. 
4. Don't describe details like the lines' width or thickness."""

def get_messages(asy_command):
    return [
        {"role": "system", "content": ASY_INSTRUCTION},
        {"role": "user", "content": asy_command}
    ]

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    from utils.openai_req import gpt3dot5_req, gpt4_req, text_davinci_003_req
    from utils.openai_req import reset, save_log

    if args.model == 'gpt-3.5-turbo':
        gpt_req = gpt3dot5_req
    elif args.model == 'gpt-4':
        gpt_req = gpt4_req
    elif args.model == 'text-davinci-003':
        gpt_req = text_davinci_003_req

    path = os.path.join(args.asy_dir, args.set)
    categories = os.listdir(path)
    output_path = os.path.join(args.output_dir + '_' + args.model, args.set)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    reset()
    for category in categories:
        category_path = os.path.join(path, category)
        output_category_path = os.path.join(output_path, category)
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)
        
        files = os.listdir(category_path)
        for file in tqdm(files, desc=category):
            # if output file exists, skip
            if os.path.exists(os.path.join(output_category_path, file.replace('.asy', '.txt'))):
                continue

            file_path = os.path.join(category_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                asy_command = f.read()

            # call model
            caption = gpt_req(get_messages(asy_command), temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p)
            
            with open(os.path.join(output_category_path, file.replace('.asy', '.txt')), 'w', encoding='utf-8') as f:
                f.write(caption)

    save_log()




