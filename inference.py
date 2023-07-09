"""
    Test MATH test set on LLaMa w and w/o picture description
    Only test the ones with [asy]
"""
# set up the environment
import os
os.environ["HF_HOME"] = "/data/cache/huggingface"
# check env
print('HF_HOME: {}'.format(os.environ["HF_HOME"]))

# use microsoft guidance
import guidance
import torch
import argparse
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='Test LLaMa test set')
    parser.add_argument('-d', '--desp', action='store_true', help='use picture description')
    parser.add_argument('-m', '--help_model', type=str, default='3.5', help='description generator', choices=['3.5', '4', 'text'])
    parser.add_argument('--log', type=str, default='log.txt', help='log file')
    parser.add_argument('--result', type=str, default='/data/xukp/result', help='result dir')
    return parser


MODEL = 'huggyllama/llama-65b'
from patterns import EXAMPLES, PATTERN_W_DESC, PATTERN_WO_DESC

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.desp:
        print('Use picture description')
        print('Description generator: {}'.format(args.help_model))
    else:
        print('Do not use picture description')
    
    
    # use LLaMa 65b
    print('Loading LLaMa 65b')
    guidance.llm = guidance.llms.transformers.LLaMA(MODEL, device_map="auto", token_healing=True, torch_dtype=torch.bfloat16)

    # we can pre-define valid option sets
    valid_judgement = ["True", "False", "Unknown"]

    # program
    pattern = PATTERN_W_DESC if args.desp else PATTERN_WO_DESC
    program = guidance(pattern)

    TOTAL_TEST = 419    # total count for tqdm

    # log and result
    if not os.path.exists(args.result):
        os.mkdir(args.result)
    # statistics
    correct = 0
    total = 0
    wrong_ids = []    

    # settings
    answer_pat = r'boxed\{(.*)\}'

    def save_log():
        with open(args.log, 'a', encoding='utf-8') as f:
            import time 
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write('\n\nTime: {}\n'.format(current_time))
            acc = correct / total
            f.write('Correct: {}/{}={:.2f}%\n'.format(correct, total, acc*100))
            for i in wrong_ids:
                f.write('{}\n'.format(i))
            f.write('\n')

    from reader import test_asy_reader, DEFAULT_PATH
    generator = test_asy_reader(path=DEFAULT_PATH, help_model=args.help_model)
    for _ in tqdm(range(TOTAL_TEST), desc='Testing'):
        try:
            problem = next(generator)
            out = program(
                examples=EXAMPLES,
                problem=problem,
            )
        except Exception as e:
            print(e)
            save_log()
            return
        
        # test
        model_answer = out['answer']
        # get answer
        import re
        
        # allow model to present box in answer
        model_answer_box = re.search(answer_pat, model_answer)
        if model_answer_box:
            model_answer = model_answer_box.groups()[0]
        # if no answer, try to get from the model's solution
        model_solution = out['solution']
        if model_answer == '':
            model_answer = re.search(answer_pat, model_solution).groups()[0]

        answer = re.search(answer_pat, problem['solution']).groups()[0]

        # check
        if model_answer == answer:
            correct += 1
        else:
            wrong_ids.append(problem['id'])

        # save result
        problem['model_answer'] = model_answer
        problem['model_solution'] = model_solution

        # save
        with open(os.path.join(args.result, '{}.json'.format(problem['id'])), 'w', encoding='utf-8') as f:
            import json
            json.dump(problem, f, ensure_ascii=False, indent=4)

        total += 1

    save_log()
        



if __name__ == "__main__":
    main()