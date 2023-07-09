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

def get_parser():
    parser = argparse.ArgumentParser(description='Test LLaMa test set')
    parser.add_argument('-d', '--desp', action='store_true', help='use picture description')
    parser.add_argument('-m', '--help_model', type=str, default='3.5', help='description generator', choices=['3.5', '4', 'text'])
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
    program = guidance(
        pattern=pattern,
    )


    # try once
    try: 
        out = program(
            examples=EXAMPLES,
            problem="Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).",
        )
    except Exception as e:
        print(e)
        return
    
    print(out)


if __name__ == "__main__":
    main()