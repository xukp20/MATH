"""
    To count and recheck the answers manually.
"""

import os
import argparse
import json

def get_parser():
    parser = argparse.ArgumentParser(description='Test LLaMa test set')
    parser.add_argument('-r', '--result', type=str, default='/data/xukp/result', help='result dir')
    parser.add_argument('-c', '--check', action='store_true', help='check the result')
    return parser

def main(args):
    total = 0
    correct = 0
    recheck_correct = 0
    out_of_length = 0
    out_of_length_ids = []
    wrong_ids = []
    recheck_ids = []
    for file in os.listdir(args.result):
        with open(os.path.join(args.result, file), 'r', encoding='utf-8') as f:
            result = json.load(f)
        total += 1
        if result['correct']:
            correct += 1
        elif result.get('recheck', False):
            recheck_correct += 1
        else:
            # input
            if args.check:
                # manual recheck
                print('ID: {}'.format(result['id']))
                print('Answer: {}'.format(result['answer']))
                print('Model Answer: {}'.format(result['model_answer']))
                recheck_answer = input('Recheck Answer: ')
            else:
                recheck_answer = 'n'
            if recheck_answer == 'y':
                recheck_correct += 1
                recheck_ids.append(result['id'])
                # save
                result['recheck'] = True

                with open(os.path.join(args.result, file), 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
            # check length
            elif result['model_answer'] == '[':
                out_of_length += 1
                out_of_length_ids.append(result['id'])
            else:
                wrong_ids.append(result['id'])
    print('Total: {}'.format(total))
    print('Correct: {}'.format(correct))
    print('Recheck Correct: {}'.format(recheck_correct))
    print('Out of Length: {}'.format(out_of_length))
    print('Wrong: {}'.format(len(wrong_ids)))
    assert total == correct + len(wrong_ids) + recheck_correct + out_of_length
    with open('recheck_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(recheck_ids))
    with open('out_of_length_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_of_length_ids))
    with open('wrong_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(wrong_ids))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
            


