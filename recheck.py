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
    parser.add_argument('-i', '--id', type=str, default='prm800k_asy_ids.txt', help='id file')
    return parser

def main(args):
    total = 0
    correct = 0
    recheck_correct = 0
    out_of_length = 0
    error = 0
    correct_ids = []
    out_of_length_ids = []
    wrong_ids = []
    recheck_ids = []
    error_ids = []


    with open(args.id, 'r', encoding='utf-8') as f:
        ids = [line.strip() for line in f.readlines()]

    for file in ids:
        file = file + '.json'
        total += 1
        file_path = os.path.join(args.result, file)
        if not os.path.exists(file_path):
            error += 1
            error_ids.append(file)
            continue

        with open(os.path.join(args.result, file), 'r', encoding='utf-8') as f:
            result = json.load(f)
        if result['correct']:
            correct += 1
            correct_ids.append(result['id'])
        elif result.get('recheck', False):
            recheck_correct += 1
            recheck_ids.append(result['id'])
        elif result.get('recheck', True) == False:
            # exist and false
            if result['model_answer'] == '[':
                out_of_length += 1
                out_of_length_ids.append(result['id'])
            else:
                wrong_ids.append(result['id'])

        else:
            # input
            if args.check:
                # manual recheck
                print('ID: {}'.format(result['id']))
                print('Answer: {}'.format(result['answer']))
                print('Model Answer: {}'.format(result['model_answer']))
                recheck_answer = input('Recheck Answer: ')

                if recheck_answer == 'y':
                    # save
                    result['recheck'] = True
                else:
                    result['recheck'] = False

                with open(os.path.join(args.result, file), 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                    
            else:
                recheck_answer = 'n'
            if recheck_answer == 'y':
                recheck_correct += 1
                recheck_ids.append(result['id'])
                
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
    print('Error: {}'.format(error))
    assert total == correct + len(wrong_ids) + recheck_correct + out_of_length + error

    if not os.path.exists('recheck'):
        os.mkdir('recheck')

    with open('recheck/correct_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(correct_ids))
    with open('recheck/recheck_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(recheck_ids))
    with open('recheck/out_of_length_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_of_length_ids))
    with open('recheck/wrong_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(wrong_ids))
    with open('recheck/error_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(error_ids))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
            


