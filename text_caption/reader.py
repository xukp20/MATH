"""
    Data reader of MATH dataset and its descriptions generated by GPTs
"""
import re

DEFAULT_PATH = '/data/xukp/MATH'
PROBLEMS = 'MATH'
DESCS = {
    '3.5': 'caption_gpt-3.5-turbo',
    '4': 'caption_gpt-4',
    'text': 'caption_text-davinci-003',
}


# for the original MATH dataset
# generator of the problems in test set with [asy]
def test_asy_reader(path=DEFAULT_PATH, help_model='3.5'):
    # go through each problem in each category of the test set
    import os, json
    test_path = os.path.join(path, PROBLEMS, 'test')
    for category in os.listdir(test_path):
        desc_path = os.path.join(path, DESCS[help_model], 'test', category)
        # go through the problems in desc dir
        problem_path = os.path.join(test_path, category)
        for desc_file in os.listdir(desc_path):
            problem_file = os.path.join(problem_path, desc_file.replace('.txt', '.json'))
            # read description
            with open(os.path.join(desc_path, desc_file), 'r', encoding='utf-8') as f:
                description = f.read()
            # read problem
            with open(problem_file, 'r', encoding='utf-8') as f:
                problem = json.loads(f.read())
            problem['description'] = description
            # mark the problem
            problem['id'] = category + '_' + desc_file.strip('.txt')

            # produce sample 
            yield problem


# strip [asy] command from the problem text
def strip_asy(problem):
    problem_text = problem['problem']

    # remove text in [asy] ... [/asy]
    pattern = re.compile(r'\[asy\].*?\[/asy\]', re.S)
    problem_text = re.sub(pattern, '', problem_text)

    problem['problem'] = problem_text

    return problem


def test_asy_strip_reader(path=DEFAULT_PATH, help_model='3.5'):
    for problem in test_asy_reader(path=path, help_model=help_model):
        yield strip_asy(problem)


# for the subset of 500 problems in test set of prm800k math
PRM800K = 'prm800k'
# generator of the 500 problems in test set of prm800k math
def test_prm800k_reader(path=DEFAULT_PATH):
    # all the test problems in jsonl file
    import os, json
    test_path = os.path.join(path, PRM800K, 'test.jsonl')
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            problem = json.loads(line)
            unique_id = problem['unique_id'].split('/')
            category = unique_id[1]
            id = unique_id[2].strip('.json')
            problem['id'] = category + '_' + id

            # produce sample
            yield problem

# generator of the 500 problems in test set of prm800k math with asy
def test_prm800k_asy_reader(path=DEFAULT_PATH, help_model='3.5'):
    # all the test problems in jsonl file
    import os
    for problem in test_prm800k_reader(path=path):
        problem_text = problem['problem']
        # if no [asy] in problem, skip
        if '[asy]' not in problem_text:
            continue

        # find the final _
        id = problem['id'].split('_')[-1]
        # find the category
        category = problem['id'].strip('_' + id)

        desc_path = os.path.join(path, DESCS[help_model], 'test', category, id + '.txt')
        # read description

        # no desciption file, skip, not asy
        if not os.path.exists(desc_path):
            print('Warning: missing description file for {}'.format(problem['id']))
            continue

        with open(desc_path, 'r', encoding='utf-8') as f:
            description = f.read()

        problem['description'] = description

        # produce sample
        yield problem



def test_prm800k_asy_strip_reader(path=DEFAULT_PATH, help_model='3.5'):
    for problem in test_prm800k_asy_reader(path=path, help_model=help_model):
        yield strip_asy(problem)


if __name__ == '__main__':
    # simple test
    # generator = test_asy_reader()
    # print(next(generator))
    # generator = test_prm800k_asy_reader(help_model='4')
    # count = 0
    # for i in generator:
    #     count += 1
    # print(count)
    generator = test_prm800k_asy_strip_reader(help_model='4')
    count = 0
    for i in generator:
        count += 1
    print(count)