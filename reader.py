"""
    Data reader of MATH dataset and its descriptions generated by GPTs
"""

DEFAULT_PATH = '/data/xukp/MATH'
PROBLEMS = 'MATH'
DESCS = {
    '3.5': 'caption_gpt-3.5-turbo',
    '4': 'caption_gpt-4',
    'text': 'caption_text-davinci-003',
}

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


if __name__ == '__main__':
    # simple test
    generator = test_asy_reader()
    print(next(generator))