"""
    To get the problem ids of prm800k or MATH
"""

from reader import test_prm800k_asy_reader, test_asy_reader, DEFAULT_PATH

if __name__ == '__main__':
    with open('prm800k_asy_ids.txt', 'w', encoding='utf-8') as f:
        for problem in test_prm800k_asy_reader(DEFAULT_PATH, '3.5'):
            f.write('{}\n'.format(problem['id']))
    with open('math_asy_ids.txt', 'w', encoding='utf-8') as f:
        for problem in test_asy_reader(DEFAULT_PATH, '3.5'):
            f.write('{}\n'.format(problem['id']))
