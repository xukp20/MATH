import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=str, help='index file', default='recheck_index.json')
    parser.add_argument('--text', type=str, help='text file', default='textbook.json')
    parser.add_argument('--work', type=str, help='work file', default='workbook.json')
    parser.add_argument('--dep_url', type=str, help='openmathdep url', default='/data/xukp/openmathdep/urls.json')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    index_file = args.index
    text_file = args.text
    work_file = args.work
    dep_url = args.dep_url

    with open(index_file, 'r') as f:
        index = json.load(f)
    
    with open(dep_url, 'r') as f:
        dep_urls = json.load(f)
        dep_names = [item.split('/')[-1] for item in dep_urls]

    text = []
    work = []

    with open(text_file, 'r') as f:
        text = json.load(f)
    
    with open(work_file, 'r') as f:
        work = json.load(f)

    for item in index:
        if item['clean']['rm'] or item['mark']['math'] != 'yes':
            continue
        
        new_item = {'path': item['path']}
        if item['path'].startswith('libgen/act_math') or item['path'].startswith('libgen/gmat') or item['path'].startswith('libgen/sat_math') or item['path'].startswith('libgen/mcat_math') or item['path'].startswith('libgen/math_for_dummies') or item['path'].startswith('other_books'):
            math = input(f"{item['path']}:(y/n) ")
            if math == 'y':
                kind = input(f"{item['path']}:(text/work) ")
                start = input(f"{item['path']}:(start page) ")
                end = input(f"{item['path']}:(end page) ")
                new_item['start'] = start
                new_item['end'] = end
                if kind == 'text':
                    text.append(new_item)
                elif kind == 'work':
                    work.append(new_item)
        elif item['path'].startswith('openstax_textbooks'):
            # all math
            kind = 'text'
            start = input(f"{item['path']}:(start page) ")
            end = input(f"{item['path']}:(end page) ")
            new_item['start'] = start
            new_item['end'] = end
            text.append(new_item)
        elif item['path'].startswith('openmathdep'):
            name = item['path'].split('/')[-1]
            url = ''
            for i in range(len(dep_names)):
                if name == dep_names[i]:
                    url = dep_urls[i]
                    break
            if 'algebra' in url or 'teaching' in url:
                math = input(f"{item['path']}:(y/n) ")
                if math == 'y':
                    kind = input(f"{item['path']}:(text/work) ")
                    start = input(f"{item['path']}:(start page) ")
                    end = input(f"{item['path']}:(end page) ")
                    new_item['start'] = start
                    new_item['end'] = end
                    if kind == 'text':
                        text.append(new_item)
                    elif kind == 'work':
                        work.append(new_item)

        with open(text_file, 'w') as f:
            json.dump(text, f, indent=4, ensure_ascii=False)

        with open(work_file, 'w') as f:
            json.dump(work, f, indent=4, ensure_ascii=False)



