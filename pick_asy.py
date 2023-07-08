"""
    To pick out the problems with asy command
"""

PATH = './MATH'
OUTPUT_DIR = './asy'
if __name__ == "__main__":
    # go through all the files to find [asy]
    import os, json
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # train and test in MATH
    count = {'train': None, 'test': None}
    total = {'train': 0, 'test': 0}
    sets = ['train', 'test']
    for set in sets:
        count[set] = {}
        total[set] = {}
        dataset_path = os.path.join(PATH, set)
        output_path = os.path.join(OUTPUT_DIR, set)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # go through all the categories
        categories = os.listdir(dataset_path)
        for category in categories:
            category_path = os.path.join(dataset_path, category)
            output_category_path = os.path.join(output_path, category)
            count[set][category] = 0
            total[set][category] = len(os.listdir(category_path))
            if not os.path.exists(output_category_path):
                os.mkdir(output_category_path)

            # go through all the problems
            for json_file in os.listdir(category_path):
                json_path = os.path.join(category_path, json_file)
                output_json_path = os.path.join(output_category_path, json_file.strip('.json') + '.asy')
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # find the problems with [asy]
                if '[asy]' in data['problem']:
                    # get the content between [asy] and [/asy]
                    start = data['problem'].find('[asy]')
                    end = data['problem'].find('[/asy]')
                    asy_content = data['problem'][start+5:end]
                    # write the content to a file
                    with open(output_json_path, 'w') as f:
                        f.write(asy_content.strip())
                    count[set][category] += 1

        print('set: {}\n count: {}\n total: {}\n'.format(set, count[set], total[set]))
    
    # count asy number and percentage in train and test
    for set in sets:
        print('set: {}'.format(set))
        temp = 0
        tot = 0
        for category in count[set]:
            temp += count[set][category]
            tot += total[set][category]

        print('count: {}\n total: {}\n percentage: {}\n'.format(temp, tot, temp/tot))
    


    