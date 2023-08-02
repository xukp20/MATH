if __name__ == '__main__':
    import argparse
    import json
    import os
    from thread_base_category import crawl_libgen_category_multi_threaded

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--i', type=int, default=0)
    args = parser.parse_args()
    i = args.i

    OUTPUT_DIR = '/data/xukp/libgen_category'
    with open('categories.json', 'r') as f:
        categories = json.load(f)

    # for i in categories:
    keyword = categories[i]['category']
    topic_id = categories[i]['topicid']
    print('Downloading %s' % keyword)
    crawl_libgen_category_multi_threaded(keyword, topic_id, OUTPUT_DIR)

    print("Finish downloading indexes")

    # for i in categories:
    keyword = categories[i]['category']
    topic_id = categories[i]['topicid']
    output_dir = os.path.join(OUTPUT_DIR, keyword)
    os.system('python batch_wget.py -i "%s" -q' % output_dir)



        