if __name__ == '__main__':
    import json
    import os
    from thread_base import crawl_libgen_multi_threaded
    
    OUTPUT_DIR = './data/xukp/libgen'
    with open('keywords.json', 'r') as f:
        keywords = json.load(f)
    for keyword in keywords:
        print('Downloading %s' % keyword)
        query = keyword
        output_dir = os.path.join(OUTPUT_DIR, keyword.replace(' ', '_'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        crawl_libgen_multi_threaded(keyword, output_dir, download=False)
        os.system('python batch_wget.py -i %s -q' % output_dir)



        