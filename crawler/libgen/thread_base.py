"""
    Base pattern for crawlling libgen
"""

import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import json

BASE_URL = 'http://libgen.rs/'
SEARCH_URL = BASE_URL + 'search.php'
PARAM_PATTERN = {
    'open': 0,
    'view': 'simple',
    'phrase': 1,
    'column': 'def',
    'res': 25,     # max number of results
    'sort': 'def',
    'sortMode': 'ASC',
}

PROXIES = {'https': 'http://127.0.0.1:7890', 'http': 'http://127.0.0.1:7890'}

### Tool
# use backoff to help recover from exception
def backoff_hook(_details):
    tqdm.write("Retrying...")

import backoff
@backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=300, on_backoff=backoff_hook)
def get_page(url, params=None, **kwargs):
    """
        Get the page content
    """
    if params is None:
        return requests.get(url, proxies=PROXIES, **kwargs)
    else:
        return requests.get(url, params=params, proxies=PROXIES, **kwargs)
    

### Multi Thread Settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# tool: find total file number
def find_total_num(first_page: str):
    soup = BeautifulSoup(first_page, 'html.parser')
    try:
        # find the second table
        table = soup.find_all('table')[1]
        total_num = int(table.find('tr').find('td', align='left').find('font').text.split(' ')[0])
    except:
        total_num = 0
    return total_num

### Build index

def parse_mirror1(page_url):
    """
        Parse the library.lol mirror page for several download links
    """
    # print("parse " + page_url)
    # first h2 of download div is the get url
    # find id="download" div
    try:
        download_page = get_page(page_url)
        soup = BeautifulSoup(download_page.content, 'html.parser')
        download_div = soup.find('div', id='download')
    except:
        return None, None

    # find first h2
    try:
        get_url = download_div.find('h2').find('a')['href']
    except:
        get_url = None

    # cloudflare maybe available
    # first ul/li in download div is the cloudflare url
    try:
        cloudflare_url = download_div.find('ul').find('li').find('a')['href']
    except:
        cloudflare_url = None

    return get_url, cloudflare_url


def parse_mirror2(page_url):
    """
        Parse libgen.lc download page
    """
    # print("parse " + page_url)

    # find table/tbody/tr/td which align="center"/a
    try:
        download_page = get_page(page_url)
        soup = BeautifulSoup(download_page.content, 'html.parser')
        get_url = soup.find('table').find('tr').find('td', align='center').find('a')['href']
    except:
        get_url = None

    return get_url


def crawl_page(query, page_num):
    """
        Return the index
    """
    base_url = SEARCH_URL
    params = PARAM_PATTERN.copy()
    params['req'] = query

    params['page'] = page_num
    
    # Send a GET request to Libgen
    try:
        response = get_page(base_url, params=params)
    except:
        return []
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table containing the search results
    table = soup.find('table', class_='c')

    # indexes
    index = []
    
    if not table or not table.find_all('tr'):
        return index
    
    # Process each row in the table (except the header)
    for row in tqdm(table.find_all('tr')[1:], desc="Parsing book info"):
        columns = row.find_all('td')
        
        # Extract relevant information from the columns
        id = columns[0].text
        author = columns[1].text
        book_title = columns[2].text
        publisher = columns[3].text
        year = columns[4].text
        pages = columns[5].text
        language = columns[6].text
        size = columns[7].text
        extension = columns[8].text

        # library.lol
        download_page_1 = columns[9].find('a')['href']
        # libgen.lc
        download_page_2 = columns[10].find('a')['href']
        
        library_get_url, cloudflare_url = parse_mirror1(download_page_1)
        libgen_get_url = parse_mirror2(download_page_2)

        # not None urls
        urls = []
        if library_get_url is not None:
            urls.append(library_get_url)
        if cloudflare_url is not None:
            urls.append(cloudflare_url)
        if libgen_get_url is not None:
            urls.append(libgen_get_url)

        # Add the book to the list
        index.append({
            'id': id,
            'author': author,
            'book_title': book_title,
            'publisher': publisher,
            'year': year,
            'pages': pages,
            'language': language,
            'size': size,
            'extension': extension,
            'urls': urls,
            'download': False,      # to be set after download
        })

    return index


### Download files
LIBGEN_PREFIX = "https://libgen.rocks/"
def download_book(book, output_dir, cover=False):
    """
        Download the book from any of the urls
        - If starts with http, download directly and get the file name as the finally part
        - Else (get.php) add LIBGEN_PREFIX and download, use title as the file name
    """
    # tqdm.write("Downloading {}".format(book['book_title']))
    urls = book['urls']
    if len(urls) == 0:
        return False, book['id']
    name = book['book_title'] + '.' + book['extension']
    for url in urls:
        if url.startswith('http'):
            # download directly
            name = url.split('/')[-1]

    if not cover and os.path.exists(os.path.join(output_dir, name)):
        print(f'File {name} already exists.')
        return False, book['id']
    
    # download
    for url in urls:
        if url.startswith('http'):
            download_url = url
        else:
            # download from libgen.rocks
            download_url = LIBGEN_PREFIX + url
        try:
            r = get_page(download_url, allow_redirects=True)
            # if succeed, update the name from Content-Disposition
            if 'Content-Disposition' in r.headers.keys():
                name = r.headers['Content-Disposition'].split('filename=')[-1].strip('"')
            # tqdm.write(f'Saving {name}')
            open(os.path.join(output_dir, name), 'wb').write(r.content)
            return True, book['id']
        except:
            continue

    return False, book['id']


### Multi Thread downloading
def download_book_multi_threaded(index, output_dir, cover=False):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_book, book, output_dir, cover) for book in index if not book['download']]
        print(f'Waiting for {len(futures)} downloads to complete out of {len(index)}...')
        count = 0

        # Process the results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc='Downloading books'):
            success, id = future.result()
            if success:
                count += 1
                # set index
                for book in index:
                    if book['id'] == id:
                        book['download'] = True
                        break
    
    print(f'Downloaded {count} books out of {len(index)}.')
    # save index
    with open(os.path.join(output_dir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=4, ensure_ascii=False)


def crawl_libgen_multi_threaded(query, output_dir, index_file='index.json', cover=False, download=True):
    index = []
    # If has index file, load it
    if os.path.exists(os.path.join(output_dir, index_file)):
        print(f'Loading index file {index_file}...')
        with open(os.path.join(output_dir, index_file), 'r') as f:
            index = json.load(f)
            no_index = False
    else:
        no_index = True

    # Multi-threaded crawling
    # Get total files
    if no_index:
        base_url = SEARCH_URL
        params = PARAM_PATTERN.copy()
        params['req'] = query
        first_page = get_page(base_url, params=params).content
        total_num = find_total_num(first_page)
        num_pages = ((total_num + PARAM_PATTERN['res'] - 1) // PARAM_PATTERN['res'])
        print(f"Total number of books: {total_num}")
        print(f"Total number of pages: {num_pages}")
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(crawl_page, query, page_num) for page_num in range(1, num_pages + 1)]

            # Process the results as they complete
            for future in tqdm(as_completed(futures), desc='Crawling pages', total=len(futures)):
                page_index = future.result()
                index.extend(page_index)
    
    print(f'Parse {len(index)} books.')
    
    # Try to download each book

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if download:
        print(f'Saving books to output directory {output_dir}')
        download_book_multi_threaded(index, output_dir, cover)


### test
if __name__ == '__main__':
    query = 'Graduate Texts in Mathematics'
    output_dir = '/data/xukp/gtm_libgen_multi_threaded'
    crawl_libgen_multi_threaded(query, output_dir)
