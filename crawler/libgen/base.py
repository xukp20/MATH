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
    'res': 100,     # max number of results
    'sort': 'def',
    'sortMode': 'ASC',
}



### Build index

def parse_mirror1(page_url):
    """
        Parse the library.lol mirror page for several download links
    """
    # first h2 of download div is the get url
    download_page = requests.get(page_url)
    soup = BeautifulSoup(download_page.content, 'html.parser')
    # find id="download" div
    try:
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
    download_page = requests.get(page_url)
    soup = BeautifulSoup(download_page.content, 'html.parser')
    # find table/tbody/tr/td which align="center"/a
    try:
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
    response = requests.get(base_url, params=params)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table containing the search results
    table = soup.find('table', class_='c')

    # indexes
    index = []
    
    if not table or not table.find_all('tr'):
        return index
    
    # Process each row in the table (except the header)
    for row in table.find_all('tr')[1:]:
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
    print(book)
    urls = book['urls']
    if len(urls) == 0:
        return False
    name = book['book_title'] + '.' + book['extension']
    for url in urls:
        if url.startswith('http'):
            # download directly
            name = url.split('/')[-1]

    if not cover and os.path.exists(os.path.join(output_dir, name)):
        print(f'File {name} already exists.')
        return False
    
    # download
    for url in urls:
        if url.startswith('http'):
            download_url = url
        else:
            # download from libgen.rocks
            download_url = LIBGEN_PREFIX + url
        try:
            print("requesting {}...".format(download_url))
            r = requests.get(download_url, allow_redirects=True)
            # if succeed, update the name from Content-Disposition
            if 'Content-Disposition' in r.headers.keys():
                name = r.headers['Content-Disposition'].split('filename=')[-1].strip('"')
            open(os.path.join(output_dir, name), 'wb').write(r.content)
            return True
        except:
            continue

    return False


def crawl_libgen(query, output_dir, index_file='index.json', cover=False, download=True):
    # Go through all the pages until index is empty
    page_num = 1
    index = []
    while True:
        print(f'Crawling page {page_num}...')
        query = query.replace(' ', '+')
        page_index = crawl_page(query, page_num)
        if len(page_index) == 0:
            break
        index.extend(page_index)
        page_num += 1
    
    print(f'Found {len(index)} books from {page_num-1} pages.')
    
    # Try to download each book
    print(f'Saving books to output directory {output_dir}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for book in tqdm(index, desc='Downloading books'):
        # success = download_book(book, output_dir)
        success = True
        if success:
            count += 1
            book['download'] = True
        
    print(f'Downloaded {count} out of {len(index)} books.')

    # Save the index
    with open(os.path.join(output_dir, index_file), 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


### test
if __name__ == '__main__':
    query = 'Graduate Texts in Mathematics'
    output_dir = './output'
    crawl_libgen(query, output_dir)
