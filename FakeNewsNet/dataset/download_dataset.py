import pandas as pd
from bs4 import BeautifulSoup
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from pathlib import Path
import logging


internet_archive_api_endpoint = 'https://archive.org/wayback/available?url='
#logfile = 'gossipcop_fake_test.log'
logfile = 'politifact_log_fake.log'

def make_request(url, stream=False, headers={}, retry_after=0.5):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=retry_after)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    response = session.get(url, stream=stream, headers=headers)
    return response

def request_html(url, article_fname, row):
    response = ''
    retries = 0
    headers = {
            'User-Agent': 'Internet-Archive-Bot/1.0.0'
    }
    #while response == '' and retries < 5:
    #    try:
    #        response = requests.get(row['news_url'], stream=True, headers=headers)
    #        break
    #    except requests.ConnectionError as conn_err:
    #        time.sleep(5)
    #        retries += 1
    try:
        response = make_request(url, stream=True, headers=headers)
    except requests.ConnectionError as conn_err:
        logging.error(f'error requesting {article_fname} for url {url} due to connection error handling')
        response = ''
    if response == '':
        logging.error(f'error requesting {article_fname} for url {url} as response is an empty string')
        row['article_filename'] = ''
    elif (response.status_code // 100) == 2:
        with open(article_fname, 'wb') as article_html:
            for chunk in response.iter_content(chunk_size=256):
                article_html.write(chunk)
        row['article_filename'] = article_fname
        logging.info(f'successfully downloaded {article_fname} for {url}')
    elif ((response.status_code // 4) == 4):
        response_headers = response.headers
        retry_after_time = response_headers['Retry-After']
        time.sleep(retry_after_time)
        response = make_request(url, stream=True, headers=headers, retry_after=retry_after_time)
        logging.info(f'successfully downloaded {article_fname} for {url} after a retry due to 400-range response code after sleeping for {retry_after_time} seconds')
        with open(article_fname, 'wb') as article_html:
            for chunk in response.iter_content(chunk_size=256):
                article_html.write(chunk)
        row['article_filename'] = article_fname
    else:
        logging.info(f'failed to download {article_fname} for {url} due to a response code in 300 range')
        row['article_filename'] = ''

def url_prefix(row):
    start_pos = 0
    m1 = re.search(r'(?<=http://).+', row['news_url'])
    m2 = re.search(r'(?<=https://).+', row['news_url'])
    if not (m1 or m2):
        row['news_url'] = 'https://' + row['news_url']
    return row

def download_dataset(row):
    row = url_prefix(row)
    article_fname = f"{row['id']}.html"
    article_fname_path = Path(f'./{article_fname}')
    retries = 5
    if (re.search(r'web\.archive\.org', row['news_url'])):
        if (row['article_filename'] == ''):
            request_html(row['news_url'], article_fname, row)
    elif (re.search(r'youtube\.com\/watch\?', row['news_url'])):
        logging.info(f'did not download {article_fname} because url is a youtube video link')
        row['article_filename'] = ''
    elif article_fname_path.exists():
        logging.info(f'did not download {article_fname} as it already exists locally')
        row['article_filename'] = article_fname
    else:
        api_url = internet_archive_api_endpoint + row['news_url']
        response = make_request(api_url)
        while (response.status_code == 429) and (retries > 0):
            response_headers = response.headers
            retry_after_time = response_headers.get('Retry-After')
            retry_after_time_lowercase = response_headers.get('retry-after')
            if retry_after_time_lowercase != None:
                time.sleep(retry_after_time_lowercase + 1)
            elif retry_after_time != None:
                time.sleep(retry_after_time + 1)
            if (retry_after_time == None) and (retry_after_time_lowercase == None):
                logging.error(f'failed to download {article_fname} as request to internet archive failed to get retry after header value')
                return row
            response = requests.get(api_url)
            retries -= 1
        status_invalid = (((response.status_code // 4) == 4) or (response.status_code == 203) or (response.status_code == 204))
        if status_invalid:
            logging.error(f'failed to download {article_fname} as request to internet archive api failed after 5 retries')
            row['article_filename'] = ''
        elif response.headers['content-type'].strip().startswith('application/json') == False:
            logging.error(f'failed to download {article_fname} as internet archive api did not return JSON')
            row['article_filename'] = ''
        else:
            json_response = response.json()
            if ((json_response.get('archived_snapshots') != {}) and json_response['archived_snapshots']['closest']['available']):
                #    if row['article_filename'] == '':
                html_archive_link = json_response['archived_snapshots']['closest']['url']
                request_html(html_archive_link, article_fname, row)
                if (row['article_filename'] == ''):
                    request_html(row['news_url'], article_fname, row)
    return row


if __name__ == '__main__':
    logging.basicConfig(
            filename=logfile,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt= '%Y-%m-%d %H:%M:%S',
            level = logging.INFO
    )
    #article_df = pd.read_csv('gossipcop_fake.csv')
    article_df = pd.read_csv('politifact_fake.csv')
    article_df = article_df.dropna(subset=['news_url', 'id'])
    article_df['news_url'] = article_df['news_url'].apply(str)
    #politifact_real.apply(url_prefix, axis=1)
    article_df['id'] = article_df['id'].apply(str)
    article_df['title'] = article_df['title'].apply(str)
    article_df['article_filename'] = ''
    article_df['article_filename'] = article_df['article_filename'].apply(str)
    article_df = article_df.apply(download_dataset, axis=1)
    article_df.to_csv('politifact_fake.csv')
    print('Done')
