import pandas as pd
from pathlib import Path
import json
import logging


logfile = 'text_lengths.log'


def find_text_length_gossipcop(row):
    row_id = row['id']
    row_id_components = row_id.split('-')
    folder_name = row_id_components[0] + '-' + row_id_components[1]
    json_path = ''
    if row_id.lower().endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/{folder_name}/news content.json'
    if row_id.lower().endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/{folder_name}/news content.json'
    with open(json_path) as json_file:
        json_file_text = json_file.read()
        json_dict = json.loads(json_file_text)
        text_exists = json_dict.get('text')
        text_content_exists = json_dict.get('text content')
        if text_exists:
            row['text_len'] = len(text_exists)
            text_length = len(text_exists)
            logging.info(f'text length for {row_id} is {text_length}')
        elif text_content_exists:
            row['text_len'] = len(text_content_exists)
            text_length = len(text_content_exists)
            logging.info(f'text length for {row_id} is {text_length}')
    return row


def find_text_length_politifact(row):
    row_id = row['id']
    row_id_components = row_id.split('-')
    folder_name = row_id_components[0]
    json_path = ''
    if row_id.lower().endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/real/{folder_name}/news content.json'
    if row_id.lower().endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/{folder_name}/news content.json'
    with open(json_path) as json_file:
        json_file_text = json_file.read()
        json_dict = json.loads(json_file_text)
        text_exists = json_dict.get('text')
        text_content_exists = json_dict.get('text content')
        if text_exists:
            row['text_len'] = len(text_exists)
            text_length = len(text_exists)
            logging.info(f'text length for {row_id} is {text_length}')
        elif text_content_exists:
            row['text_len'] = len(text_content_exists)
            text_length = len(text_content_exists)
            logging.info(f'text length for {row_id} is {text_length}')
    return row



if __name__ == '__main__':
    max_text_length = 0
    logging.basicConfig(
            filename=logfile,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt= '%Y-%m-%d %H:%M:%S',
            level = logging.INFO
    )
    politifact = pd.read_csv('politifact_filtered_by_json.csv')
    gossipcop = pd.read_csv('gossipcop_filtered_by_json.csv')
    politifact['text_len'] = 0
    gossipcop['text_len'] = 0
    gossipcop = gossipcop.apply(find_text_length_gossipcop, axis=1)
    politifact = politifact.apply(find_text_length_politifact, axis=1)
    max_len_gossipcop = gossipcop['text_len'].max()
    max_len_politifact = politifact['text_len'].max()
    print(f'Max text length in gossipcop is {max_len_gossipcop}')
    print(f'Max text length in politifact is {max_len_politifact}')
