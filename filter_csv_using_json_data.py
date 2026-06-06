import pandas as pd
from pathlib import Path
import json
import logging

logfile='json_filter_csv.log'


def assign_label(row):
    row_fname = row['id']
    label = 'False'
    if row_fname.endswith('true'):
        label = 'True'
    if row_fname.endswith('false'):
        label = 'False'
    row['label'] = label
    return row


def gossipcop_filter_valid_json(row):
    row_fname = row['article_filename']
    if not row_fname:
        return False
    fname_components = row_fname.split('.')
    basename = fname_components[0]
    basename_components = basename.split('-')
    id_name = basename_components[0] + '-' + basename_components[1]
    json_path = ''
    if basename.endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/{id_name}/news content.json'
    if basename.endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/{id_name}/news content.json'
    if json_path == '':
        return False
    fname_path = Path(json_path)
    if fname_path.exists():
        json_file = open(json_path, 'r')
        json_file_text = json_file.read()
        json_dict = json.loads(json_file_text)
        json_file.close()
        text_exists = json_dict.get('text')
        text_content_exists = json_dict.get('text content')
        if text_exists or text_content_exists:
            return True
        return False
    else:
        logging.info(f'Path {json_path} does not exist')
        return False


def politifact_filter_valid_json(row):
    row_fname = row['article_filename']
    if not row_fname:
        return False
    fname_components = row_fname.split('.')
    basename = fname_components[0]
    basename_components = basename.split('-')
    id_name = basename_components[0]
    json_path = ''
    if basename.endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/real/{id_name}/news content.json'
    if basename.endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/{id_name}/news content.json'
    if json_path == '':
        return False
    fname_path = Path(json_path)
    if fname_path.exists():
        json_file = open(json_path, 'r')
        json_file_text = json_file.read()
        json_dict = json.loads(json_file_text)
        json_file.close()
        text_exists = json_dict.get('text')
        text_content_exists = json_dict.get('text content')
        if text_exists or text_content_exists:
            return True
        return False
    else:
        logging.info(f'Path {json_path} does not exist')
        return False


if __name__ == '__main__':
    logging.basicConfig(
            filename=logfile,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt= '%Y-%m-%d %H:%M:%S',
            level = logging.INFO
    )
    gossipcop = pd.read_csv('gossipcop.csv')
    politifact = pd.read_csv('politifact.csv')
    gossipcop['label'] = ''
    politifact['label'] = ''
    gossipcop = gossipcop.loc[gossipcop.apply(gossipcop_filter_valid_json, axis=1)]
    politifact = politifact.loc[politifact.apply(politifact_filter_valid_json, axis=1)]
    gossipcop = gossipcop.apply(assign_label, axis=1)
    politifact = politifact.apply(assign_label, axis=1)
    gossipcop = gossipcop[['id', 'label', 'article_filename']]
    politifact = politifact[['id', 'label', 'article_filename']]
    gossipcop.to_csv('gossipcop_filtered_by_json.csv')
    politifact.to_csv('politifact_filtered_by_json.csv')
