import sys
import configparser
import json
import logging
import numpy as np
import pandas as pd
import random
import torch
import wandb
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split



def encode_and_label_dataset(data_id):
    y_label = 0
    encodings = []
    labels = []
    max_length = 0
    word_counts = []
    for index, article_id in enumerate(data_id):
        article_id = article_id.lower()
        article_id_components = article_id.split('-')
        id_name = ''
        #label = data.loc[index, 'label']
        label = ''
        #logging.info(f'label for {article_id} - {label} with type {type(label)}')
        if article_id.endswith('true'):
            y_label = 1
            label = 'true'
        elif article_id.endswith('false'):
            y_label = 0
            label = 'false'
        else:
            print(f'article does not end with true or false')
            continue
        if article_id.startswith('gossipcop'):
            id_name = article_id_components[0] + '-' + article_id_components[1]
        else:
            id_name = article_id_components[0]
        if article_id.startswith('gossipcop') and article_id.endswith('true'):
            json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/{id_name}/news content.json'
        elif article_id.startswith('gossipcop') and article_id.endswith('false'):
            json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/{id_name}/news content.json'
        elif article_id.startswith('politifact') and article_id.endswith('true'):
            json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/real/{id_name}/news content.json'
        elif article_id.startswith('politifact') and article_id.endswith('false'):
            json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/{id_name}/news content.json'
        else:
            print(f'could not fine json file for {id_name}')
            continue
        with open(json_path) as json_file:
            json_text = json_file.read()
            json_dict = json.loads(json_text)
            article_text = json_dict.get('text')
            article_word_count = len(article_text.split())
            if article_word_count > max_length:
                max_length = article_word_count
            word_counts.append(article_word_count)
    histogram = pd.Series(word_counts).hist(bins = max_length, color='firebrick')
    plt.xlabel('Number of Words')
    plt.ylabel('Number of texts')
    histogram.get_figure().savefig('word_count_politifact.png', dpi=300)


if __name__ == '__main__':
    #fname = sys.argv[-1]
    data = pd.read_csv('politifact_filtered_by_json.csv')
    # limit to 1000 instances for now
    #data = data.loc[0:499]
    #article_ids = data['id']
    json_text = ''
    article_text = ''
    label = ''
    y_label = 0
    article_ids = data['id']
    encode_and_label_dataset(article_ids)
