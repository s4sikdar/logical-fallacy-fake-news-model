import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import contractions
import wandb
import logging
import configparser
import json
import re
import os

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from tqdm import tqdm

from transformers import DistilBertTokenizer

from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, ModernBertForSequenceClassification, AutoModelForCausalLM
from transformers import BertTokenizer, BertModel, AutoTokenizer, ModernBertModel, BigBirdTokenizer, BigBirdModel, AutoModelForMaskedLM


class Politifact(Dataset):
    def __init__(self, text_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv('politifact_filtered.csv')
        self.text_dir = text_dir
        sefl.transform = transform
        sefl.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        fname = self.labels.iloc[index]['article_filename']
        label = self.labels.iloc[index]['label']
        fname_components = fname.split('.')
        basename = fname_components[0]
        txt_path = ''
        if basename.endswith('true'):
            txt_path = f'./FakeNewsNet/dataset/politifact_real/{basename}.txt'
        if basename.endswith('false'):
            txt_path = f'./FakeNewsNet/dataset/politifact_fake/{basename}.txt'
        text = ''
        with open(txt_path) as article_text:
            text = article_text.read()
        return text, label



class Gossipcop(Dataset):
    def __init__(self, text_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv('gossipcop_filtered.csv')
        self.text_dir = text_dir
        sefl.transform = transform
        sefl.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        fname = self.labels.iloc[index]['article_filename']
        label = self.labels.iloc[index]['label']
        fname_components = fname.split('.')
        basename = fname_components[0]
        txt_path = ''
        if basename.endswith('true'):
            txt_path = f'./FakeNewsNet/dataset/gossipcop_real/{basename}.txt'
        if basename.endswith('false'):
            txt_path = f'./FakeNewsNet/dataset/gossipcop_fake/{basename}.txt'
        text = ''
        with open(txt_path) as article_text:
            text = article_text.read()
        return text, label




class FakeNewsDataset(Dataset):
    ''' This class takes in the data, tokenizes it using the DistilBertTokenizer from the transformers library,
     and returns the input IDs, attention masks, and labels.'''
    def __init__(self, data, max_len=128):
        self.data = data
        self.max_len = max_len
        #self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), torch.tensor(label, dtype=torch.long)




class FakeNewsNetDataset(Dataset):
    '''
        extension of the Dataset class in Pytorch for the FakeNewsNet dataset
    '''
    #def __init__(self, data, max_len=4096):
    #def __init__(self, data, max_len=1024):
    def __init__(self, data, model_name, llama_model, llama_tokenizer, prompt, max_len=512, log_info=False):
        self.data = data
        self.max_len = max_len
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.llama_model = llama_model
        self.llama_tokenizer = llama_tokenizer
        self.prompt = updated_prompt
        self.log_info = log_info


    def __len__(self):
        return len(self.data)


    def query_and_encode_llama_output(self, text, log_info=False, index=None):
        #text = ''
        #with open(text_fname, 'r') as text_file:
        #    text = text_file.read()
        full_prompt = self.prompt.format(article=text)
        inputs = self.llama_tokenizer(
            full_prompt,
            return_tensors='pt',
        ).to(llama_model.device)
        with torch.no_grad():
            outputs = llama_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                do_sample=True
            )
        decoded = llama_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        analysis = decoded[len(full_prompt):]
        if (log_info):
            logging.info(f'Index: {index}')
            logging.info(f'Article text:\n{text}')
            logging.info(f'Response (logical fallacy score out of 10):\n{analysis}')
        tokenized_response = self.tokenizer(
            analysis,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return tokenized_response['input_ids'].squeeze(0), tokenized_response['attention_mask'].squeeze(0)



    def __getitem__(self, index):
        record = self.data.iloc[index]
        record_id = self.data.iloc[index]['id']
        record_id = record_id.lower()
        record_id_components = record_id.split('-')
        y_label = 0
        label = self.data.iloc[index]['label']
        id_name = ''
        json_path = ''
        if record_id.endswith('true'):
            y_label = 1
            label = 'true'
        #elif record_id.endswith('false'):
        else:
            y_label = 0
            label = 'false'
        if record_id.startswith('gossipcop'):
            id_name = record_id_components[0] + '-' + record_id_components[1]
        else:
            id_name = record_id_components[0]
        if record_id.startswith('gossipcop') and record_id.endswith('true'):
            json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/{id_name}/news content.json'
        elif record_id.startswith('gossipcop') and record_id.endswith('false'):
            json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/{id_name}/news content.json'
        elif record_id.startswith('politifact') and record_id.endswith('true'):
            json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/real/{id_name}/news content.json'
        elif record_id.startswith('politifact') and record_id.endswith('false'):
            json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/{id_name}/news content.json'
        article_text = ''
        with open(json_path) as json_file:
            json_text = json_file.read()
            json_dict = json.loads(json_text)
            article_text = json_dict.get('text')
            article_text = article_text.lower()
            article_text = re.sub(r'http[\w:/\.]+', ' ', article_text)
            article_text = re.sub(r"[^a-z\s'’:;\"]", " ", article_text)
            article_text = re.sub(r'\s\s+', ' ', article_text).strip()
            article_text = contractions.fix(article_text)
        inputs = self.tokenizer(
            article_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if (((index % 100) == 0) and self.log_info):
            tokenized_llama_output_input_ids, tokenized_llama_output_attn_mask = self.query_and_encode_llama_output(
                article_text, log_info=False, index=index
            )
        else:
            tokenized_llama_output_input_ids, tokenized_llama_output_attn_mask = self.query_and_encode_llama_output(
                article_text, log_info=False, index=None
            )
#            tokenized_article = inputs['input_ids'].squeeze(0)
#            tokenized_article_attn_mask = inputs['attention_mask'].squeeze(0)
#            logging.info(
#    f'tokenized_llama_output_input_ids dimension, tokenized_article dimension for index {index}: {tokenized_llama_output_input_ids.shape}, {tokenized_article.shape}'
#            )
#            logging.info(
#    f'tokenized_llama_output_attn_mask dimension, tokenized_article_attn_mask dimension for index {index}: {tokenized_llama_output_attn_mask.shape}, {tokenized_article_attn_mask.shape}'
#            )
        #return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), torch.tensor(y_label, dtype=torch.long)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), tokenized_llama_output_input_ids, tokenized_llama_output_attn_mask, torch.tensor(y_label, dtype=torch.long)

