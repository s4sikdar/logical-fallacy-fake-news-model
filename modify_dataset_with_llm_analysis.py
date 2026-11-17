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
import argparse
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from tqdm import tqdm

from transformers import DistilBertTokenizer

from transformers import DistilBertForSequenceClassification, BertForSequenceClassification
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM, BigBirdTokenizer, BigBirdModel





# Load train data
#train_data = pd.read_csv('./Dataset/train.csv')

# Load test data
#test_data = pd.read_csv('./Dataset/test.csv')
config = configparser.ConfigParser()
config.read('hugging_face_token.cfg')
#model_name = config['DEFAULT']['docker_distilbert_model_path']
wandb_api_key = config['DEFAULT']['wandb_api_key']
access_token = config['DEFAULT']['access_token']
num_heads = int(config['DEFAULT']['num_heads'])
batch_size = int(config['DEFAULT']['batch_size'])
#learning_rate = float(config['DEFAULT']['learning_rate'])
learning_rate = 0.001
train_epochs = int(config['DEFAULT']['train_epochs'])
test_epochs = int(config['DEFAULT']['test_epochs'])

cache_dir = config['DEFAULT']['docker_cache_dir']
transformers_cache_dir = config['DEFAULT']['docker_transformers_cache_dir']
model_name = config['DEFAULT']['docker_model_name_path']
llama_model_name = config['DEFAULT']['docker_llama_name_path']
#train_ratio = 0.75
wandb_project = 'fake-news-detection-model'
wandb_config = {
    'epochs': train_epochs,
    'lr': learning_rate,
    'batch_size': batch_size,
    'train_ratio': 0.7
}
bert_model = BertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, dtype='auto', device_map='auto')
bert_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=access_token)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    token=access_token,
    device_map='auto'
)
llama_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
llm_prompt='''
You are analyzing text from a news article. Your goal is to evaluate the presence of three logical fallacies: 
1. Slippery slope fallacy
2. Ad hominem fallacy
3. Straw man fallacy

Score Guidelines:
- 0: Fallacy is entirely absent.
- 1-3: Low intensity.
- 4-7: Moderate intensity.
- 8-10: High intensity.

IMPORTANT: If an article contains no logical fallacies, assign a score of 0 to all three categories.

Return your analysis strictly in the following JSON format:
{{
  "slippery_slope_score": <integer 0-10>,
  "ad_hominem_score": <integer 0-10>,
  "straw_man_score": <integer 0-10>,
  "explanation": "<your explanation here>"
}}

Article:
{article}

JSON Analysis:
'''


updated_prompt='''
You are an expert NLP analyst analyzing text from a news article. Your goal is to evaluate the presence of three logical fallacies:
1. Slippery slope fallacy
2. Ad hominem fallacy
3. Straw man fallacy

Score Guidelines:
- 0: Fallacy is entirely absent.
- 1-3: Low intensity.
- 4-7: Moderate intensity.
- 8-10: High intensity.

IMPORTANT: If an article contains no logical fallacies, assign a score of 0 to all three categories.

CRITICAL OUTPUT REQUIREMENTS:
- Output ONLY valid raw JSON.
- Do NOT use Markdown formatting (do not use ``` or ```json).
- Do NOT include any conversational text, introductory statements, or explanations before or after the JSON.
- Output the JSON object exactly once and immediately terminate your response.

Required JSON Structure:
{{
  "slippery_slope_score": <integer 0-10>,
  "ad_hominem_score": <integer 0-10>,
  "straw_man_score": <integer 0-10>,
  "explanation": "<your explanation here>"
}}

Article:
{article}

JSON Analysis:
'''


def add_prompt_info(row):
    article_fname = row['article_filename']
    if not article_fname:
        return
    fname_components = article_fname.split('.')
    basename = fname_components[0]
    basename_components = basename.split('-')
    index = None
    if (basename.startswith('politifact')):
        id_name = basename_components[0]
        last_digit = id_name[-1]
        index = int(last_digit)
    else:
        index = int(base_name_components[1])
    id_name = ''
    if basename.startswith('gossipcop'):
        id_name = basename_components[0] + '-' + basename_components[1]
    else:
        id_name = basename_components[0]
    if id_name == '':
        logging.info(f'Failed to match {article_fname} to a valid json file')
        return
    json_path = ''
    if basename.startswith('gossipcop') and basename.endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/{id_name}/news content.json'
    elif basename.startswith('gossipcop') and basename.endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/{id_name}/news content.json'
    elif basename.startswith('politifact') and basename.endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/real/{id_name}/news content.json'
    elif basename.startswith('politifact') and basename.endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/{id_name}/news content.json'
    else:
        logging.info(f'could not fine json file for {id_name}')
        return
    fname_path = Path(json_path)
    if (fname_path.exists()):
        json_file_text = ''
        json_dict = {}
        with open(json_path, 'r') as json_file:
            json_file_text = json_file.read()
            json_dict = json.loads(json_file_text)
        original_article = json_dict.get('text')
        #article = re.sub(r'http[\w:/\.]+', ' ', original_article)
        #article = re.sub(r"[^a-z\s'’]", " ", article)
        #article = re.sub(r'\s\s+', ' ', article).strip()
        #article_analysis = json_dict.get('llm_analysis')
        #if article_analysis:
        #    valid_llm_response = re.search(
        #        r'(ad hominem fallacy: [0-9]|slippery slope fallacy: [0-9]|straw man fallacy: [0-9])',
        #        article_analysis,
        #        re.IGNORECASE
        #    )
        #    if not valid_llm_response:
        article_prompt = updated_prompt.format(article=original_article)
        inputs = llama_tokenizer(article_prompt, return_tensors='pt').to(llama_model.device)
        outputs = None
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
        analysis = decoded[len(article_prompt):]
        json_dict['json_format_llm_analysis'] = analysis
        updated_json_text = json.dumps(json_dict)
        with open(json_path, 'w') as json_file:
            json_file.write(updated_json_text)
        if ((index % 10) == 0):
            logging.info(f'added llm analysis for article with id {id_name}:\n{analysis}')
        else:
            logging.info(f'added llm analysis for article with id {id_name}')
    else:
        logging.info(f'Path {json_path} does not exist')


def log_llm_analysis(row):
    article_fname = row['article_filename']
    if not article_fname:
        return
    fname_components = article_fname.split('.')
    basename = fname_components[0]
    basename_components = basename.split('-')
    id_name = ''
    if basename.startswith('gossipcop'):
        id_name = basename_components[0] + '-' + basename_components[1]
    else:
        id_name = basename_components[0]
    if id_name == '':
        logging.info(f'Failed to match {article_fname} to a valid json file')
        return
    json_path = ''
    if basename.startswith('gossipcop') and basename.endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/{id_name}/news content.json'
    elif basename.startswith('gossipcop') and basename.endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/{id_name}/news content.json'
    elif basename.startswith('politifact') and basename.endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/real/{id_name}/news content.json'
    elif basename.startswith('politifact') and basename.endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/{id_name}/news content.json'
    else:
        logging.info(f'could not fine json file for {id_name}')
        return
    fname_path = Path(json_path)
    if (fname_path.exists()):
        json_file_text = ''
        json_dict = {}
        with open(json_path, 'r') as json_file:
            json_file_text = json_file.read()
            json_dict = json.loads(json_file_text)
        article_analysis = json_dict.get('llm_analysis')
        if article_analysis:
            #article = re.sub(r'http[\w:/\.]+', ' ', article)
            #article = re.sub(r"[^a-z\s'’]", " ", article)
            #article = re.sub(r'\s\s+', ' ', article).strip()
            logging.info(f'llm analysis for article with id {id_name}: {article_analysis}')
        else:
            logging.info(f'llm analysis for article {id_name} does not exist')
    else:
        logging.info(f'Path {json_path} does not exist')



def check_logical_fallacy_exists(row):
    article_fname = row['article_filename']
    if not article_fname:
        return
    fname_components = article_fname.split('.')
    basename = fname_components[0]
    basename_components = basename.split('-')
    id_name = ''
    if basename.startswith('gossipcop'):
        id_name = basename_components[0] + '-' + basename_components[1]
    else:
        id_name = basename_components[0]
    if id_name == '':
        logging.info(f'Failed to match {article_fname} to a valid json file')
        return
    json_path = ''
    if basename.startswith('gossipcop') and basename.endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/{id_name}/news content.json'
    elif basename.startswith('gossipcop') and basename.endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/{id_name}/news content.json'
    elif basename.startswith('politifact') and basename.endswith('true'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/real/{id_name}/news content.json'
    elif basename.startswith('politifact') and basename.endswith('false'):
        json_path = f'./FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/{id_name}/news content.json'
    else:
        logging.info(f'could not fine json file for {id_name}')
        return
    fname_path = Path(json_path)
    if (fname_path.exists()):
        json_file_text = ''
        json_dict = {}
        with open(json_path, 'r') as json_file:
            json_file_text = json_file.read()
            json_dict = json.loads(json_file_text)
        #article_analysis = json_dict.get('llm_analysis')
        article_analysis = json_dict.get('json_format_llm_analysis')
        article_text = json_dict.get('text')
        starting_index = 0
        ending_index = 0
        if article_analysis:
            #article = re.sub(r'http[\w:/\.]+', ' ', article)
            #article = re.sub(r"[^a-z\s'’]", " ", article)
            #article = re.sub(r'\s\s+', ' ', article).strip()
            #logging.info(f'Article analysis: {article_analysis}')
            #article_analysis_result = json.loads(article_analysis)
            #slippery_slope_score = article_analysis_result['slippery_slope_score']
            #ad_hominem_score = article_analysis_result['ad_hominem_score']
            #straw_man_score = article_analysis_result['straw_man_score']
            starting_index = article_analysis.find('{')
            if (starting_index != -1):
                ending_index = article_analysis.find('}', starting_index)
                if (ending_index != -1):
                    json_analysis_text = article_analysis[starting_index:(ending_index + 1)]
                    logging.info(f'analysis found for article with id {id_name}:\n{json_analysis_text}')
            #if (slippery_slope_score > 0) or (ad_hominem_score > 0) or (straw_man_score > 0):
            #article_match = re.search(
            #    r'("slippery_slope_score": [1-9]|"straw_man_score": [1-9]|"ad_hominem_score": [1-9]){1,}',
            #    article_analysis, re.IGNORECASE
            #)
            #gibberish_analysis_search = re.search(r'^\s*(slippery slope fallacy: __|slippery slope fallacy: \?\?)', article, re.IGNORECASE)
            #if (slippery_slope_score > 0) or (ad_hominem_score > 0) or (straw_man_score > 0):
            
            #if article_match:
            #    logging.info(f'logical fallacy detected for article {id_name}')
            #    logging.info(f'article text is: {article_text}')
            #    logging.info(f'article analysis is: {article_analysis}')
                #if basename.startswith('gossipcop'):
                #    gossipcop_logical_fallacy_count += 1
                #else:
                #    politifact_logical_fallacy_count += 1
                #logging.info(f'gibberish detected for article {id_name}, article below:\n{article_text}')
                #logging.info(f'gibberish detected for article {id_name}')
            #logging.info(f'llm analysis for article with id {id_name}: {article}')
        #else:
        #    logging.info(f'llm analysis for article {id_name} does not exist')
    #else:
    #    logging.info(f'Path {json_path} does not exist')




if __name__ == '__main__':
    #gossipcop_data = pd.read_csv('./gossipcop_filtered_by_json.csv')
    #logfile = 'prompt-response-hardcoding.log'
    logfile = 'prompt-response-check.log'
    #logfile = 'prompt-responses.log'
    #logfile = 'gibberish-response-check.log'
    politifact_logical_fallacy_count = 0
    gossipcop_logical_fallacy_count = 0
    logging.basicConfig(
            filename=logfile,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt= '%Y-%m-%d %H:%M:%S',
            level = logging.INFO
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'On device {device}')
    politifact_data = pd.read_csv('./politifact_filtered_by_json.csv')
    #gossipcop_data.dropna(inplace=True)
    politifact_data.dropna(inplace=True)
    #gossipcop_data.apply(check_logical_fallacy_exists, axis=1)
    politifact_data.apply(check_logical_fallacy_exists, axis=1)
    #logging.info(f'Logical fallacy count for politifact: {politifact_logical_fallacy_count}')
    #logging.info(f'Logical fallacy count for gossipcop: {gossipcop_logical_fallacy_count}')
