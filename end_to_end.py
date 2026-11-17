import sys
import logging
import configparser
import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm





logfile='dimensions.log'
logging.basicConfig(
        filename=logfile,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt= '%Y-%m-%d %H:%M:%S',
        level = logging.INFO
)
config = configparser.ConfigParser()
config.read('hugging_face_token.cfg')
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

access_token = config['DEFAULT']['access_token']
logging.info(f'Gathered access token and using device {device}')
cache_dir = config['DEFAULT']['docker_cache_dir']
#cache_dir = config['DEFAULT']['gpu_cache_dir']
llama_model_name = config['DEFAULT']['docker_llama_name_path']
model_name = config['DEFAULT']['docker_model_name_path']
#model_name = config['DEFAULT']['gpu_model_name_path']
bert_tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=access_token)
bert_model = BertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, dtype='auto', device_map='auto')
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
bert_model.to(device)
logging.info(f'Loaded bert model and bert tokenizer, moved to {device}')
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=access_token)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    token=access_token,
    device_map='auto'
)
llama_model.to(device)
logging.info(f'Loaded llama model and llama tokenizer, moved to {device}')
prompt='''
You are analyzing text from a news article that has been scraped from a website. You aim to detect logical fallacies in an article and assign a score from 1 to 10 representing the frequency of a specific logical fallacy.

Example 1:
Article:
"Secret alien pyramids were found near antarctica. Hence, the pyramids were built by aliens."

Slippery Slope Fallacy: 10

Example 2:
Article:
"War is profitable for defense contractors. As a result, defense contractors often lobby congress to involve the US in various global conflicts due to the profit potential from selling arms to various countries' military forces."

Slippery Slope Fallacy: 2

Now analyze the following article:
{article}

Analysis:
'''




def encode_article(text):
    vector_encodings = None
    bert_model.eval()
    inputs = bert_tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=512
    ).to(bert_model.device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        vector_encodings = outputs.last_hidden_state[:,0,:]
    return vector_encodings


def query_and_encode_llama_output(text_fname):
    text = ''
    with open(text_fname, 'r') as text_file:
        text = text_file.read()
    full_prompt = prompt.format(text)
    inputs = llama_tokenizer(
        prompt,
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
    analysis = decoded[len(prompt):]
    encoded_analysis = encode_article(analysis)
    return encoded_analysis




if __name__ == '__main__':
    fname = sys.argv[-1]
    data = pd.read_csv(fname)
    article_ids = data.loc[1:3,'id']
    json_text = ''
    article_text = ''
    for article_id in article_ids:
        article_id = article_id.lower()
        article_id_components = article_id.split('-')
        id_name = ''
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
        with open(json_path) as json_file:
            json_text = json_file.read()
            json_dict = json.loads(json_text)
            article_text = json_dict.get('text')
            embeddings = encode_article(article_text)
            logging.info(f'embedding shape for article with id {article_id}: {embeddings.shape}')
    #with open(txt_filename, 'r') as text_file:
    #    text = text_file.read()
    #article_encoding = encode_article(text)
    #llm_output_encoding = query_and_encode_llama_output(txt_filename)
    #print(encodings)
