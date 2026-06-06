import sys
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
import configparser
import os





logfile='debug.log'
logging.basicConfig(
        filename=logfile,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt= '%Y-%m-%d %H:%M:%S',
        level = logging.INFO
)
config = configparser.ConfigParser()
config.read('hugging_face_token.cfg')
#transformers_cache_dir = config['DEFAULT']['gpu_transformers_cache_dir']
transformers_cache_dir = config['DEFAULT']['docker_transformers_cache_dir']
os.environ['TRANSFORMERS_CACHE'] = transformers_cache_dir
#device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
#TRANSFORMERS_CACHE
access_token = config['DEFAULT']['access_token']
logging.info(f'Gathered access token and using device cpu in test.py')
#cache_dir = config['DEFAULT']['gpu_cache_dir']
cache_dir = config['DEFAULT']['docker_cache_dir']
llama_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_name = 'distilbert-base-uncased'
logging.info(f'Set environment variable TRANSFORMERS_CACHE')
#bert_tokenizer = DistilBertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=access_token)
bert_tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, low_cpu_mem_usage=True, torch_dtype='auto')
logging.info(f'Dowloaded BERT tokenizer')
#bert_model = DistilBertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, dtype='auto', device_map='auto')
bert_model = DistilBertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, low_cpu_mem_usage=True, torch_dtype='auto')
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
#bert_model.to(device)
logging.info(f'Loaded bert model and bert tokenizer, moved to {device}')

