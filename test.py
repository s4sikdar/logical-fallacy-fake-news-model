import sys
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
import configparser
import os
import random
from utils import true_positive, false_positive, true_negative, false_negative





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
#transformers_cache_dir = config['DEFAULT']['docker_transformers_cache_dir']
#os.environ['TRANSFORMERS_CACHE'] = transformers_cache_dir
#device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
#TRANSFORMERS_CACHE
#access_token = config['DEFAULT']['access_token']
#logging.info(f'Gathered access token and using device cpu in test.py')
#cache_dir = config['DEFAULT']['gpu_cache_dir']
#cache_dir = config['DEFAULT']['docker_cache_dir']
#llama_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
#model_name = 'distilbert-base-uncased'
#logging.info(f'Set environment variable TRANSFORMERS_CACHE')
#bert_tokenizer = DistilBertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=access_token)
#bert_tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, low_cpu_mem_usage=True, torch_dtype='auto')
#logging.info(f'Dowloaded BERT tokenizer')
#bert_model = DistilBertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, dtype='auto', device_map='auto')
#bert_model = DistilBertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, low_cpu_mem_usage=True, torch_dtype='auto')
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
#bert_model.to(device)
#logging.info(f'Loaded bert model and bert tokenizer, moved to {device}')

label_tensor = torch.zeros(10, 2)
pred_tensor = torch.zeros(10, 2)
ones_tensor = torch.ones(10)

for index in range(10):
    label_tensor[index, random.randint(0, 1)] = 1
    pred_tensor[index, random.randint(0, 1)] = 1

logging.info(f'label tensor: {label_tensor}')
logging.info(f'random tensor: {pred_tensor}')
logging.info(f'ones tensor: {ones_tensor}')

#both_tensors_with_same_one_predictions = torch.argmax(label_tensor, dim=1) == torch.argmax(pred_tensor, dim=1)
#label_tensor_with_one_prediction = torch.argmax(label_tensor, dim=1) == ones_tensor
#correct_ones_predictions = torch.logical_and(both_tensors_with_same_one_predictions, label_tensor_with_one_prediction)
#logging.info(f'boolean tensor 1: {both_tensors_with_same_one_predictions}')
#logging.info(f'boolean tensor 2: {label_tensor_with_one_prediction}')
#correctly_predicted_ones_second_column = correct_ones_predictions.type(torch.float).sum().item()
#correct_ones_predictions = (both_tensors_with_same_one_predictions & label_tensor_with_one_prediction)
#logging.info(f'ones in second column for both pred tensor and label tensor: {correct_ones_predictions}')
#correctly_predicted_ones_labels = correct_ones_predictions.type(torch.float).sum().item()
#logging.info(f'number of times with 1 in both pred tensor and label tensor: {correctly_predicted_ones_labels}')
tp = true_positive(label_tensor, pred_tensor)
fp = false_positive(label_tensor, pred_tensor)
tn = true_negative(label_tensor, pred_tensor)
fn = false_negative(label_tensor, pred_tensor)
logging.info(f'true positive: {tp}')
logging.info(f'false positive: {fp}')
logging.info(f'true negative: {tn}')
logging.info(f'false negative: {fn}')
