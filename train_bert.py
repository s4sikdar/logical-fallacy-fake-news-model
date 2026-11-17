import sys
import re
import configparser
import json
import logging
import numpy as np
import pandas as pd
import random
import torch
import wandb
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM, BigBirdTokenizer, BigBirdModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import set_seed, true_positive, false_positive, true_negative, false_negative
from end_neural_network import AttentionNeuralNetwork, BERT_Arch



SEED = 42
set_seed(SEED)

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
#device = 'cpu'
#torch.set_default_device(device)
access_token = config['DEFAULT']['access_token']
wandb_api_key = config['DEFAULT']['wandb_api_key']
logging.info(f'Gathered access token and using device {device}')
cache_dir = config['DEFAULT']['docker_cache_dir']
#cache_dir = config['DEFAULT']['gpu_cache_dir']
llama_model_name = config['DEFAULT']['docker_llama_name_path']
#llama_model_name = config['DEFAULT']['gpu_llama_name_path']
#model_name = config['DEFAULT']['docker_model_name_path']
#model_name = config['DEFAULT']['gpu_model_name_path']
model_name = config['DEFAULT']['docker_big_bird_model_path']
#embedding_dimensions = int(config['DEFAULT']['embedding_dimension'])
embedding_dimensions = 768
hidden_size = int(config['DEFAULT']['hidden_size'])
output_size = int(config['DEFAULT']['output_size'])
num_heads = int(config['DEFAULT']['num_heads'])
bert_tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=access_token)
bert_model = BertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, dtype='auto', device_map='auto')
bert_model.to(device)
bert_nn_model = BERT_Arch(bert_model)
optimizer = AdamW(bert_nn_model.parameters(), lr=1e-5)
cross_entropy = nn.NLLLoss()
for param in bert_model.parameters():
    param.requires_grad = False
logging.info(f'Loaded bert model and bert tokenizer, moved to {device}')


batch_size = 32
epochs = 20
learning_rate = 0.001
train_ratio = 0.75
wandb_project = 'fake-news-detection-model'
wandb_config = {
    'epochs': 10,
    'lr': 0.001,
    'batch_size': 128,
    'train_ratio': 0.75
}


def encode_and_label_dataset(data_id):
    y_label = 0
    texts = []
    labels = []
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
            logging.info(f'article does not end with true or false')
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
            logging.info(f'could not fine json file for {id_name}')
            continue
        with open(json_path) as json_file:
            json_text = json_file.read()
            json_dict = json.loads(json_text)
            article_text = json_dict.get('text')
            article_text = article_text.lower()
            article_text = re.sub(r'http[\w:/\.]+', ' ', article_text)
            article_text = re.sub(r"[^a-z\s'’]", " ", article_text)
            article_text = re.sub(r'\s\s+', ' ', article_text).strip()
            texts.append(article_text)
            labels.append(y_label)
            #if ((index % 5) == 0):
            #    logging.info(f'article text:\n{article_text}')
    return texts, labels


def tokenize(texts):
    tokens = []
    attention_masks = []
    num_texts = len(texts)
    for text in texts:
        #inputs = bert_tokenizer(
        #    text,
        #    return_tensors='pt',
        #    truncation=True,
        #    padding=True,
        #    max_length=512,
        #    return_overflowing_tokens=True,
        #    stride=128
        #)
        tokenized_text = bert_tokenizer(
            text,
            max_length=4096,
            padding='max_length',
            truncation=True,
            pad_to_max_length=True,
            #return_overflowing_tokens=True,
            #stride=128
        )
        #tokenized_text = np.array(tokenized_text)
        tokenized_text_ids = np.array(tokenized_text['input_ids'])
        attention_mask = np.array(tokenized_text['attention_mask'])
        tokenized_text_ids = torch.from_numpy(np.reshape(tokenized_text_ids, (-1)))
        attention_mask = torch.from_numpy(np.reshape(attention_mask, (-1)))
        #logging.info(f'tokenized text shape: {tokenized_text_ids.shape}')
        logging.info(f'tokenized text shape: {tokenized_text_ids.shape}')
        logging.info(f'attention mask shape: {attention_mask.shape}')
        #logging.info(f'attention mask shape: {attention_mask.shape}')
        tokens.append(tokenized_text_ids)
        attention_masks.append(attention_mask)
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    padded_tensor_length = tokens.shape[-1]
    num_rows_per_text = padded_tensor_length // 512 # max input length of 512 for bert
    tokens = tokens.reshape(num_texts, num_rows_per_text, -1)
    attention_masks = attention_masks.reshape(num_texts, num_rows_per_text, -1)
    logging.info(f'token shape: {tokens.shape}')
    logging.info(f'attention_masks shape: {attention_masks.shape}')
    #logging.info(f'token: {tokens}')
    return (tokens, attention_masks)


def train(model, dataloader):
    model.train()
    total_loss = 0
    total_accuracy = 0
    #predictions = []
    step = 0
    for batch in dataloader:
        if ((step % 50) == 0) and (not (step == 0)):
            logging.info(f'Batch {step} of {len(dataloader)}')
        #logging.info(f'batch component lengths: {batch[0].shape} {batch[1].shape} {batch[2].shape}')
        token_ids, mask, labels = batch
        token_ids = token_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        #batch = [r for r in batch]
        model.zero_grad()
        index = 0
        for token_set in token_ids:
            #logging.info(f'token id shape: {len(token_set)}')
            #logging.info(f'mask shape: {mask[index].shape}')
            #logging.info(f'labels shape: {labels[index].shape}')
            #token_set = token_set.reshape((1, len(token_set))).to(device)
            #current_mask = mask[index].reshape((1, len(mask[index]))).to(device)
            current_mask = mask[index].to(device)
            #logging.info(f'token device: {token_set.get_device()}')
            #logging.info(f'mask device: {current_mask.get_device()}')
            #logging.info(f'model device: {model.get_device()}')
            preds = model(token_set, current_mask).to(device)
            target_label = labels[index].reshape((1)).to(device)
            #logging.info(f'predictions: {preds}')
            #logging.info(f'preds shape: {preds.shape}')
            #logging.info(f'preds type: {type(preds)}')
            #logging.info(f'labels shape: {target_label.shape}')
            #logging.info(f'labels type: {type(target_label)}')
            loss = cross_entropy(preds, target_label)
            total_loss = total_loss + loss.item()
            index += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        #predictions = predictions.detach().cpu().numpy()

    #avg_loss = total_loss / (len(dataloader) * batch)
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader):
    model.eval()
    total_loss, total_accuracy = 0, 0
    step = 0
    for batch in dataloader:
        if ((step % 50) == 0) and (not (step == 0)):
            logging.info(f'Batch {step} of {len(dataloader)}')
        batch = [t for t in batch]
        token_ids, mask, labels = batch
        with torch.no_grad():
            index = 0
            for token_set in token_ids:
                #token_set = token_set.reshape((1, len(token_set))).to(device)
                #current_mask = mask[index].reshape((1, len(mask[index]))).to(device)
                current_mask = mask[index].to(device)
                token_set = token_set.to(device)
                predictions = model(token_set, current_mask).to(device)
                target_label = labels[index].reshape((1)).to(device)
                loss = cross_entropy(predictions, target_label)
                total_loss = total_loss + loss.item()
                #predictions = predictions.detach().cpu().numpy()
                index += 1
    avg_loss = total_loss / len(dataloader)
    return avg_loss


if __name__ == '__main__':
    fname = sys.argv[-1]
    data = pd.read_csv(fname)
    # limit to 1000 instances for now
    #data = data.loc[0:499]
    article_ids = data['id']
    json_text = ''
    article_text = ''
    label = ''
    y_label = 0
    data_true = data[data['label'] == True]
    data_false = data[data['label'] == False]
    data_true = data_true[:100]
    data_false = data_false[:100]
    #data_true = data_true[:2000]
    #data_false = data_false[:2000]
    data_sample = pd.concat([data_true, data_false], ignore_index=True)
    data_sample = data_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    article_ids = data_sample['id']
    texts, labels = encode_and_label_dataset(article_ids)
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        random_state=2018,
        test_size=0.3,
        stratify=labels
    )
    validation_text, test_text, validation_labels, test_labels = train_test_split(
        temp_text,
        temp_labels,
        random_state=2018,
        test_size=0.5,
        stratify=temp_labels
    )
    #logging.info(f'amount of training texts: {len(train_text)}')
    train_text_tokens, train_text_masks = tokenize(train_text)
    validate_text_tokens, validate_text_masks = tokenize(validation_text)
    test_text_tokens, test_text_masks = tokenize(test_text)
    test_text_tokens = test_text_tokens.to(device)
    test_text_masks = test_text_masks.to(device)
    train_y = torch.tensor(train_labels)
    validate_y = torch.tensor(validation_labels)
    test_y = torch.tensor(test_labels)
    batch_size = 32
    train_data = TensorDataset(train_text_tokens, train_text_masks, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validate_text_tokens, validate_text_masks, validate_y)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler = validation_sampler, batch_size=batch_size)
    test_data = TensorDataset(test_text_tokens, test_text_masks, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    epochs = 10
    best_valid_loss = float('inf')
    train_losses = []
    validation_losses = []
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} of {epochs}')
        logging.info(f'Epoch {epoch + 1} of {epochs}')
        train_loss = train(bert_nn_model, train_dataloader)
        #logging.info(f'train loss: {train_loss}')
        validation_loss = evaluate(bert_nn_model, validation_dataloader)
        if (validation_loss < best_valid_loss):
            best_valid_loss = validation_loss
            torch.save(bert_nn_model.state_dict(), f'c2_new_model_weights.pt')
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print(f'Training loss: {train_loss}')
        logging.info(f'Training loss: {train_loss}')
        print(f'Validation loss: {validation_loss}')
        logging.info(f'Validation loss: {validation_loss}')

    test_predictions = []
    test_labels = []
    index = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = [t for t in batch]
            token_ids, mask, labels = batch
            with torch.no_grad():
                index = 0
                for token_set in token_ids:
                    token_set = token_set.to(device)
                    current_mask = mask[index].to(device)
                    predictions = bert_nn_model(token_set, current_mask).to(device)
                    target_label = labels[index].reshape((1)).to(device)
                    test_predictions.append(predictions)
                    test_labels.append(target_label)
                    #loss = cross_entropy(predictions, target_label)
                    #total_loss = total_loss + loss.item()
                    #predictions = predictions.detach().cpu().numpy()
                    index += 1
        #predictions = predictions.detach().cpu().numpy()
        test_predictions = np.array(test_predictions)
        test_labels = np.array(test_labels)
        #logging.info(f'test_text_tokens dimension: {test_text_tokens.shape}')
        logging.info(f'test_predictions dimension: {test_predictions.shape}')
        logging.info(f'test_labels dimension: {test_labels.shape}')

    #predictions = np.argmax(predictions, axis = 1)
    print(classification_report(test_labels, test_predictions))
    logging.info(f'Classification report:')
    logging.info(classification_report(test_y, predictions))
