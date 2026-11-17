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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import set_seed, true_positive, false_positive, true_negative, false_negative
from end_neural_network import AttentionNeuralNetwork



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


access_token = config['DEFAULT']['access_token']
wandb_api_key = config['DEFAULT']['wandb_api_key']
logging.info(f'Gathered access token and using device {device}')
cache_dir = config['DEFAULT']['docker_cache_dir']
#cache_dir = config['DEFAULT']['gpu_cache_dir']
llama_model_name = config['DEFAULT']['docker_llama_name_path']
#llama_model_name = config['DEFAULT']['gpu_llama_name_path']
model_name = config['DEFAULT']['docker_model_name_path']
#model_name = config['DEFAULT']['gpu_model_name_path']
#embedding_dimensions = int(config['DEFAULT']['embedding_dimension'])
embedding_dimensions = 768
hidden_size = int(config['DEFAULT']['hidden_size'])
output_size = int(config['DEFAULT']['output_size'])
num_heads = int(config['DEFAULT']['num_heads'])
bert_tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=access_token)
bert_model = BertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, dtype='auto', device_map='auto')
bert_model.to(device)
for param in bert_model.parameters():
    param.requires_grad = False
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
batch_size = 128
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



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    model.to(device)
    loss_fn.to(device)
    current_batch_size = 0
    total_norm = 0
    for batch, (X, y) in enumerate(dataloader):
        current_batch_size = X.shape[0]
        #X = X.to(device)
        #y = F.one_hot(y.long(), num_classes=2).long().to(device)
        X = X.to(device)
        y = F.one_hot(y.long(), num_classes=2).long().to(device)
        #y = y.float().to(device)
        y = y.float().to(device)
        pred = model(X).reshape(current_batch_size, 2).to(device)
        #pred = torch.flatten(pred).to('cpu')
        #pred = torch.argmax(pred, dim=1)
        logging.info(f'train function batch {batch} X value shape: {X.shape}')
        logging.info(f'train function batch {batch} Prediction shape: {pred.shape}')
        logging.info(f'train function batch {batch} y label shape: {y.shape}')
        #logging.info(f'X value: {X}')
        #logging.info(f'Prediction: {pred}')
        #logging.info(f'y label: {y}')
        loss = loss_fn(pred, y)
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        wandb.log({'gradient_magnitude': total_norm, 'loss': loss.item()})
        optimizer.step()
        optimizer.zero_grad()

        if (batch % 100) == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')
            logging.info(f'training loss: {loss} [{current}/{size}]')


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    model.to(device)
    loss_fn.to(device)
    #model.to('cpu')
    test_loss, correct = 0, 0
    current_batch_size = 0
    prediction_indices = np.array([])
    label_indices = np.array([])
    with torch.no_grad():
        for X, y in dataloader:
            current_batch_size = X.shape[0]
            X = X.to(device)
            #X = X.to(device)
            #y = y.float().to(device)
            y = F.one_hot(y.long(), num_classes=2).long()
            y = y.float().to(device)
            pred = model(X).reshape(current_batch_size, 2).to(device)
            logging.info(f'test function X value shape: {X.shape}')
            logging.info(f'test function Prediction shape: {pred.shape}')
            logging.info(f'test function y label shape: {y.shape}')
            logging.info(f'predictions: {pred}')
            logging.info(f'y value: {y}')
            #pred = torch.flatten(model(X)).to(device)
            #pred = model(X).to('cpu')
            test_loss += loss_fn(pred, y).item()
            max_indices_pred = torch.argmax(pred, dim=1).float()
            max_indices_y = torch.argmax(y, dim=1).float()
            prediction_indices = np.concatenate([prediction_indices, max_indices_pred.detach().cpu().numpy()])
            label_indices = np.concatenate([label_indices, max_indices_y.detach().cpu().numpy()])
            correct += (max_indices_pred == max_indices_y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    prediction_indices = torch.from_numpy(prediction_indices)
    label_indices = torch.from_numpy(label_indices)
    #tp = true_positive(prediction_indices, label_indices)
    #fp = false_positive(prediction_indices, label_indices)
    #tn = true_negative(prediction_indices, label_indices)
    #fn = false_negative(prediction_indices, label_indices)
    #precision = tp / (tp + fp)
    #recall = tp / (tp + fn)
    #f1 = 2 * ((precision * recall) / (precision + recall))
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #logging.info(f"Test Error: \n Accuracy: {(100*correct)}%, Avg loss: {test_loss}, Precision: {precision}, Recall: {recall}, F1-score: {f1}\n")
    logging.info(f"Test Error: \n Accuracy: {(100*correct)}%, Avg loss: {test_loss}\n")
    logging.info(f'dimensions of prediction tensor: {prediction_indices.shape}')
    logging.info(f'dimensions of label tensor: {label_indices.shape}')
    #wandb.log({'precision': precision, 'recall': recall, 'F1-Score': f1})


def encode_article(text):
    bert_model.eval()
    inputs = bert_tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512,
        return_overflowing_tokens=True,
        stride=128
    ).to(bert_model.device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    vector_encodings = outputs['pooler_output']
    return vector_encodings


def query_and_encode_llama_output(text):
    #text = ''
    #with open(text_fname, 'r') as text_file:
    #    text = text_file.read()
    full_prompt = prompt.format(article=text)
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


def encode_and_label_dataset(data_id):
    y_label = 0
    encodings = []
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
            embeddings = encode_article(article_text).cpu()
            logging.info(f'embedding shape for article with id {article_id} and label {label} before mean pooling: {embeddings.shape}')
            embedding_shape = embeddings.shape
            if (embedding_shape[0] > 1):
                embeddings = torch.mean(embeddings, dim=0)
                embeddings = torch.reshape(embeddings, (1, embedding_shape[1]))
            #llm_embeddings = query_and_encode_llama_output(article_text).cpu()
            #total_encoding = torch.cat((embeddings, llm_embeddings), dim=1).cpu()
            encodings.append(embeddings)
            #encodings.append(llm_embeddings)
            #encodings.append(total_encoding)
            labels.append(y_label)
            #logging.info(f'embedding shape for article with id {article_id} and label {label}: {llm_embeddings.shape}')
            logging.info(f'embedding shape for article with id {article_id} and label {label} after mean pooling: {embeddings.shape}')
            #if (index % 10) == 0:
            #    logging.info(f'embedding for article with id {article_id} and label {label}:\n{embeddings}')
    return encodings, labels


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
    #data_true = data_true[:50]
    #data_false = data_false[:50]
    data_true = data_true[:2000]
    data_false = data_false[:2000]
    data_sample = pd.concat([data_true, data_false], ignore_index=True)
    data_sample = data_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    article_ids = data_sample['id']
    X, y = encode_and_label_dataset(article_ids)
    X_train_encodings = []
    y_train_labels = []
    X_test_encodings = []
    y_test_encodings = []
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_ratio, random_state=SEED, shuffle=True, stratify=y
    )
    #X_train = [encoding.detach().cpu() for encoding in X_train]
    #X_test = [encoding.detach().cpu() for encoding in X_test]

    #X_train_encodings, y_train_labels = encode_and_label_dataset(train_ids)
    #logging.info(f'X train type: {type(X_train)}')
    X_train_encodings = np.array(X_train)
    y_train_labels = np.array(y_train)
    X_train_tensor = torch.tensor(X_train_encodings, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_labels, dtype=torch.long).to(device)
    freq_counts = np.unique(y_train_labels, return_counts=True)
    logging.info(f'X_train_tensor shape: {X_train_tensor.shape}')
    logging.info(f'y_train_tensor shape: {y_train_tensor.shape}')
    logging.info(f'Class distribution of train set:\n{freq_counts}')

    #X_test_encodings, y_test_labels = encode_and_label_dataset(test_ids)
    X_test_encodings = np.array(X_test)
    y_test_labels = np.array(y_test)
    X_test_tensor = torch.tensor(X_test_encodings, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_labels, dtype=torch.long).to(device)
    freq_counts = np.unique(y_test_labels, return_counts=True)
    logging.info(f'X_test_tensor shape: {X_test_tensor.shape}')
    logging.info(f'y_test_tensor shape: {y_test_tensor.shape}')
    logging.info(f'Class distribution of test set:\n{freq_counts}')

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    att_neural_network = AttentionNeuralNetwork(embedding_dimensions, hidden_size, output_size, num_heads)
    att_neural_network.to(device)
    #att_neural_network.to(device)
    optimizer = optim.Adam(att_neural_network.parameters(), lr=learning_rate)
    #for index in range(epochs):
    wandb.login(key=wandb_api_key)
    wandb.init(project=wandb_project, name='BERT-pooler-outpjut-with-attention-mechanism-4000-instances')
    wandb.watch(att_neural_network, log='all', log_freq=100)
    for index in range(10):
        print(f'Epoch {index + 1}\n------------------------')
        logging.info(f'Epoch {index + 1}\n')
        train(train_loader, att_neural_network, criterion, optimizer)
        test(test_loader, att_neural_network, criterion)
    print('Done')
    #with open(txt_filename, 'r') as text_file:
    #    text = text_file.read()
    #article_encoding = encode_article(text)
    #llm_output_encoding = query_and_encode_llama_output(txt_filename)
    #print(encodings)
