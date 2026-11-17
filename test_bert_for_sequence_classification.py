import os
import torch
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer


# ------------------Configuration------------------
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LENGTH = 512
os.makedirs(SAVE_DIR, exist_ok=True)
config = configparser.ConfigParser()
config.read('hugging_face_token.cfg')
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
MODEL_NAME = config['DEFAULT']['docker_big_bird_model_path']
SAVE_DIR = '../models'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
logfile='dimensions.log'

# ------------------Logging Setup------------------
logging.basicConfig(
        filename=logfile,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt= '%Y-%m-%d %H:%M:%S',
        level = logging.INFO
)


# ------------- Dataset Class -------------
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ------------- Training Function -------------
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(dataloader)



# ------------- Evaluation Function -------------
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_correct / total



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
            texts.append(article_text)
            labels.append(y_label)
            #if ((index % 10) == 0):
            #    logging.info(f'article text:\n{article_text}')
    return texts, labels



# ------------- Main Function -------------
def main():
    logging.info("Starting training...")

    df_fake = pd.read_csv("../data/Fake.csv")
    df_real = pd.read_csv("../data/True.csv")
    df_fake["label"] = 0
    df_real["label"] = 1
    df = pd.concat([df_fake, df_real])
    df = df.sample(frac=1).reset_index(drop=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.1, random_state=SEED
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        logging.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        logging.info(f"Training Loss: {train_loss:.4f}")

        val_accuracy = evaluate(model, val_loader, device)
        logging.info(f"Validation Accuracy: {val_accuracy:.4f}")

        model.save_pretrained(f'models/checkpoints/checkpoint_{epoch+1}')
        tokenizer.save_pretrained(f'models/tokenizers/tokenizer_epoch_{epoch+1}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_path = os.path.join(SAVE_DIR, "best_model.pt")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            logging.info(f"Best model saved to {checkpoint_path}")

    logging.info("Training complete.")


if __name__ == '__main__':
    main()
