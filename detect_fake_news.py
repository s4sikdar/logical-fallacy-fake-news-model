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

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from tqdm import tqdm

from transformers import DistilBertTokenizer

from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, ModernBertForSequenceClassification, AutoModelForCausalLM
from transformers import BertTokenizer, BertModel, AutoTokenizer, ModernBertModel, BigBirdTokenizer, BigBirdModel, AutoModelForMaskedLM
from end_neural_network import AttentionNeuralNetwork, BERT_Arch, BertClassifier, FakeNewsLogicalFallacyDetector


#logfile='bert-equal-instances.log'
#logfile='logical-fallacy-modern-bert.log'
#logfile='modern-bert-equal-instances.log'
#logfile='modern-bert.log'
#logfile='modern-bert-for-sequence-classification.log'
#logfile='bert-original.log'
#logfile='bert-mean-pooled.log'
#logfile='original-bert.log'
#logfile='slippery-slope-only-full-model-gossipcop-1e2.log'
#logfile = 'full-model-gossipcop-1e3.log'
logfile = 'full-model-without-attention-gossipcop-1e3.log'
logging.basicConfig(
        filename=logfile,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt= '%Y-%m-%d %H:%M:%S',
        level = logging.INFO
)


#logging.basicConfig(
#        filename=llama_logfile,
#        format='%(asctime)s - %(levelname)s - %(message)s',
#        datefmt= '%Y-%m-%d %H:%M:%S',
#        level = logging.INFO
#)


# Load train data
#train_data = pd.read_csv('./Dataset/train.csv')

# Load test data
#test_data = pd.read_csv('./Dataset/test.csv')
#data = pd.read_csv('./gossipcop_filtered_by_json.csv')
data = pd.read_csv('./gossipcop_filtered_by_logical_fallacy.csv')
data.dropna(inplace=True)
#data = data[:8000]
#data_true = data[data['label'] == True]
#data_false = data[data['label'] == False]
#minimum_num_labels = min(len(data_true), len(data_false))
#data_true = data_true[:(minimum_num_labels + 1)]
#data_false = data_false[:(minimum_num_labels + 1)]
#data_true = data_true[:2000]
#data_false = data_false[:2000]
#data = pd.concat([data_true, data_false], ignore_index=True)
#data = data.sample(frac=1, random_state=42).reset_index(drop=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
#torch.set_default_device(device)
logging.info(f'on device {device}')
#logging.info(f'num true instances is {len(data_true)}, portion true isntances is {len(data_true) / (minimum_num_labels * 2)}')
#logging.info(f'num false instances is {len(data_false)} portion false instances is {len(data_false) / (minimum_num_labels * 2)}')

config = configparser.ConfigParser()
config.read('hugging_face_token.cfg')
#model_name = config['DEFAULT']['docker_distilbert_model_path']
#model_name = config['DEFAULT']['docker_model_name_path']
model_name = config['DEFAULT']['docker_modernbert_path']
wandb_api_key = config['DEFAULT']['wandb_api_key']
access_token = config['DEFAULT']['access_token']
cache_dir = config['DEFAULT']['docker_cache_dir']
num_heads = int(config['DEFAULT']['num_heads'])
llama_model_name = config['DEFAULT']['docker_llama_name_path']
#batch_size = 16
batch_size = 8
#epochs = 20
learning_rate = 2e-5
train_epochs = 10
#train_ratio = 0.75
#wandb_project = 'modern-bert-fake-news-detection-model'
wandb_project = 'original-bert-fake-news-detection-model'
wandb_config = {
    'epochs': train_epochs,
    'lr': learning_rate,
    'batch_size': batch_size,
    'train_ratio': 0.7
}
#bert_model = BertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, dtype='auto', device_map='auto')
bert_model = ModernBertModel.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, dtype='auto', device_map='auto')
bert_model.to(device)
for param in bert_model.parameters():
    param.requires_grad = True
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=access_token)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    token=access_token,
    device_map='auto'
)
llama_model.to(device)
logging.info(f'Loaded llama model and llama tokenizer, moved to {device}') # AutoModelForMaskedLM
prompt='''
You are analyzing text from a news article that has been cleaned to contain all
the sentences only in lowercase words. You aim to detect logical fallacies in
an article and assign a score from 1 to 10 representing the frequency and/or
level of intensity of a specific logical fallacy. In this case you are trying
to detect the slippery slope fallacy. Your response should state the score of
the slippery slope fallacy first, and then include an explanation below this.
The scores between 1 and 10 should be as follows:
A) a score of 1 - 3 means a low level of intensity of the slippery slope fallacy
B) a score of 4 - 7 means a moderate level of intensity of the slippery slope fallacy
C) a score of 8 - 10 implies a high level of intensity of the slippery slope fallacy
Some examples are given below.

Example 1:
Article:
"Secret alien pyramids were found near antarctica. Hence, the pyramids were built by aliens."

Slippery Slope Fallacy: 10
Explanation: Assuming secret alien pyramids were found near antarctica, it is
too far away of a connection to imply that the pyramids were accordingly built
by aliens.

Example 2:
Article:
"In war, defense contractors profit from selling arms to all warring factions.
As a result, defense contractors often have an incentive to lobby congress into
voting in favour of involvement in foreign conflicts. It follows that the U.S.
obviously had no business being in any of the conflicts we were in post 2000."

Slippery Slope Fallacy: 2
Explanation: It is true that defense contractors profit from selling arms to
warring parties, and that defense contractors have an incentive to influence
congress policy decisions. However, this does not automatically mean the U.S.
had no business in the foreign conflicts they were in. There may still have been
bad actors that intended to cause harm to U.S. civilians who the U.S. would
have had to deal with. The taliban are an example.

Example 3:
Article:
"If we say that it is ethical for the Miami Herald to fire the columnist, then
the next thing you know no reporters will try to push the envelope at all
anymore for fear of their jobs. It will be the end of freedom of the press as
we know it."

Slippery slope fallacy:6
Explanation: the Miami Herald firing a columnist does not mean that freedom of
the press will end as a whole. There may have been ethical concerns that led to
the firing of the columnist in the first place. Or the firing may not have had
anything to do with the reporter's work. We do not know for sure, so we cannot
say that the firing of the columnist will end freedom of the press overall.

Now analyze the following article:
{article}

Analysis:
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
    def __init__(self, data, max_len=512):
        self.data = data
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.llama_model = llama_model
        self.llama_tokenizer = llama_tokenizer
        self.prompt = updated_prompt


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
        if ((index % 100) == 0):
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




class FakeNewsClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(FakeNewsClassifier, self).__init__()
        #self.bert = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.bert = self.bert.to(device)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]



def train_epoch(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0
    train_acc = 0
    instances = 0

    for input_ids, attention_mask, llama_input_ids, llama_attn_mask, labels in train_loader:
    #for input_ids, attention_mask, labels in train_loader:
        optimizer.zero_grad()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        #logging.info(f'line 395: input_ids dimension: {input_ids.shape}')
        #logging.info(f'line 396: mask dimension: {attention_mask.shape}')
            #label = labels[index]
        #outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        outputs = model(input_ids, attention_mask, llama_input_ids, llama_attn_mask)
        #outputs = model(input_ids, attention_mask)
        #logging.info(f'line 401: output dimension: {outputs.shape}')
        #logging.info(f'line 402: labels dimension: {labels.shape}')
        #logging.info(f'labels: {labels}')
        #logging.info(f'outputs: {outputs}')
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_acc += (outputs.argmax(1) == labels.to(device)).sum().item()
        #instances += 1

    #train_loss /= len(instances)
    #train_acc /= len(instances)
    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)
    #wandb.log({'loss': loss.item()})

    return train_loss, train_acc



def eval_epoch(model, criterion, val_loader):
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for input_ids, attention_mask, llama_input_ids, llama_attn_mask, labels in val_loader:
        #for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            #outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            outputs = model(input_ids, attention_mask, llama_input_ids, llama_attn_mask)
            #outputs = model(input_ids, attention_mask)
            #logging.info(f'output dimension: {outputs.shape}')
            #logging.info(f'input_ids dimension: {input_ids.shape}')
            #logging.info(f'labels dimension: {labels.shape}')
            loss = criterion(outputs, labels.to(device))

            val_loss += loss.item()
            val_acc += (outputs.argmax(1) == labels.to(device)).sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)

    return val_loss, val_acc



# split the data into training and validation sets

#train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
#
## Create PyTorch data loaders for the training, validation, and test sets:
#
#train_dataset = FakeNewsDataset(train_data)
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#
#val_dataset = FakeNewsDataset(val_data)
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#
#test_dataset = FakeNewsDataset(test_data)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_data, test_data = train_test_split(data, test_size=0.7, random_state=42)

validation_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_dataset = FakeNewsNetDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = FakeNewsNetDataset(validation_data)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = FakeNewsNetDataset(test_data)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


#model = FakeNewsClassifier()
#model = BERT_Arch(bert_model, num_heads)
model = FakeNewsLogicalFallacyDetector(bert_model)
#model = BertClassifier(bert_model, num_heads)
#model = ModernBertForSequenceClassification.from_pretrained(model_name)
#model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
class_weights = torch.tensor([4203.0 / 1242.0, 1.0])
#criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
criterion = nn.CrossEntropyLoss().to(device)

best_val_acc = 0
#wandb.login(key=wandb_api_key)
#wandb.init(project=wandb_project, name='full-model-all-instances-lr-1e-3-batch-size-8')
#wandb.init(project=wandb_project, name='modern-bert-12000-instances-lr-1e-3-batch-size-8')
#wandb.watch(model, log='all', log_freq=100)


for epoch in range(train_epochs):
    train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader)
    val_loss, val_acc = eval_epoch(model, criterion, val_loader)

    print(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
    logging.info(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'best_model.pt')
        best_val_acc = val_acc

# Train and evaluate the model
test_epochs = 5

for epoch in range(test_epochs):
    # Train
    model.train()
    total_train_loss = 0
    index = 0
    for batch in train_loader:
        batch_inputs, batch_masks, llama_batch_inputs, llama_batch_masks, batch_labels = batch
        #batch_inputs, batch_masks, batch_labels = batch
        batch_inputs = batch_inputs.to(device)
        batch_masks = batch_masks.to(device)
        optimizer.zero_grad()
        #outputs = model(batch_inputs, batch_masks)
        outputs = model(
            batch_inputs.to(device), batch_masks.to(device), llama_batch_inputs.to(device), llama_batch_masks.to(device)
        ).to(device)
        loss = criterion(outputs, batch_labels.to(device))
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if index % 100 == 0:
            print(f"Epoch {epoch+1} / {test_epochs} - Batch {index} / {len(train_loader)} - Loss: {loss.item()}")
            logging.info(f"Epoch {epoch+1} / {test_epochs} - Batch {index} / {len(train_loader)} - Loss: {loss.item()}")
        index += 1

    # Evaluate
    model.eval()
    total_validation_loss = 0
    predictions, true_labels = [], []
    for batch in val_loader:
        batch_inputs, batch_masks, llama_batch_inputs, llama_batch_masks, batch_labels = batch
        #batch_inputs, batch_masks, batch_labels = batch
        batch_inputs = batch_inputs.to(device)
        batch_masks = batch_masks.to(device)
        with torch.no_grad():
            #outputs = model(batch_inputs, batch_masks).to(device)
            #outputs = outputs.logits
            outputs = model(
                batch_inputs.to(device), batch_masks.to(device), llama_batch_inputs.to(device), llama_batch_masks.to(device)
            ).to(device)
            loss = criterion(outputs, batch_labels.to(device))
            total_validation_loss += loss.item()
            predictions += list(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels += list(batch_labels.cpu().numpy())

    # Print training and validation loss
    average_train_loss = total_train_loss / len(train_loader)
    average_validation_loss = total_validation_loss / len(val_loader)
    print(f"Epoch {epoch+1} / {test_epochs} - Average training loss: {average_train_loss}")
    logging.info(f"Epoch {epoch+1} / {test_epochs} - Average training loss: {average_train_loss}")
    print(f"Epoch {epoch+1} / {test_epochs} - Average validation loss: {average_validation_loss}")
    logging.info(f"Epoch {epoch+1} / {test_epochs} - Average validation loss: {average_validation_loss}")

    # Print classification report
    print(classification_report(true_labels, predictions))
    logging.info(f'classification report: {classification_report(true_labels, predictions)}')

    # Save model
    torch.save(model.state_dict(), f"full-model-epoch-{epoch+1}.pth")

