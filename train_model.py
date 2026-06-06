import sys
import numpy as np
import random
import torch
import torch.optim as optim
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from utils import set_seed



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


access_token = config['DEFAULT']['access_token']
cache_dir = '/data/huggingface_models/'
llama_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, access_token=access_token)
bert_model = BertModel.from_pretrained(model_name, cache_dir=cache_dir, access_token=access_token, dtype='auto', device_map='auto')
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
bert_model.to(device)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    torch_dype=torch.float16,
    device_map='auto'
)
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
batch_size = 32
epochs = 20
learning_rate = 0.001
train_ratio = 0.75





def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch % 100) == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')




def encode_article(text):
    bert_model.eval()
    inputs = bert_tokenizer(
        text,
        return_tensors='pt',
        truncation=False,
        padding=True
    ).to(bert_model.device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    vector_encodings = output.last_hidden_state[:,0,:]
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
    # limit to 1000 instances for now
    data = df.loc[0:999]
    article_ids = data['id']
    json_text = ''
    article_text = ''
    label = ''
    y_label = 0
    X_train_encodings = []
    y_train_labels = []
    train_ids, test_ids = train_test_split(
        article_ids, test_size=1-train_ratio, random_state=SEED
    )
    for index, article_id in tqdm(enumerate(train_ids), desc=f'assessing article {article_id}'):
        article_id = article_id.lower()
        article_id_components = article_id.split('-')
        id_name = ''
        label = article_ids.loc[index,'label']
        if label == 'True':
            y_label = 1
        elif label == 'False':
            y_label = 0
        else:
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
        with open(json_path) as json_file:
            json_text = json_file.read()
            json_dict = json.loads(json_text)
            article_text = json_dict.get('text')
            embeddings = encode_article(article_text)
            llm_embeddings = query_and_encode_llama_output(article_text)
            X_train_encodings.append([embeddings, llm_embeddings])
            y_train_labels.append(y_label)
            logging.info(f'embedding shape for article with id {article_id}: {embeddings.shape}')
    X_train_encodings = np.array(X_train_encodings)
    y_train_labels = np.array(y_train_labels)
    criterion = nn.CrossEntropyLoss()

    X_train_tensor = torch.tensor(X_train_encodings, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_labels, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #with open(txt_filename, 'r') as text_file:
    #    text = text_file.read()
    #article_encoding = encode_article(text)
    #llm_output_encoding = query_and_encode_llama_output(txt_filename)
    #print(encodings)
