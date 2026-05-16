import sys
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM


llama_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name, dtype='auto', device_map='auto')
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
    vector_encodings = output.last_hidden_state
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
    txt_filename = sys.argv[-1]
    text = ''
    with open(txt_filename, 'r') as text_file:
        text = text_file.read()
    article_encoding = encode_article(text)
    llm_output_encoding = query_and_encode_llama_output(txt_filename)
    print(encodings)
