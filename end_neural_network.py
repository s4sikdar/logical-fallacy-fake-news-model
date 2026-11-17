import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor


#logfile='dimensions.log'
#logging.basicConfig(
#        filename=logfile,
#        format='%(asctime)s - %(levelname)s - %(message)s',
#        datefmt= '%Y-%m-%d %H:%M:%S',
#        level = logging.INFO
#)


class AttentionNeuralNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_size, output_size, attention_heads, dropout=0.1):
        super(AttentionNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.attn = nn.MultiheadAttention(embed_dim, attention_heads, dropout=0.1)
        self.dropout = nn.Dropout(dropout)
        self.linear_sigmoid_stack = nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.ReLU(),
                self.dropout,
                nn.Linear(hidden_size, 2),
                #nn.Sigmoid(),
                #nn.Linear(hidden_size, hidden_size // 2),
                nn.LogSoftmax(dim=1)
                #nn.Linear(hidden_size // 2, hidden_size // 4),
                #nn.Sigmoid(),
                #nn.Linear(hidden_size // 4, hidden_size // 8),
                #nn.Sigmoid(),
                #nn.Linear(hidden_size // 8, hidden_size // 16),
                #nn.Sigmoid(),
                #nn.Linear(hidden_size // 16, output_size),
                #nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x, x_attention_mask=None, x_key_padding_mask=None):
        #x = self.flatten(x)
        #attn_output, output_weights = self.attn(x, x, x, attn_mask=x_attention_mask, key_padding_mask=x_key_padding_mask)
        #dropout_attn_output = x + self.dropout(attn_output)
        #nn_input = self.norm(dropout_attn_output)
        nn_input = self.norm(x)
        logits = self.linear_sigmoid_stack(nn_input)
        return logits


class BertClassifier(nn.Module):
    def __init__(self, bert_model, attention_heads):  
        super(BertClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        #self.attn = nn.MultiheadAttention((768 * 2), attention_heads, dropout=0.1).to(self.device)
        self.bert = bert_model.to(self.device)
        self.dropout = nn.Dropout(0.1).to(self.device)            # dropout layer
        self.relu =  nn.ReLU().to(self.device)                    # relu activation function
        self.fc1 = nn.Linear(768, 512).to(self.device)             # dense layer 1
        #self.fc2 = nn.Linear(512, 256).to(self.device)             # dense layer 1
        self.fc2 = nn.Linear(512, 256).to(self.device)             # dense layer 1
        self.fc3 = nn.Linear(256,2).to(self.device)             # dense layer 1
        #self.fc2 = nn.Linear(512,256).to(self.device)             # dense layer 1
        #self.fc3 = nn.Linear(256,128).to(self.device)             # dense layer 1
        #self.fc4 = nn.Linear(128,2).to(self.device)               # dense layer 2 (Output layer)
        #self.softmax = nn.LogSoftmax(dim=1).to(self.device)       # softmax activation function
        #self.device = 'cpu'
        #torch.set_default_device(self.device)
        self.norm = nn.LayerNorm(768).to(self.device)



    def forward(self, sent_id, mask):           # define the forward pass  
        #print(f'embedding device: {cls_hs.get_device()}')
        sent_id = sent_id.to(self.device)
        mask = mask.to(self.device)
        #logging.info(f'dimensions of bert tokenizer output: {sent_id.shape}')
        #logging.info(f'bert tokenizer output: {sent_id}')
        #cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=True).last_hidden_state
        outputs = self.bert(sent_id, attention_mask=mask)
        #logging.info(f'cls_hs: {cls_hs}')
        #cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=True)
        #logging.info(f'output: {cls_hs}')
        #article_pooled_output = cls_hs[1]
        last_hidden_state = outputs[0]
        #article_pooled_output = cls_hs[:,1,:]
        article_pooled_output = last_hidden_state[:,0]
        #input_mask_expanded = mask.unsqueeze(-1).expand(cls_hs.size()).float()
        #sum_embeddings = torch.sum(cls_hs * input_mask_expanded, 1)
        #sum_mask = input_mask_expanded.sum(1)
        #sum_mask = torch.clamp(sum_mask, min=1e-9)
        #article_pooled_output = sum_embeddings / sum_mask
        #logging.info(f'cls_hs dimensions: {cls_hs.shape}')
        #concatenated_embeddings = torch.cat((article_pooled_output, llama_pooled_output), dim=1).to(self.device)
        #logging.info(f'embedding size of cls_hs and llama_embeddings: {pooled_output.shape} {llama_pooled_output.shape}')
        #attn_output, output_weights = self.attn(
        #    article_pooled_output, concatenated_embeddings, concatenated_embeddings, attn_mask=None, key_padding_mask=None
        #)
        #pooled_output = concatenated_embeddings + self.dropout(attn_output)
        #logging.info(f'embedding size of article_pooled_output: {article_pooled_output.shape}')
        #logging.info(f'article_pooled_output: {article_pooled_output}')
        pooled_output = self.norm(article_pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc1(pooled_output)
        #logging.info(f'embedding size of pooled_output after fc1: {pooled_output.shape}')
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc2(pooled_output)
        #logging.info(f'embedding size of pooled_output after fc2: {pooled_output.shape}')
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc3(pooled_output)
        #logging.info(f'embedding size of pooled_output after fc3: {logits.shape}')
        return logits



class BERT_Arch(nn.Module):
    def __init__(self, bert, attention_heads):  
        super(BERT_Arch, self).__init__()
        self.flatten = nn.Flatten()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.attn = nn.MultiheadAttention((768 * 2), attention_heads, dropout=0.1).to(self.device)
        #self.attn = nn.MultiheadAttention(768, attention_heads, dropout=0.1).to(self.device)
        self.bert = bert.to(self.device)
        self.dropout = nn.Dropout(0.1).to(self.device)            # dropout layer
        self.relu =  nn.ReLU().to(self.device)                    # relu activation function
        self.fc1 = nn.Linear((768 * 2), 512).to(self.device)             # dense layer 1
        #self.fc1 = nn.Linear(768, 512).to(self.device)             # dense layer 1
        #self.fc2 = nn.Linear(512, 256).to(self.device)             # dense layer 1
        self.fc2 = nn.Linear(512, 256).to(self.device)             # dense layer 1
        self.fc3 = nn.Linear(256,2).to(self.device)             # dense layer 1
        #self.fc2 = nn.Linear(512,256).to(self.device)             # dense layer 1
        #self.fc3 = nn.Linear(256,128).to(self.device)             # dense layer 1
        #self.fc4 = nn.Linear(128,2).to(self.device)               # dense layer 2 (Output layer)
        #self.softmax = nn.LogSoftmax(dim=1).to(self.device)       # softmax activation function
        #self.device = 'cpu'
        #torch.set_default_device(self.device)
        self.norm = nn.LayerNorm((768 * 2)).to(self.device)
        #self.norm = nn.LayerNorm(768).to(self.device)


    def forward(self, sent_id, mask, llama_input_ids, llama_mask):           # define the forward pass  
    #def forward(self, sent_id, mask):           # define the forward pass  
        #print(f'embedding device: {cls_hs.get_device()}')
        sent_id = sent_id.to(self.device)
        mask = mask.to(self.device)
        llama_input_ids = llama_input_ids.to(self.device)
        llama_mask = llama_mask.to(self.device)
        cls_hs = self.bert(sent_id, attention_mask=mask)
        #cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=True)
        llama_embeddings = self.bert(llama_input_ids, attention_mask=llama_mask)
        last_hidden_state = cls_hs[0]
        llama_embeddings_hidden_state = cls_hs[0]
        article_pooled_output = last_hidden_state[:,0]
        llama_pooled_output = llama_embeddings_hidden_state[:,0]
        concatenated_embeddings = torch.cat((article_pooled_output, llama_pooled_output), dim=1).to(self.device)
        #logging.info(f'embedding size of cls_hs and llama_embeddings: {pooled_output.shape} {llama_pooled_output.shape}')
        attn_output, output_weights = self.attn(
            concatenated_embeddings, concatenated_embeddings, concatenated_embeddings, attn_mask=None, key_padding_mask=None
            #article_pooled_output, article_pooled_output, article_pooled_output, attn_mask=None, key_padding_mask=None
        )
        pooled_output = concatenated_embeddings + self.dropout(attn_output)
        pooled_output = self.norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc1(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc2(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc3(pooled_output)
                                            # pass the inputs to the model
        # mean pooling but accounting for the attention mask, since most tensors will have multiple rows of tokens
        #print(f'size of output: {cls_hs.size()}')
        #print(f'size of mask: {mask.size()}')
        #mask_expanded = mask.unsqueeze(-1).expand(cls_hs.size()).float()
        #print(f'size of mask_expanded: {mask_expanded.size()}')
        #sum_embeddings = torch.sum(cls_hs * mask_expanded, 1)
        #print(f'size of sum_embeddings: {sum_embeddings.size()}')
        #sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        #print(f'size of sum_mask: {sum_mask.size()}')
        #mean_pooled = sum_embeddings / sum_mask
        #document_embedding = mean_pooled.mean(dim=0, keepdim=True)
        #print(f'size of mean_pooled: {mean_pooled.size()}')
        #mean_pooled = mean_pooled.to('cpu')
        #x = self.flatten(mean_pooled)
        #x = self.fc1(x)
        #cls_hs = cls_hs.to('cpu')
        #x = self.fc1(document_embedding)
        #x = self.fc1(cls_hs)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.fc2(x)                           # output layer
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.fc3(x)                           # output layer
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.fc4(x)                           # output layer
        #x = self.softmax(x)                       # apply softmax activation
        #print(f'size of logits: {logits.size()}')
        #print(f'shape of logits: {logits.shape}')
        #print(f'type of logits: {type(x)}')
        return logits




class FakeNewsLogicalFallacyDetector(nn.Module):
    def __init__(self, bert):  
        super(FakeNewsLogicalFallacyDetector, self).__init__()
        self.flatten = nn.Flatten()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.attn = nn.MultiheadAttention((768 * 2), attention_heads, dropout=0.1).to(self.device)
        #self.attn = nn.MultiheadAttention(768, attention_heads, dropout=0.1).to(self.device)
        self.bert = bert.to(self.device)
        self.dropout = nn.Dropout(0.1).to(self.device)            # dropout layer
        self.relu =  nn.ReLU().to(self.device)                    # relu activation function
        self.fc1 = nn.Linear((768 * 2), 512).to(self.device)             # dense layer 1
        #self.fc1 = nn.Linear(768, 512).to(self.device)             # dense layer 1
        #self.fc2 = nn.Linear(512, 256).to(self.device)             # dense layer 1
        self.fc2 = nn.Linear(512, 256).to(self.device)             # dense layer 1
        self.fc3 = nn.Linear(256,2).to(self.device)             # dense layer 1
        #self.device = 'cpu'
        #torch.set_default_device(self.device)
        self.norm = nn.LayerNorm((768 * 2)).to(self.device)
        #self.norm = nn.LayerNorm(768).to(self.device)


    def forward(self, sent_id, mask, llama_input_ids, llama_mask):           # define the forward pass  
    #def forward(self, sent_id, mask):           # define the forward pass  
        #print(f'embedding device: {cls_hs.get_device()}')
        sent_id = sent_id.to(self.device)
        mask = mask.to(self.device)
        llama_input_ids = llama_input_ids.to(self.device)
        llama_mask = llama_mask.to(self.device)
        cls_hs = self.bert(sent_id, attention_mask=mask)
        #cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=True)
        llama_embeddings = self.bert(llama_input_ids, attention_mask=llama_mask)
        last_hidden_state = cls_hs[0]
        llama_embeddings_hidden_state = cls_hs[0]
        article_pooled_output = last_hidden_state[:,0]
        llama_pooled_output = llama_embeddings_hidden_state[:,0]
        concatenated_embeddings = torch.cat((article_pooled_output, llama_pooled_output), dim=1).to(self.device)
        #logging.info(f'embedding size of cls_hs and llama_embeddings: {pooled_output.shape} {llama_pooled_output.shape}')
        #attn_output, output_weights = self.attn(
        #    concatenated_embeddings, concatenated_embeddings, concatenated_embeddings, attn_mask=None, key_padding_mask=None
            #article_pooled_output, article_pooled_output, article_pooled_output, attn_mask=None, key_padding_mask=None
        #)
        pooled_output = concatenated_embeddings
        pooled_output = self.norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc1(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc2(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc3(pooled_output)
                                            # pass the inputs to the model
        # mean pooling but accounting for the attention mask, since most tensors will have multiple rows of tokens
        return logits




if __name__ == '__main__':
    #model = NeuralNetwork()
    embed_dim = 1024
    hidden_size = 512
    output_size = 2
    num_heads = 4
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    #device = 'cpu'
    model = AttentionNeuralNetwork(embed_dim, hidden_size, output_size, num_heads)
    model.to(device)
    sample_input = torch.rand(10, 32, embed_dim).to(device)
    output = model(sample_input)
    print(output)
    #print(device)
    #print(torch.__version__)
