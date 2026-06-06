import torch
from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor


class AttentionNeuralNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_size, output_size, attention_heads, dropout=0.1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.attn = nn.MultiheadAttention(embed_dim, attention_heads, dropout=0.1)
        self.linear_sigmoid_stack = nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Sigmoid(),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.Sigmoid(),
                nn.Linear(hidden_size // 4, hidden_size // 8),
                nn.Sigmoid(),
                nn.Linear(hidden_size // 8, hidden_size // 16),
                nn.Sigmoid(),
                nn.Linear(hidden_size // 16, output_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x, x_attention_mask=None, x_key_padding_mask=None):
        #x = self.flatten(x)
        attn_output, output_weights = self.attn(x, x, x, attn_mask=x_attention_mask, key_padding_mask=x_key_padding_mask)
        dropout_attn_output = x + self.dropout(attn_output)
        nn_input = self.norm(dropout_attn_output)
        logits = self.linear_sigmoid_stack(nn_input)
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
