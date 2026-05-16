import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Politifact(Dataset):
    def __init__(self, text_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv('politifact_filtered.csv')
        self.text_dir = text_dir
        sefl.transform = transform
        sefl.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        fname = self.labels.iloc[index]['article_filename']
        label = self.labels.iloc[index]['label']
        fname_components = fname.split('.')
        basename = fname_components[0]
        txt_path = ''
        if basename.endswith('true'):
            txt_path = f'./FakeNewsNet/dataset/politifact_real/{basename}.txt'
        if basename.endswith('false'):
            txt_path = f'./FakeNewsNet/dataset/politifact_fake/{basename}.txt'
        text = ''
        with open(txt_path) as article_text:
            text = article_text.read()
        return text, label



class Gossipcop(Dataset):
    def __init__(self, text_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv('gossipcop_filtered.csv')
        self.text_dir = text_dir
        sefl.transform = transform
        sefl.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        fname = self.labels.iloc[index]['article_filename']
        label = self.labels.iloc[index]['label']
        fname_components = fname.split('.')
        basename = fname_components[0]
        txt_path = ''
        if basename.endswith('true'):
            txt_path = f'./FakeNewsNet/dataset/gossipcop_real/{basename}.txt'
        if basename.endswith('false'):
            txt_path = f'./FakeNewsNet/dataset/gossipcop_fake/{basename}.txt'
        text = ''
        with open(txt_path) as article_text:
            text = article_text.read()
        return text, label
