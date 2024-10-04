import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.loc[idx, '원문'], self.data.loc[idx, '번역문']

class WMT(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
                
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        src_texts = self.data[idx]['translation']['en']
        trg_texts = self.data[idx]['translation']['de']
        src = self.tokenizer(src_texts, padding=True, truncation=True, max_length= self.tokenizer_max_len, return_tensors='pt', add_special_tokens = False).input_ids
        trg_texts = ['</s> ' + s for s in trg_texts]
        trg = self.tokenizer(trg_texts, padding=True, truncation=True, max_length = self.tokenizer_max_len, return_tensors='pt').input_ids

        return src, trg