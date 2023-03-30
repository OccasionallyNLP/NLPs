# -*- coding: utf-8 -*-
# data_utils
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import tensor as T
from typing import List
import random
import copy

# class RankerDataset(Dataset):
#     def __init__(self, args, data:List[dict], tokenizer):
#         super().__init__()
#         self.data = data
#         self.args = args
#         self.tokenizer = tokenizer
        
#     def _collate_fn(self, batch):
#         encoder_inputs = []
#         labels = []
#         for b in batch:
#             # positive
#             encoder_inputs.append(b['question']+' '+b['positive_ctxs'][0]['title']+' '+b['positive_ctxs'][0]['context'])
#             labels.append(1)            
#             # negative
#             if b.get('negative_ctxs') is not None:
#                 encoder_inputs.append(b['question']+' '+b['negative_ctxs'][0])
#                 labels.append(0)            
#         if self.args.context_max_length is None:
#             encoder_inputs = self.tokenizer(encoder_inputs, padding='longest',return_tensors = 'pt')
#         else:
#             encoder_inputs = self.tokenizer(encoder_inputs, padding=True, truncation=True, max_length=self.args.context_max_length, return_tensors = 'pt')
#         return dict(input_ids = encoder_inputs.input_ids, attention_mask = encoder_inputs.attention_mask, labels = T(labels))
    
#     def __getitem__(self, index):
#         return self.data[index]
    
#     def __len__(self):
#         return len(self.data)

class RankerDataset(Dataset):
    def __init__(self, args, data:List[dict], tokenizer):
        super().__init__()
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        
    def _collate_fn(self, batch):
        encoder_inputs = []
        labels = []
        for b in batch:
            # positive
            encoder_inputs.append(b['question']+' '+b['context'])
            labels.append(b['label'])            
        if self.args.context_max_length is None:
            encoder_inputs = self.tokenizer(encoder_inputs, padding='longest',return_tensors = 'pt')
        else:
            encoder_inputs = self.tokenizer(encoder_inputs, padding=True, truncation=True, max_length=self.args.context_max_length, return_tensors = 'pt')
        return dict(input_ids = encoder_inputs.input_ids, attention_mask = encoder_inputs.attention_mask, labels = T(labels))
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
