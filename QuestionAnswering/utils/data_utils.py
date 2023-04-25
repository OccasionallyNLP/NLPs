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

class QADataset(Dataset):
    def __init__(self, data:List[dict], tokenizer, include_title:bool, context_max_length:int):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.include_title = include_title
        self.context_max_length = context_max_length
        
    def _collate_fn(self, batch):
        encoder_inputs = []
        labels = []
        for b in batch:
            if self.include_title:
                encoder_inputs.append(b['question']+' '+b['title']+' '+b['context'])
            else:
                encoder_inputs.append(b['question']+' '+b['context'])
            labels.append(b['answer'])            
        if self.context_max_length is None:
            encoder_inputs = self.tokenizer(encoder_inputs, padding='longest',return_tensors = 'pt')
        else:
            encoder_inputs = self.tokenizer(encoder_inputs, padding=True, truncation=True, max_length=self.context_max_length, return_tensors = 'pt')
        labels = self.tokenizer(labels, padding='longest',return_tensors='pt').input_ids
        return dict(input_ids = encoder_inputs.input_ids, attention_mask = encoder_inputs.attention_mask, labels = labels)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)