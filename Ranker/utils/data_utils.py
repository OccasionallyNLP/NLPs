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

class ListWiseRankerDataset(Dataset):
    def __init__(self, data:List[dict], tokenizer, n_docs:int, include_title:bool, context_max_length:int, train_mode:bool=True):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.n_docs = n_docs if n_docs is not None else len(data[0]['retrieved_ctxs_ids'])
        self.include_title = include_title
        self.context_max_length = context_max_length
        self.train_mode = train_mode
        
    def _collate_fn(self, batch):
        encoder_inputs = []
        labels = []
        bs = len(batch)
        for b in batch:
            if self.train_mode:
                # data element - question, positive_ctxs, positive_ctxs_ids
                encoder_inputs.append(b['question']+' '+b['positive_ctxs'][0]['title']+' '+b['positive_ctxs'][0]['context'] \
                                      if self.include_title else b['question']+' '+b['positive_ctxs'][0]['context'])
                labels.append(1)
                negatives = random.sample(b['retrieved_ctxs'], k=self.n_docs)
                negatives = [i for i in negatives if i['doc_id']!=b['positive_ctxs_ids'][0]][:self.n_docs-1]
                for i in negatives:
                    encoder_inputs.append(b['question']+' '+i['title']+' '+i['context'] \
                                          if self.include_title  else b['question']+' '+i['context'])
                labels.extend([0]*(self.n_docs-1))
            else:
                for i in b['retrieved_ctxs']:
                    encoder_inputs.append(b['question']+' '+i['title']+' '+i['context'] \
                                      if self.include_title else b['question']+' '+i['context'])
        if self.context_max_length is None:
            encoder_inputs = self.tokenizer(encoder_inputs, padding='longest',return_tensors = 'pt')
        else:
            encoder_inputs = self.tokenizer(encoder_inputs, padding=True, truncation=True, max_length=self.context_max_length, return_tensors = 'pt')
        
        output = dict(input_ids = encoder_inputs.input_ids.reshape(bs,self.n_docs,-1), attention_mask = encoder_inputs.attention_mask.reshape(bs,self.n_docs,-1))
        if labels:
            output['labels'] = T(labels).reshape(bs,self.n_docs)
        return output
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
   
# class ListWiseRandomRankerDataset(Dataset):
#     def __init__(self, data:List[dict], docs:List[dict], n_docs, tokenizer):
#         super().__init__()
#         self.data = data
#         self.args = args
#         self.tokenizer = tokenizer
#         self.docs = docs
#         self.n_docs = n_docs
        
#     def _collate_fn(self, batch):
#         encoder_inputs = []
#         labels = []
#         bs = len(batch)
#         for b in batch:
#             # data element - question, positive_ctxs, positive_ctxs_ids
#             if self.args.include_title:
#                 encoder_inputs.append(b['question']+' '+b['positive_ctxs'][0]['title']+' '+['positive_ctxs'][0]['context'] )
#             else:
#                 encoder_inputs.append(b['question']+' '+['positive_ctxs'][0]['context'])
#             labels.append(1)
#             negative_pool = random.choices(self.docs, k=self.n_docs)
#             negatives = [i for i in negative_pool if i['doc_id']!=b['positive_ctxs_ids'][0]][:self.n_docs-1]
#             if self.args.include_title:
#                 for i in negatives:
#                     encoder_inputs.append(b['question']+' '+i['title']+' '+i['context'] )
#             else:
#                 for i in negatives:
#                     encoder_inputs.append(b['question']+' '+i['context'])
#             labels.extend([0]*(self.n_docs-1))
#             assert len(negatives) == self.n_docs
                        
#         if self.args.context_max_length is None:
#             encoder_inputs = self.tokenizer(encoder_inputs, padding='longest',return_tensors = 'pt')
#         else:
#             encoder_inputs = self.tokenizer(encoder_inputs, padding=True, truncation=True, max_length=self.args.context_max_length, return_tensors = 'pt')
#         return dict(input_ids = encoder_inputs.input_ids.reshape(bs,self.n_docs,-1), attention_mask = encoder_inputs.attention_mask.reshape(bs,self.n_docs,-1), labels = T(labels).reshape(bs,self.n_docs))
    
#     def __getitem__(self, index):
#         return self.data[index]
    
#     def __len__(self):
#         return len(self.data)
    
    
