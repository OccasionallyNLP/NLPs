# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:40:42 2021

@author: OK
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import tensor as T
from typing import List, Optional
from transformers import PreTrainedModel, RobertaModel, BertModel, T5EncoderModel

class Ranker(PreTrainedModel):
    def __init__(self, config, pool, model_class):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config)
        self.fc = nn.Linear(config.d_model, 2)

    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.pretrained_model(input_ids, attention_mask) 
        if self.pool=='cls':
            # T5 Encoder만 따로 만들어놔야함.
            if isinstance(self.pretrained_model, T5EncoderModel):
                out = output.last_hidden_state[:,0,:] # bs, seq_len, dim
            else:
                out = output['pooler_output'] # bs, dim
        elif self.pool == 'mean':
            out = output['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            out = out.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, 1
            out = out/(s+1e-12)
        scores = self.fc(out) # bs, 2
        
        if 'labels' in kwargs:
            loss = F.cross_entropy(scores, kwargs['labels'])
            return dict(loss=loss, score = scores)
        else:
            return dict(score = scores)
        
