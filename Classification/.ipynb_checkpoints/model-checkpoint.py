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

# point wise

# pair wise
## pointwise - Sigmoid cross entropy loss
## softmax cross entropy loss

## pairwise - Pairwise logistic loss
## 

## listwise - Softmax loss
##

# point wise ranker
class PointWiseRanker(PreTrainedModel):
    def __init__(self, config, pool, model_class):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config) # T5 Enc model
        self.fc = nn.Linear(config.d_model, 1)

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
            
        scores = F.sigmoid(self.fc(out).squeeze(1)) # bs, 1 -> bs
        
        if 'labels' in kwargs:
            loss_fn = nn.BCELoss()
            loss = loss_fn(scores, kwargs['labels'].float())
            return dict(loss=loss, score = scores)
        else:
            return dict(score = scores)

# list wise ranker
class ListWiseRanker(PreTrainedModel):
    def __init__(self, config, pool, model_class):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config)
        self.fc = nn.Linear(config.d_model, 1)

    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # input_ids - (bs, n_docs, seq_len) -> (bs*n_docs, seq_len)
        # attention_ids - (bs, n_docs, seq_len) -> (bs*n_docs, seq_len)
        bs, n_docs, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)
        output = self.pretrained_model(input_ids, attention_mask)
        if self.pool=='cls':
            # T5 Encoder만 따로 만들어놔야함.
            if isinstance(self.pretrained_model, T5EncoderModel):
                out = output.last_hidden_state[:,0,:] # bs*n_docs, seq_len, dim -> bs*n_docs, dim
            else:
                out = output['pooler_output'] # bs, dim
        elif self.pool == 'mean':
            out = output['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            out = out.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, 1
            out = out/(s+1e-12)
            
        # out ~ (bs*n_docs, dim) 
        out = out.reshape(bs, n_docs, -1) # bs, n_docs, dim
        scores = self.fc(out).squeeze(-1) # bs, n_docs, dim -> bs, n_docs, 1 -> bs, n_docs
        
        
        if 'labels' in kwargs: # binary label 1, 0, 0, ... ,0
            # bs, n_docs
            labels = kwargs['labels']
            loss = -((F.log_softmax(scores,dim=-1)*labels).sum())
            return dict(loss = loss, score = scores)
        else:
            return dict(score = scores)
        
# # point wise ranker
# class PairWiseLoss(nn.Module):
#     """
#     Pairwise Loss for Reward Model
#     """
#     def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
#         probs = torch.sigmoid(chosen_reward - reject_reward)
#         log_probs = torch.log(probs)
#         loss = -log_probs.mean()
#         return loss