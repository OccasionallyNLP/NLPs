# test T5
# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import time
import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,DistributedSampler,RandomSampler,SequentialSampler
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from tokenizers import AddedToken #XXX
# from utils.utils_data import BYTE_TOKENS #XXX
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from utils.metrics import *
from model import Ranker
import pandas as pd

def get_args():
    # parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--shard_id', type=int, default = 0) 
    parser.add_argument('--n_shards', type=int, default = 1) 
    parser.add_argument('--test_data', type=str, help = 'test_data 위치')
    # parser.add_argument('--contain_title', type=str2bool, default=True)
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    
    # PTM model
    parser.add_argument('--check_point_dir', type=str)
    
    # 데이터 관련
    parser.add_argument('--context_max_length',type= int, default = 512)
    parser.add_argument('--batch_size', default = 8, type=int)
    #parser.add_argument('--n_docs', default = 8, type=int)

    # TODO
    ## distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    args = parser.parse_args()
    return args
    
def sorting(retrieved_ids:List[List[int]], scores:List[List[float]])->List[List[int]]:
    final_result = [[(i_i,j_i) for i_i,j_i in zip(i,j)] for i,j in zip(retrieved_ids,scores)]
    final_result = [sorted(i, key=lambda i : i[1], reverse=True) for i in final_result]
    retrieved_ctxs_ids = [[j[0] for j in i] for i in final_result]
    return retrieved_ctxs_ids

if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    with open(os.path.join(args.check_point_dir,'args.txt'), 'r') as f:
        check_point_args = json.load(f)    
    ###########################################################################################
    # tokenizer, config, model
    ###########################################################################################
    config = T5Config.from_pretrained(check_point_args['ptm_path'])
    tokenizer = T5Tokenizer.from_pretrained(check_point_args['ptm_path'])
    model_type = T5EncoderModel
    model = Ranker(config, 'mean', model_type)
    model.load_state_dict(torch.load(os.path.join(check_point_args['output_dir'],'best_model')))
    
    ###########################################################################################
    # device
    ###########################################################################################
    # TODO
     if args.local_rank == -1:  # single-node multi-gpu (or cpu) mode
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    # multi gpu
    else:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device) 
        model.to(device)
    ###########################################################################################
    
    ###########################################################################################
    # data
    ###########################################################################################
    test_data = load_jsonl(args.test_data) 
    shard_size = int(len(test_data) / args.n_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    test_data = test_data[start_idx:end_idx]
    ###########################################################################################
    result_ids = [i['retrieved_ctxs_ids'] for i in data]
    n_data = len(test_data)
    data = make_2d_to_1d([[i['question']+' '+j['title']+' '+j['context'] for j in i['retrieved_ctxs']] for i in data])
    model.eval()
    result = []
    with torch.no_grad():
        # single batch
        for i,batch_start in enumerate(tqdm(range(0, len(data), args.batch_size))):
            batch = data[batch_start:batch_start+batch_size]
            model_input = tokenizer(batch, padding='max_length', truncation=True, max_length = args.context_max_length, return_tensors='pt').to(device)
            output = model(**model_input)
            result.extend(output['score'].cpu()[:,1]) 
    result = torch.cat(result).reshape(n_data,-1).tolist() # n_data*n_docs -> n_data, n_docs
    retrieved_ctxs_ids = sorting(result_ids, result)
    for i,j in zip(test_data,retrieved_ctxs_ids):
        i['reranked_retrieved_ctxs_ids'] = j
    save_jsonl(args.output_dir, test_data, 'test_data_reranked_%d'%args.shard_id)    
