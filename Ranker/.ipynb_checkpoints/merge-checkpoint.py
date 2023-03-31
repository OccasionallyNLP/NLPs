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
    parser.add_argument('--data_path', type=str, help = 'test_data 위치')
    return args

if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    # sanity check
    data = []
    for i in os.listdir(args.data_path):
        if i.endswith('jsonl'):
            data.extend(load_jsonl(os.path.join(args.data_path, i)))
    actual = [i['positive_ctxs_ids'][0] for i in data]
    predict = [i['retrieved_ctxs_ids'] for i in data]
    reranked_predict = [i['reranked_retrieved_ctxs_ids'] for i in data]
    print('before')
    before = hit(actual, predict)
    print('after')
    after = hit(actual, reranked_predict) 
    
   
    