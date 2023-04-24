import json
import os
from utils import *
from tqdm import tqdm
import numpy as np
import random
import re
from typing import List
from collections import defaultdict
from multiprocessing import Pool
import time
import argparse
from rank_bm25 import BM25Okapi
import multiprocessing as mp

class BM25OK(object):
    def __init__(self, contexts, include_title, tokenizer):
        self.include_title = include_title
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.index_id_to_db_id = {_:i['doc_id'] for _,i in enumerate(contexts)}
        if include_title:
            total_contexts = [i['title']+' '+i['context'] for i in contexts]
        else:
            total_contexts = [i['context'] for i in contexts]
        corpus = list(map(lambda i : self.tokenizer(i), total_contexts))
        print('make up bm25')
        bm25 = BM25Okapi(corpus)
        print('done')
        self.bm25 = bm25
    
    def retrieve(self, query):
        tokenized_query = self.tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        sorted_idx = np.argsort(-doc_scores)
        result = [(self.index_id_to_db_id[i], doc_scores[i]) for i in sorted_idx]
        return result
    
def list_retrieve(query_list):
    c_proc = mp.current_process()
    for i in tqdm(query_list, desc = f'retrieve in {c_proc.name}'):
        output = {}
        output[i['q_id']] = bm25.retrieve(i['question'])[args.top_n]
        cur_path = os.path.join(args.output_dir, c_proc.name)
        os.makedirs(cur_path, exist_ok = True)
        with open(os.path.join(cur_path, i['q_id']),'wb') as f:
            json.dump(output, f)
    
def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cores', type = int)
    parser.add_argument('--top_n', type = int, default = 100)
    parser.add_argument('--contexts_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type = str)
    parser.add_argument('--name', type = str)
    parser.add_argument('--include_title', type = str2bool, default = True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.num_cores is None:
        args.num_cores = os.cpu_count()
    contexts = load_jsonl(args.contexts_path)
    doc_id_to_list_id = {i['doc_id']:_ for _,i in enumerate(contexts)}
    data = load_jsonl(args.data_path)
    query_list = [dict(q_id = _, question = i['question']) for _,i in enumerate(data)]
    
    bm25 = BM25OKnvidia-(contexts, args.include_title, lambda i : i.split())
    
    # multi processing
    pool = Pool(args.num_cores)
    a = np.array_split(query_list, args.num_cores)
    a = list(map(lambda i:i.tolist(), a))
    print('============== multi-processing starts ======================')
    d = pool.map(list_retrieve, a)
    
    pool.close()
    print('============== multi-processing close is done ======================')
    pool.join()
    print('============== multi-processing join is done ======================')
    