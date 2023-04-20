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

class BM25(object):
    def __init__(self, contexts, include_title, tokenizer):
        self.include_title = include_title
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.index_id_to_db_id = {_:i['doc_id'] for _ in enumerate(contexts)}
        if include_title:
            total_contexts = [i['title']+' '+i['context'] for i in contexts]
        else:
            total_contexts = [i['context'] for i in contexts]
        corpus = list(map(lambda i : self.tokenizer(i), total_contexts))
        print('make up bm25')
        bm25 = BM25(corpus)
        print('done')
        self.bm25 = bm25
    
    def retrieve(self, query):
        tokenized_query = self.tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        sorted_idx = np.argsort(-doc_scores)
        result = [(index_id_to_db_id[i], doc_scores[i]) for i in sorted_idx]
        return result
    
    def list_retrieve(self, query_list):
        output = {}
        for i in tqdm(query_list):
            output[i['q_id']] = self.retrieve(i['question'])
        return output
    
def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cores', type = int, default = 8)
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
    contexts = load_jsonl(args.contexts_path)
    doc_id_to_list_id = {i['doc_id']:_ for _,i in enumerate(contexts)}
    data = load_jsonl(args.data_path)
    bm25 = BM25(contexts, args.include_title, lambda i : i.split())
    
    query_list = [dict(q_id = _, question = i['question']) for i in enumerate(data)]
    # multi processing
    pool = Pool(args.num_cores)
    a = np.array_split(query_list, args.num_cores)
    a = list(map(lambda i:i.tolist(), a))
    d = pool.map(lambda i: bm25.list_retrieve, a)
    pool.close()
    pool.join()
    output = {}
    for i in list(d):
        # dict
        for key,value in i.items():
            output[key]=value
    print(time.time()-now)
    for _, i in enumerate(data):
        i['retrieved_ctxs_ids'] = [j[0] for j in output[_]]
        i['retrieved_ctxs'] = [contexts[doc_id_to_list_id[j[0]]] for j in output[_]]
    os.makedirs(args.output_path, exist_ok=True)
    save_jsonl(args.output_path, output, args.name)