# train T5
# train
# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import logging
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,DistributedSampler,RandomSampler,SequentialSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from tokenizers import AddedToken #XXX
# from utils.utils_data import BYTE_TOKENS #XXX
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from utils.metrics import *
from model import *
from datasets import load_dataset

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, help = 'test_name')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    # data
    #parser.add_argument('--train_data', nargs='+', help = 'train_data 위치')
    #parser.add_argument('--val_data', nargs='+', help='val data 위치')
    parser.add_argument('--train_data', type=str, help = 'train_data 위치')
    parser.add_argument('--val_data', type=str, help='val data 위치')
    parser.add_argument('--docs', type=str, help = 'docs 위치')
    parser.add_argument('--include_title', type=str2bool)
    parser.add_argument('--n_docs', type=int, default = 100)
    parser.add_argument('--rank_type', type=str, default = 'point', choices=['point','pair','list'])
    
    # logging 관련
    parser.add_argument('--logging_term', type=int, default = 100)
    
    # 학습 관련
    parser.add_argument('--epochs', default = 10, type=int)
    parser.add_argument('--eval_epoch', type = int, default = 1, help = 'term of evaluation')
    parser.add_argument('--batch_size', default = 8, type=int)
    parser.add_argument('--lr', type=float, default = 5e-5)
    parser.add_argument('--warmup', type=float, default = 1000)
    parser.add_argument('--decay', type=float, default = 0.05)
    parser.add_argument('--fp16', type=str2bool, default = False)
    parser.add_argument('--accumulation_step', type=int, default = 1) # 221124 추가
    # PTM model
    parser.add_argument('--ptm_path', type=str)
    # further train
    parser.add_argument('--model_path', type=str)
    
    # model
    parser.add_argument('--context_max_length', type=int)
    
    # distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    parser.add_argument('--early_stop', type=str2bool, default = True) # XXX220919
    parser.add_argument('--patience', type=int, default = 3)
    args  = parser.parse_args()
    return args

# evaluation
def evaluation(args, model, tokenizer, eval_dataloader):
    # f1, kf1, bleu, rouge, ppl
    total_loss = 0.
    model.eval()
    Predict = []
    Actual = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  args.local_rank not in [-1,0]):
            data = {i:j.cuda() for i,j in data.items()}
            output = model.forward(**data)
            if output.get('loss') is not None:
                loss = output['loss'].item()
                total_loss+=loss
            # output scores - bs,n_docs -> bs
            predict = output['score'].argmax(dim=-1).cpu().tolist()
            Predict.extend(predict)
            # if args.rank_type == 'point':
            #     Actual.extend(data['labels'].cpu().tolist())
            # elif args.rank_type == 'list':
            Actual.extend(data['labels'].argmax(dim=-1).cpu().tolist())
    acc = []
    # print('predict')
    # print(Predict)
    # print('actual')
    # print(Actual)
    for i,j in zip(Predict, Actual):
        acc.append(i==j)
    acc = sum(acc)
    cnt = len(Predict)
    return dict(Loss=total_loss/len(eval_dataloader), cnt=cnt, acc=acc), Predict

def get_scores(scores):
    if args.distributed:
        cnt = sum([j.item() for j in get_global(args, torch.tensor([scores['cnt']]).cuda())])
        acc = sum([j.item() for j in get_global(args, torch.tensor([scores['acc']]).cuda())])/cnt
        total_loss = [j.item() for j in get_global(args, torch.tensor([scores['Loss']]).cuda())]
        total_loss = sum(total_loss)/len(total_loss) 
    else:
        acc = scores['acc']/scores['cnt']
        total_loss = scores['Loss']
    return dict(Loss=np.round(total_loss,3), acc=np.round(acc,3))

def train():
    # optimizer
    optimizer_grouped_parameters = make_optimizer_group(model, args.decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.decay)
    # scheduler
    scheduler = get_linear_scheduler(len(train_dataloader)*args.epochs, args.warmup, optimizer, train_dataloader)
    
    if args.local_rank in [-1,0]:
        early_stop = EarlyStopping(args.patience, args.output_dir, max = False, min_difference=1e-5)
        
    if args.fp16:
        scaler = GradScaler()
    
    flag_tensor = torch.zeros(1).cuda()
    # train
    ########################################################################################
    global_step = 0
    for epoch in range(1, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        Loss = 0.
        step = 0
        iter_bar = tqdm(train_dataloader, desc='step', disable=args.local_rank not in [-1,0])
        #train
        for data in iter_bar:
            optimizer.zero_grad()            
            data = {i:j.cuda() for i,j in data.items()}
            if args.fp16:
                with autocast():
                    loss = model.forward(**data)['loss']
                    loss = loss / args.accumulation_step
                    scaler.scale(loss).backward()
                    if (step+1)%args.accumulation_step==0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
            else:
                loss = model.forward(**data)['loss']
                loss = loss / args.accumulation_step
                loss.backward()
                if (step+1)%args.accumulation_step==0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad()
            step+=1
            scheduler.step()
            if args.distributed:
                torch.distributed.reduce(loss, 0)
                loss = loss / torch.distributed.get_world_size()
            Loss+=loss.item()
            iter_bar.set_postfix({'epoch':epoch, 'global_step':global_step, 'lr':f"{scheduler.get_last_lr()[0]:.5f}",'total_loss':f'{Loss/step:.5f}'}) # 감소한다는 것을 확인하는 것임.
            if global_step%args.logging_term == 0:
                if args.local_rank in [-1,0]:
                    logger1.info(iter_bar)
                    logger2.info(iter_bar)
            global_step+=1
        
        # epoch 당 기록.
        if args.local_rank in [-1,0]:
            logger1.info(iter_bar)
            logger2.info(iter_bar)
        # evaluation
        ###################################################################################################
        if args.eval_epoch!=0 and epoch%args.eval_epoch==0:
            scores_, _ = evaluation(args, model, tokenizer, val_dataloader)
            scores = get_scores(scores_)            
            
            if args.local_rank in [-1,0]:
                logger1.info(f'Val ---- epoch : {epoch} ----- scores:{scores}')
                logger2.info(f'Val ---- epoch : {epoch} ----- scores:{scores}')
                model_to_save = model.module if hasattr(model,'module') else model
                early_stop.check(model_to_save, scores['Loss'])  
                if early_stop.timetobreak:
                    flag_tensor += 1
            if args.distributed:
                torch.distributed.broadcast(flag_tensor, 0) 
                torch.distributed.barrier()
        ###################################################################################################
        if args.early_stop:    
            if flag_tensor:
                if args.local_rank in [-1,0]:
                    logger1.info('early stop')
                    logger2.info('early stop')
                break
    # 저장시 - gpu 0번 것만 저장 - barrier 필수
    if args.local_rank in [-1,0]:
        torch.save(early_stop.best_model, os.path.join(early_stop.save_dir,'best_model'))
        logger1.info('train_end')
        logger2.info('train end')

if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok = True)
    if args.local_rank in [-1,0]:
        with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    logger1, logger2 = get_log(args)
    if args.local_rank in [-1,0]:
        logger1.info(args)
        logger2.info(args)
        
    # tokenizer, model load
    ########################################################################################
    tokenizer = T5Tokenizer.from_pretrained(args.ptm_path, extra_ids = 0)
    config = T5Config.from_pretrained(args.ptm_path)
    if args.model_path is None:
        t5 = T5EncoderModel.from_pretrained(args.ptm_path)
    else:
        model_state_dict = torch.load(args.model_path, map_location='cpu')
        
    model_type = T5EncoderModel
    if args.rank_type == 'point':
        model = Ranker(config, 'mean', model_type)
        
    elif args.rank_type == 'list':
        model = ListWiseRanker(config, 'mean', model_type)
    
    if args.model_path is None:    
        model.init_pretrained_model(t5.state_dict())
    else:
        model.load_state_dict(model_state_dict)
    ########################################################################################
    # distributed 관련
    if args.distributed:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count()>1
        # 이 프로세스가 어느 gpu에 할당되는지 명시
        torch.cuda.set_device(args.local_rank)
        # 통신을 위한 초기화
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device = args.local_rank)
    else:
        model.cuda()
    
    # data
    ########################################################################################
    train_data = load_jsonl(args.train_data)
    if args.rank_type == 'point':
        train_dataset = RankerDataset(args, train_data, tokenizer)
    elif args.rank_type == 'list':
        train_dataset = ListWiseRankerDataset(train_data, tokenizer, args.n_docs, args.include_title, args.context_max_length,  True)
    train_sampler = DistributedSampler(train_dataset) if args.distributed else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size, sampler = train_sampler, collate_fn = train_dataset._collate_fn)
    
    val_data = load_data(args.val_data, args.local_rank, args.distributed)
    val_dataset = ListWiseRankerDataset(val_data, tokenizer, None, args.include_title, args.context_max_length,  True)
        
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size = args.batch_size, sampler = val_sampler, collate_fn = val_dataset._collate_fn)
    ########################################################################################
    
     ########################################################################################
    # train
    ########################################################################################
    train()
    ########################################################################################
