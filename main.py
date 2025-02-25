import os
import time
import random
import argparse
import logging
logger = logging.getLogger("PPGN_PYTORCH.MAIN")

import torch
import numpy as np
import tensorboardX
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from model import PPGN
from utils.metrics import metrics
from dataset import PPGN_DATASET

from collections import defaultdict
from tensorflow import keras


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_device', type=int, default=0,  # 1:3090
                        help='choose which gpu to run')

    parser.add_argument('--data_rebuild', type=bool, default=True,
                        help='whether to rebuild train/test dataset')
    parser.add_argument('--mat_rebuild', type=bool, default=True,
                        help='whether to rebuild` adjacent mat')
    parser.add_argument('--cross_data_rebuild', type=bool, default=True,
                        help='whether to rebuild cross data')

    parser.add_argument('--process_mid', type=bool, default=True,
                        help='whether to rebuild train/test dataset')
    parser.add_argument('--process_ready', type=bool, default=True,
                        help='whether to rebuild train/test dataset')
    parser.add_argument('--process_cross', type=bool, default=True,
                        help='whether to rebuild train/test dataset')
    parser.add_argument('--processor_num', type=int, default=12,
                        help='number of processors when preprocessing data')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='size of mini-batch')
    parser.add_argument('--train_neg_num', type=int, default=4,
                        help='number of negative samples per training positive sample')
    parser.add_argument('--test_size', type=int, default=1,
                        help='size of sampled test data')
    parser.add_argument('--test_neg_num', type=int, default=99,
                        help='number of negative samples for test')
    parser.add_argument('--epochs', type=int, default=20,
                        help='the number of epochs')
    parser.add_argument('--gnn_layers', nargs='?', default=[32,32,16,16,8],
                        help='the unit list of layers')
    parser.add_argument('--mlp_layers', nargs='?', default=[32,8],
                        help='the unit list of layers')
    parser.add_argument('--embedding_size', type=int, default=32,
                        help='the size for embedding user and item')
    parser.add_argument('--topK', type=int, default=10,
                        help='topk for evaluation')
    parser.add_argument('--regularizer_rate', type=float, default=1e-5,
                        help='the regularizer rate')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--dropout_message', type=float, default=0.1,
                        help='dropout rate of message')
    parser.add_argument('--NCForMF', type=str, default='NCF',  # NCF
                        help='method to propagate embeddings')
    
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether to train on gpu')
    parser.add_argument('--weight', type=float, default=1.0,
                        help='whether to train on gpu')
    parser.add_argument('--weight_tt', type=float, default=1.0,
                        help='whether to train on gpu')
    parser.add_argument('--weight_ss', type=float, default=1.0,
                        help='whether to train on gpu')

    parser.add_argument('--task_id', type=int, default=21,
                        help='设置任务')

    parser.add_argument('--graph_encoder', type=str, default="raw",
                        help='设置图的编码器, 可选[raw(light-gcn), ngcf, no-graph-ncf, semi-gcn]')

    parser.add_argument('--is_simple_pool', type=bool, default=True,
                        help='是否为每个item生成一组embedding')

    parser.add_argument('--is_douban_dataset', type=bool, default=True,
                        help='用于数据预处理')

    parser.add_argument('--attention_layer_K', type=int, default=3,
                        help='用于控制用户特征编码Attetion的长度')

    parser.add_argument('--graph_layer_K', type=int, default=3,
                        help='gnn layers 长度')
    
    parser.add_argument('--meta_dim', type=int, default=256,
                        help='meta_dim 的维度')

    parser.add_argument('--cmp_s', type=int, default=1,
                        help='alpha : beta -> alpha')
    parser.add_argument('--cmp_t', type=int, default=1,
                        help='alpha : beta -> beta')

    parser.add_argument('--max_sequence_length', default=20, help="max numb of users' historical sequence for extracting character.")

    args = parser.parse_args()
    logger.info(args)

    writer = tensorboardX.SummaryWriter("tensorboard_logs")    
    torch.cuda.set_device(args.gpu_device)

    if args.task_id == 1:

        args.weight_ss = args.weight * args.cmp_s
        args.weight_tt = args.weight * args.cmp_t

        dataset = PPGN_DATASET("./data/bo-mtv/raw/Apps.json.gz",
                                "./data/bo-mtv/raw/MoTV.json.gz", args)

        target_domain_name = 'Apps'
        source_domain_name = 'MoTV'

    else:
        exit()

    data_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    data_loader_test = DataLoader(dataset,batch_size=100)
    source_domain_user_historical_sequence = defaultdict(set)
    target_domain_user_historical_sequence = defaultdict(set)

    with tqdm(total=len(data_loader), desc="construct users' historical sequential list.") as pbar:
        for idx, batch in enumerate(data_loader):
            uid, item_s, item_t, label_s, label_t = batch

            uid_list = uid.tolist()
            item_s_list = item_s.tolist()
            item_t_list = item_t.tolist()
            label_s_list = label_s.tolist()
            label_t_list = label_t.tolist()


            for i in range(len(uid_list)):
                if label_s_list[i] == 1:
                    source_domain_user_historical_sequence[uid_list[i]].add(item_s_list[i])
                if label_t_list[i] == 1:
                    target_domain_user_historical_sequence[uid_list[i]].add(item_t_list[i])

            pbar.update(1)

    # 1.1 convert to str
    for k,v in source_domain_user_historical_sequence.items():
        source_domain_user_historical_sequence[k] = str(v)
    for k, v in target_domain_user_historical_sequence.items():
        target_domain_user_historical_sequence[k] = str(v)

    source_domain_user_historical_sequence_df = pd.DataFrame.from_dict(source_domain_user_historical_sequence, orient='index', columns=['sequence']).reset_index()
    target_domain_user_historical_sequence_df = pd.DataFrame.from_dict(target_domain_user_historical_sequence, orient='index', columns=['sequence']).reset_index()
    def seq_extractor(x):
        x = x.rstrip('}').lstrip('{').split(', ')
        for i in range(len(x)):
            x[i] = int(x[i])
        return np.array(x)

    source_pos_seq = keras.preprocessing.sequence.pad_sequences(source_domain_user_historical_sequence_df.sequence.map(seq_extractor), maxlen=args.max_sequence_length, padding='post')
    target_pos_seq = keras.preprocessing.sequence.pad_sequences(target_domain_user_historical_sequence_df.sequence.map(seq_extractor), maxlen=args.max_sequence_length, padding='post')

    source_pos_seq = torch.tensor(source_pos_seq, dtype=torch.long)
    target_pos_seq = torch.tensor(target_pos_seq, dtype=torch.long)
    
    source_domain_user_historical_sequence_after_padding_dict = dict()
    target_domain_user_historical_sequence_after_padding_dict = dict()

    # userid
    source_domain_user_historical_sequence_ulist = source_domain_user_historical_sequence_df['index'].values
    target_domain_user_historical_sequence_ulist = target_domain_user_historical_sequence_df['index'].values

    # padding
    source_uid2seq_padding = dict(zip(source_domain_user_historical_sequence_ulist, source_pos_seq))
    target_uid2seq_padding = dict(zip(target_domain_user_historical_sequence_ulist, target_pos_seq))

    print(dataset.domain1_user_number,dataset.domain1_item_number,dataset.domain2_item_number)
    model = PPGN(dataset.domain1_user_number,dataset.domain1_item_number,dataset.domain2_item_number,
    norm_adj_mat=dataset.nor_adj,args=args, source_uid2seq_padding=source_uid2seq_padding, target_uid2seq_padding=target_uid2seq_padding)
    
    if args.cuda:
        model = model.cuda()

    loss_fuction = torch.nn.BCELoss(reduction="none")
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.regularizer_rate)

    best_hr_s = 0
    best_ndcg_s = 0
    best_hr_t = 0
    best_ndcg_t = 0
    
    for epoch in range(args.epochs):
        losses = []
        losses_s = []
        losses_t = []
        for index,batch in enumerate(data_loader):
            s = time.time()
            u,si,ti,sl,tl = batch
            model.train()
            optimizer.zero_grad()
            sl_ = torch.as_tensor(sl,dtype=torch.float32)
            tl_ = torch.as_tensor(tl,dtype=torch.float32)
            def f(x):
                if x == 1.0:
                    return args.weight_ss
                else:
                    return 1.0
            def ft(x):
                if x == 1.0:
                    return args.weight_tt
                else:
                    return 1.0
            loss_w_s = torch.tensor(list(map(f,sl)),requires_grad=True)
            loss_w_t = torch.tensor(list(map(ft,tl)),requires_grad=True)
            if args.cuda:
                u = u.cuda()
                si = si.cuda()
                ti = ti.cuda()
                sl_ = sl_.cuda()
                tl_ = tl_.cuda()
                loss_w_s = loss_w_s.cuda()
                loss_w_t = loss_w_t.cuda()
            logits_s, logits_t = model(u,si,ti)
            
            loss_list_s = loss_fuction(logits_s,sl_)
            loss_list_t = loss_fuction(logits_t,tl_)
            loss_s  = torch.mean(torch.multiply(loss_list_s,loss_w_s))
            loss_t  = torch.mean(torch.multiply(loss_list_t,loss_w_t))
            loss = loss_s + loss_t
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            losses_s.append(loss_s.cpu().detach().numpy())
            losses_t.append(loss_t.cpu().detach().numpy())
            e = time.time()
            writer.add_scalar("loss",loss.cpu().detach().numpy(),index+epoch*len(data_loader))
            if index % 200 ==0:
                loss_mean = np.mean(losses)
                loss_mean_s = np.mean(losses_s)
                loss_mean_t = np.mean(losses_t)
                with torch.no_grad():
                    if epoch !=0 or index !=0: 
                        if index % 1000 ==0:
                            dataset.phase ="test"
                            model.eval()
                            HR_s, NDCG_s,HR_t, NDCG_t = 0,0,0,0
                            HR_s, NDCG_s,HR_t, NDCG_t = metrics(model, data_loader_test, args.topK,args.cuda)
                            dataset.phase = "train"
                            best_hr_s = max(best_hr_s,HR_s)
                            best_ndcg_s = max(best_ndcg_s,NDCG_s)
                            best_hr_t = max(best_hr_t,HR_t)
                            best_ndcg_t = max(best_ndcg_t,NDCG_t)
                            logger.info(f"epoch: {epoch} step: {index} train loss: {loss_mean:.3f} train loss_s: {loss_mean_s:.3f} train loss_t: {loss_mean_t:.3f}")
                            logger.info(f"epoch: {epoch} step: {index} train loss: {loss_mean:.3f} test HR_s:{HR_s:.3f} test NDCG_s:{NDCG_s:.3f} test HR_t:{HR_t:.3f} test NDCG_t:{NDCG_t:.3f}")
                            logger.info(f"【best】test HR_s:{best_hr_s:.3f} test NDCG_s:{best_ndcg_s:.3f} test HR_t:{best_hr_t:.3f} test NDCG_t:{best_ndcg_t:.3f}")

                            writer.add_scalar("hr_s",HR_s,index+epoch*len(data_loader))
                            writer.add_scalar("ndcg_s",NDCG_s,index+epoch*len(data_loader))
                            writer.add_scalar("hr_t",HR_t,index+epoch*len(data_loader))
                            writer.add_scalar("ndcf_t",NDCG_t,index+epoch*len(data_loader))
                        else:
                            logger.info(f"epoch: {epoch} step: {index} train loss: {loss_mean:.3f} train loss_s: {loss_mean_s:.3f} train loss_t: {loss_mean_t:.3f}")
                losses = []
                losses_s = []
                losses_t = []

    logger.info(f"【best】test HR_s:{best_hr_s:.3f} test NDCG_s:{best_ndcg_s:.3f} test HR_t:{best_hr_t:.3f} test NDCG_t:{best_ndcg_t:.3f}")

if __name__ == "__main__":
    train()
