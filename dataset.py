import os
import json
import time
import gzip
import logging
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
ß
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
from multiprocessing import Pool
from torch.utils.data import DataLoader


logger = logging.getLogger("PPGN_PYTORCH.DATASET")


class PPGN_DATASET(Dataset):

    def __init__(self, domain1_path, domain2_path,args) -> None:
        super(PPGN_DATASET).__init__()
        self.args = args
        self.phase = "train"
        self.domain1_user_number,self.domain1_item_number,self.domain2_user_number,self.domain2_item_number = 0,0,0,0

        self.domain1_train_dict = dict()
        self.domain1_test_dict =  dict()
        self.domain2_train_dict =  dict()
        self.domain2_test_dict =  dict()

        self.domain1_pos_dict = dict()
        self.domain2_pos_dict = dict()
        
        self.cross_train_dict = dict()
        self.cross_test_dict = dict() 

        self.nor_adj = None
        
        self.domain1_path = domain1_path
        self.domain2_path = domain2_path

        self.mid_path = os.path.abspath(os.path.join(os.path.dirname(domain1_path),"../mid"))
        self.ready_path = os.path.abspath(os.path.join(os.path.dirname(domain1_path),"../ready"))

        if not os.path.exists(self.mid_path):
            os.mkdir(self.mid_path)
        if not os.path.exists(self.ready_path):
            os.mkdir(self.ready_path)

        self.domain1_mid_path = os.path.join(self.mid_path,self.domain1_path.split("/")[-1].replace('_5.json.gz','.csv'))
        self.domain2_mid_path = os.path.join(self.mid_path,self.domain2_path.split("/")[-1].replace('_5.json.gz','.csv'))
        
        self.domain1_train_pos_path = os.path.join(self.ready_path,self.domain1_path.split("/")[-1].replace('_5.json.gz','_train.csv'))
        self.domain1_test_pos_path = os.path.join(self.ready_path,self.domain1_path.split("/")[-1].replace('_5.json.gz','_test.csv'))


        self.domain2_train_pos_path = os.path.join(self.ready_path,self.domain2_path.split("/")[-1].replace('_5.json.gz','_train.csv'))
        self.domain2_test_pos_path = os.path.join(self.ready_path,self.domain2_path.split("/")[-1].replace('_5.json.gz','_test.csv'))


        self.domain1_train_npy_path = os.path.join(self.ready_path,self.domain1_path.split("/")[-1].replace('_5.json.gz','_train.npy'))
        self.domain1_test_npy_path = os.path.join(self.ready_path,self.domain1_path.split("/")[-1].replace('_5.json.gz','_test.npy'))
        self.domain2_train_npy_path = os.path.join(self.ready_path,self.domain2_path.split("/")[-1].replace('_5.json.gz','_train.npy'))
        self.domain2_test_npy_path = os.path.join(self.ready_path,self.domain2_path.split("/")[-1].replace('_5.json.gz','_test.npy'))

        self.train_npy_path = os.path.join(self.ready_path,'cross_CDs_Digital_train.npy')
        self.test_npy_path = os.path.join(self.ready_path,'cross_CDs_Digital_test.npy')

        self.nor_adj_path = os.path.join(self.ready_path,'norm_adj_mat.npz')

        if args.data_rebuild == True:
            self.build_data()
        else:
            self.cross_train_dict = np.load(self.train_npy_path,allow_pickle=True).item()
            self.cross_test_dict = np.load(self.test_npy_path,allow_pickle=True).item()
            self.nor_adj = sp.load_npz(self.nor_adj_path) 
            self.domain1_item_number = len(set(self.cross_train_dict['item_s'])|set(self.cross_test_dict['item_s']))            
            self.domain2_item_number = len(set(self.cross_train_dict['item_t'])|set(self.cross_test_dict['item_t']))
            self.domain1_user_number = len(set(self.cross_train_dict['user']))
            self.domain2_user_number = len(set(self.cross_train_dict['user']))

    def __len__(self):

        if self.phase == "train":
            return len(self.cross_train_dict['user'])
        else:
             return len(self.cross_test_dict['user'])


    def __getitem__(self, index):

        if self.phase == "train":
            return self.cross_train_dict['user'][index],self.cross_train_dict['item_s'][index],self.cross_train_dict['item_t'][index],self.cross_train_dict['label_s'][index],self.cross_train_dict['label_t'][index]

        else:
            return self.cross_test_dict['user'][index],self.cross_test_dict['item_s'][index],self.cross_test_dict['item_t'][index],self.cross_test_dict['label_s'][index],self.cross_test_dict['label_t'][index]
    
    def get_nor_adj_matrix(self):
        if os.path.exists(self.nor_adj_path) and self.args.mat_rebuild == False:
            logger.info('Loading adjacent mats...')
            self.nor_adj = sp.load_npz(self.nor_adj_path)
        else:
            logger.info("Building adjacent matrix..")

            train_df_s = {'user':self.domain1_train_dict['user'][self.args.train_neg_num::self.args.train_neg_num+1],
                      'item':self.domain1_train_dict['item'][self.args.train_neg_num::self.args.train_neg_num+1]}
            train_df_t = {'user':self.domain2_train_dict['user'][self.args.train_neg_num::self.args.train_neg_num+1],
                      'item':self.domain2_train_dict['item'][self.args.train_neg_num::self.args.train_neg_num+1]}
            R_s = sp.dok_matrix((self.domain1_user_number,self.domain1_item_number),dtype= np.float32)
            R_t = sp.dok_matrix((self.domain1_user_number,self.domain2_item_number),dtype= np.float32)
            for user, item in zip(train_df_s['user'], train_df_s['item']):
                R_s[user, item] = 1.0

            for user, item in zip(train_df_t['user'], train_df_t['item']):
                R_t[user, item] = 1.0

            R_s, R_t = R_s.tolil(), R_t.tolil() 

            plain_adj_mat = sp.dok_matrix((self.domain1_item_number+ self.domain1_user_number+ self.domain2_item_number,self.domain1_item_number+ self.domain1_user_number+ self.domain2_item_number),
                                dtype=np.float32).tolil()

            plain_adj_mat[self.domain1_item_number: self.domain1_item_number+ self.domain1_user_number, :self.domain1_item_number] = R_s
            plain_adj_mat[:self.domain1_item_number, self.domain1_item_number: self.domain1_item_number+ self.domain1_user_number] = R_s.T
            plain_adj_mat[self.domain1_item_number: self.domain1_item_number+ self.domain1_user_number, self.domain1_item_number+ self.domain1_user_number:] = R_t
            plain_adj_mat[self.domain1_item_number+ self.domain1_user_number:, self.domain1_item_number: self.domain1_item_number+ self.domain1_user_number] = R_t.T
            plain_adj_mat = plain_adj_mat.todok()    
            

            self.nor_adj = self.normalized_adj_single(plain_adj_mat+sp.eye(plain_adj_mat.shape[0]))
            sp.save_npz(self.nor_adj_path, self.nor_adj)
        logger.info("Get normalize adjacents matrix sucessfully")
    
    def normalized_adj_single(self,adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj

    def build_data_mid (self) -> None:
        domain1 = dict()
        domain2 = dict()
        if not self.args.is_douban_dataset:
            with gzip.open(self.domain1_path,"r") as f:
                for index,line in tqdm(enumerate(f)):
                    domain1[index] = json.loads(line)
            with gzip.open(self.domain2_path,"r") as f:
                for index,line in tqdm(enumerate(f)):
                    domain2[index] = json.loads(line)
        else:
            with open(self.domain1_path, "r", encoding='utf-8') as f:
                for idx, line in tqdm(enumerate(f.readlines())):
                    domain1[idx] = json.loads(line)
            with open(self.domain2_path, "r", encoding='utf-8') as f:
                for idx, line in tqdm(enumerate(f.readlines())):
                    domain2[idx] = json.loads(line)

        domain1 = pd.DataFrame.from_dict(domain1,orient='index')
        domain2 = pd.DataFrame.from_dict(domain2,orient='index')

        domain1_users = set(domain1['reviewerID'].tolist())
        domain2_users = set(domain2['reviewerID'].tolist())

        overlapping_users = list(domain1_users & domain2_users)
        domain1 = domain1[domain1['reviewerID'].isin(overlapping_users)][['reviewerID','asin','overall','unixReviewTime']]
        domain2 = domain2[domain2['reviewerID'].isin(overlapping_users)][['reviewerID','asin','overall','unixReviewTime']]

      
        domain1.to_csv(self.domain1_mid_path,index=False)
        domain2.to_csv(self.domain2_mid_path,index=False)

        logger.info(f"Build raw data 1 to {self.domain1_mid_path}")
        logger.info(f"Build raw data 2 to {self.domain2_mid_path}")


    def build_data_ready(self):
        domain1 = pd.read_csv(self.domain1_mid_path,usecols=[0,1], sep=',')
        domain2 = pd.read_csv(self.domain2_mid_path,usecols=[0,1], sep=',')
        self.domain1_user_number = len(domain1['reviewerID'].unique().tolist())
        self.domain2_user_number = len(domain2['reviewerID'].unique().tolist())
        self.domain1_item_number = len(domain1['asin'].unique().tolist())
        self.domain2_item_number = len(domain2['asin'].unique().tolist())
        # 2. remapping
        domain1['uidx'] = domain1['reviewerID'].map(dict(zip(domain1['reviewerID'].unique(),range(self.domain1_user_number))))
        domain1['iidx'] = domain1['asin'].map(dict(zip(domain1['asin'].unique(),range(self.domain1_item_number))))
        domain2['uidx'] = domain2['reviewerID'].map(dict(zip(domain2['reviewerID'].unique(),range(self.domain2_user_number))))
        domain2['iidx'] = domain2['asin'].map(dict(zip(domain2['asin'].unique(),range(self.domain2_item_number))))
        del domain1['reviewerID'], domain1['asin']
        del domain2['reviewerID'], domain2['asin']
        # 3. overlap
        self.domain1_user_number = len(domain1['uidx'].unique().tolist())
        self.domain2_user_number = len(domain2['uidx'].unique().tolist())
        self.domain1_item_number = len(domain1['iidx'].unique().tolist())
        self.domain2_item_number = len(domain2['iidx'].unique().tolist())

        logger.info(f"Load {self.domain1_path} data successfully with {self.domain1_user_number} users, {self.domain1_item_number} products and {domain1.shape[0]} interactions.")
        logger.info(f"Load {self.domain2_path} data successfully with {self.domain2_user_number} users, {self.domain2_item_number} products and {domain2.shape[0]} interactions.")
        # 4. {uId: set(iId), .... }
        self.domain1_pos_dict, self.domain1_pos_dict_list = self.construct_pos_dict(domain1)
        logger.info("Build domain1 pos dict sucessfully")
        self.domain2_pos_dict, self.domain2_pos_dict_list = self.construct_pos_dict(domain2)
        logger.info("Build domain2 pos dict sucessfully")
        # 5. 
        domain1_train_df, domain1_test_df = self._split_train_test(domain1,self.domain1_train_pos_path,self.domain1_test_pos_path,self.domain1_user_number)
        logger.info("Build domain1 dataframe sucessfully")
        domain2_train_df, domain2_test_df = self._split_train_test(domain2,self.domain2_train_pos_path,self.domain2_test_pos_path,self.domain2_user_number)
        logger.info("Build domain2 dataframe sucessfully")
        self.domain1_train_dict = self._construct_train(domain1_train_df,self.domain1_item_number,self.domain1_pos_dict,self.domain1_train_npy_path)
        logger.info("Build domain1 train data sucessfully")
        self.domain1_test_dict = self._construct_test(domain1_test_df,self.domain1_item_number, self.domain1_pos_dict,self.domain1_test_npy_path)
        logger.info("Build domain1 test data sucessfully")
        self.domain2_train_dict = self._construct_train(domain2_train_df,self.domain2_item_number,self.domain2_pos_dict,self.domain2_train_npy_path)
        
        self.domain2_test_dict = self._construct_test(domain2_test_df,self.domain2_item_number,self.domain2_pos_dict,self.domain2_test_npy_path)
        logger.info("Build domain2 test data sucessfully")
        logger.info(len(set(self.domain2_test_dict['item'])))
        
    def build_data_cross(self):
        logger.info("Building data cross ..")
        d1_t = 0
        d2_s = 0

        nargs = [(user, self.domain1_pos_dict, self.domain2_pos_dict,self.domain1_item_number, self.domain2_item_number,
                    self.args.train_neg_num, self.domain1_pos_dict_list, self.domain2_pos_dict_list) for user in range(self.domain1_user_number)]
        

        extend_list = list()

        for each in nargs:
            extend_list.append(self._cross_build(each))

        do_t = list()
        do_s = list()
        equ_ = list()

        for (extend_users, extend_items, extend_labels, flag, user, pos_num) in extend_list:
            if flag == 't': 
                d1_t += pos_num
                do_t.append(user)
                self.domain2_train_dict['user'].extend(extend_users) # [894185] + [1060] [825795]
                self.domain2_train_dict['item'].extend(extend_items)  #[894185]
                self.domain2_train_dict['label'].extend(extend_labels) #[894185]
            elif flag == 's':
                do_s.append(user)
                d2_s += pos_num
                self.domain1_train_dict['user'].extend(extend_users)
                self.domain1_train_dict['item'].extend(extend_items)
                self.domain1_train_dict['label'].extend(extend_labels)
            else:
                equ_.append(user)

        start = time.time()
        q_s = np.argsort(np.array(self.domain1_train_dict['user']))
        q_t = np.argsort(np.array(self.domain2_train_dict['user']))

        users_s = np.array(self.domain1_train_dict['user'])[q_s].tolist()
        users_t = np.array(self.domain2_train_dict['user'])[q_t].tolist()

        assert users_s == users_t

        users = users_s

        items_s = np.array(self.domain1_train_dict['item'])[q_s].tolist()
        labels_s = np.array(self.domain1_train_dict['label'])[q_s].tolist()

        items_t = np.array(self.domain2_train_dict['item'])[q_t].tolist()
        labels_t = np.array(self.domain2_train_dict['label'])[q_t].tolist()

        self.cross_train_dict = {'user': users, 'item_s': items_s, 'item_t': items_t,'label_s': labels_s, 'label_t':labels_t}
        np.save(self.train_npy_path, self.cross_train_dict)

        assert self.domain1_test_dict['user'] == self.domain2_test_dict['user']
        self.cross_test_dict = {'user': self.domain1_test_dict['user'], 'item_s': self.domain1_test_dict['item'], 'item_t': self.domain2_test_dict['item'],
                    'label_s': self.domain1_test_dict['label'], 'label_t':self.domain2_test_dict['label']}

        np.save(self.test_npy_path, self.cross_test_dict)


    def _split_train_test(self,df,train_file_path,test_file_path,num_users):
        test_list = []
        logger.info("Spliting data of train and test")
        with Pool(self.args.processor_num) as pool:
            nargs = [(user, df, self.args.test_size) for user in range(num_users)]
            
            test_list = tqdm(pool.map(self._split, nargs))
            pool.close()
            pool.join()

        test_df = pd.concat(test_list)
        train_df = df.drop(test_df.index)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(train_file_path, index=False)
        test_df.to_csv(test_file_path, index=False)

        return train_df, test_df
    
    def _construct_train(self, df,num_items, pos_dict,path):
        # It's desperate to use df to calculate... so slow!!!
        users = []
        items = []
        labels = []
        with Pool(self.args.processor_num) as pool:
            nargs = [(user, item, num_items, pos_dict, self.args.train_neg_num, True)
                        for user, item in zip(df['uidx'], df['iidx'])]
            res_list = tqdm(pool.map(self._add_negtive, nargs))

        for (batch_users, batch_items, batch_labels) in res_list:
            users += batch_users
            items += batch_items
            labels += batch_labels


        data_dict = {'user': users, 'item': items, 'label': labels}
        np.save(path, data_dict)

        return data_dict

    def _construct_test(self, df,num_items, pos_dict,path):
        users = []
        items = []
        labels = []

        with Pool(self.args.processor_num) as pool:
            nargs = [(user, item, num_items, pos_dict, self.args.test_neg_num, False)
                     for user, item in zip(df['uidx'], df['iidx'])]
            res_list = pool.map(self._add_negtive, nargs)

        for batch_users, batch_items, batch_labels in res_list:
            users += batch_users
            items += batch_items
            labels += batch_labels

        data_dict = {'user': users, 'item': items, 'label': labels}
        np.save(path, data_dict)

        return data_dict

    @staticmethod
    def _split(args):
        user, df, test_size = args
        sample_test = df[df['uidx']==user].tail(test_size)
        return sample_test

    @staticmethod
    def _add_negtive(args):
        user, item, num_items, pos_dict, neg_num, train = args
        users, items, labels = list(), list(), list()
        neg_set = set(range(num_items)).difference(pos_dict[user])
        try:
            neg_sample_list = np.random.choice(list(neg_set),neg_num,replace=False)
        except Exception as e:
            neg_sample_list = np.random.choice(list(neg_set),neg_num,replace=True)
        for neg_sample in neg_sample_list:
            users.append(user)
            items.append(neg_sample)
            labels.append(0) if train == True else labels.append(neg_sample)      

        users.append(user)
        items.append(item)
        if train == True:
            labels.append(1)
        else:
            labels.append(item)

        return (users, items, labels)
        
    @staticmethod
    def _cross_build(args):
        user, posdict_s, posdict_t ,num_items_s, num_items_t, per_neg_num, post_list_dict_s, post_list_dict_t  = args

        num_item_s = len(post_list_dict_s[user])
        num_item_t = len(post_list_dict_t[user])
        flag = ''
        if num_item_s > num_item_t:
            flag = 't'
            pos_num = num_item_s - num_item_t  # positive sample  
            neg_num = per_neg_num * pos_num 
            pos_set = set(posdict_t[user])
            neg_set = set(range(num_items_t)) - pos_set
        elif num_item_t > num_item_s:
            flag = 's'
            pos_num = num_item_t-num_item_s
            neg_num = per_neg_num * pos_num
            pos_set = set(posdict_s[user])
            neg_set = set(range(num_items_s)) - pos_set
        else:
            return [], [], [], '', user, 0

        extend_users = (pos_num + neg_num)*[user]
        extend_items_neg = np.random.choice(list(neg_set), neg_num, replace=True)
        extend_items_pos = np.random.choice(list(pos_set), pos_num, replace=True)
        extend_items = np.concatenate([extend_items_neg, extend_items_pos])
        extend_labels = neg_num*[0] + pos_num*[1]

        return extend_users, extend_items, extend_labels, flag, user, pos_num

    def construct_pos_dict(self,df: pd.DataFrame):
        pos_dict = defaultdict(set)
        pos_list_dict = defaultdict(list)
        for user,item in zip(df['uidx'],df['iidx']):
            pos_dict[user].add(item)
            pos_list_dict[user].append(item)
        return pos_dict, pos_list_dict

    def build_data(self):
        if self.args.process_mid:
            self.build_data_mid()
        if self.args.process_ready:
            self.build_data_ready()
        self.get_nor_adj_matrix()
        if self.args.process_cross:
            self.build_data_cross()