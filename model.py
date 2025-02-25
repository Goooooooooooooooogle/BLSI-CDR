from re import S
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()
        event_K = self.event_K(emb_fea)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1)
        output = self.decoder(his_fea)
        return output.squeeze(1)
    

class PPGN(torch.nn.Module):

    def __init__(self,user_number,domain1_item_number,domain2_item_number,norm_adj_mat,args, source_uid2seq_padding, target_uid2seq_padding) -> None:
        super(PPGN,self).__init__()

        self.args = args
        self.user_number = user_number
        self.domain1_item_number = domain1_item_number
        self.domain2_item_number = domain2_item_number

        self.source_uid2seq_padding = source_uid2seq_padding
        self.target_uid2seq_padding = target_uid2seq_padding
        # for characteristic encoder
        self.event_K_source_seq = torch.nn.Sequential()
        self.event_K_target_seq = torch.nn.Sequential()

        for i in range(self.args.attention_layer_K):
            self.event_K_source_seq.add_module(f"linear_{i}", torch.nn.Linear(self.args.embedding_size, self.args.embedding_size))
            self.event_K_source_seq.add_module(f"relu_{i}", torch.nn.ReLU())

            self.event_K_target_seq.add_module(f"linear_{i}", torch.nn.Linear(self.args.embedding_size, self.args.embedding_size))
            self.event_K_target_seq.add_module(f"relu_{i}", torch.nn.ReLU())
        
        
        self.event_K_source_seq.add_module("out_s", torch.nn.Linear(self.args.embedding_size, 1, False))
        self.event_K_target_seq.add_module("out_t", torch.nn.Linear(self.args.embedding_size, 1, False))

        if self.args.graph_encoder != 'no-graph-ncf':
            self.decoder_ss = torch.nn.Sequential(torch.nn.Linear(self.args.embedding_size, self.args.meta_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(self.args.meta_dim, (self.args.embedding_size*(self.args.graph_layer_K+1)) * (self.args.embedding_size*(self.args.graph_layer_K+1))))
            self.decoder_tt = torch.nn.Sequential(torch.nn.Linear(self.args.embedding_size, self.args.meta_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(self.args.meta_dim, (self.args.embedding_size*(self.args.graph_layer_K+1)) * (self.args.embedding_size*(self.args.graph_layer_K+1))))
        else:
            self.decoder_ss = torch.nn.Sequential(torch.nn.Linear(self.args.embedding_size, self.args.meta_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(self.args.meta_dim, (self.args.embedding_size) * (self.args.embedding_size)))
            self.decoder_tt = torch.nn.Sequential(torch.nn.Linear(self.args.embedding_size, self.args.meta_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(self.args.meta_dim, (self.args.embedding_size) * (self.args.embedding_size)))
        self.event_softmax = torch.nn.Softmax(dim=1)

        self.user_embedding = torch.nn.Embedding(user_number,self.args.embedding_size)
        self.item_embeddding_s = torch.nn.Embedding(domain1_item_number,self.args.embedding_size)
        self.item_embeddding_t = torch.nn.Embedding(domain2_item_number,self.args.embedding_size)

        self.source_domain_embedding_decompose = torch.nn.Sequential(torch.nn.Linear(self.args.embedding_size * 8, self.args.embedding_size * 4), torch.nn.ReLU(),
                                            torch.nn.Linear(self.args.embedding_size * 4, self.args.embedding_size * 2), torch.nn.ReLU(),
                                           torch.nn.Linear(self.args.embedding_size * 2, self.args.embedding_size))
        self.target_domain_embedding_decompose = torch.nn.Sequential(torch.nn.Linear(self.args.embedding_size, self.args.embedding_size), torch.nn.ReLU(),
                                           torch.nn.Linear(self.args.embedding_size * 2, self.args.embedding_size))

        if not args.is_simple_pool:
            print("double Pool")
            self.item_embeddding_s_special = torch.nn.Embedding(domain1_item_number,self.args.embedding_size)
            self.item_embeddding_t_special = torch.nn.Embedding(domain2_item_number,self.args.embedding_size)

        else:
            self.item_embeddding_s_special = self.item_embeddding_s
            self.item_embeddding_t_special = self.item_embeddding_t

        self.neural_cf_source = torch.nn.Linear(self.args.embedding_size + 8, 1)
        self.neural_cf_target = torch.nn.Linear(self.args.embedding_size + 8, 1)
        
        self.n_fold = 100
        self.norm_adj_mat = norm_adj_mat
        self.A_fold_hat = self._split_A_hat(self.norm_adj_mat)
        if args.cuda:
            self.A_fold_hat = self.A_fold_hat.cuda()
      
        self.layer_plus = [self.args.embedding_size] + self.args.gnn_layers

        self.sigmod = torch.nn.Sigmoid()

        self.all_weights = torch.nn.ModuleList()
        self.graph_droupout = torch.nn.Dropout(self.args.dropout_message)
        for k in range(len(self.layer_plus)-1):
            self.all_weights.append(torch.nn.Linear(self.layer_plus[k],self.layer_plus[k+1]))
        
        self.ncf_weights_s = torch.nn.ModuleList()
        self.ncf_weights_t = torch.nn.ModuleList()
        self.ncf_drouput = torch.nn.ModuleList()
        self.args.mlp_layers_ncf = [32 * (self.args.graph_layer_K + 1)*2] + self.args.mlp_layers  # [32,8]  [256, 32, 8, 1]
        if args.graph_encoder == 'no-graph-ncf':
            self.args.mlp_layers_ncf = [32 * 2] + self.args.mlp_layers
        for k in range(len(self.args.mlp_layers_ncf)-1):
            self.ncf_weights_s.append(torch.nn.Linear(self.args.mlp_layers_ncf[k],self.args.mlp_layers_ncf[k+1]))
            self.ncf_weights_t.append(torch.nn.Linear(self.args.mlp_layers_ncf[k],self.args.mlp_layers_ncf[k+1]))

        self.ncf_dense_s = torch.nn.Linear(self.args.mlp_layers_ncf[-1],1)
        self.ncf_dense_t = torch.nn.Linear(self.args.mlp_layers_ncf[-1],1)

        self.ncf_weights_s_gmf = torch.nn.ModuleList()
        self.ncf_weights_t_gmf = torch.nn.ModuleList()
        self.ncf_drouput = torch.nn.ModuleList()
        self.args.mlp_layers_gmf = [32 * (self.args.graph_layer_K+1)*2] + self.args.mlp_layers
        for k in range(len(self.args.mlp_layers)-1):
            self.ncf_weights_s_gmf.append(torch.nn.Linear(self.args.mlp_layers_gmf[k],self.args.mlp_layers_gmf[k+1]))
            self.ncf_weights_t_gmf.append(torch.nn.Linear(self.args.mlp_layers_gmf[k],self.args.mlp_layers_gmf[k+1]))

        self.ncf_dense_s_gmf = torch.nn.Linear(self.args.embedding_size * 5 + self.args.mlp_layers[-1],1)
        self.ncf_dense_t_gmf = torch.nn.Linear(self.args.embedding_size * 5 + self.args.mlp_layers[-1],1)

        self.a_s_dropout = torch.nn.Dropout(self.args.dropout_message)
        self.a_t_dropout = torch.nn.Dropout(self.args.dropout_message)
        
        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.mapping = torch.nn.Parameter(torch.randn(128,128))
        self.mapping_s = torch.nn.Parameter(torch.randn(128,128))
        self.mapping_t = torch.nn.Parameter(torch.randn(128,128))

        if self.args.graph_encoder == 'ngcf':
            self.ngcf_weight_dict = self.init_weight_for_ngcf()

        elif self.args.graph_encoder == 'semi-gcn':
            self.semi_gcn_encoder_dict = self.init_weight_for_semi_gcn()
                    

    def init_weight_for_semi_gcn(self):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        layers = [self.args.embedding_size] * 4
        for k in range(3):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})

        return weight_dict

    def init_weight_for_ngcf(self):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        layers = [self.args.embedding_size] * 4
        for k in range(3):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
        return weight_dict

    def forward(self,u,si,ti):
        source_seq_list = list()
        target_seq_list = list()
        for each in u.tolist():
            source_seq_list.append(self.source_uid2seq_padding[each].tolist())
            target_seq_list.append(self.target_uid2seq_padding[each].tolist())
        
        source_seq_list_cuda = torch.LongTensor(source_seq_list).cuda()
        target_seq_list_cuda = torch.LongTensor(target_seq_list).cuda()

        mask_source = (source_seq_list_cuda == 0).float()
        mask_target = (target_seq_list_cuda == 0).float()

        source_seq_emb_list = self.item_embeddding_s_special(source_seq_list_cuda)
        target_seq_emb_list = self.item_embeddding_t_special(target_seq_list_cuda)

        event_K_source = self.event_K_source_seq(source_seq_emb_list)
        event_K_target = self.event_K_target_seq(target_seq_emb_list)

        t_source = event_K_source - torch.unsqueeze(mask_source, 2) * 1e8
        t_target = event_K_target - torch.unsqueeze(mask_target, 2) * 1e8

        att_source = self.event_softmax(t_source)
        att_target = self.event_softmax(t_target)

        his_fea_source = torch.sum(att_source*source_seq_emb_list, 1)
        his_fea_target = torch.sum(att_target*target_seq_emb_list, 1)

        embeddings = torch.concat([self.item_embeddding_s.weight,self.user_embedding.weight,self.item_embeddding_t.weight],dim=0)
        all_embeddings = [embeddings]

        if self.args.graph_encoder == 'raw':  # LightGCN
            for k in range(self.args.graph_layer_K):
                embeddings = torch.sparse.mm(self.A_fold_hat,embeddings)
                all_embeddings += [embeddings]

        elif self.args.graph_encoder == 'ngcf':
            ego_embeddings = embeddings
            for k in range(self.args.graph_layer_K):
                side_embeddings = torch.sparse.mm(self.A_fold_hat, ego_embeddings)

                sum_embeddings = torch.matmul(side_embeddings, self.ngcf_weight_dict['W_gc_%d' % k]) \
                                                + self.ngcf_weight_dict['b_gc_%d' % k]

                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = torch.matmul(bi_embeddings, self.ngcf_weight_dict['W_bi_%d' % k]) \
                                                + self.ngcf_weight_dict['b_bi_%d' % k]

                ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
                ego_embeddings = nn.Dropout(0.1)(ego_embeddings)
                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

                all_embeddings += [norm_embeddings]
        elif self.args.graph_encoder == 'semi-gcn':
            ego_embeddings = embeddings
            for k in range(self.args.graph_layer_K):
                side_embeddings = torch.sparse.mm(self.A_fold_hat, ego_embeddings)
                ego_embeddings = torch.matmul(side_embeddings, self.semi_gcn_encoder_dict['W_gc_%d' % k])
                ego_embeddings = self.relu(ego_embeddings)

                all_embeddings += [ego_embeddings]
        
        elif self.args.graph_encoder == 'no-graph-ncf':
            pass 

        
        else:
            print('self.args.graph_encoder 参数有误')
            exit()
        
        all_embeddings = torch.cat(all_embeddings,dim=1)
        item_embeddings_s, user_embeddings, item_embeddings_t = all_embeddings[:self.domain1_item_number,:],all_embeddings[self.domain1_item_number:self.domain1_item_number+self.user_number,:],all_embeddings[self.domain1_item_number+self.user_number:,:]

        user_embeddings_s = user_embeddings[u]
        user_embeddings_t = user_embeddings[u]

        if self.args.NCForMF == 'MF':
            self.logits_s = torch.mean(torch.multiply(user_embeddings[u],item_embeddings_s[si]),1)
            self.logits_t = torch.mean(torch.multiply(user_embeddings[u],item_embeddings_t[ti]),1)

        elif self.args.NCForMF == 'NCF':
            if self.args.graph_encoder != 'no-graph-ncf':
                his_fea_source_meta = (self.decoder_ss(his_fea_source).squeeze(1)).view(-1, self.args.embedding_size*(self.args.graph_layer_K+1), self.args.embedding_size*(self.args.graph_layer_K+1))
                his_fea_target_meta = (self.decoder_tt(his_fea_target).squeeze(1)).view(-1, self.args.embedding_size*(self.args.graph_layer_K+1), self.args.embedding_size*(self.args.graph_layer_K+1))

            else:
                his_fea_source_meta = (self.decoder_ss(his_fea_source).squeeze(1)).view(-1, self.args.embedding_size, self.args.embedding_size)
                his_fea_target_meta = (self.decoder_tt(his_fea_target).squeeze(1)).view(-1, self.args.embedding_size, self.args.embedding_size)

            uu_emb_s = torch.bmm(user_embeddings_s.unsqueeze(1), his_fea_source_meta)
            uu_emb_t = torch.bmm(user_embeddings_t.unsqueeze(1), his_fea_target_meta)
            
            a_s = torch.cat([uu_emb_s.squeeze(1),item_embeddings_s[si]],dim=1)
            a_t = torch.cat([uu_emb_t.squeeze(1),item_embeddings_t[ti]],dim=1)
            for i  in range(len(self.args.mlp_layers_ncf)-1):
                a_s = self.ncf_weights_s[i](a_s)
                a_s = self.relu(a_s)
                a_s = self.a_s_dropout(a_s)

                a_t = self.ncf_weights_t[i](a_t)
                a_t = self.relu(a_t)
                a_t = self.a_t_dropout(a_t)
            self.logits_s = self.ncf_dense_s(a_s)
            self.logits_t = self.ncf_dense_t(a_t)
        
        elif self.args.NCForMF == 'NCF+GMF':
            a_s = torch.cat([user_embeddings_s,item_embeddings_s[si]],dim=1)
            a_t = torch.cat([user_embeddings_t,item_embeddings_t[ti]],dim=1)
            for i  in range(len(self.args.mlp_layers_gmf)-1):
                a_s = self.ncf_weights_s_gmf[i](a_s)
                a_s = self.relu(a_s)
                a_s = self.a_s_dropout(a_s)

                a_t = self.ncf_weights_t_gmf[i](a_t)
                a_t = self.relu(a_t)
                a_t = self.a_t_dropout(a_t)

            gmf_source = user_embeddings_s * item_embeddings_s[si]
            gmf_target = user_embeddings_t * item_embeddings_t[ti]
            
            neural_input_source = torch.cat([gmf_source, a_s, his_fea_source], dim=1) 
            neural_input_target = torch.cat([gmf_target, a_t, his_fea_target], dim=1) 

            self.logits_s = self.ncf_dense_s_gmf(neural_input_source)
            self.logits_t = self.ncf_dense_t_gmf(neural_input_target)

        else:
            raise ValueError


        self.logits_s = torch.squeeze(self.logits_s)
        self.logits_t = torch.squeeze(self.logits_t)
        self.logits_s = self.sigmod(self.logits_s)
        self.logits_t = self.sigmod(self.logits_t)
        return self.logits_s, self.logits_t

    def _split_A_hat(self,X):
        A_fold_hat = self._convert_sp_mat_to_sp_tensor(X)
        return A_fold_hat


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
