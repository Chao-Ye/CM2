import logging
import os, pdb
import math
import collections
import json
from typing import Dict, Optional, Any, Union, Callable, List

from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer
import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd

from . import constants
dev = 'cuda'

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class CM2WordEmbedding(nn.Module):
    def __init__(self,
        vocab_size,
        hidden_dim,
        vocab_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
        vocab_freeze=False,
        use_bert=True,
        ) -> None:
        super().__init__()
        word2vec_weight = torch.load('./CM2/bert_emb.pt')
        self.word_embeddings_header = nn.Embedding.from_pretrained(word2vec_weight, freeze=vocab_freeze, padding_idx=padding_idx)
        self.word_embeddings_value = nn.Embedding(vocab_size, vocab_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings_value.weight)

        self.norm_header = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)
        weight_emb = torch.load('./CM2/bert_layernorm_weight.pt')
        bias_emb = torch.load('./CM2/bert_layernorm_bias.pt')
        self.norm_header.weight.data.copy_(weight_emb)
        self.norm_header.bias.data.copy_(bias_emb)
        if vocab_freeze:
            freeze(self.norm_header)
        self.norm_value = nn.LayerNorm(vocab_dim, eps=layer_norm_eps)

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, emb_type) -> Tensor:
        if emb_type == 'header':
            embeddings = self.word_embeddings_header(input_ids)
            embeddings = self.norm_header(embeddings)
        elif emb_type == 'value':
            embeddings = self.word_embeddings_value(input_ids)
            embeddings = self.norm_value(embeddings)
        else:
            raise RuntimeError(f'no {emb_type} word_embedding method!')

        embeddings =  self.dropout(embeddings)
        return embeddings

class CM2NumEmbedding(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim)) # add bias
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb

class CM2FeatureExtractor:
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        disable_tokenizer_parallel=False,
        ignore_duplicate_cols=False,
        **kwargs,
        ) -> None:
        if os.path.exists('./CM2/tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('./CM2/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained('./CM2/tokenizer')
        self.tokenizer.__dict__['model_max_length'] = 512
        if disable_tokenizer_parallel: # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols


    def __call__(self, x, x_cat=None, table_flag=0) -> Dict:
        encoded_inputs = {
            'x_num':None,
            'num_col_input_ids':None,
            'x_cat_input_ids':None,
        }
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c in self.categorical_columns[table_flag]] if self.categorical_columns[table_flag] is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns[table_flag]] if self.numerical_columns[table_flag] is not None else []
        
        if len(cat_cols+num_cols) == 0:
            # take all columns as categorical columns!
            cat_cols = col_names

        # TODO:
        # mask out NaN values like done in binary columns
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0) # fill Nan with zero
            x_num_ts = torch.tensor(x_num.values, dtype=torch.float32)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask'] # mask out attention

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_cat = x_cat.fillna('')

            # x_cat = x_cat.apply(lambda x: x.name + ' is ' + x) * x_mask # mask out nan features
            x_cat_str = x_cat.values.tolist()
            encoded_inputs['x_cat_input_ids'] = []
            encoded_inputs['x_cat_att_mask'] = []
            max_y = 0
            cat_cnt = len(cat_cols)
            # max_token_len = max(int(1), int(4096/cat_cnt))
            max_token_len = max(int(1), int(2048/cat_cnt))
            for sample in x_cat_str:
                x_cat_ts = self.tokenizer(sample, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
                x_cat_ts['input_ids'] = x_cat_ts['input_ids'][:,:max_token_len]
                x_cat_ts['attention_mask'] = x_cat_ts['attention_mask'][:,:max_token_len]
                encoded_inputs['x_cat_input_ids'].append(x_cat_ts['input_ids'])
                encoded_inputs['x_cat_att_mask'].append(x_cat_ts['attention_mask'])
                max_y = max(max_y, x_cat_ts['input_ids'].shape[1])
            for i in range(len(encoded_inputs['x_cat_input_ids'])):
                # tmp = torch.zeros((cat_cnt, max_y), dtype=int)
                tmp = torch.full((cat_cnt, max_y), self.pad_token_id, dtype=int)
                tmp[:, :encoded_inputs['x_cat_input_ids'][i].shape[1]] = encoded_inputs['x_cat_input_ids'][i]
                encoded_inputs['x_cat_input_ids'][i] = tmp
                tmp = torch.zeros((cat_cnt, max_y), dtype=int)
                tmp[:, :encoded_inputs['x_cat_att_mask'][i].shape[1]] = encoded_inputs['x_cat_att_mask'][i]
                encoded_inputs['x_cat_att_mask'][i] = tmp
            encoded_inputs['x_cat_input_ids'] = torch.stack(encoded_inputs['x_cat_input_ids'], dim=0)
            encoded_inputs['x_cat_att_mask'] = torch.stack(encoded_inputs['x_cat_att_mask'], dim=0)

            col_cat_ts = self.tokenizer(cat_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['col_cat_input_ids'] = col_cat_ts['input_ids']
            encoded_inputs['col_cat_att_mask'] = col_cat_ts['attention_mask']
        
        return encoded_inputs

    def save(self, path):
        '''
        save the feature extractor configuration to local dir.
        '''
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

    def load(self, path):
        '''load the feature extractor configuration from local dir.
        '''
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)


    def update(self, cat=None, num=None, bin=None):
        if cat is not None:
            self.categorical_columns = cat

        if num is not None:
            self.numerical_columns = num

        if bin is not None:
            self.binary_columns = bin

class CM2FeatureProcessor(nn.Module):
    def __init__(self,
        vocab_size=None,
        vocab_dim=768,
        hidden_dim=128,
        hidden_dropout_prob=0,
        pad_token_id=0,
        vocab_freeze=False,
        use_bert=True,
        pool_policy='avg',
        device=dev,
        ) -> None:
        super().__init__()
        self.word_embedding = CM2WordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            vocab_dim=vocab_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
        )
        self.num_embedding = CM2NumEmbedding(vocab_dim)


        self.align_layer = nn.Linear(vocab_dim, hidden_dim, bias=False)
        
        self.pool_policy=pool_policy
        self.device = device

    def _avg_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is None:
            return embs.mean(-2)
        else:
            embs[att_mask==0] = 0
            embs = embs.sum(-2) / (att_mask.sum(-1,keepdim=True).to(embs.device)+eps)
            return embs
        
    def _max_embedding_by_mask(self, embs, att_mask=None, eps=1e-12):
        if att_mask is not None:
            embs[att_mask==0] = -1e12
        embs = torch.max(embs, dim=-2)[0]
        return embs
    
    def _sa_block(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)[0]
        return x[:, 0, :]
    
    def _check_nan(self, value):
        return torch.isnan(value).any().item()

    def forward(self,
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        x_cat_input_ids=None,
        x_cat_att_mask=None,
        col_cat_input_ids=None,
        col_cat_att_mask=None,
        # cat_class=None,
        # x_cat=None,
        **kwargs,
        ) -> Tensor:
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None
        other_info = {
            'col_emb' : None,          # [num_fs+cat_fs]
            'num_cnt' : 0,             # num_fs
            'x_num' : x_num,           # [bs, num_fs]
            'cat_bert_emb' : None     # [bs, cat_fs, dim]
        }

        if other_info['x_num'] is not None:
            other_info['x_num'] = other_info['x_num'].to(self.device)

        if self.pool_policy=='avg':
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type='header') # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
    
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)
    
            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                x_cat_feat_embedding = self._avg_embedding_by_mask(x_cat_feat_embedding, x_cat_att_mask)
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                cat_col_emb = self._avg_embedding_by_mask(col_cat_feat_embedding, col_cat_att_mask)
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0],-1,-1))
                
                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._avg_embedding_by_mask(cat_feat_embedding)
    
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
    
                cat_feat_embedding = self.align_layer(cat_feat_embedding)
                cat_col_emb = self.align_layer(cat_col_emb)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
    
                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()
        elif self.pool_policy=='no':
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type='header') # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
    
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)
    
            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                col_cat_feat_embedding = col_cat_feat_embedding.unsqueeze(0).expand((x_cat_feat_embedding.shape[0],-1,-1,-1))
                cat_feat_embedding = torch.cat((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                bs,emb_dim = cat_feat_embedding.shape[0], cat_feat_embedding.shape[-1]
                cat_feat_embedding = cat_feat_embedding.reshape(bs, -1, emb_dim)
                cat_feat_embedding = self.align_layer(cat_feat_embedding)

                # mask
                col_cat_att_mask = col_cat_att_mask.unsqueeze(0).expand((x_cat_att_mask.shape[0],-1,-1))
                cat_att_mask = torch.cat((col_cat_att_mask, x_cat_att_mask), dim=-1)
                cat_att_mask = cat_att_mask.reshape(bs, -1)
                
                cat_col_emb = None
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()
        elif self.pool_policy=='max':        
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type='header') # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_col_emb = self._max_embedding_by_mask(num_col_emb, num_att_mask)
    
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)
    
            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                x_cat_feat_embedding = self._max_embedding_by_mask(x_cat_feat_embedding, x_cat_att_mask)
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                cat_col_emb = self._max_embedding_by_mask(col_cat_feat_embedding, col_cat_att_mask)
                col_cat_feat_embedding = cat_col_emb.unsqueeze(0).expand((x_cat_feat_embedding.shape[0],-1,-1))
                
                cat_feat_embedding = torch.stack((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                cat_feat_embedding = self._max_embedding_by_mask(cat_feat_embedding)
    
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._max_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
    
                cat_feat_embedding = self.align_layer(cat_feat_embedding)
                cat_col_emb = self.align_layer(cat_col_emb)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
    
                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()
        elif self.pool_policy=='self-attention':        
            if x_num is not None and num_col_input_ids is not None:
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device), emb_type='header') # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                num_emb_mask = self.add_cls(num_col_emb, num_att_mask)
                num_col_emb = num_emb_mask['embedding']
                num_att_mask = num_emb_mask['attention_mask'].to(num_col_emb.device)
                num_col_emb = self._sa_block(num_col_emb, num_att_mask)
    
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)
                num_feat_embedding = self.align_layer(num_feat_embedding)
                num_col_emb = self.align_layer(num_col_emb)
    
            if x_cat_input_ids is not None:
                x_cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='value')
                col_cat_feat_embedding = self.word_embedding(col_cat_input_ids.to(self.device), emb_type='header')
                col_cat_feat_embedding = col_cat_feat_embedding.unsqueeze(0).expand((x_cat_feat_embedding.shape[0],-1,-1,-1))
                cat_feat_embedding = torch.cat((col_cat_feat_embedding, x_cat_feat_embedding), dim=2)
                # mask
                col_cat_att_mask = col_cat_att_mask.unsqueeze(0).expand((x_cat_att_mask.shape[0],-1,-1))
                cat_att_mask = torch.cat((col_cat_att_mask, x_cat_att_mask), dim=-1)

                bs, fs, ls = cat_feat_embedding.shape[0], cat_feat_embedding.shape[1], cat_feat_embedding.shape[2]
                cat_feat_embedding = cat_feat_embedding.reshape(bs*fs, ls, -1)
                cat_att_mask = cat_att_mask.reshape(bs*fs, ls)
                cat_embedding_mask = self.add_cls(cat_feat_embedding, cat_att_mask)
                cat_feat_embedding = cat_embedding_mask['embedding'] 
                cat_att_mask = cat_embedding_mask['attention_mask'].to(cat_feat_embedding.device)
                cat_feat_embedding = self._sa_block(cat_feat_embedding, cat_att_mask).reshape(bs, fs, -1)
                cat_feat_embedding = self.align_layer(cat_feat_embedding)

                cat_col_emb = None
                x_cat_bert_embedding = self.word_embedding(x_cat_input_ids.to(self.device), emb_type='header')
                x_cat_bert_embedding = self._avg_embedding_by_mask(x_cat_bert_embedding, x_cat_att_mask)
                x_cat_bert_embedding = self.align_layer(x_cat_bert_embedding)
                other_info['cat_bert_emb'] = x_cat_bert_embedding.detach()
        else:
            raise RuntimeError(f'no such {self.pool_policy} pooling policy!!!')


        emb_list = []
        att_mask_list = []
        col_emb = []
        if num_feat_embedding is not None:
            col_emb += [num_col_emb]
            other_info['num_cnt'] = num_col_emb.shape[0]
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1]).to(self.device)]

            # emb_list += [num_feat_embedding]
            # att_mask_list += [num_att_mask]
        if cat_feat_embedding is not None:
            col_emb += [cat_col_emb]
            emb_list += [cat_feat_embedding]
            if self.pool_policy=='no':
                att_mask_list += [cat_att_mask.to(self.device)]
            else:
                att_mask_list += [torch.ones(cat_feat_embedding.shape[0], cat_feat_embedding.shape[1]).to(self.device)]
            
        if len(emb_list) == 0: raise Exception('no feature found belonging into numerical, categorical, or binary, check your data!')
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        other_info['col_emb'] = torch.cat(col_emb, 0).float()
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}, other_info

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'selu':
        return F.selu
    elif activation == 'leakyrelu':
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))

class CM2TransformerLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=False,
                 device=None, dtype=None, use_layer_norm=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        src = x
        key_padding_mask = ~key_padding_mask.bool()
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        g = self.gate_act(self.gate_linear(x))
        h = self.linear1(x)
        h = h * g # add gate
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None) -> Tensor:
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))

        else: # do not use layer norm
                x = x + self._sa_block(x, src_mask, src_key_padding_mask)
                x = x + self._ff_block(x)
        return x


class CM2InputEncoder(nn.Module):
    def __init__(self,
        feature_extractor,
        feature_processor,
        device=dev,
        ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    def forward(self, x):
        tokenized = self.feature_extractor(x)
        embeds = self.feature_processor(**tokenized)
        return embeds
    
    def load(self, ckpt_dir):
        # load feature extractor
        self.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))

        # load embedding layer
        model_name = os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

class CM2Encoder(nn.Module):
    def __init__(self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
            CM2TransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,)
            ]
            )
        if num_layer > 1:
            encoder_layer = CM2TransformerLayer(
                d_model=hidden_dim,
                nhead=num_attention_head,
                dropout=hidden_dropout_prob,
                dim_feedforward=ffn_dim,
                batch_first=True,
                layer_norm_eps=1e-5,
                norm_first=False,
                use_layer_norm=True,
                activation=activation,)
            stacked_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer-1)
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        outputs = embedding
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=attention_mask)
        return outputs

class CM2LinearClassifier(nn.Module):
    def __init__(self,
        num_class,
        hidden_dim=128) -> None:
        super().__init__()
        if num_class <= 2:
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        x = x[:,0,:] # take the cls token embedding
        x = self.norm(x)
        logits = self.fc(x)
        return logits

class CM2LinearRegression(nn.Module):
    def __init__(self,
        hidden_dim=128) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activate_fn = nn.ReLU()

    def forward(self, x) -> Tensor:
        x = x[:,0,:] # take the cls token embedding
        # x = self.norm(x)
        x = self.activate_fn(x)
        logits = self.fc(x)
        return logits

class CM2ProjectionHead(nn.Module):
    def __init__(self,
        hidden_dim=128,
        projection_dim=128):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        # x = self.norm(x)
        h = self.dense(x)
        return h

class CM2CLSToken(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        outputs = {'embedding': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1)
        outputs['attention_mask'] = attention_mask
        return outputs
    
class CM2MaskToken(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mask_emb = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.mask_emb, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, embedding, masked_indices, header_emb):
        embedding[masked_indices.bool()] = 0
        bs,fs = embedding.shape[0], embedding.shape[1]
        all_mask_token = self.mask_emb.unsqueeze(0).unsqueeze(0).expand(bs,fs,-1) + header_emb.unsqueeze(0).expand(bs,-1,-1) 
        embedding = embedding + all_mask_token*masked_indices.unsqueeze(-1)

        return embedding


class CM2Model(nn.Module):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation='relu',
        device=dev,
        vocab_freeze=False,
        use_bert=True,
        pool_policy='avg',
        **kwargs,
        ) -> None:

        super().__init__()
        self.categorical_columns=categorical_columns
        self.numerical_columns=numerical_columns
        self.binary_columns=binary_columns

        if feature_extractor is None:
            feature_extractor = CM2FeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                **kwargs,
            )

        feature_processor = CM2FeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
        )
        
        self.input_encoder = CM2InputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
        )

        self.encoder = CM2Encoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
        )

        self.cls_token = CM2CLSToken(hidden_dim=hidden_dim)
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        embeded = self.input_encoder(x)
        embeded = self.cls_token(**embeded)

        encoder_output = self.encoder(**embeded)

        return encoder_output

    def load(self, ckpt_dir):
        # load model weight state dict
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        # save model weight state dict
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        # save the input encoder separately
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None

    def update(self, config):
        col_map = {}
        for k,v in config.items():
            if k in ['cat','num','bin']: col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if 'num_class' in config:
            num_class = config['num_class']
            self.clf = CM2LinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.')

        return None


class CM2Classifier(CM2Model):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        vocab_freeze=False,
        use_bert=True,
        pool_policy='avg',
        device=dev,
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
            **kwargs,
        )
        self.num_class = num_class
        self.clf = CM2LinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        # self.fused = nn.Linear(hidden_dim * 2, hidden_dim)
        if self.num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)

    def forward(self, x, y=None, table_flag=0):
        encoder_output2 = None
        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
            # encoder_output2 = self.input_encoder2(x['x_num'])
        elif isinstance(x, pd.DataFrame):
            # input is dataframe
            inputs = self.input_encoder.feature_extractor(x, table_flag=table_flag)
            # encoder_output2 = self.input_encoder2(inputs['x_num'])
        else:
            raise ValueError(f'CM2Classifier takes inputs with dict or pd.DataFrame, find {type(x)}.')




        outputs,_ = self.input_encoder.feature_processor(**inputs)

        outputs = self.cls_token(**outputs)


        encoder_output = self.encoder(**outputs) # bs, seqlen+1, hidden_dim

        # classifier
        logits = self.clf(encoder_output)

        if y is not None:
            # compute classification loss
            if self.num_class == 2:
                if isinstance(y, pd.Series):
                    y_ts = torch.tensor(y.values).to(self.device).float()
                else:
                    y_ts = y.float().to(self.device)
                loss = self.loss_fn(logits.flatten(), y_ts)
            else:
                if isinstance(y, pd.Series):
                    y_ts = torch.tensor(y.values).to(self.device).long()
                else:
                    y_ts = y.long().to(self.device)
                loss = self.loss_fn(logits, y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss


class CM2Regression(CM2Model):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        vocab_freeze=False,
        device=dev,
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            device=device,
            **kwargs,
        )
        self.res = CM2LinearRegression(hidden_dim=hidden_dim)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.to(device)

    def forward(self, x, y=None, table_flag=0):
        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
        elif isinstance(x, pd.DataFrame):
            # input is dataframe
            inputs = self.input_encoder.feature_extractor(x, table_flag=table_flag)
        else:
            raise ValueError(f'CM2Classifier takes inputs with dict or pd.DataFrame, find {type(x)}.')

        outputs,_ = self.input_encoder.feature_processor(**inputs)
        outputs = self.cls_token(**outputs)
        encoder_output = self.encoder(**outputs) # bs, seqlen+1, hidden_dim

        logits = self.res(encoder_output)

        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values.reshape(-1,1)
                y = torch.tensor(y, dtype=torch.float32)
            y = y.float().to(self.device)
            y = y.reshape(-1,1)
            loss = self.loss_fn(logits, y)
        else:
            loss = None

        return logits, loss

class CM2ForCL(CM2Model):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128,
        overlap_ratio=0.1,
        num_partition=2,
        supervised=True,
        temperature=10,
        base_temperature=10,
        activation='relu',
        vocab_freeze=True,
        device=dev,
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            device=device,
            **kwargs,
            )
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition,int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        self.projection_head = CM2ProjectionHead(hidden_dim, projection_dim)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.supervised = supervised
        self.device = device
        self.to(device)

    def forward(self, x, y=None, table_flag=0):
        # do positive sampling
        feat_x_list = []
        if isinstance(x, pd.DataFrame):
            # print("x:",x)
            sub_x_list = self._build_positive_pairs(x, self.num_partition)
            for sub_x in sub_x_list:
                # encode two subset feature samples
                feat_x = self.input_encoder(sub_x)
                feat_x = self.cls_token(**feat_x)
                feat_x = self.encoder(**feat_x)
                feat_x_proj = feat_x[:,0,:] # take cls embedding
                feat_x_proj = self.projection_head(feat_x_proj) # bs, projection_dim
                feat_x_list.append(feat_x_proj)
        elif isinstance(x, dict):
            for input_x in x['input_sub_x']:
                feat_x,_ = self.input_encoder.feature_processor(**input_x)
                feat_x = self.cls_token(**feat_x)
                feat_x = self.encoder(**feat_x)
                feat_x_proj = feat_x[:, 0, :]
                feat_x_proj = self.projection_head(feat_x_proj)
                feat_x_list.append(feat_x_proj)
        else:
            raise ValueError(f'expect input x to be pd.DataFrame or dict(pretokenized), get {type(x)} instead')

        feat_x_multiview = torch.stack(feat_x_list, axis=1) # bs, n_view, emb_dim

        if y is not None and self.supervised:
            # take supervised loss
            if isinstance(y, pd.Series):
                y = torch.tensor(y.values, device=feat_x_multiview.device)
            loss = self.supervised_contrastive_loss(feat_x_multiview, y)
        else:
            # compute cl loss (multi-view InfoNCE loss)
            loss = self.self_supervised_contrastive_loss(feat_x_multiview)
        return None, loss

    def _build_positive_pairs(self, x, n):
        x_cols = x.columns.tolist()
        sub_col_list = np.array_split(np.array(x_cols), n)
        len_cols = len(sub_col_list[0])
        overlap = int(np.ceil(len_cols * (self.overlap_ratio)))
        sub_x_list = []
        for i, sub_col in enumerate(sub_col_list):
            if overlap > 0 and i < n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i+1][:overlap]])
            elif overlap >0 and i == n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i-1][-overlap:]])
            sub_x = x.copy()[sub_col]
            sub_x_list.append(sub_x)
        return sub_x_list

    def cos_sim(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def self_supervised_contrastive_loss(self, features):
        batch_size = features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=self.device).view(-1,1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
        # [[0,1],[2,3]] -> [0,2,1,3]
        contrast_feature = torch.cat(torch.unbind(features,dim=1),dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 
                                    1, 
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device), 
                                    0)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def supervised_contrastive_loss(self, features, labels):
        labels = labels.contiguous().view(-1,1)
        batch_size = features.shape[0]
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # contrast_mode == 'all'
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
    
    def my_supervised_contrastive_loss(self, features, labels):
        labels = labels.contiguous().view(-1,1)
        batch_size = features.shape[0]
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # contrast_mode == 'all'
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0,
        )
        n_mask = torch.logical_not(mask)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * n_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class CM2ForMask(CM2Model):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        mlm_probability=0.35,
        projection_dim=128,
        activation='relu',
        vocab_freeze=True,
        pretrain_table_num=0,
        device=dev,
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            device=device,
            **kwargs,
        )
        self.num_projection_head = [CM2ProjectionHead(hidden_dim, 1) for i in range(pretrain_table_num)]
        self.cat_projection_head = [CM2ProjectionHead(hidden_dim, hidden_dim) for i in range(pretrain_table_num)]
        self.num_projection_head = nn.ModuleList(self.num_projection_head)
        self.cat_projection_head = nn.ModuleList(self.cat_projection_head)
        self.mask_token = CM2MaskToken(hidden_dim)
        self.mse_loss_fn = nn.MSELoss(reduction="mean")
    
        self.mlm_probability = mlm_probability
        self.hidden_dim = hidden_dim
        self.negative_inf = -1e12
        self.to(device)

    def forward(self, x, y=None, table_flag=0):
        if isinstance(x, dict):
            inputs = x
        elif isinstance(x, pd.DataFrame):
            inputs = self.input_encoder.feature_extractor(x, table_flag=table_flag)
        else:
            raise ValueError(f'CM2ForMask takes inputs with dict or pd.DataFrame, find {type(x)}.')

        outputs, other_info = self.input_encoder.feature_processor(**inputs)  # {embedding, attention_mask}
        num_cnt = other_info['num_cnt']
        col_emb = other_info['col_emb']
        cat_cnt = col_emb.shape[0] - num_cnt
        
        masked_indices = self.mask_features(outputs['attention_mask'], num_cnt, cat_cnt)

        indixes_tesor = torch.full(masked_indices.shape, False)
        sum_masked = torch.sum(masked_indices, dim=-1, keepdim=True)
        indixes_tesor[:, :1] = sum_masked==0
        masked_indices[indixes_tesor] = 1
        indixes_tesor[:, :1] = sum_masked==masked_indices.shape[1]
        masked_indices[indixes_tesor] = 0

        outputs['embedding'] = self.mask_token(outputs['embedding'], masked_indices, col_emb)
        
        encoder_output = self.encoder(**outputs) # bs, seqlen, hidden_dim

        num_encoder_output = encoder_output[:, :num_cnt, :]
        cat_encoder_output = encoder_output[:, num_cnt:, :]

        num_loss = torch.tensor(0, device=self.device)
        cat_loss = torch.tensor(0, device=self.device)

        if num_encoder_output.nelement() != 0:
            num_encoder_output = self.num_projection_head[table_flag](num_encoder_output)
            num_loss = self.cal_mask_num_features_loss(num_encoder_output, masked_indices[:,:num_cnt], other_info['x_num'])

        if cat_encoder_output.nelement() != 0:
            cat_encoder_output = self.cat_projection_head[table_flag](cat_encoder_output)
            cat_loss = self.cal_mask_cat_features_loss(cat_encoder_output, masked_indices[:,num_cnt:], other_info['cat_bert_emb'])
       
        loss = 0.5*num_loss + cat_loss
        
        return None, loss
    
    def mask_features(self, attention_mask, num_cnt, cat_cnt, num_rotation=3, cat_rotation=7):
        if num_cnt==0 or cat_cnt==0:
            masked_indices = torch.bernoulli(torch.full(attention_mask.shape, self.mlm_probability))
        else:
            num_mp = min(1.0, self.mlm_probability*num_rotation*(1+cat_cnt/num_cnt)/10)
            cat_mp = min(1.0, self.mlm_probability*cat_rotation*(1+num_cnt/cat_cnt)/10)

            if num_mp >= 1.0:
                cat_mp = (self.mlm_probability*(num_cnt+cat_cnt)-num_cnt)/cat_cnt
            elif cat_mp >= 1.0:
                num_mp = (self.mlm_probability*(num_cnt+cat_cnt)-cat_cnt)/num_cnt

            num_masked_indices = torch.bernoulli(torch.full([attention_mask.shape[0], num_cnt], num_mp))
            cat_masked_indices = torch.bernoulli(torch.full([attention_mask.shape[0], cat_cnt], cat_mp))
            masked_indices = torch.cat([num_masked_indices, cat_masked_indices], dim=-1)
        
        return masked_indices.int().to(self.device)

    def cal_mask_num_features_loss(self, output_emb, masked_indices, x_num):
        # output_emb [bs, len, 1]
        # x_num [bs, len] 
        
        if masked_indices.bool().any():
            output_emb_norm = self._minmax_norm(output_emb)
            x_num = x_num.unsqueeze(dim=-1)
            loss = self.mse_loss_fn(output_emb_norm[masked_indices.bool()], x_num[masked_indices.bool()])
        else:
            loss = torch.tensor(0, device=self.device)

        return loss
    
    def cal_mask_cat_features_loss(self, output_emb, masked_indices, cat_value_emb):
        # output_emb [bs, len, dim]
        # cat_value_emb  [bs, len, dim]

        if masked_indices.bool().any():
            cosine_distance = 1-F.cosine_similarity(output_emb[masked_indices.bool()], cat_value_emb[masked_indices.bool()], dim=-1)
            loss = torch.mean(cosine_distance)
        else:
            loss = torch.tensor(0, device=self.device)

        return loss


    def cal_mask_features_loss(self, input_emb, output_emb, masked_indices):
        input_emb_copy = input_emb.detach()
        
        cosine_distance = 1-F.cosine_similarity(input_emb_copy, output_emb, dim=-1)
        loss = torch.mean(cosine_distance[masked_indices.bool()])

        return loss
    
    def _norm(self, emb, eps=1e-12):
        emb_mean = torch.mean(emb, dim=-1, keepdim=True)
        emb_var = torch.var(emb, dim=-1, keepdim=True)
        emb_norm = (emb - emb_mean) / torch.sqrt(emb_var + eps)

        return emb_norm
    
    def _minmax_norm(self, emb, eps=1e-12):
        min_vals, _ = torch.min(emb, dim=0, keepdim=True)
        max_vals, _ = torch.max(emb, dim=0, keepdim=True)
        emb_norm = (emb-min_vals) / ((max_vals-min_vals)+eps)

        return emb_norm

    def _check_nan(self, value):
        return torch.isnan(value).any().item()


class CM2ForSup(CM2Model):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        vocab_freeze=False,
        use_bert=True,
        pool_policy='avg',
        device=dev,
        **kwargs,
        ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            vocab_freeze=vocab_freeze,
            use_bert=use_bert,
            pool_policy=pool_policy,
            device=device,
            **kwargs,
        )
        self.num_class = num_class
        self.pretrain_table_num = len(self.num_class)
        self.clf = [CM2LinearClassifier(num_class=num_class[i], hidden_dim=hidden_dim) for i in range(self.pretrain_table_num)]
        self.clf = nn.ModuleList(self.clf)
        self.multiclass_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.binaryclass_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)

    def forward(self, x, y=None, table_flag=0):
        if isinstance(x, dict):
            inputs = x
        elif isinstance(x, pd.DataFrame):
            inputs = self.input_encoder.feature_extractor(x, table_flag=table_flag)
        else:
            raise ValueError(f'CM2Classifier takes inputs with dict or pd.DataFrame, find {type(x)}.')

        outputs,_ = self.input_encoder.feature_processor(**inputs)
        outputs = self.cls_token(**outputs)

        # go through transformers, get the first cls embedding
        encoder_output = self.encoder(**outputs) # bs, seqlen+1, hidden_dim

        # classifier
        logits = self.clf[table_flag](encoder_output)

        assert y is not None, 'Error! No label!'

        if self.num_class[table_flag] <= 2:
            if isinstance(y, pd.Series):
                y_ts = torch.tensor(y.values).to(self.device).float()
            else:
                y_ts = y.float().to(self.device)
            loss = self.binaryclass_loss_fn(logits.flatten(), y_ts)
        else:
            if isinstance(y, pd.Series):
                y_ts = torch.tensor(y.values).to(self.device).long()
            else:
                y_ts = y.long().to(self.device)
            loss = self.multiclass_loss_fn(logits, y_ts)
        loss = loss.mean()

        return logits, loss