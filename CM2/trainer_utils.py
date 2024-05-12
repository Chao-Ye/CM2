import pdb
import os
import random
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup
)

from .modeling_CM2 import CM2FeatureExtractor
# from .modeling_CM2 import CM2FeatureExtractor_CL

TYPE_TO_SCHEDULER_FUNCTION = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule,
    'constant_with_warmup': get_constant_schedule_with_warmup,
}

class TrainDataset(Dataset):
    def __init__(self, trainset):
        (self.x, self.y), self.table_flag = trainset

    def __len__(self):
        # return len(self.x)
        if self.x['x_num'] is not None:
            return self.x['x_num'].shape[0]
        else:
            return self.x['x_cat_input_ids'].shape[0]
    
    def __getitem__(self, index):

        if self.x['x_cat_input_ids'] is not None:
            x_cat_input_ids = self.x['x_cat_input_ids'][index:index+1]
            x_cat_att_mask = self.x['x_cat_att_mask'][index:index+1]
            col_cat_input_ids = self.x['col_cat_input_ids']
            col_cat_att_mask = self.x['col_cat_att_mask']
        else:
            x_cat_input_ids = None
            x_cat_att_mask = None
            col_cat_input_ids = None
            col_cat_att_mask = None

        if self.x['x_num'] is not None:
            x_num = self.x['x_num'][index:index+1]
            num_col_input_ids = self.x['num_col_input_ids']
            num_att_mask = self.x['num_att_mask']
        else:
            x_num = None        
            num_col_input_ids = None
            num_att_mask = None

        if self.y is not None:
            y = self.y.iloc[index:index+1]
        else:
            y = None
        return  x_cat_input_ids, x_cat_att_mask, x_num, col_cat_input_ids, col_cat_att_mask, num_col_input_ids, num_att_mask, y, self.table_flag


class SupervisedTrainCollator():
    def __init__(self,
        **kwargs,
        ):
        pass
    
    def __call__(self, data):
    
        if data[0][0] is not None:
            x_cat_input_ids = torch.cat([row[0] for row in data], 0)
        else:
            x_cat_input_ids = None
        
        if data[0][1] is not None:
            x_cat_att_mask = torch.cat([row[1] for row in data], 0)
        else:
            x_cat_att_mask = None

        if data[0][2] is not None:
            x_num = torch.cat([row[2] for row in data], 0)
        else:
            x_num = None

        col_cat_input_ids = data[0][3]
        col_cat_att_mask = data[0][4]
        num_col_input_ids = data[0][5]
        num_att_mask = data[0][6]
        y = None
        if data[0][7] is not None:
            y = pd.concat([row[7] for row in data])
        table_flag = data[0][8]

        inputs = {
            'x_cat_input_ids' : x_cat_input_ids,
            'x_cat_att_mask' : x_cat_att_mask,
            'x_num' : x_num,
            'col_cat_input_ids' : col_cat_input_ids,
            'col_cat_att_mask' : col_cat_att_mask,
            'num_col_input_ids' : num_col_input_ids,
            'num_att_mask' : num_att_mask
        }
        return inputs, y

class CM2CollatorForCL():
    def __init__(self, 
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        overlap_ratio=0.5, 
        num_partition=3,
        ignore_duplicate_cols=True,
        **kwargs) -> None:
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition,int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        self.overlap_ratio=overlap_ratio
        self.num_partition=num_partition

    def __call__(self, data):

        if data[0][0] is not None:
            x_cat_input_ids = torch.cat([row[0] for row in data], 0)
        else:
            x_cat_input_ids = None
        
        if data[0][1] is not None:
            x_cat_att_mask = torch.cat([row[1] for row in data], 0)
        else:
            x_cat_att_mask = None

        if data[0][2] is not None:
            x_num = torch.cat([row[2] for row in data], 0)
        else:
            x_num = None

        col_cat_input_ids = data[0][3]
        col_cat_att_mask = data[0][4]
        num_col_input_ids = data[0][5]
        num_att_mask = data[0][6]
        y = None
        if data[0][7] is not None:
            y = pd.concat([row[7] for row in data])
        table_flag = data[0][7]

        inputs = {'x_cat_input_ids': x_cat_input_ids,
                  'x_cat_att_mask': x_cat_att_mask,
                  'x_num': x_num,
                  'col_cat_input_ids': col_cat_input_ids,
                  'col_cat_att_mask': col_cat_att_mask,
                  'num_col_input_ids': num_col_input_ids,
                  'num_att_mask' : num_att_mask
        }
        
        if self.num_partition > 1:
            # sub_x_list = self._build_positive_pairs(inputs, self.num_partition)
            sub_x_list = self._build_positive_pairs_random(inputs, self.num_partition)
        else:
            sub_x_list = self._build_positive_pairs_single_view(inputs)

        res = {'input_sub_x':sub_x_list}
        return res, y

    
    def _build_positive_pairs_random(self, x, k):
        '''
        build multi-view of each sample by spliting columns
        '''
        # x_cat_input_ids, x_cat_att_mask, xnum, col_cat_input_ids, col_cat_att_mask, num_col_input_ids, y, self.table_flag
        if x['x_num'] is not None:
            l_num = x['x_num'].shape[1]
        else:
            l_num = 0
        
        if x['x_cat_input_ids'] is not None:
            l_cat = x['x_cat_input_ids'].shape[1]
        else:
            l_cat = 0
        
        l_all = l_num + l_cat
        sub_x_list = []
        sub_cols_len = min(l_all, math.ceil(l_all/k*(1+self.overlap_ratio)))
        x_num_cols = []
        x_cat_cols = []

        for _ in range(k):
            x_num_cols = []
            x_cat_cols = []
            sub_cols = random.sample(range(l_all), k=sub_cols_len)
            for i in sub_cols:
                if i<l_num:
                    x_num_cols.append(i)
                else:
                    x_cat_cols.append(i-l_num)
            
            if (x['x_num'] is not None) and (len(x_num_cols)>0):
                x_num = x['x_num'][:,x_num_cols]
                num_col_input_ids = x['num_col_input_ids'][x_num_cols,:]
                num_att_mask = x['num_att_mask'][x_num_cols]
            else:
                x_num = None
                num_col_input_ids = None
                num_att_mask = None
            
            if (x['x_cat_input_ids'] is not None) and (len(x_cat_cols)>0):
                x_cat_input_ids = x['x_cat_input_ids'][:,x_cat_cols,:]
                x_cat_att_mask = x['x_cat_att_mask'][:,x_cat_cols,:]
                col_cat_input_ids = x['col_cat_input_ids'][x_cat_cols,:]
                col_cat_att_mask = x['col_cat_att_mask'][x_cat_cols,:]
            else:
                x_cat_input_ids = None
                x_cat_att_mask = None
                col_cat_input_ids = None
                col_cat_att_mask = None

            encoded_info = {'x_cat_input_ids': x_cat_input_ids,
                  'x_cat_att_mask': x_cat_att_mask,
                  'x_num': x_num,
                  'col_cat_input_ids': col_cat_input_ids,
                  'col_cat_att_mask': col_cat_att_mask,
                  'num_col_input_ids': num_col_input_ids,
                  'num_att_mask': num_att_mask
            }
            sub_x_list.append(encoded_info)
        return sub_x_list

    def _build_positive_pairs_single_view(self, x):
        sub_x_list = [x]
        return sub_x_list

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_scheduler(
    name,
    optimizer,
    num_warmup_steps = None,
    num_training_steps = None,
    ):
    name = name.lower()
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == 'constant':
        return schedule_func(optimizer)
    
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == 'constant_with_warmup':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
    
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


class LinearWarmupScheduler:
    def __init__(
        self,
        optimizer,
        base_lr,
        warmup_epochs,

        warmup_start_lr=-1,
        warmup_ratio=0.1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs

        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else base_lr*warmup_ratio

    def step(self, cur_epoch):
        if cur_epoch < self.warmup_epochs:
            self._warmup_lr_schedule(
                step=cur_epoch,
                optimizer=self.optimizer,
                max_step=self.warmup_epochs,
                init_lr=self.warmup_start_lr,
                max_lr=self.base_lr,
            )
        elif cur_epoch == self.warmup_epochs:
            self._set_lr(self.optimizer, self.base_lr)
    
    def init_optimizer(self):
        self._set_lr(self.optimizer, self.warmup_start_lr)

    def _warmup_lr_schedule(self, optimizer, step, max_step, init_lr, max_lr):
        """Warmup the learning rate"""
        lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
        self._set_lr(optimizer, lr)
    
    def _set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr