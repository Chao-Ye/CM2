import logging
import os
import pdb
import math
import shutil
import sys
import time
import json
import random

import deepspeed
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import torch.optim as optim
from tqdm.autonotebook import trange
from loguru import logger

from . import constants
from .evaluator import predict, get_eval_metric_fn, EarlyStopping, evaluate
from .modeling_CM2 import CM2FeatureExtractor
from .trainer_utils import SupervisedTrainCollator, TrainDataset
from .trainer_utils import get_parameter_names
from .trainer_utils import get_scheduler

class Trainer_ds:
    def __init__(self,
        model,
        train_set_list,
        test_set_list=None,
        cmd_args=None,
        collate_fn=None,
        output_dir='./ckpt',
        num_epoch=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=0,
        patience=5,
        eval_batch_size=256,
        warmup_ratio=None,
        warmup_steps=None,
        balance_sample=False,
        load_best_at_last=True,
        ignore_duplicate_cols=False,
        eval_metric='auc',
        eval_less_is_better=False,
        num_workers=0,
        device=None,
        flag=0,
        data_weight=None,
        **kwargs,
        ):
        self.flag = flag
        self.model = model
        self.data_weight = data_weight
        if isinstance(train_set_list, tuple): train_set_list = [train_set_list]
        if isinstance(test_set_list, tuple): test_set_list = [test_set_list]

        # self.train_set_list = train_set_list
        # self.test_set_list = test_set_list
        self.collate_fn = collate_fn
        if collate_fn is None:
            self.collate_fn = SupervisedTrainCollator(
                categorical_columns=model.categorical_columns,
                numerical_columns=model.numerical_columns,
                binary_columns=model.binary_columns,
                ignore_duplicate_cols=ignore_duplicate_cols,
            )

        self.args = {
            'lr':lr,
            'weight_decay':weight_decay,
            'batch_size':batch_size,
            'num_epoch':num_epoch,
            'eval_batch_size':eval_batch_size,
            'warmup_ratio': warmup_ratio,
            'warmup_steps': warmup_steps,
            'num_training_steps': self.get_num_epoch_steps(train_set_list, batch_size),
            'eval_metric': get_eval_metric_fn(eval_metric),
            'eval_metric_name': eval_metric,
        }
        self.args['steps_per_epoch'] = int(self.args['num_training_steps'] / (num_epoch * len(train_set_list)))

        self.optimizer = None
        parameters = self.create_optimizer()
        logger.info(f'deepspeed init start')
        with open(cmd_args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        ds_config['scheduler']['params']['warmup_num_steps'] = self.args['num_training_steps']*warmup_steps
        self.model_engine, _, _, _ = deepspeed.initialize(
            model=self.model, model_parameters=parameters, 
            config_params=ds_config,
            # args=cmd_args,
        )
        logger.info(f'deepspeed init finish')

        # data preprocess once
        self.feature_extractor = CM2FeatureExtractor(
            categorical_columns=model.categorical_columns,
            numerical_columns=model.numerical_columns,
            binary_columns=model.binary_columns,
            disable_tokenizer_parallel=True,
            ignore_duplicate_cols=ignore_duplicate_cols,
        )
        new_train_list = []
        new_test_list = []
        for dataindex, trainset in enumerate(train_set_list):
            x = trainset[0]
            if trainset[1] is not None:
                y = trainset[1]
            else:
                y = None
            inputs = self.feature_extractor(x, table_flag=dataindex)            
            new_train_list.append((inputs, y))

        self.trainloader_list = []
        # self.trainsampler_list = []
        for index, trainset in enumerate(new_train_list):
            train_data = TrainDataset((trainset, index))
            train_sampler = DistributedSampler(train_data)
            # self.trainsampler_list.append(train_sampler)

            trainloader = DataLoader(train_data, collate_fn=self.collate_fn, batch_size=batch_size, sampler=train_sampler)
            self.trainloader_list.append(trainloader)
        
        if test_set_list is not None:
            for dataindex, testset in enumerate(test_set_list):
                x = testset[0]
                if testset[1] is not None:
                    y = testset[1]
                else:
                    y = None
                inputs = self.feature_extractor(x, table_flag=dataindex)            
                new_test_list.append((inputs, y))
        
            self.testloader_list = [
                self._build_dataloader((testset,index), eval_batch_size, collator=self.collate_fn, shuffle=False) for index, testset in enumerate(new_test_list)
            ]
        else:
            self.testloader_list = None

        logger.info(f'deepspeed dataload finish')

        self.output_dir = output_dir
        self.early_stopping = EarlyStopping(output_dir=output_dir, patience=patience, verbose=False, less_is_better=eval_less_is_better)

        if self.model_engine.local_rank == 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        self.lr_scheduler = None
        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last

    def train(self):
        args = self.args

        start_time = time.time()
        for epoch in trange(args['num_epoch'], desc='Epoch'):
            ite = 0
            train_loss_all = 0
            # for all datasets
            for dataindex in range(len(self.trainloader_list)):
                for data in self.trainloader_list[dataindex]:
                    data = list(data)
                    for key in data[0]:
                        if isinstance(data[0][key], list):
                            for i in range(len(data[0][key])):
                                data[0][key][i] = self.change_device(data[0][key][i], self.model_engine.local_rank)
                        else:
                            data[0] = self.change_device(data[0], self.model_engine.local_rank)
                        break
                    if data[1] is not None:
                        data[1] = torch.tensor(data[1].values).to(self.model_engine.local_rank)
                    logits, loss = self.model_engine(data[0], data[1], table_flag=dataindex)
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                    train_loss_all += loss.item()
                    ite += 1

            if self.testloader_list is not None:
                eval_res_list = self.evaluate()
                eval_res = np.mean(eval_res_list)
                model_save_path = self.output_dir + '/epoch_' + str(epoch+1) + '_' + 'valloss_' + str(eval_res)
                self.save_model(model_save_path)
                logger.info('epoch: {}, train loss: {:.4f}, test {}: {:.6f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch+1, train_loss_all, self.args['eval_metric_name'], eval_res, self.optimizer.param_groups[0]['lr'], time.time()-start_time))

        logger.info('training complete, cost {:.1f} secs.'.format(time.time()-start_time))
    
    def evaluate(self):
        # evaluate in each epoch
        self.model.eval()
        eval_res_list = []
        for dataindex in range(len(self.testloader_list)):
            y_test, pred_list, loss_list = [], [], []
            # self.testsampler_list[dataindex].set_epoch(epoch)
            for data in self.testloader_list[dataindex]:
                y_test.append(data[1])
                with torch.no_grad():
                    logits, loss = self.model(data[0], data[1], table_flag=dataindex)
                if loss is not None:
                    loss_list.append(loss.item())
                if logits is not None:
                    if logits.shape[-1] == 1: # binary classification
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else: # multi-class classification
                        pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())

            if len(pred_list)>0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()

            if self.args['eval_metric_name'] == 'val_loss':
                eval_res = np.mean(loss_list)
            else:
                y_test = pd.concat(y_test, 0)
                eval_res = self.args['eval_metric'](y_test, pred_all, self.model.num_class)

            eval_res_list.append(eval_res)
        return eval_res_list

    def change_device(self, data, dev):
        for key in data:
            if data[key] is not None:
                data[key] = data[key].to(dev)
        return data
    
    def save_model(self, output_dir=None): 
        if output_dir is None:
            print('no path assigned for save mode, default saved to ./ckpt/model.pt !')
            output_dir = self.output_dir

        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        # logger.info(f'saving model checkpoint to {output_dir}')
        self.model.save(output_dir)

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, constants.OPTIMIZER_NAME))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, constants.SCHEDULER_NAME))

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args['lr'])
            return optimizer_grouped_parameters

    def create_scheduler(self, num_training_steps, optimizer):
        self.lr_scheduler = get_scheduler(
            'cosine',
            optimizer = optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_num_epoch_steps(self, train_set_list, batch_size):
        total_step = 0
        for trainset in train_set_list:
            x_train, _ = trainset
            total_step += np.ceil(len(x_train) / batch_size)
        return total_step

    def get_warmup_steps(self, num_training_steps):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.args['warmup_steps'] if self.args['warmup_steps'] is not None else math.ceil(num_training_steps * self.args['warmup_ratio'])
        )
        return warmup_steps

    def _build_dataloader(self, trainset, batch_size, collator, shuffle=True):
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False,
            )
        return trainloader