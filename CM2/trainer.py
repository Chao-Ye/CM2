import os
import shutil
import pdb
import math
import time
import json
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm.autonotebook import trange
from loguru import logger
import logging

from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR, CosineAnnealingLR

from . import constants
from .evaluator import predict, get_eval_metric_fn, EarlyStopping, evaluate
from .modeling_CM2 import CM2FeatureExtractor
from .trainer_utils import SupervisedTrainCollator, TrainDataset
from .trainer_utils import get_parameter_names
from .trainer_utils import get_scheduler, LinearWarmupScheduler


class Trainer:
    def __init__(self,
        model,
        train_set_list,
        test_set_list=None,
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
        ignore_duplicate_cols=True,
        eval_metric='auc',
        eval_less_is_better=False,
        num_workers=0,
        regression_task=False,
        flag=0,
        data_weight=None,
        device=None,
        **kwargs,
        ):
        self.flag = flag
        self.model = model
        self.device = device
        self.data_weight = data_weight
        if isinstance(train_set_list, tuple): train_set_list = [train_set_list]
        if isinstance(test_set_list, tuple): test_set_list = [test_set_list]

        new_train_list = []
        new_test_list = []
        self.collate_fn = collate_fn
        self.regression_task = regression_task
        if collate_fn is None:
            self.collate_fn = SupervisedTrainCollator(
                categorical_columns=model.categorical_columns,
                numerical_columns=model.numerical_columns,
                binary_columns=model.binary_columns,
                ignore_duplicate_cols=ignore_duplicate_cols,
            )

        self.feature_extractor = CM2FeatureExtractor(
            categorical_columns=model.categorical_columns,
            numerical_columns=model.numerical_columns,
            binary_columns=model.binary_columns,
            disable_tokenizer_parallel=True,
            ignore_duplicate_cols=ignore_duplicate_cols,
        )       
        # prepare collate_fn for all train datasets once        
        for dataindex, trainset in enumerate(train_set_list):
            x = trainset[0]
            if trainset[1] is not None:
                y = trainset[1]
            else:
                y = None
            inputs = self.feature_extractor(x, table_flag=dataindex)            
            new_train_list.append((inputs, y))
        self.trainloader_list = [
            self._build_dataloader((trainset, dataindex), batch_size=batch_size, collator=self.collate_fn, num_workers=num_workers)
            for dataindex, trainset in enumerate(new_train_list)
        ]
        # prepare collate_fn for test/val datasets once 
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
                self._build_dataloader((testset, dataindex), batch_size=eval_batch_size, collator=self.collate_fn, num_workers=num_workers, shuffle=False) 
                for dataindex, testset in enumerate(new_test_list)
            ]
        else:
            self.testloader_list = None
        
        self.train_set_list = new_train_list
        self.test_set_list = new_test_list
        self.output_dir = output_dir
        self.early_stopping = EarlyStopping(output_dir=output_dir, patience=patience, verbose=False, less_is_better=eval_less_is_better)
        self.args = {
            'lr':lr,
            'weight_decay':weight_decay,
            'batch_size':batch_size,
            'num_epoch':num_epoch,
            'eval_batch_size':eval_batch_size,
            'warmup_ratio': warmup_ratio,
            'warmup_steps': warmup_steps,
            'num_training_steps': self.get_num_train_steps(train_set_list, num_epoch, batch_size),
            'eval_metric': get_eval_metric_fn(eval_metric),
            'eval_metric_name': eval_metric,
        }
        self.args['steps_per_epoch'] = int(self.args['num_training_steps'] / (num_epoch*len(self.train_set_list)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.optimizer = None
        self.lr_scheduler = None
        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last

    def train(self, eval_data=None):
        args = self.args
        self.create_optimizer()
        # self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.96)
        # self.lr_scheduler = StepLR(self.optimizer, step_size=5, gamma=0.95)
        # self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30], gamma=0.8)
        # self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=5, eta_min=0)
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:

            logger.info(f'set warmup training in initial {args["warmup_steps"]} steps')
            self.lr_scheduler = LinearWarmupScheduler(
                optimizer=self.optimizer, 
                base_lr=self.args['lr'],
                warmup_epochs=args['warmup_steps'],
            )
            self.lr_scheduler.init_optimizer()

        start_time = time.time()
        real_res_list = []
        for epoch in trange(args['num_epoch'], desc='Epoch'):
            ite = 0
            train_loss_all = 0
            # for all datasets
            self.model.train()
            for dataindex in range(len(self.trainloader_list)):
                # for each batch of one dataset
                for data in self.trainloader_list[dataindex]:
                    self.optimizer.zero_grad()                    
                    for key in data[0]:
                        if isinstance(data[0][key], list):
                            for i in range(len(data[0][key])):
                                data[0][key][i] = self.change_device(data[0][key][i], self.device)
                        else:
                            data[0] = self.change_device(data[0], self.device)
                        break
                    if data[1] is not None:
                        data[1] = torch.tensor(data[1].values).to(self.device)
                    logits, loss = self.model(data[0], data[1], table_flag=dataindex)
                    # print(f'{dataindex} :::: {loss.item()}')
                    loss.backward()
                    self.optimizer.step()
                    train_loss_all += loss.item()
                    ite += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(cur_epoch=epoch)


            if self.test_set_list is not None:
                eval_res_list = self.evaluate()
                eval_res = np.mean(eval_res_list)
                # print('epoch: {}, test {}: {:.6f}'.format(epoch, self.args['eval_metric_name'], eval_res))
                if self.early_stopping(-eval_res, self.model) and eval_data:
                    if self.regression_task:
                        ypred = predict(self.model, eval_data[0], regression_task=True)
                        ans = evaluate(ypred, eval_data[1], metric='rmse')
                    else:
                        ypred = predict(self.model, eval_data[0])
                        ans = evaluate(ypred, eval_data[1], metric='auc', num_class=self.model.num_class)
                    real_res_list.append(ans[0])
                    # logging.info(f'eval_res_list: {real_res_list}')
                if self.early_stopping.early_stop:
                    logging.info('early stopped')
                    break
                logging.info('epoch: {}, train loss: {:.4f}, test {}: {:.6f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch, train_loss_all, self.args['eval_metric_name'], eval_res, self.optimizer.param_groups[0]['lr'], time.time()-start_time))
            else:
                logging.info('epoch: {}, train loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch, train_loss_all, self.optimizer.param_groups[0]['lr'], time.time()-start_time))

        if os.path.exists(self.output_dir):
            if self.test_set_list is not None:
                # load checkpoints
                logger.info(f'load best at last from {self.output_dir}')
                state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

        logger.info('training complete, cost {:.1f} secs.'.format(time.time()-start_time))
        return real_res_list

    def change_device(self, data, dev):
        for key in data:
            if data[key] is not None:
                data[key] = data[key].to(dev)
        return data

    def save_epoch(self):
        save_path = './openml_pretrain_model'
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        self.save_model(save_path)
            
    def evaluate(self):
        # evaluate in each epoch
        self.model.eval()
        eval_res_list = []
        for dataindex in range(len(self.testloader_list)):
            y_test, pred_list, loss_list = [], [], []
            for data in self.testloader_list[dataindex]:
                y_test.append(data[1])
                with torch.no_grad():
                    logits, loss = self.model(data[0], data[1], table_flag=dataindex)
                if loss is not None:
                    loss_list.append(loss.item())
                if logits is not None:
                    if self.regression_task:
                        pred_list.append(logits.detach().cpu().numpy())
                    elif logits.shape[-1] == 1: # binary classification
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
                if self.regression_task:
                    eval_res = self.args['eval_metric'](y_test, pred_all)
                else:
                    eval_res = self.args['eval_metric'](y_test, pred_all, self.model.num_class)

            eval_res_list.append(eval_res)

        return eval_res_list

    def train_no_dataloader(self,
        resume_from_checkpoint = None,
        ):
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint
        args = self.args
        self.create_optimizer()
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:
            print('set warmup training.')
            self.create_scheduler(args['num_training_steps'], self.optimizer)

        for epoch in range(args['num_epoch']):
            ite = 0
            # go through all train sets
            for train_set in self.train_set_list:
                x_train, y_train = train_set
                train_loss_all = 0
                for i in range(0, len(x_train), args['batch_size']):
                    self.model.train()
                    if self.balance_sample:
                        bs_x_train_pos = x_train.loc[y_train==1].sample(int(args['batch_size']/2))
                        bs_y_train_pos = y_train.loc[bs_x_train_pos.index]
                        bs_x_train_neg = x_train.loc[y_train==0].sample(int(args['batch_size']/2))
                        bs_y_train_neg = y_train.loc[bs_x_train_neg.index]
                        bs_x_train = pd.concat([bs_x_train_pos, bs_x_train_neg], axis=0)
                        bs_y_train = pd.concat([bs_y_train_pos, bs_y_train_neg], axis=0)
                    else:
                        bs_x_train = x_train.iloc[i:i+args['batch_size']]
                        bs_y_train = y_train.loc[bs_x_train.index]

                    self.optimizer.zero_grad()
                    logits, loss = self.model(bs_x_train, bs_y_train)
                    loss.backward()

                    self.optimizer.step()
                    train_loss_all += loss.item()
                    ite += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            if self.test_set is not None:
                # evaluate in each epoch
                self.model.eval()
                x_test, y_test = self.test_set
                pred_all = predict(self.model, x_test, self.args['eval_batch_size'])
                eval_res = self.args['eval_metric'](y_test, pred_all)
                print('epoch: {}, test {}: {}'.format(epoch, self.args['eval_metric_name'], eval_res))
                self.early_stopping(-eval_res, self.model)
                if self.early_stopping.early_stop:
                    print('early stopped')
                    break

            print('epoch: {}, train loss: {}, lr: {:.6f}'.format(epoch, train_loss_all, self.optimizer.param_groups[0]['lr']))

        if os.path.exists(self.output_dir):
            if self.test_set is not None:
                # load checkpoints
                print('load best at last from', self.output_dir)
                state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

    def save_model(self, output_dir=None): 
        if output_dir is None:
            print('no path assigned for save mode, default saved to ./ckpt/model.pt !')
            output_dir = self.output_dir

        if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        logger.info(f'saving model checkpoint to {output_dir}')
        self.model.save(output_dir)

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, constants.OPTIMIZER_NAME))
        # if self.lr_scheduler is not None:
        #     torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, constants.SCHEDULER_NAME))


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

    def create_scheduler(self, num_training_steps, optimizer):
        self.lr_scheduler = get_scheduler(
            'cosine',
            optimizer = optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_num_train_steps(self, train_set_list, num_epoch, batch_size):
        total_step = 0
        for trainset in train_set_list:
            x_train, _ = trainset
            total_step += np.ceil(len(x_train) / batch_size)
        total_step *= num_epoch
        return total_step

    def get_warmup_steps(self, num_training_steps):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.args['warmup_steps'] if self.args['warmup_steps'] is not None else math.ceil(num_training_steps * self.args['warmup_ratio'])
        )
        return warmup_steps

    def _build_dataloader(self, trainset, batch_size, collator, num_workers=8, shuffle=True):
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            )
        return trainloader






