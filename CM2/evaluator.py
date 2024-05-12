import logging
from collections import defaultdict
import os
import pdb

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

from . import constants

def predict(clf, 
    x_test,
    y_test=None,
    return_loss=False,
    eval_batch_size=256,
    table_flag=0,
    regression_task=False,
    ):
    clf.eval()
    pred_list, loss_list = [], []
    x_test = clf.input_encoder.feature_extractor(x_test, table_flag=table_flag)
    if x_test['x_cat_input_ids'] is not None:
        x_len = x_test['x_cat_input_ids'].shape[0]
    else:
        x_len = x_test['x_num'].shape[0]
    for i in range(0, x_len, eval_batch_size):
        if x_test['x_cat_input_ids'] is not None:
            x_cat_input_ids = x_test['x_cat_input_ids'][i:i+eval_batch_size]
            x_cat_att_mask = x_test['x_cat_att_mask'][i:i+eval_batch_size]
            col_cat_input_ids = x_test['col_cat_input_ids']
            col_cat_att_mask = x_test['col_cat_att_mask']
        else:
            x_cat_input_ids = None
            x_cat_att_mask = None
            col_cat_input_ids = None
            col_cat_att_mask = None

        if x_test['x_num'] is not None:
            x_num = x_test['x_num'][i:i+eval_batch_size]
            num_col_input_ids = x_test['num_col_input_ids']
            num_att_mask = x_test['num_att_mask']
        else:
            x_num = None        
            num_col_input_ids = None
            num_att_mask = None

        bs_x_test = {'x_cat_input_ids': x_cat_input_ids,
                  'x_cat_att_mask': x_cat_att_mask,
                  'x_num': x_num,
                  'col_cat_input_ids': col_cat_input_ids,
                  'col_cat_att_mask': col_cat_att_mask,
                  'num_col_input_ids': num_col_input_ids,
                  'num_att_mask': num_att_mask
        }
        # bs_x_test = x_test.iloc[i:i+eval_batch_size]
        with torch.no_grad():
            logits, loss = clf(bs_x_test, y_test, table_flag=table_flag)
        
        if loss is not None:
            loss_list.append(loss.item())
        
        if regression_task:
            pred_list.append(logits.detach().cpu().numpy())
        elif logits.shape[-1] == 1: # binary classification
            pred_list.append(logits.sigmoid().detach().cpu().numpy())
        else: # multi-class classification
            pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())
    pred_all = np.concatenate(pred_list, 0)
    if logits.shape[-1] == 1:
        pred_all = pred_all.flatten()

    if return_loss:
        avg_loss = np.mean(loss_list)
        return avg_loss
    else:
        return pred_all

def evaluate(ypred, y_test, metric='auc', num_class=2, seed=123, bootstrap=False):
    np.random.seed(seed)
    eval_fn = get_eval_metric_fn(metric)
    res_list = []
    stats_dict = defaultdict(list)
    if bootstrap:
        for i in range(10):
            sub_idx = np.random.choice(np.arange(len(ypred)), len(ypred), replace=True)
            sub_ypred = ypred[sub_idx]
            sub_ytest = y_test.iloc[sub_idx]
            try:
                sub_res = eval_fn(sub_ytest, sub_ypred)
            except ValueError:
                print('evaluation went wrong!')
            stats_dict[metric].append(sub_res)
        for key in stats_dict.keys():
            stats = stats_dict[key]
            alpha = 0.95
            p = ((1-alpha)/2) * 100
            lower = max(0, np.percentile(stats, p))
            p = (alpha+((1.0-alpha)/2.0)) * 100
            upper = min(1.0, np.percentile(stats, p))
            print('{} {:.2f} mean/interval {:.4f}({:.2f})'.format(key, alpha, (upper+lower)/2, (upper-lower)/2))
            if key == metric: res_list.append((upper+lower)/2)
    else:
        res = eval_fn(y_test, ypred, num_class)
        res_list.append(res)
    # logging.info(f'Score========>{res_list}')
    return res_list

def get_eval_metric_fn(eval_metric):
    fn_dict = {
        'acc': acc_fn,
        'auc': auc_fn,
        'mse': mse_fn,
        'r2': r2_fn,
        'rae': rae_fn,
        'rmse': rmse_fn,
        'val_loss': None,
    }
    return fn_dict[eval_metric]

def acc_fn(y, p, num_class=2):
    if num_class==2:
        y_p = (p >= 0.5).astype(int)
    else:
        y_p = np.argmax(p, -1)
    return accuracy_score(y, y_p)

def auc_fn(y, p, num_class=2):
    if num_class > 2:
        return roc_auc_score(y, p, multi_class='ovo')
    else:
        return roc_auc_score(y, p)

def mse_fn(y, p, num_class=None):
    return mean_squared_error(y, p)

def r2_fn(y, p, num_class=None):
    y = y.values
    return r2_score(y, p)

def rae_fn(y_true: np.ndarray, y_pred: np.ndarray, num_class=None):
    y_true = y_true.values
    up = np.abs(y_pred - y_true).sum()
    down = np.abs(y_true.mean() - y_true).sum()
    score = 1 - up / down
    return score

def rmse_fn(y, p, num_class=None):
    return np.sqrt(mean_squared_error(y, p))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, output_dir='ckpt', trace_func=print, less_is_better=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = output_dir
        self.trace_func = trace_func
        self.less_is_better = less_is_better

    def __call__(self, val_loss, model):
        if self.patience < 0: # no early stop
            self.early_stop = False
            return
        
        if self.less_is_better:
            score = val_loss
        else:    
            score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            # if self.counter%10 == 0:
            #     return True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, constants.WEIGHTS_NAME))
        self.val_loss_min = val_loss

