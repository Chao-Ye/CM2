import argparse
import logging
import os
import shutil

import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from CM2.dataset_openml import load_single_data_all

import CM2

import warnings
warnings.filterwarnings("ignore")

# set random seed
CM2.random_seed(42)

cal_device = 'cuda'

def log_config(args):
    """
    log Configuration information, specifying the saving path of output log file, etc
    :return: None
    """
    log_name = args.log_name
    exp_dir = 'search_{}_{}'.format(log_name, time.strftime("%Y%m%d-%H%M%S"))
    exp_log_dir = Path('logs') / exp_dir
    # save argss
    setattr(args, 'exp_log_dir', exp_log_dir)

    if not os.path.exists(exp_log_dir):
        os.mkdir(exp_log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(exp_log_dir / 'log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

def parse_args():
    parser = argparse.ArgumentParser(description='CM2-sup-scratch')
    parser.add_argument('--log_name', type=str, default="CM2_scratch", help='task name')
    parser.add_argument('--task_data', type=str, default="./example/cmc.csv", help='task dataset')
    args = parser.parse_args()
    return args

_args = parse_args()
log_config(_args)

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
all_res = {}

task_dataset = _args.task_data.split(',')

for table_file_path in task_dataset:
    data_name = table_file_path.split('/')[-1]
    logging.info(f'Start========>{data_name}_DataSet==========>')
    X, y, cat_cols, num_cols, bin_cols = load_single_data_all(table_file_path)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    num_class = len(y.value_counts())
    logging.info(f'num_class : {num_class}')
    cat_cols = [cat_cols]
    num_cols = [num_cols]
    bin_cols = [bin_cols]
    idd = 0
    score_list = []
    for trn_idx, val_idx in skf.split(X, y):
        CM2.random_seed(42)
        idd += 1
        train_data = X.loc[trn_idx]
        train_label = y[trn_idx]
        X_test = X.loc[val_idx]
        y_test = y[val_idx]
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=0, stratify=train_label, shuffle=True)
        model = CM2.build_classifier(
            cat_cols, num_cols, bin_cols,
            device=cal_device,
            num_class=num_class,
            num_layer=3,

            vocab_freeze=True,
            hidden_dropout_prob=0.1,
            use_bert=True,
        )
        training_arguments = {
            'num_epoch':300,
            'batch_size':64,
            'lr':1e-4,
            'eval_metric':'auc',
            'eval_less_is_better':False,
            'output_dir':f'./models/checkpoint-scratch',
            'patience':30,
            'num_workers':0,
            'device':cal_device,
            'flag':1,
            'warmup_steps':5,
        }
        logging.info(training_arguments)
        if os.path.isdir(training_arguments['output_dir']):
            shutil.rmtree(training_arguments['output_dir'])
        trainer = CM2.train(model, (X_train, y_train), (X_val, y_val), data_weight=[True], **training_arguments)
        eval_res_list = trainer.train((X_test, y_test))

        ypred = CM2.predict(model, X_test)
        ans = CM2.evaluate(ypred, y_test, metric='auc', num_class=num_class)
        # assembling the top 5 models on the validation set
        ans[0] = max(ans[0], max(eval_res_list[-5:]))
        score_list.append(ans[0])
        logging.info(f'Test_Score_{idd}===>{data_name}_DataSet==> {ans[0]}')
    all_res[data_name] = np.mean(score_list)
    logging.info(f'Test_Score_5_fold===>{data_name}_DataSet==> {np.mean(score_list)}')

mean_list = []
for key in all_res:
    logging.info(f'meaning_5_fold=>{all_res[key]}=>{key}')
    mean_list.append(all_res[key])
result_df = pd.DataFrame(mean_list, columns=['result'])
res_path = str(_args.exp_log_dir) + os.sep +'res.csv'
result_df.to_csv(res_path, index=False)
logging.info(f'meaning all data=>{np.mean(mean_list)}')