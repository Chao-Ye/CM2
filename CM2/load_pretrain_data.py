import os
import pdb
import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class Feature_type_recognition():
    def __init__(self):
        self.df = None
    
    def get_data_type(self, col):
        if 'std' in self.df[col].describe():
            if self.df[col].nunique() < 15:
                return 'cat'
            return 'num'
        else:
            return 'cat'

    def fit(self, df):
        self.df = df.infer_objects()
        self.num = []
        self.cat = []
        self.bin = []
        for col in self.df.columns:
            cur_type = self.get_data_type(col)
            if (cur_type == 'num'):
                self.num.append(col)
            elif (cur_type == 'cat'):
                self.cat.append(col)
            else:
                raise RuntimeError('error feature type!')
        return self.cat, self.bin, self.num
    
    def check_class(self, data_path):
        self.df = pd.read_csv(data_path)
        
        target_type = self.get_data_type(self.df.columns.tolist()[-1])
        if target_type == 'cat':
            return True
        else:
            return False

def check_col_name_meaning(table_file, target, threshold=2):
    df = pd.read_csv(table_file)
    col_names = df.columns.values.tolist()
    col_names.remove(target)
    res = False
    good_cnt = 0
    one_cnt = 0
    for name in col_names:
        if len(name) <= 1:
            one_cnt += 1
            if one_cnt*2 >= len(col_names):
                return False
        if not name[-1].isdigit():
            good_cnt += 1
            if good_cnt>threshold or good_cnt==len(col_names):
                res = True
    return res

def get_col_type(col):
    if 'std' in col.describe():
        if col.nunique() < 15:
            return 'cat'
        return 'num'
    else:
        return 'cat'

def check_word_count(text):
    words = str(text).split()
    return len(words) >= 30

def check_data_quality(df):
    total_cells = df.size

    total_nulls = df.isnull().sum().sum()
    total_null_percentage = total_nulls / total_cells
    if total_null_percentage >= 0.2:
        return False

    specific_value = ['.', '#', 'null', 'NULL', '-', '*']
    specific_value_count = 0
    for val in specific_value:
        specific_value_count += (df == val).sum().sum()
    if specific_value_count >= 0.2:
        return False

    word_count_within_limit = df.applymap(check_word_count)
    if word_count_within_limit.any().any():
        return False

    if df.shape[1] <= 3:
        return False

    return True

def load_single_data(table_file, auto_feature_type, is_label=False, is_classify=False, seed=42, core_size=10000):
    if os.path.exists(table_file):
        print(f'load from local data dir {table_file}')
        df = pd.read_csv(table_file)

        if is_classify:
            target = df.columns.tolist()[-1]

            value_counts = df[target].value_counts()
            unique_values = value_counts[value_counts == 1].index
            df = df[~df[target].isin(unique_values)]

            y = df[target]
            X = df.drop([target], axis=1)

            if (X.shape[0] > core_size):
                sample_ratio = (core_size / X.shape[0])
                X, _, y, _ = train_test_split(X, y, train_size=sample_ratio, random_state=seed, stratify=y, shuffle=True)

            y = LabelEncoder().fit_transform(y.values)
            y = pd.Series(y, index=X.index)
        else:
            X = df
            if df.shape[0] > core_size:
                X = df.sample(n=core_size, random_state=seed)
            if is_label:
                target = df.columns.tolist()[-1]
                X = X.drop([target], axis=1)
            y = None

        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        if X.shape[1] > 1000:
            raise RuntimeError('too much features!')
        
        if not check_data_quality(X):
            raise RuntimeError('data quality is too poor!')

        # divide cat/bin/num feature
        cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)
        if len(cat_cols) > 0:
            for col in cat_cols: 
                X[col].fillna(X[col].mode()[0], inplace=True)       
            X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower())
        if len(num_cols) > 0:
            for col in num_cols: 
                X[col].fillna(X[col].mode()[0], inplace=True)       
            X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])
        
        # split train/val
        if is_classify:
            train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
        else:
            train_dataset, test_dataset = train_test_split(X, test_size=0.2, random_state=seed, shuffle=True)
            y_train = None
            y_test = None
    
        assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols)))
        return (train_dataset, y_train), (test_dataset, y_test), cat_cols, num_cols, bin_cols
    else:
        raise RuntimeError('no such data!')


def load_all_data(label_data_path=None, 
                  unlabel_data_path=None, 
                  seed=42, limit=10000, core_size=10000):
    
    num_col_list, cat_col_list, bin_col_list = [], [], []
    train_list, val_list = [], []
    data_weight = []
    auto_feature_type = Feature_type_recognition()

    label_data_list = os.listdir(label_data_path)
    unlabel_data_list = os.listdir(unlabel_data_path)

    for data in tqdm(unlabel_data_list, desc='load unlabel data'):
        if data[-3:]=='csv':
            data_path = unlabel_data_path + os.sep + data
        
            try:
                trainset, valset, cat_cols, num_cols, bin_cols = \
                load_single_data(data_path, auto_feature_type=auto_feature_type, seed=seed, core_size=core_size)
            except:
                continue

            num_col_list.append(num_cols)
            cat_col_list.append(cat_cols)
            bin_col_list.append(bin_cols)
            train_list.append(trainset)
            val_list.append(valset)
            data_weight.append(False)
            
            if len(train_list) >= limit-1:
                break
    
    
    for data in tqdm(label_data_list, desc='load label data'):
        if data[-3:]=='csv':
            data_path = label_data_path + os.sep + data

            try:
                trainset, valset, cat_cols, num_cols, bin_cols = \
                load_single_data(data_path, auto_feature_type=auto_feature_type, is_label=True, seed=seed, core_size=core_size)
            except:
                continue

            num_col_list.append(num_cols)
            cat_col_list.append(cat_cols)
            bin_col_list.append(bin_cols)
            train_list.append(trainset)
            val_list.append(valset)
            data_weight.append(True)

            if len(train_list) > limit-1:
                break
    
    print(f'all train data number:{len(train_list)}')
    
    
    return train_list, val_list, cat_col_list, num_col_list, bin_col_list, data_weight



    

