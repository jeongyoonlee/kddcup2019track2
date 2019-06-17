import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")

import copy
import logging
import numpy as np
import pandas as pd
from data import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from automl import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME, CATEGORY_PREFIX, MULTI_CAT_PREFIX, NUMERICAL_PREFIX, TIME_PREFIX
from merge import merge_table
from preprocess import clean_df, clean_tables
from util import Config, log, show_dataframe, timeit


class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None
        self.enc = None
        self.cat_cols = []
        self.mcat_cols = []
        self.num_cols = []
        self.ts_cols = []
        self.ts_col = self.config['time_col']
        self.count_dict = {}

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)

        self.y = y

    @timeit
    def feature_engineer(self, X, y=None, train=True):
        # Use the hour of the day as a feature
        log('add hour, month, day, weekday features')
        for col in self.ts_cols:
            X.loc[:, col] = pd.to_datetime(X[col])
            X.loc[:, '%s_hour' % col] = X[col].dt.hour
            X.loc[:, '%s_month' % col] = X[col].dt.month
            X.loc[:, '%s_day' % col] = X[col].dt.day
            X.loc[:, '%s_weekday' % col] = X[col].dt.weekday

            max_dt = X[col].max()
            X.loc[:, '{}_diff_from_max'.format(col)] = (max_dt - X[col]).dt.total_seconds() // 60

        X.drop(self.ts_cols, axis=1, inplace=True)

        s = (X.nunique() == 1)
        cols_to_drop = s[s].index.tolist()
        log('dropping constant columns: {}'.format(cols_to_drop))
        X.drop(cols_to_drop, axis=1, inplace=True)

        self.cat_cols = sorted([c for c in self.cat_cols if c not in cols_to_drop])
        self.mcat_cols = sorted([c for c in self.mcat_cols if c not in cols_to_drop])
        self.num_cols = sorted([c for c in self.num_cols if c not in cols_to_drop])

        # Label encode categorical features
        log('label encoding categorical features')
        if train:
            self.enc = LabelEncoder(min_obs=X.shape[0] * .0001)
            X.loc[:, self.cat_cols] = self.enc.fit_transform(X[self.cat_cols])
        else:
            assert self.enc is not None
            X.loc[:, self.cat_cols] = self.enc.transform(X[self.cat_cols])

        # Generate count features for categorical columns

        '''
        for c in self.cat_cols[:2]:
            if c.find('.') >= 0:
                continue
            if train:
                self.count_dict[c] = X[c].value_counts().to_frame(name='%s_count' % c.split('.')[-1]).reset_index().rename(columns={'index': c})
            if (c in self.count_dict):
                X = X.reset_index().merge(self.count_dict[c], how='left', on=c).set_index('index')
        '''

        # Generate count features for multi-categorical columns
        '''
        for c in self.mcat_cols:
            if train:
                self.count_dict[c] = X[c].value_counts().to_frame(name='%s_count' % c.split('.')[-1]).reset_index().rename(columns={'index': c})

            if (c in self.count_dict):
                X = X.reset_index().merge(self.count_dict[c], how='left', on=c).set_index('index')
        '''

        # Use the count of categories for each observation
        log('count encoding multi-cat features')
        for col in self.mcat_cols:
            X.loc[:, col] = X[col].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

        return X

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]

        main_table['y_sorted'] = self.y
        main_table.sort_values(self.ts_col, inplace=True)
        y_trn = main_table.y_sorted.copy()
        main_table.drop('y_sorted', axis=1, inplace=True)

        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)

        self.cat_cols = sorted([c for c in X.columns if c.startswith(CATEGORY_PREFIX)])
        self.mcat_cols = sorted([c for c in X.columns if c.startswith(MULTI_CAT_PREFIX)])
        self.num_cols = sorted([c for c in X.columns if c.startswith(NUMERICAL_PREFIX)])
        self.ts_cols = sorted([c for c in X.columns if c.startswith(TIME_PREFIX)])

        X = self.feature_engineer(X, train=True)

        X_trn = X[X.index.str.startswith("train")]
        X_trn.index = X_trn.index.map(lambda x: int(x.split('_')[1]))

        train(X_trn, y_trn, self.config)

        X_tst = X[X.index.str.startswith("test")]
        X_tst.index = X_tst.index.map(lambda x: int(x.split('_')[1]))
        X_tst.sort_index(inplace=True)
        result = predict(X_tst, self.config)

        return pd.Series(result)
