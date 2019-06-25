import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")

import gc

import copy
from itertools import combinations
import logging
import numpy as np
import pandas as pd
from data import LabelEncoder

from automl import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME, CATEGORY_PREFIX, MULTI_CAT_PREFIX, NUMERICAL_PREFIX, TIME_PREFIX
from merge import merge_table
from preprocess import clean_df, clean_tables
from util import Config, log, show_dataframe, timeit, check_imbalance_data


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

        self.prev_cat_cnt = None
        self.prev_multi_cat_cnt = None
        self.prev_uniq_cat_cnt = None

    @timeit
    def count_categorical(self, df, prev_cnt, mcat = False):
        if mcat == True:
            columns = self.mcat_cols
        else:
            columns = self.cat_cols

        for c in columns:
            cnt = df[c].value_counts().to_frame(name='%s_count' % c).reset_index().rename(columns={'index': c})

            if prev_cnt is None:
                prev_cnt = {}
                prev_cnt[str(c)] = cnt
            elif not str(c) in prev_cnt:
                prev_cnt[str(c)] = cnt
            else:
                prev_cnt[str(c)] = prev_cnt[str(c)].append(cnt)
                prev_cnt[str(c)] = prev_cnt[str(c)].groupby(c)['%s_count' % c].mean().to_frame(name='%s_count' % c).reset_index().rename(columns={'index': c})

            df = df.merge(prev_cnt[str(c)], how='left', on=c)

        return df, prev_cnt

    @timeit
    def uniq_categorical(self, df, prev_uniq_cnt, mcat = False):
        if mcat == True:
            m_cat_cols = self.mcat_cols
        else:
            m_cat_cols = self.cat_cols

        if len(m_cat_cols) > 1:
            m_cat_cols = m_cat_cols[0:1]

        for c in m_cat_cols:
            # count unique
            for magic in self.num_cols:
                cnt = df.groupby(c)[magic].nunique().to_frame(name='%s_%s_unique' % (c,magic)).reset_index().rename(columns={'index': c})
            if prev_uniq_cnt is None:
                prev_uniq_cnt = {}
                prev_uniq_cnt[str(c)] = {magic: cnt}
            elif not str(c) in prev_uniq_cnt:
                prev_uniq_cnt[str(c)] = {magic: cnt}
            elif not magic in prev_uniq_cnt[str(c)]:
                prev_uniq_cnt[str(c)] = {magic: cnt}
            else:
                prev_uniq_cnt[str(c)][magic] = prev_uniq_cnt[str(c)][magic].append(cnt)
                prev_uniq_cnt[str(c)][magic] = prev_uniq_cnt[str(c)][magic].groupby(c)['%s_%s_unique' % (c,magic)].mean().to_frame(name='%s_%s_unique' % (c,magic)).reset_index().rename(columns={'index': c})
            df = df.merge(prev_uniq_cnt[str(c)][magic], how='left', on=c)
            df['%s_%s_unique' % (c,magic)] = df['%s_%s_unique' % (c,magic)].rank() / df.shape[0]
            df['%s_%s_ratio' % (c,magic)] = df['%s_%s_unique' % (c,magic)] / df['%s_count' % c]

        return df, prev_uniq_cnt

    @timeit
    def aggregate_cat_cols_on_time_features(self, df, mcat = False):
        if len(self.ts_cols) == 0:
            return df

        if mcat == True:
            m_cat_cols = self.mcat_cols
        else:
            m_cat_cols = self.cat_cols

        for c in m_cat_cols:
            try:
                aggr = df.groupby(c)[self.ts_cols].agg(['min', 'max', 'std', 'sum', 'nunique'])
                aggr.columns = ['%s_%s' % (c, '_'.join(map(str,col))) for col in aggr.columns.values]
                aggr = aggr.reset_index()
                df = df.merge(aggr, how='left', on=[c])
            except:
                log('Failed: {}'.format(c))
                pass

        return df

    @timeit
    def feature_engineer(self, X, y=None):
        log('memory usage of X: {:.2f}MB'.format(X.memory_usage().sum() // 1e6))

        X, self.prev_cat_cnt = self.count_categorical(X, self.prev_cat_cnt, mcat=False)
        X, self.prev_multi_cat_cnt = self.count_categorical(X, self.prev_multi_cat_cnt, mcat=True)

        # Use the hour of the day as a feature
        self.ts_cols = sorted([c for c in X.columns if c.startswith(TIME_PREFIX)])

        for col in self.ts_cols:
            log('memory usage of X: {:.2f}MB'.format(X.memory_usage().sum() // 1e6))
            log('adding datetime features for {}'.format(col))
            s = X[col] = pd.to_datetime(X[col])
            s_n = s.dt.minute
            if s_n.min() != s_n.max():
                X.loc[:, '{}_minute'.format(col)] = s_n
            s_n = s.dt.hour
            if s_n.min() != s_n.max():
                X.loc[:, '{}_hour'.format(col)] = s_n
            s_n = s.dt.month
            if s_n.min() != s_n.max():
                X.loc[:, '{}_month'.format(col)] = s_n
            s_n = s.dt.weekday
            if s_n.min() != s_n.max():
                X.loc[:, '{}_weekday'.format(col)] = s_n

            log('adding the diff-from-max feature for {}'.format(col))
            max_ts = s.max()
            s_n = (max_ts - s).dt.total_seconds() // 60
            if s_n.min() != s_n.max():
                X.loc[:, '{}_diff_from_max'.format(col)] = s_n

        for ts_col1, ts_col2 in combinations(self.ts_cols, 2):
            log('adding the diff between {} and {}'.format(ts_col1, ts_col2))
            s = (X[ts_col1] - X[ts_col2]).dt.total_seconds() // 60
            if s.min() != s.max():
                X.loc[:, '{}_minus_{}'.format(ts_col1, ts_col2)] = s

        log('memory usage of X: {:.2f}MB'.format(X.memory_usage().sum() // 1e6))

        log('dropping timeseries features')
        X.drop(self.ts_cols, axis=1, inplace=True)
        log('memory usage of X: {:.2f}MB'.format(X.memory_usage().sum() // 1e6))

        self.cat_cols = sorted([c for c in X.columns if c.startswith(CATEGORY_PREFIX)])
        self.mcat_cols = sorted([c for c in X.columns if c.startswith(MULTI_CAT_PREFIX)])
        self.num_cols = sorted([c for c in X.columns if c.startswith(NUMERICAL_PREFIX)])

        # Label encode categorical features
        log('label encoding categorical features: {}'.format(self.cat_cols))
        self.enc = LabelEncoder(min_obs=X.shape[0] * .0001)
        X.loc[:, self.cat_cols] = self.enc.fit_transform(X[self.cat_cols])

        # Use the count of categories for each observation
        log('adding count features for multi-cat features: {}'.format(self.mcat_cols))
        for col in self.mcat_cols:
            X.loc[:, col] = X[col].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

        log('Features list: {}'.format(X.columns.tolist()))
        log('memory usage of X: {:.2f}MB'.format(X.memory_usage().sum() // 1e6))

        return X

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)
        self.y = y

    @timeit
    def predict(self, X_test, time_remain):
        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]

        main_table['y_sorted'] = self.y
        main_table.sort_values(self.ts_col, inplace=True)

        X_test['y_sorted'] = -1
        main_table = pd.concat([main_table, X_test], ignore_index=True).reset_index(drop=True)

        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)

        del Xs, main_table, X_test
        gc.collect()

        X = self.feature_engineer(X)

        X_trn = X[X['y_sorted'] != -1]
        y_trn = X_trn.y_sorted.copy()
        X_trn = X_trn.drop('y_sorted', axis=1)

        train(X_trn, y_trn, self.config)

        X_tst = X[X['y_sorted'] == -1]
        X_tst = X_tst.drop('y_sorted', axis=1)

        X_tst.sort_index(inplace=True)
        result = predict(X_tst, self.config)

        return pd.Series(result)
