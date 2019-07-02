import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")
# os.system("pip3 install psutil")

# check before install to save running time
# try:
#     from loguru import logger
# except Exception as e:
#     os.system("pip3 install loguru")
#     from loguru import logger

#try:
#    import hyperopt
#except Exception as e:
#    os.system("pip3 install hyperopt")
#    import hyperopt

#try:
#    import lightgbm as lgb
#except Exception as e:
#    os.system("pip3 install lightgbm")
#    import lightgbm as lgb

#try:
#    os.system("pip3 install pandas==0.24.2")
#    import pandas as pd
#    #if pd.__version__ < "0.24.2":
#    #    os.system("pip3 install pandas --upgrade") # should not use as taking too many time to upgrade
#    #    # os.system("pip3 install pandas==0.24.2")
#    #    import pandas as pd
#except Exception as e:
#    os.system("pip3 install pandas==0.24.2")
#    import pandas as pd
#-----------------------------------------

import copy
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from data import LabelEncoder

from automl import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME, CATEGORY_PREFIX, MULTI_CAT_PREFIX, NUMERICAL_PREFIX, TIME_PREFIX, MEMORY_LIMIT
from merge import merge_table
from preprocess import clean_df, clean_tables, feature_engineer as fe
from util import Config, log, show_dataframe, timeit, df_memory_usage, get_process_memory

import gc

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
            # cnt[c] = cnt[c].astype('category')

            if prev_cnt is None:
                prev_cnt = {}
                prev_cnt[str(c)] = cnt
            elif not str(c) in prev_cnt:
                prev_cnt[str(c)] = cnt
            else:
                # print("...Mean with previous count")
                prev_cnt[str(c)] = prev_cnt[str(c)].append(cnt)
                prev_cnt[str(c)] = prev_cnt[str(c)].groupby(c)['%s_count' % c].mean().to_frame(name='%s_count' % c).reset_index().rename(columns={'index': c})

            df = df.merge(prev_cnt[str(c)], how='left', on=c)
            # checked with D, not improve
            # if mcat == True:
            #     df['%s_count_items' % c] = df[c].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

            # df = df.drop(c, axis=1)

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
                # print("...Mean with previous count")
                # print("col", c, magic, prev_uniq_cnt[str(c)])
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
    def feature_engineer(self, X, y=None, train=True):
        log('memory usage of X before count_categorical for CAT: {:.2f}MB'.format(df_memory_usage(X) // 1e6))
        log('memory usage of process before count_categorical for CAT: {:.2f}MB'.format(get_process_memory()))
        if get_process_memory() < MEMORY_LIMIT:
            X, self.prev_cat_cnt = self.count_categorical(X, self.prev_cat_cnt, mcat = False)
        
        log('memory usage of X before count_categorical for MUL-CAT: {:.2f}MB'.format(df_memory_usage(X) // 1e6))
        log('memory usage of process before count_categorical for MUL-CAT: {:.2f}MB'.format(get_process_memory()))
        if get_process_memory() < MEMORY_LIMIT:
            X, self.prev_multi_cat_cnt = self.count_categorical(X, self.prev_multi_cat_cnt, mcat = True)
        
        log('memory usage of X before uniq_categorical for CAT: {:.2f}MB'.format(df_memory_usage(X) // 1e6))
        log('memory usage of process before uniq_categorical for CAT: {:.2f}MB'.format(get_process_memory()))
        if get_process_memory() < MEMORY_LIMIT:
            X, self.prev_uniq_cat_cnt = self.uniq_categorical(X, self.prev_uniq_cat_cnt, mcat = False)  

        #X = self.aggregate_cat_cols_on_time_features(X, mcat = False)

        # Use the hour of the day as a feature
        log('memory usage of X before time calc: {:.2f}MB'.format(df_memory_usage(X) // 1e6))
        log('memory usage of process before time calc: {:.2f}MB'.format(get_process_memory()))
        if get_process_memory() < MEMORY_LIMIT:
            for col in self.ts_cols:
                X[col] = pd.to_datetime(X[col])
                X.loc[:, '%s_minute' % col] = X[col].dt.minute
                X.loc[:, '%s_hour' % col] = X[col].dt.hour
                X.loc[:, '%s_year' % col] = X[col].dt.year
                X.loc[:, '%s_quarter' % col] = X[col].dt.quarter
                X.loc[:, '%s_month' % col] = X[col].dt.month
                X.loc[:, '%s_day' % col] = X[col].dt.day
                X.loc[:, '%s_weekday' % col] = X[col].dt.weekday

                max_ts = X[col].max()
                X.loc[:, '{}_diff_from_max'.format(col)] = (max_ts - X[col]).dt.total_seconds() // 60

                #if col != self.ts_col:
                #    X.loc[:, '{}_diff_from_ts'.format(col)] = X[[self.ts_col, col]].apply(lambda x: (x[0]-x[1]).total_seconds() // 60, axis=1)
        log('memory usage of X before diff between calc: {:.2f}MB'.format(df_memory_usage(X) // 1e6))
        log('memory usage of process before diff between calc: {:.2f}MB'.format(get_process_memory()))
        if get_process_memory() < MEMORY_LIMIT:
            for ts_col1, ts_col2 in combinations(self.ts_cols, 2):
                log('adding the diff between {} and {}'.format(ts_col1, ts_col2))
                s = (X[ts_col1] - X[ts_col2]).dt.total_seconds() // 60
                if s.min() != s.max():
                    X.loc[:, '{}_minus_{}'.format(ts_col1, ts_col2)] = s
            
        X.drop(self.ts_cols, axis=1, inplace=True)        

        drop_cols = []
        for c in X.columns:
            if c.startswith('n_'):
                if X[c].min() == X[c].max():
                    drop_cols.append(c)

        X.drop(drop_cols, axis=1, inplace=True)

        #for c in self.cat_cols:
        #    X[c] = X[c].astype('category')

        # Label encode categorical features
        if train:
            self.enc = LabelEncoder(min_obs=X.shape[0] * .0001)
            X.loc[:, self.cat_cols] = self.enc.fit_transform(X[self.cat_cols])
        else:
            assert self.enc is not None
            X.loc[:, self.cat_cols] = self.enc.transform(X[self.cat_cols])


        # Generate count features fro categorical columns
        
        '''
        for c in self.cat_cols[:2]:
            if c.find('.') >= 0:
                continue
            if train:
                self.count_dict[c] = X[c].value_counts().to_frame(name='%s_count' % c.split('.')[-1]).reset_index().rename(columns={'index': c})
            if (c in self.count_dict):
                X = X.reset_index().merge(self.count_dict[c], how='left', on=c).set_index('index')
        '''

        # Generate count features fro multi-categorical columns
        '''
        for c in self.mcat_cols:
            if train:
                self.count_dict[c] = X[c].value_counts().to_frame(name='%s_count' % c.split('.')[-1]).reset_index().rename(columns={'index': c})

            if (c in self.count_dict):
                X = X.reset_index().merge(self.count_dict[c], how='left', on=c).set_index('index')
        '''

        # Use the count of categories for each observation
        for col in self.mcat_cols:
            X.loc[:, col] = X[col].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

        log('Features list: {}'.format(X.columns.tolist()))

        log('memory usage of X after featuring process: {:.2f}MB'.format(df_memory_usage(X) // 1e6))
        log('memory usage of process after featuring process: {:.2f}MB'.format(get_process_memory()))
        
        return X

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)

        self.y = y

        '''
        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)
        

        self.cat_cols = sorted([c for c in X.columns if c.startswith(CATEGORY_PREFIX)])
        self.mcat_cols = sorted([c for c in X.columns if c.startswith(MULTI_CAT_PREFIX)])
        self.num_cols = sorted([c for c in X.columns if c.startswith(NUMERICAL_PREFIX)])
        self.ts_cols = sorted([c for c in X.columns if c.startswith(TIME_PREFIX)])

        log('sorting the training data by the main timeseries column: {}'.format(self.ts_col))
        X['y_sorted'] = y
        X.sort_values(self.ts_col, inplace=True)

        y = X.y_sorted.copy()
        X.drop('y_sorted', axis=1, inplace=True)

        X = self.feature_engineer(X, train=True)
        train(X, y, self.config)
        '''
    
    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]

        main_table['y_sorted'] = self.y
        main_table.sort_values(self.ts_col, inplace=True)
        #y_trn = main_table.y_sorted.copy()
        #main_table.drop('y_sorted', axis=1, inplace=True)

        #main_table['data_type'] = 'train'
        #X_test['data_type'] = 'test'
        X_test['y_sorted'] = -1
        main_table = pd.concat([main_table, X_test], ignore_index = True).reset_index()

        del X_test
        gc.collect()

        # main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        # main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")        

        Xs[MAIN_TABLE_NAME] = main_table
        log('memory usage of main_table: {:.2f}MB'.format(df_memory_usage(main_table) // 1e6))
        log('memory usage of process: {:.2f}MB'.format(get_process_memory()))

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)

        del Xs, main_table
        gc.collect()

        log('memory usage of X: {:.2f}MB'.format(df_memory_usage(X) // 1e6))
        log('memory usage of process: {:.2f}MB'.format(get_process_memory()))
        
        self.cat_cols = sorted([c for c in X.columns if c.startswith(CATEGORY_PREFIX)])
        self.mcat_cols = sorted([c for c in X.columns if c.startswith(MULTI_CAT_PREFIX)])
        self.num_cols = sorted([c for c in X.columns if c.startswith(NUMERICAL_PREFIX)])
        self.ts_cols = sorted([c for c in X.columns if c.startswith(TIME_PREFIX)])

        X = self.feature_engineer(X, train=True)

        # X_trn = X[X.index.str.startswith("train")]
        # X_trn.index = X_trn.index.map(lambda x: int(x.split('_')[1]))
        X_trn = X[X['y_sorted'] != -1]
        y_trn = X_trn.y_sorted.copy()
        X_trn = X_trn.drop('y_sorted', axis=1)

        # X_tst = X[X.index.str.startswith("test")]
        # X_tst.index = X_tst.index.map(lambda x: int(x.split('_')[1]))
        X_tst = X[X['y_sorted'] == -1]
        X_tst = X_tst.drop('y_sorted', axis=1)

        X_tst.sort_index(inplace=True)

        del X
        gc.collect()

        log('memory usage of X_trn: {:.2f}MB'.format(df_memory_usage(X_trn) // 1e6))
        log('memory usage of process: {:.2f}MB'.format(get_process_memory()))

        train(X_trn, y_trn, self.config)
        del X_trn, y_trn
        gc.collect()
        
        log('memory usage of X_tst: {:.2f}MB'.format(df_memory_usage(X_tst) // 1e6))
        log('memory usage of process: {:.2f}MB'.format(get_process_memory()))
        result = predict(X_tst, self.config)
        del X_tst
        gc.collect()

        return pd.Series(result)
