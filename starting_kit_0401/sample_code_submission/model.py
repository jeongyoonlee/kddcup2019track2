import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")

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
    def feature_engineer(self, X, y=None):

        self.ts_cols = sorted([c for c in X.columns if c.startswith(TIME_PREFIX)])

        X, self.prev_cat_cnt = self.count_categorical(X, self.prev_cat_cnt, mcat = False)
        X, self.prev_multi_cat_cnt = self.count_categorical(X, self.prev_multi_cat_cnt, mcat = True)
        X, self.prev_uniq_cat_cnt = self.uniq_categorical(X, self.prev_uniq_cat_cnt, mcat = False)
        #X = self.aggregate_cat_cols_on_time_features(X, mcat = False)

        # Use the hour of the day as a feature
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

        s = (X.nunique() == 1)
        cols_to_drop = s[s].index.tolist()
        X.drop(cols_to_drop + self.ts_cols, axis=1, inplace=True)

        self.cat_cols = sorted([c for c in X.columns if c.startswith(CATEGORY_PREFIX)])
        self.mcat_cols = sorted([c for c in X.columns if c.startswith(MULTI_CAT_PREFIX)])
        self.num_cols = sorted([c for c in X.columns if c.startswith(NUMERICAL_PREFIX)])

        # Label encode categorical features
        self.enc = LabelEncoder(min_obs=X.shape[0] * .0001)
        X.loc[:, self.cat_cols] = self.enc.fit_transform(X[self.cat_cols])

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
        main_table = pd.concat([main_table, X_test], ignore_index = True).reset_index()

        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)

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
