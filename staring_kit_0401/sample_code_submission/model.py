import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")

import copy
import logging
import numpy as np
import pandas as pd
from data import LabelEncoder

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

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)

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

    @timeit
    def feature_engineer(self, X, y=None, train=True):
        # Use the hour of the day as a feature
        for col in self.ts_cols:
            X.loc[:, col] = pd.to_datetime(X[col]).dt.hour

        # Label encode categorical features
        if train:
            self.enc = LabelEncoder(min_obs=X.shape[0] * .0001)
            X.loc[:, self.cat_cols] = self.enc.fit_transform(X[self.cat_cols])
        else:
            assert self.enc is not None
            X.loc[:, self.cat_cols] = self.enc.transform(X[self.cat_cols])

        # Use the count of categories for each observation
        for col in self.mcat_cols:
            X.loc[:, col] = X[col].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)

        return X

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)

        X = self.feature_engineer(X, train=False)
        X = X[X.index.str.startswith("test")]
        X.index = X.index.map(lambda x: int(x.split('_')[1]))
        X.sort_index(inplace=True)
        result = predict(X, self.config)

        return pd.Series(result)
