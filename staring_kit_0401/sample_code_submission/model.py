import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")

import copy
import numpy as np
import pandas as pd
from data import LabelEncoder

from automl import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME, CATEGORY_PREFIX, MULTI_CAT_PREFIX, NUMERICAL_PREFIX, TIME_PREFIX
from merge import merge_table
from preprocess import clean_df, clean_tables, feature_engineer
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

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        clean_df(X)

        self.cat_cols = [c for c in X.columns if c.startswith(CATEGORY_PREFIX) or X[c].dtype == np.object]
        self.mcat_cols = [c for c in X.columns if c.startswith(MULTI_CAT_PREFIX)]
        self.num_cols = [c for c in X.columns if c.startswith(NUMERICAL_PREFIX)]
        self.ts_cols = [c for c in X.columns if c.startswith(TIME_PREFIX)]

        self.enc = LabelEncoder(min_obs=X.shape[0] * .0001)
        X.loc[:, self.cat_cols] = self.enc.fit_transform(X[self.cat_cols])
        X.drop(self.ts_cols, axis=1, inplace=True)

        #feature_engineer(X, self.config)
        train(X, y, self.config)

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

        X.loc[:, self.cat_cols] = self.enc.transform(X[self.cat_cols])
        X.drop(self.ts_cols, axis=1, inplace=True)

        #feature_engineer(X, self.config)
        X = X[X.index.str.startswith("test")]
        X.index = X.index.map(lambda x: int(x.split('_')[1]))
        X.sort_index(inplace=True)
        result = predict(X, self.config)

        return pd.Series(result)
