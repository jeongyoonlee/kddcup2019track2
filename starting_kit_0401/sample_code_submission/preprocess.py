import datetime
import numpy as np
from pandas.api.types import is_categorical_dtype

import CONSTANT
from util import log, timeit


@timeit
def clean_tables(tables):
    for tname in tables:
        log(f"cleaning table {tname}")
        clean_df(tables[tname])


@timeit
def clean_df(df):
    log('memory usage: {:.2f}MB'.format(df.memory_usage().sum() // 1e6))
    fillna(df)

    cols_to_drop = []
    for col in df.columns:
        s = df[col]
        if s.dtype == np.object:
            s = s.astype('category')
            if len(s.cat.categories) == 1:
                cols_to_drop.append(col)

        elif is_categorical_dtype(s):
            if len(s.cat.categories) == 1:
                cols_to_drop.append(col)

        elif s.min() == s.max():
            cols_to_drop.append(col)

    log('dropping constant features')
    log('{}'.format(cols_to_drop))
    df.drop(cols_to_drop, axis=1, inplace=True)
    log('memory usage: {:.2f}MB'.format(df.memory_usage().sum() // 1e6))


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)


@timeit
def feature_engineer(df, config):
    transform_categorical_hash(df)
    transform_datetime(df, config)


@timeit
def transform_datetime(df, config):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x.split(',')[0]))
