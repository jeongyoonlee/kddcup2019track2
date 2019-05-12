from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from util import Config, log, timeit


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    preds = predict_lightgbm(X, config)
    return preds


@timeit
def validate(preds, y_path) -> np.float64:
    score = roc_auc_score(pd.read_csv(y_path)['label'].values, preds)
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "binary",
        "metric": "auc",
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4
    }

    X_train, X_val, y_train, y_val = ts_data_split(X, y, 0.15)

    X_sample, y_sample = ts_data_sample(X_train, y_train, 30000)
    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    config["model"] = lgb.train({**params, **hyperparams},
                                train_data,
                                1000,
                                valid_data,
                                early_stopping_rounds=10,
                                verbose_eval=100)

    n_best = config["model"].best_iteration
    log("best iteration: {}".format(n_best))

    log('training with 100% training data')
    train_data = lgb.Dataset(X, label=y)
    config["model"] = lgb.train({**params, **hyperparams},
                                train_data,
                                n_best,
                                verbose_eval=100)


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    return config["model"].predict(X)


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = ts_data_split(X, y, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.02), np.log(0.1)),
        "num_leaves": hp.choice("num_leaves", [31, 63, 127]),
        "feature_fraction": hp.quniform("feature_fraction", .5, .8, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", .5, .8, 0.1),
        "min_child_samples": hp.choice('min_child_samples', [10, 25, 100]),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 300,
                          valid_data, early_stopping_rounds=30, verbose_eval=0)

        score = model.best_score["valid_0"][params["metric"]]

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=10, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams


def ts_data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, shuffle=False)


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=1)


def ts_data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.iloc[-nrows:]
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample
