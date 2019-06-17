from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from util import Config, log, timeit
from CONSTANT import RANDOM_SEED

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

def get_top_features_lightgbm(model, feature_names, random_cols):
    feature_importances = model.feature_importance(importance_type='split')
    imp = pd.DataFrame({'feature_importances': feature_importances, 'feature_names': feature_names})
    imp = imp.sort_values('feature_importances', ascending=False).drop_duplicates()

    th = max(imp.loc[imp.feature_names.isin(random_cols)].feature_importances.max(), 0)
    imp = imp[imp.feature_importances > th]

    return imp.feature_names.tolist()

@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "binary",
        "metric": "auc",
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "num_threads": 4,
    }

    np.random.seed(RANDOM_SEED)
    n_random_cols = 3
    random_cols = []
    for i in range(1, n_random_cols + 1):
        random_col = '__random_{}__'.format(i)
        X[random_col] = np.random.rand(X.shape[0])
        random_cols.append(random_col)

    X_sample, y_sample = ts_data_sample(X, y, 100000)
    hyperparams, trials = hyperopt_lightgbm(X_sample, y_sample, params, config)
    n_best = trials.best_trial['result']['model'].best_iteration
    log('best iterations: %d' % n_best)

    feature_names = X.columns.values.tolist()
    top_features = get_top_features_lightgbm(trials.best_trial['result']['model'],
                                             feature_names,
                                             random_cols)

    log('selecting top %d out of %d features' % (len(top_features), len(feature_names)))
    X = X[top_features]
    feature_names = top_features
    config['feature_names'] = feature_names

    log('training with 100% training data')
    train_data = lgb.Dataset(X, label=y)
    config["model"] = lgb.train({**params, **hyperparams},
                                train_data,
                                n_best,
                                verbose_eval=100)

    feature_importances = config["model"].feature_importance(importance_type='split')
    imp = pd.DataFrame({'feature_importances': feature_importances, 'feature_names': feature_names})
    imp = imp.sort_values('feature_importances', ascending=False).drop_duplicates()

    print("[+] All feature importances", list(imp.values))

    config['trials'] = trials


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config, bagging: bool=False) -> List:
    X = X[config['feature_names']]
    best_pred = config["model"].predict(X)
    if bagging:
        trial_pred = 1
        sum_score = 0
        for trial, score in zip(config['trials'].results, config['trials'].losses()):
            trial_pred *= (trial['model'].predict(X)**(-score))
            sum_score += (-score)
        trial_pred = trial_pred**(1.0/sum_score)
        best_pred = (best_pred**0.7)*(trial_pred**0.3)
    return best_pred


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
        "min_child_samples": hp.choice('min_child_samples', [10, 25, 50, 100]),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 500,
                          valid_data, early_stopping_rounds=10, verbose_eval=0)

        score = model.best_score["valid_0"][params["metric"]]

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK, 'model': model}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=50, verbose=1,
                         rstate=np.random.RandomState(RANDOM_SEED))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams, trials


def ts_data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=RANDOM_SEED)

    np.random.seed(RANDOM_SEED)

    i_trn = np.arange(X_train.shape[0])
    np.random.shuffle(i_trn)

    i_val = np.arange(X_val.shape[0])
    np.random.shuffle(i_val)
    return X_train.iloc[i_trn], X_val.iloc[i_val], y_train.iloc[i_trn], y_val.iloc[i_val]
    #test_size = int(test_size*X.shape[0])
    #return X.iloc[:-test_size,:], X.iloc[test_size:,:], y[:-test_size], y[test_size:]


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)


def ts_data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):

    assert X.shape[0] == y.shape[0]

    if len(X) > nrows:
        X_sample = X.iloc[-nrows:].copy()
        y_sample = y.iloc[-nrows:].copy()
    else:
        X_sample = X.copy()
        y_sample = y.copy()

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
