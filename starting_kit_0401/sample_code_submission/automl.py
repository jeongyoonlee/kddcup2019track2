from typing import Dict, List

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from util import Config, Timer, log, timeit, check_imbalance_data
from CONSTANT import RANDOM_SEED, SAMPLE_SIZE, HYPEROPT_TEST_SIZE, BEST_ITER_THRESHOLD
from CONSTANT import KFOLD, IMBALANCE_RATE, N_RANDOM_COL

from scipy.stats import rankdata

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

def get_top_features_lightgbm(model, feature_names, random_cols=[]):
    feature_importances = model.feature_importance(importance_type='gain')
    imp = pd.DataFrame({'feature_importances': feature_importances, 'feature_names': feature_names})
    imp = imp.sort_values('feature_importances', ascending=False).drop_duplicates()

    th = imp.loc[imp.feature_names.isin(random_cols)].feature_importances.mean()
    log('feature importance:\n{}'.format(imp))
    imp = imp[(imp.feature_importances > th) &
              ~(imp.feature_importances.isin(random_cols))]
    return imp['feature_names'].tolist()

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
    random_cols = []
    for i in range(1, N_RANDOM_COL + 1):
        random_col = '__random_{}__'.format(i)
        X[random_col] = np.random.rand(X.shape[0])
        random_cols.append(random_col)

    imbalance = check_imbalance_data(y)

    X_sample, y_sample = ts_data_sample(X, y, SAMPLE_SIZE)
    #X_sample, y_sample = stratified_kfold_split(X, y, SAMPLE_SIZE)

    shuffle = imbalance is not None
    hyperparams, trials = hyperopt_lightgbm(X_sample, y_sample, params, config, shuffle)
    n_best = trials.best_trial['result']['model'].best_iteration
    log('best iterations: %d' % n_best)

    feature_names = X.columns.values.tolist()
    top_features = get_top_features_lightgbm(trials.best_trial['result']['model'], feature_names)

    log('selecting top %d out of %d features' % (len(top_features), len(feature_names)))
    X = X[top_features]
    feature_names = top_features
    config['feature_names'] = feature_names

    n_best += 10

    config["models"] = []
    if KFOLD == 1:
        # training using full data
        log('training with 100% training data')
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train({**params, **hyperparams},
                                        train_data,
                                        n_best,
                                        verbose_eval=100)
        config["models"].append(model)
    else:
        log(f"Training models in limit time {config['time_budget'] // 10}s...")
        timer = Timer()
        timer.set(config['time_budget'] // 10)
        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
        if imbalance is None:
            log(f"Not imbalance data, training using StratifiedKFold {y.value_counts()}")
            folds = skf.split(X, y)
            for n_fold, (train_idx, valid_idx) in enumerate(folds):
                try:
                    with timer.time_limit(f'Training Model k={n_fold}'):
                        log(f"Training fold {n_fold}...")
                        X_n = X.iloc[valid_idx]
                        y_n = y.iloc[valid_idx]
                        train_data = lgb.Dataset(X_n, label=y_n)
                        model = lgb.train({**params, **hyperparams},
                                                        train_data,
                                                        n_best,
                                                        verbose_eval=100)

                        config["models"].append(model)
                except Exception as e:
                    log(f'Error: training error at k={n_fold}. Error message: {str(e)}')
                    break
            log(f'Done, exec_time={timer.exec}')
        else:
            log(f"Imbalance data, training using StratifiedKFold {y.value_counts()}")
            X['y_sorted'] = y
            df_minority = X[X['y_sorted']==imbalance]
            df_majority = X[X['y_sorted']!=imbalance]
            folds = skf.split(df_majority, df_majority['y_sorted'])
            for n_fold, (train_idx, valid_idx) in enumerate(folds):
                try:
                    with timer.time_limit(f'Training Model k={n_fold}'):
                        log(f"Training fold {n_fold}...")
                        X_majority = df_majority.iloc[valid_idx]
                        y_majority = df_majority['y_sorted'].iloc[valid_idx]
                        df_concat = pd.concat([df_minority, X_majority])

                        y_n = df_concat['y_sorted'].copy()
                        X_n = df_concat.drop('y_sorted', axis=1)
                        train_data = lgb.Dataset(X_n, label=y_n)
                        model = lgb.train({**params, **hyperparams},
                                                        train_data,
                                                        n_best,
                                                        verbose_eval=100)

                        config["models"].append(model)
                except Exception as e:
                    log(f'Error: training error at k={n_fold}. Error message: {str(e)}')
                    break
            log(f'Done, exec_time={timer.exec}')

        del skf
        gc.collect()

    feature_importances = config["models"][0].feature_importance(importance_type='gain')
    imp = pd.DataFrame({'feature_importances': feature_importances, 'feature_names': feature_names})
    imp = imp.sort_values('feature_importances', ascending=False).drop_duplicates()

    log(f"[+] All feature importances {list(imp.values)}")


    config['trials'] = trials


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config, bagging: bool=False) -> List:
    X = X[config['feature_names']]
    # best_pred = config["model"].predict(X)
    with_best_iteration = True
    total_rows = X.shape[0]
    best_pred = 0.0
    for model in config["models"]:
        if with_best_iteration == True:
          best_pred += rankdata(model.predict(X, num_iteration=model.best_iteration)) / total_rows
        else:
          best_pred += rankdata(model.predict(X)) / total_rows

    best_pred /= len(config["models"])

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
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config, shuffle = True):
    log(f'Search best params with train data {X.shape}')
    X_train, X_val, y_train, y_val = ts_data_split(X, y, test_size=HYPEROPT_TEST_SIZE)
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=HYPEROPT_TEST_SIZE,
    #                                                  shuffle=shuffle, random_state=RANDOM_SEED)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
        "num_leaves": hp.choice("num_leaves", [15, 31, 63, 127]),
        "max_depth": hp.choice("max_depth", [-1, 4, 6, 8]),
        "feature_fraction": hp.quniform("feature_fraction", .5, .8, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", .5, .8, 0.1),
        "min_child_samples": hp.choice('min_child_samples', [10, 25, 100]),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 300,
                          valid_data, early_stopping_rounds=20, verbose_eval=0)

        score = model.best_score["valid_0"][params["metric"]]

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK, 'model': model}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=100, verbose=1,
                         rstate=np.random.RandomState(RANDOM_SEED))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams, trials


def ts_data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=test_size, shuffle=False,
                                                  random_state=RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    i_trn = np.arange(X_trn.shape[0])
    np.random.shuffle(i_trn)
    i_val = np.arange(X_val.shape[0])
    np.random.shuffle(i_val)
    return X_trn.iloc[i_trn], X_val.iloc[i_val], y_trn.iloc[i_trn], y_val.iloc[i_val]


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)


def ts_data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.iloc[-nrows:].copy()
        y_sample = y[X_sample.index].copy()
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

def stratified_kfold_split(X: pd.DataFrame, y: pd.Series, nrows: int=5000):
    # split data with imbalance check
    imbalance = check_imbalance_data(y)

    if imbalance is not None:
        log(f'imbalance data less than {IMBALANCE_RATE} minority is {imbalance}')
        X['y_sorted'] = y
        df_minority = X[X['y_sorted']==imbalance]
        df_majority = X[X['y_sorted']!=imbalance]

        n_split = len(df_majority) // nrows
        if n_split < 2:
            n_split = 2

        log(f'imbalance data {df_majority.shape} minority is {df_minority.shape} with n_split {n_split}')

        skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=RANDOM_SEED) # Model.N_FOLDS
        folds = skf.split(df_majority, df_majority['y_sorted'])
        for n_fold, (train_idx, valid_idx) in enumerate(folds):
            if n_fold > 1:
                break
            X_majority = df_majority.iloc[valid_idx]
            y_majority = df_majority['y_sorted'].iloc[valid_idx]

        df_concat = pd.concat([df_minority, X_majority])
        y_sample = df_concat['y_sorted'].copy()
        X_sample = df_concat.drop('y_sorted', axis=1)

        del skf
        gc.collect()
        X.drop('y_sorted', axis=1, inplace=True)
    else:
        X_sample, y_sample = ts_data_sample(X, y, nrows)

    return X_sample, y_sample