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
from CONSTANT import KFOLD, IMBALANCE_RATE, N_RANDOM_COL, N_EST, N_STOP

from scipy.stats import rankdata


pd.set_option('mode.chained_assignment', None)


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    lgb_params = {
        "objective": "binary",
        "boosting": "gbdt",
        "metric": "auc",
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "num_threads": 4,
    }

    rf_params = {
        "objective": "binary",
        "boosting": "rf",
        "metric": "auc",
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "num_threads": 4,
    }

    X, y = downsample(X, y, ratio=IMBALANCE_RATE)

    X_sample, y_sample = ts_data_sample(X, y, SAMPLE_SIZE)
    top_features = feature_selection(X_sample, y_sample, lgb_params, config)

    X = X[top_features]
    X_sample = X_sample[top_features]
    gc.collect()

    config["models"] = []

    lgb_params, lgb_n_best = tune_hyperparam_lgb(X_sample, y_sample, lgb_params, config)
    train_lightgbm(X, y, lgb_params, lgb_n_best, config)

    rf_params, rf_n_best = tune_hyperparam_rf(X_sample, y_sample, rf_params, config)
    train_lightgbm(X, y, rf_params, rf_n_best, config)


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

    th = imp.loc[imp.feature_names.isin(random_cols), 'feature_importances'].mean()
    log('feature importance (th={:.2f}):\n{}'.format(th, imp))
    imp = imp[(imp.feature_importances > th) & ~(imp.feature_names.isin(random_cols))]
    return imp['feature_names'].tolist()

@timeit
def tune_hyperparam_lgb(X, y, params, config, n_eval=100):
    log('hyper-parameter tuning for LGB wiht 100 trials')
    hyperparams, trials = hyperopt_lightgbm(X, y, params, config, n_eval=n_eval)
    params.update(hyperparams)
    n_best = trials.best_trial['result']['model'].best_iteration
    log('best parameters: {}'.format(params))
    log('best iterations: %d' % n_best)
    return params, n_best

@timeit
def tune_hyperparam_rf(X, y, params, config, n_eval=100):
    log('hyper-parameter tuning for RF wiht 100 trials')
    hyperparams, trials = hyperopt_lightgbm(X, y, params, config, n_eval=n_eval)
    params.update(hyperparams)
    n_best = trials.best_trial['result']['model'].best_iteration
    log('best parameters: {}'.format(params))
    log('best iterations: %d' % n_best)
    return params, n_best

@timeit
def feature_selection(X, y, params, config, n_eval=10):
    np.random.seed(RANDOM_SEED)
    random_cols = []
    for i in range(1, N_RANDOM_COL + 1):
        random_col = '__random_{}__'.format(i)
        X[random_col] = np.random.rand(X.shape[0])
        random_cols.append(random_col)

    log('feature selection with {} trials'.format(n_eval))
    hyperparams, trials = hyperopt_lightgbm(X, y, params, config, n_eval=n_eval)
    feature_names = X.columns.values.tolist()
    top_features = get_top_features_lightgbm(trials.best_trial['result']['model'],
                                             feature_names, random_cols)

    log('selecting top %d out of %d features' % (len(top_features), len(feature_names)))
    config['feature_names'] = top_features
    return top_features

@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, params, n_best, config: Config):

    if KFOLD == 1:
        # training using full data
        log('training with 100% training data')
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data, n_best + 10, verbose_eval=100)
        config["models"].append(model)
    else:
        log(f"Training models in limit time {config['time_budget'] // 10}s...")
        timer = Timer()
        timer.set(config['time_budget'] // 5)
        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
        folds = skf.split(X, y)
        for i_fold, (i_trn, i_val) in enumerate(folds, 1):
            try:
                with timer.time_limit(f'Training Model k={i_fold}'):
                    log(f"Training fold {i_fold}...")
                    train_data = lgb.Dataset(X.iloc[i_trn], label=y.iloc[i_trn])
                    valid_data = lgb.Dataset(X.iloc[i_val], label=y.iloc[i_val])
                    model = lgb.train(params, train_data, int(n_best * 1.5), valid_data,
                                      early_stopping_rounds=N_STOP,
                                      verbose_eval=100)

                    config["models"].append(model)
            except Exception as e:
                log(f'Error: training error at k={i_fold}. Error message: {str(e)}')
                break
        log(f'Done, exec_time={timer.exec}')

        del skf
        gc.collect()

    feature_importances = config["models"][0].feature_importance(importance_type='gain')
    imp = pd.DataFrame({'feature_importances': feature_importances, 'feature_names': config['feature_names']})
    imp = imp.sort_values('feature_importances', ascending=False).drop_duplicates()

    log("[+] All feature importances:\n {}".format(imp))

@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config, bagging: bool=False) -> List:
    X = X[config['feature_names']]
    best_pred = 1.0
    n_model = len(config["models"])
    for model in config["models"]:
        best_pred *= model.predict(X, num_iteration=model.best_iteration)

    return best_pred ** (1 / n_model)

@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config,
                      n_eval=100):
    log(f'Search best params with train data {X.shape}')
    X_train, X_val, y_train, y_val = ts_data_split(X, y, test_size=HYPEROPT_TEST_SIZE)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
        "num_leaves": hp.choice("num_leaves", [15, 31, 63, 127]),
        "max_depth": hp.choice("max_depth", [-1, 4, 6, 8]),
        "feature_fraction": hp.quniform("feature_fraction", .5, .8, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", .5, .8, 0.1),
        "min_child_samples": hp.choice('min_child_samples', [10, 25, 100]),
        "lambda_l1": hp.choice('lambda_l1', [.1, 1, 10]),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, N_EST,
                          valid_data, early_stopping_rounds=N_STOP, verbose_eval=0)

        score = model.best_score["valid_0"][params["metric"]]

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK, 'model': model}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=n_eval, verbose=1,
                         rstate=np.random.RandomState(RANDOM_SEED))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams, trials


def ts_data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    #  -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    X_trn, X_val, y_trn, y_val = train_test_split(X, y,
                                                  test_size=test_size,
                                                  shuffle=False,
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


def downsample(X, y, ratio=.01):
    minor_label = check_imbalance_data(y)
    if minor_label is None:
        return X, y

    n_minor = (y == minor_label).sum()
    org_ratio = n_minor / (y.shape[0] - n_minor)
    if org_ratio >= ratio:
        return X, y

    else:
        X['__y__'] = y
        X['__original_order__'] = np.arange(X.shape[0])
        X_minor = X.loc[y == minor_label]
        X_major = X.loc[y != minor_label]

        n_major = int(n_minor / ratio)

        log('downsampling imbalanced data: {} {}s + {} others'.format(
            n_minor, minor_label, n_major
        ))

        X_major_sample = X_major.sample(n=n_major, random_state=RANDOM_SEED)
        X_sample = pd.concat([X_minor, X_major_sample], axis=0)
        X_sample.sort_values('__original_order__', inplace=True)

        y_sample = X_sample['__y__'].copy()
        X_sample.drop(['__y__', '__original_order__'], axis=1, inplace=True)

        return X_sample, y_sample


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

def stratified_kfold_split(X: pd.DataFrame, y: pd.Series, nrows: int = 5000,
                           minor_label=None):
    # split data with imbalance check
    X['__y__'] = y
    df_minority = X[X['__y__'] == minor_label]
    df_majority = X[X['__y__'] != minor_label]

    n_split = len(df_majority) // nrows
    if n_split < 2:
        n_split = 2

    log(f'imbalance data {df_majority.shape} minority is {df_minority.shape} with n_split {n_split}')

    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=RANDOM_SEED) # Model.N_FOLDS
    folds = skf.split(df_majority, df_majority['__y__'])
    for i_fold, (i_trn, i_val) in enumerate(folds, 1):
        if i_fold > 2:
            break
        X_majority = df_majority.iloc[i_val]
        y_majority = df_majority['__y__'].iloc[i_val]

    df_concat = pd.concat([df_minority, X_majority])
    y_sample = df_concat['__y__'].copy()
    X_sample = df_concat.drop('__y__', axis=1)

    del skf
    gc.collect()
    X.drop('__y__', axis=1, inplace=True)

    return X_sample, y_sample