"""This module contains model related code"""

from collections import Counter

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np

SEED = 997

PARAMS = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 70,
          'max_depth': 9,
          'learning_rate': 0.01,
          'bagging_fraction': 0.85,
          'feature_fraction': 0.8,
          'min_split_gain': 0.02,
          'min_child_samples': 150,
          'min_child_weight': 0.02,
          'lambda_l2': 0.0475,
          'verbosity': -1,
          'data_random_seed': SEED}


def kfold_lgb(datasets, target_col='AdoptionSpeed', params=PARAMS, seed=SEED):
    """Runs K-Fold CV of LGBM models"""

    assert set(datasets.keys()) == {'train', 'test'}, "Wrong data dict structure"
    train_set = datasets['train']
    x_test = datasets['test']

    n_splits = 5
    n, _ = datasets['train'].shape
    m, _ = datasets['test'].shape
    outputs_train = np.zeros((n))
    outputs_test = np.zeros((m, n_splits))

    i = 0
    kfold = StratifiedKFold(n_splits=n_splits, random_state=seed)
    for train_idx, val_idx in kfold.split(train_set, train_set[target_col].values):

        train_set_fold = train_set.iloc[train_idx, :]
        val_set_fold = train_set.iloc[val_idx, :]

        y_train = train_set_fold[target_col].values
        x_train = train_set_fold.drop([target_col], axis=1)

        y_val = val_set_fold[target_col].values
        x_val = val_set_fold.drop([target_col], axis=1)

        print('\ny_train distribution: {}'.format(Counter(y_train)))

        model = train_lgb(x_train, x_val, y_train, y_val, params=params)

        val_pred = model.predict(x_val, num_iteration=model.best_iteration)
        test_pred = model.predict(x_test, num_iteration=model.best_iteration)

        outputs_train[val_idx] = val_pred
        outputs_test[:, i] = test_pred
        i += 1

    return {'train': outputs_train, 'test': outputs_test}


def train_lgb(x_train, x_val, y_train, y_val, params=PARAMS):
    """Train a baseline, LightGMB model"""
    early_stop = 500
    verbose_eval = 100
    num_rounds = 10000

    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_val, label=y_val)
    watchlist = [d_train, d_valid]

    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
    return model
