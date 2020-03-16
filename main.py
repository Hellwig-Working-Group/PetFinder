"""
This module contains main flow.
To generate submission run: `python main.py`

This is a regression approach explained here: https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/76107
"""

import pandas as pd
import numpy as np

from data_utils import get_dataset
from preprocessing import remove_object_cols
from models import kfold_lgb
from submission_utils import OptimizedRounder, generate_submission
from evaluation_utils import sklearn_quadratic_kappa

SEED = 997
TARGET_COL = 'AdoptionSpeed'

if __name__ == '__main__':
    # step 1 - load and transform data
    # load train and test tabular datasets
    datasets = {dataset_type: get_dataset(dataset_type) for dataset_type in ('train', 'test')}
    # remove all string columns from dataset
    # todo: investigate if there are no int/float categorical cols left that hasn't been one-hot encoded
    cleaned_datasets = {dataset_type: remove_object_cols(dataset) for dataset_type, dataset in datasets.items()}
    # extract training labels
    y_train = cleaned_datasets['train'][TARGET_COL]

    # step 2 - train a model and get it's outputs
    # get outputs from k-fold CV LGBM training
    outputs = kfold_lgb(cleaned_datasets)

    # step 3 - round the outputs, compute quadratic kappa and generate submission
    # initialize and train OptimizedRounder
    optR = OptimizedRounder()
    optR.fit(outputs['train'], y_train.values)
    # get rounding coefficients
    coefficients = optR.coefficients()
    # round outputs for training/test set
    rounded_train_outputs = optR.predict(outputs['train'], coefficients).astype(int)
    rounded_test_outputs = optR.predict(outputs['test'].mean(axis=1), coefficients).astype(int)
    # compute quadratic kappa for train set and print it
    qwk_train = sklearn_quadratic_kappa(y_train.values, rounded_train_outputs)
    print(f"\nTrain QWK: {qwk_train}")

    # print distributions of predictions vs. true distributions
    print("\nTrue Distribution:")
    print(pd.value_counts(y_train, normalize=True).sort_index())
    print("\nTrain Predicted Distribution:")
    print(pd.value_counts(rounded_train_outputs, normalize=True).sort_index())
    print("\nTest Predicted Distribution:")
    print(pd.value_counts(rounded_test_outputs, normalize=True).sort_index())

    # generate submission
    generate_submission(datasets['test']['PetID'].values, rounded_test_outputs.astype(np.int32))
