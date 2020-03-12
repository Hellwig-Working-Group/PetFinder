"""
This module contains main flow.
To generate submission run: `python main.py`
"""

from collections import Counter

import pandas as pd
import numpy as np

from data_utils import get_dataset, remove_object_cols
from models import kfold_lgb
from submission_utils import OptimizedRounder
from evaluation_utils import sklearn_quadratic_kappa


SEED = 997


# todo: major cleanup and comments

if __name__ == '__main__':
    target_col = 'AdoptionSpeed'
    datasets = {dataset_type: get_dataset(dataset_type) for dataset_type in ('train', 'test')}
    cleaned_datasets = {dataset_type: remove_object_cols(dataset) for dataset_type, dataset in datasets.items()}
    outputs = kfold_lgb(cleaned_datasets)

    y_train = cleaned_datasets['train'][target_col]

    optR = OptimizedRounder()
    optR.fit(outputs['train'], y_train.values)
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(outputs['train'], coefficients)
    print("\nValid Counts = ", Counter(y_train.values))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = sklearn_quadratic_kappa(y_train.values, pred_test_y_k)
    print("QWK = ", qwk)

    train_predictions = optR.predict(outputs['train'], coefficients).astype(int)
    print('train pred distribution: {}'.format(Counter(train_predictions)))

    test_predictions = optR.predict(outputs['test'].mean(axis=1), coefficients)
    print('test pred distribution: {}'.format(Counter(test_predictions)))

    print("True Distribution:")
    print(pd.value_counts(y_train, normalize=True).sort_index())
    print("\nTrain Predicted Distribution:")
    print(pd.value_counts(train_predictions, normalize=True).sort_index())
    print("\nTest Predicted Distribution:")
    print(pd.value_counts(test_predictions, normalize=True).sort_index())

    submission = pd.DataFrame({'PetID': datasets['test']['PetID'].values, 'AdoptionSpeed': test_predictions.astype(np.int32)})
    submission.head()
    submission.to_csv('submission.csv', index=False)
