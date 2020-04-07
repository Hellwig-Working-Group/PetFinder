"""
This module contains implementation of evaluation/loss function
Based on: https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
"""

import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def sklearn_quadratic_kappa(actuals, predictions, num_buckets=5):
    """Sklearn implementation of quadratic kappa"""
    return cohen_kappa_score(actuals, predictions, labels=[x for x in range(num_buckets)], weights='quadratic')


def quadratic_kappa(actuals, predictions, num_buckets=5):
    """Computes quadratic kappa for 2 vectors."""
    # check if vectors have same shape
    assert actuals.shape == predictions.shape, "Shape mismatch between ground truth and prediction"
    # compute confusion matrix, explicitly set labels range
    conf_mx = confusion_matrix(actuals, predictions, labels=[x for x in range(num_buckets)])
    normalized_conf_mx = conf_mx/conf_mx.sum()

    # compute weights matrix
    weights = np.zeros(conf_mx.shape)
    # todo: try to get by without nested loops?
    for i in range(len(weights)):
        for j in range(len(weights)):
            weights[i][j] = float(((i - j) ** 2) / 16)

    # compute histograms for actual and predicted labels
    actuals_histogram = np.zeros([num_buckets])
    for element in actuals:
        actuals_histogram[element] += 1
    predictions_histogram = np.zeros([num_buckets])
    for element in predictions:
        predictions_histogram[element] += 1
    print(f'Actuals value counts:{actuals_histogram}\nPredictions value counts:{predictions_histogram}')

    # compute expected matrix
    expected_mx = np.outer(actuals_histogram, predictions_histogram)
    normalized_expected_mx = expected_mx / expected_mx.sum()

    # compute quadratic kappa
    nominator, denominator = 0, 0
    # todo: try to get by without nested loops?
    for i in range(len(weights)):
        for j in range(len(weights)):
            nominator += weights[i][j] * normalized_conf_mx[i][j]
            denominator += weights[i][j] * normalized_expected_mx[i][j]

    return 1 - (nominator / denominator)

def compare_models_chisq(x, y):
    cont_table = pd.crosstab(x, y, margins=False)
    chi2, p, dof, ex = chi2_contingency(cont_table, correction=False)
    return(p)


if __name__ == '__main__':
    # simple test
    acts = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 1])
    preds = np.array([0, 2, 1, 0, 0, 0, 1, 1, 2, 1])
    assert round(quadratic_kappa(acts, preds), 15) == -0.139240506329114, 'Test failed, quadratic kappa is broken!'
    assert round(sklearn_quadratic_kappa(acts, preds), 15) == -0.139240506329114, 'Test failed, quadratic kappa from' \
                                                                                  ' sklearn is broken!'
