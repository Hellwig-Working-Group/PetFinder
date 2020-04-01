"""This model contains utils for creating submissions"""
from datetime import datetime
from functools import partial
from os.path import isdir, join
from os import makedirs, getcwd

import numpy as np
import pandas as pd
import scipy as sp

from config.path_config import SUBMISSION_DIR
from evaluation_utils import sklearn_quadratic_kappa


class OptimizedRounder:
    """
    Taken from:
        - https://www.kaggle.com/wrosinski/baselinemodeling
        - https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/76107
    """
    def __init__(self):
        self._coefficients = None

    def fit(self, x, y):
        """Learns coefficients using nelder-mead method"""
        loss_partial = partial(self._kappa_loss, x=x, y=y)
        # inital coeffs taken from the competition description
        initial_coefficients = [0.5, 1.5, 2.5, 3.5]
        self._coefficients = sp.optimize.minimize(loss_partial, initial_coefficients, method='nelder-mead')

    def predict(self, x, coefficients):
        """Rounds the input using learned coefficients"""
        x_copy = np.copy(x)
        for i, prediction in enumerate(x_copy):
            if prediction < coefficients[0]:
                x_copy[i] = 0
            elif coefficients[0] <= prediction < coefficients[1]:
                x_copy[i] = 1
            elif coefficients[1] <= prediction < coefficients[2]:
                x_copy[i] = 2
            elif coefficients[2] <= prediction < coefficients[3]:
                x_copy[i] = 3
            else:
                x_copy[i] = 4
        return x_copy

    def _kappa_loss(self, coefficients, x, y):
        """Quadratic Kappa loss function."""
        x_copy = self.predict(x, coefficients)
        return -1 * sklearn_quadratic_kappa(y, x_copy)

    def coefficients(self):
        """Returns coefficients for optimal rounding for given input"""
        return self._coefficients['x']


def generate_submission(ids, labels, save_dir=SUBMISSION_DIR):
    log_date = datetime.now().strftime("%b_%d_%H_%M")
    """Saves Kaggle submission csv file from predictions in specified directory."""
    # assign current working directory (CWD) if save_dir is not specified
    if not save_dir:
        save_dir = getcwd()
    # recursively create directory specified in save_dir if it doesn't exist yet
    if not isdir(save_dir):
        makedirs(save_dir)

    submission = pd.DataFrame(
        {'PetID': ids,
         'AdoptionSpeed': labels})
    submission.to_csv(join(save_dir, log_date + '_submission.csv'), index=False)
    submission.to_csv(join(save_dir, 'submission.csv'), index=False)
