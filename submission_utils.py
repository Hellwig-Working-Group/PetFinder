"""This model contains utils for creating submissions"""

from functools import partial

import numpy as np
import scipy as sp

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
