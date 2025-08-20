import numpy as np

from ..utils.base import BaseEstimator, ClassifierMixin
from ..utils.validation import validate_array, validate_X_y
from ..optim.gradient_descent import StochasticGradientDescent
from typing import Dict

class Perceptron(ClassifierMixin, BaseEstimator):
    # noinspection PyUnreachableCode
    def __init__(self, optimizer: StochasticGradientDescent):

        if not isinstance(optimizer, StochasticGradientDescent):
            raise TypeError("Optimizer must be an instance of StochasticGradientDescent")

        self.optimizer = optimizer
        self.weights_ = None
        self.bias_ = None

    @staticmethod
    def _calculate_gradient(X_i: np.ndarray, y_i: np.ndarray, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        y_i_scaler = y_i[0,0]
        prediction = X_i.dot(params['weights']) + params['bias']
        if y_i_scaler * prediction[0, 0] > 0:
            # No update needed
            return {
                'weights': np.zeros_like(params['weights']),
                'bias': np.zeros_like(params['bias'])
            }

        dw = -y_i_scaler * X_i.T
        db = np.array(-y_i_scaler, dtype=np.float64)

        return {
            'weights': dw,
            'bias': db
        }

    # noinspection DuplicatedCode
    def fit(self, X: np.ndarray, y: np.ndarray = None, **fit_params) -> 'Perceptron':

        X, y = validate_X_y(X, y, X_dtype = np.float64)

        m, n = X.shape # m = number of samples, n = number of features

        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Labels must be either -1 or 1 for the Perceptron algorithm.")

        self.weights_ = np.zeros((n, 1), dtype=np.float64)
        self.bias_ = 0.0

        initial_params = {
            'weights': self.weights_,
            'bias': np.array(self.bias_, dtype=np.float64)
        }

        final_params = self.optimizer.optimize(
            X=X,
            y=y,
            initial_params=initial_params,
            gradient_func=self._calculate_gradient
        )

        self.weights_ = final_params['weights']
        self.bias_ = float(final_params['bias'])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        X = validate_array(X, dtype=np.float64)

        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("The model has not been fitted yet. Call 'fit' before 'predict'.")

        prediction_2d = np.sign(X.dot(self.weights_) + self.bias_)

        prediction_2d[prediction_2d == 0] = -1

        return prediction_2d.ravel()

    @property
    def coef_(self) -> np.ndarray:
        return self.weights_.flatten()

    @property
    def intercept_(self) -> float:

        return self.bias_

    @classmethod
    def with_default_optimizer(cls, learning_rate: float = 0.01,n_epochs = 1000) -> 'Perceptron':

        optimizer = StochasticGradientDescent(learning_rate = learning_rate, n_epochs = n_epochs)
        return cls(optimizer=optimizer)
