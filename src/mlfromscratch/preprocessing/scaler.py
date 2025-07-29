# This line is used to mark the creation of this file (the body is accidentally committed with other contents).
import numpy as np
from typing import Optional
from ..utils.base import BaseEstimator, TransformerMixin

class StandardScaler(BaseEstimator, TransformerMixin):
    """
    StandardScaler standardizes features by removing the mean and scaling to unit variance.
    It is useful for algorithms that assume data is centered around zero and has unit variance.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    @property
    def n_features_(self) -> int:
        """
        Number of features in the input data after fitting.
        This property is set after calling fit() method.
        """
        if self.mean_ is None:
            raise ValueError("You must fit before calling scaler.")

        return self.mean_.shape[0]

    def fit(self, X: np.ndarray, _y: np.ndarray = None, **_fit_params) -> 'StandardScaler':
        """
        Compute the mean and standard deviation for each feature in X.

        :param X: Input data of shape (n_samples, n_features)
        :param _y: Ignored, exists for compatibility with scikit-learn API
        :param _fit_params: Additional parameters (not used)
        :return: self
        """

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize the input data X using the mean and scale computed during fit.

        :param X: Input data of shape (n_samples, n_features)
        :return: Standardized data of shape (n_samples, n_features)
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("You must fit the scaler before transforming data.")

        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the standardized data back to original scale.

        :param X: Standardized data of shape (n_samples, n_features)
        :return: Original data of shape (n_samples, n_features)
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("You must fit the scaler before inverse transforming data.")

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("Input data shape does not match the fitted scaler's feature count.")

        return X * self.scale_ + self.mean_