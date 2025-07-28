from abc import ABC, abstractmethod
import numpy as np
from ..metrics.classification import accuracy_score
from ..metrics.regression import r2_score


class BaseEstimator(ABC):
    """
    Base class for all estimators in the machine learning library.
    """

    def get_params(self, deep=True) -> dict:
        """
        Get parameters for this estimator.
        :param deep: If True, will return the parameters for this estimator and
                     contained subobjects that are estimators.
        :return: Parameters of the estimator.
        """

        params = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                params[key] = getattr(self, key)
        if deep:
            for key, value in params.items():
                if isinstance(value, BaseEstimator):
                    nest_params = value.get_params()
                    for nest_key, nest_value in nest_params.items():
                        params[f"{key}__{nest_key}"] = nest_value
        return params

    def set_params(self, **params) -> 'BaseEstimator':
        """
        Set the parameters of this estimator.
        :param params: Parameters to set.
        :return: Self with updated parameters.
        """
        if not params:
            return self

        self._validate_params(**params)
        self._internal_set_params(**params)
        return self

    def _validate_params(self, **params) -> None:
        """
        Validate the parameters before setting them.

        This method checks if the provided parameters are valid for this estimator.
        For nested parameters (containing '__'), it validates that the main parameter
        exists and is an estimator, then recursively validates the nested parameters.

        :param params: Dictionary of parameters to validate
        :raises ValueError: If any parameter is invalid or if a nested parameter
                           refers to an object that is not an estimator
        """
        if not params:
            return

        for key, value in params.items():
            if '__' in key:
                main_key, sub_key = key.split('__', 1)
                if hasattr(self, main_key):
                    nest_object = getattr(self, main_key)
                    if isinstance(nest_object, BaseEstimator):
                        nest_object._validate_params(**{sub_key: value})
                    else:
                        raise ValueError(f"Parameter {main_key} is not an estimator in {self.__class__.__name__}")
                else:
                    raise ValueError(f"Invalid parameter {main_key} for estimator {self.__class__.__name__}")
            else:
                if not hasattr(self, key):
                    raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")

    def _internal_set_params(self, **params) -> 'BaseEstimator':
        """
        Internal method to set parameters without validation.
        This is used for setting parameters in nested estimators.
        :param params: Parameters to set.
        :return: Self with updated parameters.
        """
        for key, value in params.items():
            if '__' in key:
                main_key, sub_key = key.split('__', 1)
                assert hasattr(self, main_key)
                nest_object = getattr(self, main_key)
                assert isinstance(nest_object, BaseEstimator)
                nest_object._internal_set_params(**{sub_key: value})
            else:
                setattr(self, key, value)
        return self

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'BaseEstimator':
        """
        Fit the model to the training data.
        :param X: Training data of shape (n_samples, n_features)
        :param y: Target values of shape (n_samples,), optional
        :param fit_params: Additional parameters to pass to the fitting process
        :return: Self (fitted estimator)
        """

        raise NotImplementedError("BaseEstimator.fit() is not implemented. ")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.
        :param X: Input data of shape (n_samples, n_features)
        :return: Predictions of shape (n_samples,)
        """

        raise NotImplementedError("BaseEstimator.predict() is not implemented. ")

    def fit_predict(self, X: np.ndarray, y: np.ndarray, **fit_params) -> np.ndarray:
        """
        Fit the model and then predict using the fitted model.
        :param X: Training data of shape (n_samples, n_features)
        :param y: Target values of shape (n_samples,), optional
        :param fit_params: Additional parameters to pass to the fitting process
        :return: Predictions of shape (n_samples,)
        """
        return self.fit(X, y, **fit_params).predict(X)

    def __repr__(self):
        """
        String representation of the estimator.
        :return: String representation of the estimator with its parameters
        """

        class_name = self.__class__.__name__
        params_str = ', '.join(f"{key}={value!r}" for key, value in self.get_params().items())
        return f"{class_name}({params_str})"


def clone(estimator: BaseEstimator) -> BaseEstimator:
    """
    Clone an estimator.
    :param estimator: Estimator to clone
    :return: A new instance of the estimator with the same parameters
    """
    if not isinstance(estimator, BaseEstimator):
        raise TypeError(f"{estimator} is not an estimator")

    estimator_class = estimator.__class__
    params = estimator.get_params(deep=False)

    try:
        new_estimator = estimator_class(**params)
    except TypeError:
        new_estimator = estimator_class()
        new_estimator.set_params(**params)

    return new_estimator


class ClassifierMixin(BaseEstimator, ABC):
    """
    Mixin class for classifiers.
    Provides methods for classification metrics.
    """

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the accuracy of the classifier on the given data and labels.
        :param X: Input data of shape (n_samples, n_features)
        :param y: True labels of shape (n_samples,)
        :return: Accuracy score as a float
        """
        y_pred = self.predict(X)
        return accuracy_score(y_true=y, y_pred=y_pred)


class RegressorMixin(BaseEstimator, ABC):
    """
    Mixin class for regressors.
    Provides methods for regression metrics.
    """

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R^2 score of the regressor on the given data and labels.
        :param X: Input data of shape (n_samples, n_features)
        :param y: True labels of shape (n_samples,)
        :return: R^2 score as a float
        """
        y_pred = self.predict(X)
        return r2_score(y_true=y, y_pred=y_pred)


class TransformerMixin(BaseEstimator, ABC):
    """
    Mixin class for transformers.
    Provides methods for transforming data.
    """

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data.
        :param X: Input data of shape (n_samples, n_features)
        :return: Transformed data of shape (n_samples, n_features_transformed)
        """
        raise NotImplementedError("TransformerMixin.transform() is not implemented. ")

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, **fit_params) -> np.ndarray:
        """
        Fit the transformer to the data and then transform it.
        :param X: Input data of shape (n_samples, n_features)
        :param y: Target values, optional
        :param fit_params: Additional parameters to pass to the fitting process
        :return: Transformed data of shape (n_samples, n_features_transformed)
        """
        return self.fit(X, y, **fit_params).transform(X)
