import numpy as np
from ..utils.base import BaseEstimator, RegressorMixin
from typing import Dict, Optional

class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None

    @staticmethod
    def _calculate_gradient(X: np.ndarray, y: np.ndarray, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate gradients for linear regression using Mean Squared Error loss.

        This method computes the partial derivatives of the MSE loss function with respect
        to the weights and bias parameters. The gradients are calculated using the formulas:
        - dW = (1/m) * X.T @ (y_pred - y)
        - db = (1/m) * sum(y_pred - y)

        Where y_pred = X @ W + b

        Args:
            X (np.ndarray): Input features matrix of shape (m, n) where m is the number
                           of samples and n is the number of features.
            y (np.ndarray): Target values vector of shape (m,1).
            params (Dict[str, np.ndarray]): Dictionary containing current parameters:
                - 'weights': Weight vector of shape (n,1)
                - 'bias': Bias scalar value

        Returns:
            Dict[str, np.ndarray]: Dictionary containing computed gradients:
                - 'weights': Gradient with respect to weights, shape (n,1)
                - 'bias': Gradient with respect to bias, scalar
        """
        params_weights = params['weights']
        params_bias = params['bias']
        m, n = X.shape # m = n_samples, n = n_features

        y_pred = X.dot(params_weights) + params_bias
        db = float((1 / m) * np.sum(y_pred - y))
        dw = (1 / m) * X.T.dot(y_pred - y)

        return {'weights': dw, 'bias': np.array(db)}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model to the training data using the specified optimizer.

        This method trains the linear regression model by finding optimal weights and bias
        that minimize the Mean Squared Error between predictions and actual target values.
        The optimization process is handled by the optimizer passed during initialization.

        The linear regression model follows the equation: y = X @ W + b
        where W represents weights and b represents bias.

        Args:
            X (np.ndarray): Training input features matrix of shape (m, n) where m is the
                           number of training samples and n is the number of features.
            y (np.ndarray): Training target values vector of shape (m,1) containing the
                           ground truth values to be predicted.

        Returns:
            LinearRegression: Returns self to enable method chaining. The fitted model
                             with optimized weights and bias parameters.

        Note:
            After fitting, the model's weights and bias attributes will be updated with
            the optimized values found by the optimizer.
        """

        self._validate_inputs(X, y)

        m, n = X.shape
        self.weights_ = np.zeros((n, 1))
        self.bias_ = 0

        y_reshaped = y.reshape(-1, 1)


        initial_params = {
            'weights': self.weights_,
            'bias': np.array(self.bias_)
        }

        final_params = self.optimizer.optimize(
            X=X,
            y=y_reshaped,
            initial_params=initial_params,
            gradient_function=self._calculate_gradient
        )

        self.weights_ = final_params['weights']
        self.bias_ = float(final_params['bias'])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted linear regression model.

        This method applies the learned linear transformation to input features to generate
        predictions. The prediction follows the linear equation: y_pred = X @ W + b
        where W are the learned weights and b is the learned bias.

        Args:
            X (np.ndarray): Input features matrix of shape (m, n) where m is the number
                           of samples to predict and n is the number of features. The
                           number of features must match the number used during training.

        Returns:
            np.ndarray: Predicted values vector of shape (m,1) containing the linear
                       regression predictions for each input sample.

        Raises:
            AttributeError: If the model has not been fitted yet (weights or bias is None).

        Note:
            The model must be fitted using the fit() method before making predictions.
        """
        self._validate_X(X)

        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() before predict().")

        if X.shape[1] != self.weights_.shape[0]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) does not "
                             f"match the number of features used during training ({self.weights_.shape[0]}).")
        return X.dot(self.weights_) + self.bias_

    @staticmethod
    def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate input shapes for fit and predict methods.

        This method checks if the input features X and target values y have compatible
        shapes. It raises ValueError if the shapes do not match the expected dimensions.

        Args:
            X (np.ndarray): Input features matrix of shape (m, n).
            y (np.ndarray): Target values vector of shape (m, 1).

        Raises:
            ValueError: If the shapes of X and y are incompatible.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected X to be a numpy array, got type: {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"Expected y to be a numpy array, got type: {type(y)}")

        if X.ndim != 2:
            raise ValueError(f"Expected X to be a 2D array, got shape: {X.shape}")
        if y.ndim not in [1, 2]:
            raise ValueError(f"Expected y to be a 1D or 2D array, got shape: {y.shape}")

        # Check for empty arrays
        if X.size == 0:
            raise ValueError("X cannot be an empty array")
        if y.size == 0:
            raise ValueError("y cannot be an empty array")

        # Check for NaN values
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")
        if np.isnan(y).any():
            raise ValueError("y contains NaN values")

        # Check for infinite values
        if np.isinf(X).any():
            raise ValueError("X contains infinite values")
        if np.isinf(y).any():
            raise ValueError("y contains infinite values")

        m, n = X.shape # m = n_samples, n = n_features

        if y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError(f"Expected y to be a column vector, got shape: {y.shape}")
        if m != y.shape[0]:
            raise ValueError(f"Number of samples in X and y must match: {m} != {y.shape[0]}")

    @staticmethod
    def _validate_X(X: np.ndarray) -> None:
        """
        Validate input features X for predict method.

        This method checks if the input features X have the correct number of features
        that match the model's trained weights. It raises ValueError if the shape is
        incompatible.

        Args:
            X (np.ndarray): Input features matrix of shape (m, n).

        Raises:
            ValueError: If the number of features in X does not match the trained model.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected X to be a numpy array, got type: {type(X)}")

        if X.ndim != 2:
            raise ValueError(f"Expected X to be a 2D array, got shape: {X.shape}")

        # Check for empty arrays
        if X.size == 0:
            raise ValueError("X cannot be an empty array")

        # Check for NaN values
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")

        # Check for infinite values
        if np.isinf(X).any():
            raise ValueError("X contains infinite values")

    @property
    def coef_(self) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() before accessing coef_.")
        return self.weights_.flatten()

    @property
    def intercept_(self) -> float:
        if self.bias_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() before accessing intercept_.")
        return self.bias_