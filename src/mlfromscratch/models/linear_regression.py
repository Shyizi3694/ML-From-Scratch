import numpy as np
from ..utils.base import BaseEstimator, RegressorMixin
from ..utils.validation import validate_X_y, validate_array
from typing import Dict, Optional

class LinearRegression(RegressorMixin, BaseEstimator):
    """
    Ordinary Least Squares Linear Regression using customizable optimizers.

    LinearRegression fits a linear model with coefficients W = (w1, ..., wp) to minimize
    the residual sum of squares between the observed targets in the dataset, and the
    targets predicted by the linear approximation. The model assumes a linear relationship
    between input features and target values following the equation: y = XW + b.

    This implementation allows for different optimization algorithms to be used for
    parameter estimation, providing flexibility in the training process. The model
    uses Mean Squared Error as the loss function for optimization.

    Parameters
    ----------
    optimizer : Optimizer
        The optimization algorithm to use for parameter estimation. Must implement
        an optimize method that takes gradient functions and returns optimized parameters.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features, 1) or None
        Estimated coefficients for the linear regression problem. Available after
        fitting the model.
    bias_ : float or None
        Independent term (intercept) in the linear model. Available after fitting.
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem (flattened weights).
        Read-only property, available after fitting.
    intercept_ : float
        Independent term (intercept) in the linear model. Alias for bias_.
        Read-only property, available after fitting.

    Notes
    -----
    The linear regression model follows the mathematical formulation:
    y = X @ W + b + ε
    where ε represents the error term with zero mean.

    The optimization minimizes the Mean Squared Error:
    MSE = (1/2m) * ||y - (XW + b)||²
    """

    # Annotations for properties
    coef_: np.ndarray
    intercept_: float

    def __init__(self, optimizer):
        """
        Initialize the LinearRegression model with a specified optimizer.

        Parameters
        ----------
        optimizer : Optimizer
            The optimization algorithm instance to use for parameter estimation.
            Must provide an optimize method compatible with the gradient-based
            optimization interface.
        """
        self.optimizer = optimizer
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None

    @staticmethod
    def _calculate_gradient(X: np.ndarray, y: np.ndarray, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute gradients of Mean Squared Error loss with respect to model parameters.

        This method calculates the partial derivatives of the MSE loss function with
        respect to weights and bias using analytical derivatives. The gradients are
        essential for gradient-based optimization algorithms.

        The MSE loss function is: L = (1/2m) * ||y - (XW + b)||²
        The computed gradients are:
        - ∂L/∂W = (1/m) * X^T @ (XW + b - y)
        - ∂L/∂b = (1/m) * Σ(XW + b - y)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix containing training samples.
        y : np.ndarray of shape (n_samples, 1)
            Target values vector for training samples.
        params : Dict[str, np.ndarray]
            Current model parameters containing:
            - 'weights': Weight matrix of shape (n_features, 1)
            - 'bias': Bias term as scalar in array form

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing computed gradients:
            - 'weights': Gradient w.r.t. weights, shape (n_features, 1)
            - 'bias': Gradient w.r.t. bias, scalar in array form

        Notes
        -----
        The gradient computation assumes the linear model y_pred = X @ W + b
        and uses vectorized operations for computational efficiency.
        """
        params_weights = params['weights']
        params_bias = params['bias']
        m, n = X.shape # m = n_samples, n = n_features

        y_pred = X.dot(params_weights) + params_bias
        db = float((1 / m) * np.sum(y_pred - y))
        dw = (1 / m) * X.T.dot(y_pred - y)

        return {'weights': dw, 'bias': np.array(db)}

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'LinearRegression':
        """
        Fit the linear regression model to training data using the configured optimizer.

        This method estimates optimal model parameters (weights and bias) that minimize
        the Mean Squared Error between predictions and actual target values. The
        optimization process uses the gradient-based optimizer provided during
        model initialization.

        The fitting process involves:
        1. Data validation and preprocessing
        2. Parameter initialization (weights to zeros, bias to zero)
        3. Iterative optimization using computed gradients
        4. Storage of final optimized parameters

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training input feature matrix where each row represents a sample
            and each column represents a feature.
        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Training target values. Will be reshaped to column vector internally.
        **fit_params : dict
            Additional parameters to pass to the optimizer's optimize method.
            Specific parameters depend on the optimizer implementation.

        Returns
        -------
        self : LinearRegression
            Returns the fitted estimator instance to enable method chaining.

        Raises
        ------
        ValueError
            If input arrays have incompatible shapes or contain invalid values.

        Notes
        -----
        After successful fitting, the model's weights_ and bias_ attributes
        will contain the optimized parameters ready for making predictions.
        """

        X, y = validate_X_y(X, y, dtype = np.float64)

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
        Generate predictions using the fitted linear regression model.

        This method applies the learned linear transformation to input features
        to produce predictions. The prediction computation follows the linear
        equation: y_pred = X @ W + b, where W and b are the learned parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix for which to generate predictions. The number
            of features must match the number used during model training.

        Returns
        -------
        np.ndarray of shape (n_samples, 1)
            Predicted target values for the input samples. Each row corresponds
            to the prediction for the corresponding input sample.

        Raises
        ------
        ValueError
            If the model has not been fitted yet, or if the number of features
            in X does not match the number used during training.

        Notes
        -----
        The model must be successfully fitted using the fit() method before
        predictions can be made. The prediction is a linear combination of
        input features with learned weights plus the learned bias term.
        """
        X = validate_array(X, dtype = np.float64)

        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() before predict().")

        if X.shape[1] != self.weights_.shape[0]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) does not "
                             f"match the number of features used during training ({self.weights_.shape[0]}).")
        return X.dot(self.weights_) + self.bias_

    @property
    def coef_(self) -> np.ndarray:
        """
        Get the fitted coefficients (weights) of the linear model.

        Returns
        -------
        np.ndarray of shape (n_features,)
            The estimated coefficients for each input feature in flattened form.
            These represent the linear relationship between each feature and the target.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.weights_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() before accessing coef_.")
        return self.weights_.flatten()

    @property
    def intercept_(self) -> float:
        """
        Get the fitted intercept (bias) of the linear model.

        Returns
        -------
        float
            The estimated intercept term, representing the model's prediction
            when all features are zero.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.bias_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() before accessing intercept_.")
        return self.bias_