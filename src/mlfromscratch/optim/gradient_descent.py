import numpy as np
from typing import Dict, Callable
from src.mlfromscratch.utils.validation import validate_X_y


class GradientDescent:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def optimize(self, X: np.ndarray, y: np.ndarray, initial_params: Dict[str, np.ndarray],
                 gradient_func: Callable[[np.ndarray, np.ndarray, Dict[str, np.ndarray]], Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Optimize parameters using gradient descent.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        - initial_params: Initial parameters for the model (dictionary).
        - gradient_func: Function to compute the gradient.

        Returns:
        - params: Optimized parameters (dictionary).
        """

        X, y = validate_X_y(X, y)

        params = initial_params.copy()
        # diff = {}
        for _ in range(self.n_iterations):
            gradient = gradient_func(X, y, params)

            if not gradient:
                raise ValueError("Gradient function returned an empty gradient. Check the implementation.")

            for param in params:
                if param not in gradient:
                    raise ValueError(f"Gradient for parameter '{param}' is missing. Check the gradient function.")

                # diff[param] = self.learning_rate * gradient[param]
                params[param] -= self.learning_rate * gradient[param]
        # print(diff)
        # print(params)
        return params

class StochasticGradientDescent:
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1000):

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        return

    def optimize(self, X: np.ndarray, y: np.ndarray, initial_params: Dict[str, np.ndarray],
                 gradient_func: Callable[[np.ndarray, np.ndarray, Dict[str, np.ndarray]], Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        X, y = validate_X_y(X, y)

        m = X.shape[0] # m = number of samples

        params = initial_params.copy()

        for _ in range(self.n_epochs):

            shuffled_indices = np.random.permutation(m)

            for i in range(m):

                index = shuffled_indices[i]
                x_i = X[index:index + 1]
                y_i = y[index:index + 1]

                gradient = gradient_func(x_i, y_i, params)

                if not gradient:
                    raise ValueError("Gradient function returned an empty gradient. Check the implementation.")

                for param in params:
                    if param not in gradient:
                        raise ValueError(f"Gradient for parameter '{param}' is missing. Check the gradient function.")

                    params[param] -= self.learning_rate * gradient[param]

        return params