import numpy as np
from typing import Dict, Callable

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
        params = initial_params.copy()
        for _ in range(self.n_iterations):
            gradient = gradient_func(X, y, params)

            if not gradient:
                raise ValueError("Gradient function returned an empty gradient. Check the implementation.")

            for param in params:
                if param not in gradient:
                    raise ValueError(f"Gradient for parameter '{param}' is missing. Check the gradient function.")

                params[param] -= self.learning_rate * gradient[param]

        return params