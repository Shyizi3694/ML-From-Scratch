import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (np.ndarray): True target values.
    y_pred (np.ndarray): Predicted target values.

    Returns:
    float: The Mean Squared Error.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    return float(np.mean((y_true - y_pred) ** 2))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    y_true (np.ndarray): True target values.
    y_pred (np.ndarray): Predicted target values.

    Returns:
    float: The Root Mean Squared Error.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R-squared (coefficient of determination) score.

    Parameters:
    y_true (np.ndarray): True target values.
    y_pred (np.ndarray): Predicted target values.

    Returns:
    float: The R-squared score.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - ss_res / ss_tot)

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate the input arrays for regression metrics.

    Parameters:
    y_true (np.ndarray): True target values.
    y_pred (np.ndarray): Predicted target values.

    Raises:
    ValueError: If the input arrays are not valid.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")

    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays cannot contain NaN values")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays cannot contain infinite values")

    return y_true, y_pred