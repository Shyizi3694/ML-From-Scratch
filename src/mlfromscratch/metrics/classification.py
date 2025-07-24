import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy score of predictions.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.

    Returns:
    float: Accuracy score as a percentage.
    """

    y_true, y_pred = _validate_inputs(y_true, y_pred)
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = y_true.size
    return float(correct_predictions / total_predictions)

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the precision score of predictions.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.

    Returns:
    float: Precision score as a percentage.
    """

    y_true, y_pred = _validate_inputs(y_true, y_pred)
    tp, _, fp, _ = _calculate_confusion_matrix_values(y_true, y_pred)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)

def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the recall score of predictions.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.

    Returns:
    float: Recall score as a percentage.
    """

    y_true, y_pred = _validate_inputs(y_true, y_pred)
    tp, _, _, fn = _calculate_confusion_matrix_values(y_true, y_pred)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the F1 score of predictions.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.

    Returns:
    float: F1 score as a percentage.
    """

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate the inputs for accuracy_score function.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.

    Raises:
    ValueError: If the input arrays do not have the same shape.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Input arrays must not contain infinite values.")

    return y_true, y_pred

def _calculate_confusion_matrix_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """
    Calculate the values for the confusion matrix.

    Parameters:
    y_true (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.

    Returns:
    tuple: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, tn, fp, fn