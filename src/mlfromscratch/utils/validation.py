import numpy as np

def validate_array(X: np.ndarray, ensure_2d: bool = True,
                    allow_nan: bool = False, allow_inf: bool = False, dtype: type = None) -> np.ndarray:
    """
    Validate and convert input array to a standardized format.

    This function performs comprehensive validation of input arrays, ensuring they
    meet the requirements for machine learning algorithms. It handles type conversion,
    dimensionality checks, and data quality validation.

    Parameters
    ----------
    X : array-like
        Input array to be validated. Can be a numpy array, list, or any array-like
        structure that can be converted to a numpy array.
    ensure_2d : bool, default=True
        If True, ensures the output array is 2-dimensional. Raises ValueError if
        the input cannot be reshaped to 2D.
    allow_nan : bool, default=False
        If False, raises ValueError when NaN values are detected in the array.
        If True, NaN values are allowed to pass through.
    allow_inf : bool, default=False
        If False, raises ValueError when infinite values are detected in the array.
        If True, infinite values are allowed to pass through.
    dtype : type, optional
        If specified, converts the input array to this dtype. If None, the input
        array is converted to a numpy array with the default dtype (usually float64).

    Returns
    -------
    np.ndarray
        Validated numpy array with dtype float64. If ensure_2d is True, the array
        will be 2-dimensional.

    Raises
    ------
    TypeError
        If the input cannot be converted to a numpy array.
    ValueError
        If any of the following conditions are met:
        - Array is empty (size == 0)
        - ensure_2d is True and array is not 2-dimensional
        - allow_nan is False and array contains NaN values
        - allow_inf is False and array contains infinite values
    """
    if not isinstance(X, np.ndarray):
        try:
            if dtype is not None:
                X = np.asarray(X, dtype=dtype)
            else:
                X = np.asarray(X)
        except (ValueError, TypeError):
            raise TypeError(f"Expected X to be a numpy array, got type: {type(X)}")
    else:
        if dtype is not None:
            X = X.astype(dtype, copy=False)

    if ensure_2d:
        if X.ndim != 2:
            raise ValueError(f"Expected X to be a 2D array, got shape: {X.shape}")

    # Check for empty arrays
    if X.size == 0:
        raise ValueError("X cannot be an empty array")

    if not allow_nan:
        # Check for NaN values
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")

    if not allow_inf:
        # Check for infinite values
        if np.isinf(X).any():
            raise ValueError("X contains infinite values")
    return X

def validate_X_y(X: np.ndarray, y: np.ndarray, X_ensure_2d: bool = True,
                    X_allow_nan: bool = False, X_allow_inf: bool = False, X_dtype: type = None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Validate input features and target arrays for machine learning tasks.

    This function performs comprehensive validation of feature matrix X and target
    vector y, ensuring they have compatible shapes and meet the requirements for
    supervised learning algorithms. Both arrays are validated individually and
    then checked for compatibility.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix where each row represents a sample and each column
        represents a feature.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Target values corresponding to the samples in X. Can be 1D or 2D array.
    X_ensure_2d : bool, default=True
        If True, ensures the feature matrix X is 2-dimensional. Raises ValueError
        if the input cannot be reshaped to 2D.
    X_allow_nan : bool, default=False
        If False, raises ValueError when NaN values are detected in the feature matrix X.
        If True, NaN values are allowed to pass through.
    X_allow_inf : bool, default=False
        If False, raises ValueError when infinite values are detected in the feature matrix X.
        If True, infinite values are allowed to pass through.
    X_dtype : type, optional
        If specified, converts the feature matrix X to this dtype. If None, the input
        array is converted to a numpy array with the default dtype (usually float64).

    Returns
    -------
    X_validated : np.ndarray of shape (n_samples, n_features)
        Validated feature matrix with dtype float64.
    y_validated : np.ndarray of shape (n_samples, 1)
        Validated target array reshaped to column vector with dtype float64.

    Raises
    ------
    TypeError
        If X or y cannot be converted to numpy arrays.
    ValueError
        If any of the following conditions are met:
        - X or y contain NaN or infinite values
        - X or y are empty arrays
        - X is not 2-dimensional
        - y is not 1D or 2D
        - y is 2D but not a column vector (shape[1] != 1)
        - Number of samples in X and y don't match
    """
    X = validate_array(X, ensure_2d=X_ensure_2d,
                        allow_nan=X_allow_nan, allow_inf=X_allow_inf, dtype=X_dtype)

    if y is None:
        return X, None

    y = validate_array(y, ensure_2d=False)

    if y.ndim not in [1, 2]:
        raise ValueError(f"Expected y to be 1D or 2D array, got shape: {y.shape}")

    m, n = X.shape # m = n_samples, n = n_features

    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(f"Expected y to be a column vector, got shape: {y.shape}")

    if m != y.shape[0]:
        raise ValueError(f"Number of samples in X and y must match: {m} != {y.shape[0]}")

    return X, y.reshape(-1, 1) if y.ndim == 1 else y