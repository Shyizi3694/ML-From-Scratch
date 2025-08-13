import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.

    This function splits the input data X and target y into training and testing
    sets using random sampling without replacement. The split is performed by
    shuffling indices rather than the data itself for better memory efficiency.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples to split.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values to split.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
        Should be between 0.0 and 1.0.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    X_train : ndarray of shape (n_train_samples, n_features)
        Training input samples.
    X_test : ndarray of shape (n_test_samples, n_features)
        Testing input samples.
    y_train : ndarray of shape (n_train_samples,) or (n_train_samples, n_outputs)
        Training target values.
    y_test : ndarray of shape (n_test_samples,) or (n_test_samples, n_outputs)
        Testing target values.

    Raises
    ------
    ValueError
        If test_size is not between 0.0 and 1.0, or if X and y have
        inconsistent numbers of samples.
    TypeError
        If X or y cannot be converted to numpy arrays.

    Notes
    -----
    This function shuffles indices rather than the actual data arrays for
    better memory efficiency, especially with large datasets.
    """
    # Convert inputs to numpy arrays
    try:
        X = np.asarray(X)
        y = np.asarray(y)
    except Exception as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")

    # Validate test_size
    if not 0.0 <= test_size <= 1.0:
        raise ValueError(f"test_size should be between 0.0 and 1.0, got {test_size}")

    # Validate dimensions
    if X.ndim < 2:
        X = X.reshape(-1, 1)

    n_samples_X = X.shape[0]
    n_samples_y = len(y)

    if n_samples_X != n_samples_y:
        raise ValueError(
            f"X and y have inconsistent numbers of samples: "
            f"X has {n_samples_X} samples, y has {n_samples_y} samples"
        )

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate split sizes
    n_samples = n_samples_X
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    # Generate shuffled indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Split data using indices
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
