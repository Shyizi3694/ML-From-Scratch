import numpy as np
from ..utils.base import BaseEstimator, TransformerMixin
from ..utils.validation import validate_array
import warnings

class SimpleImputer(TransformerMixin, BaseEstimator):
    """
    Univariate imputer for completing missing values with simple strategies.
    
    This transformer replaces missing values using a descriptive statistic (e.g. mean, median, 
    or most frequent) along each column, or using a constant value. The imputer can handle both
    numerical and categorical data depending on the strategy chosen.
    
    The imputer stores the imputation statistics computed during fit and applies them during
    transform to handle missing values in new data.
    
    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy to use for replacing missing values.
        - 'mean': Replace missing values using the mean along each column. 
          Only applicable for numerical features.
        - 'median': Replace missing values using the median along each column.
          Only applicable for numerical features.
        - 'most_frequent': Replace missing values using the most frequent value
          along each column. Can be used with strings or numerical features.
    
    Attributes
    ----------
    statistics_ : ndarray of shape (n_features,)
        The imputation fill value for each feature. Contains the computed
        statistics (mean, median, or most frequent) for each column after fitting.
    
    Notes
    -----
    When using 'most_frequent' strategy, if multiple values have the same frequency,
    the first encountered value will be selected.
    
    For columns containing only NaN values, a default value of 0.0 is used when
    applying 'most_frequent' strategy, and a UserWarning is issued.
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "SimpleImputer is a transformer, not a predictor. "
            "Use transform() method to impute missing values.")

    def __init__(self, strategy: str = 'mean'):
        """
        Initialize the SimpleImputer with the specified imputation strategy.

        Parameters
        ----------
        strategy : str, default='mean'
            The imputation strategy to use for replacing missing values.
            Must be one of 'mean', 'median', or 'most_frequent'.

        Raises
        ------
        ValueError
            If the provided strategy is not supported. Supported strategies
            are 'mean', 'median', and 'most_frequent'.
        """
        if strategy not in ['mean', 'median', 'most_frequent']:
            raise ValueError(f"Invalid strategy: {strategy}. Supported strategies are 'mean', 'median', 'most_frequent'.")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X: np.ndarray, _y: np.ndarray = None, **_fit_params) -> 'SimpleImputer':
        """
        Fit the imputer on input data X.
        
        Computes the imputation statistics (mean, median, or most frequent value)
        for each feature column in the input data. These statistics will be used
        later during transform to replace missing values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data where missing values are represented as NaN.
            The imputer will compute statistics along each column.
        _y : ndarray of shape (n_samples,), default=None
            Target values. Ignored in this unsupervised transformer.
        **_fit_params : dict
            Additional fitting parameters. Ignored in this implementation.

        Returns
        -------
        self : SimpleImputer
            Returns the fitted imputer instance for method chaining.
            
        Notes
        -----
        After fitting, the computed statistics are stored in the `statistics_`
        attribute and can be accessed for inspection.
        """
        X = validate_array(X, allow_nan=True)

        strategy_function = {
            'mean': lambda x: np.nanmean(x, axis=0),
            'median': lambda x: np.nanmedian(x, axis=0),
            'most_frequent': lambda x: self._calculate_most_frequent(x)
        }

        self.statistics_ = strategy_function[self.strategy](X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute all missing values in X using the fitted statistics.
        
        Replaces all NaN values in the input data with the corresponding
        imputation values computed during the fit phase. Each column's
        missing values are replaced with that column's statistic.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data containing missing values (NaN) to be imputed.
            Must have the same number of features as the data used during fit.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The input data with all missing values replaced by the appropriate
            imputation values. The output array is a copy of the input.

        Raises
        ------
        ValueError
            If the imputer has not been fitted yet (statistics_ is None).
        ValueError
            If the number of features in X does not match the number of
            features in the data used during fitting.
            
        Notes
        -----
        The transformation creates a copy of the input data, so the original
        array is not modified.
        """
        if self.statistics_ is None:
            raise ValueError("You must fit the imputer before transforming data.")

        X = validate_array(X, allow_nan=True)

        if X.shape[1] != self.statistics_.shape[0]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) does not "
                             f"match the number of features in the fitted data ({self.statistics_.shape[0]}).")
        
        X_copy = X.copy()
        for col in range(X_copy.shape[1]):
            X_copy[np.isnan(X_copy[:, col]), col] = self.statistics_[col]

        return X_copy

    @staticmethod
    def _calculate_most_frequent(X: np.ndarray) -> np.ndarray:
        """
        Calculate the most frequent (mode) value for each column in the input array.
        
        For each column, computes the value that appears most frequently among
        non-NaN values. If a column contains only NaN values, a default value
        of 0.0 is used and a warning is issued.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data array where missing values are represented as NaN.
            Each column will be processed independently to find its mode.

        Returns
        -------
        most_frequent_values : ndarray of shape (n_features,)
            Array containing the most frequent value for each column.
            For columns with only NaN values, contains the default value 0.0.

        Warns
        -----
        UserWarning
            When a column contains only NaN values, warning about using
            the default value 0.0 for imputation.
            
        Notes
        -----
        When multiple values have the same highest frequency in a column,
        the first value encountered (in sorted order) is selected as the mode.
        """
        most_frequent = []

        for col in range(X.shape[1]):
            col_data = X[:, col]
            col_data_valid = col_data[~np.isnan(col_data)]
            if len(col_data_valid) == 0:
                default_value = 0.0
                warnings.warn(f"Column {col} contains only NaN values. Using default value {default_value}.",
                              UserWarning, stacklevel=2)
                most_frequent.append(default_value)
            else: # len(col_data_valid) > 0
                unique_values, counts = np.unique(col_data_valid, return_counts=True)
                most_frequent_idx = np.argmax(counts)
                most_frequent.append(unique_values[most_frequent_idx])

        return np.array(most_frequent)