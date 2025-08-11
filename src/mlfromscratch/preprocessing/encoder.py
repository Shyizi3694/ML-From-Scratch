import numpy as np
from ..utils.base import BaseEstimator, TransformerMixin
from ..utils.validation import validate_array
import warnings

class OneHotEncoder(TransformerMixin, BaseEstimator):
    """
    One-hot encoder for transforming categorical features into binary vectors.
    
    This encoder converts categorical variables into a format suitable for machine learning
    algorithms by creating binary columns for each unique category. Each categorical feature
    is transformed into multiple binary features, where exactly one feature is 1 (hot) and
    the rest are 0 (cold) for each sample.
    
    The encoder only supports categorical (non-numeric) features and will raise an error
    if numeric features are provided. Unknown categories encountered during transformation
    will be ignored with a warning.
    
    Attributes
    ----------
    categories_ : list of ndarray or None
        The categories for each feature determined during fitting. Each element contains
        the unique categories found in the corresponding column of the training data.
        None if the encoder has not been fitted yet.
    
    Notes
    -----
    This implementation differs from scikit-learn's OneHotEncoder in that it:
    - Only accepts categorical (non-numeric) features
    - Handles unknown categories by ignoring them with a warning
    - Does not support sparse output format
    """
    def __init__(self):
        """
        Initialize the OneHotEncoder.
        
        Creates a new OneHotEncoder instance with no fitted categories.
        The encoder must be fitted before it can transform data.
        """
        self.categories_ = None
        
    def fit(self, X: np.ndarray, _y: np.ndarray = None, **_fit_params) -> 'OneHotEncoder':
        """
        Learn the unique categories for each feature in the input data.
        
        This method analyzes the input data to determine the unique categorical
        values in each column. These categories will be used to create the
        one-hot encoding structure during transformation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data containing categorical features to analyze.
            All features must be categorical (non-numeric).
        _y : array-like of shape (n_samples,), default=None
            Target values. Ignored, present for API compatibility with
            scikit-learn transformers.
        **_fit_params : dict
            Additional fit parameters. Ignored, present for API compatibility.
            
        Returns
        -------
        self : OneHotEncoder
            Returns the fitted encoder instance for method chaining.
            
        Raises
        ------
        ValueError
            If any column in X contains numeric data, as this encoder only
            supports categorical features.
        """

        self.categories_ = []

        X = validate_array(X, allow_nan = True, allow_inf=True)

        for col in range(X.shape[1]):
            if np.issubdtype(X[:, col].dtype, np.number):
                raise ValueError(f"Column {col} is numeric. OneHotEncoder only supports categorical features.")
            else:
                unique_values = np.unique(X[:, col])
                self.categories_.append(unique_values)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform categorical data into one-hot encoded binary vectors.
        
        For each categorical feature, this method creates binary columns corresponding
        to the categories learned during fitting. Each sample will have exactly one
        '1' in the columns corresponding to each original feature, indicating which
        category that sample belongs to.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform. Must have the same number of features
            as the data used during fitting. All features must be categorical.
            
        Returns
        -------
        X_encoded : ndarray of shape (n_samples, n_encoded_features)
            The one-hot encoded data where n_encoded_features is the sum of
            unique categories across all input features. Each original categorical
            feature is replaced by multiple binary features.
            
        Raises
        ------
        ValueError
            If the encoder has not been fitted yet, if the number of input features
            doesn't match the fitted data, or if any column contains numeric data.
            
        Warns
        -----
        UserWarning
            If any categories in the input data were not seen during fitting.
            These unknown categories will be ignored in the encoding.
        """
        if self.categories_ is None:
            raise ValueError("Encoder has not been fitted yet. Call 'fit' before 'transform'.")

        X = validate_array(X, allow_nan=True, allow_inf=True)

        if X.shape[1] != len(self.categories_):
            raise ValueError(f"Input data has {X.shape[1]} features, "
                             f"but encoder was fitted on {len(self.categories_)} features.")
        
        result = []

        for col in range(X.shape[1]):
            if np.issubdtype(X[:, col].dtype, np.number):
                raise ValueError(f"Column {col} is numeric. OneHotEncoder only supports categorical features.")
            
            categories = self.categories_[col]
            col_encoded = np.zeros((X.shape[0], len(categories)))

            for i, category in enumerate(categories):
                col_encoded[:, i] = np.where(X[:, col] == category, 1, 0)
            
            unknown_mask = ~np.isin(X[:, col], categories)
            if np.any(unknown_mask):
                unknown_categories = np.unique(X[unknown_mask, col])
                warnings.warn(f"Column {col} contains unknown categories: {unknown_categories}. "
                                "These will be ignored in the one-hot encoding.", UserWarning, stacklevel=2)
            result.append(col_encoded)
                
        return np.hstack(result)

    def predict(self, X: np.ndarray) -> np.ndarray:

        raise NotImplementedError("OneHotEncoder is a transformer, not a predictor. "
                                  "Use transform() method to encode data.")