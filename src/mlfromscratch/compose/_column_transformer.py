import numpy as np
from ..utils.base import BaseEstimator, TransformerMixin, clone
from ..utils.validation import validate_array, validate_X_y
import warnings

class ColumnTransformer(TransformerMixin, BaseEstimator):
    """
    Applies different transformers to different columns of a dataset.

    The ColumnTransformer allows you to selectively transform different columns
    of your data using different transformation strategies. This is particularly
    useful when you have mixed data types (e.g., numerical and categorical features)
    that require different preprocessing steps.

    Parameters
    ----------
    transformers : list of tuple
        List of (name, transformer, columns) tuples specifying the transformer
        objects to be applied to subsets of the data.
        - name : str
            Name of the transformer (must be unique).
        - transformer : TransformerMixin
            A transformer object that implements fit and transform methods.
        - columns : list of int
            Column indices to which the transformer should be applied.

    remainder : {'drop', 'passthrough'}, default='drop'
        Strategy for handling columns not specified in any transformer.
        - 'drop' : columns are dropped from the output
        - 'passthrough' : columns are passed through unchanged

    Attributes
    ----------
    fitted_transformers_ : list of tuple or None
        List of fitted (name, transformer, columns) tuples after calling fit.
        None before fitting.

    Raises
    ------
    TypeError
        If transformers is not a list, if any transformer is not a tuple of length 3,
        if any transformer object doesn't implement TransformerMixin, or if column
        lists are not lists of integers.
    ValueError
        If transformer names are not unique, if column indices overlap between
        transformers, or if remainder is not 'drop' or 'passthrough'.
    """

    def __init__(self, transformers: list[tuple[str, TransformerMixin, list[int]]], remainder = 'drop') -> None:
        """
        Initialize the ColumnTransformer.

        Parameters
        ----------
        transformers : list of tuple
            List of (name, transformer, columns) tuples specifying the transformer
            objects to be applied to subsets of the data.
            - name : str
                Name of the transformer (must be unique).
            - transformer : TransformerMixin
                A transformer object that implements fit and transform methods.
            - columns : list of int
                Column indices to which the transformer should be applied.

        remainder : {'drop', 'passthrough'}, default='drop'
            Strategy for handling columns not specified in any transformer.
            - 'drop' : columns are dropped from the output
            - 'passthrough' : columns are passed through unchanged

        Raises
        ------
        TypeError
            If transformers is not a list, if any transformer is not a tuple of
            length 3, if any transformer object doesn't implement TransformerMixin,
            or if column lists are not lists of integers.
        ValueError
            If transformer names are not unique, if column indices overlap between
            transformers, or if remainder is not 'drop' or 'passthrough'.
        """

        if not isinstance(transformers, list):
            raise TypeError("transformers must be a list of tuples (name, transformer, columns)")

        if not all(isinstance(t, tuple) and len(t) == 3 for t in transformers):
            raise ValueError("Each transformer must be a tuple of (name, transformer, columns)")

        input_names = [t[0] for t in transformers]
        input_columns = []

        input_transformers = [t[1] for t in transformers]
        if not all(isinstance(t, TransformerMixin) for t in input_transformers):
            raise TypeError("Each transformer must be an instance of TransformerMixin")

        if not all(isinstance(c, list) for c in input_columns):
            raise TypeError("Each column list must be a list of integers")

        if len(input_names) != len(set(input_names)):
            raise ValueError("Transformer names must be unique")


        for _, _, columns in transformers:
            input_columns.extend(columns)

        if len(input_columns) != len(set(input_columns)):
            raise ValueError("Column indices cannot overlap between transformers")

        if remainder not in ['drop', 'passthrough']:
            raise ValueError("remainder must be either 'drop' or 'passthrough'")

        self.transformers = transformers
        self.remainder = remainder
        self.fitted_transformers_ = None

        return

    def fit(self, X: np.ndarray, y: np.ndarray = None, **fit_params) -> 'ColumnTransformer':
        """
        Fit all transformers to their respective column subsets.

        This method fits each transformer to its designated columns in the input
        data. The transformers are fitted independently and their fitted states
        are stored for later use in transform().

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to fit the transformers on.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            Target values. Some transformers may use this for supervised
            transformations.
        **fit_params : dict
            Additional parameters to pass to the transformers' fit methods.
            Use the format 'transformer_name__parameter_name' to pass parameters
            to specific transformers.

        Returns
        -------
        self : ColumnTransformer
            Returns the fitted transformer instance.

        Raises
        ------
        ValueError
            If any column indices are out of bounds for the input data.
        KeyError
            If fit_params contains parameters for transformers not in the
            transformers list.
        RuntimeWarning
            If fit_params keys don't follow the 'transformer_name__param_name'
            format.
        """
        X, y = validate_X_y(X, y)

        for name, _, columns in self.transformers:
            if any(col < 0 or col >= X.shape[1] for col in columns):
                raise ValueError(f"Columns for transformer '{name}' are out of bounds for input data with shape {X.shape}")

        transformer_names = [name for name, _, _ in self.transformers]

        fit_params_for_transformers = {name: {} for name in transformer_names}

        for key, value in fit_params.items():
            if "__" not in key:
                warnings.warn(f"Parameter '{key}' is missing a __ character")
                continue

            transformer_name_in, params_name = key.split("__", 1)
            if transformer_name_in not in transformer_names:
                raise KeyError(f"Transformer {transformer_name_in} is not defined in the transformers list")

            fit_params_for_transformers[transformer_name_in][params_name] = value

        self.fitted_transformers_ = []
        for name, transformer, columns in self.transformers:

            X_subset = X[:, columns]
            transformer_clone = clone(transformer)
            fitted_transformer = transformer_clone.fit(X_subset, y, **fit_params_for_transformers[name])
            self.fitted_transformers_.append((name, fitted_transformer, columns))


        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the fitted transformers.

        This method applies each fitted transformer to its designated columns
        and concatenates the results. Columns not assigned to any transformer
        are handled according to the remainder strategy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform. Must have the same number of features
            as the data used during fitting.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_transformed)
            Transformed data. The number of output features depends on the
            transformers applied and the remainder strategy:
            - Each transformer may change the number of features for its columns
            - If remainder='passthrough', untransformed columns are included
            - If remainder='drop', untransformed columns are excluded

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet (fit() must be called first).
        ValueError
            If the input data has a different number of features than the data
            used during fitting.
        """
        X = validate_array(X)

        if self.fitted_transformers_ is None:
            raise RuntimeError("ColumnTransformer must be fitted before transforming data")

        transformed_parts = []

        for name, fitted_transformer, columns in self.fitted_transformers_:
            X_subset = X[:, columns]
            transformed_part = fitted_transformer.transform(X_subset)
            transformed_parts.append(transformed_part)

        if self.remainder == 'passthrough':
            all_columns = set(range(X.shape[1]))
            transformer_columns = set()
            for _, _, columns in self.fitted_transformers_:
                transformer_columns.update(columns)

            remaining_columns = sorted(all_columns - transformer_columns)
            if remaining_columns:
                remaining_X = X[:, list(remaining_columns)]
                transformed_parts.append(remaining_X)

        return np.hstack(transformed_parts) if transformed_parts else np.array([]).reshape(X.shape[0], 0)

    def predict(self, X: np.ndarray) -> np.ndarray:

        raise NotImplementedError("ColumnTransformer is a transformer, not a predictor. "
                                  "Use transform() method to transform data.")