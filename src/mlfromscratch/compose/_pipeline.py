from typing import Any, override

from ..utils.base import BaseEstimator, TransformerMixin
import numpy as np
from ..utils.validation import validate_array, validate_X_y
from collections import defaultdict

class Pipeline(BaseEstimator):
    """
    A sequential pipeline that chains multiple estimators together.

    The Pipeline allows for the sequential application of multiple transformations
    followed by a final estimator. All intermediate steps must be transformers
    (implementing fit and transform methods), while the final step can be any
    estimator (transformer or predictor).

    The pipeline provides a unified interface for fitting and predicting,
    automatically passing transformed data through each step in sequence.

    Parameters
    ----------
    steps : list of tuple(str, BaseEstimator)
        A list of (name, estimator) tuples defining the pipeline steps.
        All intermediate steps must implement TransformerMixin.

    Attributes
    ----------
    steps : list of tuple(str, BaseEstimator)
        The pipeline steps as provided during initialization.
    named_steps : dict[str, BaseEstimator]
        Dictionary mapping step names to their corresponding estimators.
    final_estimator : BaseEstimator
        The final estimator in the pipeline.

    Raises
    ------
    ValueError
        If steps is empty, step names are duplicated, or step names are invalid.
    TypeError
        If any step is not a BaseEstimator or intermediate steps don't implement
        TransformerMixin.
    """

    # Annotations for properties
    named_steps: dict[str, BaseEstimator]
    final_estimator: BaseEstimator

    def __init__(self, steps: list[tuple[str, BaseEstimator]]) -> None:
        """
        Initialize the Pipeline with a sequence of steps.

        Parameters
        ----------
        steps : list of tuple(str, BaseEstimator)
            A list of (name, estimator) tuples. Each name must be a non-empty
            string and unique within the pipeline. All intermediate estimators
            must implement TransformerMixin, while the final estimator can be
            any BaseEstimator.

        Raises
        ------
        ValueError
            If steps is empty, contains duplicate names, or has invalid step names.
        TypeError
            If any step is not a BaseEstimator or intermediate steps don't
            implement TransformerMixin.
        """

        if not steps:
            raise ValueError("Pipeline steps cannot be empty.")

        step_names = set()

        for i, (name, estimator) in enumerate(steps):

            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Step {i} name must be a non-empty string, got {name}")

            if name in step_names:
                raise ValueError(f"Step name '{name}' is duplicated in the pipeline.")
            step_names.add(name)

            if not isinstance(estimator, BaseEstimator):
                raise TypeError(f"Step {i} must be a BaseEstimator, got {type(estimator)}")

            is_final = (i == len(steps) - 1)
            is_transformer = isinstance(estimator, TransformerMixin)

            if not is_final and not is_transformer:
                raise TypeError(f"Step {i} must be a TransformerMixin, got {type(estimator)}")

        self.steps = steps
        self._fitted = False



    def fit(self, X: np.ndarray, y: np.ndarray = None, **fit_params) -> "Pipeline":
        """
        Fit all estimators in the pipeline sequentially.

        Each transformer in the pipeline is fitted on the input data and then
        transforms the data for the next step. The final estimator is fitted
        on the transformed data from all previous steps.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training input samples.
        y : np.ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Target values. Default is None.
        **fit_params : dict
            Parameters to pass to the fit methods of individual steps.
            Parameter names should be prefixed with the step name followed
            by double underscores (e.g., 'step_name__param_name').

        Returns
        -------
        self : Pipeline
            Returns the fitted pipeline instance.

        Raises
        ------
        ValueError
            If the pipeline contains invalid fit parameters.
        """
        X, y = validate_X_y(X, y)
        X_transformed = X.copy()

        sorted_fit_params = self._fit_params_sort(**fit_params)

        for i, (name, estimator) in enumerate(self.steps):
            step_params = sorted_fit_params.get(name, {})
            if i != len(self.steps) - 1:
                transformer: TransformerMixin = estimator # type: ignore
                X_transformed = transformer.fit_transform(X_transformed, y, **step_params)
            elif i == len(self.steps) - 1:
                estimator.fit(X_transformed, y, **step_params)

        self._fitted = True
        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data through the pipeline and predict using the final estimator.

        Applies all transformation steps sequentially to the input data,
        then uses the final estimator to make predictions on the transformed data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values from the final estimator.

        Raises
        ------
        AttributeError
            If the final estimator doesn't have a predict method.
        ValueError
            If the pipeline hasn't been fitted yet.
        """
        if not hasattr(self.steps[-1][1], "predict"):
            raise AttributeError(f"The last step must have a 'predict' method, got {type(self.steps[-1][1])}")

        if not self._fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit() before predict().")

        X = validate_array(X)
        X_transformed = X.copy()

        for (_, estimator) in self.steps[:-1]:
            transformer: TransformerMixin = estimator # type: ignore
            X_transformed = transformer.transform(X_transformed)

        return self.steps[-1][1].predict(X_transformed)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data through all steps in the pipeline.

        Applies all transformation steps sequentially to the input data,
        including the final step (which must also be a transformer).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples to transform.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features_out)
            Transformed input data after applying all pipeline steps.

        Raises
        ------
        AttributeError
            If the final estimator doesn't have a transform method.
        ValueError
            If the pipeline hasn't been fitted yet.
        """
        if not hasattr(self.steps[-1][1], "transform"):
            raise AttributeError(f"The last step must have a 'transform' method, got {type(self.steps[-1][1])}")

        if not self._fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit() before transform().")

        X = validate_array(X)
        X_transformed = X.copy()

        for (_, estimator) in self.steps:
            transformer: TransformerMixin = estimator # type: ignore
            X_transformed = transformer.transform(X_transformed)

        return X_transformed


    def _fit_params_sort(self, **fit_params) -> dict:
        """
        Sort and validate fit parameters by step name.

        Parses fit parameters with step-specific prefixes and organizes them
        into a dictionary structure for easy access during fitting.

        Parameters
        ----------
        **fit_params : dict
            Parameters with step-specific prefixes in the format
            'step_name__param_name=value'.

        Returns
        -------
        sorted_params : dict
            Dictionary mapping step names to their respective parameters.
            Structure: {step_name: {param_name: value, ...}, ...}

        Raises
        ------
        KeyError
            If parameter keys don't contain '__' separator.
        ValueError
            If referenced step names don't exist in the pipeline.
        """
        sorted_params = defaultdict(dict)
        step_names = {step[0] for step in self.steps}

        for key, value in fit_params.items():
            if "__" not in key:
                raise KeyError(f"Parameter '{key}' does not contain '__' to separate step name and parameter name.")

            step_name, param_name = key.split("__", 1)

            if step_name not in step_names:
                raise ValueError(f"Step '{step_name}' not found in pipeline steps.")

            sorted_params[step_name][param_name] = value

        return dict(sorted_params)

    @property
    def named_steps(self) -> dict[str, BaseEstimator]:
        """
        Get a dictionary mapping step names to their estimators.

        Returns
        -------
        named_steps : dict[str, BaseEstimator]
            Dictionary where keys are step names and values are the
            corresponding estimator instances.
        """
        return {name: estimator for name, estimator in self.steps}

    @property
    def final_estimator(self) -> BaseEstimator:
        """
        Get the final estimator in the pipeline.

        Returns
        -------
        estimator : BaseEstimator
            The last estimator in the pipeline sequence.
        """
        return self.steps[-1][1]

    @override
    def get_params(self, deep = True) -> dict[str, Any]:
        """
        Get parameters for this pipeline and all nested estimators.

        Returns parameters for the pipeline itself and, if deep=True,
        parameters for all estimators in the pipeline with step-prefixed names.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns parameters for nested estimators as well.
            Parameter names are prefixed with step names using double
            underscores (e.g., 'step_name__param_name').

        Returns
        -------
        params : dict[str, Any]
            Dictionary of parameters. For nested parameters, keys follow
            the format 'step_name__param_name'.
        """
        params = {}
        for name, estimator in self.named_steps.items():
            if deep:
                step_params = estimator.get_params(deep=True)
            else:
                step_params = estimator.get_params(deep=False)
            for param_name, param_value in step_params.items():
                params[f"{name}__{param_name}"] = param_value
        return params

    @override
    def set_params(self, **params) -> "Pipeline":
        """
        Set parameters for this pipeline and nested estimators.

        Allows setting parameters for individual steps using the format
        'step_name__param_name=value'.

        Parameters
        ----------
        **params : dict
            Parameters to set. For step-specific parameters, use the format
            'step_name__param_name=value'.

        Returns
        -------
        self : Pipeline
            Returns the pipeline instance with updated parameters.

        Raises
        ------
        KeyError
            If parameter keys don't contain '__' separator.
        ValueError
            If referenced step names don't exist in the pipeline.
        """
        for key, value in params.items():
            if "__" not in key:
                raise KeyError(f"Parameter '{key}' does not contain '__' to separate step name and parameter name.")

            step_name, param_name = key.split("__", 1)

            if step_name not in [self_step_name for self_step_name, _ in self.steps]:
                raise ValueError(f"Step '{step_name}' not found in pipeline steps.")

            step_estimator = self.named_steps[step_name]

            step_estimator.set_params(**{param_name: value})

        return self

    @classmethod
    def from_estimators(cls, *estimators: BaseEstimator, name_choice: str = "class_name") -> "Pipeline":
        """
        Create a Pipeline from a sequence of estimators with automatic naming.

        Convenience method for creating a pipeline when you have a sequence
        of estimators but don't want to manually specify step names.

        Parameters
        ----------
        *estimators : BaseEstimator
            Sequence of estimators to include in the pipeline.
            All intermediate estimators must implement TransformerMixin.
        name_choice : {'class_name', 'number'}, default='class_name'
            Strategy for generating step names:
            - 'class_name': Use lowercase class names with underscores
            - 'number': Use 'step_0', 'step_1', etc.

        Returns
        -------
        pipeline : Pipeline
            A new Pipeline instance with automatically generated step names.

        Raises
        ------
        ValueError
            If name_choice is invalid or no estimators are provided.
        TypeError
            If any estimator is not a BaseEstimator.
        """
        if name_choice not in ["number", "class_name"]:
            raise ValueError("name_choice must be either 'number' or 'class_name'.")

        if not estimators:
            raise ValueError("At least one estimator must be provided.")

        steps = []
        for i, estimator in enumerate(estimators):
            if not isinstance(estimator, BaseEstimator):
                raise TypeError(f"Expected {estimator.__class__.__name__} but got {type(estimator)}")

            name = None

            if name_choice == "number":
                name = f"step_{i}"

            elif name_choice == "class_name":
                import re
                class_name = estimator.__class__.__name__
                name = re.sub('([A-Z]+)', r'_\1', class_name).lower().strip('_')
                base_name = name
                counter = 1
                exist_name = [step[0] for step in steps]
                while name in exist_name:
                    name = f"{base_name}_{counter}"
                    counter += 1

            steps.append((name, estimator))


        return cls(steps)