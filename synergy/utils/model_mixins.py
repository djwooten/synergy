"""Methods used by both 2d and Nd synergy models."""

import logging
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from synergy.exceptions import ModelNotFitToDataError, ModelNotParameterizedError

_LOGGER = logging.Logger(__name__)


class ParametricModelMixins:
    """Utility functions for parametric models."""

    @staticmethod
    def set_init_parameters(model, parameter_names: Sequence[str], **kwargs) -> None:
        """Set parameters for a model passed to the models' constructor.

        For instance, this lets us call `model = Hill(E0=1.0, Emax=0.0, h=1.0, C=1.0)` to create a fully funcitonal
        Hill model.

        :param model: The model to set parameters for.
        :param Sequence[str] parameter_names: Names of the parameters that can be set.
        :param kwargs: The kwargs supplied to the constructor, which may contain these initial values.
        """
        for param in parameter_names:
            model.__setattr__(param, kwargs.get(param, None))

    @staticmethod
    def set_parameters(model, parameter_names: Sequence[str], *args) -> None:
        """Set parameters for a model to values in args.

        :param model: The model to set parameters for.
        :param Sequence[str] parameter_names: Names of the parameters to be set.
        :param args[float] args: The parameter values
        """
        # TODO use this for synergy_model_2d just like with synergy_model_Nd
        if len(parameter_names) != len(args):
            raise ValueError("Number of parameters must match number of parameter names.")
        for param, value in zip(parameter_names, args):
            model.__setattr__(param, value)

    @staticmethod
    def set_bounds(
        model,
        transform: Callable,
        default_bounds: Dict[str, Tuple[float, float]],
        parameter_names: Sequence[str],
        **kwargs,
    ):
        """Set model._bounds for a model, which will be used when fitting the model.

        Bounds will be set using the following priorities
        1. Explicit bounds passed in kwargs, e.g., E0_bounds=(0, 1)
        2. Generic bounds for a class of parameters, e.g., E_bounds=(0, 1) applies for E0, E1, E2, ...
        3. Default bounds for a specific parameter, e.g., E0_bounds=(0, 1)
        4. (-inf, inf) if unspecified

        Bounds are then transformed using the provided transform function. For example, if "EC50" is a parameter with
        bounds (0.1, 10), and the transform function transforms EC50 to log10 space, then its bounds will be (-1, 1).

        :param model: The model to set bounds for.
        :param Callable transform: A function to transform the bounds into the space used for fitting.
        :param dict default_bounds: Default bounds for each parameter.
        :param Sequence[str] parameter_names: Names of the parameters to set bounds for.
        :param kwargs: Bounds for specific (e.g., E0_bounds=(-1, 1)) or generic parameters (e.g., E_bounds=(-1, 1)).
        """
        lower_bounds = []
        upper_bounds = []

        # Get generic bounds (not for a specific parameter, but a whole class, e.g., "E_bounds" covers E0, E1, E2, ...)
        generic_bounds = {}
        for kwarg in list(kwargs.keys()):  # list() to avoid dict.pop() errors during iteration
            if kwarg.endswith("_bounds") and kwarg not in parameter_names:
                generic_bounds[kwarg.replace("_bounds", "")] = kwargs.pop(kwarg)

        # Loop over all parameters and get bounds, in order of (1) kwargs, (2) generic_bounds, (3) default_bounds,
        # (4) (-inf, inf)
        for param in parameter_names:
            lb, ub = ParametricModelMixins._get_bound(param, generic_bounds, default_bounds, **kwargs)
            lower_bounds.append(lb)
            upper_bounds.append(ub)
        lower_bounds = list(transform(lower_bounds))
        upper_bounds = list(transform(upper_bounds))

        model._bounds = lower_bounds, upper_bounds
        return lower_bounds, upper_bounds

    @staticmethod
    def bootstrap_parameter_ranges(
        model, E, use_jacobian: bool, bootstrap_iterations: int, max_iterations: int, *args, **kwargs
    ):
        """Identify confidence intervals for parameters using bootstrap resampling.

        Residuals are randomly sampled from a normal distribution with :math:`\sigma = \sqrt{\frac{RSS}{n - N}}`
        where :math:`RSS` is the residual sum of square, :math:`n` is the number of data points, and :math:`N` is the
        number of parameters.

        This will set the property ```model.bootstrap_parameters``` which is an np.ndarray of shape
        (bootstrap_iterations, n_parameters).

        If fewer than ```bootstrap_iterations``` iterations converge, a warning is logged, but no error is raised.

        :param model: The model to bootstrap.
        :param ArrayLike E: The observed values.
        :param bool use_jacobian: Whether to use the Jacobian when fitting the model.
        :param int bootstrap_iterations: The number of bootstrap iterations to perform.
        :param int max_iterations: The maximum number of iterations to perform when fitting the model.
        :param args: Args to pass to model.E() to get model predicted values.
        :param kwargs: Additional arguments to pass to the model's _fit method.
        """
        if bootstrap_iterations <= 0:
            model.bootstrap_parameters = None
            return
        if not model.is_specified:
            raise ModelNotParameterizedError()
        if not model.is_converged:
            raise ModelNotFitToDataError()

        n_data_points = len(E)
        n_parameters = len(model.get_parameters())

        sigma_residuals = np.sqrt(model.sum_of_squares_residuals / (n_data_points - n_parameters))  # type: ignore

        E_model = model.E(*args)
        bootstrap_parameters = []

        count = 0
        while len(bootstrap_parameters) < bootstrap_iterations and count < max_iterations:
            count += 1
            residuals_step = norm.rvs(loc=0, scale=sigma_residuals, size=n_data_points)

            # Add random noise to model prediction
            E_iteration = E_model + residuals_step

            # Fit noisy data
            with np.errstate(divide="ignore", invalid="ignore"):
                popt1 = model._fit(*args, E_iteration, use_jacobian=use_jacobian, **kwargs)

            if popt1 is not None:
                bootstrap_parameters.append(popt1)

        if len(bootstrap_parameters) < bootstrap_iterations:
            _LOGGER.warning(
                f"Bootstrap reached max_iterations={max_iterations} before converging {bootstrap_iterations} times."
            )
        if len(bootstrap_parameters) > 0:
            model.bootstrap_parameters = np.vstack(bootstrap_parameters)
        else:
            _LOGGER.warning("No bootstrap iterations successfully converged.")
            model.bootstrap_parameters = None

    @staticmethod
    def make_summary_row(
        key: str,
        comp_val: int,
        val: float,
        ci: Dict[str, Tuple[float, float]],
        tol: float,
        log: bool,
        gt_outcome: str,
        lt_outcome: str,
        default_outcome: str = "additive",
    ) -> List[str]:
        """Create a string summary row for a synergy parameter.

        :param str key: The parameter name.
        :param int comp_val: The "additive" value that the synergy value will be compared against.
        :param float val: The synergy value.
        :param Dict[str, Tuple[float, float]] ci: Confidence intervals keyed by parameter names.
        :param float tol: The tolerance for comparing the synergy value to the comparison value.
        :param bool log: Whether to log-transform the synergy and comp values before comparing, which may be appropriate
            if ```tol > 0```.
        :param str gt_outcome: The outcome if the synergy value is greater than the comparison value.
        :param str lt_outcome: The outcome if the synergy value is less than the comparison value.
        :param str default_outcome: The default outcome if the synergy value is within the tolerance of the comparison.
        :return List[str]: A list of strings for the summary row, where each element is a column in the summary table.
        """
        if ci:
            lb, ub = ci[key]
            if lb > comp_val:
                comparison = f"> {comp_val}"
                outcome = gt_outcome
            elif ub < comp_val:
                comparison = f"< {comp_val}"
                outcome = lt_outcome
            else:
                comparison = f"~= {comp_val}"
                outcome = default_outcome
            return [key, f"{val:0.3g}", f"({lb:0.3g}, {ub:0.3g})", comparison, outcome]
        val_scaled = np.log(val) if log else val
        comp_val_scaled = np.log(comp_val) if log else comp_val
        if val_scaled > comp_val_scaled + tol:
            comparison = f"> {comp_val}"
            outcome = gt_outcome
        elif val_scaled < comp_val_scaled - tol:
            comparison = f"< {comp_val}"
            outcome = lt_outcome
        else:
            comparison = f"~= {comp_val}"
            outcome = default_outcome
        return [key, f"{val:0.3g}", comparison, outcome]

    @staticmethod
    def _find_matching_parameter(parameters: Sequence[str], prefix: str) -> str:
        """Find a parameter in a list that starts with a given prefix.

        If multiple parameters start with the prefix, the shortest one is returned.
        If no parameters start with the prefix, a warning is logged and an empty string is returned.

        ```python
        parameters = ["ele", "elephant", "gooses", "gopher"]
        prefix = "ele"
        matching_parameter = "ele"

        prefix = "go"
        matching_parameter = "???"
        ```
        """
        shortest_match = ""
        for param in parameters:
            if shortest_match and len(param) > len(shortest_match):
                continue
            if param.startswith(prefix):
                shortest_match = param
        if not shortest_match:
            _LOGGER.warning(f"No parameter starting with {prefix} found in {parameters}")
        return shortest_match

    @staticmethod
    def _get_bound(
        parameter: str,
        generic_bounds: Dict[str, Tuple[float, float]],
        default_bounds: Dict[str, Tuple[float, float]],
        **kwargs,
    ) -> Tuple[float, float]:
        """Get the default bounds for a parameter if it is not explicitly set."""
        generic_parameter = ParametricModelMixins._get_generic_parameter(generic_bounds, parameter)
        if generic_parameter:
            defaults = generic_bounds[generic_parameter]
        else:
            defaults = default_bounds.get(parameter, (-np.inf, np.inf))
        return kwargs.pop(f"{parameter}_bounds", defaults)

    @staticmethod
    def _get_generic_parameter(generic_bounds: Dict[str, Tuple[float, float]], parameter: str) -> str:
        """Match a parameter to the longest matching key in generic_bounds"""
        candidates = [key for key in generic_bounds if parameter.startswith(key)]
        if not candidates:
            return ""
        return max(candidates, key=len)
