import logging
from typing import Sequence

import numpy as np
from scipy.stats import norm

from synergy.exceptions import ModelNotFitToDataError, ModelNotParameterizedError

_LOGGER = logging.Logger(__name__)


class ParametricModelMixins:
    """-"""

    @staticmethod
    def set_init_parameters(model, parameter_names: list[str], **kwargs):
        """-"""
        for param in parameter_names:
            model.__setattr__(param, kwargs.get(param, None))

    @staticmethod
    def set_parameters(model, parameter_names: list[str], *args):
        """-"""
        # TODO use this for synergy_model_2d just like with synergy_model_Nd
        if len(parameter_names) != len(args):
            raise ValueError("Number of parameters must match number of parameter names.")
        for param, value in zip(parameter_names, args):
            model.__setattr__(param, value)

    @staticmethod
    def set_bounds(model, transform, default_bounds: dict[str, tuple[float]], parameter_names: list[str], **kwargs):
        """-"""
        # TODO allow kwargs to include things like "E_bounds" or "alpha_bounds" to set bounds for ALL "E" or "alpha"
        # parameters. So the lookup would be like:
        # E0:
        #   E0_bounds
        #   E_bounds
        #   default_bounds["E0"]
        # use something like (lots of details to work out):
        # generic_bounds = {}
        # for kwarg in kwargs:
        #     if kwarg.endswith("_bounds") and kwarg not in parameter_names:  # generic bounds, not for a specific parameter
        #         generic_bounds[kwarg.split("_")[0]] = kwargs.pop(kwarg)
        # then later
        # for param in parameter_names:
        #     base_param = ""
        #     for key in generic_bounds:
        #         if param.startswith(key):  # this could be a bug - use _find_matching_parameter()
        #             base_param = key
        #             break
        #     lb, ub = kwargs.pop(
        #         f"{param}_bounds",
        #         generic_bounds.get(
        #             base_param,
        #             default_bounds.get(param, (-np.inf, np.inf))
        #         )
        #     )

        lower_bounds = []
        upper_bounds = []

        for param in parameter_names:
            lb, ub = kwargs.pop(f"{param}_bounds", default_bounds.get(param, (-np.inf, np.inf)))
            lower_bounds.append(lb)
            upper_bounds.append(ub)
        lower_bounds = list(transform(lower_bounds))
        upper_bounds = list(transform(upper_bounds))

        # Log warnings for any other "bounds" passed in
        for key in kwargs:
            if "_bounds" in key:
                _LOGGER.warning(f"Ignoring unexpected bounds for {type(model).__name__}: {key}={kwargs[key]}")

        model._bounds = lower_bounds, upper_bounds
        return lower_bounds, upper_bounds

    @staticmethod
    def bootstrap_parameter_ranges(model, E, use_jacobian, bootstrap_iterations, max_iterations, *args, **kwargs):
        """Identify confidence intervals for parameters using bootstrap resampling.

        Residuals are randomly sampled from a normal distribution with :math:`\sigma = \sqrt{\frac{RSS}{n - N}}`
        where :math:`RSS` is the residual sum of square, :math:`n` is the number of data points, and :math:`N` is the
        number of parameters.
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
        ci: dict[str, tuple[float, float]],
        tol: float,
        log: bool,
        gt_outcome: str,
        lt_outcome: str,
        default_outcome: str = "additive",
    ):
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
