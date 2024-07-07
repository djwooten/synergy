from typing import Any, Dict, Sequence

import numpy as np


def _assert_keys_equal(d1: dict, d2: dict):
    if d1.keys() != d2.keys():
        d1_only = [k for k in d1.keys() if k not in d2]
        d2_only = [k for k in d2.keys() if k not in d1]
        raise AssertionError(
            f"""Expected dicts to have identical keys.
Present in first only: {d1_only}
Present in second only: {d2_only}"""
        )


def assert_dict_allclose(
    actual: Dict[Any, float],
    desired: Dict[Any, float],
    rtol: float = 1e-07,
    atol: float = 0.0,
    equal_nan: bool = True,
    err_msg: str = "",
    verbose: bool = True,
):
    _assert_keys_equal(actual, desired)

    actual_vals = []
    desired_vals = []
    for key in actual.keys():
        actual_vals.append(actual[key])
        desired_vals.append(desired[key])

    np.testing.assert_allclose(
        np.asarray(actual_vals),
        np.asarray(desired_vals),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
        verbose=verbose,
    )


def assert_dict_values_in_intervals(
    values: dict, intervals: dict, tol: float = 0, err_msg: str = "", log_keys: Sequence[str] = []
):
    _assert_keys_equal(values, intervals)
    failed_vals = []
    for key in values.keys():
        val = values[key]
        interval = intervals[key]
        if key in log_keys:
            val = np.log(val)
            interval = tuple(np.log(x) for x in interval)
            key = f"log({key})"
        if not (interval[0] - tol <= val <= interval[1] + tol):
            failed_vals.append(f"{key}={val} not in ({interval[0]}, {interval[1]})")
    if failed_vals:
        err_msg = err_msg + "\n\t"
        raise AssertionError(err_msg + "\n\t".join(failed_vals))


def assert_dict_interval_is_contained_in_other(inner_intervals: dict, outer_intervals: dict, err_msg=""):
    _assert_keys_equal(inner_intervals, outer_intervals)

    for key in inner_intervals.keys():
        inner = inner_intervals[key]
        outer = outer_intervals[key]
        if inner[0] < outer[0] or inner[1] > outer[1]:
            raise AssertionError(f"Interval {inner} not within {outer}: {err_msg}")
