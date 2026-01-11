"""
Sensitivity analysis tools.

This module provides local (finite-difference) sensitivity estimates
to quantify how model outputs respond to small parameter changes.

The goal is not optimization, but understanding robustness and risk.
"""

import copy
import numpy as np


def local_sensitivity(model_factory, base_params, param_name, delta, observable_fn):
    """
    Compute local sensitivity of an observable with respect to one parameter.

    Parameters
    ----------
    model_factory : callable
        Function that takes params dict and returns a model instance.
    base_params : dict
        Baseline model parameters.
    param_name : str
        Parameter to perturb.
    delta : float
        Small perturbation applied to the parameter.
    observable_fn : callable
        Function mapping model -> scalar observable.

    Returns
    -------
    sensitivity : float
        Finite-difference sensitivity estimate.
    """

    params_plus = copy.deepcopy(base_params)
    params_plus[param_name] += delta

    model_base = model_factory(base_params)
    model_plus = model_factory(params_plus)

    y0 = observable_fn(model_base)
    y1 = observable_fn(model_plus)

    return (y1 - y0) / delta
