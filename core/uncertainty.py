"""
Uncertainty quantification tools.

This module propagates parameter uncertainty through models
using Monte Carlo sampling.

Focus: transparency and interpretability, not speed.
"""

import numpy as np


def monte_carlo_propagation(
    model_factory,
    param_sampler,
    observable_fn,
    n_samples=100
):
    """
    Propagate parameter uncertainty through a model.

    Parameters
    ----------
    model_factory : callable
        Function mapping params dict -> model instance.
    param_sampler : callable
        Function that returns a sampled params dict.
    observable_fn : callable
        Function mapping model -> scalar observable.
    n_samples : int
        Number of Monte Carlo samples.

    Returns
    -------
    results : dict
        Contains samples, mean, std.
    """

    samples = []

    for _ in range(n_samples):
        params = param_sampler()
        model = model_factory(params)
        y = observable_fn(model)
        samples.append(y)

    samples = np.asarray(samples)

    return {
        "samples": samples,
        "mean": np.mean(samples),
        "std": np.std(samples),
        "min": np.min(samples),
        "max": np.max(samples),
    }
