# core/solvers.py

import numpy as np

def first_derivative_central(u, dx):
    """
    Compute first spatial derivative using central differences.
    Interior points only.
    """
    du_dx = np.zeros_like(u)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    return du_dx
def laplacian_1d(u, dx):
    """
    Compute 1D Laplacian using second-order central differences.
    """
    lap = np.zeros_like(u)
    lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx ** 2)
    return lap
def diffusion_operator(u, dx, alpha):
    """
    Diffusion operator: alpha * Laplacian(u)
    """
    return alpha * laplacian_1d(u, dx)
def state_dependent_source(u, func):
    """
    Apply a state-dependent source term.
    func(u) must return array of same shape.
    """
    return func(u)
def diffusion_stability_limit(dx, alpha):
    """
    Stability limit for explicit diffusion equation.
    """
    if alpha <= 0:
        return None
    return dx ** 2 / (2.0 * alpha)
def max_gradient(u, dx):
    du_dx = first_derivative_central(u, dx)
    return np.max(np.abs(du_dx))
