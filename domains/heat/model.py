# domains/heat/model.py

import numpy as np

from core.pde_base import PDEModel
from core.solvers import diffusion_operator
from core.solvers import diffusion_stability_limit

class HeatEquation(PDEModel):
    """
    1D heat equation with fixed boundary temperatures.
    """

    def __init__(self, grid, params, dt, bc_left=0.0, bc_right=0.0):
        super().__init__(grid, params, dt)

        self.alpha = params.get("alpha", None)
        if self.alpha is None:
            raise ValueError("Thermal diffusivity 'alpha' must be provided.")

        self.bc_left = bc_left
        self.bc_right = bc_right
    def spatial_operator(self, T):
        """
        alpha * Laplacian(T)
        """
        return diffusion_operator(T, self.grid.dx, self.alpha)
    def source_term(self, t):
        """
        Optional heat source term Q(x).
        """
        Q = self.params.get("Q", 0.0)

        if np.isscalar(Q):
            return Q * np.ones(self.grid.nx)
        else:
            return np.asarray(Q)
    def apply_boundary_conditions(self, T):
        """
        Enforce fixed temperature boundaries.
        """
        T[0] = self.bc_left
        T[-1] = self.bc_right
    def stability_limit(self):
        return diffusion_stability_limit(self.grid.dx, self.alpha)
