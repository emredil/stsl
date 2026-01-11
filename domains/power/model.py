# domains/power/model.py

import numpy as np

from core.pde_base import PDEModel
from core.solvers import diffusion_operator
from core.solvers import diffusion_stability_limit

class PowerCableModel(PDEModel):
    """
    Thermal-electrical coupled cable model.
    """

    def __init__(self, grid, params, dt, bc_left=0.0, bc_right=0.0):
        super().__init__(grid, params, dt)

        required = ["alpha", "I", "R0", "beta", "T0"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing parameter: {key}")

        self.alpha = params["alpha"]
        self.I = params["I"]
        self.R0 = params["R0"]
        self.beta = params["beta"]
        self.T0 = params["T0"]

        self.bc_left = bc_left
        self.bc_right = bc_right
    def resistance(self, T):
        """
        Temperature-dependent resistance model.
        """
        return self.R0 * (1.0 + self.beta * (T - self.T0))
    def source_term(self, t, T=None):
        """
        Joule heating: Q = I^2 * R(T)
        """
        if T is None:
            raise ValueError("Temperature field required for Joule heating.")

        R_T = self.resistance(T)
        return (self.I ** 2) * R_T
    def spatial_operator(self, T):
        return diffusion_operator(T, self.grid.dx, self.alpha)
    def apply_boundary_conditions(self, T):
        T[0] = self.bc_left
        T[-1] = self.bc_right
    def step(self, T, t):
        """
        Explicit time step with thermal-electrical coupling.
        """
        diffusion = self.spatial_operator(T)
        joule = self.source_term(t, T)

        T_new = T + self.dt * (diffusion + joule)
        self.apply_boundary_conditions(T_new)

        return T_new
    def stability_limit(self):
        return diffusion_stability_limit(self.grid.dx, self.alpha)
    def max_temperature(self, T):
        return np.max(T)
