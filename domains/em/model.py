# domains/em/model.py

import numpy as np

from core.pde_base import PDEModel
from core.solvers import diffusion_operator
from core.solvers import diffusion_stability_limit

class EMDiffusionModel(PDEModel):
    """
    Quasi-static electromagnetic diffusion model.
    """

    def __init__(self, grid, params, dt, bc_left=0.0, bc_right=0.0):
        super().__init__(grid, params, dt)

        required = ["mu", "sigma"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing parameter: {key}")

        self.mu = params["mu"]
        self.sigma = params["sigma"]

        self.D = 1.0 / (self.mu * self.sigma)

        self.bc_left = bc_left
        self.bc_right = bc_right
    def spatial_operator(self, B):
        """
        D * Laplacian(B)
        """
        return diffusion_operator(B, self.grid.dx, self.D)
    def apply_boundary_conditions(self, B):
        B[0] = self.bc_left
        B[-1] = self.bc_right
    def stability_limit(self):
        return diffusion_stability_limit(self.grid.dx, self.D)
    def skin_depth(self, omega):
        """
        Compute skin depth for angular frequency omega.
        """
        return np.sqrt(2.0 / (self.mu * self.sigma * omega))
    def magnetic_energy(self, B):
        return np.trapz(B**2, self.grid.x)
