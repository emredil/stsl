# core/pde_base.py

import numpy as np
from abc import ABC, abstractmethod

class PDEGrid:
    def __init__(self, x_start, x_end, nx):
        self.x_start = x_start
        self.x_end = x_end
        self.nx = nx

        self.x = np.linspace(x_start, x_end, nx)
        self.dx = self.x[1] - self.x[0]

class PDEModel(ABC):
    def __init__(self, grid, params, dt):
        self.grid = grid
        self.params = params
        self.dt = dt

        self._check_params()
    def _check_params(self):
        if self.dt <= 0:
            raise ValueError("Time step dt must be positive.")
    @abstractmethod
    def spatial_operator(self, u):
        """
        Discretized spatial operator L_h(u).
        Example: alpha * Laplacian(u)
        """
        pass
    def source_term(self, t):
        """
        Optional source term f(x, t).
        Default: zero source.
        """
        return np.zeros(self.grid.nx)
    @abstractmethod
    def apply_boundary_conditions(self, u):
        """
        Modify u in-place to enforce boundary conditions.
        """
        pass
    def step(self, u, t):
        """
        Perform one explicit time step.
        """
        Lu = self.spatial_operator(u)
        f = self.source_term(t)

        u_new = u + self.dt * (Lu + f)
        self.apply_boundary_conditions(u_new)

        return u_new
    def run(self, u0, t_final):
        nt = int(t_final / self.dt)
        u = u0.copy()

        history = [u.copy()]
        times = [0.0]

        for n in range(nt):
            t = times[-1]
            u = self.step(u, t)

            history.append(u.copy())
            times.append(t + self.dt)

        return np.array(times), np.array(history)
    def stability_limit(self):
        """
        Returns maximum stable dt if known.
        Otherwise returns None.
        """
        return None
    def check_stability(self):
        dt_max = self.stability_limit()
        if dt_max is None:
            return "unknown"

        if self.dt <= dt_max:
            return "stable"
        else:
            return "unstable"
