# domains/solar/model.py

import numpy as np
from core.pde_base import PDEModel


class SolarPanelModel(PDEModel):
    """
    Lumped (0D) solar panel energy balance model.

    This is an ODE model intentionally implemented within the PDEModel
    framework for consistency across domains.
    """

    def __init__(self, params, dt):
        # No spatial grid (0D model)
        super().__init__(grid=None, params=params, dt=dt)

        required = ["C", "eta0", "gamma", "h", "Tamb"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing parameter: {key}")

        self.C = params["C"]
        self.eta0 = params["eta0"]
        self.gamma = params["gamma"]
        self.h = params["h"]
        self.Tamb = params["Tamb"]

    # ------------------------------------------------------------------
    # Required abstract methods (not used for ODE, implemented explicitly)
    # ------------------------------------------------------------------

    def spatial_operator(self, state):
        """
        No spatial operator for lumped (0D) model.
        """
        return 0.0

    def apply_boundary_conditions(self, state):
        """
        No boundary conditions for lumped (0D) model.
        """
        return state

    # ------------------------------------------------------------------
    # Solar physics
    # ------------------------------------------------------------------

    def efficiency(self, T):
        """
        Temperature-dependent electrical efficiency.
        """
        return self.eta0 * (1.0 - self.gamma * (T - self.Tamb))

    def irradiance(self, t):
        """
        Solar irradiance S(t).
        """
        return self.params.get("S", 1000.0)

    def step(self, T, t):
        """
        Explicit Euler time step for panel temperature.
        """
        S = self.irradiance(t)
        eta = self.efficiency(T)

        dTdt = (eta * S - self.h * (T - self.Tamb)) / self.C
        return T + self.dt * dTdt

    def stability_limit(self):
        """
        Characteristic thermal time scale.
        """
        return self.C / self.h

    def power_output(self, T, t):
        """
        Electrical power output.
        """
        return self.efficiency(T) * self.irradiance(t)
