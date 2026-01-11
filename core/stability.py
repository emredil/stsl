# core/stability.py

import numpy as np

STABLE = "stable"
UNSTABLE = "unstable"
UNKNOWN = "unknown"

class StabilityReport:
    def __init__(self, status, message, dt=None, dt_max=None):
        self.status = status
        self.message = message
        self.dt = dt
        self.dt_max = dt_max

    def as_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "dt": self.dt,
            "dt_max": self.dt_max,
        }

def check_diffusion_stability(dt, dx, alpha):
    """
    Check explicit diffusion stability condition.
    """
    if alpha <= 0:
        return StabilityReport(
            status=UNKNOWN,
            message="Diffusion coefficient alpha <= 0; stability undefined."
        )

    dt_max = dx ** 2 / (2.0 * alpha)

    if dt <= dt_max:
        return StabilityReport(
            status=STABLE,
            message="Time step satisfies diffusion stability condition.",
            dt=dt,
            dt_max=dt_max
        )
    else:
        return StabilityReport(
            status=UNSTABLE,
            message="Time step exceeds diffusion stability limit.",
            dt=dt,
            dt_max=dt_max
        )

def check_with_safety_factor(dt, dt_max, safety=0.8):
    """
    Conservative stability check using safety margin.
    """
    if dt_max is None:
        return StabilityReport(
            status=UNKNOWN,
            message="No stability limit available."
        )

    safe_dt = safety * dt_max

    if dt <= safe_dt:
        return StabilityReport(
            status=STABLE,
            message=f"Time step is within {safety*100:.0f}% safety margin.",
            dt=dt,
            dt_max=safe_dt
        )
    else:
        return StabilityReport(
            status=UNSTABLE,
            message="Time step violates safety margin.",
            dt=dt,
            dt_max=safe_dt
        )

def check_model_stability(model):
    """
    Generic stability check for a PDEModel.
    """
    dt_max = model.stability_limit()

    if dt_max is None:
        return StabilityReport(
            status=UNKNOWN,
            message="Model does not define a stability limit."
        )

    if model.dt <= dt_max:
        return StabilityReport(
            status=STABLE,
            message="Model time step is stable.",
            dt=model.dt,
            dt_max=dt_max
        )
    else:
        return StabilityReport(
            status=UNSTABLE,
            message="Model time step is unstable.",
            dt=model.dt,
            dt_max=dt_max
        )
    
def explain_report(report):
    if report.status == STABLE:
        return f"Stable: dt = {report.dt:.2e}, limit = {report.dt_max:.2e}"
    elif report.status == UNSTABLE:
        return f"Unstable: dt = {report.dt:.2e}, limit = {report.dt_max:.2e}"
    else:
        return "Stability condition unknown for this model."

