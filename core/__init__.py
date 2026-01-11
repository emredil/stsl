"""
Core numerical and modeling infrastructure.

This package contains:
- Generic PDE abstractions
- Finite-difference operators
- Stability diagnostics
- Sensitivity and uncertainty analysis tools
"""

from .pde_base import PDEGrid, PDEModel
from .solvers import diffusion_operator, diffusion_stability_limit
from .stability import check_model_stability, explain_report
