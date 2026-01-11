## Simulation Trust & Sensitivity Lab (STSL)

**Simulation Trust & Sensitivity Lab** is a lightweight, physics-based toolkit designed to **expose numerical stability limits, parameter sensitivity, and uncertainty** in scientific simulations before they are used in critical engineering decisions.

This project demonstrates how **small modeling choices** (time step, discretization, physical parameters) can lead to **unstable, misleading, or unsafe results** if not handled carefully.

---

### Why this project exists

In many engineering and deep-tech teams:

* Simulations are fragile and poorly validated
* Stability limits are implicit or ignored
* Parameter sensitivity is not quantified
* Results are trusted without understanding failure modes

This project focuses on **trust**, not speed or scale.

---

### What this project demonstrates

* Explicit numerical stability checks (CFL conditions)
* Sensitivity of outcomes to physical parameters
* Reduced-order but physically correct modeling
* Cross-domain transfer of numerical methods
* Clear separation between physics, numerics, and visualization


STSL explicitly separates three complementary analyses:

* **Stability analysis**
  - Identifies numerical instability before simulation failure
  - Exposes CFL and time-step constraints

* **Sensitivity analysis**
  - Quantifies how strongly outputs depend on physical parameters
  - Uses local finite-difference derivatives

* **Uncertainty propagation**
  - Uses Monte Carlo sampling of uncertain inputs
  - Produces distributions of physically meaningful outputs
  - Exports raw samples for downstream risk analysis

---

### Supported domains

| Domain | Model                           | Key Insight                     |
| ------ | ------------------------------- | ------------------------------- |
| Heat   | 1D diffusion equation           | Stability & thermal diffusion   |
| Power  | Thermal–electrical coupling     | Ampacity & thermal runaway      |
| EM     | Quasi-static magnetic diffusion | Conductivity & skin effect      |
| Solar  | Lumped energy balance (ODE)     | Temperature-efficiency coupling |

All models are intentionally **simple, transparent, and explainable**.

---

### Project structure

```
stsl/
├── core/
│   ├── pde_base.py        # Generic PDE abstraction
│   ├── solvers.py         # Finite-difference operators
│   ├── stability.py       # Stability diagnostics
│   ├── sensitivity.py     # Local sensitivity analysis
│   └── uncertainty.py     # Monte Carlo uncertainty propagation
│
├── domains/
│   ├── heat/
│   ├── power/
│   ├── em/
│   └── solar/
│
├── ui/
│   └── app.py             # Streamlit interactive app
│
├── requirements.txt
└── README.md

```

---

### How to run the demo

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

Use the sidebar controls to:

* Change physical parameters
* Adjust grid size and time step
* Observe stability warnings
* Visualize failure modes

---



### Author

**Emre Dil**

**Email:** *[[emredil@sakarya.edu.tr](mailto:emredil@sakarya.edu.tr)]*

Computational Physicist | Scientific Modeling | Simulation & Stability

16+ years in theoretical and applied physics, power systems, and data-driven modeling.

---

> **STSL aims to expose numerical stability limits, parameter sensitivity, and uncertainty in scientific simulations before they are used in critical engineering decisions.**
