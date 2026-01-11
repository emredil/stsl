# ui/app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core.sensitivity import local_sensitivity
from core.uncertainty import monte_carlo_propagation
import pandas as pd
from io import BytesIO


st.set_page_config(page_title="Simulation Trust Lab", layout="wide")

st.title("Simulation Trust & Sensitivity Lab")
st.markdown("""
This tool demonstrates how **numerical stability, parameter sensitivity,
and physical assumptions** affect simulation outcomes across domains.
""")

domain = st.selectbox(
    "Select Simulation Domain",
    ["Heat", "Power Cable", "Electromagnetics", "Solar Panel"]
)

st.sidebar.header("Numerical Settings")

nx = st.sidebar.slider("Grid points (nx)", 20, 200, 100)
dt = st.sidebar.number_input("Time step Δt", value=1e-4, format="%.1e")
t_final = st.sidebar.number_input("Final time", value=1.0)

mode = st.radio(
    "Analysis Mode",
    ["Simulation", "Sensitivity", "Uncertainty"],
    horizontal=True
)

def export_csv(data_dict, filename):
    df = pd.DataFrame(data_dict)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


if domain == "Heat" and mode == "Simulation":
    from core.pde_base import PDEGrid
    from domains.heat.model import HeatEquation
    from core.stability import check_model_stability, explain_report

    st.subheader("1D Heat Conduction")

    alpha = st.sidebar.number_input("Thermal diffusivity α", value=1e-4, format="%.1e")
    Q = st.sidebar.number_input("Heat source Q", value=0.0)

    grid = PDEGrid(0.0, 1.0, nx)
    params = {"alpha": alpha, "Q": Q}

    model = HeatEquation(
        grid=grid,
        params=params,
        dt=dt,
        bc_left=0.0,
        bc_right=0.0
    )

    T0 = np.zeros(nx)
    times, T_hist = model.run(T0, t_final)

    report = check_model_stability(model)
    st.info(explain_report(report))

    fig, ax = plt.subplots()
    ax.plot(grid.x, T_hist[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("Temperature")
    ax.set_title("Final Temperature Profile")
    st.pyplot(fig)
elif domain == "Heat" and mode == "Sensitivity":
    from core.pde_base import PDEGrid
    from domains.heat.model import HeatEquation

    st.subheader("Local Sensitivity Analysis (Heat Equation)")

    param_name = st.selectbox(
        "Parameter to perturb",
        ["alpha", "Q"]
    )

    delta = st.number_input(
        "Perturbation Δ",
        value=1e-5,
        format="%.1e"
    )

    base_params = {
        "alpha": 1e-4,
        "Q": 0.0
    }

    grid = PDEGrid(0.0, 1.0, nx)

    def model_factory(p):
        return HeatEquation(
            grid=grid,
            params=p,
            dt=dt,
            bc_left=0.0,
            bc_right=0.0
        )

    def observable(model):
        T0 = np.zeros(nx)
        _, T_hist = model.run(T0, t_final)
        return np.max(T_hist[-1])

    sens = local_sensitivity(
        model_factory,
        base_params,
        param_name,
        delta,
        observable
    )

    export_csv(
        {
            "parameter": [param_name],
            "sensitivity": [sens],
        },
        "heat_sensitivity.csv"
    )

    st.metric(
        label=f"Sensitivity of max temperature to {param_name}",
        value=f"{sens:.4e}"
    )

    st.caption(
        "Local finite-difference sensitivity. "
        "Valid for small perturbations around baseline."
    )
elif domain == "Heat" and mode == "Uncertainty":
    from core.pde_base import PDEGrid
    from domains.heat.model import HeatEquation

    st.subheader("Uncertainty Propagation (Heat Equation)")

    n_samples = st.slider("Monte Carlo samples", 20, 500, 100)

    grid = PDEGrid(0.0, 1.0, nx)

    def param_sampler():
        return {
            "alpha": np.random.normal(1e-4, 1e-5),
            "Q": np.random.normal(0.0, 0.1)
        }

    def model_factory(p):
        return HeatEquation(
            grid=grid,
            params=p,
            dt=dt,
            bc_left=0.0,
            bc_right=0.0
        )

    def observable(model):
        T0 = np.zeros(nx)
        _, T_hist = model.run(T0, t_final)
        return np.max(T_hist[-1])

    results = monte_carlo_propagation(
        model_factory,
        param_sampler,
        observable,
        n_samples=n_samples
    )

    export_csv(
        {
            "sample": results["samples"],
        },
        "heat_uncertainty_samples.csv"
    )

    fig, ax = plt.subplots()
    ax.hist(results["samples"], bins=20)
    ax.set_xlabel("Max Temperature")
    ax.set_ylabel("Frequency")
    ax.set_title("Uncertainty in Peak Temperature")
    st.pyplot(fig)

    st.metric("Mean", f"{results['mean']:.3f}")
    st.metric("Std Dev", f"{results['std']:.3f}")
elif domain == "Power Cable" and mode == "Simulation":
    from core.pde_base import PDEGrid
    from domains.power.model import PowerCableModel
    from core.stability import check_model_stability, explain_report

    st.subheader("Thermal–Electrical Cable Model")

    alpha = st.sidebar.number_input("Thermal diffusivity α", value=1e-4, format="%.1e")
    I = st.sidebar.number_input("Current I (A)", value=200.0)
    R0 = st.sidebar.number_input("Reference resistance R0", value=0.01)
    beta = st.sidebar.number_input("Temp coefficient β", value=0.004)
    T0_ref = st.sidebar.number_input("Reference temperature T0", value=20.0)

    grid = PDEGrid(0.0, 1.0, nx)
    params = {
        "alpha": alpha,
        "I": I,
        "R0": R0,
        "beta": beta,
        "T0": T0_ref
    }

    model = PowerCableModel(
        grid=grid,
        params=params,
        dt=dt,
        bc_left=T0_ref,
        bc_right=T0_ref
    )

    T0 = T0_ref * np.ones(nx)
    times, T_hist = model.run(T0, t_final)

    report = check_model_stability(model)
    st.warning(explain_report(report))

    max_T = [np.max(T) for T in T_hist]

    fig, ax = plt.subplots()
    ax.plot(times, max_T)
    ax.set_xlabel("Time")
    ax.set_ylabel("Max Temperature")
    ax.set_title("Thermal Runaway Risk")
    st.pyplot(fig)
elif domain == "Power Cable" and mode == "Sensitivity":
    from core.pde_base import PDEGrid
    from domains.power.model import PowerCableModel

    st.subheader("Local Sensitivity Analysis (Power Cable)")

    param_name = st.selectbox(
        "Parameter to perturb",
        ["I", "R0", "beta", "alpha"]
    )

    delta = st.number_input(
        "Perturbation Δ",
        value=0.01,
        format="%.3f"
    )

    base_params = {
        "alpha": 1e-4,
        "I": 200.0,
        "R0": 0.01,
        "beta": 0.004,
        "T0": 20.0
    }

    grid = PDEGrid(0.0, 1.0, nx)

    def model_factory(p):
        return PowerCableModel(
            grid=grid,
            params=p,
            dt=dt,
            bc_left=p["T0"],
            bc_right=p["T0"]
        )

    def observable(model):
        T0 = model.params["T0"] * np.ones(nx)
        _, T_hist = model.run(T0, t_final)
        return np.max(T_hist[-1])

    sens = local_sensitivity(
        model_factory,
        base_params,
        param_name,
        delta,
        observable
    )

    export_csv(
        {
            "parameter": [param_name],
            "sensitivity": [sens],
        },
        "power_sensitivity.csv"
    )

    st.metric(
        label=f"Sensitivity of max temperature to {param_name}",
        value=f"{sens:.4e}"
    )

    st.caption(
        "Thermal runaway sensitivity computed via local finite differences."
    )
elif domain == "Power Cable" and mode == "Uncertainty":
    from core.pde_base import PDEGrid
    from domains.power.model import PowerCableModel

    st.subheader("Uncertainty Propagation (Power Cable)")

    n_samples = st.slider("Monte Carlo samples", 20, 500, 100)

    grid = PDEGrid(0.0, 1.0, nx)

    def param_sampler():
        return {
            "alpha": 1e-4,
            "I": np.random.normal(200.0, 10.0),
            "R0": np.random.normal(0.01, 0.001),
            "beta": 0.004,
            "T0": 20.0
        }

    def model_factory(p):
        return PowerCableModel(
            grid=grid,
            params=p,
            dt=dt,
            bc_left=p["T0"],
            bc_right=p["T0"]
        )

    def observable(model):
        T0 = model.params["T0"] * np.ones(nx)
        _, T_hist = model.run(T0, t_final)
        return np.max(T_hist[-1])

    results = monte_carlo_propagation(
        model_factory,
        param_sampler,
        observable,
        n_samples=n_samples
    )

    export_csv(
        {
            "sample": results["samples"],
        },
        "power_uncertainty_samples.csv"
    )

    fig, ax = plt.subplots()
    ax.hist(results["samples"], bins=20)
    ax.set_xlabel("Max Temperature")
    ax.set_ylabel("Frequency")
    ax.set_title("Thermal Runaway Risk Distribution")
    st.pyplot(fig)

    st.metric("Mean", f"{results['mean']:.2f}")
    st.metric("Std Dev", f"{results['std']:.2f}")
elif domain == "Electromagnetics" and mode == "Simulation":
    from core.pde_base import PDEGrid
    from domains.em.model import EMDiffusionModel
    from core.stability import check_model_stability, explain_report

    st.subheader("Quasi-Static EM Diffusion")

    mu = st.sidebar.number_input("Permeability μ", value=4e-7*np.pi, format="%.2e")
    sigma = st.sidebar.number_input("Conductivity σ", value=5.8e7, format="%.2e")

    grid = PDEGrid(0.0, 0.1, nx)
    params = {"mu": mu, "sigma": sigma}

    model = EMDiffusionModel(
        grid=grid,
        params=params,
        dt=dt,
        bc_left=1.0,
        bc_right=0.0
    )

    B0 = np.zeros(nx)
    times, B_hist = model.run(B0, t_final)

    report = check_model_stability(model)
    st.info(explain_report(report))

    fig, ax = plt.subplots()
    ax.plot(grid.x, B_hist[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("Magnetic Field B")
    ax.set_title("Final Field Distribution")
    st.pyplot(fig)
elif domain == "Electromagnetics" and mode == "Sensitivity":
    from core.pde_base import PDEGrid
    from domains.em.model import EMDiffusionModel

    st.subheader("Local Sensitivity Analysis (EM Diffusion)")

    param_name = st.selectbox(
        "Parameter to perturb",
        ["mu", "sigma"]
    )

    delta = st.number_input(
        "Perturbation Δ",
        value=1e-6,
        format="%.1e"
    )

    base_params = {
        "mu": 4e-7 * np.pi,
        "sigma": 5.8e7
    }

    grid = PDEGrid(0.0, 0.1, nx)

    def model_factory(p):
        return EMDiffusionModel(
            grid=grid,
            params=p,
            dt=dt,
            bc_left=1.0,
            bc_right=0.0
        )

    def observable(model):
        B0 = np.zeros(nx)
        _, B_hist = model.run(B0, t_final)
        B = B_hist[-1]
        return np.trapz(B**2, grid.x)

    sens = local_sensitivity(
        model_factory,
        base_params,
        param_name,
        delta,
        observable
    )

    export_csv(
        {
            "parameter": [param_name],
            "sensitivity": [sens],
        },
        "em_sensitivity.csv"
    )

    st.metric(
        label=f"Sensitivity of magnetic energy to {param_name}",
        value=f"{sens:.4e}"
    )

    st.caption(
        "Local sensitivity of EM diffusion energy to material parameters."
    )
elif domain == "Electromagnetics" and mode == "Uncertainty":
    from core.pde_base import PDEGrid
    from domains.em.model import EMDiffusionModel

    st.subheader("Uncertainty Propagation (EM Diffusion)")

    n_samples = st.slider("Monte Carlo samples", 20, 500, 100)

    grid = PDEGrid(0.0, 0.1, nx)

    def param_sampler():
        return {
            "mu": np.random.normal(4e-7 * np.pi, 1e-8),
            "sigma": np.random.normal(5.8e7, 5e6)
        }

    def model_factory(p):
        return EMDiffusionModel(
            grid=grid,
            params=p,
            dt=dt,
            bc_left=1.0,
            bc_right=0.0
        )

    def observable(model):
        B0 = np.zeros(nx)
        _, B_hist = model.run(B0, t_final)
        B = B_hist[-1]
        return np.trapz(B**2, grid.x)

    results = monte_carlo_propagation(
        model_factory,
        param_sampler,
        observable,
        n_samples=n_samples
    )

    export_csv(
        {
            "sample": results["samples"],
        },
        "em_uncertainty_samples.csv"
    )

    fig, ax = plt.subplots()
    ax.hist(results["samples"], bins=20)
    ax.set_xlabel("Magnetic Energy")
    ax.set_ylabel("Frequency")
    ax.set_title("Uncertainty in EM Energy")
    st.pyplot(fig)

    st.metric("Mean", f"{results['mean']:.3e}")
    st.metric("Std Dev", f"{results['std']:.3e}")
elif domain == "Solar Panel" and mode == "Simulation":
    from domains.solar.model import SolarPanelModel

    st.subheader("Solar Panel Energy Balance (0D)")

    C = st.sidebar.number_input("Thermal capacitance C", value=5000.0)
    eta0 = st.sidebar.number_input("Base efficiency η0", value=0.18)
    gamma = st.sidebar.number_input("Temp coefficient γ", value=0.004)
    h = st.sidebar.number_input("Heat loss coefficient h", value=15.0)
    Tamb = st.sidebar.number_input("Ambient temperature", value=25.0)
    S = st.sidebar.number_input("Solar irradiance S", value=900.0)

    params = {
        "C": C,
        "eta0": eta0,
        "gamma": gamma,
        "h": h,
        "Tamb": Tamb,
        "S": S
    }

    model = SolarPanelModel(params=params, dt=dt)

    T = Tamb
    times, temps, powers, etas = [], [], [], []

    for n in range(int(t_final / dt)):
        t = n * dt
        T = model.step(T, t)
        times.append(t)
        temps.append(T)
        powers.append(model.power_output(T, t))
        etas.append(model.efficiency(T))

    fig, ax = plt.subplots()
    ax.plot(times, temps, label="Temperature (°C)")
    ax.plot(times, powers, label="Power (arb.)")
    ax.plot(times, etas, label="Efficiency")
    ax.set_xlabel("Time")
    ax.legend()
    ax.set_title("Solar Panel Thermal–Electrical Response")
    st.pyplot(fig)

    st.info(
        f"Characteristic thermal time scale τ ≈ {model.stability_limit():.1f} s"
    )
elif domain == "Solar Panel" and mode == "Sensitivity":
    from domains.solar.model import SolarPanelModel

    st.subheader("Local Sensitivity Analysis (Solar Panel)")

    param_name = st.selectbox(
        "Parameter to perturb",
        ["C", "eta0", "gamma", "h"]
    )

    delta = st.number_input(
        "Perturbation Δ",
        value=0.01,
        format="%.3f"
    )

    base_params = {
        "C": 5000.0,
        "eta0": 0.18,
        "gamma": 0.004,
        "h": 15.0,
        "Tamb": 25.0,
        "S": 900.0
    }

    def model_factory(p):
        return SolarPanelModel(params=p, dt=dt)

    def observable(model):
        T = model.params["Tamb"]
        for n in range(int(t_final / dt)):
            T = model.step(T, n * dt)
        return model.power_output(T, t_final)

    sens = local_sensitivity(
        model_factory,
        base_params,
        param_name,
        delta,
        observable
    )

    export_csv(
        {
            "parameter": [param_name],
            "sensitivity": [sens],
        },
        "solar_sensitivity.csv"
    )

    st.metric(
        label=f"Sensitivity of final power to {param_name}",
        value=f"{sens:.4e}"
    )

    st.caption(
        "Computed via finite-difference local sensitivity. "
        "Valid only for small perturbations."
    )
elif domain == "Solar Panel" and mode == "Uncertainty":
    from domains.solar.model import SolarPanelModel

    st.subheader("Uncertainty Propagation (Solar Panel)")

    n_samples = st.slider("Monte Carlo samples", 20, 500, 100)

    def param_sampler():
        return {
            "C": 5000.0,
            "eta0": 0.18,
            "gamma": 0.004,
            "h": np.random.normal(15.0, 2.0),
            "Tamb": 25.0,
            "S": np.random.normal(900.0, 100.0)
        }

    def model_factory(p):
        return SolarPanelModel(params=p, dt=dt)

    def observable(model):
        T = model.params["Tamb"]
        for n in range(int(t_final / dt)):
            T = model.step(T, n * dt)
        return model.power_output(T, t_final)

    results = monte_carlo_propagation(
        model_factory,
        param_sampler,
        observable,
        n_samples=n_samples
    )

    export_csv(
        {
            "sample": results["samples"],
        },
        "solar_uncertainty_samples.csv"
    )

    fig, ax = plt.subplots()
    ax.hist(results["samples"], bins=20)
    ax.set_xlabel("Final Power Output")
    ax.set_ylabel("Frequency")
    ax.set_title("Uncertainty in Solar Power Output")
    st.pyplot(fig)

    st.metric("Mean", f"{results['mean']:.2f}")
    st.metric("Std Dev", f"{results['std']:.2f}")
