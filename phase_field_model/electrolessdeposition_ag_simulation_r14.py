#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
streamlit_electroless_final.py
Final integrated Streamlit app that:
 - supports single‑run and batch‑run (multiple concentrations)
 - two growth models (fully non‑reversible, soft reversible)
 - default total time (n_steps) = 500 (adjustable)
 - batch final‑thickness histogram
 - auto‑save CSVs to ./results/
 - retains diagnostics, 2D/3D modes, and high‑quality plotting
 - **NEW**: “Run Single Concentration” now runs the *full* list of concentrations
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, time, io
from datetime import datetime

# ----------------------------------------------------------------------
# Optional libraries detection
# ----------------------------------------------------------------------
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Electroless Ag — Final Simulator", layout="wide")
st.title("Electroless Ag — Final Simulator")

# ----------------------------------------------------------------------
# Plotting style
# ----------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 140,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
})

# ----------------------------------------------------------------------
# Results directory
# ----------------------------------------------------------------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Sidebar: Controls
# ----------------------------------------------------------------------
st.sidebar.header("Simulation controls")

# Growth model
growth_model = st.sidebar.selectbox(
    "Growth model",
    ["Model A — Fully non-reversible (strictly additive)",
     "Model B — Soft reversible (0.01× bulk smoothing)"]
)

# Dimension selection
mode = st.sidebar.selectbox("Mode", ["2D (planar)", "3D (spherical)"])

# Physics parameters
st.sidebar.subheader("Physics params (non‑dim)")
gamma = st.sidebar.slider("γ (curvature)", 1e-4, 0.5, 0.02, 1e-4, format="%.4f")
beta = st.sidebar.slider("β (double‑well)", 0.1, 20.0, 4.0, 0.1)
k0 = st.sidebar.slider("k₀ (reaction)", 0.01, 2.0, 0.4, 0.01)
M = st.sidebar.slider("M (mobility)", 1e-3, 1.0, 0.2, 1e-3, format="%.3f")
D = st.sidebar.slider("D (diffusion)", 0.0, 1.0, 0.05, 0.005)

# Grid & time
st.sidebar.subheader("Grid & time")
if mode.startswith("2D"):
    Nx = st.sidebar.slider("Nx", 40, 400, 120, 10)
    Ny = st.sidebar.slider("Ny", 40, 400, 120, 10)
    Nz = 1
else:
    Nx = st.sidebar.slider("Nx", 16, 80, 40, 4)
    Ny = Nx
    Nz = st.sidebar.slider("Nz", 16, 80, 40, 4)

# Default total time = 500
n_steps = st.sidebar.number_input("n_steps (total time steps)", min_value=10, max_value=20000, value=500, step=10)
dt = st.sidebar.number_input("dt (non‑dim)", 1e-6, 2e-2, 2e-4, format="%.6f")
save_every = st.sidebar.slider("save every (frames)", 1, 400, max(1, n_steps//20), 1)

# Geometry & detection
st.sidebar.subheader("Geometry & detection")
core_radius_frac = st.sidebar.slider("Core radius (fraction of L)", 0.05, 0.45, 0.18, 0.01)
shell_thickness_frac = st.sidebar.slider("Initial shell thickness (Δr/r_core)", 0.01, 0.6, 0.2, 0.01)
phi_threshold = st.sidebar.slider("phi threshold for shell detection", 0.1, 0.9, 0.5, 0.05)

# ----------------------------------------------------------------------
# Concentration inputs
# ----------------------------------------------------------------------
st.sidebar.subheader("Concentration input")
conc_mode = st.sidebar.selectbox("Concentration input mode", ["Manual (comma list)", "Slider range"])

if conc_mode.startswith("Manual"):
    manual_text = st.sidebar.text_area(
        "Enter concentrations (comma‑separated)",
        value="1.0, 0.5, 0.333333, 0.25, 0.2"
    )
else:
    slider_start = st.sidebar.number_input("Start c_bulk", 0.05, 10.0, 1.0, 0.05)
    slider_end   = st.sidebar.number_input("End c_bulk",   0.05, 10.0, 0.2, 0.05)
    slider_steps = st.sidebar.slider("Steps", 2, 20, 5)

# ----------------------------------------------------------------------
# Single‑concentration quick‑test slider (still useful for a single value)
# ----------------------------------------------------------------------
st.sidebar.subheader("Single concentration (quick test)")
single_c = st.sidebar.number_input("Single c_bulk", 0.05, 10.0, 1.0, 0.05)

# ----------------------------------------------------------------------
# Buttons
# ----------------------------------------------------------------------
run_single = st.sidebar.button("Run Single Concentration (ALL defined concentrations)")
run_batch  = st.sidebar.button("Run Batch Concentrations")

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def nd_to_real(length_nd, L0_nm=20.0e-9):
    """Convert non‑dimensional length to metres (kept for compatibility)."""
    return length_nd * L0_nm

def parse_concentrations():
    """Return the *full* list of concentrations exactly as the batch mode uses."""
    if conc_mode.startswith("Manual"):
        s = manual_text.strip()
        if not s:
            return [1.0]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except Exception:
                pass
        return vals if vals else [1.0]
    else:
        if slider_steps <= 1:
            return [slider_start]
        if abs(slider_end - slider_start) < 1e-12:
            return [slider_start]
        return list(np.linspace(slider_start, slider_end, slider_steps))

# ----------------------------------------------------------------------
# Numerical utilities
# ----------------------------------------------------------------------
def laplacian_explicit_2d(u, dx):
    out = np.zeros_like(u)
    out[1:-1,1:-1] = (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4*u[1:-1,1:-1])
    return out / (dx*dx)

def laplacian_explicit_3d(u, dx):
    out = np.zeros_like(u)
    out[1:-1,1:-1,1:-1] = (u[2:,1:-1,1:-1] + u[:-2,1:-1,1:-1] +
                           u[1:-1,2:,1:-1] + u[1:-1,:-2,1:-1] +
                           u[1:-1,1:-1,2:] + u[1:-1,1:-1,:-2] - 6*u[1:-1,1:-1,1:-1])
    return out / (dx*dx)

def grad_mag_2d(u, dx):
    ux = np.zeros_like(u); uy = np.zeros_like(u)
    ux[:,1:-1] = (u[:,2:] - u[:,:-2]) / (2*dx)
    ux[:,0]    = (u[:,1]  - u[:,0])   / dx
    ux[:,-1]   = (u[:,-1] - u[:,-2]) / dx
    uy[1:-1,:] = (u[2:,:] - u[:-2,:]) / (2*dx)
    uy[0,:]    = (u[1,:] - u[:,0])   / dx
    uy[-1,:]   = (u[-1,:] - u[-2,:]) / dx
    return np.sqrt(ux**2 + uy**2 + 1e-30)

# ----------------------------------------------------------------------
# Shell‑thickness computation
# ----------------------------------------------------------------------
def compute_shell_thickness(phi, psi, coords, core_radius_frac_local, threshold=0.5, mode="2D"):
    if mode.startswith("2D"):
        x, y = coords
        cx = cy = 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        distances = np.sqrt((X-cx)**2 + (Y-cy)**2)
    else:
        x, y, z = coords
        cx = cy = cz = 0.5
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        distances = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)

    shell_mask = (phi > threshold) & (psi < 0.5)
    if np.any(shell_mask):
        max_shell_radius = np.max(distances[shell_mask])
        core_radius = core_radius_frac_local
        thickness_nd = max(0.0, max_shell_radius - core_radius)
        thickness_m  = nd_to_real(thickness_nd)
        return thickness_nd, thickness_m
    return 0.0, 0.0

# ----------------------------------------------------------------------
# Core simulation functions (2D / 3D)
# ----------------------------------------------------------------------
def run_simulation_2d_local(params):
    Nx, Ny = params['Nx'], params['Ny']
    L = 1.0; dx = L/(Nx-1)
    x = np.linspace(0, L, Nx); y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2)

    psi = (dist <= params['core_radius_frac']*L).astype(np.float64)
    r_core = params['core_radius_frac']*L
    r_outer = r_core*(1.0 + params['shell_thickness_frac'])
    phi = np.where(dist <= r_core, 0.0,
                   np.where(dist <= r_outer, 1.0, 0.0)).astype(np.float64)

    eps = max(4*dx, 1e-6)
    phi = phi * (1.0 - 0.5*(1.0 - np.tanh((dist-r_core)/eps))) \
              * (1.0 - 0.5*(1.0 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0.0, 1.0)

    # ---- initial concentration ----
    if params['bc_type'].startswith("Dirichlet"):
        c = params['c_bulk'] * np.ones_like(X) * (1.0 - phi) * (1.0 - psi)
    else:
        c = params['c_bulk'] * (Y/L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots, diagnostics, thickness_data = [], [], []
    max_thickness_nd_so_far = 0.0
    softness = 0.0 if params['growth_model'].startswith("Model A") else 0.01

    for step in range(params['n_steps'] + 1):
        t = step * params['dt']

        gphi = grad_mag_2d(phi, dx)
        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps,dx))

        # ---- phi BCs ----
        if params['bc_type'].startswith("Dirichlet"):
            phi[0,:] = phi[-1,:] = phi[:,0] = phi[:,-1] = 0.0
        else:
            phi[0,:] = phi[1,:]; phi[-1,:] = phi[-2,:]
            phi[:,0] = phi[:,1]; phi[:,-1] = phi[:,-2]

        lap_phi = laplacian_explicit_2d(phi, dx)
        f_bulk  = 2.0*params['beta']*phi*(1.0-phi)*(1.0-2.0*phi)
        c_mol   = c*(1.0-phi)*(1.0-psi)
        i_loc   = params['k0']*c_mol*delta_int
        i_loc   = np.clip(i_loc, 0.0, 1e6)

        deposition    = params['M'] * i_loc
        curvature     = params['M']*params['gamma']*lap_phi
        curvature_pos = np.maximum(curvature, 0.0)

        delta_phi = params['dt'] * (deposition + curvature_pos - softness * params['M'] * f_bulk)
        if softness == 0.0:
            delta_phi = np.maximum(delta_phi, 0.0)

        phi = phi + delta_phi
        phi = np.clip(phi, 0.0, 1.0)

        # ---- concentration update ----
        lap_c = laplacian_explicit_2d(c, dx)
        sink  = i_loc
        c += params['dt'] * (params['D']*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk'])

        # ---- save frame ----
        if step % params['save_every'] == 0 or step == params['n_steps']:
            thickness_nd, thickness_m = compute_shell_thickness(
                phi, psi, (x, y), params['core_radius_frac'], params['phi_threshold'], "2D")
            max_thickness_nd_so_far = max(max_thickness_nd_so_far, thickness_nd)

            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((
                t,
                np.sqrt(np.mean(f_bulk**2)),
                np.sqrt(np.mean((params['M']*params['gamma']*lap_phi)**2)),
                0.0,
                params['M']*np.mean(c)
            ))
            thickness_data.append((t, max_thickness_nd_so_far, nd_to_real(max_thickness_nd_so_far)))

    return snapshots, diagnostics, thickness_data, (x, y)


def run_simulation_3d_local(params):
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    L = 1.0; dx = L/(Nx-1)
    x = np.linspace(0, L, Nx); y = np.linspace(0, L, Ny); z = np.linspace(0, L, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)

    psi = (dist <= params['core_radius_frac']*L).astype(np.float64)
    r_core  = params['core_radius_frac']*L
    r_outer = r_core*(1.0 + params['shell_thickness_frac'])
    phi = np.where(dist <= r_core, 0.0,
                   np.where(dist <= r_outer, 1.0, 0.0)).astype(np.float64)

    eps = max(4*dx, 1e-6)
    phi = phi * (1.0 - 0.5*(1.0 - np.tanh((dist-r_core)/eps))) \
              * (1.0 - 0.5*(1.0 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0.0, 1.0)

    if params['bc_type'].startswith("Dirichlet"):
        c = params['c_bulk'] * np.ones_like(X) * (1.0 - phi) * (1.0 - psi)
    else:
        c = params['c_bulk'] * (Z/L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots, diagnostics, thickness_data = [], [], []
    max_thickness_nd_so_far = 0.0
    softness = 0.0 if params['growth_model'].startswith("Model A") else 0.01

    for step in range(params['n_steps'] + 1):
        t = step * params['dt']

        lap_phi = laplacian_explicit_3d(phi, dx)
        gx, gy, gz = np.gradient(phi, dx, edge_order=2)
        gphi = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-30)

        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps,dx))

        # ---- phi BCs ----
        if params['bc_type'].startswith("Dirichlet"):
            phi[[0,-1],:,:] = phi[:,[0,-1],:] = phi[:,:,[0,-1]] = 0.0
        else:
            phi[0,:,:] = phi[1,:,:]; phi[-1,:,:] = phi[-2,:,:]
            phi[:,0,:] = phi[:,1,:]; phi[:,-1,:] = phi[:,-2,:]
            phi[:,:,0] = phi[:,:,1]; phi[:,:,-1] = phi[:,:,-2]

        f_bulk = 2.0*params['beta']*phi*(1.0-phi)*(1.0-2.0*phi)
        c_mol  = c*(1.0-phi)*(1.0-psi)
        i_loc  = params['k0']*c_mol*delta_int
        i_loc  = np.clip(i_loc, 0.0, 1e6)

        deposition    = params['M']*i_loc
        curvature     = params['M']*params['gamma']*lap_phi
        curvature_pos = np.maximum(curvature, 0.0)

        delta_phi = params['dt']*(deposition + curvature_pos - softness * params['M'] * f_bulk)
        if softness == 0.0:
            delta_phi = np.maximum(delta_phi, 0.0)

        phi = phi + delta_phi
        phi = np.clip(phi, 0.0, 1.0)

        # ---- concentration ----
        lap_c = laplacian_explicit_3d(c, dx)
        sink  = i_loc
        c += params['dt']*(params['D']*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk'])

        # ---- save frame ----
        if step % params['save_every'] == 0 or step == params['n_steps']:
            thickness_nd, thickness_m = compute_shell_thickness(
                phi, psi, (x, y, z), params['core_radius_frac'], params['phi_threshold'], "3D")
            max_thickness_nd_so_far = max(max_thickness_nd_so_far, thickness_nd)

            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((
                t,
                np.sqrt(np.mean(f_bulk**2)),
                np.sqrt(np.mean((params['M']*params['gamma']*lap_phi)**2)),
                0.0,
                params['M']*np.mean(c)
            ))
            thickness_data.append((t, max_thickness_nd_so_far, nd_to_real(max_thickness_nd_so_far)))

    return snapshots, diagnostics, thickness_data, (x, y, z)

# ----------------------------------------------------------------------
# Parameter builder (common for single & batch)
# ----------------------------------------------------------------------
def build_base_params(c_val):
    return {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'dt': dt, 'n_steps': int(n_steps), 'save_every': save_every,
        'gamma': gamma, 'beta': beta, 'k0': k0, 'M': M, 'D': D,
        'alpha': 0.0, 'c_bulk': float(c_val),
        'core_radius_frac': core_radius_frac,
        'shell_thickness_frac': shell_thickness_frac,
        'use_semi_implicit': False, 'use_numba': False,
        'bc_type': "Dirichlet (fixed values)",
        'phi_threshold': phi_threshold,
        'growth_model': growth_model,
        'mode': mode
    }

# ----------------------------------------------------------------------
# Run a *single* concentration (used by both modes)
# ----------------------------------------------------------------------
def run_one_concentration(c_val):
    params = build_base_params(c_val)
    if mode.startswith("2D"):
        return run_simulation_2d_local(params)
    else:
        return run_simulation_3d_local(params)

# ----------------------------------------------------------------------
# Run *all* concentrations (used by both “single” and “batch” buttons)
# ----------------------------------------------------------------------
def run_all_concentrations(concentrations):
    results = {}
    prog = st.progress(0)
    status = st.empty()
    total = len(concentrations)

    for i, c_val in enumerate(concentrations):
        status.text(f"Running c = {c_val:.6g}  ({i+1}/{total}) …")
        snaps, diags, thick, coords = run_one_concentration(c_val)
        results[c_val] = {'snapshots': snaps,
                         'diagnostics': diags,
                         'thickness': thick,
                         'coords': coords}

        # ---- auto‑save CSV ----
        df = pd.DataFrame(
            [(t, nd, real) for (t, nd, real) in thick],
            columns=["t_nd", "thickness_nd", "thickness_m"]
        )
        df["thickness_nm"] = df["thickness_m"] * 1e9
        safe_c = f"{c_val:.6g}".replace(".", "_")
        fname = os.path.join(
            RESULTS_DIR,
            f"thickness_c{safe_c}_{mode.replace(' ','')}_{growth_model.split('—')[0].strip()}.csv"
        )
        df.to_csv(fname, index=False)

        prog.progress((i+1)/total)

    status.text("All simulations finished.")
    prog.progress(1.0)
    return results

# ----------------------------------------------------------------------
# UI: Single‑run section (now runs *all* concentrations)
# ----------------------------------------------------------------------
st.header("Single Concentration Run (runs **all** defined concentrations)")
st.write(
    "Press the button below – the app will execute the **exact same list** that the batch mode uses. "
    "All thickness curves, a final‑thickness histogram and per‑concentration CSVs will appear."
)

if run_single:
    concentrations = parse_concentrations()
    if not concentrations:
        st.error("No valid concentrations parsed.")
    else:
        t0 = time.time()
        st.info(f"Running **{len(concentrations)}** concentrations …")
        multiple_results = run_all_concentrations(concentrations)
        st.session_state['single_batch_results'] = multiple_results
        st.success(f"Single‑run batch finished in {time.time()-t0:.2f}s – CSVs saved to `{RESULTS_DIR}/`")

# ---- show results of the “single‑run batch” ----
if 'single_batch_results' in st.session_state and st.session_state['single_batch_results']:
    batch_res = st.session_state['single_batch_results']

    # thickness vs time for every concentration
    fig, ax = plt.subplots(figsize=(10,6))
    cmap = plt.get_cmap("viridis")
    keys = sorted(batch_res.keys(), reverse=True)
    colors = cmap(np.linspace(0,1,len(keys)))
    final_nm = []

    for i, k in enumerate(keys):
        thick = batch_res[k]['thickness']
        if not thick:
            continue
        times = [t for (t,_,_) in thick]
        th_nm = [m*1e9 for (_,_,m) in thick]
        ax.plot(times, th_nm, marker='o', markersize=4,
                label=f"c={k:.4g}", color=colors[i])
        final_nm.append(th_nm[-1])

    ax.set_xlabel("Time (non‑dim)")
    ax.set_ylabel("Thickness (nm)")
    ax.set_title(f"Thickness vs Time — {mode} — {growth_model}")
    ax.grid(True, alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', title="c_bulk")
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)

    # histogram of final thicknesses
    if final_nm:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.hist(final_nm, bins=min(12, max(3, len(final_nm))), alpha=0.9, edgecolor='k')
        ax2.set_xlabel("Final thickness (nm)")
        ax2.set_ylabel("Count")
        ax2.set_title("Histogram of final thickness (single‑run batch)")
        ax2.grid(True, alpha=0.2)
        plt.tight_layout(pad=2.0)
        st.pyplot(fig2)

    # download links for saved CSVs
    st.subheader("Saved CSVs (single‑run batch)")
    csv_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")])
    for f in csv_files:
        path = os.path.join(RESULTS_DIR, f)
        with open(path, "rb") as fh:
            st.download_button(f"Download {f}", fh, file_name=f, mime="text/csv")

# ----------------------------------------------------------------------
# UI: Batch‑run section (unchanged – kept for backward compatibility)
# ----------------------------------------------------------------------
st.header("Batch Concentration Run")
st.write(
    "Define a list (manual or slider) **and** press the button below. "
    "The same routine as the single‑run batch is executed, but results are stored in a separate session key."
)

concs = parse_concentrations()
st.write(f"Parsed concentrations ({len(concs)}): {concs}")

if run_batch:
    t0 = time.time()
    st.info(f"Running batch for {len(concs)} concentrations …")
    batch_results = run_all_concentrations(concs)
    st.session_state['batch_results'] = batch_results
    st.success(f"Batch finished in {time.time()-t0:.2f}s – CSVs saved to `{RESULTS_DIR}/`")

# ---- show batch results (identical code to the single‑run batch) ----
if 'batch_results' in st.session_state and st.session_state['batch_results']:
    batch_res = st.session_state['batch_results']

    fig, ax = plt.subplots(figsize=(10,6))
    cmap = plt.get_cmap("viridis")
    keys = sorted(batch_res.keys(), reverse=True)
    colors = cmap(np.linspace(0,1,len(keys)))
    final_nm = []

    for i, k in enumerate(keys):
        thick = batch_res[k]['thickness']
        if not thick:
            continue
        times = [t for (t,_,_) in thick]
        th_nm = [m*1e9 for (_,_,m) in thick]
        ax.plot(times, th_nm, marker='o', markersize=4,
                label=f"c={k:.4g}", color=colors[i])
        final_nm.append(th_nm[-1])

    ax.set_xlabel("Time (non‑dim)")
    ax.set_ylabel("Thickness (nm)")
    ax.set_title(f"Batch Thickness vs Time — {mode} — {growth_model}")
    ax.grid(True, alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', title="c_bulk")
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)

    if final_nm:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.hist(final_nm, bins=min(12, max(3, len(final_nm))), alpha=0.9, edgecolor='k')
        ax2.set_xlabel("Final thickness (nm)")
        ax2.set_ylabel("Count")
        ax2.set_title("Histogram of final thickness (batch)")
        ax2.grid(True, alpha=0.2)
        plt.tight_layout(pad=2.0)
        st.pyplot(fig2)

    st.subheader("Saved CSVs (batch)")
    csv_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")])
    for f in csv_files:
        path = os.path.join(RESULTS_DIR, f)
        with open(path, "rb") as fh:
            st.download_button(f"Download {f}", fh, file_name=f, mime="text/csv")

# ----------------------------------------------------------------------
# Footer info
# ----------------------------------------------------------------------
st.sidebar.header("Info")
st.sidebar.write("All CSV files are saved to the local `results/` folder.")
st.sidebar.write("Tip: keep 3‑D grids ≤ 40 for fast runs; use Numba if available.")
