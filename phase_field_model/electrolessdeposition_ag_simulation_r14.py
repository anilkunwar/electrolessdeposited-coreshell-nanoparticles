#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
streamlit_electroless_enhanced_batch.py
--------------------------------------
Full app with:
 - growth-model dropdown (Model A / Model B)
 - concentration input modes (manual comma list OR slider range)
 - batch multi-concentration runs for selected dimension (2D or 3D)
 - publication-quality plotting for thickness-vs-time across concentrations
 - **NEW** heat-maps of concentration (c) for every saved frame
 - **NEW** extended diagnostics (final thickness, mean c, max c, total Ag)
 - preserves diagnostics, single-run playback, and other UI
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import io, zipfile, time, csv, os
from datetime import datetime
import tempfile
import math

# optional libs
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
try:
    import meshio
    MESHIO_AVAILABLE = True
except Exception:
    MESHIO_AVAILABLE = False
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ----------------------------------------------------------------------
# Page & style
# ----------------------------------------------------------------------
st.set_page_config(page_title="Electroless Ag — Enhanced Simulator (Batch)", layout="wide")
st.title("Electroless Ag — Enhanced Simulator — Batch & High-quality Plots")

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

CMAPS = [c for c in plt.colormaps() if c not in {"jet", "jet_r"}]

# ----------------------------------------------------------------------
# Sidebar – scales
# ----------------------------------------------------------------------
st.sidebar.header("Physical scales (change units)")
L0 = st.sidebar.number_input("Length scale L₀ (nm)", min_value=1.0, max_value=1e6, value=20.0, step=1.0)
E0 = st.sidebar.number_input("Energy scale E₀ (×10⁻¹⁴ J)", min_value=1e-6, max_value=1e6, value=1.0, step=0.1)
tau0 = st.sidebar.number_input("Time scale τ₀ (×10⁻⁴ s)", min_value=1e-6, max_value=1e6, value=1.0, step=0.1)
L0 = L0 * 1e-9          # nm → m
E0 = E0 * 1e-14         # ×10⁻¹⁴ J → J
tau0 = tau0 * 1e-4      # ×10⁻⁴ s → s

# ----------------------------------------------------------------------
# Concentration study mode
# ----------------------------------------------------------------------
st.sidebar.header("Concentration Study Mode")
study_mode = st.sidebar.selectbox(
    "Study Mode",
    ["Single concentration (1:1)", "Multiple concentrations (batch)"],
    help="Single run or batch runs across many concentrations"
)

st.sidebar.header("Concentration Input")
conc_input_mode = st.sidebar.selectbox(
    "Concentration input mode",
    ["Manual (comma-separated values)", "Slider range (start,end,steps)"]
)

default_manual = "1.0, 0.5, 0.333333, 0.25, 0.2"
if conc_input_mode.startswith("Manual"):
    manual_text = st.sidebar.text_area(
        "Enter concentrations (comma-separated)", value=default_manual,
        help="Example: 1.0,0.5,0.333,0.25,0.2"
    )
else:
    slider_start = st.sidebar.number_input("Start c_bulk", 0.05, 10.0, 1.0, 0.05)
    slider_end   = st.sidebar.number_input("End c_bulk",   0.05, 10.0, 0.2, 0.05)
    slider_steps = st.sidebar.slider("Number of steps", 2, 20, 5, 1)

# single-run fallback
if study_mode == "Single concentration (1:1)":
    c_bulk_nd = st.sidebar.slider("c_bulk ([Ag]/[Cu])", 0.1, 10.0, 1.0, 0.1)
else:
    c_bulk_nd = 1.0

# ----------------------------------------------------------------------
# Boundary conditions
# ----------------------------------------------------------------------
st.sidebar.header("Boundary Conditions")
bc_type = st.sidebar.selectbox(
    "Boundary condition type",
    ["Neumann (zero flux)", "Dirichlet (fixed values)"]
)

# ----------------------------------------------------------------------
# Simulation mode & grid
# ----------------------------------------------------------------------
st.sidebar.header("Simulation mode")
mode = st.sidebar.selectbox("Mode", ["2D (planar)", "3D (spherical)"])

st.sidebar.header("Grid & time")
if mode.startswith("2D"):
    Nx = st.sidebar.slider("Nx", 40, 400, 120, 10)
    Ny = st.sidebar.slider("Ny", 40, 400, 120, 10)
    Nz = 1
else:
    Nx = st.sidebar.slider("Nx", 16, 80, 40, 4)
    Ny = Nx
    Nz = st.sidebar.slider("Nz", 16, 80, 40, 4)

dt_nd = st.sidebar.number_input("dt (non-dim)", 1e-6, 2e-2, 2e-4, format="%.6f")
n_steps = st.sidebar.slider("n_steps", 50, 8000, 800, 50)
save_every = st.sidebar.slider("save every (frames)", 1, 400, max(1, n_steps//20), 1)

# ----------------------------------------------------------------------
# Physics (non-dimensional)
# ----------------------------------------------------------------------
st.sidebar.header("Physics params (non-dim)")
gamma_nd = st.sidebar.slider("γ (curvature)", 1e-4, 0.5, 0.02, 1e-4, format="%.4f")
beta_nd  = st.sidebar.slider("β (double-well)", 0.1, 20.0, 4.0, 0.1)
k0_nd    = st.sidebar.slider("k₀ (reaction)", 0.01, 2.0, 0.4, 0.01)
M_nd     = st.sidebar.slider("M (mobility)", 1e-3, 1.0, 0.2, 1e-3, format="%.3f")
alpha_nd = st.sidebar.slider("α (coupling)", 0.0, 10.0, 2.0, 0.1)
D_nd     = st.sidebar.slider("D (diffusion)", 0.0, 1.0, 0.05, 0.005)

st.sidebar.header("Solver & performance")
use_numba = st.sidebar.checkbox("Use numba (if available)", value=NUMBA_AVAILABLE)
use_semi_implicit = st.sidebar.checkbox(
    "Semi-implicit IMEX for Laplacian (requires scipy)", value=False
)
if use_semi_implicit and not SCIPY_AVAILABLE:
    st.sidebar.warning("SciPy not found — semi-implicit disabled.")
    use_semi_implicit = False

st.sidebar.header("Visualization")
cmap_choice = st.sidebar.selectbox("Matplotlib colormap", CMAPS,
                                   index=CMAPS.index("viridis"))
use_plotly = st.sidebar.checkbox("Use Plotly for interactive thickness plot (if available)", value=False)

# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
st.sidebar.header("Core & shell geometry")
core_radius_frac = st.sidebar.slider(
    "Core radius (fraction of L)", 0.05, 0.45, 0.18, 0.01
)
shell_thickness_frac = st.sidebar.slider(
    "Initial shell thickness (Δr / r_core)", 0.05, 0.6, 0.2, 0.01
)

# ----------------------------------------------------------------------
# Shell-thickness analysis
# ----------------------------------------------------------------------
st.sidebar.header("Shell Thickness Analysis")
phi_threshold = st.sidebar.slider(
    "Phi threshold for shell detection", 0.1, 0.9, 0.5, 0.05,
    help="phi > threshold → part of Ag shell"
)

# ----------------------------------------------------------------------
# Growth model
# ----------------------------------------------------------------------
st.sidebar.header("Growth model")
growth_model = st.sidebar.selectbox(
    "Choose growth model",
    ["Model A — Fully non-reversible (strictly additive)",
     "Model B — Soft reversible (0.01× bulk smoothing)"],
    index=0,
    help="Model A: no bulk dissolution, φ cannot decrease. Model B: small (1%) bulk smoothing allowed."
)

# ----------------------------------------------------------------------
# Run buttons
# ----------------------------------------------------------------------
run_single_button = st.sidebar.button("Run Single Simulation (current c_bulk)")
run_batch_button  = st.sidebar.button("Run Batch (multiple concentrations)")
export_vtu_button = st.sidebar.button("Export VTU/PVD/ZIP")
download_diags_button = st.sidebar.button("Download diagnostics CSV")

# ----------------------------------------------------------------------
# Scaling helpers
# ----------------------------------------------------------------------
def nd_to_real(length_nd): return length_nd * L0
def real_to_nd(length_m): return length_m / L0
def scale_time(t_nd): return t_nd * tau0
def scale_diffusion(D_nd): return D_nd * (L0**2 / tau0)
def scale_mobility(M_nd): return M_nd * (L0**3 / (E0 * tau0))
def scale_reaction(k0_nd): return k0_nd * (L0 / tau0)
def scale_energy_term(beta_nd): return beta_nd * (E0 / L0**3)
def scale_alpha(alpha_nd): return alpha_nd * (E0 / L0**3)

# ----------------------------------------------------------------------
# Shell-thickness computation
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
        thickness_m = nd_to_real(thickness_nd)
        return thickness_nd, thickness_m
    return 0.0, 0.0

# ----------------------------------------------------------------------
# Laplacian (Numba optional)
# ----------------------------------------------------------------------
if NUMBA_AVAILABLE and use_numba:
    @njit(parallel=True)
    def laplacian_explicit_2d(u, dx):
        nx, ny = u.shape
        out = np.zeros_like(u)
        for i in prange(1, nx-1):
            for j in prange(1, ny-1):
                out[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
        return out / (dx*dx)

    @njit(parallel=True)
    def laplacian_explicit_3d(u, dx):
        nx, ny, nz = u.shape
        out = np.zeros_like(u)
        for i in prange(1, nx-1):
            for j in prange(1, ny-1):
                for k in prange(1, nz-1):
                    out[i,j,k] = (u[i+1,j,k] + u[i-1,j,k] + u[i,j+1,k] + u[i,j-1,k] +
                                  u[i,j,k+1] + u[i,j,k-1] - 6*u[i,j,k])
        return out / (dx*dx)
else:
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
    ux[:,0] = (u[:,1] - u[:,0]) / dx; ux[:,-1] = (u[:,-1] - u[:,-2]) / dx
    uy[1:-1,:] = (u[2:,:] - u[:-2,:]) / (2*dx)
    uy[0,:] = (u[1,:] - u[:,0]) / dx; uy[-1,:] = (u[-1,:] - u[-2,:]) / dx
    return np.sqrt(ux**2 + uy**2 + 1e-30)

# ----------------------------------------------------------------------
# Core simulation (2D)
# ----------------------------------------------------------------------
def run_simulation_2d(params):
    Nx, Ny = params['Nx'], params['Ny']
    L = 1.0; dx = L/(Nx-1)
    x = np.linspace(0, L, Nx); y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    cx = cy = 0.5
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)

    psi = (dist <= params['core_radius_frac']*L).astype(np.float64)
    r_core = params['core_radius_frac']*L
    r_outer = r_core*(1.0 + params['shell_thickness_frac'])
    phi = np.where(dist <= r_core, 0.0,
                   np.where(dist <= r_outer, 1.0, 0.0)).astype(np.float64)

    eps = max(4*dx, 1e-6)
    phi = phi * (1.0 - 0.5*(1.0 - np.tanh((dist-r_core)/eps))) \
              * (1.0 - 0.5*(1.0 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0.0, 1.0)

    if params['bc_type'] == "Dirichlet (fixed values)":
        c = params['c_bulk'] * np.ones_like(X) * (1.0 - phi) * (1.0 - psi)
    else:
        c = params['c_bulk'] * (Y/L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots, diagnostics, thickness_data = [], [], []
    n_steps = params['n_steps']; dt = params['dt']; save_every = params['save_every']
    gamma = params['gamma']; beta = params['beta']; k0 = params['k0']
    M = params['M']; D = params['D']; alpha = params['alpha']
    max_thickness_nd_so_far = 0.0

    # semi-implicit matrix (optional)
    if params['use_semi_implicit'] and SCIPY_AVAILABLE:
        N = Nx*Ny
        A = sp.lil_matrix((N,N))
        for i in range(Nx):
            for j in range(Ny):
                idx = i*Ny + j
                A[idx, idx] = -4.0
                for ii,jj in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                    if 0 <= ii < Nx and 0 <= jj < Ny:
                        A[idx, ii*Ny + jj] = 1.0
                    else:
                        A[idx, idx] += 1.0
        A = A.tocsr()
        Implicit_mat = sp.eye(N) - (dt*M*gamma)*A
        lu = spla.factorized(Implicit_mat.tocsc())
        has_factor = True
    else:
        has_factor = False

    softness = 0.0 if params['growth_model'].startswith("Model A") else 0.01

    for step in range(n_steps+1):
        t = step*dt
        gphi = grad_mag_2d(phi, dx)
        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps,dx))

        # phi BCs
        if params['bc_type'] == "Dirichlet (fixed values)":
            phi[0,:] = phi[-1,:] = phi[:,0] = phi[:,-1] = 0.0
        else:
            phi[0,:] = phi[1,:]; phi[-1,:] = phi[-2,:]
            phi[:,0] = phi[:,1]; phi[:,-1] = phi[:,-2]

        lap_phi = laplacian_explicit_2d(phi, dx)
        f_bulk = 2.0*beta*phi*(1.0-phi)*(1.0-2.0*phi)
        c_mol = c*(1.0-phi)*(1.0-psi)
        i_loc = k0*c_mol*delta_int
        i_loc = np.clip(i_loc, 0.0, 1e6)

        deposition = M * i_loc
        curvature = M * gamma * lap_phi
        curvature_pos = np.maximum(curvature, 0.0)

        delta_phi = dt * (deposition + curvature_pos - softness * M * f_bulk)
        if softness == 0.0:
            delta_phi = np.maximum(delta_phi, 0.0)

        if has_factor:
            phi_temp = phi + delta_phi
            phi = lu(phi_temp.ravel()).reshape(phi.shape)
        else:
            phi = phi + delta_phi
        phi = np.clip(phi, 0.0, 1.0)

        # concentration update
        lap_c = laplacian_explicit_2d(c, dx)
        sink = i_loc
        c += dt*(D*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk'])

        # BCs for c
        if params['bc_type'] == "Dirichlet (fixed values)":
            c[0,:] = c[-1,:] = c[:,0] = c[:,-1] = params['c_bulk']
        else:
            c[:, -1] = params['c_bulk']

        # diagnostics
        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm_raw = np.sqrt(np.mean((M*gamma*lap_phi)**2))
        grad_norm_phys = np.sqrt(np.mean(((M*gamma*lap_phi)*(dx*dx))**2))
        alpha_c_norm = alpha*np.mean(c)

        if step % save_every == 0 or step == n_steps:
            thickness_nd, thickness_m = compute_shell_thickness(
                phi, psi, (x, y), params['core_radius_frac'], params['phi_threshold'], "2D"
            )
            max_thickness_nd_so_far = max(max_thickness_nd_so_far, thickness_nd)

            # **NEW** concentration heat-map data
            c_mean = np.mean(c)
            c_max  = np.max(c)
            total_ag = np.sum(i_loc) * dt  # integrated deposition

            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((
                t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm,
                c_mean, c_max, total_ag
            ))
            thickness_data.append((
                t, max_thickness_nd_so_far, nd_to_real(max_thickness_nd_so_far),
                c_mean, c_max, total_ag
            ))

    return snapshots, diagnostics, thickness_data, (x, y)

# ----------------------------------------------------------------------
# Core simulation (3D)
# ----------------------------------------------------------------------
def run_simulation_3d(params):
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    L = 1.0; dx = L/(Nx-1)
    x = np.linspace(0, L, Nx); y = np.linspace(0, L, Ny); z = np.linspace(0, L, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cx = cy = cz = 0.5
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)

    psi = (dist <= params['core_radius_frac']*L).astype(np.float64)
    r_core = params['core_radius_frac']*L
    r_outer = r_core*(1.0 + params['shell_thickness_frac'])
    phi = np.where(dist <= r_core, 0.0,
                   np.where(dist <= r_outer, 1.0, 0.0)).astype(np.float64)

    eps = max(4*dx, 1e-6)
    phi = phi * (1.0 - 0.5*(1.0 - np.tanh((dist-r_core)/eps))) \
              * (1.0 - 0.5*(1.0 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0.0, 1.0)

    if params['bc_type'] == "Dirichlet (fixed values)":
        c = params['c_bulk'] * np.ones_like(X) * (1.0 - phi) * (1.0 - psi)
    else:
        c = params['c_bulk'] * (Z/L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots, diagnostics, thickness_data = [], [], []
    n_steps = params['n_steps']; dt = params['dt']; save_every = params['save_every']
    gamma = params['gamma']; beta = params['beta']; k0 = params['k0']
    M = params['M']; D = params['D']; alpha = params['alpha']
    max_thickness_nd_so_far = 0.0
    softness = 0.0 if params['growth_model'].startswith("Model A") else 0.01

    for step in range(n_steps+1):
        t = step*dt
        lap_phi = laplacian_explicit_3d(phi, dx)
        gx, gy, gz = np.gradient(phi, dx, edge_order=2)
        gphi = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-30)
        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps,dx))

        if params['bc_type'] == "Dirichlet (fixed values)":
            phi[[0,-1],:,:] = phi[:,[0,-1],:] = phi[:,:,[0,-1]] = 0.0
        else:
            phi[0,:,:] = phi[1,:,:]; phi[-1,:,:] = phi[-2,:,:]
            phi[:,0,:] = phi[:,1,:]; phi[:,-1,:] = phi[:,-2,:]
            phi[:,:,0] = phi[:,:,1]; phi[:,:,-1] = phi[:,:,-2]

        f_bulk = 2.0*beta*phi*(1.0-phi)*(1.0-2.0*phi)
        c_mol = c*(1.0-phi)*(1.0-psi)
        i_loc = k0*c_mol*delta_int
        i_loc = np.clip(i_loc, 0.0, 1e6)

        deposition = M*i_loc
        curvature = M*gamma*lap_phi
        curvature_pos = np.maximum(curvature, 0.0)

        delta_phi = dt*(deposition + curvature_pos - softness * M * f_bulk)
        if softness == 0.0:
            delta_phi = np.maximum(delta_phi, 0.0)

        phi += delta_phi
        phi = np.clip(phi, 0.0, 1.0)

        lap_c = laplacian_explicit_3d(c, dx)
        sink = i_loc
        c += dt*(D*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk'])

        if params['bc_type'] == "Dirichlet (fixed values)":
            c[[0,-1],:,:] = c[:,[0,-1],:] = c[:,:,[0,-1]] = params['c_bulk']
        else:
            c[:, :, -1] = params['c_bulk']

        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm_raw = np.sqrt(np.mean((M*gamma*lap_phi)**2))
        grad_norm_phys = np.sqrt(np.mean(((M*gamma*lap_phi)*(dx*dx))**2))
        alpha_c_norm = alpha*np.mean(c)

        if step % save_every == 0 or step == n_steps:
            thickness_nd, thickness_m = compute_shell_thickness(
                phi, psi, (x, y, z), params['core_radius_frac'], params['phi_threshold'], "3D"
            )
            max_thickness_nd_so_far = max(max_thickness_nd_so_far, thickness_nd)

            c_mean = np.mean(c)
            c_max  = np.max(c)
            total_ag = np.sum(i_loc) * dt

            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((
                t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm,
                c_mean, c_max, total_ag
            ))
            thickness_data.append((
                t, max_thickness_nd_so_far, nd_to_real(max_thickness_nd_so_far),
                c_mean, c_max, total_ag
            ))

    return snapshots, diagnostics, thickness_data, (x, y, z)

# ----------------------------------------------------------------------
# Multi-concentration runner
# ----------------------------------------------------------------------
def run_multiple_simulations(params_base, concentrations):
    results = {}
    prog = st.progress(0)
    status = st.empty()
    total = len(concentrations)

    for i, conc in enumerate(concentrations):
        status.text(f"Running c_bulk = {conc:.6g} ({i+1}/{total}) …")
        p = params_base.copy()
        p['c_bulk'] = conc
        p['growth_model'] = params_base['growth_model']

        if p['mode'].startswith("2D"):
            snapshots, diags, thick, coords = run_simulation_2d(p)
        else:
            snapshots, diags, thick, coords = run_simulation_3d(p)

        results[conc] = {
            'snapshots': snapshots,
            'diagnostics': diags,
            'thickness': thick,
            'coords': coords
        }
        prog.progress((i+1)/total)

    status.text("All batch simulations finished.")
    return results

# ----------------------------------------------------------------------
# Base parameters
# ----------------------------------------------------------------------
params_base = {
    'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
    'dt': dt_nd, 'n_steps': n_steps, 'save_every': save_every,
    'gamma': gamma_nd, 'beta': beta_nd, 'k0': k0_nd,
    'M': M_nd, 'alpha': alpha_nd, 'c_bulk': c_bulk_nd, 'D': D_nd,
    'core_radius_frac': core_radius_frac,
    'shell_thickness_frac': shell_thickness_frac,
    'use_semi_implicit': use_semi_implicit,
    'use_numba': use_numba,
    'bc_type': bc_type,
    'phi_threshold': phi_threshold,
    'mode': mode,
    'growth_model': growth_model
}

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
for key in ["snapshots","diagnostics","thickness_data","grid_coords","multiple_results","last_batch_params"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ----------------------------------------------------------------------
# Parse concentrations
# ----------------------------------------------------------------------
def get_concentrations_from_ui():
    if conc_input_mode.startswith("Manual"):
        txt = manual_text.strip()
        if txt == "":
            return [1.0]
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except Exception:
                continue
        return vals if vals else [1.0]
    else:
        start = slider_start
        end = slider_end
        steps = slider_steps
        if steps <= 1:
            return [start]
        if start == end:
            return [start]
        return list(np.linspace(start, end, steps))

# ----------------------------------------------------------------------
# Run single simulation
# ----------------------------------------------------------------------
if run_single_button:
    t0 = time.time()
    st.info("Running single simulation …")
    params_base['growth_model'] = growth_model
    params_base['c_bulk'] = c_bulk_nd
    if mode.startswith("2D"):
        snapshots, diagnostics, thickness_data, coords = run_simulation_2d(params_base)
    else:
        snapshots, diagnostics, thickness_data, coords = run_simulation_3d(params_base)

    st.session_state.snapshots = snapshots
    st.session_state.diagnostics = diagnostics
    st.session_state.thickness_data = thickness_data
    st.session_state.grid_coords = coords
    st.session_state.multiple_results = None
    st.success(f"Single simulation done in {time.time()-t0:.2f}s — {len(snapshots)} frames")

# ----------------------------------------------------------------------
# Run batch
# ----------------------------------------------------------------------
if run_batch_button:
    concentrations = get_concentrations_from_ui()
    concentrations = sorted(list({float(round(c, 12)) for c in concentrations}), reverse=True)
    if len(concentrations) == 0:
        st.error("No valid concentrations provided.")
    else:
        t0 = time.time()
        st.info(f"Running batch for {len(concentrations)} concentrations …")
        params_base['growth_model'] = growth_model
        params_base['mode'] = mode
        multiple_results = run_multiple_simulations(params_base, concentrations)
        st.session_state.multiple_results = multiple_results
        st.session_state.last_batch_params = {
            'concentrations': concentrations,
            'growth_model': growth_model,
            'mode': mode,
            'params': params_base.copy()
        }
        st.success(f"Batch finished in {time.time()-t0:.2f}s")

# ----------------------------------------------------------------------
# Show batch results
# ----------------------------------------------------------------------
if st.session_state.multiple_results:
    st.header("Batch Results — Thickness vs Time (all concentrations)")
    multiple_results = st.session_state.multiple_results
    concentrations = list(multiple_results.keys())
    concentrations_sorted = sorted(concentrations, reverse=True)
    n = len(concentrations_sorted)
    cmap = plt.get_cmap(cmap_choice)
    colors = cmap(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(10, 6))
    final_nm = []
    for i, conc in enumerate(concentrations_sorted):
        data = multiple_results[conc]['thickness']
        if not data:
            continue
        ts = [scale_time(t) for t, _, _, _, _, _ in data]
        th_nm = [th*1e9 for _, _, th, _, _, _ in data]
        label = f"c={conc:.4g}"
        ax.plot(ts, th_nm, label=label, color=colors[i], lw=2, marker='o', markersize=4)
        final_nm.append(th_nm[-1])
    ax.set_xlabel("Time (s)", labelpad=10)
    ax.set_ylabel("Shell Thickness (nm)", labelpad=10)
    ax.set_title(f"Shell Thickness vs Time — {mode} — {growth_model}", pad=14)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Concentrations", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)

    if use_plotly and PLOTLY_AVAILABLE:
        df_lines = []
        for conc in concentrations_sorted:
            data = multiple_results[conc]['thickness']
            for t, ndth, real_th, cmean, cmax, ag in data:
                df_lines.append({"c_bulk": conc, "t_s": scale_time(t), "thickness_nm": real_th*1e9,
                                 "c_mean": cmean, "c_max": cmax, "total_Ag": ag})
        df_plot = pd.DataFrame(df_lines)
        if not df_plot.empty:
            figp = px.line(df_plot, x="t_s", y="thickness_nm", color="c_bulk",
                           labels={"t_s": "Time (s)", "thickness_nm": "Thickness (nm)"},
                           title=f"Interactive Thickness vs Time — {mode}")
            st.plotly_chart(figp, use_container_width=True)

    # Extended final summary
    summary_rows = []
    for conc in concentrations_sorted:
        data = multiple_results[conc]['thickness']
        if not data:
            continue
        final = data[-1]
        final_nd = final[1]
        final_nm = final[2]*1e9
        c_mean   = final[3]
        c_max    = final[4]
        total_ag = final[5]
        summary_rows.append({
            "c_bulk": conc,
            "Thickness_nd": final_nd,
            "Thickness_nm": final_nm,
            "c_mean": c_mean,
            "c_max": c_max,
            "Total_Ag": total_ag
        })
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).sort_values("c_bulk", ascending=False)
        st.subheader("Final Diagnostics Summary")
        st.dataframe(df_summary.style.format({
            "c_bulk": "{:.6g}", "Thickness_nm": "{:.2f}", "Thickness_nd": "{:.4f}",
            "c_mean": "{:.4f}", "c_max": "{:.4f}", "Total_Ag": "{:.3e}"
        }))
        csv_bytes = df_summary.to_csv(index=False).encode()
        st.download_button(
            "Download final diagnostics (CSV)",
            csv_bytes,
            file_name=f"batch_diagnostics_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )

# ----------------------------------------------------------------------
# Single-run playback & post-processing
# ----------------------------------------------------------------------
if st.session_state.snapshots and st.session_state.thickness_data:
    snapshots = st.session_state.snapshots
    diagnostics = st.session_state.diagnostics
    thickness_data = st.session_state.thickness_data
    coords = st.session_state.grid_coords
    st.header("Single Run — Results & Playback")

    cols = st.columns([3,1])
    with cols[0]:
        frame_idx = st.slider("Frame", 0, len(snapshots)-1, len(snapshots)-1)
        auto_play = st.checkbox("Autoplay", value=False)
        autoplay_interval = st.number_input("Interval (s)", 0.1, 5.0, 0.4, 0.1)
        field = st.selectbox("Field", ["phi (shell)", "c (concentration)", "psi (core)"])

        t_nd, phi_view, c_view, psi_view = snapshots[frame_idx]
        t_real = scale_time(t_nd)
        th_nd, th_m, c_mean, c_max, total_ag = thickness_data[frame_idx][1:6]

        st.write(f"**Current shell thickness:** {th_nd:.4f} (non-dim) = {th_m*1e9:.2f} nm")
        st.write(f"**Mean c:** {c_mean:.4f} | **Max c:** {c_max:.4f} | **Total Ag:** {total_ag:.3e}")
        st.write(f"**Concentration (c_bulk):** {params_base['c_bulk']}")
        st.write(f"**Growth model:** {params_base['growth_model']}")

        cmap = plt.get_cmap(cmap_choice)

        if mode.startswith("2D"):
            fig, ax = plt.subplots(figsize=(6,5))
            if field == "phi (shell)":
                im = ax.imshow(phi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap)
            elif field == "c (concentration)":
                im = ax.imshow(c_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap, vmin=0, vmax=params_base['c_bulk'])
            else:
                im = ax.imshow(psi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap)
            cbar = plt.colorbar(im, ax=ax, pad=0.01)
            cbar.ax.tick_params(labelsize=10)
            ax.set_title(f"{field} @ t = {t_real:.3e} s")
            st.pyplot(fig)

            # **NEW** concentration heat-map
            if field == "c (concentration)":
                st.subheader("Concentration Heat-map (2D)")
                fig_c, ax_c = plt.subplots(figsize=(6,5))
                im_c = ax_c.imshow(c_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap, vmin=0, vmax=params_base['c_bulk'])
                cbar_c = plt.colorbar(im_c, ax=ax_c)
                ax_c.set_title(f"Concentration (c) @ t = {t_real:.3e} s")
                st.pyplot(fig_c)

            # line profile
            mid = phi_view.shape[0]//2
            fig2, ax2 = plt.subplots(figsize=(6,2.2))
            if field == "phi (shell)":
                ax2.plot(np.linspace(0,1,phi_view.shape[1]), phi_view[mid,:], label='phi', lw=2)
            elif field == "c (concentration)":
                ax2.plot(np.linspace(0,1,c_view.shape[1]), c_view[mid,:], label='c', lw=2)
            else:
                ax2.plot(np.linspace(0,1,psi_view.shape[1]), psi_view[mid,:], label='psi', lw=2)
            ax2.set_xlabel("y/L"); ax2.legend(); ax2.grid(True)
            st.pyplot(fig2)

        else:
            # 3D slices
            fig, axes = plt.subplots(1,3, figsize=(12,4))
            cx = phi_view.shape[0]//2; cy = phi_view.shape[1]//2; cz = phi_view.shape[2]//2
            for ax, sl, title in zip(axes,
                                     [phi_view[cx,:,:], phi_view[:,cy,:], phi_view[:,:,cz]],
                                     ["x-slice","y-slice","z-slice"]):
                ax.imshow(sl.T, origin='lower', cmap=cmap); ax.set_title(title); ax.axis('off')
            fig.suptitle(f"{field} @ t = {t_real:.3e} s")
            st.pyplot(fig)

            if field == "c (concentration)":
                st.subheader("Concentration Heat-maps (3D slices)")
                fig_c, axes_c = plt.subplots(1,3, figsize=(12,4))
                for ax, sl, title in zip(axes_c,
                                         [c_view[cx,:,:], c_view[:,cy,:], c_view[:,:,cz]],
                                         ["x-slice","y-slice","z-slice"]):
                    im = ax.imshow(sl.T, origin='lower', cmap=cmap, vmin=0, vmax=params_base['c_bulk'])
                    ax.set_title(title); ax.axis('off')
                fig_c.colorbar(im, ax=axes_c, orientation='horizontal', pad=0.05)
                fig_c.suptitle(f"Concentration (c) @ t = {t_real:.3e} s")
                st.pyplot(fig_c)

        if auto_play:
            for i in range(frame_idx, len(snapshots)):
                time.sleep(autoplay_interval)
                st.session_state._rerun = True

    with cols[1]:
        st.subheader("Diagnostics")
        df = pd.DataFrame(diagnostics,
                          columns=["t*","||bulk||₂","||grad||₂ raw","||grad||₂ scaled","α·mean(c)",
                                   "c_mean","c_max","total_Ag"])
        st.dataframe(df.tail(20).style.format("{:.3e}"))

        fig3, ax3 = plt.subplots(figsize=(4,3))
        ax3.semilogy(df["t*"], np.maximum(df["||bulk||₂"],1e-30), label='bulk')
        ax3.semilogy(df["t*"], np.maximum(df["||grad||₂ raw"],1e-30), label='grad raw')
        ax3.semilogy(df["t*"], np.maximum(df["||grad||₂ scaled"],1e-30), label='grad scaled')
        ax3.semilogy(df["t*"], np.maximum(df["α·mean(c)"],1e-30), label='α·c')
        ax3.semilogy(df["t*"], np.maximum(df["c_mean"],1e-30), label='c_mean')
        ax3.semilogy(df["t*"], np.maximum(df["c_max"],1e-30), label='c_max')
        ax3.legend(fontsize=8); ax3.grid(True)
        st.pyplot(fig3)

        st.subheader("Shell Thickness Evolution")
        times = [scale_time(t) for t,_,_,_,_,_ in thickness_data]
        thick_nm = [th*1e9 for _,_,th,_,_,_ in thickness_data]
        fig_th, ax_th = plt.subplots(figsize=(4,3))
        ax_th.plot(times, thick_nm, 'b-', lw=2)
        ax_th.set_xlabel('Time (s)'); ax_th.set_ylabel('Thickness (nm)')
        ax_th.grid(True, alpha=0.3); ax_th.set_title('Growth curve')
        st.pyplot(fig_th)

else:
    st.info("Run a simulation (single or batch) to see results. Tip: keep 3D grid ≤ 40 for fast runs.")
