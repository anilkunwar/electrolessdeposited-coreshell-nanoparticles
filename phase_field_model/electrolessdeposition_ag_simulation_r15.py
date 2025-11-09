#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
streamlit_electroless_batch_only.py
----------------------------------------
BATCH-ONLY electroless Ag simulator:
- Full physics per concentration
- Heat-maps of c, thickness evolution
- Interactive playback of any concentration
- CSV export of all diagnostics
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import time
from datetime import datetime

# Optional performance
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
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ----------------------------------------------------------------------
# Page & Style
# ----------------------------------------------------------------------
st.set_page_config(page_title="Electroless Ag — Batch Simulator", layout="wide")
st.title("Electroless Ag — Batch Simulator (Full Physics per c)")

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 140, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
})
CMAPS = [c for c in plt.colormaps() if c not in {"jet", "jet_r"}]

# ----------------------------------------------------------------------
# Sidebar: Scales
# ----------------------------------------------------------------------
st.sidebar.header("Physical Scales")
L0 = st.sidebar.number_input("L₀ (nm)", 1.0, 1e6, 20.0, 1.0) * 1e-9
E0 = st.sidebar.number_input("E₀ (×10⁻¹⁴ J)", 1e-6, 1e6, 1.0, 0.1) * 1e-14
tau0 = st.sidebar.number_input("τ₀ (×10⁻⁴ s)", 1e-6, 1e6, 1.0, 0.1) * 1e-4

# ----------------------------------------------------------------------
# Concentration Input
# ----------------------------------------------------------------------
st.sidebar.header("Concentrations (c_bulk)")
input_mode = st.sidebar.radio("Input mode", ["Manual (comma list)", "Range (start, end, steps)"])

if input_mode == "Manual (comma list)":
    default = "1.0, 0.5, 0.333, 0.25, 0.2"
    text = st.sidebar.text_area("Enter values", default, height=100)
    concentrations = [float(x) for x in text.replace("\n", ",").split(",") if x.strip()]
else:
    start = st.sidebar.number_input("Start", 0.05, 10.0, 0.2, 0.05)
    end = st.sidebar.number_input("End", 0.05, 10.0, 1.0, 0.05)
    steps = st.sidebar.slider("Steps", 2, 20, 5)
    concentrations = list(np.linspace(start, end, steps))

if not concentrations:
    concentrations = [1.0]
concentrations = sorted(set(round(c, 12) for c in concentrations), reverse=True)

# ----------------------------------------------------------------------
# Simulation Settings
# ----------------------------------------------------------------------
st.sidebar.header("Simulation")
mode = st.sidebar.selectbox("Mode", ["2D (planar)", "3D (spherical)"])
bc_type = st.sidebar.selectbox("BC", ["Neumann (zero flux)", "Dirichlet (fixed values)"])

if mode.startswith("2D"):
    Nx = st.sidebar.slider("Nx", 40, 400, 120, 10)
    Ny = st.sidebar.slider("Ny", 40, 400, 120, 10)
    Nz = 1
else:
    Nx = st.sidebar.slider("Nx", 16, 80, 40, 4)
    Ny = Nx
    Nz = st.sidebar.slider("Nz", 16, 80, 40, 4)

dt_nd = st.sidebar.number_input("dt (nd)", 1e-6, 2e-2, 2e-4, format="%.6f")
n_steps = st.sidebar.slider("Steps", 50, 8000, 800, 50)
save_every = st.sidebar.slider("Save every", 1, 400, max(1, n_steps//20), 1)

st.sidebar.header("Physics (nd)")
gamma_nd = st.sidebar.slider("γ", 1e-4, 0.5, 0.02, 1e-4)
beta_nd  = st.sidebar.slider("β", 0.1, 20.0, 4.0, 0.1)
k0_nd    = st.sidebar.slider("k₀", 0.01, 2.0, 0.4, 0.01)
M_nd     = st.sidebar.slider("M", 1e-3, 1.0, 0.2, 1e-3)
alpha_nd = st.sidebar.slider("α", 0.0, 10.0, 2.0, 0.1)
D_nd     = st.sidebar.slider("D", 0.0, 1.0, 0.05, 0.005)

use_numba = st.sidebar.checkbox("Numba", NUMBA_AVAILABLE)
use_semi = st.sidebar.checkbox("Semi-implicit", SCIPY_AVAILABLE and False)

cmap_choice = st.sidebar.selectbox("Colormap", CMAPS, CMAPS.index("viridis"))
use_plotly = st.sidebar.checkbox("Plotly", PLOTLY_AVAILABLE)

st.sidebar.header("Geometry")
core_radius_frac = st.sidebar.slider("Core radius / L", 0.05, 0.45, 0.18, 0.01)
shell_thickness_frac = st.sidebar.slider("Δr / r_core", 0.05, 0.6, 0.2, 0.01)
phi_threshold = st.sidebar.slider("φ threshold", 0.1, 0.9, 0.5, 0.05)

growth_model = st.sidebar.selectbox(
    "Model", ["Model A (irreversible)", "Model B (soft reversible)"]
)

# ----------------------------------------------------------------------
# Run Button
# ----------------------------------------------------------------------
run_button = st.sidebar.button(f"Run {len(concentrations)} Simulations")

# ----------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------
def scale_time(t): return t * tau0
def nd_to_real(x): return x * L0

# ----------------------------------------------------------------------
# Laplacian
# ----------------------------------------------------------------------
if NUMBA_AVAILABLE and use_numba:
    @njit(parallel=True)
    def lap2d(u, dx):
        n, m = u.shape
        out = np.zeros_like(u)
        for i in prange(1, n-1):
            for j in prange(1, m-1):
                out[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / (dx*dx)
        return out
    @njit(parallel=True)
    def lap3d(u, dx):
        n, m, p = u.shape
        out = np.zeros_like(u)
        for i in prange(1, n-1):
            for j in prange(1, m-1):
                for k in prange(1, p-1):
                    out[i,j,k] = (u[i+1,j,k] + u[i-1,j,k] + u[i,j+1,k] + u[i,j-1,k] +
                                  u[i,j,k+1] + u[i,j,k-1] - 6*u[i,j,k]) / (dx*dx)
        return out
else:
    def lap2d(u, dx): 
        out = np.zeros_like(u)
        out[1:-1,1:-1] = (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4*u[1:-1,1:-1]) / (dx*dx)
        return out
    def lap3d(u, dx): 
        out = np.zeros_like(u)
        out[1:-1,1:-1,1:-1] = (u[2:,1:-1,1:-1] + u[:-2,1:-1,1:-1] +
                               u[1:-1,2:,1:-1] + u[1:-1,:-2,1:-1] +
                               u[1:-1,1:-1,2:] + u[1:-1,1:-1,:-2] - 6*u[1:-1,1:-1,1:-1]) / (dx*dx)
        return out

# ----------------------------------------------------------------------
# Shell Thickness
# ----------------------------------------------------------------------
def compute_thickness(phi, psi, coords, core_frac, thresh, mode):
    if mode == "2D":
        x, y = coords
        X, Y = np.meshgrid(x, y, indexing='ij')
        dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
    else:
        x, y, z = coords
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
    mask = (phi > thresh) & (psi < 0.5)
    return np.max(dist[mask]) - core_frac if np.any(mask) else 0.0

# ----------------------------------------------------------------------
# Simulation Core
# ----------------------------------------------------------------------
def run_sim(params):
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    L = 1.0; dx = L/(Nx-1)
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny) if Ny > 1 else [0.5]
    z = np.linspace(0, L, Nz) if Nz > 1 else [0.5]
    X, Y = np.meshgrid(x, y, indexing='ij')
    dist2d = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
    psi = (dist2d <= params['core']*L).astype(float)
    r_core = params['core'] * L
    r_outer = r_core * (1 + params['shell_frac'])
    phi = np.where(dist2d <= r_core, 0.0,
                   np.where(dist2d <= r_outer, 1.0, 0.0))
    eps = max(4*dx, 1e-6)
    phi = phi * (1 - 0.5*(1 - np.tanh((dist2d-r_core)/eps))) \
              * (1 - 0.5*(1 + np.tanh((dist2d-r_outer)/eps)))
    phi = np.clip(phi, 0, 1)

    if params['bc'] == "Dirichlet (fixed values)":
        c = params['c_bulk'] * (1 - phi) * (1 - psi)
    else:
        c = params['c_bulk'] * (Y/L if Ny > 1 else Z/L if Nz > 1 else 1.0) * (1 - phi) * (1 - psi)
    c = np.clip(c, 0, params['c_bulk'])

    snapshots, diags, thick = [], [], []
    softness = 0.01 if "B" in params['model'] else 0.0
    max_th = 0.0

    for step in range(params['n_steps'] + 1):
        t = step * params['dt']

        # phi update
        if params['mode'] == "2D":
            lap_phi = lap2d(phi, dx)
            grad_phi = np.gradient(phi, dx)
            gphi = np.sqrt(grad_phi[0]**2 + grad_phi[1]**2 + 1e-30)
        else:
            X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')
            dist3 = np.sqrt((X3-0.5)**2 + (Y3-0.5)**2 + (Z3-0.5)**2)
            psi3 = (dist3 <= params['core']*L).astype(float)
            phi3 = phi.copy()
            c3 = c.copy()
            lap_phi = lap3d(phi3, dx)
            gx, gy, gz = np.gradient(phi3, dx)
            gphi = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-30)
            psi, phi, c = psi3, phi3, c3

        delta_int = 6*phi*(1-phi)*(1-psi)*gphi
        delta_int = np.clip(delta_int, 0, 6/max(eps,dx))
        f_bulk = 2*params['beta']*phi*(1-phi)*(1-2*phi)
        i_loc = params['k0'] * c * (1-phi) * (1-psi) * delta_int
        i_loc = np.clip(i_loc, 0, 1e6)

        dep = params['M'] * i_loc
        curv = params['M'] * params['gamma'] * lap_phi
        curv_pos = np.maximum(curv, 0)
        dphi = params['dt'] * (dep + curv_pos - softness * params['M'] * f_bulk)
        if softness == 0: dphi = np.maximum(dphi, 0)
        phi = np.clip(phi + dphi, 0, 1)

        # c update
        lap_c = lap2d(c, dx) if params['mode'] == "2D" else lap3d(c, dx)
        c += params['dt'] * (params['D'] * lap_c - i_loc)
        c = np.clip(c, 0, params['c_bulk'])

        # BC
        if params['bc'] == "Dirichlet (fixed values)":
            if params['mode'] == "2D":
                c[[0,-1],:] = c[:,[0,-1]] = params['c_bulk']
            else:
                c[[0,-1],:,:] = c[:,[0,-1],:] = c[:,:,[0,-1]] = params['c_bulk']
        else:
            if params['mode'] == "2D":
                c[:,-1] = params['c_bulk']
            else:
                c[:,:,-1] = params['c_bulk']

        if step % params['save_every'] == 0 or step == params['n_steps']:
            th_nd = compute_thickness(phi, psi, (x,y,z)[:params['mode'].count('D')+1],
                                    params['core'], params['thresh'], params['mode'])
            max_th = max(max_th, th_nd)
            c_mean, c_max = np.mean(c), np.max(c)
            total_ag = np.sum(i_loc) * params['dt']

            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diags.append((t, c_mean, c_max, total_ag))
            thick.append((t, max_th, nd_to_real(max_th), c_mean, c_max, total_ag))

    return snapshots, diags, thick, (x, y, z)[:len(phi.shape)]

# ----------------------------------------------------------------------
# Run All
# ----------------------------------------------------------------------
if run_button:
    results = {}
    prog = st.progress(0)
    status = st.empty()
    base = {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'dt': dt_nd, 'n_steps': n_steps, 'save_every': save_every,
        'gamma': gamma_nd, 'beta': beta_nd, 'k0': k0_nd, 'M': M_nd, 'D': D_nd, 'alpha': alpha_nd,
        'core': core_radius_frac, 'shell_frac': shell_thickness_frac,
        'bc': bc_type, 'thresh': phi_threshold, 'mode': mode, 'model': growth_model
    }

    for i, c_bulk in enumerate(concentrations):
        status.text(f"Running c = {c_bulk:.4g} ({i+1}/{len(concentrations)})")
        p = base.copy()
        p['c_bulk'] = c_bulk
        snaps, d, th, coords = run_sim(p)
        results[c_bulk] = {'snaps': snaps, 'diag': d, 'thick': th, 'coords': coords}
        prog.progress((i+1)/len(concentrations))

    st.session_state.results = results
    st.success(f"Batch complete: {len(concentrations)} simulations")

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
if 'results' in st.session_state:
    results = st.session_state.results
    concs = sorted(results.keys(), reverse=True)

    # Thickness vs Time
    st.header("Thickness vs Time")
    fig, ax = plt.subplots(figsize=(10,6))
    cmap = plt.get_cmap(cmap_choice)
    colors = cmap(np.linspace(0,1,len(concs)))
    for i, c in enumerate(concs):
        data = results[c]['thick']
        t = [scale_time(t) for t,_,_,_,_,_ in data]
        th = [th*1e9 for _,_,th,_,_,_ in data]
        ax.plot(t, th, label=f'c={c:.3g}', color=colors[i], lw=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Thickness (nm)")
    ax.legend(title="c_bulk", bbox_to_anchor=(1.02,1), loc='upper left')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Summary Table
    rows = []
    for c in concs:
        final = results[c]['thick'][-1]
        rows.append({
            'c_bulk': c,
            'Th_nm': final[2]*1e9,
            'c_mean': final[3],
            'c_max': final[4],
            'Total_Ag': final[5]
        })
    df = pd.DataFrame(rows).sort_values('c_bulk', ascending=False)
    st.subheader("Final State")
    st.dataframe(df.style.format({
        'c_bulk': '{:.4g}', 'Th_nm': '{:.2f}', 'c_mean': '{:.4f}',
        'c_max': '{:.4f}', 'Total_Ag': '{:.3e}'
    }))
    st.download_button("Download CSV", df.to_csv(index=False), "batch_summary.csv", "text/csv")

    # Playback
    st.header("Playback (select concentration)")
    sel_c = st.selectbox("Concentration", concs)
    snaps = results[sel_c]['snaps']
    frame = st.slider("Frame", 0, len(snaps)-1, len(snaps)-1)
    t, phi, c, psi = snaps[frame]
    t_real = scale_time(t)
    th_nm = results[sel_c]['thick'][frame][2]*1e9

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**t = {t_real:.3e} s** | **Th = {th_nm:.2f} nm**")
        fig, ax = plt.subplots()
        im = ax.imshow(c.T if mode=="2D" else c[c.shape[0]//2], cmap=cmap_choice,
                       vmin=0, vmax=sel_c, origin='lower')
        plt.colorbar(im, ax=ax, label='c')
        ax.set_title(f"Concentration (c) @ t = {t_real:.3e} s")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        im = ax.imshow(phi.T if mode=="2D" else phi[phi.shape[0]//2], cmap=cmap_choice,
                       vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=ax, label='φ')
        ax.set_title(f"Shell (φ)")
        st.pyplot(fig)
