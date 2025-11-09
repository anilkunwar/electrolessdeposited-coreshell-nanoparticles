#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
streamlit_electroless_enhanced.py
--------------------------------
2D/3D Phase-Field Electroless Ag Deposition
- [Ag]/[Cu] = 1:1 → c_bulk = fixed reservoir
- [Ag]/[Cu] = 1:2 → c_bulk = reservoir / 2
- Shell thickness grows with ratio
- Real time: 150 s, real length: nm
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io, zipfile, time, csv, os
from datetime import datetime
import tempfile

# ------------------- Optional Libraries -------------------
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

st.set_page_config(page_title="Electroless Ag — Enhanced", layout="wide")
st.title("Electroless Ag Deposition — Phase-Field Simulator")

# ------------------- Colormaps -------------------
CMAPS = [c for c in plt.colormaps() if c not in {"jet", "jet_r"}]

# ------------------- Sidebar: Scales -------------------
st.sidebar.header("Physical Scales")
L0 = st.sidebar.number_input("L₀ (nm)", 1.0, 1e6, 20.0, 1.0)
E0 = st.sidebar.number_input("E₀ (×10⁻¹⁴ J)", 1e-6, 1e6, 1.0, 0.1)
tau0 = st.sidebar.number_input("τ₀ (×10⁻⁴ s)", 1e-6, 1e6, 1.0, 0.1)

L0 *= 1e-9      # nm → m
E0 *= 1e-14
tau0 *= 1e-4

# ------------------- Molar Ratio Control -------------------
st.sidebar.header("Concentration Control")
conc_mode = st.sidebar.selectbox(
    "Mode", 
    ["Fixed c_bulk", "Molar Ratio [Ag]/[Cu]"]
)

# Fixed reservoir (used as base for ratios)
c_bulk_reservoir = st.sidebar.slider("Reservoir c_bulk (non-dim)", 0.1, 10.0, 2.0, 0.1)

if conc_mode == "Fixed c_bulk":
    molar_ratios = [1.0]
    c_bulk_nd = c_bulk_reservoir
else:
    st.sidebar.markdown("**Select [Ag]/[Cu] ratios**")
    ratio_options = {
        "1:5": 1/5, "1:4": 1/4, "1:3": 1/3, "1:2": 1/2, "1:1": 1.0, "2:1": 2.0
    }
    selected = st.sidebar.multiselect(
        "Ratios", list(ratio_options.keys()), default=["1:2", "1:1"]
    )
    molar_ratios = [ratio_options[k] for k in selected]
    if not molar_ratios:
        molar_ratios = [1.0]
    c_bulk_nd = c_bulk_reservoir * molar_ratios[0]  # for single run

# ------------------- BCs -------------------
st.sidebar.header("Boundary Conditions")
bc_type = st.sidebar.selectbox("BC Type", ["Neumann (zero-flux)", "Dirichlet (fixed)"])

# ------------------- Mode & Grid -------------------
st.sidebar.header("Simulation Mode")
mode = st.sidebar.selectbox("Mode", ["2D (planar)", "3D (spherical)"])

st.sidebar.header("Grid & Time")
if mode.startswith("2D"):
    Nx = st.sidebar.slider("Nx", 40, 400, 120, 10)
    Ny = st.sidebar.slider("Ny", 40, 400, 120, 10)
    Nz = 1
else:
    Nx = st.sidebar.slider("Nx", 16, 80, 40, 4)
    Ny = Nx
    Nz = st.sidebar.slider("Nz", 16, 80, 40, 4)

dt_nd = st.sidebar.number_input("dt (non-dim)", 1e-6, 2e-2, 2e-4, format="%.6f")
t_max_real = st.sidebar.slider("Simulation time (s)", 10.0, 300.0, 150.0, 10.0)
n_steps = int(t_max_real / (dt_nd * tau0)) + 1
save_every = st.sidebar.slider("Save every (frames)", 1, 200, max(1, n_steps//50), 1)

# ------------------- Physics -------------------
st.sidebar.header("Physics (non-dim)")
gamma_nd = st.sidebar.slider("γ (curvature)", 1e-4, 0.5, 0.02, 1e-4)
beta_nd = st.sidebar.slider("β (double-well)", 0.1, 20.0, 4.0, 0.1)
k0_nd = st.sidebar.slider("k₀ (reaction)", 0.01, 2.0, 0.4, 0.01)
M_nd = st.sidebar.slider("M (mobility)", 1e-3, 1.0, 0.2, 1e-3)
alpha_nd = st.sidebar.slider("α (coupling)", 0.0, 10.0, 2.0, 0.1)
D_nd = st.sidebar.slider("D (diffusion)", 0.0, 1.0, 0.05, 0.005)

# ------------------- Solver & Viz -------------------
st.sidebar.header("Solver")
use_numba = st.sidebar.checkbox("Use Numba", NUMBA_AVAILABLE)
use_semi_implicit = st.sidebar.checkbox("Semi-implicit IMEX", False) and SCIPY_AVAILABLE

st.sidebar.header("Visualization")
cmap_choice = st.sidebar.selectbox("Colormap", CMAPS, CMAPS.index("viridis"))

# ------------------- Geometry -------------------
st.sidebar.header("Geometry")
core_radius_frac = st.sidebar.slider("Core radius (frac L)", 0.05, 0.45, 0.18, 0.01)
shell_thickness_frac = st.sidebar.slider("Initial Δr / r_core", 0.05, 0.6, 0.2, 0.01)
phi_threshold = st.sidebar.slider("Phi threshold", 0.1, 0.9, 0.5, 0.05)

run_button = st.sidebar.button("Run Simulation")
run_multi = st.sidebar.button("Run All Ratios") if conc_mode == "Molar Ratio [Ag]/[Cu]" else None
export_vtu = st.sidebar.button("Export VTU/PVD/ZIP")
download_csv = st.sidebar.button("Download Diagnostics CSV")

# ------------------- Scaling -------------------
def nd_to_real(x): return x * L0
def scale_time(t): return t * tau0

# ------------------- Shell Thickness -------------------
def compute_thickness(phi, psi, coords, mode="2D", thresh=0.5):
    if mode.startswith("2D"):
        x, y = coords
        X, Y = np.meshgrid(x, y, indexing='ij')
        dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
    else:
        x, y, z = coords
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
    
    shell = (phi > thresh) & (psi < 0.5)
    if np.any(shell):
        r_max = np.max(dist[shell])
        thick_nd = r_max - core_radius_frac
        thick_m = nd_to_real(thick_nd)
        return thick_nd, thick_m
    return 0.0, 0.0

# ------------------- Laplacian -------------------
if NUMBA_AVAILABLE and use_numba:
    @njit(parallel=True)
    def laplacian_2d(u, dx):
        n = u.shape[0]
        out = np.zeros_like(u)
        for i in prange(1, n-1):
            for j in prange(1, n-1):
                out[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / (dx*dx)
        return out
    @njit(parallel=True)
    def laplacian_3d(u, dx):
        n = u.shape[0]
        out = np.zeros_like(u)
        for i in prange(1, n-1):
            for j in prange(1, n-1):
                for k in prange(1, n-1):
                    out[i,j,k] = (u[i+1,j,k]+u[i-1,j,k]+u[i,j+1,k]+u[i,j-1,k]+u[i,j,k+1]+u[i,j,k-1]-6*u[i,j,k])/(dx*dx)
        return out
else:
    def laplacian_2d(u, dx):
        out = np.zeros_like(u)
        out[1:-1,1:-1] = (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4*u[1:-1,1:-1]) / (dx*dx)
        return out
    def laplacian_3d(u, dx):
        out = np.zeros_like(u)
        out[1:-1,1:-1,1:-1] = (u[2:,1:-1,1:-1] + u[:-2,1:-1,1:-1] +
                               u[1:-1,2:,1:-1] + u[1:-1,:-2,1:-1] +
                               u[1:-1,1:-1,2:] + u[1:-1,1:-1,:-2] - 6*u[1:-1,1:-1,1:-1]) / (dx*dx)
        return out

# ------------------- Simulation -------------------
def run_sim(params):
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    L, dx = 1.0, 1.0/(Nx-1)
    x = np.linspace(0, L, Nx)
    if mode.startswith("2D"):
        y = np.linspace(0, L, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
        coords = (x, y)
    else:
        y = z = np.linspace(0, L, Nx)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
        coords = (x, y, z)

    # Geometry
    r_core = core_radius_frac * L
    r_outer = r_core * (1 + shell_thickness_frac)
    psi = (dist <= r_core).astype(float)
    eps = max(4*dx, 1e-6)
    phi = np.where(dist <= r_core, 0.0,
                   np.where(dist <= r_outer, 1.0, 0.0))
    phi *= (1 - 0.5*(1 - np.tanh((dist-r_core)/eps))) * (1 - 0.5*(1 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0.0, 1.0)

    # Concentration: c_bulk = reservoir × ratio
    c_bulk = params['c_bulk']
    if mode.startswith("2D"):
        c = c_bulk * (Y/L) * (1-phi) * (1-psi)
    else:
        c = c_bulk * (Z/L) * (1-phi) * (1-psi)
    c = np.clip(c, 0.0, c_bulk)

    snapshots, diags, thicks = [], [], []
    dt = params['dt']
    gamma, beta, k0, M, D, alpha = params['gamma'], params['beta'], params['k0'], params['M'], params['D'], params['alpha']

    for step in range(params['n_steps'] + 1):
        t = step * dt

        # Interface
        if mode.startswith("2D"):
            gphi = np.sqrt(
                ((phi[:,2:] - phi[:,:-2])/(2*dx))**2 +
                ((phi[2:,:] - phi[:-2,:])/(2*dx))**2 + 1e-30
            )
            gphi = np.pad(gphi, 1, mode='edge')
            lap_phi = laplacian_2d(phi, dx)
            lap_c = laplacian_2d(c, dx)
        else:
            gx, gy, gz = np.gradient(phi, dx)
            gphi = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-30)
            lap_phi = laplacian_3d(phi, dx)
            lap_c = laplacian_3d(c, dx)

        delta_int = 6*phi*(1-phi)*(1-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/eps)

        # BCs
        if params['bc_type'] == "Dirichlet (fixed)":
            phi[[0,-1],:,:] = phi[:,[0,-1],:] = phi[:,:,[0,-1]] = 0.0 if mode == "3D" else phi[[0,-1],:] = phi[:,[0,-1]] = 0.0
            c[[0,-1],:,:] = c[:,[0,-1],:] = c[:,:,[0,-1]] = c_bulk if mode == "3D" else c[[0,-1],:] = c[:,[0,-1]] = c_bulk
        else:
            if mode.startswith("2D"):
                phi[0,:] = phi[1,:]; phi[-1,:] = phi[-2,:]
                phi[:,0] = phi[:,1]; phi[:,-1] = phi[:,-2]
                c[:, -1] = c_bulk
            else:
                phi[0,:,:] = phi[1,:,:]; phi[-1,:,:] = phi[-2,:,:]
                phi[:,0,:] = phi[:,1,:]; phi[:,-1,:] = phi[:,-2,:]
                phi[:,:,0] = phi[:,:,1]; phi[:,:,-1] = phi[:,:,-2]
                c[:, :, -1] = c_bulk

        f_bulk = 2*beta*phi*(1-phi)*(1-2*phi)
        c_mol = c * (1-phi) * (1-psi)
        i_loc = k0 * c_mol * delta_int
        i_loc = np.clip(i_loc, 0.0, 1e6)

        deposition = M * i_loc
        curvature = M * gamma * lap_phi
        phi += dt * (deposition + curvature - M*f_bulk)
        phi = np.clip(phi, 0.0, 1.0)

        c += dt * (D * lap_c - i_loc)
        c = np.clip(c, 0.0, c_bulk*5)

        if step % params['save_every'] == 0 or step == params['n_steps']:
            thick_nd, thick_m = compute_thickness(phi, psi, coords, mode, params['phi_threshold'])
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diags.append((t, np.sqrt(np.mean(f_bulk**2)), np.sqrt(np.mean((M*gamma*lap_phi)**2)), alpha*np.mean(c)))
            thicks.append((t, thick_nd, thick_m))

    return snapshots, diags, thicks, coords

# ------------------- Run Single -------------------
if run_button:
    params = {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'dt': dt_nd, 'n_steps': n_steps, 'save_every': save_every,
        'gamma': gamma_nd, 'beta': beta_nd, 'k0': k0_nd, 'M': M_nd, 'D': D_nd, 'alpha': alpha_nd,
        'c_bulk': c_bulk_nd, 'core_radius_frac': core_radius_frac, 'shell_thickness_frac': shell_thickness_frac,
        'bc_type': bc_type, 'phi_threshold': phi_threshold
    }
    with st.spinner("Running..."):
        snapshots, diags, thicks, coords = run_sim(params)
    st.session_state.snapshots = snapshots
    st.session_state.diags = diags
    st.session_state.thicks = thicks
    st.session_state.coords = coords
    st.success("Done!")

# ------------------- Run Multiple -------------------
if run_multi and conc_mode == "Molar Ratio [Ag]/[Cu]":
    results = {}
    prog = st.progress(0)
    for i, ratio in enumerate(molar_ratios):
        c_bulk = c_bulk_reservoir * ratio
        st.write(f"Running [Ag]/[Cu] = {ratio:.3f} → c_bulk = {c_bulk:.3f}")
        params = {
            'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'dt': dt_nd, 'n_steps': n_steps, 'save_every': save_every,
            'gamma': gamma_nd, 'beta': beta_nd, 'k0': k0_nd, 'M': M_nd, 'D': D_nd, 'alpha': alpha_nd,
            'c_bulk': c_bulk, 'core_radius_frac': core_radius_frac, 'shell_thickness_frac': shell_thickness_frac,
            'bc_type': bc_type, 'phi_threshold': phi_threshold
        }
        snapshots, diags, thicks, coords = run_sim(params)
        results[ratio] = {'thicks': thicks}
        prog.progress((i+1)/len(molar_ratios))
    st.session_state.multi_results = results
    st.success("All ratios done!")

# ------------------- Results -------------------
if st.session_state.get('snapshots'):
    snapshots = st.session_state.snapshots
    diags = st.session_state.diags
    thicks = st.session_state.thicks
    coords = st.session_state.coords

    st.header("Results")
    frame = st.slider("Frame", 0, len(snapshots)-1, len(snapshots)-1)
    t_nd, phi, c, psi = snapshots[frame]
    t_real = scale_time(t_nd)
    thick_nd, thick_m = thicks[frame][1], thicks[frame][2]

    st.write(f"**Time:** {t_real:.1f} s | **Thickness:** {thick_m*1e9:.2f} nm")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(phi.T, origin='lower', extent=[0,1,0,1], cmap=cmap_choice)
    ax[0].set_title(f"ϕ (shell) @ {t_real:.1f} s"); plt.colorbar(im, ax=ax[0])
    im2 = ax[1].imshow(c.T, origin='lower', extent=[0,1,0,1], cmap=cmap_choice)
    ax[1].set_title("c (concentration)"); plt.colorbar(im2, ax=ax[1])
    st.pyplot(fig)

    times = [scale_time(t) for t, _, _ in thicks]
    thick_nm = [th*1e9 for _, _, th in thicks]
    fig2, ax2 = plt.subplots()
    ax2.plot(times, thick_nm, 'b-', lw=2)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Thickness (nm)")
    ax2.grid(True); st.pyplot(fig2)

# ------------------- Multi-Ratio Plot -------------------
if st.session_state.get('multi_results'):
    st.header("Thickness vs [Ag]/[Cu]")
    ratios = list(st.session_state.multi_results.keys())
    final_thick = [r['thicks'][-1][2]*1e9 for r in st.session_state.multi_results.values()]

    fig, ax = plt.subplots()
    ax.plot(ratios, final_thick, 'o-', lw=2, ms=8)
    ax.set_xlabel("[Ag]/[Cu]"); ax.set_ylabel("Final Thickness (nm)")
    ax.grid(True); st.pyplot(fig)

    df = pd.DataFrame({'[Ag]/[Cu]': ratios, 'Thickness (nm)': final_thick})
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "thickness_vs_ratio.csv")

st.info("Higher [Ag]/[Cu] → thicker shell. Use **Run All Ratios**.")
