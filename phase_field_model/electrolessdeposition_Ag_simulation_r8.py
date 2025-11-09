#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
streamlit_electroless_enhanced.py
--------------------------------
2-D / 3-D phase-field electroless Ag deposition with molar ratio control
and shell thickness tracking.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import io, zipfile, time, csv, os
from datetime import datetime
import tempfile

# ------------------- optional libs -------------------
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

st.set_page_config(page_title="Electroless Ag — Enhanced Simulator", layout="wide")
st.title("Electroless Ag — Enhanced Simulator (2D / 3D)")

# ------------------- colormap list -------------------
CMAPS = [c for c in plt.colormaps() if c not in {"jet", "jet_r"}]

# ------------------- sidebar – scales -------------------
st.sidebar.header("Physical scales (change units)")
L0 = st.sidebar.number_input(
    "Length scale L₀ (nm)", min_value=1.0, max_value=1e6, value=20.0, step=1.0,
    help="Reference length. 20 nm = 20 × 10⁻⁹ m (default Cu core diameter)."
)
E0 = st.sidebar.number_input(
    "Energy scale E₀ (×10⁻¹⁴ J)", min_value=1e-6, max_value=1e6, value=1.0, step=0.1,
    help="Reference energy (double-well depth)."
)
tau0 = st.sidebar.number_input(
    "Time scale τ₀ (×10⁻⁴ s)", min_value=1e-6, max_value=1e6, value=1.0, step=0.1,
    help="Reference time step scaling."
)

# conversion factors (internal non-dim → real)
L0 = L0 * 1e-9               # nm → m
E0 = E0 * 1e-14              # ×10⁻¹⁴ J → J
tau0 = tau0 * 1e-4           # ×10⁻⁴ s → s

# ------------------- UPDATED: Molar ratio control -------------------
st.sidebar.header("Molar Ratio & Concentration")
molar_ratio_mode = st.sidebar.selectbox(
    "Concentration control mode", 
    ["Fixed c_bulk", "Molar ratio c = [Ag]/[Cu]"]
)

if molar_ratio_mode == "Fixed c_bulk":
    c_bulk_nd = st.sidebar.slider("Fixed c_bulk (reservoir concentration)", 0.1, 10.0, 2.0, 0.1)
    molar_ratio_values = [c_bulk_nd]
else:
    st.sidebar.markdown("**Preset Ag:Cu ratios** (1:5 → 1:1)")
    molar_ratio_values = np.array([1/5, 1/4, 1/3, 1/2, 1.0])
    st.sidebar.write("Active ratios → [Ag]/[Cu] =", molar_ratio_values.round(3))
    c_bulk_nd = molar_ratio_values[0]  # start with first ratio

# ------------------- NEW: Boundary condition selection -------------------
st.sidebar.header("Boundary Conditions")
bc_type = st.sidebar.selectbox(
    "Boundary condition type",
    ["Neumann (zero flux)", "Dirichlet (fixed values)"]
)

# ------------------- sidebar – mode -------------------
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

# ------------------- physics (non-dimensional) -------------------
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

# ------------------- geometry (non-dim) -------------------
st.sidebar.header("Core & shell geometry")
core_radius_frac = st.sidebar.slider(
    "Core radius (fraction of L)", 0.05, 0.45, 0.18, 0.01
)
shell_thickness_frac = st.sidebar.slider(
    "Initial shell thickness (Δr / r_core)", 0.05, 0.6, 0.2, 0.01
)

# ------------------- NEW: Shell thickness analysis -------------------
st.sidebar.header("Shell Thickness Analysis")
phi_threshold = st.sidebar.slider(
    "Phi threshold for shell detection", 0.1, 0.9, 0.5, 0.05,
    help="Value of phi above which we consider it part of the shell"
)

run_button = st.sidebar.button("Run Simulation")
run_multiple_ratios = st.sidebar.button("Run Multiple Molar Ratios") if molar_ratio_mode == "Molar ratio c = [Ag]/[Cu]" else None

export_vtu_button = st.sidebar.button("Export VTU/PVD/ZIP")
download_diags_button = st.sidebar.button("Download diagnostics CSV")

# ------------------- scaling helpers -------------------
def nd_to_real(length_nd):
    """Non-dim length → metres."""
    return length_nd * L0

def real_to_nd(length_m):
    """Metres → non-dim."""
    return length_m / L0

def scale_time(t_nd):
    """Non-dim time → seconds."""
    return t_nd * tau0

def scale_diffusion(D_nd):
    """Non-dim D → m² s⁻¹."""
    return D_nd * (L0**2 / tau0)

def scale_mobility(M_nd):
    """Non-dim M → m⁴ J⁻¹ s⁻¹."""
    return M_nd * (L0**3 / (E0 * tau0))

def scale_reaction(k0_nd):
    """Non-dim k₀ → m s⁻¹ (mol m⁻³)⁻¹."""
    return k0_nd * (L0 / tau0)

def scale_energy_term(beta_nd):
    """Non-dim β → J m⁻³."""
    return beta_nd * (E0 / L0**3)

def scale_alpha(alpha_nd):
    """Non-dim α → (J m⁻³) / (mol m⁻³)."""
    return alpha_nd * (E0 / L0**3)

def scale_c(c_nd):
    """Non-dim concentration → mol m⁻³ (reference = c_bulk_nd)."""
    return c_nd * 1.0   # c_bulk_nd is already the reference

# ------------------- NEW: Shell thickness computation -------------------
def compute_shell_thickness(phi, psi, coords, threshold=0.5, mode="2D"):
    """
    Compute shell thickness from phase field phi.
    Returns thickness in non-dimensional units and real units (m).
    """
    if mode.startswith("2D"):
        x, y = coords
        cx, cy = 0.5, 0.5  # center
        X, Y = np.meshgrid(x, y, indexing='ij')
        distances = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Mask for shell region (phi > threshold and not in core)
        shell_mask = (phi > threshold) & (psi < 0.5)
        
        if np.any(shell_mask):
            shell_distances = distances[shell_mask]
            if len(shell_distances) > 0:
                # Shell thickness = max distance in shell - core radius
                max_shell_radius = np.max(shell_distances)
                core_radius = core_radius_frac  # known core radius
                thickness_nd = max_shell_radius - core_radius
                thickness_real = nd_to_real(thickness_nd)
                return thickness_nd, thickness_real
        
        return 0.0, 0.0
    
    else:  # 3D
        x, y, z = coords
        cx, cy, cz = 0.5, 0.5, 0.5  # center
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        distances = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        
        # Mask for shell region
        shell_mask = (phi > threshold) & (psi < 0.5)
        
        if np.any(shell_mask):
            shell_distances = distances[shell_mask]
            if len(shell_distances) > 0:
                max_shell_radius = np.max(shell_distances)
                core_radius = core_radius_frac
                thickness_nd = max_shell_radius - core_radius
                thickness_real = nd_to_real(thickness_nd)
                return thickness_nd, thickness_real
        
        return 0.0, 0.0

# ------------------- operators (non-dim) -------------------
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
    ux[:,0] = (u[:,1] - u[:,0]) / dx
    ux[:,-1] = (u[:,-1] - u[:,-2]) / dx
    uy[1:-1,:] = (u[2:,:] - u[:-2,:]) / (2*dx)
    uy[0,:] = (u[1,:] - u[:,0]) / dx
    uy[-1,:] = (u[-1,:] - u[-2,:]) / dx
    return np.sqrt(ux**2 + uy**2 + 1e-30)

# ------------------- simulation core (non-dim) -------------------
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

    # NEW: Different BC handling
    if params['bc_type'] == "Dirichlet (fixed values)":
        c = params['c_bulk'] * np.ones_like(X) * (1.0 - phi) * (1.0 - psi)
    else:  # Neumann
        c = params['c_bulk'] * (Y/L) * (1.0 - phi) * (1.0 - psi)
    
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots, diagnostics, shell_thickness_data = [], [], []
    n_steps = params['n_steps']; dt = params['dt']; save_every = params['save_every']
    gamma = params['gamma']; beta = params['beta']; k0 = params['k0']
    M = params['M']; D = params['D']; alpha = params['alpha']
    phi_threshold = params.get('phi_threshold', 0.5)

    # ----- semi-implicit matrix (optional) -----
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

    for step in range(n_steps+1):
        t = step*dt

        gphi = grad_mag_2d(phi, dx)
        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps,dx))

        # BCs handling
        if params['bc_type'] == "Dirichlet (fixed values)":
            # Dirichlet BCs - fixed values at boundaries
            phi[0,:] = 0.0; phi[-1,:] = 0.0
            phi[:,0] = 0.0; phi[:,-1] = 0.0
            c[0,:] = params['c_bulk']; c[-1,:] = params['c_bulk']
            c[:,0] = params['c_bulk']; c[:,-1] = params['c_bulk']
        else:
            # Neumann BCs - zero flux
            phi[0,:] = phi[1,:]; phi[-1,:] = phi[-2,:]
            phi[:,0] = phi[:,1]; phi[:,-1] = phi[:,-2]
            # For c in Neumann, only top boundary is fixed to c_bulk
            c[:, -1] = params['c_bulk']

        lap_phi = laplacian_explicit_2d(phi, dx)
        f_bulk = 2.0*beta*phi*(1.0-phi)*(1.0-2.0*phi)
        c_mol = c*(1.0-phi)*(1.0-psi)
        i_loc = k0*c_mol*delta_int
        i_loc = np.clip(i_loc, 0.0, 1e6)

        deposition = M*i_loc
        curvature = M*gamma*lap_phi
        phi_temp = phi + dt*(deposition - M*f_bulk)

        if has_factor:
            phi = lu(phi_temp.ravel()).reshape(phi.shape)
        else:
            phi = phi_temp + dt*curvature
        phi = np.clip(phi, 0.0, 1.0)

        lap_c = laplacian_explicit_2d(c, dx)
        sink = i_loc
        c += dt*(D*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk']*5.0)
        
        # Apply BCs again after update
        if params['bc_type'] == "Dirichlet (fixed values)":
            c[0,:] = params['c_bulk']; c[-1,:] = params['c_bulk']
            c[:,0] = params['c_bulk']; c[:,-1] = params['c_bulk']
        else:
            c[:, -1] = params['c_bulk']

        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm_raw = np.sqrt(np.mean((M*gamma*lap_phi)**2))
        grad_norm_phys = np.sqrt(np.mean(((M*gamma*lap_phi)*(dx*dx))**2))
        alpha_c_norm = alpha*np.mean(c)

        if step % save_every == 0 or step == n_steps:
            # NEW: Compute shell thickness
            thickness_nd, thickness_real = compute_shell_thickness(
                phi, psi, (x, y), phi_threshold, "2D"
            )
            
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm))
            shell_thickness_data.append((t, thickness_nd, thickness_real))

    return snapshots, diagnostics, shell_thickness_data, (x, y)

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

    # NEW: Different BC handling
    if params['bc_type'] == "Dirichlet (fixed values)":
        c = params['c_bulk'] * np.ones_like(X) * (1.0 - phi) * (1.0 - psi)
    else:  # Neumann
        c = params['c_bulk'] * (Z/L) * (1.0 - phi) * (1.0 - psi)
    
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots, diagnostics, shell_thickness_data = [], [], []
    n_steps = params['n_steps']; dt = params['dt']; save_every = params['save_every']
    gamma = params['gamma']; beta = params['beta']; k0 = params['k0']
    M = params['M']; D = params['D']; alpha = params['alpha']
    phi_threshold = params.get('phi_threshold', 0.5)

    for step in range(n_steps+1):
        t = step*dt

        lap_phi = laplacian_explicit_3d(phi, dx)
        gx, gy, gz = np.gradient(phi, dx, edge_order=2)
        gphi = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-30)
        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps,dx))

        # BCs handling
        if params['bc_type'] == "Dirichlet (fixed values)":
            # Dirichlet BCs
            phi[0,:,:] = 0.0; phi[-1,:,:] = 0.0
            phi[:,0,:] = 0.0; phi[:,-1,:] = 0.0
            phi[:,:,0] = 0.0; phi[:,:,-1] = 0.0
            c[0,:,:] = params['c_bulk']; c[-1,:,:] = params['c_bulk']
            c[:,0,:] = params['c_bulk']; c[:,-1,:] = params['c_bulk']
            c[:,:,0] = params['c_bulk']; c[:,:,-1] = params['c_bulk']
        else:
            # Neumann BCs
            phi[0,:,:] = phi[1,:,:]; phi[-1,:,:] = phi[-2,:,:]
            phi[:,0,:] = phi[:,1,:]; phi[:,-1,:] = phi[:,-2,:]
            phi[:,:,0] = phi[:,:,1]; phi[:,:,-1] = phi[:,:,-2]
            c[:, :, -1] = params['c_bulk']

        f_bulk = 2.0*beta*phi*(1.0-phi)*(1.0-2.0*phi)
        c_mol = c*(1.0-phi)*(1.0-psi)
        i_loc = k0*c_mol*delta_int
        i_loc = np.clip(i_loc, 0.0, 1e6)

        deposition = M*i_loc
        curvature = M*gamma*lap_phi
        phi += dt*(deposition + curvature - M*f_bulk)
        phi = np.clip(phi, 0.0, 1.0)

        lap_c = laplacian_explicit_3d(c, dx)
        sink = i_loc
        c += dt*(D*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk']*5.0)
        
        # Apply BCs again after update
        if params['bc_type'] == "Dirichlet (fixed values)":
            c[0,:,:] = params['c_bulk']; c[-1,:,:] = params['c_bulk']
            c[:,0,:] = params['c_bulk']; c[:,-1,:] = params['c_bulk']
            c[:,:,0] = params['c_bulk']; c[:,:,-1] = params['c_bulk']
        else:
            c[:, :, -1] = params['c_bulk']

        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm_raw = np.sqrt(np.mean((M*gamma*lap_phi)**2))
        grad_norm_phys = np.sqrt(np.mean(((M*gamma*lap_phi)*(dx*dx))**2))
        alpha_c_norm = alpha*np.mean(c)

        if step % save_every == 0 or step == n_steps:
            # NEW: Compute shell thickness
            thickness_nd, thickness_real = compute_shell_thickness(
                phi, psi, (x, y, z), phi_threshold, "3D"
            )
            
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm))
            shell_thickness_data.append((t, thickness_nd, thickness_real))

    return snapshots, diagnostics, shell_thickness_data, (x, y, z)

# ------------------- UPDATED: Multiple ratio runner -------------------
def run_multiple_simulations(params_base, molar_ratios):
    """Run multiple simulations for different [Ag]/[Cu] molar ratios."""
    all_results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ratio in enumerate(molar_ratios):
        status_text.text(f"Running simulation for [Ag]/[Cu] = {ratio:.3f}")
        params = params_base.copy()
        params['c_bulk'] = ratio

        if params_base['mode'].startswith("2D"):
            snapshots, diagnostics, shell_thickness_data, coords = run_simulation_2d(params)
        else:
            snapshots, diagnostics, shell_thickness_data, coords = run_simulation_3d(params)

        all_results[ratio] = {
            'snapshots': snapshots,
            'diagnostics': diagnostics,
            'shell_thickness': shell_thickness_data,
            'coords': coords
        }

        progress_bar.progress((i + 1) / len(molar_ratios))

    status_text.text("✅ All simulations completed.")
    return all_results

# ------------------- pack parameters -------------------
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
    'mode': mode
}

# ------------------- session state -------------------
if "snapshots" not in st.session_state:
    st.session_state.snapshots = None
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = None
if "shell_thickness_data" not in st.session_state:
    st.session_state.shell_thickness_data = None
if "grid_coords" not in st.session_state:
    st.session_state.grid_coords = None
if "multiple_results" not in st.session_state:
    st.session_state.multiple_results = None

# ------------------- run -------------------
if run_button:
    t0 = time.time()
    st.info("Running simulation …")
    if mode.startswith("2D"):
        snapshots, diagnostics, shell_thickness_data, coords = run_simulation_2d(params_base)
    else:
        snapshots, diagnostics, shell_thickness_data, coords = run_simulation_3d(params_base)
    
    st.session_state.snapshots = snapshots
    st.session_state.diagnostics = diagnostics
    st.session_state.shell_thickness_data = shell_thickness_data
    st.session_state.grid_coords = coords
    st.session_state.multiple_results = None  # Clear multiple results
    st.success(f"Done in {time.time()-t0:.2f}s — {len(snapshots)} frames")

if run_multiple_ratios and molar_ratio_mode == "Molar ratio c = [Ag]/[Cu]":
    t0 = time.time()
    st.info(f"Running {len(molar_ratio_values)} simulations for different molar ratios…")
    
    multiple_results = run_multiple_simulations(params_base, molar_ratio_values)
    st.session_state.multiple_results = multiple_results
    st.session_state.snapshots = None  # Clear single results
    
    st.success(f"All simulations completed in {time.time()-t0:.2f}s")

# ------------------- UPDATED: Shell thickness analysis display -------------------
if st.session_state.multiple_results:
    st.header("Shell Thickness vs Ag:Cu Molar Ratio")

    molar_ratios = list(st.session_state.multiple_results.keys())
    final_thickness_nd, final_thickness_real = [], []

    for ratio in molar_ratios:
        shell_data = st.session_state.multiple_results[ratio]['shell_thickness']
        if shell_data:
            t_final, th_nd, th_real = shell_data[-1]
            final_thickness_nd.append(th_nd)
            final_thickness_real.append(th_real)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Non-dimensional
    ax1.plot(molar_ratios, final_thickness_nd, 'o-', lw=2, ms=8, label='Non-dim thickness')
    ax1.set_xlabel('Molar Ratio [Ag]/[Cu]')
    ax1.set_ylabel('Final Shell Thickness (non-dim)')
    ax1.set_title('Non-dimensional Shell Thickness')
    ax1.grid(True, alpha=0.3)

    # Real units (nm)
    ax2.plot(molar_ratios, np.array(final_thickness_real)*1e9, 's--', lw=2, ms=8, c='tab:orange', label='Thickness (nm)')
    ax2.set_xlabel('Molar Ratio [Ag]/[Cu]')
    ax2.set_ylabel('Shell Thickness (nm)')
    ax2.set_title('Physical Shell Thickness')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Data table
    thickness_df = pd.DataFrame({
        '[Ag]/[Cu] Ratio': molar_ratios,
        'Shell_Thickness_nd': final_thickness_nd,
        'Shell_Thickness_nm': np.array(final_thickness_real)*1e9
    })
    st.dataframe(thickness_df.style.format({'Shell_Thickness_nm': '{:.2f}'}))

    csv_thickness = thickness_df.to_csv(index=False)
    st.download_button(
        "📥 Download Shell Thickness Data (CSV)",
        csv_thickness,
        file_name=f"shell_thickness_vs_molar_ratio_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv"
    )

    # NEW: Shell thickness evolution for all ratios
    st.subheader("Shell Thickness Evolution Over Time")
    
    fig_evolution, ax_evolution = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(molar_ratios)))
    
    for i, ratio in enumerate(molar_ratios):
        shell_data = st.session_state.multiple_results[ratio]['shell_thickness']
        if shell_data:
            times = [scale_time(data[0]) for data in shell_data]
            thickness_nm = [data[2] * 1e9 for data in shell_data]
            ax_evolution.plot(times, thickness_nm, 
                            label=f'[Ag]/[Cu] = {ratio:.3f}', 
                            color=colors[i], 
                            linewidth=2,
                            marker='o', 
                            markersize=4)
    
    ax_evolution.set_xlabel('Time (s)')
    ax_evolution.set_ylabel('Shell Thickness (nm)')
    ax_evolution.set_title('Shell Thickness Evolution for Different Molar Ratios')
    ax_evolution.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_evolution.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_evolution)

    # NEW: Download comprehensive evolution data
    st.subheader("Download Complete Evolution Data")
    
    evolution_data = []
    for ratio in molar_ratios:
        shell_data = st.session_state.multiple_results[ratio]['shell_thickness']
        if shell_data:
            for t_nd, thickness_nd, thickness_real in shell_data:
                evolution_data.append({
                    '[Ag]/[Cu]': ratio,
                    'Time_s': scale_time(t_nd),
                    'Shell_Thickness_nd': thickness_nd,
                    'Shell_Thickness_nm': thickness_real * 1e9
                })
    
    evolution_df = pd.DataFrame(evolution_data)
    csv_evolution = evolution_df.to_csv(index=False)
    
    st.download_button(
        "📥 Download Complete Evolution Data (CSV)",
        csv_evolution,
        file_name=f"shell_evolution_all_ratios_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv"
    )

# ------------------- playback & post-processing -------------------
if st.session_state.snapshots and st.session_state.shell_thickness_data:
    snapshots = st.session_state.snapshots
    diagnostics = st.session_state.diagnostics
    shell_thickness_data = st.session_state.shell_thickness_data
    coords = st.session_state.grid_coords

    st.header("Results & Playback")
    cols = st.columns([3,1])
    
    with cols[0]:
        frame_idx = st.slider("Frame", 0, len(snapshots)-1, len(snapshots)-1)
        auto_play = st.checkbox("Autoplay", value=False)
        autoplay_interval = st.number_input("Interval (s)", 0.1, 5.0, 0.4, 0.1)
        field = st.selectbox("Field", ["phi (shell)", "c (concentration)", "psi (core)"])
        t_nd, phi_view, c_view, psi_view = snapshots[frame_idx]
        t_real = scale_time(t_nd)
        
        # NEW: Display current shell thickness
        current_thickness_nd, current_thickness_real = shell_thickness_data[frame_idx][1], shell_thickness_data[frame_idx][2]
        st.write(f"Current shell thickness: {current_thickness_nd:.4f} (non-dim) = {current_thickness_real*1e9:.2f} nm")

        cmap = plt.get_cmap(cmap_choice)

        if mode.startswith("2D"):
            fig, ax = plt.subplots(figsize=(6,5))
            if field == "phi (shell)":
                im = ax.imshow(phi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap)
            elif field == "c (concentration)":
                im = ax.imshow(c_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap)
            else:
                im = ax.imshow(psi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap)
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{field} @ t = {t_real:.3e} s")
            st.pyplot(fig)

            mid = phi_view.shape[0]//2
            fig2, ax2 = plt.subplots(figsize=(6,2.2))
            if field == "phi (shell)":
                ax2.plot(np.linspace(0,1,phi_view.shape[1]), phi_view[mid,:], label='phi')
            elif field == "c (concentration)":
                ax2.plot(np.linspace(0,1,c_view.shape[1]), c_view[mid,:], label='c')
            else:
                ax2.plot(np.linspace(0,1,psi_view.shape[1]), psi_view[mid,:], label='psi')
            ax2.set_xlabel("y/L"); ax2.legend(); ax2.grid(True)
            st.pyplot(fig2)
        else:
            fig, axes = plt.subplots(1,3, figsize=(12,4))
            cx = phi_view.shape[0]//2; cy = phi_view.shape[1]//2; cz = phi_view.shape[2]//2
            for ax, sl, title in zip(axes,
                                     [phi_view[cx,:,:], phi_view[:,cy,:], phi_view[:,:,cz]],
                                     ["x-slice","y-slice","z-slice"]):
                ax.imshow(sl.T, origin='lower', cmap=cmap); ax.set_title(title); ax.axis('off')
            fig.suptitle(f"{field} @ t = {t_real:.3e} s")
            st.pyplot(fig)

        if auto_play:
            for i in range(frame_idx, len(snapshots)):
                time.sleep(autoplay_interval)
                st.session_state._rerun = True

    with cols[1]:
        st.subheader("Diagnostics")
        df = pd.DataFrame(diagnostics,
                          columns=["t*","||bulk||₂","||grad||₂ raw","||grad||₂ scaled","α·mean(c)"])
        st.dataframe(df.tail(20).style.format("{:.3e}"))

        fig3, ax3 = plt.subplots(figsize=(4,3))
        ax3.semilogy(df["t*"], np.maximum(df["||bulk||₂"],1e-30), label='bulk')
        ax3.semilogy(df["t*"], np.maximum(df["||grad||₂ raw"],1e-30), label='grad raw')
        ax3.semilogy(df["t*"], np.maximum(df["||grad||₂ scaled"],1e-30), label='grad scaled')
        ax3.semilogy(df["t*"], np.maximum(df["α·mean(c)"],1e-30), label='α·c')
        ax3.legend(fontsize=8); ax3.grid(True)
        st.pyplot(fig3)
        
        # NEW: Shell thickness evolution plot
        st.subheader("Shell Thickness Evolution")
        times = [scale_time(data[0]) for data in shell_thickness_data]
        thickness_nm = [data[2] * 1e9 for data in shell_thickness_data]  # Convert to nm
        
        fig_thickness, ax_thickness = plt.subplots(figsize=(4,3))
        ax_thickness.plot(times, thickness_nm, 'b-', linewidth=2)
        ax_thickness.set_xlabel('Time (s)')
        ax_thickness.set_ylabel('Shell Thickness (nm)')
        ax_thickness.grid(True, alpha=0.3)
        ax_thickness.set_title('Shell Thickness vs Time')
        st.pyplot(fig_thickness)

    # ------------------- diagnostics export -------------------
    if download_diags_button:
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["t (s)","||bulk||2","||grad||2_raw","||grad||2_scaled","alpha_mean_c"])
        for t_nd, b, gr, gs, ac in diagnostics:
            writer.writerow([scale_time(t_nd), b, gr, gs, ac])
        st.download_button(
            "Download diagnostics CSV",
            csv_buffer.getvalue().encode(),
            file_name=f"diagnostics_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )
        
        # NEW: Download shell thickness data
        csv_thickness = io.StringIO()
        writer_thickness = csv.writer(csv_thickness)
        writer_thickness.writerow(["t (s)", "Shell_Thickness_nd", "Shell_Thickness_nm"])
        for t_nd, thickness_nd, thickness_real in shell_thickness_data:
            writer_thickness.writerow([scale_time(t_nd), thickness_nd, thickness_real * 1e9])
        st.download_button(
            "Download Shell Thickness CSV",
            csv_thickness.getvalue().encode(),
            file_name=f"shell_thickness_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )

else:
    st.info("Run a simulation to see results. Tip: keep 3D grid ≤ 40 for fast runs.")
