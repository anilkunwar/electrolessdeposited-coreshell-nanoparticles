#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
streamlit_electroless_enhanced.py
--------------------------------
2-D / 3-D phase-field electroless Ag deposition.
All non-dimensional parameters are kept exactly as in the original
script; three user-adjustable scales (length, energy, time) convert
the simulation to real physical units while preserving numerical
behaviour and convergence.

NEW FEATURES (from the second code):
* molar-ratio control ([Ag]/[Cu])
* selectable BCs (Neumann / Dirichlet)
* automatic shell-thickness tracking (non-dim + nm)
* multi-ratio sweep + plots + CSV
* thickness-evolution plot for single run
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
L0 = L0 * 1e-9          # nm → m
E0 = E0 * 1e-14         # ×10⁻¹⁴ J → J
tau0 = tau0 * 1e-4      # ×10⁻⁴ s → s

# ------------------- FIXED: Molar ratio control -------------------
st.sidebar.header("Molar Ratio & Concentration")
molar_ratio_mode = st.sidebar.selectbox(
    "Concentration control mode",
    ["Fixed c_bulk", "Molar ratio c = [Ag]/[Cu]"]
)

# Base reservoir concentration
c_bulk_reservoir = st.sidebar.slider("Base reservoir concentration c_bulk", 0.1, 10.0, 2.0, 0.1)

if molar_ratio_mode == "Fixed c_bulk":
    c_bulk_nd = c_bulk_reservoir
    molar_ratio_values = [1.0]  # Single ratio for fixed mode
else:
    st.sidebar.markdown("**Preset Ag:Cu ratios** (1:5 → 1:1)")
    molar_ratios = np.array([1/5, 1/4, 1/3, 1/2, 1.0])
    molar_ratio_values = [c_bulk_reservoir * ratio for ratio in molar_ratios]
    st.sidebar.write("Active concentrations → c_bulk =", [f"{x:.3f}" for x in molar_ratio_values])
    c_bulk_nd = molar_ratio_values[0]  # start value for single run

# ------------------- boundary-condition selector -------------------
st.sidebar.header("Boundary Conditions")
bc_type = st.sidebar.selectbox(
    "Boundary condition type",
    ["Neumann (zero flux)", "Dirichlet (fixed values)"]
)

# ------------------- simulation mode & grid -------------------
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

# FIXED: Set simulation time to approximately 150 seconds
desired_real_time = 150.0  # seconds
dt_nd = st.sidebar.number_input("dt (non-dim)", 1e-6, 2e-2, 1e-3, format="%.6f")
# Calculate n_steps to achieve ~150 seconds
n_steps_target = int(desired_real_time / (dt_nd * tau0))
n_steps = st.sidebar.slider("n_steps", 50, 10000, min(n_steps_target, 10000), 50)
save_every = st.sidebar.slider("save every (frames)", 1, 400, max(1, n_steps//20), 1)

# Display estimated simulation time
estimated_time = n_steps * dt_nd * tau0
st.sidebar.write(f"Estimated simulation time: {estimated_time:.1f} s")

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

# ------------------- geometry -------------------
st.sidebar.header("Core & shell geometry")
core_radius_frac = st.sidebar.slider(
    "Core radius (fraction of L)", 0.05, 0.45, 0.18, 0.01
)
shell_thickness_frac = st.sidebar.slider(
    "Initial shell thickness (Δr / r_core)", 0.05, 0.6, 0.2, 0.01
)

# ------------------- shell-thickness analysis -------------------
st.sidebar.header("Shell Thickness Analysis")
phi_threshold = st.sidebar.slider(
    "Phi threshold for shell detection", 0.1, 0.9, 0.5, 0.05,
    help="phi > threshold → part of Ag shell"
)

run_button = st.sidebar.button("Run Simulation")
run_multiple_ratios = (st.sidebar.button("Run Multiple Molar Ratios")
                       if molar_ratio_mode == "Molar ratio c = [Ag]/[Cu]" else None)
export_vtu_button = st.sidebar.button("Export VTU/PVD/ZIP")
download_diags_button = st.sidebar.button("Download diagnostics CSV")

# ------------------- scaling helpers -------------------
def nd_to_real(length_nd):   return length_nd * L0
def real_to_nd(length_m):    return length_m / L0
def scale_time(t_nd):       return t_nd * tau0
def scale_diffusion(D_nd):  return D_nd * (L0**2 / tau0)
def scale_mobility(M_nd):   return M_nd * (L0**3 / (E0 * tau0))
def scale_reaction(k0_nd):  return k0_nd * (L0 / tau0)
def scale_energy_term(beta_nd): return beta_nd * (E0 / L0**3)
def scale_alpha(alpha_nd):  return alpha_nd * (E0 / L0**3)

# ------------------- shell-thickness computation -------------------
def compute_shell_thickness(phi, psi, coords, threshold=0.5, mode="2D"):
    """Return (thickness_nd, thickness_m)."""
    if mode.startswith("2D"):
        x, y = coords
        cx = cy = 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        distances = np.sqrt((X-cx)**2 + (Y-cy)**2)
    else:  # 3D
        x, y, z = coords
        cx = cy = cz = 0.5
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        distances = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)

    shell_mask = (phi > threshold) & (psi < 0.5)
    if np.any(shell_mask):
        max_shell_radius = np.max(distances[shell_mask])
        core_radius = core_radius_frac
        thickness_nd = max_shell_radius - core_radius
        thickness_m  = nd_to_real(thickness_nd)
        return thickness_nd, thickness_m
    return 0.0, 0.0

# ------------------- Laplacian (Numba) -------------------
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

# ------------------- simulation core -------------------
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

    # ---- initial concentration (BC dependent) ----
    if params['bc_type'] == "Dirichlet (fixed values)":
        c = params['c_bulk'] * np.ones_like(X) * (1.0 - phi) * (1.0 - psi)
    else:
        c = params['c_bulk'] * (Y/L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots, diagnostics, thickness_data = [], [], []
    n_steps = params['n_steps']; dt = params['dt']; save_every = params['save_every']
    gamma = params['gamma']; beta = params['beta']; k0 = params['k0']
    M = params['M']; D = params['D']; alpha = params['alpha']

    # ---- semi-implicit matrix (optional) ----
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

        # ---- BCs for phi ----
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

        deposition = M*i_loc
        curvature = M*gamma*lap_phi
        phi_temp = phi + dt*(deposition - M*f_bulk)

        if has_factor:
            phi = lu(phi_temp.ravel()).reshape(phi.shape)
        else:
            phi = phi_temp + dt*curvature
        phi = np.clip(phi, 0.0, 1.0)

        # ---- concentration update ----
        lap_c = laplacian_explicit_2d(c, dx)
        sink = i_loc
        c += dt*(D*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk']*5.0)

        # ---- BCs for c (after update) ----
        if params['bc_type'] == "Dirichlet (fixed values)":
            c[0,:] = c[-1,:] = c[:,0] = c[:,-1] = params['c_bulk']
        else:
            c[:, -1] = params['c_bulk']          # reservoir at top

        # ---- diagnostics ----
        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm_raw = np.sqrt(np.mean((M*gamma*lap_phi)**2))
        grad_norm_phys = np.sqrt(np.mean(((M*gamma*lap_phi)*(dx*dx))**2))
        alpha_c_norm = alpha*np.mean(c)

        if step % save_every == 0 or step == n_steps:
            thickness_nd, thickness_m = compute_shell_thickness(
                phi, psi, (x, y), params['phi_threshold'], "2D"
            )
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm))
            thickness_data.append((t, thickness_nd, thickness_m))

    return snapshots, diagnostics, thickness_data, (x, y)


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

    for step in range(n_steps+1):
        t = step*dt
        lap_phi = laplacian_explicit_3d(phi, dx)
        gx, gy, gz = np.gradient(phi, dx, edge_order=2)
        gphi = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-30)
        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps,dx))

        # ---- BCs phi ----
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
        phi += dt*(deposition + curvature - M*f_bulk)
        phi = np.clip(phi, 0.0, 1.0)

        lap_c = laplacian_explicit_3d(c, dx)
        sink = i_loc
        c += dt*(D*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk']*5.0)

        # ---- BCs c ----
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
                phi, psi, (x, y, z), params['phi_threshold'], "3D"
            )
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm))
            thickness_data.append((t, thickness_nd, thickness_m))

    return snapshots, diagnostics, thickness_data, (x, y, z)


# ------------------- multi-ratio runner -------------------
def run_multiple_simulations(params_base, concentrations):
    results = {}
    prog = st.progress(0)
    status = st.empty()
    for i, conc in enumerate(concentrations):
        status.text(f"Running c_bulk = {conc:.3f} …")
        p = params_base.copy()
        p['c_bulk'] = conc
        if p['mode'].startswith("2D"):
            snapshots, diags, thick, coords = run_simulation_2d(p)
        else:
            snapshots, diags, thick, coords = run_simulation_3d(p)
        results[conc] = {'snapshots': snapshots,
                      'diagnostics': diags,
                      'thickness': thick,
                      'coords': coords}
        prog.progress((i+1)/len(concentrations))
    status.text("All simulations finished.")
    return results


# ------------------- pack base parameters -------------------
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
for key in ["snapshots","diagnostics","thickness_data","grid_coords","multiple_results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------- run -------------------
if run_button:
    t0 = time.time()
    st.info("Running single simulation …")
    if mode.startswith("2D"):
        snapshots, diagnostics, thickness_data, coords = run_simulation_2d(params_base)
    else:
        snapshots, diagnostics, thickness_data, coords = run_simulation_3d(params_base)
    st.session_state.snapshots = snapshots
    st.session_state.diagnostics = diagnostics
    st.session_state.thickness_data = thickness_data
    st.session_state.grid_coords = coords
    st.session_state.multiple_results = None
    st.success(f"Done in {time.time()-t0:.2f}s — {len(snapshots)} frames")

if run_multiple_ratios and molar_ratio_mode == "Molar ratio c = [Ag]/[Cu]":
    t0 = time.time()
    st.info(f"Running {len(molar_ratio_values)} molar-ratio simulations …")
    multiple_results = run_multiple_simulations(params_base, molar_ratio_values)
    st.session_state.multiple_results = multiple_results
    st.session_state.snapshots = None
    st.success(f"All finished in {time.time()-t0:.2f}s")

# ------------------- FIXED: multi-ratio results -------------------
if st.session_state.multiple_results:
    st.header("Shell Thickness vs Molar Ratio")
    
    # Calculate the actual molar ratios from concentrations
    concentrations = list(st.session_state.multiple_results.keys())
    molar_ratios = [conc / c_bulk_reservoir for conc in concentrations]
    
    final_nd, final_nm = [], []
    for conc in concentrations:
        data = st.session_state.multiple_results[conc]['thickness']
        if data:
            final_nd.append(data[-1][1])
            final_nm.append(data[-1][2]*1e9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    
    # Plot against molar ratios
    ax1.plot(molar_ratios, final_nd, 'o-', lw=2, ms=8)
    ax1.set_xlabel('[Ag]/[Cu]'); 
    ax1.set_ylabel('Final thickness (non-dim)'); 
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Shell Thickness vs Molar Ratio')
    
    ax2.plot(molar_ratios, final_nm, 's--', lw=2, ms=8, c='tab:orange')
    ax2.set_xlabel('[Ag]/[Cu]'); 
    ax2.set_ylabel('Final thickness (nm)'); 
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Shell Thickness vs Molar Ratio')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Data table with both concentrations and ratios
    df_thick = pd.DataFrame({
        '[Ag]/[Cu]': molar_ratios,
        'c_bulk': concentrations,
        'Thickness_nd': final_nd,
        'Thickness_nm': final_nm
    })
    st.dataframe(df_thick.style.format({'Thickness_nm':'{:.2f}', 'c_bulk': '{:.3f}'}))
    
    st.download_button(
        "Download final-thickness CSV",
        df_thick.to_csv(index=False),
        file_name=f"final_thickness_vs_ratio_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv"
    )

    st.subheader("Thickness Evolution (all ratios)")
    fig_evo, ax_evo = plt.subplots(figsize=(10,6))
    colors = plt.cm.viridis(np.linspace(0,1,len(concentrations)))
    
    for i, (conc, ratio) in enumerate(zip(concentrations, molar_ratios)):
        data = st.session_state.multiple_results[conc]['thickness']
        if data:
            ts = [scale_time(t) for t,_,_ in data]
            th = [th*1e9 for _,_,th in data]
            ax_evo.plot(ts, th, label=f'[Ag]/[Cu] = {ratio:.3f}', color=colors[i], lw=2, marker='o', markersize=4)
    
    ax_evo.set_xlabel('Time (s)'); 
    ax_evo.set_ylabel('Thickness (nm)')
    ax_evo.set_title('Shell Growth for Different [Ag]/[Cu] Ratios')
    ax_evo.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax_evo.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_evo)

    # ---- full evolution CSV ----
    evo_rows = []
    for conc, ratio in zip(concentrations, molar_ratios):
        for t_nd, th_nd, th_m in st.session_state.multiple_results[conc]['thickness']:
            evo_rows.append({
                '[Ag]/[Cu]': ratio,
                'c_bulk': conc,
                't_s': scale_time(t_nd),
                'Thickness_nd': th_nd, 
                'Thickness_nm': th_m*1e9
            })
    
    evo_df = pd.DataFrame(evo_rows)
    st.download_button(
        "Download full evolution CSV",
        evo_df.to_csv(index=False),
        file_name=f"thickness_evolution_all_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv"
    )

# ------------------- single-run playback & post-processing -------------------
if st.session_state.snapshots and st.session_state.thickness_data:
    snapshots = st.session_state.snapshots
    diagnostics = st.session_state.diagnostics
    thickness_data = st.session_state.thickness_data
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
        th_nd, th_m = thickness_data[frame_idx][1], thickness_data[frame_idx][2]
        
        # Display current molar ratio for context
        current_ratio = params_base['c_bulk'] / c_bulk_reservoir if molar_ratio_mode == "Molar ratio c = [Ag]/[Cu]" else 1.0
        st.write(f"**Current configuration:** [Ag]/[Cu] = {current_ratio:.3f}, c_bulk = {params_base['c_bulk']:.3f}")
        st.write(f"**Current shell thickness:** {th_nd:.4f} (non-dim) = {th_m*1e9:.2f} nm")

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

        # ---- thickness evolution (single run) ----
        st.subheader("Shell Thickness Evolution")
        times = [scale_time(t) for t,_,_ in thickness_data]
        thick_nm = [th*1e9 for _,_,th in thickness_data]
        fig_th, ax_th = plt.subplots(figsize=(4,3))
        ax_th.plot(times, thick_nm, 'b-', lw=2)
        ax_th.set_xlabel('Time (s)'); ax_th.set_ylabel('Thickness (nm)')
        ax_th.grid(True, alpha=0.3); ax_th.set_title('Growth curve')
        st.pyplot(fig_th)

    # --------------------------------------------------------------
    # POST-PROCESSOR : material field + electric-potential proxy
    # --------------------------------------------------------------
    st.subheader("Material composition & electric-potential proxy")
    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        material_method = st.selectbox(
            "Material interpolation",
            ["phi + 2*psi (simple)",
             "phi*(1-psi) + 2*psi",
             "h·(phi² + psi²)",
             "h·(4*phi² + 2*psi²)",
             "max(phi, psi) + psi"],
            index=3,
            help="Choose how the two phase fields are merged into one colour map."
        )
    with col_b:
        show_potential = st.checkbox("Overlay electric-potential proxy (-α·c)", value=True)
    with col_c:
        if "h·" in material_method:
            h_factor = st.slider("h (scaling)", 0.1, 2.0, 0.5, 0.05,
                                 help="Scale factor for continuous material fields")
        else:
            h_factor = 1.0

    # ---------- build material ----------
    def build_material(phi, psi, method, h=1.0):
        if method == "phi + 2*psi (simple)":
            return phi + 2.0*psi
        elif method == "phi*(1-psi) + 2*psi":
            return phi*(1.0-psi) + 2.0*psi
        elif method == "h·(phi² + psi²)":
            return h*(phi**2 + psi**2)
        elif method == "h·(4*phi² + 2*psi²)":
            return h*(4.0*phi**2 + 2.0*psi**2)
        elif method == "max(phi, psi) + psi":
            return np.where(psi > 0.5, 2.0,
                   np.where(phi > 0.5, 1.0, 0.0))
        else:
            raise ValueError("unknown material method")

    material = build_material(phi_view, psi_view, material_method, h=h_factor)
    potential = -alpha_nd * c_view

    # ---------- colormap logic ----------
    if material_method in ["phi + 2*psi (simple)",
                           "phi*(1-psi) + 2*psi",
                           "max(phi, psi) + psi"]:
        cmap_mat = plt.cm.get_cmap("Set1", 3)
        vmin_mat, vmax_mat = 0, 2
    else:
        cmap_mat = cmap_choice
        vmin_mat = vmax_mat = None

    # ---------- 2-D visualisation ----------
    if mode.startswith("2D"):
        fig_mat, ax_mat = plt.subplots(figsize=(6,5))
        im_mat = ax_mat.imshow(material.T, origin='lower', extent=[0,1,0,1],
                               cmap=cmap_mat, vmin=vmin_mat, vmax=vmax_mat)
        if material_method in ["phi + 2*psi (simple)",
                               "phi*(1-psi) + 2*psi",
                               "max(phi, psi) + psi"]:
            cbar = plt.colorbar(im_mat, ax=ax_mat, ticks=[0,1,2])
            cbar.ax.set_yticklabels(['electrolyte','Ag shell','Cu core'])
        else:
            plt.colorbar(im_mat, ax=ax_mat, label="material")
        ax_mat.set_title(f"Material @ t = {t_real:.3e} s")
        st.pyplot(fig_mat)

        if show_potential:
            fig_pot, ax_pot = plt.subplots(figsize=(6,5))
            im_pot = ax_pot.imshow(potential.T, origin='lower', extent=[0,1,0,1],
                                   cmap="RdBu_r")
            plt.colorbar(im_pot, ax=ax_pot, label="Potential proxy -α·c")
            ax_pot.set_title(f"Potential proxy @ t = {t_real:.3e} s")
            st.pyplot(fig_pot)

            fig_comb, ax_comb = plt.subplots(figsize=(6,5))
            ax_comb.imshow(material.T, origin='lower', extent=[0,1,0,1],
                           cmap=cmap_mat, vmin=vmin_mat, vmax=vmax_mat, alpha=0.7)
            cs = ax_comb.contour(potential.T, levels=12, cmap="plasma",
                                 linewidths=0.8, alpha=0.9)
            ax_comb.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
            ax_comb.set_title("Material + Potential contours")
            st.pyplot(fig_comb)

    # ---------- 3-D visualisation ----------
    else:
        cx = phi_view.shape[0]//2
        cy = phi_view.shape[1]//2
        cz = phi_view.shape[2]//2
        fig_mat, axes = plt.subplots(1,3, figsize=(12,4))
        for ax, sl, label in zip(axes,
                                 [material[cx,:,:], material[:,cy,:], material[:,:,cz]],
                                 ["x-slice","y-slice","z-slice"]):
            im = ax.imshow(sl.T, origin='lower',
                           cmap=cmap_mat, vmin=vmin_mat, vmax=vmax_mat)
            ax.set_title(label); ax.axis('off')
        fig_mat.suptitle(f"Material (3-D slices) @ t = {t_real:.3e} s")
        st.pyplot(fig_mat)

        if show_potential:
            fig_pot, axes = plt.subplots(1,3, figsize=(12,4))
            for ax, sl, label in zip(axes,
                                     [potential[cx,:,:], potential[:,cy,:], potential[:,:,cz]],
                                     ["x-slice","y-slice","z-slice"]):
                im = ax.imshow(sl.T, origin='lower', cmap="RdBu_r")
                ax.set_title(label); ax.axis('off')
            fig_pot.suptitle(f"Potential proxy (-α·c) @ t = {t_real:.3e} s")
            plt.colorbar(im, ax=axes, orientation='horizontal',
                         fraction=0.05, label="-α·c")
            st.pyplot(fig_pot)

    # ------------------- diagnostics + thickness export -------------------
    if download_diags_button:
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["t (s)","||bulk||2","||grad||2_raw","||grad||2_scaled","alpha_mean_c"])
        for t_nd, b, gr, gs, ac in diagnostics:
            writer.writerow([scale_time(t_nd), b, gr, gs, ac])
        st.download_button(
            "Download diagnostics CSV",
            csv_buf.getvalue().encode(),
            file_name=f"diagnostics_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )

        # thickness CSV (single run)
        csv_th = io.StringIO()
        writer_th = csv.writer(csv_th)
        writer_th.writerow(["t (s)","Thickness_nd","Thickness_nm"])
        for t_nd, th_nd, th_m in thickness_data:
            writer_th.writerow([scale_time(t_nd), th_nd, th_m*1e9])
        st.download_button(
            "Download thickness CSV",
            csv_th.getvalue().encode(),
            file_name=f"thickness_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )

    # ------------------- PNG snapshot -------------------
    img_buf = io.BytesIO()
    if mode.startswith("2D"):
        fig_snap, ax_snap = plt.subplots(figsize=(5,4))
        if field == "phi (shell)":
            ax_snap.imshow(phi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap_choice)
        elif field == "c (concentration)":
            ax_snap.imshow(c_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap_choice)
        else:
            ax_snap.imshow(psi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap_choice)
        ax_snap.set_title(f"{field} t = {t_real:.3e} s")
        plt.colorbar(ax_snap.images[0], ax=ax_snap)
        fig_snap.tight_layout()
        fig_snap.savefig(img_buf, format='png', dpi=150); plt.close(fig_snap)
    else:
        fig_snap, axes_snap = plt.subplots(1,3,figsize=(10,3))
        cx = phi_view.shape[0]//2; cy = phi_view.shape[1]//2; cz = phi_view.shape[2]//2
        for ax, sl, title in zip(axes_snap,
                                 [phi_view[cx,:,:], phi_view[:,cy,:], phi_view[:,:,cz]],
                                 ["x","y","z"]):
            ax.imshow(sl.T, origin='lower', cmap=cmap_choice); ax.set_title(title); ax.axis('off')
        fig_snap.suptitle(f"{field} t = {t_real:.3e} s")
        fig_snap.tight_layout()
        fig_snap.savefig(img_buf, format='png', dpi=150); plt.close(fig_snap)
    img_buf.seek(0)
    st.download_button(
        "Download current snapshot (PNG)",
        img_buf,
        file_name=f"snapshot_t{t_real:.3e}s.png",
        mime="image/png"
    )

    # ------------------- VTU / PVD / ZIP -------------------
    if export_vtu_button:
        if not MESHIO_AVAILABLE:
            st.error("`meshio` not installed — VTU export disabled.")
        else:
            tmpdir = tempfile.mkdtemp()
            vtus = []
            for idx, (t_nd, phi_s, c_s, psi_s) in enumerate(snapshots):
                fname = os.path.join(tmpdir, f"frame_{idx:04d}.vtu")
                if mode.startswith("2D"):
                    xv, yv = coords
                    Xg, Yg = np.meshgrid(xv, yv, indexing='ij')
                    points = np.column_stack([nd_to_real(Xg.ravel()),
                                             nd_to_real(Yg.ravel()),
                                             np.zeros_like(Xg.ravel())])
                else:
                    xv, yv, zv = coords
                    Xg, Yg, Zg = np.meshgrid(xv, yv, zv, indexing='ij')
                    points = np.column_stack([nd_to_real(Xg.ravel()),
                                             nd_to_real(Yg.ravel()),
                                             nd_to_real(Zg.ravel())])
                mat_s = build_material(phi_s, psi_s, material_method, h=h_factor)
                point_data = {
                    "phi": phi_s.ravel().astype(np.float32),
                    "c": c_s.ravel().astype(np.float32),
                    "psi": psi_s.ravel().astype(np.float32),
                    "material": mat_s.ravel().astype(np.float32)
                }
                meshio.write_points_cells(fname, points, [], point_data=point_data)
                vtus.append(fname)

            pvd_path = os.path.join(tmpdir, "collection.pvd")
            with open(pvd_path, "w") as f:
                f.write("<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
                f.write(" <Collection>\n")
                for idx, v in enumerate(vtus):
                    f.write(f' <DataSet timestep="{scale_time(idx*params_base["dt"]):.3e}" file="{os.path.basename(v)}"/>\n')
                f.write(" </Collection>\n")
                f.write("</VTKFile>\n")

            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in vtus:
                    zf.write(p, arcname=os.path.basename(p))
                zf.write(pvd_path, arcname=os.path.basename(pvd_path))
            zipbuf.seek(0)
            st.download_button(
                "Download VTU/PVD ZIP",
                zipbuf.read(),
                file_name=f"frames_{datetime.now():%Y%m%d_%H%M%S}.zip",
                mime="application/zip"
            )
else:
    st.info("Run a simulation to see results. Tip: keep 3D grid ≤ 40 for fast runs.")
