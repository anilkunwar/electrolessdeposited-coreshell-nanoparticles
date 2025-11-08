"""
streamlit_electroless_enhanced.py

Features:
 - 2D / 3D electroless deposition shell growth (simple physics)
 - Numba acceleration if available
 - Optional semi-implicit IMEX solver using scipy.sparse (fallback to explicit)
 - Animation slider + autoplay
 - Export diagnostics CSV, PNG snapshots, VTU/PVD outputs, and zipped VTUs
 - Selectable Matplotlib colormap from a large list (~50)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import io, zipfile, time, csv, os
from datetime import datetime
import base64
import tempfile

# Optional acceleration / I/O libs (use if available)
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
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except Exception:
    PYVISTA_AVAILABLE = False

st.set_page_config(page_title="Electroless Ag — Enhanced Simulator", layout="wide")
st.title("Electroless Ag — Enhanced Simulator (2D / 3D)")

# -----------------------
# Utility: large colormap list (~50)
# -----------------------
# We'll present many common colormaps including jet, rainbow, turbo, viridis...
CMAPS = [
    "viridis","plasma","inferno","magma","cividis",
    "turbo","jet","rainbow","nipy_spectral","cubehelix",
    "summer","autumn","winter","spring","cool",
    "hot","copper","bone","pink","ocean",
    "gist_earth","terrain","gnuplot","gnuplot2","seismic",
    "Spectral","coolwarm","bwr","BrBG","PuOr",
    "RdYlBu","viridis_r","plasma_r","inferno_r","magma_r",
    "cividis_r","twilight","twilight_shifted","gist_rainbow","rainbow_r",
    "turbo_r","jet_r","Greys","Purples","Oranges","Greens","YlOrRd","YlGnBu","YlGn"
]
# ensure unique and existing
CMAPS = [c for c in CMAPS if c in plt.colormaps()]

# -----------------------
# Sidebar: simulation config
# -----------------------
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

dt = st.sidebar.number_input("dt (time step)", 1e-6, 2e-2, 2e-4, format="%.6f")
n_steps = st.sidebar.slider("n_steps", 50, 8000, 800, 50)
save_every = st.sidebar.slider("save every (frames)", 1, 400, max(1, n_steps//20), 1)

st.sidebar.header("Physics params")
gamma = st.sidebar.slider("γ (curvature strength)", 1e-4, 0.5, 0.02, 1e-4, format="%.4f")
beta = st.sidebar.slider("β (double-well strength)", 0.1, 20.0, 4.0, 0.1)
k0 = st.sidebar.slider("k₀ (reaction prefactor)", 0.01, 2.0, 0.4, 0.01)
M = st.sidebar.slider("M (mobility)", 1e-3, 1.0, 0.2, 1e-3, format="%.3f")
alpha = st.sidebar.slider("α (coupling)", 0.0, 10.0, 2.0, 0.1)
c_bulk = st.sidebar.slider("c_bulk (reservoir)", 0.1, 10.0, 2.0, 0.1)
D = st.sidebar.slider("D (diffusion)", 0.0, 1.0, 0.05, 0.005)

st.sidebar.header("Solver & performance")
use_numba = st.sidebar.checkbox("Use numba (if available)", value=NUMBA_AVAILABLE)
use_semi_implicit = st.sidebar.checkbox("Semi-implicit IMEX for Laplacian (requires scipy)", value=False)
if use_semi_implicit and not SCIPY_AVAILABLE:
    st.sidebar.warning("SciPy not found — semi-implicit will be disabled.")
    use_semi_implicit = False

st.sidebar.header("Visualization")
cmap_choice = st.sidebar.selectbox("Matplotlib colormap", CMAPS, index=0)
show_contours = st.sidebar.checkbox("Show Plotly contour overlay (2D)", value=True)

# -----------------------
# Helpers: discrete operators (2D and 3D)
# -----------------------
def make_grid(L=1.0):
    if Nz == 1:
        x = np.linspace(0, L, Nx)
        y = np.linspace(0, L, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return (x, y, X, Y)
    else:
        x = np.linspace(0, L, Nx)
        y = np.linspace(0, L, Ny)
        z = np.linspace(0, L, Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return (x, y, z, X, Y, Z)

# laplacian explicit (returns Laplacian / dx^2)
if NUMBA_AVAILABLE and use_numba:
    @njit(parallel=True)
    def laplacian_explicit_3d(u, dx):
        nx, ny, nz = u.shape
        out = np.zeros_like(u)
        for i in prange(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    out[i,j,k] = (u[i+1,j,k] + u[i-1,j,k] + u[i,j+1,k] + u[i,j-1,k] + u[i,j,k+1] + u[i,j,k-1] - 6*u[i,j,k])
        # edges will be handled outside (Neumann mirror)
        return out / (dx*dx)

    @njit(parallel=True)
    def laplacian_explicit_2d(u, dx):
        nx, ny = u.shape
        out = np.zeros_like(u)
        for i in prange(1, nx-1):
            for j in prange(1, ny-1):
                out[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
        return out / (dx*dx)
else:
    def laplacian_explicit_2d(u, dx):
        out = np.zeros_like(u)
        out[1:-1,1:-1] = (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4*u[1:-1,1:-1])
        # Neumann boundaries (copy neighbor differences)
        out[0,1:-1]   = (u[1,1:-1]   + u[0,2:]   + u[0,:-2]   - 3*u[0,1:-1])
        out[-1,1:-1]  = (u[-2,1:-1]  + u[-1,2:]  + u[-1,:-2]  - 3*u[-1,1:-1])
        out[1:-1,0]   = (u[2:,0]     + u[:-2,0]  + u[1:-1,1]  - 3*u[1:-1,0])
        out[1:-1,-1]  = (u[2:,-1]    + u[:-2,-1] + u[1:-1,-2] - 3*u[1:-1,-1])
        out[0,0] = (u[1,0] + u[0,1] - 2*u[0,0])
        out[0,-1] = (u[0,-2] + u[1,-1] - 2*u[0,-1])
        out[-1,0] = (u[-2,0] + u[-1,1] - 2*u[-1,0])
        out[-1,-1] = (u[-2,-1] + u[-1,-2] - 2*u[-1,-1])
        return out / (dx*dx)

    def laplacian_explicit_3d(u, dx):
        out = np.zeros_like(u)
        out[1:-1,1:-1,1:-1] = (
            u[2:,1:-1,1:-1] + u[:-2,1:-1,1:-1] +
            u[1:-1,2:,1:-1] + u[1:-1,:-2,1:-1] +
            u[1:-1,1:-1,2:] + u[1:-1,1:-1,:-2] - 6*u[1:-1,1:-1,1:-1]
        )
        # boundaries approximate
        return out / (dx*dx)

def grad_mag_2d(u, dx):
    uy = np.zeros_like(u); ux = np.zeros_like(u)
    ux[:,1:-1] = (u[:,2:] - u[:,:-2]) / (2*dx)
    ux[:,0] = (u[:,1] - u[:,0]) / dx
    ux[:,-1] = (u[:,-1] - u[:,-2]) / dx
    uy[1:-1,:] = (u[2:,:] - u[:-2,:]) / (2*dx)
    uy[0,:] = (u[1,:] - u[0,:]) / dx
    uy[-1,:] = (u[-1,:] - u[-2,:]) / dx
    return np.sqrt(ux*ux + uy*uy + 1e-30)

# -----------------------
# Simulation core (2D or 3D)
# -----------------------
def run_simulation_2d(params):
    Nx, Ny = params['Nx'], params['Ny']
    L = 1.0
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx); y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # core psi - spherical core in 2D is circle
    cx = cy = 0.5
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    psi = (dist <= params['core_radius_frac'] * L).astype(np.float64)

    # initial phi shell
    r_core = params['core_radius_frac'] * L
    r_outer = r_core * (1.0 + params['shell_thickness_frac'])
    phi = np.where(dist <= r_core, 0.0, np.where(dist <= r_outer, 1.0, 0.0)).astype(np.float64)
    eps = max(4.0*dx, 1e-6)
    phi = phi * (1.0 - 0.5*(1.0 - np.tanh((dist - r_core)/eps))) * (1.0 - 0.5*(1.0 + np.tanh((dist - r_outer)/eps)))
    phi = np.clip(phi, 0.0, 1.0)

    # concentration
    c = params['c_bulk'] * (Y / L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots = []
    diagnostics = []
    n_steps = params['n_steps']; dt = params['dt']; save_every = params['save_every']
    gamma = params['gamma']; beta = params['beta']; k0 = params['k0']; M = params['M']; D = params['D']; alpha = params['alpha']

    # prepare semi-implicit matrix if requested
    if params['use_semi_implicit'] and SCIPY_AVAILABLE:
        N = Nx * Ny
        main = np.ones(N) * -4.0
        offsets = [0, -1, 1, -Nx, Nx]
        diags = np.zeros((5, N))
        diags[0,:] = main
        # create Laplacian matrix (interior only) with Neumann approx is more involved;
        # we assemble a simple 5-pt Laplacian with row-major ordering and fix boundaries via identity.
        A = sp.lil_matrix((N,N))
        for i in range(Nx):
            for j in range(Ny):
                idx = i*Ny + j
                A[idx, idx] = -4.0
                # neighbors with boundary checks (Neumann mirror)
                for (ii,jj) in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                    if 0 <= ii < Nx and 0 <= jj < Ny:
                        A[idx, ii*Ny + jj] = 1.0
                    else:
                        A[idx, idx] += 1.0  # mirror -> increase diag
        A = A.tocsr()
        # operator for implicit solve: (I - dt * M * gamma * Lap)
        Implicit_mat = sp.eye(N) - (dt * M * gamma) * A
        # we'll reuse a factorization if possible:
        try:
            lu = spla.factorized(Implicit_mat.tocsc())
            has_factor = True
        except Exception:
            has_factor = False
    else:
        has_factor = False

    for step in range(n_steps + 1):
        t = step * dt

        # interface indicator
        gphi = grad_mag_2d(phi, dx)
        delta_int = 6.0 * phi * (1.0 - phi) * (1.0 - psi) * gphi
        delta_int = np.clip(delta_int, 0.0, 6.0 / max(eps, dx))

        # Neumann boundaries (mirror)
        phi[0,:]  = phi[1,:]; phi[-1,:] = phi[-2,:]; phi[:,0] = phi[:,1]; phi[:,-1] = phi[:,-2]

        lap_phi = laplacian_explicit_2d(phi, dx)
        f_bulk = 2.0 * beta * phi * (1 - phi) * (1 - 2 * phi)
        c_mol = c * (1.0 - phi) * (1.0 - psi)
        i_loc = k0 * c_mol * delta_int
        i_loc = np.clip(i_loc, 0.0, 1e6)

        deposition = M * i_loc
        # curvature smoothing term (implicit handled below optionally)
        curvature_explicit = M * gamma * lap_phi

        # explicit update for deposition and bulk
        phi_temp = phi + dt * (deposition - M * f_bulk)

        # semi-implicit step for curvature: solve (I - dt*M*gamma*Lap) phi_new = phi_temp
        if has_factor:
            b = phi_temp.reshape(-1)
            phi_new_flat = lu(b)
            phi = phi_new_flat.reshape(phi.shape)
        else:
            # explicit fallback
            phi = phi_temp + dt * curvature_explicit
            phi = np.clip(phi, 0.0, 1.0)

        phi = np.clip(phi, 0.0, 1.0)

        # concentration diffusion + sink
        lap_c = laplacian_explicit_2d(c, dx)
        sink = i_loc
        c += dt * (D * lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk'] * 5.0)
        # BCs for c: top reservoir (y = max) -> using last column as top
        c[:, -1] = params['c_bulk']
        c[:, 0] = c[:, 1]
        c[0, :] = c[1, :]
        c[-1,:] = c[-2,:]

        # diagnostics
        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm_raw = np.sqrt(np.mean((M * gamma * lap_phi)**2))
        grad_norm_phys = np.sqrt(np.mean(((M * gamma * lap_phi) * (dx*dx))**2))
        alpha_c_norm = alpha * np.mean(c)

        if step % save_every == 0 or step == n_steps:
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm))

    return snapshots, diagnostics, (x, y)

def run_simulation_3d(params):
    # A simple 3D spherical variant. Keep grid sizes small for interactive runs.
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    L = 1.0
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx); y = np.linspace(0, L, Ny); z = np.linspace(0, L, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # core psi: sphere
    cx = cy = cz = 0.5
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
    psi = (dist <= params['core_radius_frac'] * L).astype(np.float64)

    # phi initial shell
    r_core = params['core_radius_frac'] * L
    r_outer = r_core * (1.0 + params['shell_thickness_frac'])
    phi = np.where(dist <= r_core, 0.0, np.where(dist <= r_outer, 1.0, 0.0)).astype(np.float64)
    eps = max(4.0*dx, 1e-6)
    phi = phi * (1.0 - 0.5*(1.0 - np.tanh((dist - r_core)/eps))) * (1.0 - 0.5*(1.0 + np.tanh((dist - r_outer)/eps)))
    phi = np.clip(phi, 0.0, 1.0)

    # concentration (z-direction acts as "height")
    c = params['c_bulk'] * (Z / L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots = []
    diagnostics = []
    n_steps = params['n_steps']; dt = params['dt']; save_every = params['save_every']
    gamma = params['gamma']; beta = params['beta']; k0 = params['k0']; M = params['M']; D = params['D']; alpha = params['alpha']

    for step in range(n_steps + 1):
        t = step * dt
        # 3D laplacian (explicit)
        lap_phi = laplacian_explicit_3d(phi, dx)
        # gradient magnitude for interface indicator: central differences approx
        # use simple approximate grad magnitude
        # For speed we use np.gradient (not numba)
        gx, gy, gz = np.gradient(phi, dx, edge_order=2)
        gphi = np.sqrt(gx*gx + gy*gy + gz*gz + 1e-30)
        delta_int = 6.0 * phi * (1.0 - phi) * (1.0 - psi) * gphi
        delta_int = np.clip(delta_int, 0.0, 6.0 / max(eps, dx))

        # boundaries (mirror)
        phi[0,:,:] = phi[1,:,:]; phi[-1,:,:] = phi[-2,:,:]
        phi[:,0,:] = phi[:,1,:]; phi[:,-1,:] = phi[:,-2,:]
        phi[:,:,0] = phi[:,:,1]; phi[:,:,-1] = phi[:,:,-2]

        f_bulk = 2.0 * beta * phi * (1 - phi) * (1 - 2 * phi)
        c_mol = c * (1.0 - phi) * (1.0 - psi)
        i_loc = k0 * c_mol * delta_int
        i_loc = np.clip(i_loc, 0.0, 1e6)
        deposition = M * i_loc
        curvature = M * gamma * lap_phi

        phi += dt * (deposition + curvature - M * f_bulk)
        phi = np.clip(phi, 0.0, 1.0)

        # concentration
        lap_c = laplacian_explicit_3d(c, dx)
        sink = i_loc
        c += dt * (D * lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk'] * 5.0)
        c[:, -1, :] = params['c_bulk']

        # diagnostics
        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm_raw = np.sqrt(np.mean((M * gamma * lap_phi)**2))
        grad_norm_phys = np.sqrt(np.mean(((M * gamma * lap_phi) * (dx*dx))**2))
        alpha_c_norm = alpha * np.mean(c)

        if step % save_every == 0 or step == n_steps:
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm))

    return snapshots, diagnostics, (x, y, z)

# -----------------------
# UI: run controls
# -----------------------
st.sidebar.header("Core & shell geometry")
core_radius_frac = st.sidebar.slider("Core radius (fraction of L)", 0.05, 0.45, 0.18, 0.01)
shell_thickness_frac = st.sidebar.slider("Shell thickness (Δr / r_core)", 0.05, 0.6, 0.2, 0.01)

run_button = st.sidebar.button("Run Simulation ▶")
export_vtu_button = st.sidebar.button("Export VTU/PVD/ZIP")
download_diags_button = st.sidebar.button("Download diagnostics CSV")

# pack params
params = {
    'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
    'dt': dt, 'n_steps': n_steps, 'save_every': save_every,
    'gamma': gamma, 'beta': beta, 'k0': k0, 'M': M, 'alpha': alpha, 'c_bulk': c_bulk, 'D': D,
    'core_radius_frac': core_radius_frac, 'shell_thickness_frac': shell_thickness_frac,
    'use_semi_implicit': use_semi_implicit, 'use_numba': use_numba
}

# storage in session_state
if "snapshots" not in st.session_state:
    st.session_state.snapshots = None
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = None
if "grid_coords" not in st.session_state:
    st.session_state.grid_coords = None

if run_button:
    t0 = time.time()
    st.info("Running simulation — this may take a few seconds...")
    if mode.startswith("2D"):
        snapshots, diagnostics, coords = run_simulation_2d({**params, 'n_steps': n_steps, 'dt': dt, 'save_every': save_every, 'gamma': gamma, 'beta': beta, 'k0':k0, 'M':M, 'D':D, 'alpha':alpha, 'c_bulk':c_bulk, 'Nx':Nx, 'Ny':Ny, 'core_radius_frac':core_radius_frac, 'shell_thickness_frac':shell_thickness_frac, 'use_semi_implicit':use_semi_implicit})
    else:
        snapshots, diagnostics, coords = run_simulation_3d({**params, 'n_steps': n_steps, 'dt': dt, 'save_every': save_every, 'gamma': gamma, 'beta': beta, 'k0':k0, 'M':M, 'D':D, 'alpha':alpha, 'c_bulk':c_bulk, 'Nx':Nx, 'Ny':Ny, 'Nz':Nz, 'core_radius_frac':core_radius_frac, 'shell_thickness_frac':shell_thickness_frac})
    st.session_state.snapshots = snapshots
    st.session_state.diagnostics = diagnostics
    st.session_state.grid_coords = coords
    st.success(f"Simulation finished in {time.time() - t0:.2f}s — frames saved: {len(snapshots)}")

# -----------------------
# Visualization & playback
# -----------------------
if st.session_state.snapshots:
    snapshots = st.session_state.snapshots
    diagnostics = st.session_state.diagnostics
    coords = st.session_state.grid_coords

    st.header("Results & Playback")

    cols = st.columns([3,1])
    with cols[0]:
        frame_idx = st.slider("Frame", 0, len(snapshots)-1, len(snapshots)-1)
        auto_play = st.checkbox("Autoplay", value=False)
        autoplay_interval = st.number_input("Autoplay interval (s)", 0.1, 5.0, 0.4, 0.1)

        # choose field
        field = st.selectbox("Field to display", ["phi (shell)", "c (concentration)", "psi (core)"])
        cmap = plt.get_cmap(cmap_choice)

        t, phi_view, c_view, psi_view = snapshots[frame_idx]

        if mode.startswith("2D"):
            # show 2D heatmap with chosen cmap
            fig, ax = plt.subplots(figsize=(6,5))
            if field == "phi (shell)":
                im = ax.imshow(phi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap)
            elif field == "c (concentration)":
                im = ax.imshow(c_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap)
            else:
                im = ax.imshow(psi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap)
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{field} at t*={t:.5f}")
            st.pyplot(fig)

            # midline profile
            mid = phi_view.shape[0] // 2
            fig2, ax2 = plt.subplots(figsize=(6,2.2))
            if field == "phi (shell)":
                ax2.plot(np.linspace(0,1,phi_view.shape[1]), phi_view[mid,:], label='phi')
            elif field == "c (concentration)":
                ax2.plot(np.linspace(0,1,c_view.shape[1]), c_view[mid,:], label='c')
            else:
                ax2.plot(np.linspace(0,1,psi_view.shape[1]), psi_view[mid,:], label='psi')
            ax2.set_xlabel("y / L"); ax2.legend(); ax2.grid(True)
            st.pyplot(fig2)

        else:
            # 3D: show three orthogonal slices using matplotlib
            fig, axes = plt.subplots(1,3, figsize=(12,4))
            # pick center index
            cx = phi_view.shape[0]//2; cy = phi_view.shape[1]//2; cz = phi_view.shape[2]//2
            if field == "phi (shell)":
                axes[0].imshow(phi_view[cx,:,:].T, origin='lower', cmap=cmap); axes[0].set_title("slice x=center")
                axes[1].imshow(phi_view[:,cy,:].T, origin='lower', cmap=cmap); axes[1].set_title("slice y=center")
                axes[2].imshow(phi_view[:,:,cz].T, origin='lower', cmap=cmap); axes[2].set_title("slice z=center")
            elif field == "c (concentration)":
                axes[0].imshow(c_view[cx,:,:].T, origin='lower', cmap=cmap); axes[0].set_title("slice x=center")
                axes[1].imshow(c_view[:,cy,:].T, origin='lower', cmap=cmap); axes[1].set_title("slice y=center")
                axes[2].imshow(c_view[:,:,cz].T, origin='lower', cmap=cmap); axes[2].set_title("slice z=center")
            else:
                axes[0].imshow(psi_view[cx,:,:].T, origin='lower', cmap=cmap); axes[0].set_title("slice x=center")
                axes[1].imshow(psi_view[:,cy,:].T, origin='lower', cmap=cmap); axes[1].set_title("slice y=center")
                axes[2].imshow(psi_view[:,:,cz].T, origin='lower', cmap=cmap); axes[2].set_title("slice z=center")
            for ax in axes: ax.axis('off')
            st.pyplot(fig)

        # autoplay control (simple)
        if auto_play:
            for idx in range(frame_idx, len(snapshots)):
                time.sleep(autoplay_interval)
                st.session_state._rerun = True  # hint to rerun (works in some environments)

    with cols[1]:
        st.subheader("Diagnostics (latest frame)")
        # show latest diagnostics table
        import pandas as pd
        df = pd.DataFrame(diagnostics, columns=["t*", "||bulk||₂", "||grad||₂ (raw)", "||grad||₂ (scaled)", "α·mean(c)"])
        st.dataframe(df.tail(20).style.format("{:.3e}"))

        # quick plot
        fig3, ax3 = plt.subplots(figsize=(4,3))
        times = df["t*"].values
        ax3.semilogy(times, np.maximum(df["||bulk||₂"].values,1e-30), label='bulk')
        ax3.semilogy(times, np.maximum(df["||grad||₂ (raw)"].values,1e-30), label='grad raw')
        ax3.semilogy(times, np.maximum(df["||grad||₂ (scaled)"].values,1e-30), label='grad scaled')
        ax3.semilogy(times, np.maximum(df["α·mean(c)"].values,1e-30), label='α·c')
        ax3.legend(fontsize=8); ax3.grid(True)
        st.pyplot(fig3)

    # -----------------------
    # Export: CSV diagnostics / PNG snapshot / VTU/PVD/ZIP
    # -----------------------
    # Diagnostics CSV
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["t*", "||bulk||2", "||grad||2_raw", "||grad||2_scaled", "alpha_mean_c"])
    writer.writerows(diagnostics)
    csv_data = csv_buffer.getvalue().encode()

    st.download_button("Download diagnostics CSV", csv_data, file_name=f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    # PNG snapshot of current frame
    img_buf = io.BytesIO()
    # draw a simple PNG of the current display
    if mode.startswith("2D"):
        fig_snap, ax_snap = plt.subplots(figsize=(5,4))
        if field == "phi (shell)":
            ax_snap.imshow(phi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap_choice)
        elif field == "c (concentration)":
            ax_snap.imshow(c_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap_choice)
        else:
            ax_snap.imshow(psi_view.T, origin='lower', extent=[0,1,0,1], cmap=cmap_choice)
        ax_snap.set_title(f"{field} t*={t:.5f}")
        plt.colorbar(ax_snap.images[0], ax=ax_snap)
        fig_snap.tight_layout()
        fig_snap.savefig(img_buf, format='png', dpi=150); plt.close(fig_snap)
    else:
        # 3-slice PNG
        fig_snap, axes_snap = plt.subplots(1,3,figsize=(10,3))
        cx = phi_view.shape[0]//2; cy = phi_view.shape[1]//2; cz = phi_view.shape[2]//2
        if field == "phi (shell)":
            axes_snap[0].imshow(phi_view[cx,:,:].T, origin='lower', cmap=cmap_choice); axes_snap[0].set_title("x slice")
            axes_snap[1].imshow(phi_view[:,cy,:].T, origin='lower', cmap=cmap_choice); axes_snap[1].set_title("y slice")
            axes_snap[2].imshow(phi_view[:,:,cz].T, origin='lower', cmap=cmap_choice); axes_snap[2].set_title("z slice")
        elif field == "c (concentration)":
            axes_snap[0].imshow(c_view[cx,:,:].T, origin='lower', cmap=cmap_choice); axes_snap[1].imshow(c_view[:,cy,:].T, origin='lower', cmap=cmap_choice); axes_snap[2].imshow(c_view[:,:,cz].T, origin='lower', cmap=cmap_choice)
        else:
            axes_snap[0].imshow(psi_view[cx,:,:].T, origin='lower', cmap=cmap_choice); axes_snap[1].imshow(psi_view[:,cy,:].T, origin='lower', cmap=cmap_choice); axes_snap[2].imshow(psi_view[:,:,cz].T, origin='lower', cmap=cmap_choice)
        for ax in axes_snap: ax.axis('off')
        fig_snap.tight_layout()
        fig_snap.savefig(img_buf, format='png', dpi=150); plt.close(fig_snap)
    img_buf.seek(0)
    st.download_button("Download current snapshot (PNG)", img_buf, file_name=f"snapshot_t{t:.5f}.png", mime="image/png")

    # Export VTU / PVD / ZIP (if meshio available) when requested
    if export_vtu_button:
        if not MESHIO_AVAILABLE:
            st.error("meshio not installed; cannot export VTU/PVD. Install `meshio` to enable VTU export.")
        else:
            tmpdir = tempfile.mkdtemp()
            vtus = []
            for idx, (tframe, phi_s, c_s, psi_s) in enumerate(snapshots):
                fname = os.path.join(tmpdir, f"frame_{idx:04d}.vtu")
                # build simple point coordinates and cell (structured grid -> write as point data only)
                # For simplicity we write a rectilinear point cloud with 'meshio' as unstructured points
                # Note: VTU for structured grids is more elaborate; this is a pragmatic export.
                if mode.startswith("2D"):
                    xv, yv = coords
                    Xg, Yg = np.meshgrid(xv, yv, indexing='ij')
                    points = np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros(Xg.size)])
                    # write point data arrays
                    point_data = {
                        "phi": phi_s.ravel().astype(np.float32),
                        "c": c_s.ravel().astype(np.float32),
                        "psi": psi_s.ravel().astype(np.float32)
                    }
                    meshio.write_points_cells(fname, points, [], point_data=point_data)
                else:
                    xv, yv, zv = coords
                    Xg, Yg, Zg = np.meshgrid(xv, yv, zv, indexing='ij')
                    points = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
                    point_data = {
                        "phi": phi_s.ravel().astype(np.float32),
                        "c": c_s.ravel().astype(np.float32),
                        "psi": psi_s.ravel().astype(np.float32)
                    }
                    meshio.write_points_cells(fname, points, [], point_data=point_data)
                vtus.append(fname)
            # create a simple .pvd (XML) pointing to the vtus
            pvd_path = os.path.join(tmpdir, "collection.pvd")
            with open(pvd_path, "w") as f:
                f.write("<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
                f.write("  <Collection>\n")
                for idx, v in enumerate(vtus):
                    fname = os.path.basename(v)
                    f.write(f"    <DataSet timestep=\"{idx}\" group=\"\" part=\"0\" file=\"{fname}\"/>\n")
                f.write("  </Collection>\n")
                f.write("</VTKFile>\n")
            # zip them
            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in vtus:
                    zf.write(p, arcname=os.path.basename(p))
                zf.write(pvd_path, arcname=os.path.basename(pvd_path))
            zipbuf.seek(0)
            st.download_button("Download VTU/PVD ZIP", zipbuf.read(), file_name=f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")

    # Download single VTU for selected frame
    if MESHIO_AVAILABLE:
        sel_frame_for_vtu = st.number_input("Select frame to save as single VTU", 0, len(snapshots)-1, 0)
        if st.button("Download selected frame as VTU"):
            tframe, phi_s, c_s, psi_s = snapshots[int(sel_frame_for_vtu)]
            # write to buffer
            buf = io.BytesIO()
            fname = f"frame_{int(sel_frame_for_vtu):04d}.vtu"
            if mode.startswith("2D"):
                xv, yv = coords
                Xg, Yg = np.meshgrid(xv, yv, indexing='ij')
                points = np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros(Xg.size)])
                point_data = {"phi": phi_s.ravel().astype(np.float32), "c": c_s.ravel().astype(np.float32), "psi": psi_s.ravel().astype(np.float32)}
                meshio.write_points_cells(buf, points, [], file_format="vtu", point_data=point_data)
            else:
                xv, yv, zv = coords
                Xg, Yg, Zg = np.meshgrid(xv, yv, zv, indexing='ij')
                points = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
                point_data = {"phi": phi_s.ravel().astype(np.float32), "c": c_s.ravel().astype(np.float32), "psi": psi_s.ravel().astype(np.float32)}
                meshio.write_points_cells(buf, points, [], file_format="vtu", point_data=point_data)
            buf.seek(0)
            st.download_button("Download VTU (selected frame)", buf.read(), file_name=f"{fname}", mime="application/octet-stream")

else:
    st.info("Run a simulation to see results. Tip: keep 3D grid small (<= 40) for interactive runs.")

# -----------------------
# End of app
# -----------------------
