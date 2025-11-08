"""
streamlit_electroless_enhanced.py
Features:
 - 2D / 3D electroless deposition shell growth (simple physics)
 - Numba acceleration if available
 - Optional semi-implicit IMEX solver using scipy.sparse (fallback to explicit)
 - Animation slider + autoplay
 - Export diagnostics CSV, PNG snapshots, VTU/PVD outputs, and zipped VTUs
 - Selectable Matplotlib colormap from a large list (~50)
 - NEW : post-processor that merges phi/psi into a single material field
   and visualises the electric-potential proxy -α·c
   → now includes h·(4·phi² + 2·psi²) with a live h-slider
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

# ------------------- sidebar -------------------
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
cmap_choice = st.sidebar.selectbox("Matplotlib colormap", CMAPS, index=CMAPS.index("viridis"))

# ------------------- geometry -------------------
st.sidebar.header("Core & shell geometry")
core_radius_frac = st.sidebar.slider("Core radius (fraction of L)", 0.05, 0.45, 0.18, 0.01)
shell_thickness_frac = st.sidebar.slider("Shell thickness (Δr / r_core)", 0.05, 0.6, 0.2, 0.01)

run_button = st.sidebar.button("Run Simulation")
export_vtu_button = st.sidebar.button("Export VTU/PVD/ZIP")
download_diags_button = st.sidebar.button("Download diagnostics CSV")

# ------------------- operators -------------------
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
    uy[0,:] = (u[1,:] - u[0,:]) / dx
    uy[-1,:] = (u[-1,:] - u[-2,:]) / dx
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

    c = params['c_bulk'] * (Y/L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots, diagnostics = [], []
    n_steps = params['n_steps']; dt = params['dt']; save_every = params['save_every']
    gamma = params['gamma']; beta = params['beta']; k0 = params['k0']
    M = params['M']; D = params['D']; alpha = params['alpha']

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

    for step in range(n_steps+1):
        t = step*dt
        gphi = grad_mag_2d(phi, dx)
        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps,dx))

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

        lap_c = laplacian_explicit_2d(c, dx)
        sink = i_loc
        c += dt*(D*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk']*5.0)
        c[:, -1] = params['c_bulk']

        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm_raw = np.sqrt(np.mean((M*gamma*lap_phi)**2))
        grad_norm_phys = np.sqrt(np.mean(((M*gamma*lap_phi)*(dx*dx))**2))
        alpha_c_norm = alpha*np.mean(c)

        if step % save_every == 0 or step == n_steps:
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm))
    return snapshots, diagnostics, (x, y)


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

    c = params['c_bulk'] * (Z/L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])

    snapshots, diagnostics = [], []
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
        c[:, -1, :] = params['c_bulk']

        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm_raw = np.sqrt(np.mean((M*gamma*lap_phi)**2))
        grad_norm_phys = np.sqrt(np.mean(((M*gamma*lap_phi)*(dx*dx))**2))
        alpha_c_norm = alpha*np.mean(c)

        if step % save_every == 0 or step == n_steps:
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm))
    return snapshots, diagnostics, (x, y, z)

# ------------------- run -------------------
params = {
    'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
    'dt': dt, 'n_steps': n_steps, 'save_every': save_every,
    'gamma': gamma, 'beta': beta, 'k0': k0, 'M': M,
    'alpha': alpha, 'c_bulk': c_bulk, 'D': D,
    'core_radius_frac': core_radius_frac,
    'shell_thickness_frac': shell_thickness_frac,
    'use_semi_implicit': use_semi_implicit,
    'use_numba': use_numba
}

if "snapshots" not in st.session_state:
    st.session_state.snapshots = None
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = None
if "grid_coords" not in st.session_state:
    st.session_state.grid_coords = None

if run_button:
    t0 = time.time()
    st.info("Running simulation …")
    if mode.startswith("2D"):
        snapshots, diagnostics, coords = run_simulation_2d({**params})
    else:
        snapshots, diagnostics, coords = run_simulation_3d({**params})
    st.session_state.snapshots = snapshots
    st.session_state.diagnostics = diagnostics
    st.session_state.grid_coords = coords
    st.success(f"Done in {time.time()-t0:.2f}s — {len(snapshots)} frames")

# ------------------- playback -------------------
if st.session_state.snapshots:
    snapshots = st.session_state.snapshots
    diagnostics = st.session_state.diagnostics
    coords = st.session_state.grid_coords

    st.header("Results & Playback")
    cols = st.columns([3,1])
    with cols[0]:
        frame_idx = st.slider("Frame", 0, len(snapshots)-1, len(snapshots)-1)
        auto_play = st.checkbox("Autoplay", value=False)
        autoplay_interval = st.number_input("Interval (s)", 0.1, 5.0, 0.4, 0.1)

        field = st.selectbox("Field", ["phi (shell)", "c (concentration)", "psi (core)"])
        t, phi_view, c_view, psi_view = snapshots[frame_idx]
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
            ax.set_title(f"{field} @ t*={t:.5f}")
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
    
    
    
    # --------------------------------------------------------------
    # POST-PROCESSOR : material field + electric-potential proxy
    # --------------------------------------------------------------
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
    potential = -alpha * c_view

    # ---------- Define colormap logic ONCE (for 2D and 3D) ----------
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
        ax_mat.set_title(f"Material @ t* = {t:.5f}")
        st.pyplot(fig_mat)

        if show_potential:
            fig_pot, ax_pot = plt.subplots(figsize=(6,5))
            im_pot = ax_pot.imshow(potential.T, origin='lower', extent=[0,1,0,1],
                                   cmap="RdBu_r")
            plt.colorbar(im_pot, ax=ax_pot, label="Potential proxy -α·c")
            ax_pot.set_title(f"Potential proxy @ t* = {t:.5f}")
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
            ax.set_title(label)
            ax.axis('off')
        fig_mat.suptitle(f"Material (3-D slices) @ t* = {t:.5f}")
        st.pyplot(fig_mat)

        if show_potential:
            fig_pot, axes = plt.subplots(1,3, figsize=(12,4))
            for ax, sl, label in zip(axes,
                                     [potential[cx,:,:], potential[:,cy,:], potential[:,:,cz]],
                                     ["x-slice","y-slice","z-slice"]):
                im = ax.imshow(sl.T, origin='lower', cmap="RdBu_r")
                ax.set_title(label)
                ax.axis('off')
            fig_pot.suptitle(f"Potential proxy (-α·c) @ t* = {t:.5f}")
            plt.colorbar(im, ax=axes, orientation='horizontal',
                         fraction=0.05, label="-α·c")
            st.pyplot(fig_pot)
    
    # ------------------- diagnostics export -------------------
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["t*","||bulk||2","||grad||2_raw","||grad||2_scaled","alpha_mean_c"])
    writer.writerows(diagnostics)
    st.download_button("Download diagnostics CSV",
                       csv_buffer.getvalue().encode(),
                       file_name=f"diagnostics_{datetime.now():%Y%m%d_%H%M%S}.csv",
                       mime="text/csv")

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
        ax_snap.set_title(f"{field} t*={t:.5f}")
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
        fig_snap.tight_layout()
        fig_snap.savefig(img_buf, format='png', dpi=150); plt.close(fig_snap)
    img_buf.seek(0)
    st.download_button("Download current snapshot (PNG)",
                       img_buf,
                       file_name=f"snapshot_t{t:.5f}.png",
                       mime="image/png")

    # ------------------- VTU / PVD / ZIP -------------------
    if export_vtu_button:
        if not MESHIO_AVAILABLE:
            st.error("`meshio` not installed — VTU export disabled.")
        else:
            tmpdir = tempfile.mkdtemp()
            vtus = []
            for idx, (tframe, phi_s, c_s, psi_s) in enumerate(snapshots):
                fname = os.path.join(tmpdir, f"frame_{idx:04d}.vtu")
                if mode.startswith("2D"):
                    xv, yv = coords
                    Xg, Yg = np.meshgrid(xv, yv, indexing='ij')
                    points = np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros_like(Xg.ravel())])
                else:
                    xv, yv, zv = coords
                    Xg, Yg, Zg = np.meshgrid(xv, yv, zv, indexing='ij')
                    points = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

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
                f.write("  <Collection>\n")
                for idx, v in enumerate(vtus):
                    f.write(f'    <DataSet timestep="{idx}" file="{os.path.basename(v)}"/>\n')
                f.write("  </Collection>\n")
                f.write("</VTKFile>\n")

            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in vtus:
                    zf.write(p, arcname=os.path.basename(p))
                zf.write(pvd_path, arcname=os.path.basename(pvd_path))
            zipbuf.seek(0)
            st.download_button("Download VTU/PVD ZIP",
                               zipbuf.read(),
                               file_name=f"frames_{datetime.now():%Y%m%d_%H%M%S}.zip",
                               mime="application/zip")

    if MESHIO_AVAILABLE:
        sel = st.number_input("Select frame for single VTU", 0, len(snapshots)-1, 0)
        if st.button("Download selected frame as VTU"):
            tframe, phi_s, c_s, psi_s = snapshots[int(sel)]
            buf = io.BytesIO()
            if mode.startswith("2D"):
                xv, yv = coords
                Xg, Yg = np.meshgrid(xv, yv, indexing='ij')
                points = np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros_like(Xg.ravel())])
            else:
                xv, yv, zv = coords
                Xg, Yg, Zg = np.meshgrid(xv, yv, zv, indexing='ij')
                points = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

            mat_s = build_material(phi_s, psi_s, material_method, h=h_factor)
            point_data = {
                "phi": phi_s.ravel().astype(np.float32),
                "c": c_s.ravel().astype(np.float32),
                "psi": psi_s.ravel().astype(np.float32),
                "material": mat_s.ravel().astype(np.float32)
            }
            meshio.write_points_cells(buf, points, [], file_format="vtu", point_data=point_data)
            buf.seek(0)
            st.download_button("Download VTU",
                               buf.read(),
                               file_name=f"frame_{int(sel):04d}.vtu",
                               mime="application/octet-stream")
else:
    st.info("Run a simulation to see results. Tip: keep 3D grid ≤ 40 for fast runs.")
