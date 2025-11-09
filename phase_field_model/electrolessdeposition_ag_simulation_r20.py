#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Electroless Ag — FULL BATCH SIMULATOR
All features:
  • Bulk vs. gradient norms
  • Material + potential proxy
  • Multi-run history + combined plots
  • VTU/PVD/ZIP export
  • GIF animation
  • PNG snapshot, CSV, autoplay
  • BATCH SIMULATION (parallel)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import io
import zipfile
import os
import tempfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------- OPTIONAL LIBS -------------------
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

try:
    import meshio
    MESHIO_AVAILABLE = True
except Exception:
    MESHIO_AVAILABLE = False

try:
    import imageio
    from matplotlib.animation import FuncAnimation
    GIF_AVAILABLE = True
except Exception:
    GIF_AVAILABLE = False

# ------------------- PAGE -------------------
st.set_page_config(page_title="Electroless Ag — Batch Simulator", layout="wide")
st.title("Electroless Ag — Batch + Full Simulator")

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 140, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
})
CMAPS = [c for c in plt.colormaps() if c not in {"jet", "jet_r"}]

# ------------------- SIDEBAR -------------------
st.sidebar.header("Batch c_bulk Values")
c_options = {
    "1.0 (1:1)": 1.0, "0.5 (1:2)": 0.5, "0.333 (1:3)": 0.333,
    "0.25 (1:4)": 0.25, "0.2 (1:5)": 0.2, "0.1 (1:10)": 0.1
}
selected_labels = st.sidebar.multiselect(
    "Choose c_bulk values for batch run",
    list(c_options.keys()),
    default=["1.0 (1:1)", "0.5 (1:2)", "0.2 (1:5)"]
)
c_bulk_list = [c_options[l] for l in selected_labels]

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
D_nd     = st.sidebar.slider("D", 0.0, 1.0, 0.05, 0.005)
alpha_nd = st.sidebar.slider("α (coupling)", 0.0, 10.0, 2.0, 0.1)

use_numba = st.sidebar.checkbox("Numba", NUMBA_AVAILABLE)
cmap_choice = st.sidebar.selectbox("Colormap", CMAPS, CMAPS.index("viridis"))

st.sidebar.header("Geometry")
core_radius_frac = st.sidebar.slider("Core / L", 0.05, 0.45, 0.18, 0.01)
shell_thickness_frac = st.sidebar.slider("Δr / r_core", 0.05, 0.6, 0.2, 0.01)
phi_threshold = st.sidebar.slider("φ threshold", 0.1, 0.9, 0.5, 0.05)

growth_model = st.sidebar.selectbox("Model", ["Model A (irreversible)", "Model B (soft reversible)"])

st.sidebar.header("Physical Scales")
L0 = st.sidebar.number_input("L₀ (nm)", 1.0, 1e6, 20.0) * 1e-9
tau0 = st.sidebar.number_input("τ₀ (×10⁻⁴ s)", 1e-6, 1e6, 1.0) * 1e-4

run_batch_button = st.sidebar.button("Run BATCH Simulation")
run_single_button = st.sidebar.button("Run SINGLE (last selected c)")

# ------------------- HELPERS -------------------
def scale_time(t): return t * tau0
def nd_to_real(x): return x * L0

# ------------------- NUMBA LAPLACIANS -------------------
if NUMBA_AVAILABLE and use_numba:
    @njit(parallel=True)
    def lap2d(u, dx):
        n, m = u.shape
        out = np.zeros((n, m), dtype=u.dtype)
        for i in prange(1, n-1):
            for j in prange(1, m-1):
                out[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / (dx*dx)
        return out

    @njit(parallel=True)
    def lap3d(u, dx):
        n, m, p = u.shape
        out = np.zeros((n, m, p), dtype=u.dtype)
        for i in prange(1, n-1):
            for j in prange(1, m-1):
                for k in prange(1, p-1):
                    out[i,j,k] = (u[i+1,j,k] + u[i-1,j,k] +
                                  u[i,j+1,k] + u[i,j-1,k] +
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

# ------------------- THICKNESS -------------------
def compute_thickness(phi, psi, coords, core_frac, thresh):
    x, y, z = coords
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
    mask = (phi > thresh) & (psi < 0.5)
    return np.max(dist[mask]) - core_frac if np.any(mask) else 0.0

# ------------------- SIMULATION CORE -------------------
def run_simulation(c_bulk_val):
    L = 1.0; dx = L/(Nx-1)
    x = np.linspace(0, L, Nx)
    if mode == "2D (planar)":
        y = np.linspace(0, L, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
        lap_phi_func = lap2d
        lap_c_func = lap2d
        coords = (x, y)
        grad_axis = (0, 1)
    else:
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
        lap_phi_func = lap3d
        lap_c_func = lap3d
        coords = (x, x, x)
        grad_axis = (0, 1, 2)

    psi = (dist <= core_radius_frac*L).astype(np.float64)
    r_core = core_radius_frac * L
    r_outer = r_core * (1 + shell_thickness_frac)
    phi = np.where(dist <= r_core, 0.0, np.where(dist <= r_outer, 1.0, 0.0))
    eps = max(4*dx, 1e-6)
    phi = phi * (1 - 0.5*(1 - np.tanh((dist-r_core)/eps))) \
              * (1 - 0.5*(1 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0, 1)

    c = c_bulk_val * (Y/L if mode == "2D (planar)" else Z/L) * (1 - phi) * (1 - psi) if bc_type == "Neumann (zero flux)" else \
        c_bulk_val * (1 - phi) * (1 - psi)
    c = np.clip(c, 0, c_bulk_val)

    snapshots, diags, thick = [], [], []
    softness = 0.01 if "B" in growth_model else 0.0
    max_th = 0.0

    for step in range(n_steps + 1):
        t = step * dt_nd

        grad = np.gradient(phi, dx, axis=grad_axis)
        gphi = np.sqrt(sum(g**2 for g in grad) + 1e-30) if mode != "2D (planar)" else np.sqrt(grad[0]**2 + grad[1]**2 + 1e-30)

        delta_int = 6*phi*(1-phi)*(1-psi)*gphi
        delta_int = np.clip(delta_int, 0, 6/max(eps,dx))
        f_bulk = 2 * beta_nd * phi * (1 - phi) * (1 - 2 * phi)
        i_loc = k0_nd * c * (1-phi) * (1-psi) * delta_int
        i_loc = np.clip(i_loc, 0, 1e6)

        lap_phi = lap_phi_func(phi, dx)
        dep = M_nd * i_loc
        curv = M_nd * gamma_nd * lap_phi
        dphi = dt_nd * (dep + np.maximum(curv, 0) - softness * M_nd * f_bulk)
        if softness == 0: dphi = np.maximum(dphi, 0)
        phi = np.clip(phi + dphi, 0, 1)

        lap_c = lap_c_func(c, dx)
        c += dt_nd * (D_nd * lap_c - i_loc)
        c = np.clip(c, 0, c_bulk_val)
        if bc_type == "Neumann (zero flux)":
            if mode == "2D (planar)":
                c[:,-1] = c_bulk_val
            else:
                c[:,:,-1] = c_bulk_val
        else:
            if mode == "2D (planar)":
                c[[0,-1],:] = c[:,[0,-1]] = c_bulk_val
            else:
                c[[0,-1],:,:] = c[:,[0,-1],:] = c[:,:,[0,-1]] = c_bulk_val

        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm = np.sqrt(np.mean((M_nd*gamma_nd*lap_phi)**2))

        if step % save_every == 0 or step == n_steps:
            th_nd = compute_thickness(phi, psi, coords + (np.array([0.5]),) if mode == "2D (planar)" else coords, core_radius_frac, phi_threshold)
            max_th = max(max_th, th_nd)
            c_mean, c_max = np.mean(c), np.max(c)
            total_ag = np.sum(i_loc) * dt_nd

            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            #diags.append((t, c_mean, c_max, total_ag, bulk_norm, grad_norm))
            diags.append((t, c_mean, c_max, total_ag, float(bulk_norm), float(grad_norm)))
            thick.append((t, max_th, nd_to_real(max_th), c_mean, c_max, total_ag))

    return c_bulk_val, snapshots, diags, thick, coords

# ------------------- HISTORY -------------------
if "history" not in st.session_state:
    st.session_state.history = {}
if "selected_c" not in st.session_state:
    st.session_state.selected_c = None

# ------------------- BATCH RUN -------------------
if run_batch_button and c_bulk_list:
    if len(c_bulk_list) == 0:
        st.warning("Select  Select at least one c_bulk value.")
    else:
        with st.spinner(f"Running BATCH of {len(c_bulk_list)} simulations..."):
            results = []
            with ProcessPoolExecutor() as executor:
                future_to_c = {executor.submit(run_simulation, c): c for c in c_bulk_list}
                for future in as_completed(future_to_c):
                    c_val = future_to_c[future]
                    try:
                        c_bulk_val, snaps, diag, thick, coords = future.result()
                        results.append((c_bulk_val, snaps, diag, thick, coords))
                    except Exception as e:
                        st.error(f"Simulation failed for c = {c_val}: {e}")
            for c_bulk_val, snaps, diag, thick, coords in results:
                st.session_state.history[c_bulk_val] = {
                    "snaps": snaps, "diag": diag, "thick": thick, "coords": coords
                }
            if results:
                st.session_state.selected_c = results[0][0]
            st.success(f"Batch completed: {len(results)} runs")

# ------------------- SINGLE RUN -------------------
if run_single_button and selected_labels:
    c_bulk_val = c_options[selected_labels[-1]]
    with st.spinner(f"Running SINGLE c = {c_bulk_val}..."):
        t0 = time.time()
        c_bulk_val, snaps, diag, thick, coords = run_simulation(c_bulk_val)
        st.session_state.history[c_bulk_val] = {
            "snaps": snaps, "diag": diag, "thick": thick, "coords": coords
        }
        st.session_state.selected_c = c_bulk_val
        st.success(f"Done in {time.time()-t0:.2f}s")

# ------------------- COMBINED PLOTS -------------------
if len(st.session_state.history) > 1:
    st.header("Batch Comparison")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap(cmap_choice)
    colors = cmap(np.linspace(0, 1, len(st.session_state.history)))

    for idx, (c, data) in enumerate(st.session_state.history.items()):
        times = [scale_time(t) for t, _, _, _, _, _ in data["thick"]]
        ths   = [th*1e9 for _, _, th, _, _, _ in data["thick"]]
        ax1.plot(times, ths, label=f"c = {c:.3g}", color=colors[idx], lw=2)

        tdiag = [scale_time(t) for t, _, _, _, _, _, _ in data["diag"]]
        bulk  = [b for _, _, _, _, b, _ in data["diag"]]
        grad  = [g for _, _, _, _, _, g in data["diag"]]
        ax2.semilogy(tdiag, np.maximum(bulk, 1e-30), label=f"c={c:.3g} bulk", color=colors[idx], lw=1.5)
        ax2.semilogy(tdiag, np.maximum(grad, 1e-30), label=f"c={c:.3g} grad", color=colors[idx], ls='--', lw=1.5)

    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Thickness (nm)"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("L²-norm"); ax2.legend(); ax2.grid(True, alpha=0.3)
    st.pyplot(fig)

# ------------------- PLAYBACK -------------------
if st.session_state.history:
    st.header("Select Run for Playback")
    selected_c = st.selectbox("Choose run", sorted(st.session_state.history.keys(), reverse=True),
                              index=sorted(st.session_state.history.keys(), reverse=True).index(st.session_state.selected_c)
                              if st.session_state.selected_c in st.session_state.history else 0)
    st.session_state.selected_c = selected_c
    data = st.session_state.history[selected_c]
    snaps, thick, diag, coords = data["snaps"], data["thick"], data["diag"], data["coords"]

    frame = st.slider("Frame", 0, len(snaps)-1, len(snaps)-1)
    auto = st.checkbox("Autoplay", False)
    interval = st.number_input("Interval (s)", 0.1, 5.0, 0.4, 0.1)
    field = st.selectbox("Field", ["phi (shell)", "c (concentration)", "psi (core)"])

    t, phi, c, psi = snaps[frame]
    t_real = scale_time(t)
    th_nm = thick[frame][2] * 1e9
    c_mean, c_max, total_ag, bulk_norm, grad_norm = diag[frame][1:]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**c = {selected_c:.3g}** | **t = {t_real:.3e} s** | **Th = {th_nm:.2f} nm** | "
                 f"**||bulk||₂ = {bulk_norm:.2e}** | **||grad||₂ = {grad_norm:.2e}**")

        fig, ax = plt.subplots(figsize=(6,5))
        field_data = {"phi (shell)": phi, "c (concentration)": c, "psi (core)": psi}[field]
        vmin = 0
        vmax = 1 if field != "c (concentration)" else selected_c
        im = ax.imshow(field_data.T if mode=="2D (planar)" else field_data[field_data.shape[0]//2],
                       cmap=cmap_choice, vmin=vmin, vmax=vmax, origin='lower')
        plt.colorbar(im, ax=ax, label=field.split()[0])
        ax.set_title(f"{field} @ t = {t_real:.3e} s")
        st.pyplot(fig)

    with col2:
        st.subheader("Thickness")
        times = [scale_time(t) for t,_,_,_,_,_ in thick]
        ths   = [th*1e9 for _,_,th,_,_,_ in thick]
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(times, ths, 'b-', lw=2)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("nm"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Diagnostics")
        df = pd.DataFrame(diag, columns=["t", "c_mean", "c_max", "total_Ag", "||bulk||₂", "||grad||₂"])
        st.dataframe(df.tail(10).style.format("{:.3e}"))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"diag_c_{selected_c:.3g}.csv", "text/csv")

    # ------------------- MATERIAL & POTENTIAL -------------------
    st.subheader("Material & Potential Proxy")
    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        material_method = st.selectbox("Interpolation", [
            "phi + 2*psi (simple)", "phi*(1-psi) + 2*psi",
            "h·(phi² + psi²)", "h·(4*phi² + 2*psi²)", "max(phi, psi) + psi"
        ], index=3)
    with col_b:
        show_potential = st.checkbox("Show -α·c", True)
    with col_c:
        h_factor = st.slider("h", 0.1, 2.0, 0.5, 0.05) if "h·" in material_method else 1.0

    def build_material(phi, psi, method, h=1.0):
        if method == "phi + 2*psi (simple)": return phi + 2.0*psi
        elif method == "phi*(1-psi) + 2*psi": return phi*(1.0-psi) + 2.0*psi
        elif method == "h·(phi² + psi²)": return h*(phi**2 + psi**2)
        elif method == "h·(4*phi² + 2*psi²)": return h*(4.0*phi**2 + 2.0*psi**2)
        elif method == "max(phi, psi) + psi":
            return np.where(psi > 0.5, 2.0, np.where(phi > 0.5, 1.0, 0.0))
        return phi

    material = build_material(phi, psi, material_method, h_factor)
    potential = -alpha_nd * c

    if material_method in ["phi + 2*psi (simple)", "phi*(1-psi) + 2*psi", "max(phi, psi) + psi"]:
        cmap_mat = plt.cm.get_cmap("Set1", 3)
        vmin_mat, vmax_mat = 0, 2
    else:
        cmap_mat = cmap_choice
        vmin_mat = vmax_mat = None

    if mode == "2D (planar)":
        fig_mat, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(material.T, origin='lower', extent=[0,1,0,1], cmap=cmap_mat, vmin=vmin_mat, vmax=vmax_mat)
        if "max" in material_method or "2*psi" in material_method:
            cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2])
            cbar.ax.set_yticklabels(['electrolyte','Ag','Cu'])
        else:
            plt.colorbar(im, ax=ax, label="material")
        ax.set_title("Material")
        st.pyplot(fig_mat)

        if show_potential:
            fig_pot, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(potential.T, origin='lower', extent=[0,1,0,1], cmap="RdBu_r")
            plt.colorbar(im, ax=ax, label="-α·c")
            ax.set_title("Potential Proxy")
            st.pyplot(fig_pot)
    else:
        cx = phi.shape[0]//2
        fig, axes = plt.subplots(1,3,figsize=(12,4))
        for ax, sl, label in zip(axes, [material[cx,:,:], material[:,cx,:], material[:,:,cx]], ["x","y","z"]):
            ax.imshow(sl.T, origin='lower', cmap=cmap_mat, vmin=vmin_mat, vmax=vmax_mat)
            ax.set_title(label); ax.axis('off')
        fig.suptitle("Material (3D slices)")
        st.pyplot(fig)

    # ------------------- VTU/PVD/ZIP -------------------
    if st.sidebar.button("Export VTU/PVD/ZIP") and MESHIO_AVAILABLE:
        with st.spinner("Exporting VTU/PVD..."):
            tmpdir = tempfile.mkdtemp()
            vtus = []
            for idx, (t_nd, phi_s, c_s, psi_s) in enumerate(snaps):
                fname = os.path.join(tmpdir, f"frame_{idx:04d}.vtu")
                if mode == "2D (planar)":
                    xv, yv = coords
                    Xg, Yg = np.meshgrid(xv, yv, indexing='ij')
                    points = np.column_stack([nd_to_real(Xg.ravel()), nd_to_real(Yg.ravel()), np.zeros_like(Xg.ravel())])
                else:
                    xv, yv, zv = coords
                    Xg, Yg, Zg = np.meshgrid(xv, yv, zv, indexing='ij')
                    points = np.column_stack([nd_to_real(Xg.ravel()), nd_to_real(Yg.ravel()), nd_to_real(Zg.ravel())])
                mat_s = build_material(phi_s, psi_s, material_method, h_factor)
                point_data = {
                    "phi": phi_s.ravel().astype(np.float32),
                    "c": c_s.ravel().astype(np.float32),
                    "psi": psi_s.ravel().astype(np.float32),
                    "material": mat_s.ravel().astype(np.float32),
                    "potential": (-alpha_nd * c_s).ravel().astype(np.float32)
                }
                meshio.write_points_cells(fname, points, [], point_data=point_data)
                vtus.append(fname)

            pvd_path = os.path.join(tmpdir, "collection.pvd")
            with open(pvd_path, "w") as f:
                f.write('<VTKFile type="Collection" version="0.1">\n <Collection>\n')
                for idx, v in enumerate(vtus):
                    f.write(f'  <DataSet timestep="{scale_time(idx * dt_nd * save_every):.3e}" file="{os.path.basename(v)}"/>\n')
                f.write(' </Collection>\n</VTKFile>\n')

            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in vtus + [pvd_path]:
                    zf.write(p, arcname=os.path.basename(p))
            zipbuf.seek(0)
            st.download_button("Download VTU/PVD ZIP", zipbuf.read(), f"frames_{datetime.now():%Y%m%d_%H%M%S}.zip", "application/zip")

    # ------------------- GIF -------------------
    if st.sidebar.button("Generate GIF") and GIF_AVAILABLE:
        with st.spinner("Generating GIF..."):
            fig, ax = plt.subplots(figsize=(6,5))
            field_data = {"phi (shell)": phi, "c (concentration)": c, "psi (core)": psi}[field]
            vmin = 0
            vmax = 1 if field != "c (concentration)" else selected_c
            im = ax.imshow(field_data.T if mode=="2D (planar)" else field_data[field_data.shape[0]//2],
                           cmap=cmap_choice, vmin=vmin, vmax=vmax, origin='lower', animated=True)
            plt.colorbar(im, ax=ax)

            def animate(i):
                t, phi_f, c_f, psi_f = snaps[i]
                data = {"phi (shell)": phi_f, "c (concentration)": c_f, "psi (core)": psi_f}[field]
                im.set_array(data.T if mode=="2D (planar)" else data[data.shape[0]//2])
                ax.set_title(f"{field} @ t = {scale_time(t):.3e} s")
                return [im]

            anim = FuncAnimation(fig, animate, frames=len(snaps), interval=200, blit=True)
            gif_buf = io.BytesIO()
            anim.save(gif_buf, writer='pillow', fps=5)
            gif_buf.seek(0)
            st.download_button("Download GIF", gif_buf.read(), f"animation_{field.split()[0]}.gif", "image/gif")

    # ------------------- PNG SNAPSHOT -------------------
    if st.sidebar.button("Download PNG Snapshot"):
        fig_snap, ax = plt.subplots(figsize=(6,5))
        data = {"phi (shell)": phi, "c (concentration)": c, "psi (core)": psi}[field]
        im = ax.imshow(data.T if mode=="2D (planar)" else data[data.shape[0]//2],
                       cmap=cmap_choice, vmin=0, vmax=1 if field != "c (concentration)" else selected_c, origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{field} @ t = {t_real:.3e} s")
        buf = io.BytesIO()
        fig_snap.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button("Download PNG", buf.read(), f"snapshot_{field.split()[0]}.png", "image/png")

    if auto:
        for i in range(frame, len(snaps)):
            time.sleep(interval)
            st.rerun()

    col_clear, col_del = st.columns([1, 1])
    with col_clear:
        if st.button("Clear ALL"): st.session_state.history.clear(); st.session_state.selected_c = None; st.experimental_rerun()
    with col_del:
        if st.button(f"Delete c = {selected_c:.3g}"):
            del st.session_state.history[selected_c]
            st.session_state.selected_c = list(st.session_state.history.keys())[0] if st.session_state.history else None
            st.experimental_rerun()

else:
    st.info("Run at least one simulation (single or batch) to see results.")
