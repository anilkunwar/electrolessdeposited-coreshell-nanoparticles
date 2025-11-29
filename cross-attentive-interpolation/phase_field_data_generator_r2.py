#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELECTROLESS Ag — FULLY UPGRADED WITH .PKL EXPORT + DOMAIN SIZE CONTROL
* Auto-saves .pkl after every run
* One-click .pkl download per c_bulk in playback panel
* Domain multiplier increases effective distance to boundary (core scales with domain)
* Filename includes c_bulk, BC, EDL, k0, M, D, domain size, resolution
* 100% backward compatible
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import time
import io
import zipfile
import os
import tempfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

# ------------------- SAFE GPU SETUP -------------------
GPU_AVAILABLE = False
try:
    import cupy as cp
    cp.cuda.Device(0).use()
    cp.zeros(1)
    GPU_AVAILABLE = True
    st.sidebar.success("GPU (CuPy) ready!")
except Exception as e:
    import numpy as cp
    from numpy.fft import fft2, ifft2
    GPU_AVAILABLE = False
    st.sidebar.warning(f"GPU unavailable: {e}")

# ------------------- OPTIONAL LIBS -------------------
try:
    import meshio
    MESHIO_AVAILABLE = True
except:
    MESHIO_AVAILABLE = False
try:
    import imageio
    from matplotlib.animation import FuncAnimation
    GIF_AVAILABLE = True
except:
    GIF_AVAILABLE = False

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Electroless Ag — EDL + PKL", layout="wide")
st.title("Electroless Ag — EDL Catalyst + .PKL Export")

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
    "Choose c_bulk values",
    list(c_options.keys()),
    default=["1.0 (1:1)", "0.5 (1:2)", "0.2 (1:5)"]
)
c_bulk_list = [c_options[l] for l in selected_labels]

st.sidebar.header("Simulation")
mode = st.sidebar.selectbox("Mode", ["2D (planar)", "3D (spherical)"])
bc_type = st.sidebar.selectbox("BC", ["Neumann (zero flux)", "Dirichlet (fixed values)"])

domain_multiplier = st.sidebar.slider(
    "Domain Size Multiplier", 1.0, 5.0, 1.0, 0.25,
    help="1.0 = original. >1.0 = larger bath → less boundary effect"
)

max_res = 1024 if GPU_AVAILABLE else 512
max_steps = 4000 if GPU_AVAILABLE else 2000
Nx = st.sidebar.slider("Nx", 64, max_res, 256, 32)
Ny = st.sidebar.slider("Ny", 64, max_res, 256, 32) if mode == "2D (planar)" else Nx
Nz = st.sidebar.slider("Nz", 32, max_res // 4, 64, 8) if mode != "2D (planar)" else 1
dt_nd = st.sidebar.number_input("dt (nd)", 1e-6, 1e-2, 1e-4, format="%.6f")
n_steps = st.sidebar.slider("Steps", 50, max_steps, 100000, 50)
save_every = st.sidebar.slider("Save every", 1, 200, max(1, n_steps // 20), 1)

st.sidebar.header("Physics (nd)")
gamma_nd = st.sidebar.slider("γ", 1e-4, 0.5, 0.02, 1e-4)
beta_nd = st.sidebar.slider("β", 0.1, 20.0, 4.0, 0.1)
k0_nd = st.sidebar.slider("k₀", 0.01, 2.0, 0.4, 0.01)
M_nd = st.sidebar.slider("M", 1e-3, 1.0, 0.2, 1e-3)
D_nd = st.sidebar.slider("D", 0.0, 1.0, 0.05, 0.005)
alpha_nd = st.sidebar.slider("α (coupling)", 0.0, 10.0, 2.0, 0.1)
use_fft = st.sidebar.checkbox("Use FFT Laplacian", GPU_AVAILABLE)
cmap_choice = st.sidebar.selectbox("Colormap", CMAPS, CMAPS.index("viridis"))

st.sidebar.header("Geometry")
core_radius_frac = st.sidebar.slider("Core / L", 0.05, 0.45, 0.18, 0.01)
shell_thickness_frac = st.sidebar.slider("Δr / r_core", 0.05, 0.6, 0.2, 0.01)
phi_threshold = st.sidebar.slider("φ threshold", 0.1, 0.9, 0.5, 0.05)
growth_model = st.sidebar.selectbox("Model", ["Model A (irreversible)", "Model B (soft reversible)"])

st.sidebar.header("Physical Scales")
L0 = st.sidebar.number_input("L₀ (nm)", 1.0, 1e6, 20.0) * 1e-9
tau0 = st.sidebar.number_input("τ₀ (×10⁻⁴ s)", 1e-6, 1e6, 1.0) * 1e-4

# ------------------- EDL -------------------
st.sidebar.header("EDL Catalyst (Optional)")
use_edl = st.sidebar.checkbox("Enable EDL Nucleation Boost", False)
if use_edl:
    lambda0_edl = st.sidebar.slider("λ₀ (initial boost)", 0.0, 5.0, 2.0, 0.1)
    tau_edl_nd = st.sidebar.slider("τ_edl (decay time, nd)", 1e-3, 1.0, 0.05, 0.005)
    alpha_edl = st.sidebar.slider("EDL strength α", 0.0, 10.0, 3.0, 0.1)
else:
    lambda0_edl = tau_edl_nd = alpha_edl = 0.0

run_batch_button = st.sidebar.button("Run BATCH")
run_single_button = st.sidebar.button("Run SINGLE (last c)")

# ------------------- HELPERS -------------------
def scale_time(t): return t * tau0
def nd_to_real(x): return x * L0
def to_cpu(arr): return cp.asnumpy(arr) if GPU_AVAILABLE else arr
def to_gpu(arr): return cp.asarray(arr) if GPU_AVAILABLE else arr

def get_edl_factor(t_nd, use_edl, lambda0_edl, tau_edl_nd):
    if not use_edl or lambda0_edl <= 0 or tau_edl_nd <= 0:
        return 0.0
    return lambda0_edl * cp.exp(-t_nd / tau_edl_nd)

def laplacian(u, dx):
    if GPU_AVAILABLE and use_fft:
        if u.ndim == 2:
            kx = cp.fft.fftfreq(u.shape[0], d=1.0/u.shape[0]) * 2 * cp.pi
            ky = cp.fft.fftfreq(u.shape[1], d=1.0/u.shape[1]) * 2 * cp.pi
            KX, KY = cp.meshgrid(kx, ky, indexing='ij')
            K2 = KX**2 + KY**2
            u_hat = cp.fft.fft2(u)
            return cp.fft.ifft2(-K2 * u_hat).real
        else:
            kx = cp.fft.fftfreq(u.shape[0], d=1.0/u.shape[0]) * 2 * cp.pi
            ky = cp.fft.fftfreq(u.shape[1], d=1.0/u.shape[1]) * 2 * cp.pi
            kz = cp.fft.fftfreq(u.shape[2], d=1.0/u.shape[2]) * 2 * cp.pi
            KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing='ij')
            K2 = KX**2 + KY**2 + KZ**2
            u_hat = cp.fft.fftn(u)
            return cp.fft.ifftn(-K2 * u_hat).real
    else:
        if u.ndim == 2:
            return (cp.roll(u, 1, 0) + cp.roll(u, -1, 0) +
                    cp.roll(u, 1, 1) + cp.roll(u, -1, 1) - 4*u) / (dx*dx)
        else:
            return (cp.roll(u, 1, 0) + cp.roll(u, -1, 0) +
                    cp.roll(u, 1, 1) + cp.roll(u, -1, 1) +
                    cp.roll(u, 1, 2) + cp.roll(u, -1, 2) - 6*u) / (dx*dx)

# ------------------- SIMULATION CORE -------------------
def run_simulation(c_bulk_val):
    L = domain_multiplier
    dx = L / (Nx - 1)
    center = L / 2.0

    x = np.linspace(0, L, Nx)
    x_gpu = to_gpu(x)

    if mode == "2D (planar)":
        y = np.linspace(0, L, Ny)
        y_gpu = to_gpu(y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Xg, Yg = to_gpu(X), to_gpu(Y)
        dist = cp.sqrt((Xg - center)**2 + (Yg - center)**2)
        coords = (x, y)
    else:
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        Xg, Yg, Zg = to_gpu(X), to_gpu(Y), to_gpu(Z)
        dist = cp.sqrt((Xg - center)**2 + (Yg - center)**2 + (Zg - center)**2)
        coords = (x, x, x)

    psi = (dist <= core_radius_frac * L).astype(cp.float64)
    r_core = core_radius_frac * L
    r_outer = r_core * (1 + shell_thickness_frac)

    phi = cp.where(dist <= r_core, 0.0, cp.where(dist <= r_outer, 1.0, 0.0))
    eps = max(4 * dx, 1e-6)
    phi = phi * (1 - 0.5 * (1 - cp.tanh((dist - r_core) / eps))) \
              * (1 - 0.5 * (1 + cp.tanh((dist - r_outer) / eps)))
    phi = cp.clip(phi, 0, 1)

    if bc_type == "Neumann (zero flux)":
        if mode == "2D (planar)":
            c = c_bulk_val * (Yg / L) * (1 - phi) * (1 - psi)
        else:
            c = c_bulk_val * (Zg / L) * (1 - phi) * (1 - psi)
    else:
        c = c_bulk_val * (1 - phi) * (1 - psi)
    c = cp.clip(c, 0, c_bulk_val)

    snapshots, diags, thick = [], [], []
    softness = 0.01 if "B" in growth_model else 0.0
    max_th = 0.0

    for step in range(n_steps + 1):
        t = step * dt_nd

        grad_phi = cp.gradient(phi, dx)
        gphi = cp.sqrt(sum(g**2 for g in grad_phi) + 1e-30)
        delta_int = 6 * phi * (1 - phi) * (1 - psi) * gphi
        delta_int = cp.clip(delta_int, 0, 6 / max(eps, dx))

        f_bulk = 2 * beta_nd * phi * (1 - phi) * (1 - 2 * phi)

        lambda_edl_t = get_edl_factor(t, use_edl, lambda0_edl, tau_edl_nd)
        edl_boost = 1.0 + lambda_edl_t * alpha_edl * (delta_int / (cp.max(delta_int) + 1e-12))
        edl_boost = cp.where(use_edl, edl_boost, 1.0)

        i_loc = k0_nd * c * (1 - phi) * (1 - psi) * delta_int * edl_boost
        i_loc = cp.clip(i_loc, 0, 1e6)

        lap_phi = laplacian(phi, dx)
        dep = M_nd * i_loc
        curv = M_nd * gamma_nd * lap_phi
        dphi = dt_nd * (dep + cp.maximum(curv, 0) - softness * M_nd * f_bulk)
        if softness == 0:
            dphi = cp.maximum(dphi, 0)
        phi = cp.clip(phi + dphi, 0, 1)

        lap_c = laplacian(c, dx)
        c += dt_nd * (D_nd * lap_c - i_loc)
        c = cp.clip(c, 0, c_bulk_val)

        if bc_type == "Neumann (zero flux)":
            if mode == "2D (planar)":
                c[:, -1] = c_bulk_val
            else:
                c[:, :, -1] = c_bulk_val
        else:
            if mode == "2D (planar)":
                c[[0, -1], :] = c_bulk_val
                c[:, [0, -1]] = c_bulk_val
            else:
                c[[0, -1], :, :] = c_bulk_val
                c[:, [0, -1], :] = c_bulk_val
                c[:, :, [0, -1]] = c_bulk_val

        bulk_norm = cp.sqrt(cp.mean(f_bulk**2))
        grad_norm = cp.sqrt(cp.mean((M_nd * gamma_nd * lap_phi)**2))
        edl_flux = float(cp.sum(edl_boost - 1.0)) if use_edl else 0.0

        if step % save_every == 0 or step == n_steps:
            phi_cpu = to_cpu(phi)
            psi_cpu = to_cpu(psi)
            c_cpu = to_cpu(c)

            Xc, Yc, Zc = np.meshgrid(
                x,
                y if mode == "2D (planar)" else x,
                x if mode != "2D (planar)" else [center],
                indexing='ij'
            )
            dist_cpu = np.sqrt((Xc - center)**2 + (Yc - center)**2 + (Zc - center)**2)
            mask = (phi_cpu > phi_threshold) & (psi_cpu < 0.5)
            th_nd = np.max(dist_cpu[mask]) - r_core if np.any(mask) else 0.0
            max_th = max(max_th, th_nd)

            c_mean = float(cp.mean(c))
            c_max = float(cp.max(c))
            total_ag = float(cp.sum(i_loc) * dt_nd)

            snapshots.append((t, phi_cpu, c_cpu, psi_cpu))
            diags.append((t, c_mean, c_max, total_ag, float(bulk_norm), float(grad_norm), edl_flux))
            thick.append((t, max_th, nd_to_real(max_th), c_mean, c_max, total_ag))

    # ------------------- AUTO SAVE .PKL -------------------
    os.makedirs("electroless_pkl_solutions", exist_ok=True)

    bc_str = "Neu" if bc_type == "Neumann (zero flux)" else "Dir"
    mode_str = "2D" if mode == "2D (planar)" else "3D"
    edl_str = f"EDL{lambda0_edl:.1f}" if use_edl else "noEDL"
    dom_str = f"dom{domain_multiplier:.2f}".replace(".", "p")

    filename = (
        f"Ag_{mode_str}_c{c_bulk_val:.3f}_{bc_str}_{edl_str}_"
        f"k{k0_nd:.2f}_M{M_nd:.2f}_D{D_nd:.3f}_{dom_str}_"
        f"Nx{Nx}_steps{n_steps}.pkl"
    )
    filepath = os.path.join("electroless_pkl_solutions", filename)

    save_data = {
        "meta": {
            "c_bulk": float(c_bulk_val),
            "domain_multiplier": float(domain_multiplier),
            "bc_type": bc_type,
            "mode": mode,
            "use_edl": use_edl,
            "timestamp": datetime.now().isoformat(),
        },
        "parameters": {
            "gamma_nd": float(gamma_nd), "beta_nd": float(beta_nd),
            "k0_nd": float(k0_nd), "M_nd": float(M_nd), "D_nd": float(D_nd),
            "alpha_nd": float(alpha_nd), "lambda0_edl": float(lambda0_edl),
            "tau_edl_nd": float(tau_edl_nd), "alpha_edl": float(alpha_edl),
            "core_radius_frac": float(core_radius_frac),
            "shell_thickness_frac": float(shell_thickness_frac),
            "phi_threshold": float(phi_threshold),
            "Nx": Nx, "Ny": Ny, "Nz": Nz,
            "n_steps": n_steps, "dt_nd": float(dt_nd),
            "L0_m": float(L0), "tau0_s": float(tau0),
        },
        "coords_nd": coords,
        "snapshots": snapshots,
        "diagnostics": diags,
        "thickness_history_nm": thick,
    }

    with open(filepath, "wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_data["_filepath"] = filepath
    save_data["_filename"] = filename

    st.sidebar.success(f"Auto-saved → {filename}")

    return c_bulk_val, snapshots, diags, thick, coords, save_data

# ------------------- HISTORY & RUN -------------------
if "history" not in st.session_state:
    st.session_state.history = {}
if "selected_c" not in st.session_state:
    st.session_state.selected_c = None

if run_batch_button and c_bulk_list:
    with st.spinner("Running BATCH..."):
        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_simulation, c) for c in c_bulk_list]
            for future in as_completed(futures):
                try:
                    c, s, d, t, co, save_data = future.result()
                    results.append((c, s, d, t, co, save_data))
                except Exception as e:
                    st.error(f"Sim failed: {e}")
        for c, s, d, t, co, save_data in results:
            st.session_state.history[c] = {"snaps": s, "diag": d, "thick": t, "coords": co, "pkl": save_data}
        if results:
            st.session_state.selected_c = results[0][0]
        st.success(f"Batch done: {len(results)} runs")

if run_single_button and selected_labels:
    c_val = c_options[selected_labels[-1]]
    with st.spinner(f"Running SINGLE c = {c_val}..."):
        c, s, d, t, co, save_data = run_simulation(c_val)
        st.session_state.history[c_val] = {"snaps": s, "diag": d, "thick": t, "coords": co, "pkl": save_data}
        st.session_state.selected_c = c_val
        st.success("Done")

# ------------------- BATCH COMPARISON -------------------
if len(st.session_state.history) > 1:
    st.header("Batch Comparison")
    
    # Styling controls
    col1, col2, col3 = st.columns(3)
    with col1:
        logx = st.checkbox("Log time", False)
    with col2:
        logy = st.checkbox("Log thickness", False)
    with col3:
        show_final_only = st.checkbox("Final thickness only", True)
    
    # Plot customization
    col1, col2 = st.columns(2)
    with col1:
        line_width = st.slider("Line width", 1.0, 5.0, 2.5, 0.5)
    with col2:
        marker_size = st.slider("Marker size", 0, 10, 4, 1)
    
    # Create comparison plots
    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(st.session_state.history)))
    
    for i, (c_bulk, data) in enumerate(st.session_state.history.items()):
        thick_data = data["thick"]
        if not thick_data:
            continue
            
        times = [t[0] * tau0 * 1e3 for t in thick_data]  # ms
        thickness_nm = [t[2] * 1e9 for t in thick_data]  # nm
        
        label = f"c={c_bulk:.3f}"
        color = colors[i]
        
        if show_final_only:
            # Only plot final thickness vs c_bulk
            ax1.scatter(c_bulk, thickness_nm[-1], s=80, color=color, 
                       label=label, alpha=0.7, edgecolors='black', linewidth=1)
        else:
            # Plot full evolution
            ax1.plot(times, thickness_nm, linewidth=line_width, 
                    marker='o' if marker_size > 0 else None, 
                    markersize=marker_size, label=label, color=color)
        
        # Plot deposition rate
        diag_data = data["diag"]
        if len(diag_data) > 1:
            diag_times = [d[0] * tau0 * 1e3 for d in diag_data]
            total_ag = [d[3] for d in diag_data]
            ax2.plot(diag_times, total_ag, linewidth=line_width, 
                    marker='s' if marker_size > 0 else None, 
                    markersize=marker_size, label=label, color=color)
    
    # Configure plot 1
    ax1.set_xlabel("Time (ms)" if not show_final_only else "c_bulk")
    ax1.set_ylabel("Shell thickness (nm)")
    ax1.set_title("Growth Kinetics" if not show_final_only else "Final Thickness vs Concentration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if logx and not show_final_only:
        ax1.set_xscale('log')
    if logy:
        ax1.set_yscale('log')
    
    # Configure plot 2
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Total Ag deposited")
    ax2.set_title("Cumulative Deposition")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if logx:
        ax2.set_xscale('log')
    
    plt.tight_layout()
    st.pyplot(fig_comp)
    
    # Summary table
    st.subheader("Batch Summary")
    summary_data = []
    for c_bulk, data in st.session_state.history.items():
        if data["thick"]:
            final_thick_nm = data["thick"][-1][2] * 1e9
            final_time_ms = data["thick"][-1][0] * tau0 * 1e3
            summary_data.append({
                "c_bulk": c_bulk,
                "Final Thickness (nm)": f"{final_thick_nm:.1f}",
                "Final Time (ms)": f"{final_time_ms:.1f}",
                "Snapshots": len(data["snaps"])
            })
    
    if summary_data:
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ------------------- PLAYBACK + PKL DOWNLOAD -------------------
if st.session_state.history:
    st.header("Select Run for Playback & Download")
    selected_c = st.selectbox(
        "Choose run",
        sorted(st.session_state.history.keys(), reverse=True),
        index=sorted(st.session_state.history.keys(), reverse=True).index(st.session_state.selected_c)
        if st.session_state.selected_c in st.session_state.history else 0
    )
    st.session_state.selected_c = selected_c
    data = st.session_state.history[selected_c]
    snaps, thick, diag, coords = data["snaps"], data["thick"], data["diag"], data["coords"]
    save_data = data.get("pkl")

    # === ONE-CLICK PKL DOWNLOAD ===
    if save_data and "_filepath" in save_data and os.path.exists(save_data["_filepath"]):
        with open(save_data["_filepath"], "rb") as f:
            st.download_button(
                label=f"📦 Download FULL simulation as .pkl — {save_data['_filename']}",
                data=f.read(),
                file_name=save_data["_filename"],
                mime="application/octet-stream",
                use_container_width=True
            )
    else:
        st.info("Re-run this simulation to generate downloadable .pkl")

    # Playback controls
    if snaps:
        st.subheader("Playback")
        frame_idx = st.slider("Frame", 0, len(snaps)-1, len(snaps)-1, 
                             help="Select simulation frame to display")
        
        t_nd, phi, c, psi = snaps[frame_idx]
        t_real_ms = t_nd * tau0 * 1e3
        
        # Display current frame info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Time (nd)", f"{t_nd:.3f}")
        with col2:
            st.metric("Time (ms)", f"{t_real_ms:.1f}")
        with col3:
            if thick and frame_idx < len(thick):
                st.metric("Thickness (nm)", f"{thick[frame_idx][2]*1e9:.1f}")
        with col4:
            st.metric("Frame", f"{frame_idx}/{len(snaps)-1}")
        
        # Visualization
        fig_playback = plt.figure(figsize=(12, 5))
        
        if mode == "2D (planar)":
            # 2D plots
            ax1 = fig_playback.add_subplot(121)
            im1 = ax1.imshow(phi.T, cmap=cmap_choice, origin='lower', 
                           extent=[0, domain_multiplier, 0, domain_multiplier])
            ax1.set_title(f"φ @ t = {t_real_ms:.1f} ms")
            ax1.set_xlabel("x (nd)")
            ax1.set_ylabel("y (nd)")
            plt.colorbar(im1, ax=ax1, label="Phase field φ")
            
            ax2 = fig_playback.add_subplot(122)
            im2 = ax2.imshow(c.T, cmap='plasma', origin='lower',
                           extent=[0, domain_multiplier, 0, domain_multiplier])
            ax2.set_title(f"Concentration @ t = {t_real_ms:.1f} ms")
            ax2.set_xlabel("x (nd)")
            ax2.set_ylabel("y (nd)")
            plt.colorbar(im2, ax=ax2, label="Concentration c")
            
        else:
            # 3D - show middle slice
            mid_z = phi.shape[2] // 2
            ax1 = fig_playback.add_subplot(121)
            im1 = ax1.imshow(phi[:, :, mid_z].T, cmap=cmap_choice, origin='lower',
                           extent=[0, domain_multiplier, 0, domain_multiplier])
            ax1.set_title(f"φ @ z-slice {mid_z}, t = {t_real_ms:.1f} ms")
            ax1.set_xlabel("x (nd)")
            ax1.set_ylabel("y (nd)")
            plt.colorbar(im1, ax=ax1, label="Phase field φ")
            
            ax2 = fig_playback.add_subplot(122)
            im2 = ax2.imshow(c[:, :, mid_z].T, cmap='plasma', origin='lower',
                           extent=[0, domain_multiplier, 0, domain_multiplier])
            ax2.set_title(f"Concentration @ z-slice {mid_z}, t = {t_real_ms:.1f} ms")
            ax2.set_xlabel("x (nd)")
            ax2.set_ylabel("y (nd)")
            plt.colorbar(im2, ax=ax2, label="Concentration c")
        
        plt.tight_layout()
        st.pyplot(fig_playback)
        
        # Diagnostics plot
        if diag:
            fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            times = [d[0] * tau0 * 1e3 for d in diag]
            c_means = [d[1] for d in diag]
            c_maxs = [d[2] for d in diag]
            total_ag = [d[3] for d in diag]
            
            ax1.plot(times, c_means, 'b-', label='Mean c', linewidth=2)
            ax1.plot(times, c_maxs, 'r--', label='Max c', linewidth=2)
            ax1.set_xlabel("Time (ms)")
            ax1.set_ylabel("Concentration")
            ax1.set_title("Concentration Evolution")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(times, total_ag, 'g-', linewidth=2)
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Total Ag")
            ax2.set_title("Cumulative Silver Deposited")
            ax2.grid(True, alpha=0.3)
            
            # Mark current frame
            current_time = t_nd * tau0 * 1e3
            ax1.axvline(current_time, color='black', linestyle=':', alpha=0.7)
            ax2.axvline(current_time, color='black', linestyle=':', alpha=0.7)
            
            plt.tight_layout()
            st.pyplot(fig_diag)
        
        # Export options
        st.subheader("Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PNG export
            buf = io.BytesIO()
            plt.figure(figsize=(8, 6))
            if mode == "2D (planar)":
                plt.imshow(phi.T, cmap=cmap_choice, origin='lower',
                          extent=[0, domain_multiplier, 0, domain_multiplier])
            else:
                mid_z = phi.shape[2] // 2
                plt.imshow(phi[:, :, mid_z].T, cmap=cmap_choice, origin='lower',
                          extent=[0, domain_multiplier, 0, domain_multiplier])
            plt.title(f"φ @ t = {t_real_ms:.1f} ms, c = {selected_c:.3f}")
            plt.colorbar(label="Phase field φ")
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150)
            plt.close()
            
            st.download_button(
                label="📷 Download current frame as PNG",
                data=buf.getvalue(),
                file_name=f"Ag_frame_{frame_idx}_t{t_real_ms:.1f}ms.png",
                mime="image/png"
            )
        
        with col2:
            # GIF export (if available)
            if GIF_AVAILABLE and len(snaps) > 1:
                if st.button("🎬 Create GIF animation"):
                    with st.spinner("Creating GIF..."):
                        gif_buf = io.BytesIO()
                        fig, ax = plt.subplots(figsize=(6, 5))
                        
                        def animate_frame(i):
                            ax.clear()
                            t_nd, phi_frame, c_frame, psi_frame = snaps[i]
                            t_ms = t_nd * tau0 * 1e3
                            if mode == "2D (planar)":
                                im = ax.imshow(phi_frame.T, cmap=cmap_choice, origin='lower',
                                             extent=[0, domain_multiplier, 0, domain_multiplier])
                            else:
                                mid_z = phi_frame.shape[2] // 2
                                im = ax.imshow(phi_frame[:, :, mid_z].T, cmap=cmap_choice, origin='lower',
                                             extent=[0, domain_multiplier, 0, domain_multiplier])
                            ax.set_title(f"t = {t_ms:.1f} ms")
                            return [im]
                        
                        anim = FuncAnimation(fig, animate_frame, frames=len(snaps), 
                                           interval=200, blit=True)
                        anim.save(gif_buf, writer='pillow', fps=5)
                        plt.close()
                        
                        st.download_button(
                            label="📥 Download GIF",
                            data=gif_buf.getvalue(),
                            file_name=f"Ag_animation_c{selected_c:.3f}.gif",
                            mime="image/gif"
                        )
            else:
                st.info("GIF export requires imageio")
        
        with col3:
            # CSV export of diagnostics
            if diag:
                csv_buf = io.StringIO()
                df = pd.DataFrame(diag, columns=[
                    'time_nd', 'c_mean', 'c_max', 'total_ag', 
                    'bulk_norm', 'grad_norm', 'edl_flux'
                ])
                df['time_ms'] = df['time_nd'] * tau0 * 1e3
                df.to_csv(csv_buf, index=False)
                
                st.download_button(
                    label="📊 Download diagnostics CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"Ag_diagnostics_c{selected_c:.3f}.csv",
                    mime="text/csv"
                )

else:
    st.info("Run a simulation (single or batch) to see results.")
