#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELECTROLESS Ag — FULLY UPGRADED & BACKWARD‑COMPATIBLE SIMULATOR
* EDL as catalytic nucleation booster: i_loc *= (1 + λ(t)·α·δ_int)
* λ(t) = λ₀ exp(-t/τ_edl) → decays over time
* δ_int = 6ϕ(1-ϕ)|∇ϕ| → interface‑localized
* When EDL disabled → 100% identical to original
* GPU (CuPy) / CPU (NumPy) safe
* Batch, VTU/PVD/ZIP, GIF, PNG, CSV, material & potential proxy
* **PKL export** – auto‑saves every run, one‑click download
* **Phase statistics** – area/volume of electrolyte, Ag, Cu (threshold‑based)
* **Default material proxy** = max(ϕ,ψ) + ψ with discrete colourmap
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
import pickle                       # <-- for PKL export

# ------------------- SAFE GPU SETUP -------------------
GPU_AVAILABLE = False
try:
    import cupy as cp
    cp.cuda.Device(0).use()
    cp.zeros(1)                      # force context init
    GPU_AVAILABLE = True
    st.sidebar.success("GPU (CuPy) detected and ready!")
except Exception as e:
    import numpy as cp
    from numpy.fft import fft2, ifft2
    GPU_AVAILABLE = False
    st.sidebar.warning(f"GPU not available: {e}\nUsing CPU (NumPy).")

# ------------------- OPTIONAL LIBS -------------------
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
st.set_page_config(page_title="Electroless Ag — EDL + PKL + Phase Stats", layout="wide")
st.title("Electroless Ag — EDL Catalyst + PKL Export + Phase Statistics")

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
L0 = st.sidebar.number_input("L₀ (nm)", 1.0, 1e6, 20.0) * 1e-9   # nm → cm
tau0 = st.sidebar.number_input("τ₀ (×10⁻⁴ s)", 1e-6, 1e6, 1.0) * 1e-4

# ------------------- EDL CATALYST (OPTIONAL) -------------------
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

# ------------------- EDL DECAY -------------------
def get_edl_factor(t_nd, use_edl, lambda0_edl, tau_edl_nd):
    if not use_edl or lambda0_edl <= 0 or tau_edl_nd <= 0:
        return 0.0
    return lambda0_edl * cp.exp(-t_nd / tau_edl_nd)

# ------------------- LAPLACIAN -------------------
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

# ------------------- PHASE STATISTICS (AREA/VOLUME) -------------------
def compute_phase_stats(phi, psi, dx, dy, dz, L0, threshold):
    """
    Compute area (2D) or volume (3D) of electrolyte, Ag, and Cu.
    Returns dict with counts and real units.
    """
    # Thresholds
    ag_mask = (phi > threshold) & (psi <= 0.5)
    cu_mask = psi > 0.5
    electrolyte_mask = ~(ag_mask | cu_mask)  # everything else

    if phi.ndim == 2:  # 2D
        cell_area_nd = dx * dy
        cell_area_real = cell_area_nd * (L0**2)
        electrolyte_area_nd = np.sum(electrolyte_mask) * cell_area_nd
        ag_area_nd = np.sum(ag_mask) * cell_area_nd
        cu_area_nd = np.sum(cu_mask) * cell_area_nd
        electrolyte_area_real = electrolyte_area_nd * (L0**2)
        ag_area_real = ag_area_nd * (L0**2)
        cu_area_real = cu_area_nd * (L0**2)
        return {
            "Electrolyte": (electrolyte_area_nd, electrolyte_area_real),
            "Ag": (ag_area_nd, ag_area_real),
            "Cu": (cu_area_nd, cu_area_real),
        }
    else:  # 3D
        cell_vol_nd = dx * dy * dz
        cell_vol_real = cell_vol_nd * (L0**3)
        electrolyte_vol_nd = np.sum(electrolyte_mask) * cell_vol_nd
        ag_vol_nd = np.sum(ag_mask) * cell_vol_nd
        cu_vol_nd = np.sum(cu_mask) * cell_vol_nd
        electrolyte_vol_real = electrolyte_vol_nd * (L0**3)
        ag_vol_real = ag_vol_nd * (L0**3)
        cu_vol_real = cu_vol_nd * (L0**3)
        return {
            "Electrolyte": (electrolyte_vol_nd, electrolyte_vol_real),
            "Ag": (ag_vol_nd, ag_vol_real),
            "Cu": (cu_vol_nd, cu_vol_real),
        }

# ------------------- SIMULATION CORE (with PKL export) -------------------
def run_simulation(c_bulk_val):
    L = 1.0
    dx = L / (Nx - 1)
    dy = L / (Ny - 1) if mode == "2D (planar)" else dx
    dz = L / (Nz - 1) if mode != "2D (planar)" else 1.0
    x = np.linspace(0, L, Nx)
    x_gpu = to_gpu(x)

    if mode == "2D (planar)":
        y = np.linspace(0, L, Ny)
        y_gpu = to_gpu(y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Xg, Yg = to_gpu(X), to_gpu(Y)
        dist = cp.sqrt((Xg - 0.5)**2 + (Yg - 0.5)**2)
        coords = (x, y)
    else:
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        Xg, Yg, Zg = to_gpu(X), to_gpu(Y), to_gpu(Z)
        dist = cp.sqrt((Xg - 0.5)**2 + (Yg - 0.5)**2 + (Zg - 0.5)**2)
        coords = (x, x, x)

    psi = (dist <= core_radius_frac * L).astype(cp.float64)
    r_core = core_radius_frac * L
    r_outer = r_core * (1 + shell_thickness_frac)
    phi = cp.where(dist <= r_core, 0.0, cp.where(dist <= r_outer, 1.0, 0.0))
    eps = max(4 * dx, 1e-6)
    phi = phi * (1 - 0.5 * (1 - cp.tanh((dist - r_core) / eps))) \
              * (1 - 0.5 * (1 + cp.tanh((dist - r_outer) / eps)))
    phi = cp.clip(phi, 0, 1)

    c = c_bulk_val * (Yg / L if mode == "2D (planar)" else Zg / L) * (1 - phi) * (1 - psi) \
        if bc_type == "Neumann (zero flux)" else c_bulk_val * (1 - phi) * (1 - psi)
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

        # === EDL CATALYST: interface‑localized, time‑decaying boost ===
        lambda_edl_t = get_edl_factor(t, use_edl, lambda0_edl, tau_edl_nd)
        edl_boost = 1.0 + lambda_edl_t * alpha_edl * (delta_int / (cp.max(delta_int) + 1e-12))
        edl_boost = cp.where(use_edl, edl_boost, 1.0)   # ← forces 1.0 when disabled

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

        # BCs
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
                x if mode != "2D (planar)" else [0.5],
                indexing='ij'
            )
            dist_cpu = np.sqrt((Xc - 0.5)**2 + (Yc - 0.5)**2 + (Zc - 0.5)**2)
            mask = (phi_cpu > phi_threshold) & (psi_cpu < 0.5)
            th_nd = np.max(dist_cpu[mask]) - core_radius_frac if np.any(mask) else 0.0
            max_th = max(max_th, th_nd)

            c_mean = float(cp.mean(c))
            c_max = float(cp.max(c))
            total_ag = float(cp.sum(i_loc) * dt_nd)

            snapshots.append((t, phi_cpu, c_cpu, psi_cpu))
            diags.append((t, c_mean, c_max, total_ag, float(bulk_norm), float(grad_norm), edl_flux))
            thick.append((t, max_th, nd_to_real(max_th), c_mean, c_max, total_ag))

    # ------------------- AUTO‑SAVE .PKL -------------------
    os.makedirs("electroless_pkl_solutions", exist_ok=True)

    bc_str = "Neu" if bc_type == "Neumann (zero flux)" else "Dir"
    mode_str = "2D" if mode == "2D (planar)" else "3D"
    edl_str = f"EDL{lambda0_edl:.1f}" if use_edl else "noEDL"

    filename = (
        f"Ag_{mode_str}_c{c_bulk_val:.3f}_{bc_str}_{edl_str}_"
        f"k{k0_nd:.2f}_M{M_nd:.2f}_D{D_nd:.3f}_"
        f"Nx{Nx}_steps{n_steps}.pkl"
    )
    filepath = os.path.join("electroless_pkl_solutions", filename)

    save_data = {
        "meta": {
            "c_bulk": float(c_bulk_val),
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

    st.sidebar.success(f"Auto‑saved → {filename}")

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
                    results.append(future.result())
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
        st.session_state.history[c] = {"snaps": s, "diag": d, "thick": t, "coords": co, "pkl": save_data}
        st.session_state.selected_c = c
        st.success("Done")

# ------------------- BATCH COMPARISON + EDL DECAY -------------------
if len(st.session_state.history) > 1:
    st.header("Batch Comparison")

    # --- Styling sliders ---
    st.sidebar.subheader("Plot Styling")
    font_size        = st.sidebar.slider("Font size", 8, 20, 12)
    axes_lw          = st.sidebar.slider("Axes linewidth", 0.5, 3.0, 1.0)
    tick_lw          = st.sidebar.slider("Tick width", 0.5, 3.0, 1.0)
    curve_lw         = st.sidebar.slider("Curve linewidth", 0.5, 5.0, 2.0)
    figsize_x        = st.sidebar.slider("Figure width", 8, 24, 18)
    figsize_y        = st.sidebar.slider("Figure height", 4, 12, 6)
    hspace           = st.sidebar.slider("Horizontal spacing", 0.1, 1.0, 0.3)
    legend_loc       = st.sidebar.selectbox("Legend location", ["best", "upper right", "upper left", "lower right", "lower left", "none"], index=0)
    cmap_choice_curves = st.sidebar.selectbox("Curve Colormap", plt.colormaps(), index=plt.colormaps().index("viridis"))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(figsize_x, figsize_y))
    fig.subplots_adjust(wspace=hspace)

    cmap = plt.get_cmap(cmap_choice_curves)
    colors = cmap(np.linspace(0, 1, len(st.session_state.history)))

    for idx, (c, data) in enumerate(st.session_state.history.items()):
        # Thickness
        times_th = [scale_time(t) for t, _, _, _, _, _ in data["thick"]]
        ths_nm   = [th * 1e9 for _, _, th, _, _, _ in data["thick"]]
        ax1.plot(times_th, ths_nm, label=f"c = {c:.3g}", color=colors[idx], lw=curve_lw)

        # Diagnostics
        tdiag = [scale_time(t) for t, _, _, _, _, _, _ in data["diag"]]
        bulk  = [b for _, _, _, _, b, _, _ in data["diag"]]
        grad  = [g for _, _, _, _, _, g, _ in data["diag"]]
        edl   = [e for _, _, _, _, _, _, e in data["diag"]]

        ax2.semilogy(tdiag, np.maximum(bulk, 1e-30), label=f"bulk", color=colors[idx], lw=curve_lw)
        ax2.semilogy(tdiag, np.maximum(grad, 1e-30), label=f"grad", color=colors[idx], ls='--', lw=curve_lw)
        if any(e != 0 for e in edl):
            ax3.plot(tdiag, edl, label=f"EDL boost", color=colors[idx], ls=':', lw=curve_lw)

    # Set styling for all axes
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Time (s)", fontsize=font_size)
        ax.tick_params(width=tick_lw, labelsize=font_size)
        ax.yaxis.set_tick_params(width=tick_lw, labelsize=font_size)
        for spine in ax.spines.values():
            spine.set_linewidth(axes_lw)
        ax.grid(True, alpha=0.3)

        # Force exponential format for x‑axis
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 0))

        # Legend on/off
        if legend_loc != "none":
            ax.legend(fontsize=font_size, loc=legend_loc)
        else:
            ax.legend_.remove() if ax.legend_ else None

    ax1.set_ylabel("Thickness (nm)", fontsize=font_size)
    ax2.set_ylabel("L²‑norm", fontsize=font_size)
    ax3.set_ylabel("EDL Boost", fontsize=font_size)

    st.pyplot(fig)

    # ------------------- EDL Decay Plot -------------------
    if use_edl:
        st.subheader("EDL Catalyst Decay")
        t_nd_range = np.linspace(0, n_steps * dt_nd, 200)
        lambda_t = [float(to_cpu(get_edl_factor(t, True, lambda0_edl, tau_edl_nd))) for t in t_nd_range]

        fig_decay, ax = plt.subplots(figsize=(figsize_x/2, figsize_y))
        ax.plot([scale_time(t) for t in t_nd_range], lambda_t, 'r-', lw=curve_lw)

        ax.set_xlabel("Time (s)", fontsize=font_size)
        ax.set_ylabel("λ_edl(t)", fontsize=font_size)
        ax.tick_params(width=tick_lw, labelsize=font_size)
        for spine in ax.spines.values():
            spine.set_linewidth(axes_lw)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.set_title(f"λ₀={lambda0_edl}, τ_edl={tau_edl_nd*tau0:.2e} s", fontsize=font_size)
        st.pyplot(fig_decay)

# ------------------- PLAYBACK -------------------
if st.session_state.history:
    st.header("Select Run for Playback")
    selected_c = st.selectbox(
        "Choose run",
        sorted(st.session_state.history.keys(), reverse=True),
        index=sorted(st.session_state.history.keys(), reverse=True).index(st.session_state.selected_c)
        if st.session_state.selected_c in st.session_state.history else 0
    )
    st.session_state.selected_c = selected_c
    data = st.session_state.history[selected_c]
    snaps, thick, diag, coords = data["snaps"], data["thick"], data["diag"], data["coords"]
    save_data = data.get("pkl")   # for PKL download

    # === ONE‑CLICK PKL DOWNLOAD ===
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
        st.info("Re‑run this simulation to generate downloadable .pkl")

    frame = st.slider("Frame", 0, len(snaps) - 1, len(snaps) - 1)
    auto = st.checkbox("Autoplay", False)
    interval = st.number_input("Interval (s)", 0.1, 5.0, 0.4, 0.1)
    field = st.selectbox("Field", ["phi (shell)", "c (concentration)", "psi (core)"])

    t, phi, c, psi = snaps[frame]
    t_real = scale_time(t)
    th_nm = thick[frame][2] * 1e9
    c_mean, c_max, total_ag, bulk_norm, grad_norm, edl_flux = diag[frame][1:]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(
            f"**c = {selected_c:.3g}** | **t = {t_real:.3e} s** | **Th = {th_nm:.2f} nm** | "
            f"**||bulk||₂ = {bulk_norm:.2e}** | **||grad||₂ = {grad_norm:.2e}** | **EDL boost = {edl_flux:.2e}**"
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        field_data = {"phi (shell)": phi, "c (concentration)": c, "psi (core)": psi}[field]
        vmin = 0
        vmax = 1 if field != "c (concentration)" else selected_c
        im = ax.imshow(
            field_data.T if mode == "2D (planar)" else field_data[field_data.shape[0] // 2],
            cmap=cmap_choice, vmin=vmin, vmax=vmax, origin='lower'
        )
        plt.colorbar(im, ax=ax, label=field.split()[0])
        ax.set_title(f"{field} @ t = {t_real:.3e} s")
        st.pyplot(fig)

    with col2:
        st.subheader("Thickness")
        times = [scale_time(t) for t, _, _, _, _, _ in thick]
        ths   = [th * 1e9 for _, _, th, _, _, _ in thick]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(times, ths, 'b-', lw=2)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("nm"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Diagnostics")
        df = pd.DataFrame(diag,
            columns=["t", "c_mean", "c_max", "total_Ag", "||bulk||₂", "||grad||₂", "EDL_boost"])
        st.dataframe(df.tail(10).style.format("{:.3e}"))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"diag_c_{selected_c:.3g}.csv", "text/csv")

    # ------------------- PHASE STATISTICS (new) -------------------
    st.subheader("Phase Statistics (current frame)")
    L_dom = 1.0
    dx = L_dom / (Nx - 1)
    dy = L_dom / (Ny - 1) if mode == "2D (planar)" else dx
    dz = L_dom / (Nz - 1) if mode != "2D (planar)" else 1.0
    stats = compute_phase_stats(phi, psi, dx, dy, dz, L0, phi_threshold)

    if mode == "2D (planar)":
        cols = st.columns(3)
        with cols[0]:
            st.metric("Electrolyte", f"{stats['Electrolyte'][0]:.3f} (nd²)",
                     help=f"Real: {stats['Electrolyte'][1]*1e18:.1f} nm²")
        with cols[1]:
            st.metric("Ag", f"{stats['Ag'][0]:.3f} (nd²)",
                     help=f"Real: {stats['Ag'][1]*1e18:.1f} nm²")
        with cols[2]:
            st.metric("Cu", f"{stats['Cu'][0]:.3f} (nd²)",
                     help=f"Real: {stats['Cu'][1]*1e18:.1f} nm²")
    else:
        cols = st.columns(3)
        with cols[0]:
            st.metric("Electrolyte", f"{stats['Electrolyte'][0]:.3f} (nd³)",
                     help=f"Real: {stats['Electrolyte'][1]*1e27:.1f} nm³")
        with cols[1]:
            st.metric("Ag", f"{stats['Ag'][0]:.3f} (nd³)",
                     help=f"Real: {stats['Ag'][1]*1e27:.1f} nm³")
        with cols[2]:
            st.metric("Cu", f"{stats['Cu'][0]:.3f} (nd³)",
                     help=f"Real: {stats['Cu'][1]*1e27:.1f} nm³")

    # ------------------- MATERIAL & POTENTIAL PROXY -------------------
    st.subheader("Material & Potential Proxy")
    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        material_method = st.selectbox(
            "Interpolation",
            ["phi + 2*psi (simple)", "phi*(1-psi) + 2*psi",
             "h·(phi² + psi²)", "h·(4*phi² + 2*psi²)", "max(phi, psi) + psi"],
            index=4                    # <-- default to max(phi,psi)+psi
        )
    with col_b:
        show_potential = st.checkbox("Show -α·c", True)
    with col_c:
        h_factor = st.slider("h", 0.1, 2.0, 0.5, 0.05) if "h·" in material_method else 1.0

    def build_material(phi, psi, method, h=1.0):
        phi, psi = np.array(phi), np.array(psi)
        if method == "phi + 2*psi (simple)":
            return phi + 2.0 * psi
        elif method == "phi*(1-psi) + 2*psi":
            return phi * (1.0 - psi) + 2.0 * psi
        elif method == "h·(phi² + psi²)":
            return h * (phi**2 + psi**2)
        elif method == "h·(4*phi² + 2*psi²)":
            return h * (4.0 * phi**2 + 2.0 * psi**2)
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
        fig_mat, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(material.T, origin='lower', extent=[0, 1, 0, 1],
                       cmap=cmap_mat, vmin=vmin_mat, vmax=vmax_mat)
        if "max" in material_method or "2*psi" in material_method:
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['electrolyte', 'Ag', 'Cu'])
        else:
            plt.colorbar(im, ax=ax, label="material")
        ax.set_title("Material")
        st.pyplot(fig_mat)

        if show_potential:
            fig_pot, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(potential.T, origin='lower', extent=[0, 1, 0, 1], cmap="RdBu_r")
            plt.colorbar(im, ax=ax, label="-α·c")
            ax.set_title("Potential Proxy")
            st.pyplot(fig_pot)
    else:
        cx = phi.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, sl, label in zip(axes,
                                 [material[cx, :, :], material[:, cx, :], material[:, :, cx]],
                                 ["x", "y", "z"]):
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
                    "potential": (-alpha_nd * c_s).ravel().astype(np.float32),
                    "EDL_boost": (to_cpu(get_edl_factor(t_nd, use_edl, lambda0_edl, tau_edl_nd)) * np.ones_like(phi_s)).ravel().astype(np.float32)
                }
                meshio.write_points_cells(fname, points, [], point_data=point_data)
                vtus.append(fname)

            pvd_path = os.path.join(tmpdir, "collection.pvd")
            with open(pvd_path, "w") as f:
                f.write('<VTKFile type="Collection" version="0.1">\n <Collection>\n')
                for idx, v in enumerate(vtus):
                    f.write(f'  <DataSet timestep="{scale_time(snaps[idx][0]):.3e}" file="{os.path.basename(v)}"/>\n')
                f.write(' </Collection>\n</VTKFile>\n')

            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in vtus + [pvd_path]:
                    zf.write(p, arcname=os.path.basename(p))
            zipbuf.seek(0)
            st.download_button(
                "Download VTU/PVD ZIP",
                zipbuf.read(),
                f"frames_{datetime.now():%Y%m%d_%H%M%S}.zip",
                "application/zip"
            )

    # ------------------- GIF -------------------
    if st.sidebar.button("Generate GIF") and GIF_AVAILABLE:
        with st.spinner("Generating GIF..."):
            fig, ax = plt.subplots(figsize=(6, 5))
            field_data = {"phi (shell)": phi, "c (concentration)": c, "psi (core)": psi}[field]
            vmin = 0
            vmax = 1 if field != "c (concentration)" else selected_c
            im = ax.imshow(
                field_data.T if mode == "2D (planar)" else field_data[field_data.shape[0] // 2],
                cmap=cmap_choice, vmin=vmin, vmax=vmax, origin='lower', animated=True
            )
            plt.colorbar(im, ax=ax)

            def animate(i):
                t_i, phi_i, c_i, psi_i = snaps[i]
                data = {"phi (shell)": phi_i, "c (concentration)": c_i, "psi (core)": psi_i}[field]
                im.set_array(data.T if mode == "2D (planar)" else data[data.shape[0] // 2])
                ax.set_title(f"{field} @ t = {scale_time(t_i):.3e} s")
                return [im]

            anim = FuncAnimation(fig, animate, frames=len(snaps), interval=200, blit=True)
            gif_buf = io.BytesIO()
            anim.save(gif_buf, writer='pillow', fps=5)
            gif_buf.seek(0)
            st.download_button(
                "Download GIF",
                gif_buf.read(),
                f"animation_{field.split()[0]}.gif",
                "image/gif"
            )

    # ------------------- PNG SNAPSHOT -------------------
    if st.sidebar.button("Download PNG Snapshot"):
        fig_snap, ax = plt.subplots(figsize=(6, 5))
        data = {"phi (shell)": phi, "c (concentration)": c, "psi (core)": psi}[field]
        im = ax.imshow(
            data.T if mode == "2D (planar)" else data[data.shape[0] // 2],
            cmap=cmap_choice, vmin=0, vmax=1 if field != "c (concentration)" else selected_c,
            origin='lower'
        )
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{field} @ t = {t_real:.3e} s")
        buf = io.BytesIO()
        fig_snap.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            "Download PNG",
            buf.read(),
            f"snapshot_{field.split()[0]}.png",
            "image/png"
        )

    if auto:
        for i in range(frame, len(snaps)):
            time.sleep(interval)
            st.rerun()

    col_clear, col_del = st.columns([1, 1])
    with col_clear:
        if st.button("Clear ALL"):
            st.session_state.history.clear()
            st.session_state.selected_c = None
            st.rerun()
    with col_del:
        if st.button(f"Delete c = {selected_c:.3g}"):
            del st.session_state.history[selected_c]
            st.session_state.selected_c = list(st.session_state.history.keys())[0] if st.session_state.history else None
            st.rerun()

else:
    st.info("Run a simulation (single or batch) to see results.")
