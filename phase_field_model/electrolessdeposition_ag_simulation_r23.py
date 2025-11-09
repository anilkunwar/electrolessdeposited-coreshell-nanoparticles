#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELECTROLESS Ag — DAMPED EDL + FULLY COMPATIBLE WITH ORIGINAL
* GPU (CuPy) when available → 10–20× speed-up
* CPU fallback (NumPy) → works on Streamlit Cloud
* Damped EDL: λ_edl ∈ [0,1] → λ_edl=0 recovers original behavior
* Batch, VTU/PVD/ZIP, GIF, PNG, CSV, material & potential proxy
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

# ------------------- SAFE GPU SETUP -------------------
GPU_AVAILABLE = False
try:
    import cupy as cp
    cp.cuda.Device(0).use()
    cp.zeros(1)
    GPU_AVAILABLE = True
    st.sidebar.success("GPU (CuPy) detected!")
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
st.set_page_config(page_title="Electroless Ag — Damped EDL", layout="wide")
st.title("Electroless Ag — Damped EDL Phase-Field Model")

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
n_steps = st.sidebar.slider("Steps", 50, max_steps, 1000, 50)
save_every = st.sidebar.slider("Save every", 1, 200, max(1, n_steps // 20), 1)

st.sidebar.header("Physics (nd)")
gamma_nd = st.sidebar.slider("γ", 1e-4, 0.5, 0.02, 1e-4)
beta_nd  = st.sidebar.slider("β", 0.1, 20.0, 4.0, 0.1)
k0_nd    = st.sidebar.slider("k₀", 0.01, 2.0, 0.4, 0.01)
M_nd     = st.sidebar.slider("M", 1e-3, 1.0, 0.2, 1e-3)
D_nd     = st.sidebar.slider("D", 0.0, 1.0, 0.05, 0.005)
alpha_nd = st.sidebar.slider("α (proxy)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.header("Damped EDL")
use_edl = st.sidebar.checkbox("Enable EDL", True)
if use_edl:
    lambda_edl = st.sidebar.slider("λ_edl (damping)", 0.0, 1.0, 0.1, 0.01, help="0 = no EDL, 1 = full EDL")
    kappa_cm = st.sidebar.slider("Debye length κ", 1e-8, 1e-6, 1e-7, 1e-8)
    alpha_edl = st.sidebar.slider("EDL α", 1e2, 1e5, 1e3, 1e2)
    eta_mob = st.sidebar.slider("Mobility boost η", 0.0, 5.0, 1.0, 0.1)
else:
    lambda_edl = kappa_cm = alpha_edl = eta_mob = 0.0

use_fft = st.sidebar.checkbox("Use FFT Laplacian", GPU_AVAILABLE)
cmap_choice = st.sidebar.selectbox("Colormap", CMAPS, CMAPS.index("viridis"))

st.sidebar.header("Geometry")
core_radius_frac = st.sidebar.slider("Core / L", 0.05, 0.45, 0.18, 0.01)
shell_thickness_frac = st.sidebar.slider("Δr / r_core", 0.05, 0.6, 0.2, 0.01)
phi_threshold = st.sidebar.slider("φ threshold", 0.1, 0.9, 0.5, 0.05)

growth_model = st.sidebar.selectbox("Model", ["Model A (irreversible)", "Model B (soft reversible)"])

st.sidebar.header("Physical Scales")
L0 = st.sidebar.number_input("L₀ (nm)", 1.0, 1e6, 20.0) * 1e-7  # nm → cm
tau0 = st.sidebar.number_input("τ₀ (×10⁻⁴ s)", 1e-6, 1e6, 1.0) * 1e-4

run_batch_button = st.sidebar.button("Run BATCH")
run_single_button = st.sidebar.button("Run SINGLE (last c)")

# ------------------- HELPERS -------------------
def scale_time(t): return t * tau0
def nd_to_real(x): return x * L0
def to_cpu(arr): return cp.asnumpy(arr) if GPU_AVAILABLE else arr
def to_gpu(arr): return cp.asarray(arr) if GPU_AVAILABLE else arr

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

# ------------------- SIMULATION CORE -------------------
def run_simulation(c_bulk_val):
    L = 1.0
    dx = L / (Nx - 1)
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

    # EDL scaling
    kappa_nd = kappa_cm / L0 if use_edl and kappa_cm > 0 else 0.0
    alpha_nd_edl = alpha_edl * 1e-3 * tau0 / (L0 * 1e7) if use_edl else 0.0
    mu_nd = D_nd * 38.9  # Einstein relation (z=1, 298K)

    for step in range(n_steps + 1):
        t = step * dt_nd

        grad_phi = cp.gradient(phi, dx)
        gphi = cp.sqrt(sum(g**2 for g in grad_phi) + 1e-30)
        delta_int = 6 * phi * (1 - phi) * (1 - psi) * gphi
        delta_int = cp.clip(delta_int, 0, 6 / max(eps, dx))
        f_bulk = 2 * beta_nd * phi * (1 - phi) * (1 - 2 * phi)
        i_loc = k0_nd * c * (1 - phi) * (1 - psi) * delta_int
        i_loc = cp.clip(i_loc, 0, 1e6)

        lap_phi = laplacian(phi, dx)
        dep = M_nd * i_loc
        curv = M_nd * gamma_nd * lap_phi
        M_local = M_nd * (1 + eta_mob * phi)
        dphi = dt_nd * M_local * (dep + cp.maximum(curv, 0) - softness * f_bulk)
        if softness == 0:
            dphi = cp.maximum(dphi, 0)
        phi = cp.clip(phi + dphi, 0, 1)

        # CONCENTRATION: DAMPED EDL
        lap_c = laplacian(c, dx)
        div_D_grad_c = D_nd * lap_c

        div_mu_c_grad_V = 0.0
        if use_edl and lambda_edl > 0 and kappa_nd > 0:
            lap2_c = laplacian(lap_c, dx)
            V = -alpha_nd_edl * c + kappa_nd * lap2_c
            grad_V = cp.gradient(V, dx)
            flux_edl = sum(mu_nd * cp.maximum(c, 1e-12) * g for g in grad_V)
            div_mu_c_grad_V = lambda_edl * flux_edl  # DAMPED

        c += dt_nd * (div_D_grad_c + div_mu_c_grad_V - i_loc)
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
        edl_flux = float(cp.sum(div_mu_c_grad_V)) if use_edl else 0.0

        if step % save_every == 0 or step == n_steps:
            phi_cpu = to_cpu(phi)
            psi_cpu = to_cpu(psi)
            c_cpu = to_cpu(c)

            Xc, Yc, Zc = np.meshgrid(
                x, y if mode == "2D (planar)" else x,
                x if mode != "2D (planar)" else [0.5], indexing='ij'
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

    return c_bulk_val, snapshots, diags, thick, coords

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
        for c, s, d, t, co in results:
            st.session_state.history[c] = {"snaps": s, "diag": d, "thick": t, "coords": co}
        if results:
            st.session_state.selected_c = results[0][0]
        st.success(f"Batch done: {len(results)} runs")

if run_single_button and selected_labels:
    c_val = c_options[selected_labels[-1]]
    with st.spinner(f"Running SINGLE c = {c_val}..."):
        c, s, d, t, co = run_simulation(c_val)
        st.session_state.history[c] = {"snaps": s, "diag": d, "thick": t, "coords": co}
        st.session_state.selected_c = c
        st.success("Done")

# ------------------- BATCH COMPARISON -------------------
if len(st.session_state.history) > 1:
    st.header("Batch Comparison")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    cmap = plt.get_cmap(cmap_choice)
    colors = cmap(np.linspace(0, 1, len(st.session_state.history)))

    for idx, (c, data) in enumerate(st.session_state.history.items()):
        times_th = [scale_time(t) for t, _, _, _, _, _ in data["thick"]]
        ths_nm   = [th * 1e9 for _, _, th, _, _, _ in data["thick"]]
        ax1.plot(times_th, ths_nm, label=f"c = {c:.3g}", color=colors[idx], lw=2)

        tdiag = [scale_time(t) for t, _, _, _, _, _, _ in data["diag"]]
        bulk  = [b for _, _, _, _, b, _, _ in data["diag"]]
        grad  = [g for _, _, _, _, _, g, _ in data["diag"]]
        edl   = [e for _, _, _, _, _, _, e in data["diag"]]
        ax2.semilogy(tdiag, np.maximum(bulk, 1e-30), label=f"bulk", color=colors[idx], lw=1.5)
        ax2.semilogy(tdiag, np.maximum(grad, 1e-30), label=f"grad", color=colors[idx], ls='--', lw=1.5)
        if any(e != 0 for e in edl):
            ax3.plot(tdiag, edl, label=f"EDL flux", color=colors[idx], ls=':', lw=2)

    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Thickness (nm)"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("L²-norm"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("EDL Flux"); ax3.legend(); ax3.grid(True, alpha=0.3)
    st.pyplot(fig)

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
            f"**||bulk||₂ = {bulk_norm:.2e}** | **||grad||₂ = {grad_norm:.2e}** | **EDL = {edl_flux:.2e}**"
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
            columns=["t", "c_mean", "c_max", "total_Ag", "||bulk||₂", "||grad||₂", "EDL_flux"])
        st.dataframe(df.tail(10).style.format("{:.3e}"))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"diag_c_{selected_c:.3g}.csv", "text/csv")

    # ------------------- MATERIAL & POTENTIAL -------------------
    st.subheader("Material & Potential Proxy")
    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        material_method = st.selectbox(
            "Interpolation",
            ["phi + 2*psi (simple)", "phi*(1-psi) + 2*psi",
             "h·(phi² + psi²)", "h·(4*phi² + 2*psi²)", "max(phi, psi) + psi"],
            index=3
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
