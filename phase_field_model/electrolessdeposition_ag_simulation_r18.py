#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Electroless Ag — SINGLE run (dropdown c) – NUMBA-SAFE
Now with:
  • bulk vs. gradient term diagnostics
  • material-composition + electric-potential proxy
  • multi-run history + combined thickness & norm plots
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# ------------------- NUMBA -------------------
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ------------------- PAGE -------------------
st.set_page_config(page_title="Electroless Ag — Single (Dropdown)", layout="wide")
st.title("Electroless Ag — Single Simulation (Dropdown c)")

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 140, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
})
CMAPS = [c for c in plt.colormaps() if c not in {"jet", "jet_r"}]

# ------------------- SIDEBAR -------------------
st.sidebar.header("Molar Fraction [Ag]/[Cu]")
c_options = {
    "1.0 (1:1)": 1.0, "0.5 (1:2)": 0.5, "0.333 (1:3)": 0.333,
    "0.25 (1:4)": 0.25, "0.2 (1:5)": 0.2, "0.1 (1:10)": 0.1
}
c_label = st.sidebar.selectbox("Choose c_bulk", list(c_options.keys()))
c_bulk_nd = c_options[c_label]

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
alpha_nd = st.sidebar.slider("α (coupling)", 0.0, 10.0, 2.0, 0.1)   # NEW

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

run_button = st.sidebar.button("Run Simulation")

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

# ------------------- 2D SIMULATION -------------------
def run_2d():
    L = 1.0; dx = L/(Nx-1)
    x = np.linspace(0, L, Nx); y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2)

    psi = (dist <= core_radius_frac*L).astype(np.float64)
    r_core = core_radius_frac * L
    r_outer = r_core * (1 + shell_thickness_frac)
    phi = np.where(dist <= r_core, 0.0, np.where(dist <= r_outer, 1.0, 0.0))
    eps = max(4*dx, 1e-6)
    phi = phi * (1 - 0.5*(1 - np.tanh((dist-r_core)/eps))) \
              * (1 - 0.5*(1 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0, 1)

    c = c_bulk_nd * (Y/L) * (1 - phi) * (1 - psi) if bc_type == "Neumann (zero flux)" else \
        c_bulk_nd * (1 - phi) * (1 - psi)
    c = np.clip(c, 0, c_bulk_nd)

    snapshots, diags, thick = [], [], []
    softness = 0.01 if "B" in growth_model else 0.0
    max_th = 0.0

    for step in range(n_steps + 1):
        t = step * dt_nd

        # ---- phi ----
        gx, gy = np.gradient(phi, dx)
        gphi = np.sqrt(gx**2 + gy**2 + 1e-30)
        delta_int = 6*phi*(1-phi)*(1-psi)*gphi
        delta_int = np.clip(delta_int, 0, 6/max(eps,dx))
        f_bulk = 2 * beta_nd * phi * (1 - phi) * (1 - 2 * phi)               # double-well
        i_loc = k0_nd * c * (1-phi) * (1-psi) * delta_int
        i_loc = np.clip(i_loc, 0, 1e6)

        lap_phi = lap2d(phi, dx)
        dep = M_nd * i_loc
        curv = M_nd * gamma_nd * lap_phi
        dphi = dt_nd * (dep + np.maximum(curv, 0) - softness * M_nd * f_bulk)
        if softness == 0: dphi = np.maximum(dphi, 0)
        phi = np.clip(phi + dphi, 0, 1)

        # ---- c ----
        lap_c = lap2d(c, dx)
        c += dt_nd * (D_nd * lap_c - i_loc)
        c = np.clip(c, 0, c_bulk_nd)
        if bc_type == "Neumann (zero flux)":
            c[:,-1] = c_bulk_nd
        else:
            c[[0,-1],:] = c[:,[0,-1]] = c_bulk_nd

        # ---- diagnostics (bulk / grad norms) ----
        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm = np.sqrt(np.mean((M_nd*gamma_nd*lap_phi)**2))

        # ---- save ----
        if step % save_every == 0 or step == n_steps:
            th_nd = compute_thickness(phi, psi, (x, y, np.array([0.5])), core_radius_frac, phi_threshold)
            max_th = max(max_th, th_nd)
            c_mean, c_max = np.mean(c), np.max(c)
            total_ag = np.sum(i_loc) * dt_nd

            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            #diags.append((t, c_mean, c_max, total_ag, bulk_norm, grad_norm))
            diags.append((t, c_mean, c_max, total_ag, float(bulk_norm), float(grad_norm)))
            thick.append((t, max_th, nd_to_real(max_th), c_mean, c_max, total_ag))

    return snapshots, diags, thick, (x, y)

# ------------------- 3D SIMULATION -------------------
def run_3d():
    L = 1.0; dx = L/(Nx-1)
    x = np.linspace(0, L, Nx)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)

    psi = (dist <= core_radius_frac*L).astype(np.float64)
    r_core = core_radius_frac * L
    r_outer = r_core * (1 + shell_thickness_frac)
    phi = np.where(dist <= r_core, 0.0, np.where(dist <= r_outer, 1.0, 0.0))
    eps = max(4*dx, 1e-6)
    phi = phi * (1 - 0.5*(1 - np.tanh((dist-r_core)/eps))) \
              * (1 - 0.5*(1 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0, 1)

    c = c_bulk_nd * (Z/L) * (1 - phi) * (1 - psi) if bc_type == "Neumann (zero flux)" else \
        c_bulk_nd * (1 - phi) * (1 - psi)
    c = np.clip(c, 0, c_bulk_nd)

    snapshots, diags, thick = [], [], []
    softness = 0.01 if "B" in growth_model else 0.0
    max_th = 0.0

    for step in range(n_steps + 1):
        t = step * dt_nd

        grad = np.gradient(phi, dx)
        gphi = np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2 + 1e-30)
        delta_int = 6*phi*(1-phi)*(1-psi)*gphi
        delta_int = np.clip(delta_int, 0, 6/max(eps,dx))
        f_bulk = 2 * beta_nd * phi * (1 - phi) * (1 - 2 * phi)
        i_loc = k0_nd * c * (1-phi) * (1-psi) * delta_int
        i_loc = np.clip(i_loc, 0, 1e6)

        lap_phi = lap3d(phi, dx)
        dep = M_nd * i_loc
        curv = M_nd * gamma_nd * lap_phi
        dphi = dt_nd * (dep + np.maximum(curv, 0) - softness * M_nd * f_bulk)
        if softness == 0: dphi = np.maximum(dphi, 0)
        phi = np.clip(phi + dphi, 0, 1)

        lap_c = lap3d(c, dx)
        c += dt_nd * (D_nd * lap_c - i_loc)
        c = np.clip(c, 0, c_bulk_nd)
        if bc_type == "Neumann (zero flux)":
            c[:,:,-1] = c_bulk_nd
        else:
            c[[0,-1],:,:] = c[:,[0,-1],:] = c[:,:,[0,-1]] = c_bulk_nd

        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm = np.sqrt(np.mean((M_nd*gamma_nd*lap_phi)**2))

        if step % save_every == 0 or step == n_steps:
            th_nd = compute_thickness(phi, psi, (x, x, x), core_radius_frac, phi_threshold)
            max_th = max(max_th, th_nd)
            c_mean, c_max = np.mean(c), np.max(c)
            total_ag = np.sum(i_loc) * dt_nd

            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diags.append((t, c_mean, c_max, total_ag, bulk_norm, grad_norm))
            thick.append((t, max_th, nd_to_real(max_th), c_mean, c_max, total_ag))

    return snapshots, diags, thick, (x, x, x)

# ------------------- HISTORY INITIALISATION -------------------
if "history" not in st.session_state:
    st.session_state.history = {}          # {c_bulk: {"snaps":..., "diag":..., "thick":..., "coords":...}}
if "selected_c" not in st.session_state:
    st.session_state.selected_c = None

# ------------------- RUN -------------------
if run_button:
    with st.spinner(f"Running c = {c_bulk_nd:.3g} …"):
        t0 = time.time()
        if mode == "2D (planar)":
            snaps, diag, thick, coords = run_2d()
        else:
            snaps, diag, thick, coords = run_3d()

        st.session_state.history[c_bulk_nd] = {
            "snaps": snaps,
            "diag": diag,
            "thick": thick,
            "coords": coords
        }
        st.session_state.selected_c = c_bulk_nd
        st.success(f"Done in {time.time()-t0:.2f}s — {len(snaps)} frames")

# ------------------- COMBINED THICKNESS PLOT -------------------
if len(st.session_state.history) > 1:
    st.header("Combined Thickness Evolution")
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap(cmap_choice)
    colors = cmap(np.linspace(0, 1, len(st.session_state.history)))

    for idx, (c, data) in enumerate(st.session_state.history.items()):
        times = [scale_time(t) for t, _, _, _, _, _ in data["thick"]]
        ths   = [th*1e9 for _, _, th, _, _, _ in data["thick"]]
        ax.plot(times, ths, label=f"c = {c:.3g}", color=colors[idx], lw=2)

    ax.set_xlabel("Time (s)"); ax.set_ylabel("Thickness (nm)")
    ax.legend(title="c_bulk", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ------------------- COMBINED BULK/GRAD NORM PLOT -------------------
if len(st.session_state.history) > 1:
    st.header("Combined Bulk vs. Gradient Norms")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap(cmap_choice)
    colors = cmap(np.linspace(0, 1, len(st.session_state.history)))

    for idx, (c, data) in enumerate(st.session_state.history.items()):
        times = [scale_time(t) for t, _, _, _, _, _, _ in data["diag"]]
        bulk  = [b for _, _, _, _, b, _ in data["diag"]]
        grad  = [g for _, _, _, _, _, g in data["diag"]]
        ax2.semilogy(times, np.maximum(bulk, 1e-30), label=f"c={c:.3g} bulk", color=colors[idx], lw=1.5)
        ax2.semilogy(times, np.maximum(grad, 1e-30), label=f"c={c:.3g} grad", color=colors[idx], ls='--', lw=1.5)

    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("L²-norm")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# ------------------- SELECT RUN FOR PLAYBACK -------------------
if st.session_state.history:
    st.header("Select Run for Detailed Playback")
    selected_c = st.selectbox(
        "Choose a run",
        options=sorted(st.session_state.history.keys(), reverse=True),
        index=sorted(st.session_state.history.keys(), reverse=True).index(st.session_state.selected_c)
        if st.session_state.selected_c in st.session_state.history else 0
    )
    st.session_state.selected_c = selected_c

    data = st.session_state.history[selected_c]
    snaps = data["snaps"]
    thick = data["thick"]
    diag  = data["diag"]
    coords = data["coords"]

    # ---- playback controls ----
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
                 f"**c_mean = {c_mean:.3f}** | **Total Ag = {total_ag:.2e}** | "
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
        st.subheader("Thickness (this run)")
        times = [scale_time(t) for t,_,_,_,_,_ in thick]
        ths   = [th*1e9 for _,_,th,_,_,_ in thick]
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(times, ths, 'b-', lw=2)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("nm")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Diagnostics (last 10 rows)")
        df = pd.DataFrame(diag,
                          columns=["t", "c_mean", "c_max", "total_Ag", "||bulk||₂", "||grad||₂"])
        st.dataframe(df.tail(10).style.format("{:.3e}"))

        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV (this run)",
            csv,
            f"diagnostics_c_{selected_c:.3g}.csv",
            "text/csv"
        )

    # ------------------- MATERIAL & POTENTIAL POST-PROCESSOR -------------------
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
            index=3
        )
    with col_b:
        show_potential = st.checkbox("Overlay electric-potential proxy (-α·c)", value=True)
    with col_c:
        if "h·" in material_method:
            h_factor = st.slider("h (scaling)", 0.1, 2.0, 0.5, 0.05)
        else:
            h_factor = 1.0

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

    material = build_material(phi, psi, material_method, h=h_factor)
    potential = -alpha_nd * c

    if material_method in ["phi + 2*psi (simple)",
                           "phi*(1-psi) + 2*psi",
                           "max(phi, psi) + psi"]:
        cmap_mat = plt.cm.get_cmap("Set1", 3)
        vmin_mat, vmax_mat = 0, 2
    else:
        cmap_mat = cmap_choice
        vmin_mat = vmax_mat = None

    if mode == "2D (planar)":
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

    else:   # 3-D
        cx = phi.shape[0]//2; cy = phi.shape[1]//2; cz = phi.shape[2]//2
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

    # ---- autoplay ----
    if auto:
        for i in range(frame, len(snaps)):
            time.sleep(interval)
            st.rerun()

    # ---- management buttons ----
    col_clear, col_del = st.columns([1, 1])
    with col_clear:
        if st.button("Clear ALL runs"):
            st.session_state.history.clear()
            st.session_state.selected_c = None
            st.experimental_rerun()
    with col_del:
        if st.button(f"Delete run c = {selected_c:.3g}"):
            del st.session_state.history[selected_c]
            if not st.session_state.history:
                st.session_state.selected_c = None
            else:
                st.session_state.selected_c = list(st.session_state.history.keys())[0]
            st.experimental_rerun()

else:
    st.info("Run at least one simulation to see results.")
