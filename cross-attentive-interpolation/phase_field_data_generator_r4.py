#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELECTROLESS Ag — FULLY UPGRADED SIMULATOR WITH MODIFIED DEPOSITION
* Modified deposition rate: max(φ, ψ) + ψ
* Robust PKL generator with batch variants
* Integrated postprocessed metrics: refined thickness, deposition rates, nucleations, onset
* Fixed 3D grid, visualization, and playback
* Batch comparison, playback, exports (PNG, GIF, CSV, ZIP)
* Auto-saves .pkl with all data for interpolator
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import io
import zipfile
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from scipy.ndimage import distance_transform_edt, label

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
st.set_page_config(page_title="Electroless Ag — Upgraded Simulator", layout="wide")
st.title("Electroless Ag — Fully Upgraded Simulator with Modified Deposition")

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 140, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
})
CMAPS = [c for c in plt.colormaps() if c not in {"jet", "jet_r"}]

# ------------------- SIDEBAR -------------------
st.sidebar.header("Batch Variants Control (Shell Growth Influencers)")

st.sidebar.subheader("c_bulk Variants (Concentration)")
c_options = {
    "1.0 (1:1)": 1.0, "0.5 (1:2)": 0.5, "0.333 (1:3)": 0.333,
    "0.25 (1:4)": 0.25, "0.2 (1:5)": 0.2, "0.1 (1:10)": 0.1
}
selected_c_labels = st.sidebar.multiselect(
    "Choose c_bulk values",
    list(c_options.keys()),
    default=["1.0 (1:1)", "0.5 (1:2)", "0.2 (1:5)"]
)
c_bulk_list = [c_options[l] for l in selected_c_labels]

st.sidebar.subheader("Domain Size Variants")
domain_variants = st.sidebar.multiselect(
    "Choose domain multipliers",
    [1.0, 1.5, 2.0, 2.5, 3.0],
    default=[1.0, 2.0],
    help="Larger domains reduce boundary effects on diffusion-limited growth"
)
scale_core_with_domain = st.sidebar.checkbox("Scale core/shell with domain?", False)

st.sidebar.subheader("k0_nd Variants (Kinetic Coefficient)")
k0_variants = st.sidebar.multiselect(
    "Choose k0_nd values",
    [0.2, 0.4, 0.6, 0.8, 1.0],
    default=[0.4],
    help="Higher k0 increases reaction rate, accelerating shell growth"
)

st.sidebar.subheader("D_nd Variants (Diffusion Coefficient)")
D_variants = st.sidebar.multiselect(
    "Choose D_nd values",
    [0.01, 0.05, 0.1, 0.2],
    default=[0.05],
    help="Higher D enhances mass transport, influencing growth regime"
)

st.sidebar.subheader("gamma_nd Variants (Interfacial Energy)")
gamma_variants = st.sidebar.multiselect(
    "Choose gamma_nd values",
    [0.01, 0.02, 0.05, 0.1],
    default=[0.02],
    help="Higher gamma stabilizes interfaces, affecting morphology"
)

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
early_stop_thick_tol = st.sidebar.number_input("Early stop if Δthickness < tol (nd)", 0.0, 1e-3, 1e-5, format="%.1e")

st.sidebar.header("Physics (nd) — Fixed for Batch")
beta_nd = st.sidebar.slider("β", 0.1, 20.0, 4.0, 0.1)
M_nd = st.sidebar.slider("M", 1e-3, 1.0, 0.2, 1e-3)
alpha_nd = st.sidebar.slider("α (coupling)", 0.0, 10.0, 2.0, 0.1)
use_fft = st.sidebar.checkbox("Use FFT Laplacian", GPU_AVAILABLE)
cmap_choice = st.sidebar.selectbox("Colormap", CMAPS, CMAPS.index("viridis"))

st.sidebar.header("Geometry — Fixed for Batch")
core_radius_frac = st.sidebar.slider("Core / L_base", 0.05, 0.45, 0.18, 0.01)
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

run_batch_button = st.sidebar.button("Run BATCH (All Variants)")
run_single_button = st.sidebar.button("Run SINGLE (default settings)")

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
        ndim = u.ndim
        freqs = [cp.fft.fftfreq(s, d=1.0/s) * 2 * cp.pi for s in u.shape]
        K = cp.meshgrid(*freqs, indexing='ij')
        K2 = sum(k**2 for k in K)
        u_hat = cp.fft.fftn(u)
        return cp.fft.ifftn(-K2 * u_hat).real
    else:
        shifts = [(1,0), (-1,0), (0,1), (0,-1)]
        if u.ndim == 3:
            shifts += [(0,0,1), (0,0,-1)]
        lap = sum(cp.roll(u, shift, axis=list(range(u.ndim))) for shift in shifts) - 2 * len(shifts) // 2 * u
        return lap / (dx * dx)

# ------------------- POSTPROCESSED METRICS HELPERS -------------------
def compute_material_map(phi_cpu, psi_cpu):
    return np.maximum(phi_cpu, psi_cpu) + psi_cpu

def compute_refined_thickness(m, dx):
    shell_mask = (m == 1)
    if not np.any(shell_mask):
        return 0.0
    core_mask = (m == 2)
    dist_to_core = distance_transform_edt(~core_mask) * dx
    thickness_map = dist_to_core * shell_mask
    return np.max(thickness_map)

def detect_nucleations(m, min_area=5):
    shell_mask = (m == 1)
    labeled_shell, num_components = label(shell_mask)
    if num_components == 0:
        return 0
    areas = np.bincount(labeled_shell.ravel())[1:]
    nucleations = np.sum(areas < min_area) if len(areas) > 0 else 0
    return nucleations

def compute_deposition_rates(thick_history_nd, dt_nd):
    if len(thick_history_nd) < 2:
        return []
    times = np.array([entry[0] for entry in thick_history_nd])
    ths = np.array([entry[1] for entry in thick_history_nd])
    rates = np.diff(ths) / np.diff(times) / dt_nd
    return rates.tolist()

# ------------------- MODIFIED SIMULATION CORE -------------------
def run_simulation(params):
    try:
        c_bulk_val, domain_mult_val, k0_nd_val, D_nd_val, gamma_nd_val = params
        L_base = 1.0
        L = L_base * domain_mult_val
        dx = L / max(Nx, Ny, Nz)
        center = L / 2.0

        x = np.linspace(0, L, Nx)
        if mode == "2D (planar)":
            y = np.linspace(0, L, Ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            Xg, Yg = to_gpu(X), to_gpu(Y)
            dist = cp.sqrt((Xg - center)**2 + (Yg - center)**2)
            coords = (x, y)
            shape = (Nx, Ny)
        else:
            y = np.linspace(0, L, Ny)
            z = np.linspace(0, L, Nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            Xg, Yg, Zg = to_gpu(X), to_gpu(Y), to_gpu(Z)
            dist = cp.sqrt((Xg - center)**2 + (Yg - center)**2 + (Zg - center)**2)
            coords = (x, y, z)
            shape = (Nx, Ny, Nz)

        r_core_base = core_radius_frac * L_base
        r_core = r_core_base * domain_mult_val if scale_core_with_domain else r_core_base
        r_outer = r_core * (1 + shell_thickness_frac)
        psi = (dist <= r_core).astype(cp.float64)
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
        material_snaps = []
        nucleation_counts = []
        softness = 0.01 if "B" in growth_model else 0.0
        max_th = prev_th = 0.0
        onset_time = None
        initial_m = compute_material_map(to_cpu(phi), to_cpu(psi))
        initial_ag_volume = np.sum(initial_m == 1) * (dx ** len(shape))

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

            # ========== MODIFIED DEPOSITION RATE ==========
            deposition_factor = cp.maximum(phi, psi) + psi  # max(phi, psi) + psi
            i_loc = k0_nd_val * c * deposition_factor * delta_int * edl_boost
            i_loc = cp.clip(i_loc, 0, 1e6)

            lap_phi = laplacian(phi, dx)
            dep = M_nd * i_loc
            curv = M_nd * gamma_nd_val * lap_phi
            dphi = dt_nd * (dep + cp.maximum(curv, 0) - softness * M_nd * f_bulk)
            if softness == 0:
                dphi = cp.maximum(dphi, 0)
            phi = cp.clip(phi + dphi, 0, 1)

            lap_c = laplacian(c, dx)
            c += dt_nd * (D_nd_val * lap_c - i_loc)
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
            grad_norm = cp.sqrt(cp.mean((M_nd * gamma_nd_val * lap_phi)**2))
            edl_flux = float(cp.sum(edl_boost - 1.0)) if use_edl else 0.0

            if step % save_every == 0 or step == n_steps:
                phi_cpu = to_cpu(phi)
                psi_cpu = to_cpu(psi)
                c_cpu = to_cpu(c)

                yc_use = y if mode == "2D (planar)" else y
                zc_use = np.array([center]) if mode == "2D (planar)" else z
                Xc, Yc, Zc = np.meshgrid(x, yc_use, zc_use, indexing='ij')
                dist_cpu = np.sqrt((Xc - center)**2 + (Yc - center)**2 + (Zc - center)**2)
                mask = (phi_cpu > phi_threshold) & (psi_cpu < 0.5)
                th_nd_original = np.max(dist_cpu[mask]) - r_core if np.any(mask) else 0.0

                m = compute_material_map(phi_cpu, psi_cpu)
                th_nd_refined = compute_refined_thickness(m, dx)
                num_nucleations = detect_nucleations(m)
                current_ag_volume = np.sum(m == 1) * (dx ** len(shape))
                if onset_time is None and current_ag_volume > initial_ag_volume * 1.01:
                    onset_time = t
                nucleation_counts.append((t, num_nucleations))

                max_th = max(max_th, th_nd_refined)
                c_mean = float(cp.mean(c))
                c_max = float(cp.max(c))
                total_ag = float(cp.sum(i_loc) * dt_nd)

                snapshots.append((t, phi_cpu, c_cpu, psi_cpu))
                diags.append((t, c_mean, c_max, total_ag, float(bulk_norm), float(grad_norm), edl_flux))
                thick.append((t, th_nd_refined, nd_to_real(th_nd_refined), c_mean, c_max, total_ag))

            if abs(max_th - prev_th) < early_stop_thick_tol and step > 10:
                break
            prev_th = max_th

        deposition_rates = compute_deposition_rates(thick, dt_nd)

        os.makedirs("electroless_pkl_solutions", exist_ok=True)

        bc_str = "Neu" if bc_type == "Neumann (zero flux)" else "Dir"
        mode_str = "2D" if mode == "2D (planar)" else "3D"
        edl_str = f"EDL{lambda0_edl:.1f}" if use_edl else "noEDL"
        dom_str = f"dom{domain_mult_val:.2f}".replace(".", "p")
        k0_str = f"k{k0_nd_val:.2f}".replace(".", "p")
        D_str = f"D{D_nd_val:.3f}".replace(".", "p")
        gamma_str = f"g{gamma_nd_val:.3f}".replace(".", "p")

        filename = f"Ag_MOD_{mode_str}_c{c_bulk_val:.3f}_{bc_str}_{edl_str}_{k0_str}_M{M_nd:.2f}_{D_str}_{gamma_str}_{dom_str}_N{Nx}x{Ny}x{Nz}_steps{step}.pkl"
        filepath = os.path.join("electroless_pkl_solutions", filename)

        save_data = {
            "meta": {
                "c_bulk": float(c_bulk_val),
                "domain_multiplier": float(domain_mult_val),
                "k0_nd": float(k0_nd_val),
                "D_nd": float(D_nd_val),
                "gamma_nd": float(gamma_nd_val),
                "bc_type": bc_type,
                "mode": mode,
                "use_edl": use_edl,
                "scale_core_with_domain": scale_core_with_domain,
                "timestamp": datetime.now().isoformat(),
                "deposition_model": "MODIFIED_max_phi_psi_plus_psi",
                "onset_time_nd": onset_time if onset_time else n_steps * dt_nd,
            },
            "parameters": {
                "beta_nd": float(beta_nd),
                "M_nd": float(M_nd), "alpha_nd": float(alpha_nd),
                "lambda0_edl": float(lambda0_edl),
                "tau_edl_nd": float(tau_edl_nd), "alpha_edl": float(alpha_edl),
                "core_radius_frac": float(core_radius_frac),
                "shell_thickness_frac": float(shell_thickness_frac),
                "phi_threshold": float(phi_threshold),
                "Nx": Nx, "Ny": Ny, "Nz": Nz,
                "n_steps": step, "dt_nd": float(dt_nd),
                "L0_m": float(L0), "tau0_s": float(tau0),
                "early_stop_thick_tol": float(early_stop_thick_tol),
            },
            "coords_nd": coords,
            "snapshots": snapshots,
            "diagnostics": diags,
            "thickness_history_nm": thick,
            "growth_metrics": {
                "onset_time_nd": onset_time if onset_time else None,
                "deposition_rates_nd_per_unit_time": deposition_rates,
                "nucleation_counts": nucleation_counts,
            }
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return params, snapshots, diags, thick, coords, filepath, filename

    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return params, None, None, None, None, None, None

# ------------------- HISTORY & RUN -------------------
if "history" not in st.session_state:
    st.session_state.history = {}
if "selected_run" not in st.session_state:
    st.session_state.selected_run = None

def get_run_key(params):
    c, dom, k0, D, gamma = params
    return f"c{c:.3f}_dom{dom:.2f}_k0{k0:.2f}_D{D:.3f}_gamma{gamma:.3f}"

if run_batch_button:
    variants = [(c, dom, k0, D, gamma) for c in c_bulk_list for dom in domain_variants for k0 in k0_variants for D in D_variants for gamma in gamma_variants]
    total_runs = len(variants)
    
    if total_runs == 0:
        st.warning("Please select at least one variant for each parameter to run batch simulation.")
    elif total_runs > 50:
        st.warning(f"Too many variants ({total_runs}). Please select fewer combinations (max 50 recommended).")
    else:
        with st.spinner(f"Running BATCH: {total_runs} variants..."):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, params in enumerate(variants):
                try:
                    status_text.text(f"Running {i+1}/{total_runs}: c={params[0]}, dom={params[1]}, k0={params[2]}, D={params[3]}, gamma={params[4]}")
                    result = run_simulation(params)
                    results.append(result)
                    progress_bar.progress((i + 1) / total_runs)
                except Exception as e:
                    st.error(f"Simulation failed for params {params}: {str(e)}")
                    results.append((params, None, None, None, None, None, None))
            
            # Process successful results
            successful_runs = 0
            for p, s, d, t, co, fp, fn in results:
                if s is not None and len(s) > 0:
                    key = get_run_key(p)
                    st.session_state.history[key] = {
                        "params": p, "snaps": s, "diag": d, "thick": t, 
                        "coords": co, "pkl_path": fp, "pkl_name": fn
                    }
                    successful_runs += 1
            
            if successful_runs > 0:
                st.success(f"Batch completed: {successful_runs}/{total_runs} successful runs")
                # Auto-select the first successful run
                first_successful = next((r for r in results if r[1] is not None and len(r[1]) > 0), None)
                if first_successful:
                    st.session_state.selected_run = get_run_key(first_successful[0])
            else:
                st.error("No simulations completed successfully. Check parameters and error messages.")

if run_single_button:
    default_params = (1.0, 1.0, 0.4, 0.05, 0.02)
    with st.spinner("Running SINGLE..."):
        p, s, d, t, co, fp, fn = run_simulation(default_params)
        if s is not None and len(s) > 0:
            key = get_run_key(p)
            st.session_state.history[key] = {"params": p, "snaps": s, "diag": d, "thick": t, "coords": co, "pkl_path": fp, "pkl_name": fn}
            st.session_state.selected_run = key
            st.success("Single run completed")

# ------------------- BATCH COMPARISON -------------------
if len(st.session_state.history) > 1:
    st.header("Batch Comparison - Modified Deposition Rate")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(st.session_state.history)))

    for i, (key, data) in enumerate(st.session_state.history.items()):
        if not data["thick"] or len(data["thick"]) == 0:
            continue
            
        thick_data = data["thick"]
        times = [scale_time(t[0]) for t in thick_data]
        ths_nm = [t[2] * 1e9 for t in thick_data]
        ax1.plot(times, ths_nm, label=key, color=colors[i], linewidth=2)

        diag = data["diag"]
        if diag and len(diag) > 0:
            tdiag = [scale_time(d[0]) for d in diag]
            bulk = [d[4] for d in diag]
            grad = [d[5] for d in diag]
            ax2.semilogy(tdiag, bulk, label=f'{key} bulk', color=colors[i])
            ax2.semilogy(tdiag, grad, ls='--', label=f'{key} grad', color=colors[i])

            if use_edl:
                edl = [d[6] for d in diag]
                ax3.plot(tdiag, edl, label=key, color=colors[i])

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Thickness (nm)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Shell Thickness Evolution")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("L2-norm")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Bulk and Gradient Norms")

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("EDL Boost")
    if use_edl:
        ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("EDL Catalyst Effect")

    st.pyplot(fig)

# ------------------- PLAYBACK -------------------
st.header("Simulation Playback & Analysis")

if st.session_state.history:
    selected_key = st.selectbox(
        "Choose run",
        list(st.session_state.history.keys()),
        index=0
    )
    data = st.session_state.history[selected_key]
    snaps, thick, diag, coords = data["snaps"], data["thick"], data["diag"], data["coords"]
    params = data["params"]

    # Display run parameters
    st.subheader("Run Parameters")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("c_bulk", f"{params[0]:.3f}")
    with col2:
        st.metric("Domain Mult", f"{params[1]:.2f}")
    with col3:
        st.metric("k0_nd", f"{params[2]:.3f}")
    with col4:
        st.metric("D_nd", f"{params[3]:.3f}")
    with col5:
        st.metric("gamma_nd", f"{params[4]:.3f}")

    # --- Safe playback slider block (replace existing playback guard and slider) ---
    if snaps and len(snaps) > 0:
        # enforce Python int and non-negative
        max_frame = int(len(snaps) - 1)

        if max_frame < 0:
            st.warning("No frames available for this run.")
        else:
            # default to first frame or clamp to boundary
            default_frame = 0

            frame = st.slider(
                "Frame",
                0,
                max_frame,
                default_frame,
                step=1,
                format="%d"
            )

            field = st.selectbox(
                "Field",
                ["phi (shell)", "c (concentration)", "psi (core)", "material map"]
            )

            if 0 <= frame < len(snaps):
                # unpack snapshot
                t, phi, c, psi = snaps[frame]
                t_real = scale_time(t)
                th_nm = thick[frame][2] * 1e9 if (thick and frame < len(thick)) else 0.0

                # choose field
                if field == "material map":
                    m = compute_material_map(phi, psi)
                    field_data = m
                    vmin, vmax = 0, 3
                else:
                    field_data = {
                        "phi (shell)": phi,
                        "c (concentration)": c,
                        "psi (core)": psi
                    }[field]
                    if field == "c (concentration)":
                        vmin, vmax = 0, params[0]  # c_bulk
                    else:
                        vmin, vmax = 0, 1

                # --- Plot ---
                fig, ax = plt.subplots(figsize=(8, 6))

                if mode == "2D (planar)":
                    im = ax.imshow(
                        field_data.T,
                        cmap=cmap_choice,
                        vmin=vmin,
                        vmax=vmax,
                        origin="lower",
                        extent=[0, coords[0][-1], 0, coords[1][-1]]
                    )
                    ax.set_xlabel("x (nd)")
                    ax.set_ylabel("y (nd)")

                else:
                    # 3D: take mid slice
                    Nz_mid = field_data.shape[2] // 2
                    im = ax.imshow(
                        field_data[:, :, Nz_mid].T,
                        cmap=cmap_choice,
                        vmin=vmin,
                        vmax=vmax,
                        origin="lower",
                        extent=[0, coords[0][-1], 0, coords[1][-1]]
                    )
                    ax.set_xlabel("x (nd)")
                    ax.set_ylabel("y (nd)")
                    ax.set_title(f"Slice at z = {coords[2][Nz_mid]:.2f}")

                plt.colorbar(im, ax=ax)
                ax.set_title(f"{field} @ t = {t_real:.3e} s | Thickness: {th_nm:.2f} nm")
                st.pyplot(fig)

            else:
                st.warning("Selected frame is out of range.")

    else:
        st.warning("No snapshots available for this run. The simulation may have failed or produced no output.")

    # Thickness evolution plot
    if thick and len(thick) > 0:
        fig_th, ax_th = plt.subplots(figsize=(8, 5))
        times_th = [scale_time(t[0]) for t in thick]
        ths_nm = [t[2] * 1e9 for t in thick]
        ax_th.plot(times_th, ths_nm, 'b-', linewidth=3, label='Shell thickness')
        ax_th.set_xlabel("Time (s)")
        ax_th.set_ylabel("Thickness (nm)")
        ax_th.grid(True, alpha=0.3)
        ax_th.legend()
        ax_th.set_title("Shell Thickness Evolution")
        st.pyplot(fig_th)

    # PKL Download
    fp = data.get("pkl_path")
    if fp and os.path.exists(fp):
        with open(fp, "rb") as f:
            st.download_button(
                "📦 Download this simulation as .pkl", 
                f.read(), 
                data["pkl_name"], 
                "application/octet-stream",
                use_container_width=True
            )

    # Additional metrics
    if 'growth_metrics' in data and data['growth_metrics'].get('onset_time_nd'):
        st.subheader("Growth Metrics")
        onset_time = data['growth_metrics']['onset_time_nd']
        if onset_time:
            st.metric("Onset Time", f"{scale_time(onset_time):.3e} s")
        
        if data['growth_metrics'].get('deposition_rates_nd_per_unit_time'):
            rates = data['growth_metrics']['deposition_rates_nd_per_unit_time']
            if rates:
                st.metric("Max Deposition Rate", f"{max(rates):.3e} nd/t")

else:
    st.info("Run a simulation (single or batch) to see results.")
