#!/usr/bin/env python
# -*- coding: utf-8 -*-
## """
streamlit_electroless_final.py
Final integrated Streamlit app that:
 - supports single-run and batch-run (multiple concentrations)
 - two growth models (fully non-reversible, soft reversible)
 - default total time (n_steps) = 500 (adjustable)
 - batch final-thickness histogram
 - auto-save CSVs to ./results/
 - retains diagnostics, 2D/3D modes, and high-quality plotting
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, time, io
from datetime import datetime
# Optional libraries detection
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
# Page setup
st.set_page_config(page_title="Electroless Ag — Final Simulator", layout="wide")
st.title("Electroless Ag — Final Simulator")
# Plotting style
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 140,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
})
# Create results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ---------------- Sidebar: Controls ----------------
st.sidebar.header("Simulation controls")
# Growth model
growth_model = st.sidebar.selectbox(
    "Growth model",
    ["Model A — Fully non-reversible (strictly additive)",
     "Model B — Soft reversible (0.01× bulk smoothing)"]
)
# Dimension selection
mode = st.sidebar.selectbox("Mode", ["2D (planar)", "3D (spherical)"])
# Physics parameters (kept simple here but fully adjustable)
st.sidebar.subheader("Physics params (non-dim)")
gamma = st.sidebar.slider("γ (curvature)", 1e-4, 0.5, 0.02, 1e-4, format="%.4f")
beta = st.sidebar.slider("β (double-well)", 0.1, 20.0, 4.0, 0.1)
k0 = st.sidebar.slider("k₀ (reaction)", 0.01, 2.0, 0.4, 0.01)
M = st.sidebar.slider("M (mobility)", 1e-3, 1.0, 0.2, 1e-3, format="%.3f")
D = st.sidebar.slider("D (diffusion)", 0.0, 1.0, 0.05, 0.005)
# Additional physics parameter for better control
alpha = st.sidebar.slider("α (coupling)", 0.0, 10.0, 2.0, 0.1)
# Boundary conditions
bc_type = st.sidebar.selectbox(
    "Boundary condition type",
    ["Dirichlet (fixed values)", "Neumann (zero flux)"]
)
# Grid & time
st.sidebar.subheader("Grid & time")
if mode.startswith("2D"):
    Nx = st.sidebar.slider("Nx", 40, 400, 120, 10)
    Ny = st.sidebar.slider("Ny", 40, 400, 120, 10)
    Nz = 1
else:
    Nx = st.sidebar.slider("Nx", 16, 80, 40, 4)
    Ny = Nx
    Nz = st.sidebar.slider("Nz", 16, 80, 40, 4)
# Default total time (number of steps) = 500 as requested
n_steps = st.sidebar.number_input("n_steps (total time steps)", min_value=10, max_value=20000, value=500, step=10)
dt = st.sidebar.number_input("dt (non-dim)", 1e-6, 2e-2, 2e-4, format="%.6f")
save_every = st.sidebar.slider("save every (frames)", 1, 400, max(1, n_steps//20), 1)
# Geometry & detection
st.sidebar.subheader("Geometry & detection")
core_radius_frac = st.sidebar.slider("Core radius (fraction of L)", 0.05, 0.45, 0.18, 0.01)
shell_thickness_frac = st.sidebar.slider("Initial shell thickness (Δr/r_core)", 0.01, 0.6, 0.2, 0.01)
phi_threshold = st.sidebar.slider("phi threshold for shell detection", 0.1, 0.9, 0.5, 0.05)
# Concentration inputs and modes
st.sidebar.subheader("Concentration input")
conc_mode = st.sidebar.selectbox("Concentration input mode", ["Manual (comma list)", "Slider range"])
if conc_mode.startswith("Manual"):
    manual_text = st.sidebar.text_area("Enter concentrations (comma-separated)", value="1.0, 0.5, 0.333333, 0.25, 0.2")
else:
    slider_start = st.sidebar.number_input("Start c_bulk", 0.05, 10.0, 1.0, 0.05)
    slider_end = st.sidebar.number_input("End c_bulk", 0.05, 10.0, 0.2, 0.05)
    slider_steps = st.sidebar.slider("Steps", 2, 20, 5)
# Enhanced Single concentration control with more options
st.sidebar.subheader("Single concentration")
single_c = st.sidebar.number_input("Single c_bulk", 0.01, 20.0, 1.0, 0.01, 
                                  help="Concentration ratio [Ag]/[Cu] for single simulation")
# Performance options
st.sidebar.subheader("Performance")
use_numba_single = st.sidebar.checkbox("Use Numba acceleration (if available)", value=False)
use_semi_implicit_single = st.sidebar.checkbox("Use semi-implicit solver (if scipy available)", value=False)

# Buttons
run_single = st.sidebar.button("Run Single Concentration")
run_batch = st.sidebar.button("Run Batch Concentrations")
# ---------------- Helpers ----------------
def nd_to_real(length_nd, L0_nm=20.0e-9):
    return length_nd * L0_nm

def parse_concentrations():
    if conc_mode.startswith("Manual"):
        s = manual_text.strip()
        if s == "":
            return [1.0]
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except Exception:
                pass
        return vals if len(vals) > 0 else [1.0]
    else:
        if slider_steps <= 1:
            return [slider_start]
        if abs(slider_end - slider_start) < 1e-12:
            return [slider_start]
        return list(np.linspace(slider_start, slider_end, slider_steps))

# Enhanced concentration validation
def validate_concentration(c_val):
    """Validate concentration value and return with appropriate precision"""
    if c_val <= 0:
        st.warning(f"Concentration {c_val} is not positive. Using 0.01 instead.")
        return 0.01
    return float(round(c_val, 6))  # Reasonable precision

# ---------------- Numerical utilities ----------------
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

# ---------------- Core functions (2D/3D simulation) ----------------
def compute_shell_thickness(phi, psi, coords, core_radius_frac_local, threshold=0.5, mode="2D"):
    if mode.startswith("2D"):
        x, y = coords
        cx = cy = 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        distances = np.sqrt((X-cx)**2 + (Y-cy)**2)
    else:
        x, y, z = coords
        cx = cy = cz = 0.5
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        distances = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)
    shell_mask = (phi > threshold) & (psi < 0.5)
    if np.any(shell_mask):
        max_shell_radius = np.max(distances[shell_mask])
        core_radius = core_radius_frac_local
        thickness_nd = max(0.0, max_shell_radius - core_radius)
        thickness_m = nd_to_real(thickness_nd)
        return thickness_nd, thickness_m
    return 0.0, 0.0

def run_simulation_2d_local(params):
    Nx, Ny = params['Nx'], params['Ny']
    L = 1.0
    dx = L/(Nx-1)
    x = np.linspace(0, L, Nx); y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
    psi = (dist <= params['core_radius_frac']*L).astype(np.float64)
    r_core = params['core_radius_frac']*L
    r_outer = r_core*(1.0 + params['shell_thickness_frac'])
    phi = np.where(dist <= r_core, 0.0,
                   np.where(dist <= r_outer, 1.0, 0.0)).astype(np.float64)
    eps = max(4*dx, 1e-6)
    phi = phi * (1.0 - 0.5*(1.0 - np.tanh((dist-r_core)/eps))) * (1.0 - 0.5*(1.0 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0.0, 1.0)
    if params['bc_type'].startswith("Dirichlet"):
        c = params['c_bulk'] * np.ones_like(X) * (1.0 - phi) * (1.0 - psi)
    else:
        c = params['c_bulk'] * (Y/L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])
    snapshots, diagnostics, thickness_data = [], [], []
    max_thickness_nd_so_far = 0.0
    softness = 0.0 if params['growth_model'].startswith("Model A") else 0.01
    for step in range(params['n_steps'] + 1):
        t = step * params['dt']
        gphi = grad_mag_2d(phi, dx)
        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps,dx))
        # phi BCs
        if params['bc_type'].startswith("Dirichlet"):
            phi[0,:] = phi[-1,:] = phi[:,0] = phi[:,-1] = 0.0
        else:
            phi[0,:] = phi[1,:]; phi[-1,:] = phi[-2,:]
            phi[:,0] = phi[:,1]; phi[:,-1] = phi[:,-2]
        lap_phi = laplacian_explicit_2d(phi, dx)
        f_bulk = 2.0*params['beta']*phi*(1.0-phi)*(1.0-2.0*phi)
        c_mol = c*(1.0-phi)*(1.0-psi)
        i_loc = params['k0']*c_mol*delta_int
        i_loc = np.clip(i_loc, 0.0, 1e6)
        deposition = params['M'] * i_loc
        curvature = params['M']*params['gamma']*lap_phi
        curvature_pos = np.maximum(curvature, 0.0)
        delta_phi = params['dt'] * (deposition + curvature_pos - softness * params['M'] * f_bulk)
        if softness == 0.0:
            delta_phi = np.maximum(delta_phi, 0.0)
        phi = phi + delta_phi
        phi = np.clip(phi, 0.0, 1.0)
        # concentration update
        lap_c = laplacian_explicit_2d(c, dx)
        sink = i_loc
        c += params['dt'] * (params['D']*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk'])
        if step % params['save_every'] == 0 or step == params['n_steps']:
            thickness_nd, thickness_m = compute_shell_thickness(phi, psi, (x, y), params['core_radius_frac'], params['phi_threshold'], "2D")
            max_thickness_nd_so_far = max(max_thickness_nd_so_far, thickness_nd)
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, np.sqrt(np.mean(f_bulk**2)), np.sqrt(np.mean((params['M']*params['gamma']*lap_phi)**2)), 0.0, params['alpha']*np.mean(c)))
            thickness_data.append((t, max_thickness_nd_so_far, nd_to_real(max_thickness_nd_so_far)))
    return snapshots, diagnostics, thickness_data, (x, y)

def run_simulation_3d_local(params):
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
    L = 1.0; dx = L/(Nx-1)
    x = np.linspace(0, L, Nx); y = np.linspace(0, L, Ny); z = np.linspace(0, L, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    dist = np.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
    psi = (dist <= params['core_radius_frac']*L).astype(np.float64)
    r_core = params['core_radius_frac']*L
    r_outer = r_core*(1.0 + params['shell_thickness_frac'])
    phi = np.where(dist <= r_core, 0.0,
                   np.where(dist <= r_outer, 1.0, 0.0)).astype(np.float64)
    eps = max(4*dx, 1e-6)
    phi = phi * (1.0 - 0.5*(1.0 - np.tanh((dist-r_core)/eps))) * (1.0 - 0.5*(1.0 + np.tanh((dist-r_outer)/eps)))
    phi = np.clip(phi, 0.0, 1.0)
    if params['bc_type'].startswith("Dirichlet"):
        c = params['c_bulk'] * np.ones_like(X) * (1.0 - phi) * (1.0 - psi)
    else:
        c = params['c_bulk'] * (Z/L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, params['c_bulk'])
    snapshots, diagnostics, thickness_data = [], [], []
    max_thickness_nd_so_far = 0.0
    softness = 0.0 if params['growth_model'].startswith("Model A") else 0.01
    for step in range(params['n_steps'] + 1):
        t = step * params['dt']
        lap_phi = laplacian_explicit_3d(phi, L/(Nx-1))
        gx, gy, gz = np.gradient(phi, L/(Nx-1), edge_order=2)
        gphi = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-30)
        delta_int = 6.0*phi*(1.0-phi)*(1.0-psi)*gphi
        delta_int = np.clip(delta_int, 0.0, 6.0/max(eps, L/(Nx-1)))
        if params['bc_type'].startswith("Dirichlet"):
            phi[[0,-1],:,:] = phi[:,[0,-1],:] = phi[:,:,[0,-1]] = 0.0
        else:
            phi[0,:,:] = phi[1,:,:]; phi[-1,:,:] = phi[-2,:,:]
            phi[:,0,:] = phi[:,1,:]; phi[:,-1,:] = phi[:,-2,:]
            phi[:,:,0] = phi[:,:,1]; phi[:,:,-1] = phi[:,:,-2]
        f_bulk = 2.0*params['beta']*phi*(1.0-phi)*(1.0-2.0*phi)
        c_mol = c*(1.0-phi)*(1.0-psi)
        i_loc = params['k0']*c_mol*delta_int
        i_loc = np.clip(i_loc, 0.0, 1e6)
        deposition = params['M']*i_loc
        curvature = params['M']*params['gamma']*lap_phi
        curvature_pos = np.maximum(curvature, 0.0)
        delta_phi = params['dt']*(deposition + curvature_pos - softness * params['M'] * f_bulk)
        if softness == 0.0:
            delta_phi = np.maximum(delta_phi, 0.0)
        phi = phi + delta_phi
        phi = np.clip(phi, 0.0, 1.0)
        lap_c = laplacian_explicit_3d(c, L/(Nx-1))
        sink = i_loc
        c += params['dt']*(params['D']*lap_c - sink)
        c = np.clip(c, 0.0, params['c_bulk'])
        if step % params['save_every'] == 0 or step == params['n_steps']:
            thickness_nd, thickness_m = compute_shell_thickness(phi, psi, (x, y, z), params['core_radius_frac'], params['phi_threshold'], "3D")
            max_thickness_nd_so_far = max(max_thickness_nd_so_far, thickness_nd)
            snapshots.append((t, phi.copy(), c.copy(), psi.copy()))
            diagnostics.append((t, np.sqrt(np.mean(f_bulk**2)), np.sqrt(np.mean((params['M']*params['gamma']*lap_phi)**2)), 0.0, params['alpha']*np.mean(c)))
            thickness_data.append((t, max_thickness_nd_so_far, nd_to_real(max_thickness_nd_so_far)))
    return snapshots, diagnostics, thickness_data, (x, y, z)

# ---------------- Run wrappers ----------------
def run_single_concentration(c_val):
    # Validate concentration
    c_val = validate_concentration(c_val)
    
    params = {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'dt': dt, 'n_steps': int(n_steps), 'save_every': save_every,
        'gamma': gamma, 'beta': beta, 'k0': k0, 'M': M, 'D': D,
        'alpha': alpha, 'c_bulk': float(c_val),
        'core_radius_frac': core_radius_frac, 'shell_thickness_frac': shell_thickness_frac,
        'use_semi_implicit': use_semi_implicit_single and SCIPY_AVAILABLE, 
        'use_numba': use_numba_single and NUMBA_AVAILABLE,
        'bc_type': bc_type,
        'phi_threshold': phi_threshold,
        'growth_model': growth_model,
        'mode': mode
    }
    if mode.startswith("2D"):
        return run_simulation_2d_local(params)
    else:
        return run_simulation_3d_local(params)

def run_batch_concentrations(concentrations):
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, c_val in enumerate(concentrations):
        c_val = validate_concentration(c_val)
        status_text.text(f"Running concentration {i+1}/{len(concentrations)}: c = {c_val:.6f}")
        snaps, diags, thick, coords = run_single_concentration(c_val)
        results[c_val] = {'snapshots': snaps, 'diagnostics': diags, 'thickness': thick, 'coords': coords}
        
        # Save CSV for this run (thickness vs time)
        csv_buf = io.StringIO()
        df = pd.DataFrame([(t, nd, real) for (t, nd, real) in thick], columns=["t_nd", "thickness_nd", "thickness_m"])
        df["thickness_nm"] = df["thickness_m"] * 1e9
        df.to_csv(csv_buf, index=False)
        
        # Create filename with timestamp and parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = growth_model.split('—')[0].strip().replace(' ', '_')
        filename = os.path.join(RESULTS_DIR, f"thickness_c_{c_val:.6f}_{mode.replace(' ','_')}_{safe_model_name}_{timestamp}.csv")
        
        with open(filename, "w", newline='') as f:
            f.write(csv_buf.getvalue())
            
        progress_bar.progress((i + 1) / len(concentrations))
    
    status_text.text("Batch simulation completed!")
    return results

# ---------------- UI: Single-run ----------------
st.header("Single Concentration Run")
st.write("Run a single concentration and inspect the thickness curve and fields.")

# Enhanced single run section with more details
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Simulation Control")
    sc_val = single_c
    st.write(f"**Current concentration:** {sc_val:.6f}")
    st.write(f"**Growth model:** {growth_model}")
    st.write(f"**Mode:** {mode}")
    st.write(f"**Grid size:** {Nx}×{Ny}" + (f"×{Nz}" if mode.startswith("3D") else ""))
    
    if st.button("Run single concentration now", key="single_run_main"):
        with st.spinner(f"Running simulation for c = {sc_val:.6f}..."):
            t0 = time.time()
            snaps, diags, thick, coords = run_single_concentration(sc_val)
            st.session_state['single_result'] = {
                'snapshots': snaps, 
                'diagnostics': diags, 
                'thickness': thick, 
                'coords': coords, 
                'c': sc_val,
                'run_time': time.time() - t0
            }
            
            # Save CSV with enhanced metadata
            df = pd.DataFrame([(t, nd, real) for (t, nd, real) in thick], columns=["t_nd", "thickness_nd", "thickness_m"])
            df["thickness_nm"] = df["thickness_m"] * 1e9
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = growth_model.split('—')[0].strip().replace(' ', '_')
            fname = os.path.join(RESULTS_DIR, f"single_thickness_c_{sc_val:.6f}_{mode.replace(' ','_')}_{safe_model_name}_{timestamp}.csv")
            df.to_csv(fname, index=False)
            
            st.success(f"Single run finished in {time.time()-t0:.2f}s — CSV saved to {fname}")

with col2:
    if 'single_result' in st.session_state and st.session_state['single_result'] is not None:
        res = st.session_state['single_result']
        thick = res['thickness']
        
        if thick:
            times = [t for (t,*,*) in thick]
            thick_nm = [m*1e9 for (*,*,m) in thick]
            
            # Create enhanced plot
            fig, ax = plt.subplots(figsize=(8,4.5))
            ax.plot(np.array(times), thick_nm, '-o', markersize=4, linewidth=2)
            ax.set_xlabel("Time (non-dim)")
            ax.set_ylabel("Thickness (nm)")
            ax.set_title(f"Single-run Thickness vs Time (c={res['c']:.4f}, {mode}, {growth_model.split('—')[0]})")
            ax.grid(True, alpha=0.3)
            
            # Add final thickness annotation
            final_thickness = thick_nm[-1] if thick_nm else 0
            ax.annotate(f'Final: {final_thickness:.2f} nm', 
                       xy=(times[-1], final_thickness), 
                       xytext=(times[-1]*0.7, final_thickness*1.1),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=12, color='red')
            
            plt.tight_layout(pad=2.0)
            st.pyplot(fig)
            
            # Display simulation summary
            st.subheader("Simulation Summary")
            st.write(f"**Concentration:** {res['c']:.6f}")
            st.write(f"**Final thickness:** {final_thickness:.2f} nm")
            st.write(f"**Run time:** {res.get('run_time', 'N/A'):.2f}s")
            st.write(f"**Number of frames:** {len(thick)}")
            
            # Download CSV
            csv_files = sorted([f for f in os.listdir(RESULTS_DIR) 
                              if f.startswith(f"single_thickness_c_{res['c']:.6f}") and f.endswith(".csv")])
            if csv_files:
                latest_file = csv_files[-1]  # Get most recent
                file_path = os.path.join(RESULTS_DIR, latest_file)
                with open(file_path, "rb") as f:
                    st.download_button(
                        "Download single-run CSV", 
                        f, 
                        file_name=latest_file, 
                        mime="text/csv",
                        key="single_download"
                    )

# ---------------- UI: Batch-run ----------------
st.header("Batch Concentration Run")
st.write("Define multiple concentrations (manual list or slider range) and run batch. Final-thickness histogram will be shown.")

# Enhanced batch interface
batch_col1, batch_col2 = st.columns([1,1])
with batch_col1:
    concs = parse_concentrations()
    st.write(f"**Parsed concentrations ({len(concs)}):** {[f'{c:.6f}' for c in concs]}")
    
    if len(concs) > 10:
        st.warning(f"Large number of concentrations ({len(concs)}). This may take a while to complete.")

with batch_col2:
    if st.button("Run batch now", key="batch_run_main"):
        if not concs:
            st.error("No valid concentrations provided.")
        else:
            with st.spinner(f"Running batch simulation for {len(concs)} concentrations..."):
                t0 = time.time()
                multiple_results = run_batch_concentrations(concs)
                st.session_state['batch_results'] = multiple_results
                st.session_state['batch_run_time'] = time.time() - t0
                st.success(f"Batch finished in {time.time()-t0:.2f}s — {len(concs)} CSV files saved to '{RESULTS_DIR}/'")

# Show batch results if available
if 'batch_results' in st.session_state and st.session_state['batch_results']:
    batch_res = st.session_state['batch_results']
    
    # Display batch summary
    st.subheader("Batch Simulation Summary")
    st.write(f"**Total concentrations:** {len(batch_res)}")
    st.write(f"**Total run time:** {st.session_state.get('batch_run_time', 'N/A'):.2f}s")
    
    # plot thickness vs time for all concentrations
    fig, ax = plt.subplots(figsize=(10,6))
    cmap = plt.get_cmap("viridis")
    keys = sorted(batch_res.keys(), reverse=True)
    colors = cmap(np.linspace(0,1,len(keys)))
    final_thicknesses_nm = []
    concentrations_list = []
    
    for i, k in enumerate(keys):
        thick = batch_res[k]['thickness']
        if not thick:
            continue
        times = [t for (t,*,*) in thick]
        th_nm = [m*1e9 for (*,*,m) in thick]
        ax.plot(times, th_nm, marker='o', markersize=4, label=f"c={k:.4f}", color=colors[i], linewidth=2)
        final_thicknesses_nm.append(th_nm[-1])
        concentrations_list.append(k)
    
    ax.set_xlabel("Time (non-dim)")
    ax.set_ylabel("Thickness (nm)")
    ax.set_title(f"Batch Thickness vs Time — {mode} — {growth_model}")
    ax.grid(True, alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', title="Concentrations")
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)
    
    # Histogram of final thicknesses
    if final_thicknesses_nm:
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12,4))
        
        # Histogram
        ax2.hist(final_thicknesses_nm, bins=min(12, max(3, len(final_thicknesses_nm))), alpha=0.9, color='skyblue', edgecolor='black')
        ax2.set_xlabel("Final thickness (nm)")
        ax2.set_ylabel("Count")
        ax2.set_title("Histogram of final thickness across concentrations")
        ax2.grid(True, alpha=0.2)
        
        # Scatter plot: concentration vs final thickness
        ax3.scatter(concentrations_list, final_thicknesses_nm, s=60, alpha=0.7, color='coral', edgecolor='black')
        ax3.set_xlabel("Concentration c_bulk")
        ax3.set_ylabel("Final thickness (nm)")
        ax3.set_title("Concentration vs Final Thickness")
        ax3.grid(True, alpha=0.2)
        
        plt.tight_layout(pad=2.0)
        st.pyplot(fig2)
        
        # Final thickness table
        st.subheader("Final Thickness Summary")
        summary_data = []
        for conc, thick_nm in zip(concentrations_list, final_thicknesses_nm):
            summary_data.append({
                "Concentration": conc,
                "Final Thickness (nm)": f"{thick_nm:.2f}",
                "Final Thickness (non-dim)": f"{(thick_nm * 1e-9) / 20e-9:.4f}"  # Convert back to non-dim
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
    
    # Provide zip or single CSV download links for saved CSVs
    st.subheader("Download Results")
    csv_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv") and "single_thickness" not in f])
    
    if csv_files:
        st.write(f"Found {len(csv_files)} batch result files:")
        
        # Create a zip file with all batch results
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for file in csv_files:
                file_path = os.path.join(RESULTS_DIR, file)
                zip_file.write(file_path, file)
        
        st.download_button(
            "Download All Batch Results (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            key="batch_zip"
        )
        
        # Individual file downloads
        st.write("Or download individual files:")
        for file in csv_files[-5:]:  # Show last 5 files
            file_path = os.path.join(RESULTS_DIR, file)
            with open(file_path, "rb") as f:
                st.download_button(
                    f"Download {file}",
                    f,
                    file_name=file,
                    mime="text/csv",
                    key=f"batch_{file}"
                )

# ---------------- Diagnostics / Info ----------------
st.sidebar.header("Info")
st.sidebar.write("CSV files are saved to the local results/ folder.")
st.sidebar.write("Tip: reduce grid (Nx/Ny/Nz) for faster runs; keep 3D grids small.")

# Add system information
st.sidebar.header("System Info")
st.sidebar.write(f"Numba available: {NUMBA_AVAILABLE}")
st.sidebar.write(f"SciPy available: {SCIPY_AVAILABLE}")
st.sidebar.write(f"Results directory: {RESULTS_DIR}")

# Add quick concentration presets
st.sidebar.header("Quick Presets")
if st.sidebar.button("Load Common Concentrations"):
    manual_text = "1.0, 0.8, 0.6, 0.5, 0.4, 0.333333, 0.3, 0.25, 0.2, 0.15, 0.1"
    st.sidebar.text_area("Enter concentrations (comma-separated)", value=manual_text, key="preset_update")
    st.experimental_rerun()
