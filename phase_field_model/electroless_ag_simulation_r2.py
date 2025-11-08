# --------------------------------------------------------------
# NON-DIMENSIONAL ELECTROLESS Ag – BOUNDED & STABLE
# --------------------------------------------------------------
import streamlit as st
import numpy as np
from numba import njit, prange
import plotly.graph_objects as go
import pyvista as pv
import sqlite3, pickle, hashlib, zipfile, tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------- 1. SQLite --------------------
DB_PATH = Path("simulations_nondim.db")
def _hash_params(**kw):
    s = "".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, params BLOB, x BLOB, y BLOB,
                    phi_hist BLOB, c_hist BLOB, t_hist BLOB, psi BLOB
               )""")

def save_run(run_id, params, x, y, phi_hist, c_hist, t_hist, psi):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)""",
                    (run_id, pickle.dumps(params), pickle.dumps(x), pickle.dumps(y),
                     pickle.dumps(phi_hist), pickle.dumps(c_hist), pickle.dumps(t_hist), pickle.dumps(psi)))

def load_run(run_id):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row:
            keys = ["params","x","y","phi_hist","c_hist","t_hist","psi"]
            return {k: pickle.loads(v) for k, v in zip(keys, row[1:])}
    return None

# -------------------- 2. Numba kernels --------------------
@njit(parallel=True, fastmath=True)
def _laplacian(arr, h2):
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            out[i,j] = (arr[i+1,j] + arr[i-1,j] + arr[i,j+1] + arr[i,j-1] - 4*arr[i,j]) / h2
    return out

@njit(parallel=True, fastmath=True)
def _grad_x(arr, h):
    ny, nx = arr.shape; out = np.zeros_like(arr)
    for i in prange(ny):
        for j in prange(1, nx-1):
            out[i,j] = (arr[i,j+1] - arr[i,j-1]) / (2*h)
    return out

@njit(parallel=True, fastmath=True)
def _grad_y(arr, h):
    ny, nx = arr.shape; out = np.zeros_like(arr)
    for i in prange(1, ny-1):
        for j in prange(nx):
            out[i,j] = (arr[i+1,j] - arr[i-1,j]) / (2*h)
    return out

# -------------------- 3. NON-DIMENSIONAL FREE ENERGY --------------------
@njit(parallel=True, fastmath=True)
def free_energy_derivative_nondim(phi, psi, a_index, beta_tilde, h, W_tilde, kBT_tilde, eps):
    ny, nx = phi.shape
    f_prime = np.zeros_like(phi)
    phi_cl = np.where(phi < eps, eps, phi)
    phi_cl = np.where(phi_cl > 1.0-eps, 1.0-eps, phi_cl)

    # Polynomial: W_tilde * (2phi-1)*(1-2phi+2phi^2) = W_tilde * 2*(2phi-1)*(phi*(1-phi))
    z = 2.0*phi - 1.0
    poly = 2.0 * z * phi * (1.0 - phi)  # = 2*z*phi*(1-phi)
    poly_deriv = W_tilde * poly

    # Template harmonic
    harm = 2.0 * beta_tilde * (phi - h)

    # Logarithmic barrier: kBT_tilde * [ln(phi/(1-phi))]
    log_term = kBT_tilde * (np.log(phi_cl) - np.log(1.0 - phi_cl))

    for i in prange(ny):
        for j in prange(nx):
            f_prime[i,j] = ((1.0 + a_index)*(1.0 - psi[i,j])*poly_deriv[i,j] +
                            (1.0 - a_index)*psi[i,j]*harm[i,j] +
                            log_term[i,j])
    return f_prime

# -------------------- 4. Simulation (Non-dimensional) --------------------
def run_simulation_nondim(
    run_id, L,, Nx, Ny, eps_tilde, core_radius_frac, core_center,
    M_tilde, dt_tilde, t_max_tilde, D_tilde, c_bulk_tilde,
    alpha_tilde, i0_tilde, c_ref_tilde, beta_tilde, a_index, h,
    W_tilde, kBT_tilde, eps_log, ratio_top_factor, ratio_surface_factor, ratio_decay_tilde,
    save_every, ui
):
    h = L / (Nx - 1)  # grid spacing
    h2 = h * h
    x = np.linspace(0, L, Nx, dtype=np.float32)
    y = np.linspace(0, L, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- CORE TEMPLATE (≥30% area) ---
    r_core = core_radius_frac * L
    cx, cy = core_center[0] * L, core_center[1] * L
    psi = (((X - cx)**2 + (Y - cy)**2) <= r_core**2).astype(np.float32)
    area_core = np.sum(psi) * h * h
    area_domain = L * L
    st.write(f"Core area: {area_core/area_domain:.1%} of domain")

    # --- Initial phi ---
    phi = (0.5 * (1 - np.tanh(3 * (np.sqrt((X-cx)**2 + (Y-cy)**2) - r_core) / eps_tilde))).astype(np.float32)
    phi = np.clip(phi, 1e-6, 1-1e-6)

    # --- Ratio field ---
    base_ratio = 1.0
    vertical = Y / L
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2) - r_core
    surface = np.exp(-np.abs(dist) / ratio_decay_tilde)
    ratio_field = base_ratio * (
        (1.0 - ratio_surface_factor) * (0.2 + 0.8 * vertical) +
        ratio_surface_factor * surface
    )
    ratio_field = np.clip(ratio_field, 0.1, 8.0)

    # --- Concentration ---
    c = c_bulk_tilde * vertical * (1 - phi) * (1 - psi)

    # --- Storage ---
    phi_hist, c_hist, t_hist = [], [], []
    n_steps = int(np.ceil(t_max_tilde / dt_tilde))
    save_step = max(1, save_every)
    phi_old = phi.copy()

    progress = ui['progress']; status = ui['status']
    plot = ui['plot']; line = ui['line']; metrics = ui['metrics']

    for step in range(n_steps + 1):
        t = step * dt_tilde

        # Neumann BCs
        phi[:,0] = phi[:,1]; phi[:,-1] = phi[:,-2]
        phi[0,:] = phi[1,:]; phi[-1,:] = phi[-2,:]

        # Gradients
        phi_x = _grad_x(phi, h); phi_y = _grad_y(phi, h)
        grad_phi_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-30)
        delta_int = 6 * phi * (1 - phi) * (1 - psi) * grad_phi_mag
        delta_int = np.clip(delta_int, 0.0, 6.0 / eps_tilde)

        phi_xx = _laplacian(phi, h2)

        # Free energy derivative (non-dim)
        f_prime = free_energy_derivative_nondim(phi, psi, a_index, beta_tilde, h,
                                                W_tilde, kBT_tilde, eps_log)

        mu = -eps_tilde**2 * phi_xx + f_prime - alpha_tilde * c
        mu_xx = _laplacian(mu, h2)

        # Current & advection
        c_mol = c * (1 - phi) * (1 - psi) * ratio_field
        i_loc = i0_tilde * (c_mol / c_ref_tilde) * np.exp(0.5 * eta_chem_tilde)
        i_loc = i_loc * delta_int
        i_loc = np.clip(i_loc, -1e3, 1e3)

        u = -i_loc * MAg_rho_tilde
        advection = u * (1 - psi) * phi_y

        # Phase evolution
        dphi_dt = M_tilde * mu_xx - advection
        phi += dt_tilde * dphi_dt
        phi = np.clip(phi, 1e-6, 1-1e-6)  # soft safety only

        # Concentration
        c_eff = (1 - phi) * (1 - psi) * c
        c_xx = _laplacian(c_eff, h2)
        sink = -i_loc * delta_int
        c += dt_tilde * (D_tilde * c_xx + sink)
        c = np.clip(c, 0.0, c_bulk_tilde * 2)

        # BCs
        c[:,0] = 0.0
        c[:,-1] = c_bulk_tilde * (y / L)

        # Save & UI
        if step % save_step == 0 or step == n_steps:
            phi_hist.append(phi.copy()); c_hist.append(c.copy()); t_hist.append(t)
            try:
                fig = go.Figure(go.Contour(z=phi_hist[-1].T, x=x, y=y,
                                          contours_coloring='heatmap',
                                          colorbar=dict(title='ϕ')))
                fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=20))
                plot.plotly_chart(fig, use_container_width=True)

                mid = Nx//2
                fig2, ax = plt.subplots()
                ax.plot(y, phi_hist[-1][mid,:], label='ϕ')
                ax.plot(y, psi[mid,:], '--', label='ψ')
                ax.set_xlabel('y/L'); ax.set_title(f't* = {t:.3f}')
                ax.legend(); ax.grid(True)
                line.pyplot(fig2); plt.close(fig2)

                metrics.metric('t*', f"{t:.3f}")
                metrics.metric('ϕ range', f"[{phi.min():.3f}, {phi.max():.3f}]")
            except: pass

        if step > 100 and np.max(np.abs(phi - phi_old)) < 1e-6:
            status.info("Converged"); break
        if step % 50 == 0: phi_old = phi.copy()

        progress.progress(min(1.0, step/n_steps))
        if st.session_state.get('stop_sim', False): break

    progress.empty()
    return {
        'x': x, 'y': y, 'phi_hist': np.array(phi_hist), 'c_hist': np.array(c_hist),
        't_hist': np.array(t_hist), 'psi': psi, 'ratio_field': ratio_field
    }

# -------------------- 5. UI (Non-dimensional) --------------------
st.title("Non-Dimensional Electroless Ag – Stable & Bounded")
st.markdown("**Core ≥ 30% area, movable center, ϕ ∈ [0,1] guaranteed**")

st.sidebar.header("Domain")
L = st.sidebar.slider("L (char. length, cm)", 1e-6, 1e-5, 5e-6, 1e-7, format="%e")
Nx = st.sidebar.slider("Nx", 100, 300, 180, 10)
Ny = st.sidebar.slider("Ny", 100, 300, 180, 10)

st.sidebar.header("Core Template")
core_radius_frac = st.sidebar.slider("Core radius / L", 0.3, 0.7, 0.55, 0.01,
                                     help="≥30% area → πr²/L² ≥ 0.3 → r/L ≥ √(0.3/π) ≈ 0.31")
core_center_x = st.sidebar.slider("Core center x/L", 0.2, 0.8, 0.5, 0.01)
core_center_y = st.sidebar.slider("Core center y/L", 0.2, 0.8, 0.5, 0.01)

st.sidebar.header("Interface")
eps_tilde = st.sidebar.slider("ε* = ε/L", 0.01, 0.1, 0.03, 0.005)

st.sidebar.header("Kinetics")
M_tilde = st.sidebar.number_input("M* = M·t₀/L²", 1e-3, 1.0, 0.1, 0.01)
dt_tilde = st.sidebar.number_input("Δt*", 1e-4, 1e-2, 5e-4, 1e-5)
t_max_tilde = st.sidebar.number_input("t*_max", 1.0, 20.0, 10.0, 0.5)
D_tilde = st.sidebar.number_input("D* = D·t₀/L²", 0.01, 1.0, 0.1, 0.01)
c_bulk_tilde = st.sidebar.number_input("c*_bulk", 0.1, 10.0, 1.0, 0.1)

st.sidebar.header("Coupling")
alpha_tilde = st.sidebar.number_input("α*", 0.0, 10.0, 1.0, 0.1)
i0_tilde = st.sidebar.number_input("i₀*", 0.01, 1.0, 0.1, 0.01)
c_ref_tilde = st.sidebar.number_input("c*_ref", 0.1, 10.0, 1.0, 0.1)
eta_chem_tilde = st.sidebar.slider("η*_chem", 0.1, 1.0, 0.5, 0.05)

st.sidebar.header("Free Energy")
W_tilde = st.sidebar.number_input("W*", 1.0, 100.0, 30.0, 1.0)
kBT_tilde = st.sidebar.number_input("kT*", 0.1, 10.0, 1.0, 0.1)
beta_tilde = st.sidebar.slider("β*", 0.1, 10.0, 1.0, 0.1)
a_index = st.sidebar.slider("a-index", -1.0, 1.0, 0.0, 0.1)
h = st.sidebar.slider("h", 0.0, 1.0, 0.5, 0.1)
eps_log = st.sidebar.number_input("ϵ_log", 1e-6, 1e-2, 1e-4, 1e-5)

st.sidebar.header("Ratio Field")
ratio_top_factor = st.sidebar.slider("Top-weight", 0.0, 1.0, 0.7, 0.05)
ratio_surface_factor = st.sidebar.slider("Surface-boost", 0.0, 1.0, 0.5, 0.05)
ratio_decay_tilde = st.sidebar.slider("Decay λ*/L", 0.01, 0.2, 0.05, 0.01)

save_every = st.sidebar.number_input("Save every", 10, 100, 30, 5)

# Constants (non-dim)
MAg_rho_tilde = 1.0  # scaled
core_center = (core_center_x, core_center_y)

init_db()
if "run_id" not in st.session_state: st.session_state.run_id = None
if "results" not in st.session_state: st.session_state.results = None
if "stop_sim" not in st.session_state: st.session_state.stop_sim = False

col1, col2 = st.columns([3,1])
plot_area = col1.container(); line_area = col1.container(); metrics_area = col2.container()
status = st.empty(); progress = st.empty()

run_col, stop_col = st.columns(2)
if run_col.button("Run"):
    st.session_state.stop_sim = False
    run_id = _hash_params(**locals())
    cached = load_run(run_id)
    if cached:
        st.session_state.results = cached
        status.success("Loaded cached run")
    else:
        status.info("Starting simulation…")
        ui = {
            'progress': progress,
            'status': status,
            'plot': plot_area,
            'line': line_area,
            'metrics': metrics_area
        }
        results = run_simulation_nondim(
            run_id, L, Nx, Ny, eps_tilde, core_radius_frac, core_center,
            M_tilde, dt_tilde, t_max_tilde, D_tilde, c_bulk_tilde,
            alpha_tilde, i0_tilde, c_ref_tilde, beta_tilde, a_index, h,
            W_tilde, kBT_tilde, eps_log, ratio_top_factor, ratio_surface_factor, ratio_decay_tilde,
            save_every, ui
        )
        save_run(run_id, {}, results['x'], results['y'],
                 results['phi_hist'], results['c_hist'],
                 results['t_hist'], results['psi'])
        st.session_state.results = results
        status.success("Finished & saved!")

if stop_col.button("Stop"):
    st.session_state.stop_sim = True

# Results
if st.session_state.results:
    r = st.session_state.results
    x, y = r['x'], r['y']
    phi_hist, c_hist = r['phi_hist'], r['c_hist']
    t_hist = r['t_hist']; psi = r['psi']

    st.subheader("Results")
    time_idx = st.slider("Time t*", 0, len(t_hist)-1, len(t_hist)//2)
    var = st.selectbox("Field", ["ϕ", "c", "ψ"])
    data = phi_hist[time_idx] if var == "ϕ" else c_hist[time_idx] if var == "c" else psi

    fig = go.Figure(go.Contour(z=data.T, x=x, y=y, contours_coloring='heatmap',
                               colorbar=dict(title=var)))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    mid = Nx//2
    fig2, ax = plt.subplots()
    ax.plot(y/L, data[mid,:]); ax.set_xlabel('y/L'); ax.grid(True)
    st.pyplot(fig2); plt.close(fig2)
