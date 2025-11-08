# --------------------------------------------------------------
# ELECTRODEPOSITION Ag – CORRECT PHYSICS, STABLE SHELL GROWTH
# --------------------------------------------------------------
import streamlit as st
import numpy as np
from numba import njit, prange
import plotly.graph_objects as go
import pyvista as pv
import sqlite3, pickle, hashlib
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------- 1. SQLite --------------------
DB_PATH = Path("simulations_electrodeposition.db")
def _hash_params(**kw):
    s = "".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, params BLOB, x BLOB, y BLOB,
                    phi_hist BLOB, c_hist BLOB, phi_l_hist BLOB, t_hist BLOB, psi BLOB
               )""")

def save_run(run_id, params, x, y, phi_hist, c_hist, phi_l_hist, t_hist, psi):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)""",
                    (run_id, pickle.dumps(params), pickle.dumps(x), pickle.dumps(y),
                     pickle.dumps(phi_hist), pickle.dumps(c_hist), pickle.dumps(phi_l_hist),
                     pickle.dumps(t_hist), pickle.dumps(psi)))

def load_run(run_id):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row:
            keys = ["params","x","y","phi_hist","c_hist","phi_l_hist","t_hist","psi"]
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

# -------------------- 3. DOUBLE-WELL FREE ENERGY --------------------
@njit(parallel=True, fastmath=True)
def f_prime_double_well(phi, psi, a_index, beta_tilde, h):
    ny, nx = phi.shape
    f_prime = np.zeros_like(phi)
    dw = 2.0 * beta_tilde * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
    harm = 2.0 * beta_tilde * (phi - h)
    for i in prange(ny):
        for j in prange(nx):
            f_prime[i,j] = ((1.0 + a_index) * (1.0 - psi[i,j]) * dw[i,j] / 8.0 +
                            (1.0 - a_index) * psi[i,j] * harm[i,j] / 8.0)
    return f_prime

# -------------------- 4. ELECTRODEPOSITION SIMULATION --------------------
def run_simulation(
    run_id, L, Nx, Ny, eps_tilde, core_radius_frac, core_center,
    M_tilde, dt_tilde, t_max_tilde, D_tilde, c_bulk_tilde,
    Phi_anode_tilde, alpha_tilde, i0_tilde, c_ref_tilde,
    beta_tilde, a_index, h, save_every, ui
):
    h = L / (Nx - 1); h2 = h * h
    x = np.linspace(0, L, Nx, dtype=np.float32)
    y = np.linspace(0, L, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Core template ---
    r_core = core_radius_frac * L
    cx, cy = core_center[0] * L, core_center[1] * L
    psi = (((X - cx)**2 + (Y - cy)**2) <= r_core**2).astype(np.float32)
    core_area = np.sum(psi) * h * h / (L * L)
    st.write(f"**Core: {core_area:.1%} of domain**")

    # --- Initial fields ---
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2) - r_core
    phi = 0.5 * (1.0 - np.tanh(3.0 * dist / eps_tilde))
    phi = np.clip(phi, 1e-6, 1.0 - 1e-6)

    c = c_bulk_tilde * (Y / L) * (1.0 - phi) * (1.0 - psi)
    phi_l = (Y / L) * Phi_anode_tilde  # Linear potential

    # --- Constants ---
    F_tilde = 1.0; R_tilde = 1.0; T_tilde = 1.0  # non-dim
    z = 1
    MAg_rho_tilde = 1.0

    # --- Storage ---
    phi_hist, c_hist, phi_l_hist, t_hist = [], [], [], []
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
        grad_phi = np.sqrt(phi_x**2 + phi_y**2 + 1e-30)
        delta_int = 6.0 * phi * (1.0 - phi) * (1.0 - psi) * grad_phi
        delta_int = np.clip(delta_int, 0.0, 6.0 / eps_tilde)

        phi_xx = _laplacian(phi, h2)

        # Free energy
        f_prime = f_prime_double_well(phi, psi, a_index, beta_tilde, h)
        mu = -eps_tilde**2 * phi_xx + f_prime - alpha_tilde * c
        mu_xx = _laplacian(mu, h2)

        # --- Butler-Volmer current ---
        eta = -phi_l
        c_mol = c * (1.0 - phi) * (1.0 - psi)
        exp_c = np.exp(1.5 * F_tilde * eta / (R_tilde * T_tilde))
        exp_a = np.exp(-0.5 * F_tilde * eta / (R_tilde * T_tilde))
        i_loc = i0_tilde * (exp_c * c_mol / c_ref_tilde - exp_a)
        i_loc = i_loc * delta_int
        i_loc = np.clip(i_loc, -1e3, 1e3)

        # --- Advection velocity (volume addition) ---
        u = -i_loc / (z * F_tilde) * MAg_rho_tilde
        advection = u * (1.0 - psi) * phi_y

        # --- Phase evolution ---
        dphi_dt = M_tilde * mu_xx - advection
        phi += dt_tilde * dphi_dt
        phi = np.clip(phi, 0.0, 1.0)

        # --- Concentration ---
        c_eff = (1.0 - phi) * (1.0 - psi) * c
        c_xx = _laplacian(c_eff, h2)
        c_y = _grad_y(c_eff, h)

        # Electromigration: (z F D / RT) * E * c_y
        E_field = Phi_anode_tilde / L
        migration = (z * F_tilde * D_tilde / (R_tilde * T_tilde)) * E_field * c_y

        sink = -i_loc * delta_int / (z * F_tilde)
        c += dt_tilde * (D_tilde * c_xx + migration + sink)
        c = np.clip(c, 0.0, c_bulk_tilde * 2)

        # BCs
        c[:,0] = 0.0
        c[:,-1] = c_bulk_tilde * (y / L)

        # --- Store ---
        if step % save_step == 0 or step == n_steps:
            phi_hist.append(phi.copy())
            c_hist.append(c.copy())
            phi_l_hist.append(phi_l.copy())
            t_hist.append(t)

            try:
                fig = go.Figure(go.Contour(z=phi_hist[-1].T, x=x/L, y=y/L,
                                          contours_coloring='heatmap',
                                          colorbar=dict(title='ϕ')))
                fig.update_layout(height=360)
                plot.plotly_chart(fig, use_container_width=True)

                mid = Nx // 2
                fig2, ax = plt.subplots()
                ax.plot(y/L, phi_hist[-1][mid,:], label='ϕ')
                ax.plot(y/L, psi[mid,:], '--', label='ψ')
                ax.set_xlabel('y/L'); ax.set_title(f't* = {t:.3f}')
                ax.legend(); ax.grid(True)
                line.pyplot(fig2); plt.close(fig2)

                metrics.metric('t*', f"{t:.3f}")
                metrics.metric('ϕ range', f"[{phi.min():.4f}, {phi.max():.4f}]")
            except: pass

        if step > 100 and np.max(np.abs(phi - phi_old)) < 1e-6:
            status.info("Converged")
            break
        if step % 50 == 0: phi_old = phi.copy()

        progress.progress(min(1.0, step / n_steps))
        if st.session_state.get('stop_sim', False): break

    progress.empty()
    return {
        'x': x, 'y': y,
        'phi_hist': np.array(phi_hist), 'c_hist': np.array(c_hist),
        'phi_l_hist': np.array(phi_l_hist), 't_hist': np.array(t_hist),
        'psi': psi
    }

# -------------------- 5. UI --------------------
st.title("Electrodeposition Ag – Correct Physics")
st.markdown("**Higher driving force far from template → stable outward shell**")

st.sidebar.header("Domain")
L = st.sidebar.slider("L (cm)", 1e-6, 1e-5, 5e-6, 1e-7, format="%e")
Nx = st.sidebar.slider("Nx", 100, 300, 180, 10)
Ny = st.sidebar.slider("Ny", 100, 300, 180, 10)

st.sidebar.header("Core Template")
core_radius_frac = st.sidebar.slider("Core r/L", 0.31, 0.7, 0.55, 0.01)
core_center_x = st.sidebar.slider("Core x/L", 0.2, 0.8, 0.5, 0.01)
core_center_y = st.sidebar.slider("Core y/L", 0.2, 0.8, 0.5, 0.01)

st.sidebar.header("Interface")
eps_tilde = st.sidebar.slider("ε*", 0.01, 0.1, 0.03, 0.005)

st.sidebar.header("Kinetics")
M_tilde = st.sidebar.number_input("M*", 1e-3, 1.0, 0.1, 0.01)
dt_tilde = st.sidebar.number_input("Δt*", 1e-4, 1e-2, 5e-4, 1e-5)
t_max_tilde = st.sidebar.number_input("t*_max", 1.0, 30.0, 15.0, 0.5)
D_tilde = st.sidebar.number_input("D*", 0.01, 1.0, 0.1, 0.01)
c_bulk_tilde = st.sidebar.number_input("c*_bulk", 0.1, 10.0, 1.0, 0.1)

st.sidebar.header("Electrochemistry")
Phi_anode_tilde = st.sidebar.slider("Φ*_anode", 0.1, 5.0, 2.0, 0.1)
i0_tilde = st.sidebar.number_input("i₀*", 0.01, 1.0, 0.1, 0.01)
c_ref_tilde = st.sidebar.number_input("c*_ref", 0.1, 10.0, 1.0, 0.1)

st.sidebar.header("Coupling")
alpha_tilde = st.sidebar.number_input("α*", 0.0, 10.0, 1.0, 0.1)
beta_tilde = st.sidebar.slider("β*", 0.1, 10.0, 1.0, 0.1)
a_index = st.sidebar.slider("a-index", -1.0, 1.0, 0.0, 0.1)
h = st.sidebar.slider("h", 0.0, 1.0, 0.5, 0.1)

save_every = st.sidebar.number_input("Save every", 10, 100, 30, 5)

init_db()
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
        st.session_state.results = cached; status.success("Loaded")
    else:
        status.info("Running...")
        ui = {'progress': progress, 'status': status, 'plot': plot_area, 'line': line_area, 'metrics': metrics_area}
        results = run_simulation(
            run_id, L, Nx, Ny, eps_tilde, core_radius_frac, (core_center_x, core_center_y),
            M_tilde, dt_tilde, t_max_tilde, D_tilde, c_bulk_tilde,
            Phi_anode_tilde, alpha_tilde, i0_tilde, c_ref_tilde,
            beta_tilde, a_index, h, save_every, ui
        )
        save_run(run_id, {}, results['x'], results['y'], results['phi_hist'], results['c_hist'],
                 results['phi_l_hist'], results['t_hist'], results['psi'])
        st.session_state.results = results
        status.success("Done!")

if stop_col.button("Stop"): st.session_state.stop_sim = True

# -------------------- Results --------------------
if st.session_state.results:
    r = st.session_state.results
    x, y = r['x'], r['y']
    phi_hist, c_hist, phi_l_hist = r['phi_hist'], r['c_hist'], r['phi_l_hist']
    t_hist = r['t_hist']; psi = r['psi']

    st.subheader("Results")
    time_idx = st.slider("Time t*", 0, len(t_hist)-1, len(t_hist)//2)
    var = st.selectbox("Field", ["ϕ", "c", "ϕ_l", "ψ"])
    data = phi_hist[time_idx] if var == "ϕ" else c_hist[time_idx] if var == "c" else phi_l_hist[time_idx] if var == "ϕ_l" else psi

    fig = go.Figure(go.Contour(z=data.T, x=x/L, y=y/L, contours_coloring='heatmap',
                               colorbar=dict(title=var)))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    mid = Nx//2
    fig2, ax = plt.subplots()
    ax.plot(y/L, data[mid,:]); ax.set_xlabel('y/L'); ax.grid(True)
    st.pyplot(fig2); plt.close(fig2)
