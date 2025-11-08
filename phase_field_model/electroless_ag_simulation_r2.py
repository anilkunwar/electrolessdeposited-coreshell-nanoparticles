# --------------------------------------------------------------
# ELECTROLESS Ag – SHELL-FIRST + BALANCED BULK/INTERFACE
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
DB_PATH = Path("simulations_balanced.db")
def _hash_params(**kw):
    s = "".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, params BLOB, x BLOB, y BLOB,
                    phi_hist BLOB, c_hist BLOB, t_hist BLOB, psi BLOB,
                    diag BLOB
               )""")

def save_run(run_id, params, x, y, phi_hist, c_hist, t_hist, psi, diag):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)""",
                    (run_id, pickle.dumps(params), pickle.dumps(x), pickle.dumps(y),
                     pickle.dumps(phi_hist), pickle.dumps(c_hist),
                     pickle.dumps(t_hist), pickle.dumps(psi), pickle.dumps(diag)))

def load_run(run_id):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row:
            keys = ["params","x","y","phi_hist","c_hist","t_hist","psi","diag"]
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

# -------------------- 3. DOUBLE-WELL (bulk) --------------------
@njit(parallel=True, fastmath=True)
def f_prime_bulk(phi, psi, a_index, beta_tilde, h):
    """Only the double-well + template harmonic part."""
    ny, nx = phi.shape
    f = np.zeros_like(phi)
    dw  = 2.0 * beta_tilde * phi * (1.0 - phi) * (1.0 - 2.0 * phi)   # bulk
    harm = 2.0 * beta_tilde * (phi - h)                               # template
    for i in prange(ny):
        for j in prange(nx):
            f[i,j] = ((1.0 + a_index) * (1.0 - psi[i,j]) * dw[i,j] / 8.0 +
                      (1.0 - a_index) * psi[i,j] * harm[i,j] / 8.0)
    return f

# -------------------- 4. SIMULATION (balanced) --------------------
def run_simulation(
    run_id, L, Nx, Ny, eps_tilde, W_tilde, core_radius_frac, shell_thickness_frac, core_center,
    M_tilde, dt_tilde, t_max_tilde, D_tilde, c_bulk_tilde,
    k0_tilde, c_ref_tilde, alpha_tilde, beta_tilde, a_index, h,
    ratio_top_factor, ratio_surface_factor, ratio_decay_tilde,
    save_every, ui
):
    h  = L / (Nx - 1); h2 = h * h
    x  = np.linspace(0, L, Nx, dtype=np.float32)
    y  = np.linspace(0, L, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # ---- Core ----
    r_core = core_radius_frac * L
    cx, cy = core_center[0] * L, core_center[1] * L
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    psi  = (dist <= r_core).astype(np.float32)

    # ---- Shell (sharp inside, smoothed interfaces) ----
    r_inner = r_core
    r_outer = r_core * (1.0 + shell_thickness_frac)
    # sharp interior
    phi = np.where(dist <= r_inner, 0.0,
            np.where(dist <= r_outer, 1.0, 0.0)).astype(np.float32)
    # smooth the *two* interfaces with tanh of width eps
    phi = phi * (1.0 - 0.5*(1.0-np.tanh((dist-r_inner)/eps_tilde))) \
            * (1.0 - 0.5*(1.0+np.tanh((dist-r_outer)/eps_tilde)))
    phi = np.clip(phi, 0.0, 1.0)

    # ---- Concentration & ratio ----
    c = c_bulk_tilde * (Y / L) * (1.0 - phi) * (1.0 - psi)
    surface = np.exp(-np.abs(dist - (r_inner+r_outer)/2) / ratio_decay_tilde)
    vertical = Y / L
    ratio = (1.0 - ratio_surface_factor)*(0.2 + 0.8*vertical) + ratio_surface_factor*surface
    ratio = np.clip(ratio, 0.1, 8.0)

    # ---- Storage & diagnostics ----
    phi_hist, c_hist, t_hist, diag_hist = [], [], [], []
    n_steps = int(np.ceil(t_max_tilde / dt_tilde))
    save_step = max(1, save_every)
    phi_old = phi.copy()
    MAg_rho_tilde = 1.0

    progress = ui['progress']; status = ui['status']
    plot = ui['plot']; line = ui['line']; metrics = ui['metrics']

    for step in range(n_steps + 1):
        t = step * dt_tilde

        # ---- BCs ----
        phi[:,0] = phi[:,1]; phi[:,-1] = phi[:,-2]
        phi[0,:] = phi[1,:]; phi[-1,:] = phi[-2:]

        # ---- Gradients & interface indicator ----
        phi_x = _grad_x(phi, h); phi_y = _grad_y(phi, h)
        grad_phi = np.sqrt(phi_x**2 + phi_y**2 + 1e-30)
        delta_int = 6.0 * phi * (1.0 - phi) * (1.0 - psi) * grad_phi
        delta_int = np.clip(delta_int, 0.0 streak, 6.0 / eps_tilde)

        phi_xx = _laplacian(phi, h2)

        # ---- Free-energy terms ----
        f_bulk = f_prime_bulk(phi, psi, a_index, beta_tilde, h)          # bulk
        grad_term = -W_tilde * eps_tilde**2 * phi_xx                     # interfacial
        mu = grad_term + f_bulk - alpha_tilde * c
        mu_xx = _laplacian(mu, h2)

        # ---- Diagnostics (L2 norms) ----
        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_norm = np.sqrt(np.mean(grad_term**2))
        conc_norm = alpha_tilde * np.mean(c)

        # ---- Current & advection ----
        c_mol = c * (1.0 - phi) * (1.0 - psi) * ratio
        i_loc = k0_tilde * c_mol / c_ref_tilde * delta_int
        i_loc = np.clip(i_loc, 0.0, 1e3)

        u = i_loc * MAg_rho_tilde
        advection = u * (1.0 - psi) * phi_y

        # ---- Phase evolution ----
        dphi_dt = M_tilde * mu_xx + advection
        phi += dt_tilde * dphi_dt
        phi = np.clip(phi, 0.0, 1.0)

        # ---- Concentration ----
        c_eff = (1.0 - phi) * (1.0 - psi) * c
        c_xx  = _laplacian(c_eff, h2)
        sink  = -i_loc * delta_int
        c += dt_tilde * (D_tilde * c_xx + sink)
        c = np.clip(c, 0.0, c_bulk_tilde*2)

        c[:,0] = 0.0
        c[:,-1] = c_bulk_tilde * (y / L)

        # ---- Save & UI ----
        if step % save_step == 0 or step == n_steps:
            phi_hist.append(phi.copy())
            c_hist.append(c.copy())
            t_hist.append(t)
            diag_hist.append((bulk_norm, grad_norm, conc_norm))

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
                metrics.metric('ϕ range', f"[{phi.min():.4f},{phi.max():.4f}]")
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
        't_hist': np.array(t_hist), 'psi': psi,
        'diag': np.array(diag_hist)   # (bulk, grad, conc) per saved step
    }

# -------------------- 5. UI --------------------
st.title("Electroless Ag – Balanced Bulk vs. Interface")
st.markdown("**Shell-first, sharp interior, smooth interfaces, diagnostics**")

st.sidebar.header("Domain")
L  = st.sidebar.slider("L (cm)", 1e-6, 1e-5, 5e-6, 1e-7, format="%e")
Nx = st.sidebar.slider("Nx", 80, 300, 160, 10)
Ny = st.sidebar.slider("Ny", 80, 300, 160, 10)

st.sidebar.header("Core & Shell")
core_radius_frac = st.sidebar.slider("Core r/L", 0.31, 0.7, 0.5, 0.01)
shell_thickness_frac = st.sidebar.slider("Shell Δr / r_core", 0.1, 0.5, 0.25, 0.01)
core_center_x = st.sidebar.slider("Core x/L", 0.2, 0.8, 0.5, 0.01)
core_center_y = st.sidebar.slider("Core y/L", 0.2, 0.8, 0.5, 0.01)

st.sidebar.header("Interface Energy")
eps_tilde = st.sidebar.slider("ε* (interface width)", 0.01, 0.08, 0.03, 0.005)
W_tilde   = st.sidebar.slider("W* (gradient prefactor)", 0.1, 10.0, 1.0, 0.1)

st.sidebar.header("Kinetics")
M_tilde   = st.sidebar.number_input("M*", 1e-3, 1.0, 0.1, 0.01)
dt_tilde  = st.sidebar.number_input("Δt*", 1e-4, 1e-2, 5e-4, 1e-5)
t_max_tilde = st.sidebar.number_input("t*_max", 1.0, 30.0, 12.0, 0.5)
D_tilde   = st.sidebar.number_input("D*", 0.01, 1.0, 0.1, 0.01)
c_bulk_tilde = st.sidebar.number_input("c*_bulk", 0.1, 10.0, 1.0, 0.1)

st.sidebar.header("Electroless Reaction")
k0_tilde  = st.sidebar.number_input("k₀*", 0.01, 1.0, 0.1, 0.01)
c_ref_tilde = st.sidebar.number_input("c*_ref", 0.1, 10.0, 1.0, 0.1)

st.sidebar.header("Coupling")
alpha_tilde = st.sidebar.number_input("α*", 0.0, 10.0, 1.0, 0.1)
beta_tilde  = st.sidebar.slider("β* (double-well depth)", 0.1, 5.0, 1.0, 0.1)
a_index     = st.sidebar.slider("a-index", -1.0, 1.0, 0.0, 0.1)
h           = st.sidebar.slider("h", 0.0, 1.0, 0.5, 0.1)

st.sidebar.header("Ratio Field")
ratio_top_factor      = st.sidebar.slider("Top-weight", 0.0, 1.0, 0.7, 0.05)
ratio_surface_factor  = st.sidebar.slider("Surface-boost", 0.0, 1.0, 0.5, 0.05)
ratio_decay_tilde     = st.sidebar.slider("Decay λ*/L", 0.01, 0.2, 0.05, 0.01)

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
        status.info("Running …")
        ui = {'progress': progress, 'status': status,
              'plot': plot_area, 'line': line_area, 'metrics': metrics_area}
        results = run_simulation(
            run_id, L, Nx, Ny, eps_tilde, W_tilde,
            core_radius_frac, shell_thickness_frac, (core_center_x, core_center_y),
            M_tilde, dt_tilde, t_max_tilde, D_tilde, c_bulk_tilde,
            k0_tilde, c_ref_tilde, alpha_tilde, beta_tilde, a_index, h,
            ratio_top_factor, ratio_surface_factor, ratio_decay_tilde,
            save_every, ui
        )
        save_run(run_id, {}, results['x'], results['y'],
                 results['phi_hist'], results['c_hist'],
                 results['t_hist'], results['psi'], results['diag'])
        st.session_state.results = results
        status.success("Finished!")

if stop_col.button("Stop"): st.session_state.stop_sim = True

# -------------------- Results & Diagnostics --------------------
if st.session_state.results:
    r = st.session_state.results
    x, y = r['x'], r['y']
    phi_hist, c_hist = r['phi_hist'], r['c_hist']
    t_hist = r['t_hist']; psi = r['psi']
    diag = r['diag']                     # (bulk, grad, conc) per saved step

    st.subheader("Results")
    time_idx = st.slider("Time t*", 0, len(t_hist)-1, 0, format="t* = %.3f")
    var = st.selectbox("Field", ["ϕ", "c", "ψ"])
    data = phi_hist[time_idx] if var == "ϕ" else c_hist[time_idx] if var == "c" else psi

    fig = go.Figure(go.Contour(z=data.T, x=x/L, y=y/L,
                               contours_coloring='heatmap',
                               colorbar=dict(title=var)))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Line plot ----
    mid = Nx//2
    fig2, ax = plt.subplots()
    ax.plot(y/L, data[mid,:], label=var)
    ax.set_xlabel('y/L'); ax.grid(True); ax.legend()
    st.pyplot(fig2); plt.close(fig2)

    # ---- Diagnostics ----
    st.subheader("Energy-Term Balance (L² norms)")
    bulk_norm, grad_norm, conc_norm = diag.T
    df = {
        "t*": t_hist[::save_every],
        "‖bulk‖₂": bulk_norm,
        "‖grad‖₂": grad_norm,
        "‖αc‖": conc_norm
    }
    import pandas as pd
    df = pd.DataFrame(df)
    st.dataframe(df.style.format("{:.2e}"))

    # Plot norms vs time
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t_hist[::save_every], y=bulk_norm, name='bulk'))
    fig3.add_trace(go.Scatter(x=t_hist[::save_every], y=grad_norm, name='gradient'))
    fig3.add_trace(go.Scatter(x=t_hist[::save_every], y=conc_norm, name='αc'))
    fig3.update_layout(xaxis_title='t*', yaxis_type='log', height=400)
    st.plotly_chart(fig3, use_container_width=True)
