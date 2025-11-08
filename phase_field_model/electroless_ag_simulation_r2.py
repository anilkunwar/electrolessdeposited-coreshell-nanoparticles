# --------------------------------------------------------------
# ELECTROLESS Ag – LOGARITHMIC OBSTACLE + POLYNOMIAL FREE ENERGY
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
DB_PATH = Path("simulations.db")
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
            return {k: pickle.loads(v) for k, v in zip(["params","x","y","phi_hist","c_hist","t_hist","psi"], row[1:])}
    return None

def list_runs():
    with sqlite3.connect(DB_PATH) as con:
        return [r[0][:8] for r in con.execute("SELECT run_id FROM runs").fetchall()]

def delete_all():
    DB_PATH.unlink(missing_ok=True); init_db()

# -------------------- 2. Numba kernels --------------------
@njit(parallel=True, fastmath=True)
def _laplacian(arr, dx2):
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for i in prange(1, ny-1):
        for j in prange(1, nx-1):
            out[i,j] = (arr[i+1,j] + arr[i-1,j] + arr[i,j+1] + arr[i,j-1] - 4*arr[i,j]) / dx2
    return out

@njit(parallel=True, fastmath=True)
def _grad_x(arr, dx):
    ny, nx = arr.shape; out = np.zeros_like(arr)
    for i in prange(ny):
        for j in prange(1, nx-1):
            out[i,j] = (arr[i,j+1] - arr[i,j-1]) / (2*dx)
    return out

@njit(parallel=True, fastmath=True)
def _grad_y(arr, dy):
    ny, nx = arr.shape; out = np.zeros_like(arr)
    for i in prange(1, ny-1):
        for j in prange(nx):
            out[i,j] = (arr[i+1,j] - arr[i-1,j]) / (2*dy)
    return out

@njit(parallel=True, fastmath=True)
def _distance_to_edge(psi, dx, dy):
    ny, nx = psi.shape; dist = np.zeros_like(psi)
    for i in prange(ny):
        for j in prange(nx):
            if psi[i,j] > 0.5: dist[i,j] = -1.0
            else:
                dmin = 1e9
                for di in (-1,0,1):
                    for dj in (-1,0,1):
                        ii, jj = i+di, j+dj
                        if 0 <= ii < ny and 0 <= jj < nx and psi[ii,jj] > 0.5:
                            d = np.sqrt((di*dy)**2 + (dj*dx)**2)
                            if d < dmin: dmin = d
                dist[i,j] = dmin if dmin < 1e9 else 1.0
    return dist

# -------------------- 3. Simulation --------------------
def run_simulation_live(
    run_id, Lx, Ly, Nx, Ny, epsilon, y0,
    M, dt, t_max, D, c_bulk,
    z, F, R, T, alpha, i0, c_ref,
    M_Ag, rho_Ag, beta, a_index, h, kBT, eps_log,
    psi, AgNH3_conc, Cu_ion_conc, eta_chem,
    ratio_top_factor, ratio_surface_factor, ratio_decay_len,
    save_every, ui
):
    dx = Lx / (Nx - 1); dy = Ly / (Ny - 1); dx2 = dx*dx
    x = np.linspace(0, Lx, Nx, dtype=np.float32)
    y = np.linspace(0, Ly, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initial phi
    phi = ((1 - psi) * 0.5 * (1 - np.tanh((Y - y0) / epsilon))).astype(np.float32)

    # Ratio field
    base_ratio = np.float32(AgNH3_conc / (Cu_ion_conc + 1e-12))
    vertical = Y / Ly
    dist = _distance_to_edge(psi, dx, dy)
    surface_enh = np.exp(-np.abs(dist) / ratio_decay_len)
    ratio_field = base_ratio * (
        (1.0 - ratio_surface_factor) * (0.2 + 0.8 * vertical) +
        ratio_surface_factor * surface_enh
    )
    ratio_field = np.clip(ratio_field, 0.1*base_ratio, 8.0*base_ratio)

    # Concentration
    c = (c_bulk * vertical * (1 - phi) * (1 - psi)).astype(np.float32)

    Fz = np.float32(z * F); RT = np.float32(R * T)
    MAg_rho = np.float32(M_Ag / rho_Ag * 1e-2)

    phi_hist, c_hist, t_hist = [], [], []
    n_steps = int(np.ceil(t_max / dt)); save_step = max(1, save_every)
    phi_old = phi.copy()

    progress_bar = ui['progress']; status = ui['status']
    plot_area = ui['plot']; line_area = ui['line']; metrics_area = ui['metrics']

    for step in range(n_steps + 1):
        t = step * dt

        # --- Neumann BCs (no flux) ---
        phi[:, 0] = phi[:, 1]; phi[:, -1] = phi[:, -2]
        phi[0, :] = phi[1, :]; phi[-1, :] = phi[-2, :]

        # --- Gradients ---
        phi_x = _grad_x(phi, dx); phi_y = _grad_y(phi, dy)
        grad_phi_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-30)
        delta_int = 6 * phi * (1 - phi) * (1 - psi) * grad_phi_mag
        delta_int = np.clip(delta_int, 0.0, 6.0 / epsilon)

        phi_xx = _laplacian(phi, dx2)

        # --- FREE ENERGY DERIVATIVE f'(ϕ,ψ) ---
        # Polynomial part: W(2ϕ−1)(1−2ϕ+2ϕ²)
        z = 2*phi - 1
        poly_term = (1 - 2*phi + 2*phi*phi)  # = 2(z²)
        f_prime_poly = (1 + a_index) * (1 - psi) * poly_term

        # Template pinning
        f_prime_template = (1 - a_index) * psi * 2 * beta * (phi - h)

        # Logarithmic entropic term: k_B T [ln(ϕ/(1−ϕ)) + ϵ/(ϕ(1−ϕ))]
        eps = 1e-12  # prevent log(0)
        phi_safe = np.clip(phi, eps, 1-eps)
        log_term = kBT * (np.log(phi_safe / (1 - phi_safe)) + eps_log / (phi_safe * (1 - phi_safe)))

        f_prime = f_prime_poly + f_prime_template + log_term

        mu = -epsilon**2 * phi_xx + f_prime - alpha * c
        mu_xx = _laplacian(mu, dx2)

        # --- Current & Advection ---
        c_mol = c * 1e6 * (1 - phi) * (1 - psi) * ratio_field
        i_loc = i0 * (c_mol / c_ref) * np.exp(0.5 * Fz * eta_chem / RT)
        i_loc = i_loc * delta_int
        i_loc = np.clip(i_loc, -1e5, 1e5)

        u = - (i_loc / Fz) * MAg_rho
        advection = u * (1 - psi) * phi_y

        # --- Phase evolution ---
        dphi_dt = M * mu_xx - advection
        phi += dt * dphi_dt

        # --- Concentration ---
        c_eff = (1 - phi) * (1 - psi) * c
        c_xx = _laplacian(c_eff, dx2)
        sink = - i_loc * delta_int / (Fz * 1e6)
        c += dt * (D * c_xx + sink)
        c = np.clip(c, 0.0, c_bulk * 2.0)

        # --- BCs ---
        c[:, 0] = 0.0; c[:, -1] = c_bulk * vertical[:, -1]

        # --- Save & UI ---
        if step % save_step == 0 or step == n_steps:
            phi_hist.append(phi.copy()); c_hist.append(c.copy()); t_hist.append(t)
            try:
                fig = go.Figure(go.Contour(z=phi_hist[-1].T, x=x, y=y, contours_coloring='heatmap',
                                          colorbar=dict(title='ϕ')))
                fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=20))
                plot_area.plotly_chart(fig, use_container_width=True)

                mid = Nx//2
                fig2, ax = plt.subplots()
                ax.plot(y, phi_hist[-1][mid,:], label='ϕ')
                ax.plot(y, ratio_field[mid,:], '--', label='ratio')
                ax.set_xlabel('y (cm)'); ax.set_title(f't={t:.3f}s'); ax.legend(); ax.grid(True)
                line_area.pyplot(fig2); plt.close(fig2)

                metrics_area.metric('t (s)', f"{t:.3f}")
                metrics_area.metric('ϕ range', f"[{phi.min():.3f}, {phi.max():.3f}]")
            except: pass

        if step > 100 and np.max(np.abs(phi - phi_old)) < 1e-6:
            status.info(f"Converged at t={t:.4f}s"); break
        if step % 50 == 0: phi_old = phi.copy()

        progress_bar.progress(min(1.0, step / n_steps))
        if st.session_state.get('stop_sim', False): status.warning('Stopped'); break

    progress_bar.empty()
    return {
        'x': x, 'y': y, 'phi_hist': np.array(phi_hist), 'c_hist': np.array(c_hist),
        't_hist': np.array(t_hist), 'psi': psi, 'ratio_field': ratio_field
    }

# -------------------- 4. Template --------------------
def create_template(Lx, Ly, Nx, Ny, template_type, radius, side_length, param1, param2, param_func):
    x = np.linspace(0, Lx, Nx); y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij"); psi = np.zeros((Nx, Ny), dtype=np.float32)
    if template_type == "Circle":
        psi = ((X - Lx/2)**2 + (Y - Ly/2)**2 <= radius**2).astype(np.float32)
    elif template_type == "Semicircle":
        psi = (((X - Lx/2)**2 + Y**2 <= radius**2) & (Y >= 0)).astype(np.float32)
    elif template_type == "Square":
        psi = (np.abs(X - Lx/2) <= side_length/2) & (Y >= 0) & (Y <= side_length)
        psi = psi.astype(np.float32)
    elif template_type == "Parametric":
        try:
            g = eval(param_func, {"x": X-Lx/2, "y": Y, "p1": param1, "p2": param2, "np": np})
            psi = (g <= 0).astype(np.float32)
        except Exception as e: st.error(f"Parametric error: {e}")
    return psi

# -------------------- 5. UI --------------------
st.title("Electroless Ag – Logarithmic + Polynomial Free Energy")
st.markdown("**Thermodynamically consistent, bounded ϕ ∈ (0,1), stable shell growth.**")

st.sidebar.header("Domain")
Lx = st.sidebar.slider("Lx (cm)", 1e-6, 1e-5, 5e-6, 1e-7, format="%e")
Ly = st.sidebar.slider("Ly (cm)", 1e-6, 1e-5, 5e-6, 1e-7, format="%e")
Nx = st.sidebar.slider("Nx", 60, 250, 120, 10)
Ny = st.sidebar.slider("Ny", 60, 250, 120, 10)
epsilon = st.sidebar.slider("ε (cm)", 1e-8, 1e-7, 5e-8, 1e-8, format="%e")
y0 = Lx / 2

st.sidebar.header("Physics")
M = st.sidebar.number_input("M (cm²/s)", 1e-7, 1e-5, 1e-6, 1e-7, format="%e")
dt = st.sidebar.number_input("Δt (s)", 1e-7, 5e-6, 5e-7, 1e-7, format="%e")
t_max = st.sidebar.number_input("t_max (s)", 1.0, 20.0, 8.0, 0.5)
c_bulk = st.sidebar.number_input("c_bulk (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6, format="%e")
D = st.sidebar.number_input("D (cm²/s)", 5e-6, 2e-5, 1e-5, 1e-6, format="%e")
alpha = st.sidebar.number_input("α", 0.0, 1.0, 0.1, 0.01)
i0 = st.sidebar.number_input("i₀ (A/m²)", 0.1, 2.0, 0.5, 0.1)
c_ref = st.sidebar.number_input("c_ref (mol/m³)", 100, 2000, 1000, 100)
beta = st.sidebar.slider("β", 0.1, 10.0, 1.0, 0.1)
a_index = st.sidebar.slider("a-index", -1.0, 1.0, 0.0, 0.1)
h = st.sidebar.slider("h", 0.0, 1.0, 0.5, 0.1)
kBT = st.sidebar.number_input("k_B T (J)", 1e-21, 1e-19, 4.14e-21, 1e-22, help="~kT at 300K")  # 4.14e-21 J
eps_log = st.sidebar.number_input("ϵ_log", 1e-3, 10.0, 1.0, 0.1)

st.sidebar.header("Species")
AgNH3_conc = st.sidebar.number_input("[Ag(NH₃)₂]⁺ (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6, format="%e")
Cu_ion_conc = st.sidebar.number_input("[Cu²⁺] (mol/cm³)", 1e-6, 5e-4, 5e-5, 1e-6, format="%e")
eta_chem = st.sidebar.slider("η_chem (V)", 0.1, 0.5, 0.3, 0.05)

st.sidebar.header("Ratio")
ratio_top_factor = st.sidebar.slider("Top-weight", 0.0, 1.0, 0.7, 0.05)
ratio_surface_factor = st.sidebar.slider("Surface-boost", 0.0, 1.0, 0.5, 0.05)
ratio_decay_len = st.sidebar.slider("Decay (cm)", 1e-8, 5e-7, 2e-7, 1e-8, format="%e")

save_every = st.sidebar.number_input("Save every", 5, 50, 15, 1)

st.sidebar.header("Template")
template_type = st.sidebar.selectbox("Shape", ["Circle", "Semicircle", "Square", "Parametric"])
radius = 2e-7
side_length = st.sidebar.slider("Side (cm)", 1e-7, 1e-6, 2e-7, 1e-8, format="%e") if template_type == "Square" else 2e-7
param_func = st.sidebar.text_input("g(x,y,p1,p2)", "(x/p1)**2 + (y/p2)**2 - 1") if template_type == "Parametric" else ""
param1 = st.sidebar.slider("p1", 1e-7, 1e-6, 2e-7, 1e-8, format="%e") if template_type == "Parametric" else 2e-7
param2 = st.sidebar.slider("p2", 1e-7, 1e-6, 2e-7, 1e-8, format="%e") if template_type == "Parametric" else 2e-7

psi = create_template(Lx, Ly, Nx, Ny, template_type, radius, side_length, param1, param2, param_func)

init_db()
if "run_id" not in st.session_state: st.session_state.run_id = None
if "results" not in st.session_state: st.session_state.results = None
if "stop_sim" not in st.session_state: st.session_state.stop_sim = False

col1, col2 = st.columns([3, 1])
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
        results = run_simulation_live(
            run_id, Lx, Ly, Nx, Ny, epsilon, y0, M, dt, t_max, D, c_bulk,
            1, 96485, 8.314, 298, alpha, i0, c_ref, 0.10787, 10500, beta, a_index, h, kBT, eps_log,
            psi, AgNH3_conc, Cu_ion_conc, eta_chem,
            ratio_top_factor, ratio_surface_factor, ratio_decay_len, save_every, ui
        )
        save_run(run_id, {}, results['x'], results['y'], results['phi_hist'], results['c_hist'], results['t_hist'], results['psi'])
        st.session_state.results = results; status.success("Done!")

if stop_col.button("Stop"): st.session_state.stop_sim = True

if st.session_state.results:
    r = st.session_state.results
    x, y = r['x'], r['y']
    phi_hist, c_hist = r['phi_hist'], r['c_hist']
    t_hist = r['t_hist']; psi = r['psi']; ratio_field = r.get('ratio_field')

    st.subheader("Results")
    time_idx = st.slider("Time", 0, len(t_hist)-1, len(t_hist)//2, format="t = %.2f s")
    var = st.selectbox("Variable", ["ϕ", "c", "ψ", "ratio"])
    data = phi_hist[time_idx] if var == "ϕ" else c_hist[time_idx] if var == "c" else psi if var == "ψ" else ratio_field

    fig = go.Figure(go.Contour(z=data.T, x=x, y=y, contours_coloring='heatmap', colorbar=dict(title=var)))
    fig.update_layout(height=460, margin=dict(l=10,r=10,t=30,b=40))
    st.plotly_chart(fig, use_container_width=True)

    mid = Nx//2
    fig2, ax = plt.subplots()
    ax.plot(y, data[mid,:]); ax.set_xlabel('y (cm)'); ax.set_title(f'{var} at x=Lx/2')
    ax.grid(True); st.pyplot(fig2); plt.close(fig2)

    def vtr_bytes(phi, c, psi, x, y):
        grid = pv.RectilinearGrid(x, y, [0])
        grid.point_data["phi"] = phi.T.ravel(order="F")
        grid.point_data["c"] = c.T.ravel(order="F")
        grid.point_data["psi"] = psi.T.ravel(order="F")
        bio = BytesIO(); grid.save(bio, file_format="vtr"); return bio.getvalue()
    vtr = vtr_bytes(phi_hist[time_idx], c_hist[time_idx], psi, x, y)
    st.download_button(f"VTR t={t_hist[time_idx]:.2f}s", vtr, f"ag_{t_hist[time_idx]:.2f}.vtr", "application/octet-stream")

if DB_PATH.exists():
    st.download_button("Download DB", DB_PATH.read_bytes(), "simulations.db", "application/octet-stream")
