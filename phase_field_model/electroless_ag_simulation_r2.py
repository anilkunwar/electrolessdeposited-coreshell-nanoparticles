# --------------------------------------------------------------
# ELECTROLESS Ag DEPOSITION – SHELL-GROWTH + SPATIAL RATIO
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

# -------------------- 1. SQLite helper --------------------
DB_PATH = Path("simulations.db")
def _hash_params(**kw):
    s = "".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    params BLOB, x BLOB, y BLOB,
                    phi_hist BLOB, c_hist BLOB, phi_l_hist BLOB,
                    t_hist BLOB, psi BLOB
               )"""
        )

def save_run(run_id, params, x, y, phi_hist, c_hist, phi_l_hist, t_hist, psi):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """INSERT OR REPLACE INTO runs
               (run_id,params,x,y,phi_hist,c_hist,phi_l_hist,t_hist,psi)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (run_id,
             pickle.dumps(params),
             pickle.dumps(x), pickle.dumps(y),
             pickle.dumps(phi_hist), pickle.dumps(c_hist),
             pickle.dumps(phi_l_hist), pickle.dumps(t_hist),
             pickle.dumps(psi)),
        )

def load_run(run_id):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "SELECT params,x,y,phi_hist,c_hist,phi_l_hist,t_hist,psi FROM runs WHERE run_id=?",
            (run_id,),
        )
        row = cur.fetchone()
        if row:
            keys = ["params","x","y","phi_hist","c_hist","phi_l_hist","t_hist","psi"]
            return {k: pickle.loads(v) for k, v in zip(keys, row)}
    return None

def list_runs():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT run_id FROM runs")
        return [r[0][:8] for r in cur.fetchall()]

def delete_all():
    DB_PATH.unlink(missing_ok=True)
    init_db()

# -------------------- 2. Numba kernels --------------------
@njit(parallel=True, fastmath=True)
def _laplacian(arr, dx2):
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for i in prange(1, ny - 1):
        for j in prange(1, nx - 1):
            out[i, j] = (arr[i + 1, j] + arr[i - 1, j] +
                         arr[i, j + 1] + arr[i, j - 1] - 4 * arr[i, j]) / dx2
    return out

@njit(parallel=True, fastmath=True)
def _grad_x(arr, dx):
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for i in prange(ny):
        for j in prange(1, nx - 1):
            out[i, j] = (arr[i, j + 1] - arr[i, j - 1]) / (2 * dx)
    return out

@njit(parallel=True, fastmath=True)
def _grad_y(arr, dy):
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for i in prange(1, ny - 1):
        for j in prange(nx):
            out[i, j] = (arr[i + 1, j] - arr[i - 1, j]) / (2 * dy)
    return out

# -------------------- 3. Helper: distance to template edge --------------------
@njit(parallel=True, fastmath=True)
def _distance_to_edge(psi, dx, dy):
    """Signed-distance-like field (positive outside, zero on edge, negative inside)."""
    ny, nx = psi.shape
    dist = np.zeros_like(psi)
    for i in prange(ny):
        for j in prange(nx):
            if psi[i, j] > 0.5:                     # inside template
                dist[i, j] = -1.0
            else:
                # look for nearest inside point (simple 4-neighbour search)
                dmin = 1e9
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < ny and 0 <= jj < nx and psi[ii, jj] > 0.5:
                            d = np.sqrt((di * dy)**2 + (dj * dx)**2)
                            if d < dmin:
                                dmin = d
                dist[i, j] = dmin if dmin < 1e9 else 1.0
    return dist

# -------------------- 4. Live simulation wrapper --------------------
def run_simulation_live(
    run_id,
    Lx, Ly,  Ly, Nx, Ny, epsilon, y0,
    M, dt, t_max, c_bulk, D,
    z, F, R, T, alpha, i0, c_ref,
    M_Ag, rho_Ag, beta, a_index, h,
    psi, AgNH3_conc, Cu_ion_conc, eta_chem,
    ratio_top_factor, ratio_surface_factor, ratio_decay_len,
    save_every,
    ui,
):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dx2 = dx * dx
    x = np.linspace(0, Lx, Nx, dtype=np.float32)
    y = np.linspace(0, Ly, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # ---- initialise phase field ----
    phi = ((1 - psi) * 0.5 * (1 - np.tanh((Y - y0) / epsilon))).astype(np.float32)

    # ---- SPATIAL RATIO FIELD -------------------------------------------------
    base_ratio = np.float32(AgNH3_conc / (Cu_ion_conc + 1e-12))

    # 1) Vertical gradient (high at top)
    vertical = Y / Ly                                   # 0 at bottom, 1 at top

    # 2) Surface enhancement (high near template edge)
    dist = _distance_to_edge(psi, dx, dy)               # >0 outside, <0 inside
    surface_enh = np.exp(-np.abs(dist) / ratio_decay_len)  # peaks at edge

    # Combine:  ratio = base * (a*vertical + b*surface)  with a+b≈1
    ratio_field = base_ratio * (
        (1.0 - ratio_surface_factor) * (0.2 + 0.8 * vertical) +
        ratio_surface_factor * surface_enh
    )
    # clamp to avoid numerical blow-up
    ratio_field = np.clip(ratio_field, 0.1 * base_ratio, 10.0 * base_ratio)

    # ---- concentration field (no ratio here – it will be applied in i_loc) ----
    c = (c_bulk * vertical * (1 - phi) * (1 - psi)).astype(np.float32)

    # pre-compute constants
    Fz = np.float32(z * F)
    RT = np.float32(R * T)
    MAg_rho = np.float32(M_Ag / rho_Ag * 1e-2)

    # storage
    phi_hist, c_hist, phi_l_hist, t_hist = [], [], [], []
    n_steps = int(np.ceil(t_max / dt))
    save_step = max(1, save_every)
    phi_old = phi.copy()

    # UI placeholders
    progress_bar = ui['progress']
    status = ui['status']
    plot_area = ui['plot']
    line_area = ui['line']
    metrics_area = ui['metrics']

    for step in range(n_steps + 1):
        t = step * dt

        # ---- gradients & interface term ----
        phi_x = _grad_x(phi, dx)
        phi_y = _grad_y(phi, dy)
        grad_phi_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-30)
        delta_int = 6 * phi * (1 - phi) * (1 - psi) * grad_phi_mag

        phi_xx = _laplacian(phi, dx2)
        f_prime_ed = beta * 2 * phi * (1 - phi) * (1 - 2 * phi)
        f_prime_tp = beta * 2 * (phi - h)
        f_prime = ((1 + a_index) / 8) * (1 - psi) * f_prime_ed + \
                  ((1 - a_index) / 8) * psi * f_prime_tp
        mu = -epsilon**2 * phi_xx + f_prime - alpha * c
        mu_xx = _laplacian(mu, dx2)

        # ---- local current (driving force) ----
        c_mol = c * 1e6 * (1 - phi) * (1 - psi) * ratio_field
        i_loc = i0 * (c_mol / c_ref) * np.exp(0.5 * Fz * eta_chem / RT)
        i_loc = i_loc * delta_int

        # ---- phase-field evolution ----
        u = - (i_loc / Fz) * MAg_rho
        advection = u * (1 - psi) * phi_y
        phi += dt * (M * mu_xx - advection)

        # ---- concentration evolution ----
        c_eff = (1 - phi) * (1 - psi) * c
        c_xx = _laplacian(c_eff, dx2)
        sink = - i_loc * delta_int / (Fz * 1e6)
        c_t = D * c_xx + sink
        c += dt * c_t

        # ---- BCs (preserve vertical gradient) ----
        c[:, 0] = 0.0
        c[:, -1] = c_bulk * vertical[:, -1]

        # ---- store & UI update ----
        if step % save_step == 0 or step == n_steps:
            phi_hist.append(phi.copy())
            c_hist.append(c.copy())
            phi_l_hist.append(np.zeros_like(phi))
            t_hist.append(t)

            try:
                fig = go.Figure(go.Contour(z=phi_hist[-1].T, x=x, y=y,
                                          contours_coloring='heatmap',
                                          colorbar=dict(title='φ')))
                fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=20))
                plot_area.plotly_chart(fig, use_container_width=True)

                mid_x = Nx // 2
                fig2, ax = plt.subplots()
                ax.plot(y, phi_hist[-1][mid_x, :], label='φ')
                ax.plot(y, ratio_field[mid_x, :], '--', label='ratio')
                ax.set_xlabel('y (cm)'); ax.set_title(f't={t:.3f}s')
                ax.legend(); ax.grid(True)
                line_area.pyplot(fig2); plt.close(fig2)

                metrics_area.metric('t (s)', f"{t:.3f}")
                metrics_area.metric('max(φ)', f"{phi.max():.4f}")
                metrics_area.metric('mean(φ)', f"{phi.mean():.4f}")
            except Exception:
                pass

        # convergence / early stop
        if step > 100 and np.max(np.abs(phi - phi_old)) < 1e-6:
            status.info(f"Converged at t={t:.4f}s")
            break
        if step % 50 == 0:
            phi_old = phi.copy()

        progress_bar.progress(min(1.0, step / n_steps))
        if st.session_state.get('stop_sim', False):
            status.warning('Stop requested – terminating...')
            break

    progress_bar.empty()
    results = {
        'x': x, 'y': y,
        'phi_hist': np.array(phi_hist), 'c_hist': np.array(c_hist),
        'phi_l_hist': np.array(phi_l_hist), 't_hist': np.array(t_hist),
        'psi': psi, 'ratio_field': ratio_field,
    }
    return results

# -------------------- 5. Template geometry --------------------
def create_template(Lx, Ly, Nx, Ny, template_type,
                    radius, side_length, param1, param2, param_func):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    psi = np.zeros((Nx, Ny), dtype=np.float32)
    if template_type == "Circle":
        psi = ((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2 <= radius ** 2).astype(np.float32)
    elif template_type == "Semicircle":
        psi = (((X - Lx / 2) ** 2 + Y ** 2 <= radius ** 2) & (Y >= 0)).astype(np.float32)
    elif template_type == "Square":
        psi = (np.abs(X - Lx / 2) <= side_length / 2) & (Y >= 0) & (Y <= side_length)
        psi = psi.astype(np.float32)
    elif template_type == "Parametric":
        try:
            g = eval(param_func,
                     {"x": X - Lx / 2, "y": Y, "p1": param1, "p2": param2, "np": np})
            psi = (g <= 0).astype(np.float32)
        except Exception as e:
            st.error(f"Parametric error: {e}")
    return psi

# -------------------- 6. UI ------------------------------------
st.title("Electroless Ag Shell Growth – Spatial Ratio Control")
st.markdown("""
**2-D phase-field** with **spatially varying [Ag]/[Cu] ratio** → controlled shell growth.  
Stop button works live. All runs cached in `simulations.db`.
""")

# ---------- sidebar ----------
st.sidebar.header("Domain & Discretisation")
Lx = st.sidebar.slider("Lx (cm)", 1e-6, 1e-5, 5e-6, 1e-7, format="%e")
Ly = st.sidebar.slider("Ly (cm)", 1e-6, 1e-5, 5e-6, 1e-7, format="%e")
Nx = st.sidebar.slider("Nx", 60, 250, 120, 10)
Ny = st.sidebar.slider("Ny", 60, 250, 120, 10)
epsilon = st.sidebar.slider("ε (cm)", 1e-8, 1e-7, 5e-8, 1e-8, format="%e")
y0 = Lx / 2

st.sidebar.header("Physics")
M = st.sidebar.number_input("M (cm²/s)", 1e-6, 1e-4, 1e-5, 1e-6, format="%e")
dt = st.sidebar.number_input("Δt (s)", 5e-7, 5e-5, 5e-6, 1e-7, format="%e")
t_max = st.sidebar.number_input("t_max (s)", 1.0, 20.0, 6.0, 0.5)
c_bulk = st.sidebar.number_input("c_bulk (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6, format="%e")
D = st.sidebar.number_input("D (cm²/s)", 5e-6, 2e-5, 1e-5, 1e-6, format="%e")
z, F, R, T = 1, 96485, 8.314, 298
alpha = st.sidebar.number_input("α", 0.0, 1.0, 0.1, 0.01)
i0 = st.sidebar.number_input("i₀ (A/m²)", 0.1, 10.0, 0.5, 0.1)
c_ref = st.sidebar.number_input("c_ref (mol/m³)", 100, 2000, 1000, 100)
M_Ag, rho_Ag = 0.10787, 10500
beta = st.sidebar.slider("β", 0.1, 10.0, 1.0, 0.1)
a_index = st.sidebar.slider("a-index", -1.0, 1.0, 0.0, 0.1)
h = st.sidebar.slider("h", 0.0, 1.0, 0.5, 0.1)

st.sidebar.header("Species")
AgNH3_conc = st.sidebar.number_input("[Ag(NH₃)₂]⁺ (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6, format="%e")
Cu_ion_conc = st.sidebar.number_input("[Cu²⁺] (mol/cm³)", 1e-6, 5e-4, 5e-5, 1e-6, format="%e")
eta_chem = st.sidebar.slider("η_chem (V)", 0.1, 0.5, 0.3, 0.05)

st.sidebar.header("Ratio Gradient Control")
ratio_top_factor = st.sidebar.slider("Top-weight (vertical)", 0.0, 1.0, 0.8, 0.05,
                                     help="1 → ratio max at top, 0 → uniform")
ratio_surface_factor = st.sidebar.slider("Surface-boost", 0.0, 1.0, 0.6, 0.05,
                                         help="1 → strong peak at template edge")
ratio_decay_len = st.sidebar.slider("Decay length (cm)", 1e-8, 5e-7, 1e-7, 1e-8, format="%e")

save_every = st.sidebar.number_input("Save every N steps", 5, 50, 20, 1)

st.sidebar.header("Template")
template_type = st.sidebar.selectbox("Shape", ["Circle", "Semicircle", "Square", "Parametric"])
radius = 1e-6
side_length = st.sidebar.slider("Square side (cm)", 1e-7, 1e-6, 2e-7, 1e-8, format="%e") if template_type == "Square" else 2e-7
param_func = st.sidebar.text_input("g(x,y,p1,p2)", "(x/p1)**2 + (y/p2)**2 - 1") if template_type == "Parametric" else ""
param1 = st.sidebar.slider("p1", 1e-7, 1e-6, 2e-7, 1e-8, format="%e") if template_type == "Parametric" else 2e-7
param2 = st.sidebar.slider("p2", 1e-7, 1e-6, 2e-7, 1e-8, format="%e") if template_type == "Parametric" else 2e-7

psi = create_template(Lx, Ly, Nx, Ny, template_type,
                      radius, side_length, param1, param2, param_func)

# ---------- DB ----------
init_db()
if "run_id" not in st.session_state:   st.session_state.run_id = None
if "results" not in st.session_state:  st.session_state.results = None
if "stop_sim" not in st.session_state: st.session_state.stop_sim = False

# UI placeholders
col1, col2 = st.columns([3, 1])
plot_area = col1.container()
line_area = col1.container()
metrics_area = col2.container()
status = st.empty()
progress = st.empty()

# Run / Stop
run_col, stop_col = st.columns(2)
if run_col.button("Run Simulation"):
    st.session_state.stop_sim = False
    run_id = _hash_params(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, epsilon=epsilon, y0=y0,
        M=M, dt=dt, t_max=t_max, c_bulk=c_bulk, D=D,
        alpha=alpha, i0=i0, c_ref=c_ref,
        beta=beta, a_index=a_index, h=h,
        AgNH3_conc=AgNH3_conc, Cu_ion_conc=Cu_ion_conc,
        eta_chem=eta_chem, save_every=save_every,
        template_type=template_type, radius=radius,
        side_length=side_length, param1=param1, param2=param2,
        param_func=param_func,
        ratio_top_factor=ratio_top_factor,
        ratio_surface_factor=ratio_surface_factor,
        ratio_decay_len=ratio_decay_len,
    )
    cached = load_run(run_id)
    if cached:
        st.session_state.run_id = run_id
        st.session_state.results = cached
        status.success("Loaded cached run")
    else:
        status.info("Starting simulation…")
        ui = {'progress': progress, 'status': status,
              'plot': plot_area, 'line': line_area, 'metrics': metrics_area}
        results = run_simulation_live(
            run_id, Lx, Ly, Nx, Ny, epsilon, y0,
            M, dt, t_max, c_bulk, D,
            1, 96485, 8.314, 298, alpha, i0, c_ref,
            0.10787, 10500, beta, a_index, h,
            psi, AgNH3_conc, Cu_ion_conc, eta_chem,
            ratio_top_factor, ratio_surface_factor, ratio_decay_len,
            save_every, ui,
        )
        save_run(run_id, {}, results['x'], results['y'],
                 results['phi_hist'], results['c_hist'],
                 results['phi_l_hist'], results['t_hist'], results['psi'])
        st.session_state.run_id = run_id
        st.session_state.results = results
        status.success("Done & saved!")

if stop_col.button("Stop Simulation"):
    st.session_state.stop_sim = True

# Saved runs
if st.sidebar.button("Show saved run IDs"):
    ids = list_runs()
    if ids:
        st.sidebar.write("**Saved runs (first 8 chars):**")
        for i in ids: st.sidebar.code(i)
    else:
        st.sidebar.info("No runs yet.")
if st.sidebar.button("Reset DB (delete all)"):
    delete_all()
    st.session_state.run_id = None
    st.session_state.results = None
    st.success("Database cleared.")

# -------------------- Results --------------------
if st.session_state.results:
    r = st.session_state.results
    x, y = r['x'], r['y']
    phi_hist, c_hist = r['phi_hist'], r['c_hist']
    t_hist = r['t_hist']
    psi = r['psi']
    ratio_field = r.get('ratio_field', None)

    st.subheader("Results")
    time_idx = st.slider("Time step", 0, len(t_hist)-1, len(t_hist)//2,
                         format="t = %.2f s")
    var = st.selectbox("Variable", ["φ – Ag phase", "c – concentration",
                                   "ψ – template", "ratio field"])
    if var.startswith("φ"): data = phi_hist[time_idx]
    elif var.startswith("c"): data = c_hist[time_idx]
    elif var.startswith("ψ"): data = psi
    else: data = ratio_field

    fig = go.Figure(go.Contour(z=data.T, x=x, y=y,
                               contours_coloring='heatmap',
                               colorbar=dict(title=var)))
    fig.update_layout(height=460, margin=dict(l=10,r=10,t=30,b=40))
    st.plotly_chart(fig, use_container_width=True)

    mid = Nx//2
    fig2, ax = plt.subplots()
    ax.plot(y, data[mid, :])
    ax.set_xlabel('y (cm)'); ax.set_title(f'{var} at x=Lx/2, t={t_hist[time_idx]:.2f}s')
    ax.grid(True)
    st.pyplot(fig2); plt.close(fig2)

    # VTR export (selected timestep)
    def vtr_bytes(phi, c, psi, x, y):
        grid = pv.RectilinearGrid(x, y, [0])
        grid.point_data["phi"] = phi.T.ravel(order="F")
        grid.point_data["c"]   = c.T.ravel(order="F")
        grid.point_data["psi"] = psi.T.ravel(order="F")
        bio = BytesIO()
        grid.save(bio, file_format="vtr")
        return bio.getvalue()
    vtr = vtr_bytes(phi_hist[time_idx], c_hist[time_idx], psi, x, y)
    st.download_button(f"VTR t={t_hist[time_idx]:.2f}s", vtr,
                       f"ag_{t_hist[time_idx]:.2f}.vtr", "application/octet-stream")

    # ZIP of all VTRs
    if st.button("Download All VTRs (ZIP)"):
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, tt in enumerate(t_hist):
                grid = pv.RectilinearGrid(x, y, [0])
                grid.point_data["phi"] = phi_hist[i].T.ravel(order="F")
                grid.point_data["c"]   = c_hist[i].T.ravel(order="F")
                grid.point_data["psi"] = psi.T.ravel(order="F")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".vtr")
                grid.save(tmp.name)
                zf.write(tmp.name, f"ag_t{tt:.2f}.vtr")
                tmp.close(); Path(tmp.name).unlink()
        st.download_button("ZIP All VTRs", bio.getvalue(),
                           "ag_all_vtrs.zip", "application/zip")

# DB download
if DB_PATH.exists():
    st.download_button("Download DB", DB_PATH.read_bytes(),
                       "simulations.db", "application/octet-stream")
