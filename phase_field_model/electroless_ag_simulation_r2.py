# --------------------------------------------------------------
#  ELECTROLESS Ag DEPOSITION – FASTER + CLOUD-READY (IMPROVED)
#  Adds: live on-the-fly plotting, progress + stop button, realtime metrics,
#        and downloadable results (single VTR + ZIP of all timesteps).
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


# -------------------- 3. Live simulation wrapper --------------------
# This function runs the simulation in Python (Numba kernels used for heavy ops)
# and updates Streamlit placeholders so the user sees live progress.
def run_simulation_live(
    run_id,
    Lx, Ly, Nx, Ny, epsilon, y0,
    M, dt, t_max, c_bulk, D,
    z, F, R, T, alpha, i0, c_ref,
    M_Ag, rho_Ag, beta, a_index, h,
    psi, AgNH3_conc, Cu_ion_conc, eta_chem,
    save_every,
    ui,
):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dx2 = dx * dx
    x = np.linspace(0, Lx, Nx, dtype=np.float32)
    y = np.linspace(0, Ly, Ny, dtype=np.float32)

    # ---- initialise (float32) ----
    phi = ((1 - psi) * 0.5 * (1 - np.tanh((np.meshgrid(x, y, indexing='ij')[1] - y0) / epsilon))).astype(np.float32)
    ratio = np.float32(AgNH3_conc / Cu_ion_conc if Cu_ion_conc > 0 else 1.0)
    c = (c_bulk * (np.meshgrid(x, y, indexing='ij')[1] / Ly) * (1 - phi) * (1 - psi) * ratio).astype(np.float32)

    # pre-compute constants
    Fz = np.float32(z * F)
    RT = np.float32(R * T)
    exp_term = np.exp(0.5 * Fz * eta_chem / RT)
    MAg_rho = np.float32(M_Ag / rho_Ag * 1e-2)

    # storage
    phi_hist, c_hist, phi_l_hist, t_hist = [], [], [], []
    n_steps = int(np.ceil(t_max / dt))
    save_step = max(1, save_every)

    phi_old = phi.copy()

    # progress UI
    progress_bar = ui['progress']
    status = ui['status']
    plot_area = ui['plot']
    line_area = ui['line']
    metrics_area = ui['metrics']

    for step in range(n_steps + 1):
        t = step * dt

        # compute gradients & laplacians using numba functions
        phi_x = _grad_x(phi, dx)
        phi_y = _grad_y(phi, dy)
        grad_phi_mag = np.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-30)
        delta_int = 6 * phi * (1 - phi) * (1 - psi) * grad_phi_mag

        phi_xx = _laplacian(phi, dx2)

        f_prime_ed = beta * 2 * phi * (1 - phi) * (1 - 2 * phi)
        f_prime_tp = beta * 2 * (phi - h)
        f_prime = ((1 + a_index) / 8) * (1 - psi) * f_prime_ed + \
                  ((1 - a_index) / 8) * psi * f_prime_tp

        mu = -epsilon ** 2 * phi_xx + f_prime - alpha * c
        mu_xx = _laplacian(mu, dx2)

        c_mol = c * 1e6 * (1 - phi) * (1 - psi) * ratio
        i_loc = i0 * (c_mol / c_ref) * exp_term * delta_int

        u = - (i_loc / Fz) * MAg_rho
        advection = u * (1 - psi) * phi_y

        phi += dt * (M * mu_xx - advection)

        c_eff = (1 - phi) * (1 - psi) * c
        c_xx = _laplacian(c_eff, dx2)
        sink = - i_loc * delta_int / (Fz * 1e6)
        c += dt * (D * c_xx + sink)

        # BCs
        c[:, 0] = 0.0
        c[:, -1] = c_bulk * ratio

        # store
        if step % save_step == 0 or step == n_steps:
            phi_hist.append(phi.copy())
            c_hist.append(c.copy())
            phi_l_hist.append(np.zeros_like(phi))
            t_hist.append(t)

            # update UI every saved step
            try:
                # Contour
                data = phi_hist[-1]
                fig = go.Figure(go.Contour(z=data.T, x=x, y=y))
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=20))
                plot_area.plotly_chart(fig, use_container_width=True)

                # Line at mid-x
                mid_x = len(x) // 2
                fig2, ax = plt.subplots()
                ax.plot(y, data[mid_x, :])
                ax.set_xlabel('y (cm)')
                ax.set_ylabel('phi')
                ax.set_title(f'phi at x=Lx/2, t={t:.3f}s')
                ax.grid(True)
                line_area.pyplot(fig2)
                plt.close(fig2)

                # metrics
                max_phi = float(np.max(phi))
                deposited = float(np.mean(phi))
                metrics_area.metric('t (s)', f"{t:.3f}")
                metrics_area.metric('max(phi)', f"{max_phi:.4f}")
                metrics_area.metric('mean(phi)', f"{deposited:.4f}")
            except Exception:
                # UI can fail in some headless environments; ignore and continue
                pass

        # early exit
        if step > 100 and np.max(np.abs(phi - phi_old)) < 1e-6:
            status.info(f"Converged at step {step}, t={t:.4f}s")
            break
        if step % 50 == 0:
            phi_old = phi.copy()

        # progress and stop handling
        progress_bar.progress(min(1.0, step / n_steps))
        if st.session_state.get('stop_sim', False):
            status.warning('Stop requested – terminating simulation...')
            break

    progress_bar.empty()

    results = {
        'x': x, 'y': y,
        'phi_hist': np.array(phi_hist), 'c_hist': np.array(c_hist),
        'phi_l_hist': np.array(phi_l_hist), 't_hist': np.array(t_hist),
        'psi': psi,
    }
    return results


# -------------------- 4. Template geometry --------------------
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


# -------------------- 5. UI ------------------------------------
st.title("Fast Electroless Ag Deposition – Live UI")
st.markdown("""
**2-D phase-field** – live display while simulating. Use the Stop button to interrupt.
All runs are stored in `simulations.db`. Press **Reset DB** to wipe everything.
""")

# ---------- sidebar ----------
st.sidebar.header("Simulation Parameters")
Lx = st.sidebar.slider("Lx (cm)", 1e-6, 1e-5, 5e-6, 1e-7)
Ly = st.sidebar.slider("Ly (cm)", 1e-6, 1e-5, 5e-6, 1e-7)
Nx = st.sidebar.slider("Nx", 60, 250, 100, 10)
Ny = st.sidebar.slider("Ny", 60, 250, 100, 10)
epsilon = st.sidebar.slider("ε (cm)", 1e-8, 1e-7, 5e-8, 1e-8)
y0 = Lx / 2
M = st.sidebar.number_input("M (cm²/s)", 1e-6, 1e-4, 1e-5, 1e-6)
dt = st.sidebar.number_input("Δt (s)", 5e-7, 5e-5, 5e-6, 1e-7)
t_max = st.sidebar.number_input("t_max (s)", 1.0, 20.0, 5.0, 0.5)
c_bulk = st.sidebar.number_input("c_bulk (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6)
D = st.sidebar.number_input("D (cm²/s)", 5e-6, 2e-5, 1e-5, 1e-6)
z, F, R, T = 1, 96485, 8.314, 298
alpha = st.sidebar.number_input("α", 0.0, 1.0, 0.1, 0.01)
i0 = st.sidebar.number_input("i₀ (A/m²)", 0.1, 10.0, 0.5, 0.1)
c_ref = st.sidebar.number_input("c_ref (mol/m³)", 100, 2000, 1000, 100)
M_Ag, rho_Ag = 0.10787, 10500
beta = st.sidebar.slider("β", 0.1, 10.0, 1.0, 0.1)
a_index = st.sidebar.slider("a-index", -1.0, 1.0, 0.0, 0.1)
h = st.sidebar.slider("h", 0.0, 1.0, 0.5, 0.1)
AgNH3_conc = st.sidebar.number_input("[Ag(NH₃)₂]⁺ (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6)
Cu_ion_conc = st.sidebar.number_input("[Cu²⁺] (mol/cm³)", 1e-6, 5e-4, 5e-5, 1e-6)
eta_chem = st.sidebar.slider("η_chem (V)", 0.1, 0.5, 0.3, 0.05)

save_every = st.sidebar.number_input("Save every N steps", 5, 50, 20, 1)

st.sidebar.header("Template")
template_type = st.sidebar.selectbox("Shape", ["Circle", "Semicircle", "Square", "Parametric"]) 
radius = 1e-6
side_length = st.sidebar.slider("Square side (cm)", 1e-7, 1e-6, 2e-7, 1e-8) if template_type == "Square" else 2e-7
param_func = st.sidebar.text_input(
    "g(x,y,p1,p2)", "(x/p1)**2 + (y/p2)**2 - 1",
    help="g ≤ 0 → ψ = 1") if template_type == "Parametric" else ""
param1 = st.sidebar.slider("p1", 1e-7, 1e-6, 2e-7, 1e-8) if template_type == "Parametric" else 2e-7
param2 = st.sidebar.slider("p2", 1e-7, 1e-6, 2e-7, 1e-8) if template_type == "Parametric" else 2e-7

psi = create_template(Lx, Ly, Nx, Ny, template_type,
                      radius, side_length, param1, param2, param_func)

# ---------- DB ----------
init_db()

if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "results" not in st.session_state:
    st.session_state.results = None
if "stop_sim" not in st.session_state:
    st.session_state.stop_sim = False

# UI placeholders for live updates
col1, col2 = st.columns([3, 1])
plot_area = col1.container()
line_area = col1.container()
metrics_area = col2.container()
status = st.empty()
progress = st.empty()

# Run / Stop controls
run_col, stop_col = st.columns(2)
if run_col.button("Run Simulation"):
    # reset stop flag
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
        param_func=param_func)

    cached = load_run(run_id)
    if cached:
        st.session_state.run_id = run_id
        st.session_state.results = cached
        status.success("Loaded cached run")
    else:
        status.info("Starting simulation – JITting numba kernels if needed...")
        ui = {'progress': progress, 'status': status, 'plot': plot_area, 'line': line_area, 'metrics': metrics_area}
        results = run_simulation_live(
            run_id, Lx, Ly, Nx, Ny, epsilon, y0,
            M, dt, t_max, c_bulk, D,
            1, 96485, 8.314, 298, alpha, i0, c_ref,
            0.10787, 10500, beta, a_index, h,
            psi, AgNH3_conc, Cu_ion_conc, eta_chem,
            save_every,
            ui,
        )
        save_run(run_id, {}, results['x'], results['y'], results['phi_hist'], results['c_hist'], results['phi_l_hist'], results['t_hist'], results['psi'])
        st.session_state.run_id = run_id
        st.session_state.results = results
        status.success("Simulation complete and saved!")

if stop_col.button("Stop Simulation"):
    st.session_state.stop_sim = True

# Saved runs
if st.sidebar.button("Show saved run IDs"):
    ids = list_runs()
    if ids:
        st.sidebar.write("**Saved runs (first 8 chars):**")
        for i in ids:
            st.sidebar.code(i)
    else:
        st.sidebar.info("No runs in DB yet.")

if st.sidebar.button("Reset DB (delete all)"):
    delete_all()
    st.session_state.run_id = None
    st.session_state.results = None
    st.success("Database cleared.")

# Results display + downloads
if st.session_state.results:
    r = st.session_state.results
    x, y = r['x'], r['y']
    phi_hist, c_hist = r['phi_hist'], r['c_hist']
    t_hist = r['t_hist']
    psi = r['psi']

    st.subheader("Results Visualization")
    time_idx = st.slider("Select Time Step", 0, len(t_hist) - 1, 0, format="t = %.2f s")
    variable = st.selectbox("Select Variable", ["φ - Phase Field (Ag)", "c - Concentration", "ψ - Template"]) 
    if variable.startswith("φ"):
        data = phi_hist[time_idx]
    elif variable.startswith("c"):
        data = c_hist[time_idx]
    else:
        data = psi

    fig = go.Figure(go.Contour(z=data.T, x=x, y=y))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)

    mid_x = len(x) // 2
    fig2, ax = plt.subplots()
    ax.plot(y, data[mid_x, :])
    ax.set_xlabel("y (cm)")
    ax.set_ylabel(variable)
    ax.set_title(f"{variable} at x = Lx/2, t = {t_hist[time_idx]:.2f} s")
    ax.grid(True)
    st.pyplot(fig2)
    plt.close(fig2)

    # VTR export for selected time step
    def generate_vtr_bytes(_phi, _c, _psi, _x, _y):
        grid = pv.RectilinearGrid(_x, _y, [0])
        grid.point_data["phi"] = _phi.T.ravel(order="F")
        grid.point_data["c"] = _c.T.ravel(order="F")
        grid.point_data["psi"] = _psi.T.ravel(order="F")
        bio = BytesIO()
        grid.save(bio, file_format="vtr")
        return bio.getvalue()

    vtr_bytes = generate_vtr_bytes(phi_hist[time_idx], c_hist[time_idx], psi, x, y)
    st.download_button(f"Download VTR File (t = {t_hist[time_idx]:.2f} s)", vtr_bytes, f"ag_t{t_hist[time_idx]:.2f}.vtr", "application/octet-stream")

    # Zip all VTRs download
    if st.button("Download All VTRs (ZIP)"):
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, tt in enumerate(t_hist):
                grid = pv.RectilinearGrid(x, y, [0])
                grid.point_data["phi"] = phi_hist[i].T.ravel(order="F")
                grid.point_data["c"] = c_hist[i].T.ravel(order="F")
                grid.point_data["psi"] = psi.T.ravel(order="F")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".vtr")
                grid.save(tmp.name)
                zf.write(tmp.name, f"ag_t{tt:.2f}.vtr")
                tmp.close()
                Path(tmp.name).unlink()
        zip_bytes = bio.getvalue()
        st.download_button("Download ZIP of All VTRs", zip_bytes, "ag_all_vtrs.zip", "application/zip")

# End of script
