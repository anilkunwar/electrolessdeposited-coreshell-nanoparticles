# --------------------------------------------------------------
#  ELECTROLESS Ag DEPOSITION – Streamlit Cloud (SQLite + cache)
# --------------------------------------------------------------
import streamlit as st
import numpy as np
from scipy.signal import convolve2d
import plotly.graph_objects as go
import pyvista as pv
from pathlib import Path
import shutil
import tempfile
import zipfile
import sqlite3
import pickle
import hashlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO

# ------------------- 1. SQLite helper -------------------------
DB_PATH = Path("simulations.db")


def _hash_params(**kwargs):
    """Deterministic hash of all simulation parameters – used as primary key."""
    s = "".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    params BLOB,
                    x BLOB, y BLOB,
                    phi_hist BLOB, c_hist BLOB, phi_l_hist BLOB,
                    time_hist BLOB, psi BLOB
               )"""
        )


def save_run(run_id, params, x, y, phi_hist, c_hist, phi_l_hist, time_hist, psi):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, params, x, y, phi_hist, c_hist, phi_l_hist, time_hist, psi)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                pickle.dumps(params),
                pickle.dumps(x),
                pickle.dumps(y),
                pickle.dumps(phi_hist),
                pickle.dumps(c_hist),
                pickle.dumps(phi_l_hist),
                pickle.dumps(time_hist),
                pickle.dumps(psi),
            ),
        )


def load_run(run_id):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "SELECT params,x,y,phi_hist,c_hist,phi_l_hist,time_hist,psi FROM runs WHERE run_id=?",
            (run_id,),
        )
        row = cur.fetchone()
        if row:
            return {k: pickle.loads(v) for k, v in zip(
                ["params","x","y","phi_hist","c_hist","phi_l_hist","time_hist","psi"], row)}
    return None


def delete_run(run_id):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM runs WHERE run_id=?", (run_id,))


def list_runs():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT run_id FROM runs")
        return [r[0] for r in cur.fetchall()]


# ------------------- 2. Cached simulation --------------------
@st.cache_data(show_spinner=False)
def _run_simulation_cached(_run_id, Lx, Ly, Nx, Ny, epsilon, y0, M, dt, t_max,
                          c_bulk, D, z, F, R, T, alpha, i0, c_ref,
                          M_Ag, rho_Ag, beta, a_index, h, psi,
                          AgNH3_conc, Cu_ion_conc, eta_chem):
    """The *real* heavy work – returns the same objects that the UI needs."""
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # ---- initialise fields -------------------------------------------------
    phi = (1 - psi) * 0.5 * (1 - np.tanh((Y - y0) / epsilon))
    ratio = AgNH3_conc / Cu_ion_conc if Cu_ion_conc > 0 else 1.0
    c = c_bulk * (Y / Ly) * (1 - phi) * (1 - psi) * ratio
    phi_l = np.zeros_like(Y)

    # ---- kernels -----------------------------------------------------------
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / (dx ** 2)
    grad_x_kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / (2 * dx)
    grad_y_kernel = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / (2 * dy)

    times_to_plot = np.arange(0, t_max + 1, 1.0)
    phi_hist, c_hist, phi_l_hist, t_hist = [], [], [], []

    # ---- time loop ---------------------------------------------------------
    for t in np.arange(0, t_max + dt, dt):
        phi_x = convolve2d(phi, grad_x_kernel, mode="same", boundary="symm")
        phi_y = convolve2d(phi, grad_y_kernel, mode="same", boundary="symm")
        grad_phi_mag = np.sqrt(phi_x ** 2 + phi_y ** 2)
        delta_int = 6 * phi * (1 - phi) * (1 - psi) * grad_phi_mag
        phi_xx = convolve2d(phi, laplacian_kernel, mode="same", boundary="symm")

        f_prime_electrodeposit = beta * 2 * phi * (1 - phi) * (1 - 2 * phi)
        f_prime_template = beta * 2 * (phi - h)
        f_prime_total = ((1 + a_index) / 8) * (1 - psi) * f_prime_electrodeposit + \
                        ((1 - a_index) / 8) * psi * f_prime_template

        mu = -epsilon ** 2 * phi_xx + f_prime_total - alpha * c
        mu_xx = convolve2d(mu, laplacian_kernel, mode="same", boundary="symm")

        # Butler-Volmer (cathodic only)
        c_mol_m3 = c * 1e6 * (1 - phi) * (1 - psi) * ratio
        i_loc = i0 * (c_mol_m3 / c_ref) * np.exp(0.5 * z * F * eta_chem / (R * T))
        i_loc = i_loc * delta_int

        u = -(i_loc / (z * F)) * (M_Ag / rho_Ag) * 1e-2
        advection = u * (1 - psi) * phi_y
        phi += dt * (M * mu_xx - advection)

        # concentration
        c_eff = (1 - phi) * (1 - psi) * c
        c_eff_xx = convolve2d(c_eff, laplacian_kernel, mode="same", boundary="symm")
        sink = -i_loc * delta_int / (z * F * 1e6)
        c_t = D * c_eff_xx + sink
        c += dt * c_t

        c[:, 0] = 0
        c[:, -1] = c_bulk * ratio

        if np.any(np.isclose(t, times_to_plot, atol=dt / 2)):
            phi_hist.append(phi.copy())
            c_hist.append(c.copy())
            phi_l_hist.append(phi_l.copy())
            t_hist.append(t)

    return x, y, phi_hist, c_hist, phi_l_hist, t_hist, psi


# ------------------- 3. Template geometry -------------------
def create_template(Lx, Ly, Nx, Ny, template_type,
                    radius, side_length, param1, param2, param_func):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    psi = np.zeros((Ny, Nx))

    if template_type == "Circle":
        dist = np.sqrt((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2)
        psi = np.where(dist <= radius, 1, 0)
    elif template_type == "Semicircle":
        dist = np.sqrt((X - Lx / 2) ** 2 + Y ** 2)
        psi = np.where((dist <= radius) & (Y >= 0), 1, 0)
    elif template_type == "Square":
        psi = np.where((np.abs(X - Lx / 2) <= side_length / 2) &
                       (Y >= 0) & (Y <= side_length), 1, 0)
    elif template_type == "Parametric":
        try:
            g = eval(param_func,
                     {"x": X - Lx / 2, "y": Y, "p1": param1, "p2": param2, "np": np})
            psi = np.where(g <= 0, 1, 0)
        except Exception as e:
            st.error(f"Parametric error: {e}")
    return psi


# ------------------- 4. UI ----------------------------------
st.title("Electroless Ag Deposition – Cloud-ready")
st.markdown("""
2-D phase-field simulation of Ag shell growth on a **fixed Cu core**.  
All heavy data are stored in an **SQLite** DB (`simulations.db`).  
Press **Reset** to wipe the DB and start fresh.
""")

# ---- sidebar ------------------------------------------------
st.sidebar.header("Simulation Parameters")
Lx = st.sidebar.slider("Domain width Lx (cm)", 1e-6, 1e-5, 5e-6, 1e-7)
Ly = st.sidebar.slider("Domain height Ly (cm)", 1e-6, 1e-5, 5e-6, 1e-7)
Nx = st.sidebar.slider("Grid points X", 50, 200, 100, 10)
Ny = st.sidebar.slider("Grid points Y", 50, 200, 100, 10)
epsilon = st.sidebar.slider("Interface width ε (cm)", 1e-8, 1e-7, 5e-8, 1e-8)
y0 = Lx / 2
M = st.sidebar.number_input("Mobility M (cm²/s)", 1e-6, 1e-4, 1e-5, 1e-6)
dt = st.sidebar.number_input("Δt (s)", 1e-6, 1e-4, 1e-5, 1e-6)
t_max = st.sidebar.number_input("t_max (s)", 1.0, 20.0, 10.0, 1.0)
c_bulk = st.sidebar.number_input("Bulk c (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6)
D = st.sidebar.number_input("Diffusion D (cm²/s)", 1e-6, 1e-5, 1e-5, 1e-6)
z, F, R, T = 1, 96485, 8.314, 298
alpha = st.sidebar.number_input("α", 0.0, 1.0, 0.1, 0.01)
i0 = st.sidebar.number_input("i₀ (A/m²)", 0.1, 10.0, 0.5, 0.1)
c_ref = st.sidebar.number_input("c_ref (mol/m³)", 100, 2000, 1000, 100)
M_Ag, rho_Ag = 0.10787, 10500
beta = st.sidebar.slider("β", 0.1, 10.0, 1.0, 0.1)
a_index = st.sidebar.slider("a-index", -1.0, 1.0, 0.0, 0.1)
h = st.sidebar.slider("hydrophobicity h", 0.0, 1.0, 0.5, 0.1)
AgNH3_conc = st.sidebar.number_input("[Ag(NH₃)₂]⁺ (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6)
Cu_ion_conc = st.sidebar.number_input("[Cu²⁺] (mol/cm³)", 1e-6, 5e-4, 5e-5, 1e-6)
eta_chem = st.sidebar.slider("η_chem (V)", 0.1, 0.5, 0.3, 0.05)

st.sidebar.header("Template Geometry")
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

# ---- initialise DB ------------------------------------------------
init_db()

# ---- session state ------------------------------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "results" not in st.session_state:
    st.session_state.results = None

# ---- run / reload -------------------------------------------------
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running …"):
        run_id = _hash_params(
            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, epsilon=epsilon, y0=y0, M=M, dt=dt,
            t_max=t_max, c_bulk=c_bulk, D=D, z=z, F=F, R=R, T=T,
            alpha=alpha, i0=i0, c_ref=c_ref, M_Ag=M_Ag, rho_Ag=rho_Ag,
            beta=beta, a_index=a_index, h=h,
            AgNH3_conc=AgNH3_conc, Cu_ion_conc=Cu_ion_conc, eta_chem=eta_chem,
            template_type=template_type, radius=radius,
            side_length=side_length, param1=param1, param2=param2,
            param_func=param_func)

        # 1. try to load from DB
        cached = load_run(run_id)
        if cached:
            x, y = cached["x"], cached["y"]
            phi_hist = cached["phi_hist"]
            c_hist = cached["c_hist"]
            phi_l_hist = cached["phi_l_hist"]
            t_hist = cached["time_hist"]
            psi = cached["psi"]
        else:
            # 2. heavy calculation (cached by Streamlit)
            x, y, phi_hist, c_hist, phi_l_hist, t_hist, psi = _run_simulation_cached(
                run_id, Lx, Ly, Nx, Ny, epsilon, y0, M, dt, t_max,
                c_bulk, D, z, F, R, T, alpha, i0, c_ref,
                M_Ag, rho_Ag, beta, a_index, h, psi,
                AgNH3_conc, Cu_ion_conc, eta_chem)

            # 3. store in DB for later sessions
            save_run(run_id,
                     {"Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny, "epsilon": epsilon,
                      "M": M, "dt": dt, "t_max": t_max, "c_bulk": c_bulk, "D": D,
                      "alpha": alpha, "i0": i0, "c_ref": c_ref,
                      "beta": beta, "a_index": a_index, "h": h,
                      "AgNH3_conc": AgNH3_conc, "Cu_ion_conc": Cu_ion_conc,
                      "eta_chem": eta_chem},
                     x, y, phi_hist, c_hist, phi_l_hist, t_hist, psi)

        st.session_state.run_id = run_id
        st.session_state.results = {
            "x": x, "y": y,
            "phi_hist": phi_hist, "c_hist": c_hist,
            "phi_l_hist": phi_l_hist, "t_hist": t_hist,
            "psi": psi
        }
    st.success("Done!")

# ---- optional: pick a previous run ---------------------------------
prev_runs = list_runs()
if prev_runs:
    chosen = st.sidebar.selectbox("Load previous run", ["(none)"] + prev_runs)
    if chosen != "(none)":
        data = load_run(chosen)
        if data:
            st.session_state.run_id = chosen
            st.session_state.results = {
                "x": data["x"], "y": data["y"],
                "phi_hist": data["phi_hist"], "c_hist": data["c_hist"],
                "phi_l_hist": data["phi_l_hist"], "t_hist": data["time_hist"],
                "psi": data["psi"]
            }
            st.success(f"Loaded run `{chosen[:8]}…`")

# ---- reset -------------------------------------------------------
if st.sidebar.button("Reset DB (delete all runs)"):
    DB_PATH.unlink(missing_ok=True)
    init_db()
    st.session_state.run_id = None
    st.session_state.results = None
    st.success("Database cleared")

# ---- visualisation ------------------------------------------------
if st.session_state.results:
    r = st.session_state.results
    x, y = r["x"], r["y"]
    phi_hist = r["phi_hist"]
    c_hist = r["c_hist"]
    phi_l_hist = r["phi_l_hist"]
    t_hist = r["t_hist"]
    psi = r["psi"]

    st.subheader("Results")
    time_idx = st.slider("Time step", 0, len(t_hist) - 1, 0, format="t = %.1f s")
    t = t_hist[time_idx]
    phi = phi_hist[time_idx]
    c = c_hist[time_idx]
    phi_l = phi_l_hist[time_idx]

    # ----- colour scheme -------------------------------------------------
    schemes = ['Viridis', 'Plasma', 'Magma', 'Inferno', 'Cividis', 'Jet',
               'Turbo', 'Rainbow', 'Blues', 'Greens', 'Reds']
    cs = st.selectbox("Plotly colour", schemes)

    # ----- contour plots -------------------------------------------------
    for arr, name in [(phi, "φ (Ag)"), (c, "c (mol/cm³)"),
                      (phi_l, "φₗ (V)"), (psi, "ψ (Cu core)")]:
        fig = go.Figure(data=go.Contour(z=arr, x=x, y=y, colorscale=cs))
        fig.update_layout(title=f"{name} @ t = {t:.1f}s",
                          xaxis_title="x (cm)", yaxis_title="y (cm)",
                          height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ----- line plots ----------------------------------------------------
    mid = Nx // 2
    for arr, label, col in [(phi[:, mid], "φ", "steelblue"),
                            (c[:, mid], "c", "crimson"),
                            (phi_l[:, mid], "φₗ", "seagreen")]:
        fig, ax = plt.subplots()
        ax.plot(y, arr, color=col, label=label)
        ax.set_xlabel("y (cm)"); ax.set_ylabel(label)
        ax.set_title(f"{label} @ x = Lx/2, t = {t:.1f}s")
        ax.legend(); ax.grid(True)
        st.pyplot(fig); plt.close(fig)

    # ----- VTR download (cached) ----------------------------------------
    @st.cache_data
    def _make_vtr(_t, _phi, _c, _phi_l, _psi, _x, _y):
        grid = pv.RectilinearGrid(_x, _y, [0])
        grid.point_data["phi"] = _phi.T.ravel(order="F")
        grid.point_data["c"] = _c.T.ravel(order="F")
        grid.point_data["phi_l"] = _phi_l.T.ravel(order="F")
        grid.point_data["psi"] = _psi.T.ravel(order="F")
        bio = BytesIO()
        grid.save(bio, file_format="vtr")
        return bio.getvalue()

    vtr_bytes = _make_vtr(t, phi, c, phi_l, psi, x, y)
    st.download_button(
        label=f"Download VTR (t = {t:.1f}s)",
        data=vtr_bytes,
        file_name=f"ag_depo_t{t:.1f}.vtr",
        mime="application/octet-stream",
        key=f"vtr_{time_idx}"
    )

    # ----- ZIP of **all** timesteps (cached) -----------------------------
    @st.cache_data
    def _make_zip(_phi_hist, _c_hist, _phi_l_hist, _t_hist, _psi, _x, _y):
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
            for i, tt in enumerate(_t_hist):
                grid = pv.RectilinearGrid(_x, _y, [0])
                grid.point_data["phi"] = _phi_hist[i].T.ravel(order="F")
                grid.point_data["c"] = _c_hist[i].T.ravel(order="F")
                grid.point_data["phi_l"] = _phi_l_hist[i].T.ravel(order="F")
                grid.point_data["psi"] = _psi.T.ravel(order="F")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".vtr")
                grid.save(tmp.name)
                z.write(tmp.name, f"ag_depo_t{tt:.1f}.vtr")
                tmp.close()
                Path(tmp.name).unlink()
        return bio.getvalue()

    if st.button("Download **all** VTR files (ZIP)"):
        zip_bytes = _make_zip(phi_hist, c_hist, phi_l_hist, t_hist, psi, x, y)
        st.download_button(
            label="Download ZIP",
            data=zip_bytes,
            file_name="ag_deposition_all.zip",
            mime="application/zip"
        )

# --------------------------------------------------------------
st.markdown("""
### What changed?
* **SQLite** (`simulations.db`) stores every finished run – survives app restarts.  
* **`run_id`** = deterministic hash of *all* input parameters → perfect cache key.  
* **Session state** holds only the *current* result; the DB holds the archive.  
* **VTR creation** is cached (`@st.cache_data`) → download never re-runs the simulation.  
* **Reset** button wipes the DB cleanly.
""")
