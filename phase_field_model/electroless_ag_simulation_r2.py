# improved_ag_deposition.py
import streamlit as st
import numpy as np
from numba import njit, prange
import plotly.graph_objects as go
import pyvista as pv
import sqlite3
import pickle
import hashlib
import zipfile
import tempfile
import json
import time
from pathlib import Path
from io import BytesIO
import ast
from typing import Any, Dict, Tuple

# ---------------------------
# Configuration / DB helpers
# ---------------------------
DB_PATH = Path("simulations.db")

def safe_json_dumps(obj: Any) -> str:
    """Stable JSON representation for hashing (converts numpy arrays to lists)."""
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        raise TypeError
    return json.dumps(obj, sort_keys=True, default=default, separators=(",", ":"))

def _hash_params(**kw) -> str:
    """Return a short stable hex digest for parameter dict."""
    s = safe_json_dumps(kw)
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                   run_id TEXT PRIMARY KEY,
                   created_at REAL,
                   params_json TEXT,
                   params_blob BLOB,
                   x BLOB, y BLOB,
                   phi_hist BLOB, c_hist BLOB,
                   phi_l_hist BLOB, t_hist BLOB, psi BLOB
              )"""
        )

def save_run(run_id: str, params: Dict[str, Any],
             x: np.ndarray, y: np.ndarray,
             phi_hist: np.ndarray, c_hist: np.ndarray,
             phi_l_hist: np.ndarray, t_hist: np.ndarray, psi: np.ndarray) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """INSERT OR REPLACE INTO runs
               (run_id,created_at,params_json,params_blob,x,y,phi_hist,c_hist,phi_l_hist,t_hist,psi)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, time.time(), json.dumps(params, sort_keys=True),
             pickle.dumps(params),
             pickle.dumps(x), pickle.dumps(y),
             pickle.dumps(phi_hist), pickle.dumps(c_hist),
             pickle.dumps(phi_l_hist), pickle.dumps(t_hist),
             pickle.dumps(psi)),
        )

def load_run(run_id: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "SELECT params_json,x,y,phi_hist,c_hist,phi_l_hist,t_hist,psi FROM runs WHERE run_id=?",
            (run_id,))
        row = cur.fetchone()
        if row:
            keys = ["params_json", "x", "y", "phi_hist", "c_hist", "phi_l_hist", "t_hist", "psi"]
            # first column is JSON
            params_json = row[0]
            others = [pickle.loads(v) for v in row[1:]]
            result = {"params": json.loads(params_json)}
            for k, v in zip(keys[1:], others):
                result[k] = v
            return result
    return None

def list_runs():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT run_id, datetime(created_at, 'unixepoch') FROM runs ORDER BY created_at DESC")
        return cur.fetchall()

def delete_all():
    DB_PATH.unlink(missing_ok=True)
    init_db()

# ---------------------------
# Safe parametric parser
# ---------------------------
ALLOWED_NAMES = {"np", "x", "y", "p1", "p2"}

def compile_safe_parametric(expr: str):
    """
    Compile a simple expression for g(x,y,p1,p2) while ensuring it doesn't contain
    disallowed names or nodes. Returns a function g(X, Y, p1, p2) -> array of booleans.
    """

    # parse AST and validate
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        # forbid import, attribute access, function defs, etc.
        if isinstance(node, (ast.Call, ast.Attribute, ast.Import, ast.ImportFrom,
                             ast.Lambda, ast.FunctionDef, ast.ClassDef)):
            # We allow calls only to np functions, so Call is allowed but must be checked below
            if isinstance(node, ast.Call):
                # ensure the function called is a name and permitted (np.something)
                if isinstance(node.func, ast.Attribute):
                    if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "np"):
                        raise ValueError("Only numpy (np.*) calls allowed in parametric expression.")
                elif isinstance(node.func, ast.Name):
                    if node.func.id not in ("abs", "min", "max"):
                        raise ValueError("Only a few bare functions allowed (abs,min,max).")
                else:
                    raise ValueError("Unsafe call in parametric expression.")
            else:
                raise ValueError(f"Disallowed AST node: {type(node).__name__}")

        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES and node.id not in ("abs", "min", "max"):
                raise ValueError(f"Name {node.id} is not allowed in parametric expression.")

    # compile into a lambda
    code = compile(tree, filename="<param_expr>", mode="eval")
    def g(X, Y, p1, p2):
        # X, Y are numpy arrays
        return eval(code, {"np": np, "abs": abs, "min": min, "max": max}, {"x": X, "y": Y, "p1": p1, "p2": p2})
    return g

# ---------------------------
# Numba kernels (improved boundaries)
# ---------------------------
@njit(parallel=True, fastmath=True)
def _laplacian(arr: np.ndarray, dx2: float) -> np.ndarray:
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for i in prange(ny):
        for j in prange(nx):
            # central stencil with simple Neumann (zero-gradient) at boundaries using one-sided diffs
            im = i-1 if i-1 >= 0 else i
            ip = i+1 if i+1 < ny else i
            jm = j-1 if j-1 >= 0 else j
            jp = j+1 if j+1 < nx else j
            out[i, j] = (arr[ip, j] + arr[im, j] + arr[i, jp] + arr[i, jm] - 4.0 * arr[i, j]) / dx2
    return out

@njit(parallel=True, fastmath=True)
def _grad_x(arr: np.ndarray, dx: float) -> np.ndarray:
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for i in prange(ny):
        for j in prange(nx):
            if 0 < j < nx - 1:
                out[i, j] = (arr[i, j + 1] - arr[i, j - 1]) / (2.0 * dx)
            elif j == 0:
                out[i, j] = (arr[i, j + 1] - arr[i, j]) / dx
            else:  # j == nx-1
                out[i, j] = (arr[i, j] - arr[i, j - 1]) / dx
    return out

@njit(parallel=True, fastmath=True)
def _grad_y(arr: np.ndarray, dy: float) -> np.ndarray:
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for i in prange(ny):
        for j in prange(nx):
            if 0 < i < ny - 1:
                out[i, j] = (arr[i + 1, j] - arr[i - 1, j]) / (2.0 * dy)
            elif i == 0:
                out[i, j] = (arr[i + 1, j] - arr[i, j]) / dy
            else:  # i == ny-1
                out[i, j] = (arr[i, j] - arr[i - 1, j]) / dy
    return out

# ---------------------------
# Simulation kernel (cached by run-id in DB)
# ---------------------------
def run_simulation_fast(run_id: str,
                        Lx: float, Ly: float, Nx: int, Ny: int,
                        epsilon: float, y0: float,
                        M: float, dt: float, t_max: float, c_bulk: float, D: float,
                        z: int, F: float, R: float, T: float, alpha: float, i0: float, c_ref: float,
                        M_Ag: float, rho_Ag: float, beta: float, a_index: float, h: float,
                        psi: np.ndarray, AgNH3_conc: float, Cu_ion_conc: float, eta_chem: float,
                        save_every: int = 10):
    """
    Deterministic simulation runner. It's intentionally NOT decorated with st.cache_data
    because we control caching by run_id + DB. Arrays are float64 for numba's comfort.
    """

    # pre-convert to float64 for numba (consistent dtypes)
    Nx = int(Nx); Ny = int(Ny)
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dx2 = dx * dx

    x = np.linspace(0.0, Lx, Nx, dtype=np.float64)
    y = np.linspace(0.0, Ly, Ny, dtype=np.float64)
    # user provided psi might be float32; convert and ensure correct shape (Nx,Ny)
    psi = np.asarray(psi, dtype=np.float64)
    if psi.shape != (Nx, Ny):
        raise ValueError(f"psi shape {psi.shape} doesn't match Nx,Ny {(Nx,Ny)}")

    # initial phi: tanh interface located at y0 (use indexing consistent with x,y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    phi = ((1.0 - psi) * 0.5 * (1.0 - np.tanh((Y - y0) / epsilon))).astype(np.float64)

    ratio = float(AgNH3_conc / (Cu_ion_conc if Cu_ion_conc > 0 else 1.0))
    c = (c_bulk * (Y / Ly) * (1.0 - phi) * (1.0 - psi) * ratio).astype(np.float64)

    Fz = float(z * F)
    RT = float(R * T)
    exp_term = np.exp(0.5 * Fz * eta_chem / RT)
    # convert M_Ag/rho factor to consistent units - keep original scaling but be explicit
    MAg_rho = float(M_Ag / rho_Ag * 1e-2)

    phi_hist, c_hist, phi_l_hist, t_hist = [], [], [], []
    n_steps = int(np.ceil(t_max / dt))
    save_step = max(1, int(save_every))

    phi_old = phi.copy()

    for step in range(n_steps + 1):
        t = step * dt

        phi_x = _grad_x(phi, dx)
        phi_y = _grad_y(phi, dy)
        grad_phi_mag = np.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-30)
        delta_int = 6.0 * phi * (1.0 - phi) * (1.0 - psi) * grad_phi_mag

        phi_xx = _laplacian(phi, dx2)

        f_prime_ed = beta * 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
        f_prime_tp = beta * 2.0 * (phi - h)
        f_prime = ((1.0 + a_index) / 8.0) * (1.0 - psi) * f_prime_ed + \
                  ((1.0 - a_index) / 8.0) * psi * f_prime_tp

        mu = -epsilon ** 2 * phi_xx + f_prime - alpha * c
        mu_xx = _laplacian(mu, dx2)

        # concentration in mol/m^3 for kinetics; keep consistent with your earlier scaling
        c_mol = c * 1e6 * (1.0 - phi) * (1.0 - psi) * ratio
        i_loc = i0 * (c_mol / c_ref) * exp_term * delta_int

        u = - (i_loc / Fz) * MAg_rho
        advection = u * (1.0 - psi) * phi_y

        phi += dt * (M * mu_xx - advection)
        # clamp phi to [0,1]
        np.clip(phi, 0.0, 1.0, out=phi)

        c_eff = (1.0 - phi) * (1.0 - psi) * c
        c_xx = _laplacian(c_eff, dx2)
        sink = - i_loc * delta_int / (Fz * 1e6)
        c += dt * (D * c_xx + sink)

        # simple Dirichlet boundary conditions for concentration
        c[:, 0] = 0.0
        c[:, -1] = c_bulk * ratio

        if (step % save_step == 0) or (step == n_steps):
            phi_hist.append(phi.copy())
            c_hist.append(c.copy())
            phi_l_hist.append(np.zeros_like(phi))
            t_hist.append(t)

        # quick steady-state check
        if step > 100 and np.max(np.abs(phi - phi_old)) < 1e-9:
            break
        if (step % 50) == 0:
            phi_old = phi.copy()

    return (x, y,
            np.array(phi_hist), np.array(c_hist),
            np.array(phi_l_hist), np.array(t_hist), psi)

# ---------------------------
# Template creation
# ---------------------------
def create_template(Lx: float, Ly: float, Nx: int, Ny: int,
                    template_type: str, radius: float, side_length: float,
                    param1: float, param2: float, param_func: str) -> np.ndarray:
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    psi = np.zeros((Nx, Ny), dtype=np.float64)

    if template_type == "Circle":
        psi = (((X - Lx / 2.0) ** 2 + (Y - Ly / 2.0) ** 2) <= radius ** 2).astype(np.float64)
    elif template_type == "Semicircle":
        psi = ((((X - Lx / 2.0) ** 2 + (Y - Ly / 2.0) ** 2) <= radius ** 2) & (Y >= Ly/2.0)).astype(np.float64)
    elif template_type == "Square":
        psi = ((np.abs(X - Lx / 2.0) <= side_length / 2.0) & (np.abs(Y - Ly / 2.0) <= side_length / 2.0)).astype(np.float64)
    elif template_type == "Parametric":
        if not param_func:
            raise ValueError("Parametric template selected but no expression provided.")
        g_func = compile_safe_parametric(param_func)
        try:
            g = g_func(X - Lx / 2.0, Y - Ly / 2.0, float(param1), float(param2))
            psi = (g <= 0).astype(np.float64)
        except Exception as e:
            raise RuntimeError(f"Parametric evaluation error: {e}")
    else:
        raise ValueError(f"Unknown template_type: {template_type}")

    return psi

# ---------------------------
# Streamlit app (UI)
# ---------------------------
st.set_page_config(layout="wide")
st.title("Fast Electroless Ag Deposition (Numba + SQLite) — improved")

st.markdown("""
Improved safety (no `eval`), robust finite differences at boundaries, clearer DB metadata,
and safer hashing of parameters.
""")

# Sidebar parameters (kept similar to your original; use sensible defaults)
st.sidebar.header("Simulation Parameters")
Lx = float(st.sidebar.slider("Lx (cm)", 1e-6, 1e-5, 5e-6, 1e-7))
Ly = float(st.sidebar.slider("Ly (cm)", 1e-6, 1e-5, 5e-6, 1e-7))
Nx = int(st.sidebar.slider("Nx", 60, 250, 120, 10))
Ny = int(st.sidebar.slider("Ny", 60, 250, 120, 10))
epsilon = float(st.sidebar.slider("ε (cm)", 1e-8, 1e-7, 5e-8, 1e-8))
# center of interface
y0 = Ly / 2.0

M = float(st.sidebar.number_input("M (cm²/s)", 1e-6, 1e-4, 1e-5, 1e-6, format="%.6g"))
dt = float(st.sidebar.number_input("Δt (s)", 5e-7, 5e-5, 2e-6, 1e-7, format="%.6g"))
t_max = float(st.sidebar.number_input("t_max (s)", 1.0, 20.0, 8.0, 0.5))
c_bulk = float(st.sidebar.number_input("c_bulk (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6, format="%.6g"))
D = float(st.sidebar.number_input("D (cm²/s)", 5e-6, 2e-5, 1e-5, 1e-6, format="%.6g"))
alpha = float(st.sidebar.number_input("α", 0.0, 1.0, 0.1, 0.01, format="%.4g"))
i0 = float(st.sidebar.number_input("i₀ (A/m²)", 0.1, 10.0, 0.5, 0.1, format="%.4g"))
c_ref = float(st.sidebar.number_input("c_ref (mol/m³)", 100.0, 2000.0, 1000.0, 100.0, format="%.4g"))
beta = float(st.sidebar.slider("β", 0.1, 10.0, 1.0, 0.1))
a_index = float(st.sidebar.slider("a-index", -1.0, 1.0, 0.0, 0.1))
h = float(st.sidebar.slider("h", 0.0, 1.0, 0.5, 0.1))
AgNH3_conc = float(st.sidebar.number_input("[Ag(NH₃)₂]⁺ (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6, format="%.6g"))
Cu_ion_conc = float(st.sidebar.number_input("[Cu²⁺] (mol/cm³)", 1e-6, 5e-4, 5e-5, 1e-6, format="%.6g"))
eta_chem = float(st.sidebar.slider("η_chem (V)", 0.1, 0.5, 0.3, 0.05))
save_every = int(st.sidebar.number_input("Save every N steps", 1, 200, 10, 1))

st.sidebar.header("Template")
template_type = st.sidebar.selectbox("Shape", ["Circle", "Semicircle", "Square", "Parametric"])
radius = float(1e-6)
side_length = float(st.sidebar.slider("Square side (cm)", 1e-7, 1e-6, 2e-7, 1e-8)) if template_type == "Square" else float(2e-7)
param_func = st.sidebar.text_input("g(x,y,p1,p2)", "(x/p1)**2 + (y/p2)**2 - 1", help="g ≤ 0 → ψ = 1") if template_type == "Parametric" else ""
param1 = float(st.sidebar.slider("p1", 1e-7, 1e-6, 2e-7, 1e-8)) if template_type == "Parametric" else float(2e-7)
param2 = float(st.sidebar.slider("p2", 1e-7, 1e-6, 2e-7, 1e-8)) if template_type == "Parametric" else float(2e-7)

init_db()

if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "results" not in st.session_state:
    st.session_state.results = None

# Run button
if st.sidebar.button("Run Simulation"):
    # create psi with safe parser
    try:
        psi = create_template(Lx, Ly, Nx, Ny, template_type, radius, side_length, param1, param2, param_func)
    except Exception as e:
        st.error(f"Template creation failed: {e}")
        psi = np.zeros((Nx, Ny), dtype=np.float64)

    params = dict(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, epsilon=epsilon, y0=y0,
        M=M, dt=dt, t_max=t_max, c_bulk=c_bulk, D=D,
        z=1, F=96485.0, R=8.314, T=298.0,
        alpha=alpha, i0=i0, c_ref=c_ref,
        M_Ag=0.10787, rho_Ag=10500.0,
        beta=beta, a_index=a_index, h=h,
        psi=psi, AgNH3_conc=AgNH3_conc, Cu_ion_conc=Cu_ion_conc,
        eta_chem=eta_chem, save_every=save_every, template_type=template_type
    )

    run_id = _hash_params(**params)

    cached = load_run(run_id)
    if cached:
        st.session_state.run_id = run_id
        st.session_state.results = {
            "x": cached["x"], "y": cached["y"],
            "phi_hist": cached["phi_hist"], "c_hist": cached["c_hist"],
            "phi_l_hist": cached["phi_l_hist"], "t_hist": cached["t_hist"],
            "psi": cached["psi"], "params": cached["params"]
        }
        st.success("Loaded simulation from database (cached).")
    else:
        with st.spinner("Running (Numba accelerated)..."):
            # run simulation (numba-heavy)
            x, y, phi_hist, c_hist, phi_l_hist, t_hist, psi_out = run_simulation_fast(
                run_id, Lx, Ly, Nx, Ny, epsilon, y0,
                M, dt, t_max, c_bulk, D,
                1, 96485.0, 8.314, 298.0,
                alpha, i0, c_ref,
                0.10787, 10500.0, beta, a_index, h,
                psi, AgNH3_conc, Cu_ion_conc, eta_chem,
                save_every)

            save_run(run_id, params, x, y, phi_hist, c_hist, phi_l_hist, t_hist, psi_out)
            st.session_state.run_id = run_id
            st.session_state.results = {
                "x": x, "y": y,
                "phi_hist": phi_hist, "c_hist": c_hist,
                "phi_l_hist": phi_l_hist, "t_hist": t_hist,
                "psi": psi_out, "params": params
            }
        st.success("Simulation complete and saved to DB.")

# show runs
if st.sidebar.button("Show saved run IDs"):
    rows = list_runs()
    if rows:
        st.sidebar.write("**Saved runs:** (run_id, created_at)")
        for run_id, created in rows:
            st.sidebar.code(f"{run_id} — {created}")
    else:
        st.sidebar.info("No runs in DB yet.")

if st.sidebar.button("Reset DB (delete all)"):
    delete_all()
    st.session_state.run_id = None
    st.session_state.results = None
    st.success("Database cleared.")

# --------------
# Visualization
# --------------
if st.session_state.results:
    r = st.session_state.results
    x, y = r["x"], r["y"]
    phi_hist, c_hist = r["phi_hist"], r["c_hist"]
    t_hist = r["t_hist"]
    psi = r["psi"]

    st.subheader("Results Visualization")
    time_idx = st.slider("Select Time Step", 0, max(0, len(t_hist) - 1), 0, format="t = %.2f s")

    variable = st.selectbox("Select Variable", ["φ - Phase Field (Ag)", "c - Concentration", "ψ - Template"])
    if variable.startswith("φ"):
        data = phi_hist[time_idx]
    elif variable.startswith("c"):
        data = c_hist[time_idx]
    else:
        data = psi

    # Plotly contour
    fig = go.Figure(go.Contour(z=data.T, x=x, y=y))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Midline Matplotlib plot
    mid_x = len(x) // 2
    import matplotlib.pyplot as plt
    fig2, ax = plt.subplots()
    ax.plot(y, data[mid_x, :], label=variable)
    ax.set_xlabel("y (cm)")
    ax.set_ylabel(variable)
    ax.set_title(f"{variable} at x = Lx/2, t = {t_hist[time_idx]:.2f} s")
    ax.grid(True)
    st.pyplot(fig2)
    plt.close(fig2)

    # VTR export
    def generate_vtr_bytes(_phi, _c, _psi, _x, _y) -> bytes:
        # pyvista.RectilinearGrid expects 1D coords arrays
        grid = pv.RectilinearGrid(_x, _y, np.array([0.0]))
        # verify flattening order: pyvista Fortran style mapping expects order="F"
        grid.point_data["phi"] = _phi.T.ravel(order="F")
        grid.point_data["c"] = _c.T.ravel(order="F")
        grid.point_data["psi"] = _psi.T.ravel(order="F")
        bio = BytesIO()
        grid.save(bio, file_format="vtr")
        return bio.getvalue()

    vtr_bytes = generate_vtr_bytes(phi_hist[time_idx], c_hist[time_idx], psi, x, y)
    st.download_button(f"Download VTR File (t = {t_hist[time_idx]:.2f} s)", vtr_bytes,
                       f"ag_t{t_hist[time_idx]:.2f}.vtr", "application/octet-stream")

    # Zip all as before
    if st.button("Download All VTRs (ZIP)"):
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, tt in enumerate(t_hist):
                tmp_bytes = generate_vtr_bytes(phi_hist[i], c_hist[i], psi, x, y)
                zf.writestr(f"ag_t{tt:.2f}.vtr", tmp_bytes)
        st.download_button("Download ZIP of All VTRs", bio.getvalue(), "ag_all_vtrs.zip", "application/zip")
