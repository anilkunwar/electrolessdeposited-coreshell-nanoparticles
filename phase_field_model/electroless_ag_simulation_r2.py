import numpy as np
import numba as nb
import sqlite3
import json
import hashlib
import time
from contextlib import contextmanager

# ============================================================
# 1. Simulation Core
# ============================================================

@nb.njit(fastmath=True, parallel=True)
def evolve(phi, c, dt, dx, D, k_dep, n_steps):
    """
    Simple explicit finite-difference evolution for electroless Ag deposition.
    """
    nx, ny = phi.shape
    phi_hist = np.zeros((n_steps, nx, ny))
    c_hist = np.zeros((n_steps, nx, ny))

    for step in range(n_steps):
        # Finite differences with Neumann boundary
        phi_xx = np.zeros_like(phi)
        phi_yy = np.zeros_like(phi)
        c_xx = np.zeros_like(c)
        c_yy = np.zeros_like(c)

        for i in range(nx):
            for j in range(ny):
                # Handle boundary via Neumann (zero-gradient)
                ip = min(i + 1, nx - 1)
                im = max(i - 1, 0)
                jp = min(j + 1, ny - 1)
                jm = max(j - 1, 0)

                phi_xx[i, j] = (phi[ip, j] - 2 * phi[i, j] + phi[im, j]) / (dx * dx)
                phi_yy[i, j] = (phi[i, jp] - 2 * phi[i, j] + phi[i, jm]) / (dx * dx)
                c_xx[i, j] = (c[ip, j] - 2 * c[i, j] + c[im, j]) / (dx * dx)
                c_yy[i, j] = (c[i, jp] - 2 * c[i, j] + c[i, jm]) / (dx * dx)

        # PDE update (simplified reaction-diffusion)
        dphi_dt = D * (phi_xx + phi_yy) + k_dep * c * (1.0 - phi)
        dc_dt = D * (c_xx + c_yy) - k_dep * c * (1.0 - phi)

        phi += dt * dphi_dt
        c += dt * dc_dt

        # Clamp physical bounds
        phi = np.clip(phi, 0.0, 1.0)
        c = np.clip(c, 0.0, 1.0)

        phi_hist[step] = phi
        c_hist[step] = c

    return phi_hist, c_hist


# ============================================================
# 2. Database Utilities
# ============================================================

def ensure_db_schema(con):
    """Ensure that the table schema exists."""
    con.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        params_json TEXT,
        x BLOB,
        y BLOB,
        phi_hist BLOB,
        c_hist BLOB,
        phi_l_hist BLOB,
        t_hist BLOB,
        psi BLOB
    )
    """)
    con.commit()


@contextmanager
def db_connection(db_path="runs.db"):
    """Context-managed SQLite connection for Streamlit-safe operations."""
    con = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    ensure_db_schema(con)
    try:
        yield con
    finally:
        con.close()


def make_run_id(params: dict) -> str:
    """Generate a deterministic hash ID from params."""
    json_bytes = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


def save_run(run_id, params, x, y, phi_hist, c_hist, phi_l_hist=None, t_hist=None, psi=None):
    """Insert or replace a simulation run into the SQLite DB."""
    with db_connection() as con:
        con.execute("""
        INSERT OR REPLACE INTO runs
        (run_id, params_json, x, y, phi_hist, c_hist, phi_l_hist, t_hist, psi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            json.dumps(params),
            x.tobytes(),
            y.tobytes(),
            phi_hist.tobytes(),
            c_hist.tobytes(),
            phi_l_hist.tobytes() if phi_l_hist is not None else None,
            t_hist.tobytes() if t_hist is not None else None,
            psi.tobytes() if psi is not None else None
        ))
        con.commit()


def load_run(run_id):
    """Load a simulation run by its ID."""
    if not run_id:
        raise ValueError("Empty run_id: no simulation ID provided.")

    with db_connection() as con:
        cur = con.execute("""
        SELECT params_json,x,y,phi_hist,c_hist,phi_l_hist,t_hist,psi
        FROM runs WHERE run_id=?
        """, (run_id,))
        row = cur.fetchone()

        if row is None:
            raise KeyError(f"No run found with run_id={run_id}")

        params = json.loads(row[0])
        x = np.frombuffer(row[1], dtype=np.float64)
        y = np.frombuffer(row[2], dtype=np.float64)
        phi_hist = np.frombuffer(row[3], dtype=np.float64)
        c_hist = np.frombuffer(row[4], dtype=np.float64)

        nx = int(params["nx"])
        ny = int(params["ny"])
        n_steps = int(params["n_steps"])
        phi_hist = phi_hist.reshape((n_steps, nx, ny))
        c_hist = c_hist.reshape((n_steps, nx, ny))

        return params, x, y, phi_hist, c_hist


# ============================================================
# 3. Simulation Runner
# ============================================================

def run_simulation(nx=64, ny=64, n_steps=100, dx=1.0, dt=0.01,
                   D=0.05, k_dep=1.0):
    """Run an electroless Ag deposition simulation."""
    x = np.linspace(0, dx * (nx - 1), nx)
    y = np.linspace(0, dx * (ny - 1), ny)
    phi = np.zeros((nx, ny))
    c = np.ones((nx, ny)) * 0.8

    # Seed initial Ag nuclei
    phi[nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2] = 1.0

    phi_hist, c_hist = evolve(phi, c, dt, dx, D, k_dep, n_steps)
    return x, y, phi_hist, c_hist


# ============================================================
# 4. Entry Point
# ============================================================

if __name__ == "__main__":
    params = {
        "nx": 64,
        "ny": 64,
        "n_steps": 200,
        "dx": 1.0,
        "dt": 0.01,
        "D": 0.05,
        "k_dep": 1.0
    }

    run_id = make_run_id(params)
    print(f"Running simulation ID: {run_id[:10]}...")

    # Try loading cached run
    try:
        params_loaded, x, y, phi_hist, c_hist = load_run(run_id)
        print("Loaded cached run from database.")
    except KeyError:
        print("No cached run found, running simulation...")
        x, y, phi_hist, c_hist = run_simulation(**params)
        save_run(run_id, params, x, y, phi_hist, c_hist)
        print("Simulation saved to database.")

    print("Done.")
