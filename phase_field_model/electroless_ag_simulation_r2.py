import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------------------------------------
# utility functions
# ---------------------------------------------
def laplacian(u, dx):
    ny, nx = u.shape
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4*u[1:-1,1:-1])
    out[0,1:-1]   = (u[1,1:-1]   + u[0,2:]   + u[0,:-2]   - 3*u[0,1:-1])
    out[-1,1:-1]  = (u[-2,1:-1]  + u[-1,2:]  + u[-1,:-2]  - 3*u[-1,1:-1])
    out[1:-1,0]   = (u[2:,0]     + u[:-2,0]  + u[1:-1,1]  - 3*u[1:-1,0])
    out[1:-1,-1]  = (u[2:,-1]    + u[:-2,-1] + u[1:-1,-2] - 3*u[1:-1,-1])
    return out / (dx * dx)

def grad_mag(u, dx):
    uy = np.zeros_like(u); ux = np.zeros_like(u)
    ux[:,1:-1] = (u[:,2:] - u[:,:-2]) / (2*dx)
    ux[:,0]    = (u[:,1]  - u[:,0])   / dx
    ux[:,-1]   = (u[:,-1] - u[:,-2])  / dx
    uy[1:-1,:] = (u[2:,:] - u[:-2,:]) / (2*dx)
    uy[0,:]    = (u[1,:]  - u[0,:])   / dx
    uy[-1,:]   = (u[-1,:] - u[-2,:])  / dx
    return np.sqrt(ux*ux + uy*uy + 1e-30)

# ---------------------------------------------
# electroless deposition simulation
# ---------------------------------------------
def simulate_electroless(
    Nx=128, Ny=128, dt=1e-4, n_steps=3000, save_every=200,
    gamma=0.02, beta=4.0, k0=0.4, M=0.2, alpha=2.0, c_bulk=2.0, D=0.05
):
    L = 1.0
    dx = L / (Nx - 1)
    eps = 4.0 * dx
    rho_m = 1.0

    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Core and initial shell
    cx, cy = 0.5, 0.5
    r_core = 0.18 * L
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    psi = (dist <= r_core).astype(np.float64)
    phi = np.zeros_like(psi)
    shell_mask = (dist > r_core) & (dist < r_core * 1.05)
    phi[shell_mask] = 0.6
    phi = np.clip(phi, 0.0, 1.0)

    c = c_bulk * (Y / L) * (1.0 - phi) * (1.0 - psi)
    c = np.clip(c, 0.0, c_bulk)

    snapshots = []
    diagnostics = []

    for step in range(n_steps + 1):
        t = step * dt
        gphi = grad_mag(phi, dx)
        delta_int = 6.0 * phi * (1.0 - phi) * (1.0 - psi) * gphi
        delta_int = np.clip(delta_int, 0.0, 6.0 / max(eps, dx))

        phi[0,:] = phi[1,:]; phi[-1,:] = phi[-2,:]; phi[:,0] = phi[:,1]; phi[:,-1] = phi[:,-2]
        lap_phi = laplacian(phi, dx)
        f_bulk = 2.0 * beta * phi * (1 - phi) * (1 - 2 * phi)
        c_mol = c * (1.0 - phi) * (1.0 - psi)
        i_loc = k0 * c_mol * delta_int
        i_loc = np.clip(i_loc, 0.0, 1e2)

        deposition = M * rho_m * i_loc
        curvature = M * gamma * lap_phi

        phi += dt * (deposition + curvature - M * f_bulk)
        phi = np.clip(phi, 0.0, 1.0)

        lap_c = laplacian(c, dx)
        sink = i_loc
        c += dt * (D * lap_c - sink)
        c = np.clip(c, 0.0, c_bulk * 4.0)
        c[:, -1] = c_bulk
        c[:, 0] = c[:, 1]
        c[0, :] = c[1, :]
        c[-1,:] = c[-2,:]

        # diagnostics
        bulk_norm = np.sqrt(np.mean(f_bulk**2))
        grad_term_raw = M * gamma * lap_phi
        grad_norm_raw = np.sqrt(np.mean(grad_term_raw**2))
        grad_phys = grad_term_raw * (dx*dx)
        grad_norm_phys = np.sqrt(np.mean(grad_phys**2))
        alpha_c_norm = alpha * np.mean(c)

        if step % save_every == 0 or step == n_steps:
            snapshots.append((t, phi.copy(), c.copy()))
            diagnostics.append((t, bulk_norm, grad_norm_raw, grad_norm_phys, alpha_c_norm))
    return snapshots, diagnostics

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="Electroless Ag Shell Growth", layout="wide")
st.title("🧪 Electroless Deposition: Silver Shell Growth Simulation")

st.sidebar.header("Simulation Controls")
gamma = st.sidebar.slider("γ (curvature strength)", 0.001, 0.05, 0.02, 0.001)
beta = st.sidebar.slider("β (double-well strength)", 1.0, 10.0, 4.0, 0.1)
k0 = st.sidebar.slider("k₀ (reaction prefactor)", 0.1, 1.0, 0.4, 0.05)
M = st.sidebar.slider("M (mobility)", 0.05, 0.5, 0.2, 0.05)
alpha = st.sidebar.slider("α (coupling)", 0.5, 5.0, 2.0, 0.1)
c_bulk = st.sidebar.slider("c_bulk (top concentration)", 0.5, 5.0, 2.0, 0.1)
n_steps = st.sidebar.slider("Simulation Steps", 1000, 8000, 4000, 500)

if st.button("▶ Run Simulation"):
    with st.spinner("Running simulation..."):
        snapshots, diagnostics = simulate_electroless(
            gamma=gamma, beta=beta, k0=k0, M=M, alpha=alpha,
            c_bulk=c_bulk, n_steps=n_steps
        )

    st.success("Simulation completed!")

    # Visualization section
    t_start, phi_start, c_start = snapshots[0]
    t_mid, phi_mid, c_mid = snapshots[len(snapshots)//2]
    t_end, phi_end, c_end = snapshots[-1]

    st.subheader("Evolution Snapshots (ϕ: shell phase, c: concentration)")

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    for axrow, (t, ph, cc) in zip(axes, [snapshots[0], snapshots[len(snapshots)//2], snapshots[-1]]):
        im0 = axrow[0].imshow(ph.T, origin="lower", extent=[0,1,0,1])
        axrow[0].set_title(f"ϕ (shell) at t*={t:.4f}")
        plt.colorbar(im0, ax=axrow[0])
        im1 = axrow[1].imshow(cc.T, origin="lower", extent=[0,1,0,1], cmap=cm.viridis)
        axrow[1].set_title(f"c (concentration) at t*={t:.4f}")
        plt.colorbar(im1, ax=axrow[1])
    st.pyplot(fig)

    # Diagnostic plots
    st.subheader("Diagnostic Norms")
    times = [r[0] for r in diagnostics]
    bulk = [r[1] for r in diagnostics]
    grad_raw = [r[2] for r in diagnostics]
    grad_phys = [r[3] for r in diagnostics]
    alpha_c = [r[4] for r in diagnostics]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.semilogy(times, np.maximum(bulk,1e-30), label='bulk ||·||₂')
    ax2.semilogy(times, np.maximum(grad_raw,1e-30), label='grad (raw) ||·||₂')
    ax2.semilogy(times, np.maximum(grad_phys,1e-30), label='grad (scaled dx²) ||·||₂')
    ax2.semilogy(times, np.maximum(alpha_c,1e-30), label='α·c (mean)')
    ax2.set_xlabel('t*'); ax2.set_ylabel('norm (log)')
    ax2.legend(); ax2.grid(True)
    st.pyplot(fig2)
