import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io, zipfile, time, base64, tempfile, os
from matplotlib import cm
from numba import njit
import meshio

st.set_page_config(page_title="Electroless Ag Deposition Simulator", layout="wide")

st.title("🧪 Electroless Silver Shell Growth over Cu Core")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Controls")
Nx, Ny = st.sidebar.slider("Grid size", 64, 256, 128, step=32), st.sidebar.slider("Grid size (y)", 64, 256, 128, step=32)
dt = st.sidebar.number_input("Time step", 1e-5, 1e-2, 1e-3, format="%.5f")
steps = st.sidebar.number_input("Simulation steps", 10, 2000, 200, step=10)
A_phi = st.sidebar.number_input("Kinetic parameter Aφ", 0.1, 10.0, 1.0)
A_psi = st.sidebar.number_input("Kinetic parameter Aψ", 0.1, 10.0, 1.0)
D_phi = st.sidebar.number_input("Diffusion coefficient Dφ", 0.001, 1.0, 0.01)
D_psi = st.sidebar.number_input("Diffusion coefficient Dψ", 0.001, 1.0, 0.01)
semi_implicit = st.sidebar.checkbox("Use semi-implicit solver", value=False)
use_numba = st.sidebar.checkbox("Accelerate with Numba", value=True)
enable_3d = st.sidebar.checkbox("Enable 3D spherical mode", value=False)

# --- Colormap Selection ---
cmap_choice = st.sidebar.selectbox(
    "Colormap", sorted(m for m in plt.colormaps() if not m.endswith("_r")), index=sorted(plt.colormaps()).index("turbo")
)

# --- Animation Controls ---
st.sidebar.header("Animation Controls")
auto_play = st.sidebar.checkbox("Autoplay", value=False)
autoplay_interval = st.sidebar.slider("Frame interval (s)", 0.05, 1.0, 0.2)
frame_idx = st.sidebar.slider("Frame index", 0, 100, 0)

# --- Initialize Fields ---
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)
r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
phi = np.zeros((Nx, Ny))
psi = np.zeros((Nx, Ny))

phi[r < 0.2] = 0.0   # Ag shell
psi[r < 0.2] = 1.0   # Cu core

# --- Numerical Laplacian ---
@njit
def laplacian(f, dx):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4*f[1:-1, 1:-1]
    ) / (dx*dx)
    return lap

# --- Time Integration ---
@njit
def evolve(phi, psi, D_phi, D_psi, A_phi, A_psi, dt, dx):
    phi_new = phi + dt * (D_phi * laplacian(phi, dx) + A_phi * phi * (1 - phi) * (phi - 0.5 + psi))
    psi_new = psi + dt * (D_psi * laplacian(psi, dx) + A_psi * psi * (1 - psi) * (psi - 0.5 + phi))
    return phi_new, psi_new

# --- Simulation Loop ---
st.header("Simulation Progress")
progress_bar = st.progress(0)
snapshots = []
dx = 1.0 / Nx

for step in range(int(steps)):
    phi, psi = evolve(phi, psi, D_phi, D_psi, A_phi, A_psi, dt, dx)
    if step % 10 == 0:
        snapshots.append((phi.copy(), psi.copy()))
    progress_bar.progress((step + 1) / steps)
progress_bar.empty()

# --- Visualization ---
st.header("Results & Playback")
frame_idx = min(frame_idx, len(snapshots)-1)
phi_view, psi_view = snapshots[frame_idx]
t = frame_idx * dt * 10

st.subheader(f"Frame {frame_idx}/{len(snapshots)-1}  |  Time = {t:.5f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
im0 = axs[0].imshow(phi_view.T, origin="lower", cmap=cmap_choice)
axs[0].set_title("Ag shell (φ)")
plt.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(psi_view.T, origin="lower", cmap=cmap_choice)
axs[1].set_title("Cu core (ψ)")
plt.colorbar(im1, ax=axs[1])
st.pyplot(fig)

# --- Postprocessor: Combined Electric Potential ---
st.subheader("Electric Potential Post-Processor")
combine_formula = st.selectbox(
    "Select potential interpolation formula",
    ["h * (phi^2 + psi^2)", "h * (phi + psi)", "h * (phi^3 + psi^3)", "h * (phi^2 - psi^2)"]
)
h_value = st.number_input("h (scaling factor)", 0.0, 10.0, 1.0, 0.1)
show_combined = st.checkbox("Show combined potential field", value=True)

def compute_potential(phi, psi, formula, h):
    if "phi^2 + psi^2" in formula:
        return h * (phi**2 + psi**2)
    elif "phi + psi" in formula:
        return h * (phi + psi)
    elif "phi^3 + psi^3" in formula:
        return h * (phi**3 + psi**3)
    elif "phi^2 - psi^2" in formula:
        return h * (phi**2 - psi**2)
    else:
        return h * (phi**2 + psi**2)

# --- Flux Computation ---
st.subheader("Flux Computation")
flux_option = st.selectbox("Compute flux based on", ["∇φ", "∇ψ", "∇(φ+ψ)", "None"])
show_flux = flux_option != "None"

def compute_flux(field, dx):
    ux = np.zeros_like(field)
    uy = np.zeros_like(field)
    ux[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
    uy[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dx)
    return np.sqrt(ux**2 + uy**2)

if show_combined:
    potential = compute_potential(phi_view, psi_view, combine_formula, h_value)
    fig_pot, ax_pot = plt.subplots(figsize=(6, 5))
    im_pot = ax_pot.imshow(potential.T, origin='lower', cmap=cmap_choice)
    plt.colorbar(im_pot, ax=ax_pot, label="Potential")
    ax_pot.set_title(f"Combined potential — {combine_formula}")
    st.pyplot(fig_pot)

    if show_flux:
        if flux_option == "∇φ":
            flux = compute_flux(phi_view, dx)
        elif flux_option == "∇ψ":
            flux = compute_flux(psi_view, dx)
        elif flux_option == "∇(φ+ψ)":
            flux = compute_flux(phi_view + psi_view, dx)
        fig_flux, ax_flux = plt.subplots(figsize=(6, 5))
        im_flux = ax_flux.imshow(flux.T, origin='lower', cmap="inferno")
        plt.colorbar(im_flux, ax=ax_flux, label="|Flux|")
        ax_flux.set_title("Flux Magnitude")
        st.pyplot(fig_flux)

# --- Export Options ---
st.subheader("Export Results")
export_fmt = st.selectbox("Export format", ["vtu", "pvd", "zip", "csv", "png"])

def export_to_vtu(phi, psi, path):
    cells = [("quad", np.array([[i, i+1, i+Nx+1, i+Nx] for i in range((Nx-1)*(Ny-1))]))]
    mesh = meshio.Mesh(points=np.column_stack((X.flatten(), Y.flatten(), np.zeros(X.size))),
                       cells=cells,
                       point_data={"phi": phi.flatten(), "psi": psi.flatten()})
    mesh.write(path)

if st.button("Download"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{export_fmt}")
    if export_fmt == "vtu":
        export_to_vtu(phi_view, psi_view, tmp.name)
    elif export_fmt == "csv":
        np.savetxt(tmp.name, np.c_[X.flatten(), Y.flatten(), phi_view.flatten(), psi_view.flatten()],
                   delimiter=",", header="x,y,phi,psi", comments="")
    elif export_fmt == "png":
        fig.savefig(tmp.name)
    st.download_button("Download File", open(tmp.name, "rb"), file_name=os.path.basename(tmp.name))

# --- Autoplay Animation ---
if auto_play:
    st.markdown(f"""
    <script>
    let idx = 0;
    let maxFrames = {len(snapshots)};
    let interval = {autoplay_interval*1000};
    let playing = true;
    const btnPlay = document.createElement('button');
    btnPlay.innerText = '⏸ Pause';
    btnPlay.style = 'margin:4px;padding:4px 10px;';
    document.body.appendChild(btnPlay);
    btnPlay.onclick = function() {{
        playing = !playing;
        btnPlay.innerText = playing ? '⏸ Pause' : '▶ Resume';
    }}
    function step() {{
        if (!playing) return;
        idx = (idx + 1) % maxFrames;
        window.parent.postMessage({{type:'streamlit:setComponentValue', key:'frame_idx', value: idx}}, '*');
        setTimeout(step, interval);
    }}
    step();
    </script>
    """, unsafe_allow_html=True)
