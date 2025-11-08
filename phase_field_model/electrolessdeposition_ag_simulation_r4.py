import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
import os
from numba import njit
from st_js_eval import streamlit_js_eval  # for autoplay JS events
import tempfile
import meshio

st.set_page_config(page_title="Electroless Ag Deposition Simulator", layout="wide")

st.title("🧪 Electroless Ag Deposition with Cu Core")
st.markdown("""
**Visualization of ψ (Cu core), ϕ (Ag shell), and c (concentration) evolution with post-processing electric potential.**  
This model mimics silver shell growth over copper via electroless deposition.
""")

# -----------------------------------------------------------------
# Sidebar parameters
# -----------------------------------------------------------------
L = st.sidebar.slider("Domain length (L, m)", 1e-6, 1e-5, 5e-6, 1e-7, format="%e")
Nx = st.sidebar.slider("Grid Nx", 50, 300, 100, 10)
Ny = st.sidebar.slider("Grid Ny", 50, 300, 100, 10)
core_radius_frac = st.sidebar.slider("Core radius / L", 0.1, 0.7, 0.4, 0.05)
shell_thickness_frac = st.sidebar.slider("Shell thickness / core radius", 0.05, 0.5, 0.25, 0.05)
eps_tilde = st.sidebar.slider("Interface width (ε*)", 0.005, 0.05, 0.02, 0.005)
t_steps = st.sidebar.slider("Number of time steps", 5, 200, 50)
growth_rate = st.sidebar.slider("Growth rate", 0.0, 0.05, 0.01, 0.005)
cmap_name = st.sidebar.selectbox("Matplotlib colormap", plt.colormaps(), index=plt.colormaps().index("turbo"))

# -----------------------------------------------------------------
# Create domain and initialize fields
# -----------------------------------------------------------------
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
dx = L / Nx
dy = L / Ny
X, Y = np.meshgrid(x, y, indexing="ij")
dist = np.sqrt((X - L/2)**2 + (Y - L/2)**2)

r_core = core_radius_frac * L
r_inner = r_core
r_outer = r_core * (1.0 + shell_thickness_frac)

psi = (dist <= r_core).astype(float)
phi = np.where(dist <= r_inner, 0.0, np.where(dist <= r_outer, 1.0, 0.0)).astype(float)
phi = phi * (1.0 - 0.5*(1.0 - np.tanh((dist - r_inner)/eps_tilde))) * \
           (1.0 - 0.5*(1.0 + np.tanh((dist - r_outer)/eps_tilde)))
phi = np.clip(phi, 0.0, 1.0)
c = (Y/L) * (1.0 - phi) * (1.0 - psi)

# -----------------------------------------------------------------
# Numba optimized deposition update
# -----------------------------------------------------------------
@njit
def update_fields(phi, psi, c, growth_rate, t_steps):
    Nx, Ny = phi.shape
    phi_hist = np.zeros((t_steps, Nx, Ny))
    c_hist = np.zeros((t_steps, Nx, Ny))
    phi_hist[0] = phi
    c_hist[0] = c

    for t in range(1, t_steps):
        for i in range(Nx):
            for j in range(Ny):
                if 0.01 < phi[i, j] < 0.99:
                    phi[i, j] += growth_rate * (1 - psi[i, j])
        phi = np.clip(phi, 0.0, 1.0)
        c = c * (1.0 - 0.05 * phi)
        phi_hist[t] = phi
        c_hist[t] = c
    return phi_hist, c_hist

phi_hist, c_hist = update_fields(phi.copy(), psi.copy(), c.copy(), growth_rate, t_steps)

# -----------------------------------------------------------------
# Electric potential and post-processing
# -----------------------------------------------------------------
interp_fn = st.sidebar.selectbox("Post-processing function", [
    "h = (ϕ² + ψ²)",
    "h = ϕ² + 2ψ²",
    "h = ϕ² − ψ²",
    "h = sqrt(ϕ² + ψ²)"
])
frame = st.sidebar.slider("Frame", 0, t_steps - 1, 0)
play = st.sidebar.checkbox("▶️ Autoplay animation")

phi_t = phi_hist[frame]
c_t = c_hist[frame]

# Compute potential field h(ϕ, ψ)
if "sqrt" in interp_fn:
    potential = np.sqrt(phi_t**2 + psi**2)
elif "−" in interp_fn:
    potential = phi_t**2 - psi**2
elif "2ψ²" in interp_fn:
    potential = phi_t**2 + 2*psi**2
else:
    potential = phi_t**2 + psi**2

# -----------------------------------------------------------------
# Flux computation (∇c)
# -----------------------------------------------------------------
dcx, dcy = np.gradient(c_t, dx, dy)
flux = np.sqrt(dcx**2 + dcy**2)

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    field_choice = st.selectbox("Field to visualize", ["ϕ (Ag shell)", "ψ (Cu core)", "c (concentration)", "Electric potential", "Flux magnitude"])
    if field_choice.startswith("ϕ"):
        field = phi_t
    elif field_choice.startswith("ψ"):
        field = psi
    elif "concentration" in field_choice:
        field = c_t
    elif "Electric" in field_choice:
        field = potential
    else:
        field = flux

    fig = go.Figure(go.Contour(
        z=field.T, x=x/L, y=y/L, colorscale=cmap_name,
        contours_coloring="heatmap", colorbar=dict(title=field_choice)
    ))
    fig.update_layout(height=600, title=f"{field_choice} (step {frame})", xaxis_title="x/L", yaxis_title="y/L")
    st.plotly_chart(fig, use_container_width=True)

    if play:
        streamlit_js_eval(js_expressions="parent.document.querySelector('input[type=range]').value = (parseInt(parent.document.querySelector('input[type=range]').value) + 1) % parent.document.querySelector('input[type=range]').max;", key="animate")

with col2:
    mid = Nx // 2
    fig2, ax = plt.subplots()
    ax.plot(y/L, field[mid, :], label=field_choice, color="k")
    ax.set_xlabel("y/L"); ax.set_ylabel(field_choice); ax.legend(); ax.grid(True)
    st.pyplot(fig2)

# -----------------------------------------------------------------
# Export section
# -----------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Export Data")

# Create temp directory
tmpdir = tempfile.mkdtemp()

# Save selected field to CSV
csv_path = os.path.join(tmpdir, "field.csv")
np.savetxt(csv_path, field, delimiter=",")
# Save snapshot as PNG
png_path = os.path.join(tmpdir, "snapshot.png")
plt.imsave(png_path, field, cmap=cmap_name)

# Save VTU (meshio)
points = np.column_stack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
cells = [("quad", np.array([[i + j*Nx, i+1 + j*Nx, i+1 + (j+1)*Nx, i + (j+1)*Nx] for i in range(Nx-1) for j in range(Ny-1)]))]
mesh = meshio.Mesh(points, cells, point_data={"field": field.flatten()})
vtu_path = os.path.join(tmpdir, "snapshot.vtu")
meshio.write(vtu_path, mesh)

# Zip all
zip_path = os.path.join(tmpdir, "exports.zip")
with zipfile.ZipFile(zip_path, "w") as zf:
    zf.write(csv_path, "field.csv")
    zf.write(png_path, "snapshot.png")
    zf.write(vtu_path, "snapshot.vtu")

with open(zip_path, "rb") as f:
    st.sidebar.download_button("⬇️ Download Results (ZIP)", data=f, file_name="Ag_deposition_results.zip")

st.sidebar.success("Ready for export!")
