import streamlit as st
import numpy as np
from scipy.signal import convolve2d
import plotly.graph_objects as go
import pyvista as pv
from pathlib import Path
import shutil
import tempfile
import zipfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO

# Cached simulation function
@st.cache_data
def run_simulation(Lx, Ly, Nx, Ny, epsilon, r0, M, dt, t_max, c_bulk, D, z, F, R, T, alpha, i0, c_ref, M_Ag, rho_Ag, beta, a_index, h, psi, AgNH3_conc, Cu_ion_conc, eta_chem):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize phi (Ag shell = 1, electrolyte = 0) as thin nanofilm around Cu core
    R = np.sqrt((X - Lx/2)**2 + (Y - Ly/2)**2)
    phi = (1 - psi) * 0.5 * (1 - np.tanh((R - r0) / epsilon))  # r0 = core radius + initial shell thickness
    ratio = AgNH3_conc / Cu_ion_conc if Cu_ion_conc > 0 else 1.0  # Avoid div by zero
    c = c_bulk * (1 - phi) * (1 - psi) * ratio  # Concentration in electrolyte
    c[psi == 1] = 0  # c = 0 at Cu core
    c[phi > 0.5] = 0  # c = 0 in Ag shell
    phi_l = np.zeros_like(Y)  # No potential in electroless
    
    # Kernels for derivatives
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / (dx**2)
    grad_x_kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / (2 * dx)
    grad_y_kernel = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / (2 * dy)

    times_to_plot = np.arange(0, t_max + 1, 1.0)
    phi_history = []
    c_history = []
    phi_l_history = []
    time_history = []

    # Time evolution loop
    for t in np.arange(0, t_max + dt, dt):
        phi_x = convolve2d(phi, grad_x_kernel, mode='same', boundary='symm')
        phi_y = convolve2d(phi, grad_y_kernel, mode='same', boundary='symm')
        grad_phi_mag = np.sqrt(phi_x**2 + phi_y**2)
        delta_int = 6 * phi * (1 - phi) * (1 - psi) * grad_phi_mag
        phi_xx = convolve2d(phi, laplacian_kernel, mode='same', boundary='symm')
        
        # Free energy derivatives
        f_prime_electrodeposit = beta * 2 * phi * (1 - phi) * (1 - 2 * phi)
        f_prime_template = beta * 2 * (phi - h)
        f_prime_total = ((1 + a_index) / 8) * (1 - psi) * f_prime_electrodeposit + ((1 - a_index) / 8) * psi * f_prime_template
        
        mu = -epsilon**2 * phi_xx + f_prime_total - alpha * c
        mu_xx = convolve2d(mu, laplacian_kernel, mode='same', boundary='symm')
        
        # Butler-Volmer for electroless (cathodic only)
        eta = eta_chem
        c_mol_m3 = c * 1e6 * (1 - phi) * (1 - psi) * ratio  # mol/m³, scaled by ratio
        i_loc = i0 * (c_mol_m3 / c_ref) * np.exp(0.5 * z * F * eta / (R * T))
        i_loc = i_loc * delta_int
        
        # Velocity for Ag deposition
        u = -(i_loc / (z * F)) * (M_Ag / rho_Ag) * 1e-2  # cm to m
        advection = u * (1 - psi) * phi_y
        phi += dt * (M * mu_xx - advection)
        
        # Concentration update (no migration)
        c_eff = (1 - phi) * (1 - psi) * c
        c_eff_xx = convolve2d(c_eff, laplacian_kernel, mode='same', boundary='symm')
        sink = -i_loc * delta_int / (z * F * 1e6)
        c_t = D * c_eff_xx + sink
        c += dt * c_t
        
        # Enforce boundary conditions
        c[psi == 1] = 0  # c = 0 at Cu core
        c[phi > 0.5] = 0  # c = 0 in Ag shell
        c[:, 0] = c_bulk * ratio  # Left boundary
        c[:, -1] = c_bulk * ratio  # Right boundary
        c[0, :] = c_bulk * ratio  # Bottom boundary
        c[-1, :] = c_bulk * ratio  # Top boundary

        if np.any(np.isclose(t, times_to_plot, atol=dt/2)):
            phi_history.append(phi.copy())
            c_history.append(c.copy())
            phi_l_history.append(phi_l.copy())
            time_history.append(t)

    return x, y, phi_history, c_history, phi_l_history, time_history, psi

# Template geometry function
def create_template(Lx, Ly, Nx, Ny, template_type, radius, side_length, param1, param2, param_func):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    psi = np.zeros((Ny, Nx))
    
    if template_type == "Circle":
        center = (Lx / 2, Ly / 2)
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        psi = np.where(dist <= radius, 1, 0)
    elif template_type == "Semicircle":
        center = (Lx / 2, 0)
        dist = np.sqrt((X - center[0])**2 + Y**2)
        psi = np.where((dist <= radius) & (Y >= 0), 1, 0)
    elif template_type == "Square":
        center = (Lx / 2, 0)
        psi = np.where((np.abs(X - center[0]) <= side_length / 2) & (Y >= 0) & (Y <= side_length), 1, 0)
    elif template_type == "Parametric":
        try:
            g = eval(param_func, {"x": X - Lx / 2, "y": Y - Ly / 2, "p1": param1, "p2": param2, "np": np})
            psi = np.where(g <= 0, 1, 0)
        except Exception as e:
            st.error(f"Invalid parametric function: {e}")
            psi = np.zeros((Ny, Nx))
    
    return psi

# Streamlit app configuration
st.title("Electroless Ag Deposition on 2 nm Cu Core Phase Field Simulation")
st.markdown("""
    This app simulates 2D electroless silver deposition on a fixed 2 nm diameter Cu core template (ψ = 1) immiscible with Ag shell (φ = 1) and electrolyte (φ = 0). The Ag ion concentration (c) is zero at the Cu core (ψ = 1) and Ag shell (φ > 0.5), and maximum at the far-field boundaries (all four sides). Vary c_bulk to assess shell thickness. Features include adjustable parameters and VTR downloads.
""")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")
Lx = st.sidebar.slider("Domain Width (Lx, cm)", 5e-7, 5e-6, 1e-6, 1e-7)
Ly = st.sidebar.slider("Domain Height (Ly, cm)", 5e-7, 5e-6, 1e-6, 1e-7)
Nx = st.sidebar.slider("Grid Points X (Nx)", 100, 300, 150, 10)
Ny = st.sidebar.slider("Grid Points Y (Ny)", 100, 300, 150, 10)
epsilon = st.sidebar.slider("Interface Width (epsilon, cm)", 1e-8, 5e-8, 2e-8, 1e-9)
r0 = 1.5e-7  # Cu core radius (1e-7 cm) + initial Ag shell thickness (~0.5 nm)
M = st.sidebar.number_input("Mobility (M, cm²/s)", 1e-6, 1e-4, 1e-5, 1e-6)
dt = st.sidebar.number_input("Time Step (dt, s)", 1e-7, 1e-5, 1e-6, 1e-7)
t_max = st.sidebar.number_input("Total Time (t_max, s)", 1.0, 20.0, 5.0, 1.0)
c_bulk = st.sidebar.number_input("Bulk Ag Concentration (c_bulk, mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6)
D = st.sidebar.number_input("Diffusion Coefficient (D, cm²/s)", 1e-6, 1e-5, 1e-5, 1e-6)
z = 1  # Ag⁺
F = 96485
R = 8.314
T = 298
alpha = st.sidebar.number_input("Coupling Constant (alpha)", 0.0, 1.0, 0.1, 0.01)
i0 = st.sidebar.number_input("Exchange Current Density (i0, A/m²)", 0.1, 10.0, 0.5, 0.1)
c_ref = st.sidebar.number_input("Reference Concentration (c_ref, mol/m³)", 100, 2000, 1000, 100)
M_Ag = 0.10787  # kg/mol
rho_Ag = 10500  # kg/m³
beta = st.sidebar.slider("Free Energy Scale (beta)", 0.1, 10.0, 1.0, 0.1)
a_index = st.sidebar.slider("Free Energy Weight (a_index)", -1.0, 1.0, 0.0, 0.1)
h = st.sidebar.slider("Hydrophobicity Index (h)", 0.0, 1.0, 0.5, 0.1)
AgNH3_conc = st.sidebar.number_input("AgNH3 Concentration (mol/cm³)", 1e-6, 1e-4, 5e-5, 1e-6)
Cu_ion_conc = st.sidebar.number_input("Cu²⁺ Concentration (mol/cm³)", 1e-6, 5e-4, 5e-5, 1e-6)
eta_chem = st.sidebar.slider("Chemical Overpotential (eta_chem, V)", 0.1, 0.5, 0.3, 0.05)

# Template geometry selection
st.sidebar.header("Template Geometry")
template_type = st.sidebar.selectbox("Template Type", ["Circle", "Semicircle", "Square", "Parametric"])
radius = 1e-7  # 2 nm diameter Cu core = 1 nm radius
side_length = st.sidebar.slider("Square Side Length (cm)", 1e-7, 1e-6, 2e-7, 1e-8) if template_type == "Square" else 2e-7
param_func = st.sidebar.text_input("Parametric Function g(x, y, p1, p2)", "(x/p1)**2 + (y/p2)**2 - 1", help="Define g(x, y; p1, p2) ≤ 0 for ψ = 1") if template_type == "Parametric" else ""
param1 = st.sidebar.slider("Parametric p1", 1e-7, 1e-6, 2e-7, 1e-8) if template_type == "Parametric" else 2e-7
param2 = st.sidebar.slider("Parametric p2", 1e-7, 1e-6, 2e-7, 1e-8) if template_type == "Parametric" else 2e-7

# Create template
psi = create_template(Lx, Ly, Nx, Ny, template_type, radius, side_length, param1, param2, param_func)

# Output directory
output_dir = Path("electrodeposition_outputs")
output_dir.mkdir(exist_ok=True)

# Session state
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None

# Run simulation
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        x, y, phi_history, c_history, phi_l_history, time_history, psi = run_simulation(
            Lx, Ly, Nx, Ny, epsilon, r0, M, dt, t_max, c_bulk, D, z, F, R, T, alpha, i0, c_ref, M_Ag, rho_Ag, beta, a_index, h, psi, AgNH3_conc, Cu_ion_conc, eta_chem
        )
        st.session_state.simulation_results = {
            "x": x, "y": y, "phi_history": phi_history, "c_history": c_history,
            "phi_l_history": phi_l_history, "time_history": time_history, "psi": psi
        }
    st.success("Simulation complete!")

# Display results
if st.session_state.simulation_results:
    results = st.session_state.simulation_results
    x, y, phi_history, c_history, phi_l_history, time_history, psi = (
        results["x"], results["y"], results["phi_history"], results["c_history"],
        results["phi_l_history"], results["time_history"], results["psi"]
    )

    # Time slider
    st.subheader("Simulation Results")
    time_index = st.slider("Select Time Step", 0, len(time_history) - 1, 0, format="t=%.1f s")
    t = time_history[time_index]
    phi = phi_history[time_index]
    c = c_history[time_index]
    phi_l = phi_l_history[time_index]

    # Calculate shell thickness
    mid_x = Nx // 2
    phi_mid = phi[:, mid_x]
    y_shell = y[phi_mid > 0.5]
    if len(y_shell) > 0:
        shell_thickness = (max(y_shell) - min(y_shell)) / 2  # Approximate thickness in cm
        st.write(f"**Estimated Ag Shell Thickness at t = {t:.1f} s**: {shell_thickness*1e7:.2f} nm")
    else:
        st.write("**Estimated Ag Shell Thickness**: Not detectable (φ < 0.5)")

    # Color scheme
    color_schemes = [
        'Viridis', 'Plasma', 'Magma', 'Inferno', 'Cividis', 'Jet', 'Rainbow',
        'Turbo', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds'
    ]
    selected_scheme = st.selectbox("Select Plotly Color Scheme", color_schemes)

    # Contour plots
    st.write("**Plotly Contour Plots**")
    for data, title in [
        (phi, "Phase Field (φ)"), (c, "Concentration (c, mol/cm³)"), (phi_l, "Potential (φ_l, V)"), (psi, "Template (ψ)")
    ]:
        fig = go.Figure(data=go.Contour(z=data, x=x, y=y, colorscale=selected_scheme))
        fig.update_layout(title=f"{title} at t = {t:.1f} s", xaxis_title="x (cm)", yaxis_title="y (cm)", height=400)
        st.plotly_chart(fig)

    # Matplotlib line plots at x = Lx/2
    st.write("**Matplotlib Line Plots at x = Lx/2**")
    mid_x = Nx // 2
    for data, label, color in [
        (phi[:, mid_x], "φ (Phase Field)", "blue"),
        (c[:, mid_x], "c (Concentration, mol/cm³)", "red"),
        (phi_l[:, mid_x], "φ_l (Potential, V)", "green")
    ]:
        fig, ax = plt.subplots()
        ax.plot(y, data, color=color, label=label)
        ax.set_xlabel("y (cm)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} at x = Lx/2, t = {t:.1f} s")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # Additional visualizations
    st.write("**Additional Visualizations**")
    vis_type = st.selectbox("Select Visualization Type", [
        "None", "Heatmap for φ", "Heatmap for c", "Heatmap for ψ", "3D Surface for φ"
    ])
    if vis_type != "None":
        if "Heatmap" in vis_type:
            if "ψ" in vis_type:
                data = psi
                title = "Heatmap for ψ"
            elif "c" in vis_type:
                data = c
                title = "Heatmap for c"
            else:
                data = phi
                title = "Heatmap for φ"
            fig_heatmap = go.Figure(data=go.Heatmap(z=data, x=x, y=y, colorscale=selected_scheme))
            fig_heatmap.update_layout(title=title, xaxis_title="x (cm)", yaxis_title="y (cm)")
            st.plotly_chart(fig_heatmap)
        elif "3D Surface" in vis_type:
            data = phi
            title = "3D Surface for φ"
            fig_surface = go.Figure(data=[go.Surface(z=data, x=x, y=y, colorscale=selected_scheme)])
            fig_surface.update_layout(title=title, scene=dict(xaxis_title="x (cm)", yaxis_title="y (cm)", zaxis_title="φ"), height=600)
            st.plotly_chart(fig_surface)

    # VTR download for selected timestep
    st.write("**Download VTR File for Selected Timestep**")
    with tempfile.TemporaryDirectory() as tmpdirname:
        vtr_path = Path(tmpdirname) / f"electrodeposition_t{t:.1f}.vtr"
        grid = pv.RectilinearGrid(x, y, [0])
        grid.point_data['phi'] = phi.T.ravel(order='F')
        grid.point_data['c'] = c.T.ravel(order='F')
        grid.point_data['phi_l'] = phi_l.T.ravel(order='F')
        grid.point_data['psi'] = psi.T.ravel(order='F')
        grid.save(vtr_path)
        with open(vtr_path, "rb") as f:
            vtr_data = f.read()
        st.download_button(
            label=f"Download VTR for t = {t:.1f} s",
            data=vtr_data,
            file_name=f"electrodeposition_t{t:.1f}.vtr",
            mime="application/octet-stream",
            key=f"vtr_download_{time_index}"
        )

    # Download all VTR files
    st.write("**Download All VTR Files**")
    if st.button("Download All VTR Files as ZIP"):
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = Path(tmpdirname) / "electrodeposition_vtr_files.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for idx, t in enumerate(time_history):
                    vtr_path = Path(tmpdirname) / f"electrodeposition_t{t:.1f}.vtr"
                    grid = pv.RectilinearGrid(x, y, [0])
                    grid.point_data['phi'] = phi_history[idx].T.ravel(order='F')
                    grid.point_data['c'] = c_history[idx].T.ravel(order='F')
                    grid.point_data['phi_l'] = phi_l_history[idx].T.ravel(order='F')
                    grid.point_data['psi'] = psi.T.ravel(order='F')
                    grid.save(vtr_path)
                    zipf.write(vtr_path, f"electrodeposition_t{t:.1f}.vtr")
            with open(zip_path, "rb") as f:
                zip_data = f.read()
            st.download_button(
                label="Download ZIP of All VTR Files",
                data=zip_data,
                file_name="electrodeposition_vtr_files.zip",
                mime="application/zip",
                key="vtr_zip_download"
            )

# Cleanup
if st.button("Clear Output Files"):
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)
    st.success("Output files cleared!")

st.markdown("""
### Parametric Template Design Guideline
Define a function \( g(x, y; p_1, p_2) \leq 0 \) for \(\psi = 1\), centered at \((L_x/2, L_y/2)\):
- **Formula**: \(\psi = 1\) if \( g(x - L_x/2, y - L_y/2; p_1, p_2) \leq 0 \), else \(\psi = 0\).
- **Examples**:
  - Ellipse: \((x/p_1)^2 + (y/p_2)^2 - 1\)
  - Star: \(\sqrt{x^2 + y^2} - p_1 (1 + 0.2 \cos(p_2 \arctan2(y, x)))\)
- **Input**: Enter \( g(x, y, p1, p2) \) in the text box, use \( p1 \), \( p2 \) as parameters.
- **Tips**: Ensure \( g \) is continuous and centered at \((0, 0)\) in shifted coordinates.

### Instructions
1. Adjust parameters in the sidebar, especially c_bulk to assess shell thickness.
2. Select template geometry (Circle for 2 nm Cu core).
3. Click 'Run Simulation'.
4. Use time slider to view results and check shell thickness.
5. Choose Plotly color scheme.
6. View Matplotlib plots at \( x = L_x/2 \).
7. Download VTR files (single or all timesteps).
8. Select additional visualizations, including ψ.
9. Clear files with 'Clear Output Files'.

### Notes
- **φ = 1** (Ag shell), **φ = 0** (electrolyte), **ψ = 1** (2 nm Cu core, fixed).
- **c = 0** at Cu core (ψ = 1) and Ag shell (φ > 0.5), **c = c_bulk * ratio** at far-field boundaries.
- Shell thickness increases with higher c_bulk (test 1e-6 to 1e-4 mol/cm³).
- VTR files are compatible with Paraview for 3D visualization.
""")
