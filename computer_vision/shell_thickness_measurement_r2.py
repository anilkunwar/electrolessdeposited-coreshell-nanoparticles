import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="CoreShellGPT: Geometric EDS Extractor", layout="wide")
st.title("🔬 CoreShellGPT: Geometric EDS Shell Thickness Extractor")
st.markdown("""
**CRITICAL PRE-PROCESSING:** Crop your image to contain **ONLY** the colored line-scan curves. 
Remove axes, legends, TEM insets, and text labels before uploading.
""")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Cropped EDS Curves Only", type=["png", "jpg", "jpeg"])
    
    st.header("⚙️ Calibration")
    nm_per_px = st.number_input("nm per pixel", value=0.4, step=0.05, 
                                help="From TEM scale bar. E.g., 20nm = 50px → 0.4")
    
    st.header("🔍 Color Detection")
    red_thresh = st.slider("Red dominance threshold", 5, 50, 20,
                           help="R must exceed G,B by this much to count as Cu")
    green_thresh = st.slider("Green dominance threshold", 5, 50, 20,
                             help="G must exceed R,B by this much to count as Ag")
    min_intensity = st.slider("Min color intensity", 50, 200, 100,
                              help="Exclude faint anti-aliasing artifacts")
    text_thresh = st.slider("Text exclusion threshold", 10, 100, 30,
                          help="Pixels darker than this in all channels are treated as text")
    
    st.header("📊 Geometric Analysis")
    baseline = st.number_input("Background baseline", value=10, step=5)
    intensity_threshold = st.number_input("Signal threshold (for D_total)", value=15, step=1)
    core_threshold_mult = st.slider("Core threshold multiplier", 1.0, 3.0, 1.5, step=0.1,
                                    help="Multiplier for intensity_threshold to define Cu core boundary")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    R = img_array[:, :, 0].astype(float)
    G = img_array[:, :, 1].astype(float)
    B = img_array[:, :, 2].astype(float)
    
    # ==========================================
    # COLOR-DOMINANCE EXTRACTION (not max projection)
    # ==========================================
    # Exclude text pixels (very dark)
    text_mask = (R < text_thresh) & (G < text_thresh) & (B < text_thresh)
    
    # Detect specifically colored pixels
    red_pixels = ((R > G + red_thresh) & (R > B + red_thresh) & 
                  (R > min_intensity) & (~text_mask))
    green_pixels = ((G > R + green_thresh) & (G > B + green_thresh) & 
                    (G > min_intensity) & (~text_mask))
    
    # Extract profiles: max of specifically colored pixels per column
    R_profile = np.array([np.max(R[red_pixels[:, x], x]) if np.any(red_pixels[:, x]) else 0 
                          for x in range(img_array.shape[1])])
    G_profile = np.array([np.max(G[green_pixels[:, x], x]) if np.any(green_pixels[:, x]) else 0 
                            for x in range(img_array.shape[1])])
    
    # Baseline subtraction
    R_net = np.maximum(R_profile - baseline, 0)
    G_net = np.maximum(G_profile - baseline, 0)
    
    # ==========================================
    # GEOMETRIC METHOD (Physically Correct)
    # ==========================================
    
    # 1. Total Particle Diameter (D_total)
    total_signal = R_net + G_net
    particle_mask = total_signal > intensity_threshold
    
    # 2. Core Diameter (D_core)
    core_mask = R_net > (intensity_threshold * core_threshold_mult)
    
    def get_width_and_bounds(mask):
        if not np.any(mask):
            return 0, 0, 0
        start = np.argmax(mask)
        end = len(mask) - 1 - np.argmax(mask[::-1])
        return (end - start), start, end

    D_total_px, start_t, end_t = get_width_and_bounds(particle_mask)
    D_core_px, start_c, end_c = get_width_and_bounds(core_mask)
    
    D_total_nm = D_total_px * nm_per_px
    D_core_nm = D_core_px * nm_per_px
    
    # 3. Shell Thickness (delta)
    if D_total_px > 0:
        delta_px = max(0, (D_total_px - D_core_px) / 2)
        delta_nm = delta_px * nm_per_px
    else:
        delta_nm = 0

    # 4. Structure Classification
    if D_total_px == 0:
        structure_type = "No particle detected"
    elif D_core_px < (0.2 * D_total_px):
        structure_type = "⚠️ Homogeneous Ag (No distinct core)"
    elif delta_nm < 1.0:
        structure_type = "⚠️ Discontinuous / Ultra-thin shell"
    else:
        structure_type = "✅ Valid Core-Shell"

    # Mole fraction proxy (AUC)
    valid_signal = (R_net > intensity_threshold) | (G_net > intensity_threshold)
    area_Cu = np.sum(R_net[valid_signal])
    area_Ag = np.sum(G_net[valid_signal])
    total_area = area_Cu + area_Ag
    ag_frac = area_Ag / total_area if total_area > 0 else 0
    cu_frac = area_Cu / total_area if total_area > 0 else 0

    # ==========================================
    # UI DISPLAY
    # ==========================================
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Image & Detection")
        st.image(image, caption="Uploaded Cropped Curves", use_column_width=True)
        
        # Detection mask
        mask_vis = np.zeros_like(img_array)
        mask_vis[red_pixels] = [255, 0, 0]
        mask_vis[green_pixels] = [0, 255, 0]
        st.image(mask_vis, caption="Detected: Red=Cu, Green=Ag", use_column_width=True)
        
    with col2:
        st.subheader("📊 Geometric Metrics")
        
        m1, m2 = st.columns(2)
        m1.metric("Structure Type", structure_type)
        m2.metric("Ag Shell Thickness (δ)", f"{delta_nm:.2f} nm")
        
        m3, m4 = st.columns(2)
        m3.metric("Total Diameter (D_total)", f"{D_total_nm:.2f} nm")
        m4.metric("Core Diameter (D_core)", f"{D_core_nm:.2f} nm")
        
        st.divider()
        st.success(f"**Formula:** δ = (D_total − D_core) / 2 = ({D_total_nm:.2f} − {D_core_nm:.2f}) / 2 = **{delta_nm:.2f} nm**")
        
        m5, m6 = st.columns(2)
        m5.metric("Ag Mole Fraction Proxy", f"{ag_frac:.1%}")
        m6.metric("Cu Mole Fraction Proxy", f"{cu_frac:.1%}")
        
        # Profiles with geometric boundaries
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(R_profile))
        ax.plot(x, R_profile, 'r-', label='Cu Signal', alpha=0.8, linewidth=2)
        ax.plot(x, G_profile, 'g-', label='Ag Signal', alpha=0.8, linewidth=2)
        
        if D_total_px > 0:
            ax.axvline(start_t, color='blue', linestyle='--', linewidth=2)
            ax.axvline(end_t, color='blue', linestyle='--', linewidth=2)
            ax.axvspan(start_t, end_t, color='blue', alpha=0.1, 
                       label=f'D_total = {D_total_nm:.1f} nm')
            
        if D_core_px > 0:
            ax.axvline(start_c, color='darkred', linestyle=':', linewidth=2)
            ax.axvline(end_c, color='darkred', linestyle=':', linewidth=2)
            ax.axvspan(start_c, end_c, color='red', alpha=0.2, 
                       label=f'D_core = {D_core_nm:.1f} nm')
            
        ax.axhline(y=baseline + intensity_threshold, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Pixel Position"); ax.set_ylabel("Intensity")
        ax.set_title("EDS Line Scan with Geometric Boundaries")
        ax.legend(loc='upper right', fontsize='small')
        st.pyplot(fig)

    # Data export
    df = pd.DataFrame({
        'Pixel_X': x, 'Cu_Intensity': R_profile, 'Ag_Intensity': G_profile,
        'Cu_Net': R_net, 'Ag_Net': G_net,
        'Is_Particle': particle_mask.astype(int),
        'Is_Core': core_mask.astype(int)
    })
    st.download_button("Download CSV", df.to_csv(index=False), "eds_geometric_data.csv", "text/csv")

else:
    st.info("👈 Please upload an image in the sidebar to begin analysis.")

with st.expander("🧠 Methodology: Why Geometric Measurement?"):
    st.markdown("""
    **The Flaw of Intensity Crossover:** Previous methods measured the width where `Ag > Cu`. This fails for:
    - **1:1 ratios**: Ag is dominant everywhere → measures particle radius, not shell
    - **5:1 ratios**: Ag is too weak to cross threshold → misses shell entirely
    
    **The Correct Physical Theory:** Shell thickness ($\\delta$) must be calculated geometrically:
    1. **D_total**: Spatial width where combined signal (Cu + Ag) exceeds background
    2. **D_core**: Spatial width where Cu signal exceeds noise floor
    3. **Shell thickness**: $\\delta = (D_{total} - D_{core}) / 2$
    
    **Structure Classification:**
    - $D_{core} < 20\\%$ of $D_{total}$ → **Homogeneous Ag** (no core, typical of 1:1)
    - $\\delta < 1.0$ nm → **Discontinuous/Ultra-thin** (typical of 5:1)
    - Otherwise → **Valid Core-Shell** (typical of 3:1 and 4:1)
    
    **Color-Dominance Detection:** Unlike simple max projection (which captures white background at 255), we identify pixels where one RGB channel significantly exceeds the others. This excludes white/gray background and anti-aliasing artifacts.
    """)
