import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd

st.set_page_config(page_title="CoreShellGPT: EDS Extractor", layout="wide")
st.title("🔬 CoreShellGPT: Automated EDS Line-Scan Extractor")
st.markdown("""
**CRITICAL PRE-PROCESSING:** Crop your image to contain **ONLY** the colored line-scan curves. 
Remove axes, legends, TEM insets, and text labels before uploading.
""")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Cropped EDS Curves Only", type=["png", "jpg", "jpeg"])
    
    st.header("⚙️ Calibration")
    nm_per_px = st.number_input("nm per pixel", value=0.4, step=0.1, 
                                help="From TEM scale bar. E.g., 20nm = 50px → 0.4")
    
    st.header("🔍 Color Detection")
    red_thresh = st.slider("Red dominance threshold", 5, 50, 20,
                           help="How much R must exceed G,B to count as Cu")
    green_thresh = st.slider("Green dominance threshold", 5, 50, 20,
                             help="How much G must exceed R,B to count as Ag")
    min_intensity = st.slider("Min color intensity", 50, 200, 100,
                              help="Exclude faint anti-aliasing artifacts")
    
    st.header("📊 Analysis")
    baseline = st.number_input("Background baseline", value=0, step=5)
    intensity_threshold = st.number_input("Signal threshold", value=5, step=1)
    min_length_px = st.number_input("Min contiguous length (px)", value=2, step=1)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    R = img_array[:, :, 0].astype(float)
    G = img_array[:, :, 1].astype(float)
    B = img_array[:, :, 2].astype(float)
    
    # CRITICAL FIX: Color-dominance detection instead of max projection
    # Exclude text pixels (very dark)
    text_mask = (R < 30) & (G < 30) & (B < 30)
    
    red_pixels = ((R > G + red_thresh) & (R > B + red_thresh) & 
                  (R > min_intensity) & (~text_mask))
    green_pixels = ((G > R + green_thresh) & (G > B + green_thresh) & 
                    (G > min_intensity) & (~text_mask))
    
    # Extract profiles: max of specifically colored pixels per column
    R_profile = np.array([np.max(R[red_pixels[:, x], x]) if np.any(red_pixels[:, x]) else 0 
                          for x in range(img_array.shape[1])])
    G_profile = np.array([np.max(G[green_pixels[:, x], x]) if np.any(green_pixels[:, x]) else 0 
                          for x in range(img_array.shape[1])])
    
    # Baseline and threshold
    R_net = np.maximum(R_profile - baseline, 0)
    G_net = np.maximum(G_profile - baseline, 0)
    valid_signal = (R_net > intensity_threshold) | (G_net > intensity_threshold)
    
    # Ag-dominant regions
    ag_dominant = (G_net > R_net) & valid_signal
    labeled_array, num_features = ndimage.label(ag_dominant)
    
    regions = []
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        if len(indices) >= min_length_px:
            regions.append({
                'start_px': indices[0], 'end_px': indices[-1],
                'length_px': len(indices),
                'position': 'left' if indices[0] < len(R_profile)/2 else 'right'
            })
    
    # Shell thickness: average of left and right edge regions
    edge_regions = [r for r in regions if r['position'] in ['left', 'right']]
    thickness_nm = np.mean([r['length_px'] for r in edge_regions]) * nm_per_px if edge_regions else 0
    
    # Mole fraction proxy (AUC)
    area_Cu = np.sum(R_net[valid_signal])
    area_Ag = np.sum(G_net[valid_signal])
    total = area_Cu + area_Ag
    ag_frac = area_Ag / total if total > 0 else 0
    cu_frac = area_Cu / total if total > 0 else 0
    
    # UI Display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Image & Detection")
        st.image(image, caption="Uploaded Cropped Curves", use_column_width=True)
        
        # Detection mask visualization
        mask_vis = np.zeros_like(img_array)
        mask_vis[red_pixels] = [255, 0, 0]
        mask_vis[green_pixels] = [0, 255, 0]
        st.image(mask_vis, caption="Detected: Red=Cu, Green=Ag", use_column_width=True)
    
    with col2:
        st.subheader("📊 Extracted Metrics")
        
        # Structure classification
        if len(regions) == 0:
            structure = "No Ag shell detected"
        elif any(r['position'] == 'left' for r in regions) and any(r['position'] == 'right' for r in regions):
            if all(r['length_px'] > len(R_profile) * 0.3 for r in regions):
                structure = "Homogeneous / No distinct core"
            else:
                structure = "Core-shell structure"
        else:
            structure = "Partial / Discontinuous shell"
        
        m1, m2 = st.columns(2)
        m1.metric("Structure Type", structure)
        m2.metric("Ag Shell Thickness", f"{thickness_nm:.2f} nm" if thickness_nm > 0 else "N/A")
        
        m3, m4 = st.columns(2)
        m3.metric("Ag Mole Fraction Proxy", f"{ag_frac:.1%}")
        m4.metric("Cu Mole Fraction Proxy", f"{cu_frac:.1%}")
        
        # Ratio
        if ag_frac > 0.05:
            ratio_str = f"1 : {cu_frac/ag_frac:.2f}"
        else:
            ratio_str = "Ag below detection"
        st.metric("Cu:Ag Ratio Proxy", ratio_str)
        
        # Profiles plot
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(R_profile))
        ax.plot(x, R_profile, 'r-', label='Cu Signal', alpha=0.8, linewidth=2)
        ax.plot(x, G_profile, 'g-', label='Ag Signal', alpha=0.8, linewidth=2)
        for r in regions:
            ax.axvspan(r['start_px'], r['end_px'], color='lime', alpha=0.2)
        ax.axhline(y=baseline + intensity_threshold, color='gray', linestyle='--')
        ax.set_xlabel("Pixel Position"); ax.set_ylabel("Intensity")
        ax.set_title("EDS Line Scan with Ag-Dominant Regions")
        ax.legend()
        st.pyplot(fig)
    
    # Data export
    df = pd.DataFrame({
        'Pixel_X': x, 'Cu_Intensity': R_profile, 'Ag_Intensity': G_profile,
        'Is_Ag_Shell': ag_dominant.astype(int)
    })
    st.download_button("Download CSV", df.to_csv(index=False), "eds_data.csv", "text/csv")
    
    # Theory
    with st.expander("🧠 Methodology"):
        st.markdown("""
        **1. Color-Dominance Detection:** Unlike simple max projection, we identify pixels where one RGB channel significantly exceeds the others. This excludes white background and gray anti-aliasing artifacts.
        
        **2. Text Exclusion:** Black text pixels (R≈G≈B≈0) are masked out to prevent false signal detection.
        
        **3. Shell Thickness:** Contiguous Ag-dominant regions at the left and right edges are identified via connected component analysis. Their average width × calibration factor = physical thickness.
        
        **4. Mole Fraction Proxy:** Area Under Curve (AUC) ratio approximates atomic fraction assuming linear detector response:
        $$\\text{Ag \\%} \\approx \\frac{\\int I_G(x)dx}{\\int I_G(x)dx + \\int I_R(x)dx}$$
        """)
