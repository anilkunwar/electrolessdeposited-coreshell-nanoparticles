import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="CoreShellGPT: EDS Extractor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Academic Look ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 CoreShellGPT: Automated EDS Line-Scan Extractor")
st.markdown("""
**Objective:** Automatically extract Ag shell thickness and estimate Cu/Ag mole fraction proxies from EDS line-scan images.
*⚠️ Crucial Pre-processing: Please crop your image to contain ONLY the line-scan curves (remove axes, legends, and TEM insets) before uploading.*
""")

# ==========================================
# SIDEBAR: Calibration & Parameters
# ==========================================
with st.sidebar:
    st.header("⚙️ Calibration & Settings")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Cropped EDS Image", type=["png", "jpg", "jpeg"])
    
    st.subheader("Physical Calibration")
    pixels_per_nm = st.number_input(
        "Pixels per nm (Scale Factor)", 
        value=1.0, step=0.1, min_value=0.1,
        help="Calculate from your TEM scale bar. E.g., if 20nm = 50px, enter 2.5."
    )
    
    st.subheader("Signal Processing")
    baseline = st.number_input("Background Baseline Intensity (0-255)", value=15, step=5)
    intensity_threshold = st.number_input("Min Signal Threshold (0-255)", value=30, step=5)
    min_length_px = st.number_input("Min Contiguous Length (pixels)", value=4, step=1)
    
    st.subheader("Channel Mapping")
    st.info("Default: Red = Cu (Core), Green = Ag (Shell)")
    # In case users have inverted colors
    invert_channels = st.checkbox("Invert Channels (Red=Ag, Green=Cu)", value=False)

# ==========================================
# MAIN LOGIC
# ==========================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # Extract Channels
    R = img_array[:, :, 0].astype(float)
    G = img_array[:, :, 1].astype(float)
    
    if invert_channels:
        R, G = G, R  # Swap if user checked the box

    # 1D Max Projection (Captures the peak of the plotted lines)
    R_profile = np.max(R, axis=0)
    G_profile = np.max(G, axis=0)
    
    # Subtract baseline and apply threshold
    R_net = np.maximum(R_profile - baseline, 0)
    G_net = np.maximum(G_profile - baseline, 0)
    valid_signal = (R_net > intensity_threshold) | (G_net > intensity_threshold)
    
    # Identify Ag-dominant regions (Shell)
    ag_dominant_mask = (G_net > R_net) & valid_signal
    
    # Connected Component Analysis
    labeled_array, num_features = ndimage.label(ag_dominant_mask)
    
    # Extract contiguous regions
    regions = []
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        length = len(indices)
        if length >= min_length_px:
            regions.append({
                'id': i,
                'start_px': indices[0],
                'end_px': indices[-1],
                'length_px': length,
                'position': 'left' if indices[0] < len(R_profile)/2 else 'right'
            })
            
    # ==========================================
    # CALCULATIONS
    # ==========================================
    # 1. Shell Thickness (Average of left and right edge shells)
    left_shells = [r for r in regions if r['position'] == 'left']
    right_shells = [r for r in regions if r['position'] == 'right']
    
    thickness_nm = 0.0
    if left_shells or right_shells:
        valid_lengths = [r['length_px'] for r in regions]
        avg_length_px = np.mean(valid_lengths)
        thickness_nm = avg_length_px / pixels_per_nm
        
    # 2. Mole Fraction Proxy (Area Under Curve)
    total_area_Cu = np.sum(R_net[valid_signal])
    total_area_Ag = np.sum(G_net[valid_signal])
    total_area = total_area_Cu + total_area_Ag
    
    ag_fraction = (total_area_Ag / total_area) if total_area > 0 else 0
    cu_fraction = (total_area_Cu / total_area) if total_area > 0 else 0
    ratio_proxy = (cu_fraction / ag_fraction) if ag_fraction > 0 else float('inf')

    # ==========================================
    # UI DISPLAY
    # ==========================================
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Image & Mask Analysis")
        st.image(image, caption="Uploaded Cropped Image", use_column_width=True)
        
        # Generate 2D Binary Mask for Visual Proof
        # Create a 2D mask by extending the 1D profile vertically
        mask_2d = np.zeros_like(R, dtype=bool)  # Same shape as R channel (2D)
        mask_2d[ag_dominant_mask, :] = True  # Apply 1D mask to all rows
        
        # Create RGB mask image
        mask_img = np.zeros_like(img_array)
        mask_img[mask_2d] = [0, 255, 0]  # Highlight Ag regions in pure green
        
        st.image(mask_img, caption="Algorithm Mask (Green = Identified Ag Shell Regions)", use_column_width=True)
        
        # Show the 1D profile mask
        st.markdown("**1D Ag-Dominant Mask Profile:**")
        fig_mask, ax_mask = plt.subplots(figsize=(10, 1))
        ax_mask.imshow(ag_dominant_mask.reshape(1, -1), aspect='auto', cmap='Greens', vmin=0, vmax=1)
        ax_mask.set_xlabel("Pixel Position (x)")
        ax_mask.set_yticks([])
        ax_mask.set_title("Binary Mask: Green = Ag > Cu")
        st.pyplot(fig_mask)

    with col2:
        st.subheader("📊 Extracted Metrics")
        
        m1, m2 = st.columns(2)
        m1.metric("Estimated Ag Shell Thickness", f"{thickness_nm:.2f} nm", delta="Target: 3-5 nm")
        m2.metric("Ag Intensity Fraction (Proxy)", f"{ag_fraction:.2%}")
        
        m3, m4 = st.columns(2)
        m3.metric("Cu Intensity Fraction (Proxy)", f"{cu_fraction:.2%}")
        m4.metric("Cu:Ag Intensity Ratio", f"1 : {1/ratio_proxy:.2f}" if ratio_proxy != float('inf') else "N/A")
        
        st.markdown("---")
        st.subheader("📈 1D Intensity Profiles")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(R_profile))
        ax.plot(x, R_profile, color='red', label='Cu Signal (Red)', alpha=0.8, linewidth=2)
        ax.plot(x, G_profile, color='green', label='Ag Signal (Green)', alpha=0.8, linewidth=2)
        
        # Shade Ag dominant regions
        for region in regions:
            ax.axvspan(region['start_px'], region['end_px'], color='lime', alpha=0.2, label='Ag Shell Region' if region['id']==1 else "")
            
        ax.axhline(y=baseline + intensity_threshold, color='gray', linestyle='--', label='Threshold')
        ax.set_xlabel("Pixel Position (x)")
        ax.set_ylabel("Raw Intensity")
        ax.set_title("EDS Line Scan Extraction")
        ax.legend(loc='upper right')
        st.pyplot(fig)

    # ==========================================
    # DATA EXPORT
    # ==========================================
    st.markdown("---")
    st.subheader("💾 Export Extracted Data")
    
    df = pd.DataFrame({
        'Pixel_X': x,
        'Cu_Intensity': R_profile,
        'Ag_Intensity': G_profile,
        'Is_Ag_Shell': ag_dominant_mask.astype(int)
    })
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Profile Data as CSV",
        data=csv,
        file_name='eds_line_scan_data.csv',
        mime='text/csv',
    )

else:
    st.info("👈 Please upload an image in the sidebar to begin analysis.")

# ==========================================
# THEORETICAL FOOTER
# ==========================================
with st.expander("🧠 Read the Underlying Theory & Methodology"):
    st.markdown("""
    **1. Color Space Decomposition & 1D Projection:**
    The EDS plot is decomposed into Red ($I_R$) and Green ($I_G$) channels. A vertical max-projection $I(x) = \max_y[I(x,y)]$ extracts the 1D curve, ignoring the white background.
    
    **2. Signal Dominance Thresholding:**
    The Ag shell physically resides at the particle edges. We create a boolean mask $M(x) = 1$ where $I_G(x) > I_R(x)$ and $I(x) > I_{threshold}$. The central region (Cu core) naturally yields $M(x) = 0$.
    
    **3. Contiguous Edge Extraction:**
    Using connected component analysis, we isolate the left-most and right-most contiguous regions where $M(x)=1$. The average pixel width of these edge regions, divided by the calibration factor ($k$), yields the physical shell thickness: $\Delta r = L_{px} / k$.
    
    **4. Mole Fraction Proxy (AUC):**
    Assuming linear detector response, the atomic fraction is approximated via the Area Under the Curve (AUC): 
    $$ \text{Ag Fraction} \approx \frac{\int I_G(x)dx}{\int I_G(x)dx + \int I_R(x)dx} $$
    *Note: This is an intensity proxy. True atomic percentages require ZAF correction factors from the EDS software, but this proxy is highly consistent for comparative trend analysis across different Cu:Ag molar ratios.*
    """)
