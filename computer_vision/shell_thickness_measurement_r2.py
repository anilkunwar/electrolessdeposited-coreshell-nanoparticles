"""
CoreShellGPT: Geometric EDS Shell Thickness Extractor
=====================================================
Streamlit Cloud-ready application with AUTO-LOADING of all EDS images
in the app directory. Processes all Cu:Ag ratios automatically on startup.

Version: 2.1 (Auto-Loading + Ag Validation)
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import re
import os
from pathlib import Path


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="CoreShellGPT: Geometric EDS Extractor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 CoreShellGPT: Geometric EDS Shell Thickness Extractor")
st.markdown("""
**Automated extraction of Ag shell thickness and Cu/Ag mole fraction proxies from EDS line-scan images.**

⚠️ **PRE-PROCESSING:** Place your cropped EDS images (containing ONLY the colored curves, no axes/legends/TEM insets)
in the same directory as this app. The app will auto-discover and process all of them on startup.
""")


# =============================================================================
# CORE ANALYSIS FUNCTION
# =============================================================================

def extract_eds_geometric(img_array, nm_per_px=0.35,
                           red_threshold=20, green_threshold=20, intensity_min=80,
                           baseline=10, intensity_threshold=15,
                           text_threshold=50):
    """
    Extract EDS line-scan profiles using color-dominance detection and
    calculate shell thickness via the geometric diameter method with Ag validation.
    """
    R = img_array[:, :, 0].astype(float)
    G = img_array[:, :, 1].astype(float)
    B = img_array[:, :, 2].astype(float)
    
    # Step 1: Exclude text and dark artifacts
    text_mask = (R < text_threshold) & (G < text_threshold) & (B < text_threshold)
    
    # Step 2: Color-dominance pixel detection
    red_pixels = (
        (R > G + red_threshold) & 
        (R > B + red_threshold) & 
        (R > intensity_min) & 
        (~text_mask)
    )
    green_pixels = (
        (G > R + green_threshold) & 
        (G > B + green_threshold) & 
        (G > intensity_min) & 
        (~text_mask)
    )
    
    # Step 3: Extract 1D intensity profiles
    R_profile = np.array([
        np.max(R[red_pixels[:, x], x]) if np.any(red_pixels[:, x]) else 0
        for x in range(img_array.shape[1])
    ])
    G_profile = np.array([
        np.max(G[green_pixels[:, x], x]) if np.any(green_pixels[:, x]) else 0
        for x in range(img_array.shape[1])
    ])
    
    # Step 4: Baseline subtraction
    R_net = np.maximum(R_profile - baseline, 0)
    G_net = np.maximum(G_profile - baseline, 0)
    
    # ==========================================
    # GEOMETRIC METHOD WITH AG VALIDATION
    # ==========================================
    
    # D_total: spatial extent where ANY signal exceeds background
    total_signal = R_net + G_net
    particle_mask = total_signal > intensity_threshold
    
    # D_core: spatial extent where Cu DOMINATES Ag (critical fix)
    cu_dominant = (R_net > G_net) & (R_net > intensity_threshold)
    
    # Ag validation checks
    ag_detectable = np.any(G_net > intensity_threshold)
    ag_dominant_edges = np.any((G_net > R_net) & (G_net > intensity_threshold))
    
    # Connected component analysis for Ag-dominant regions
    ag_dominant_mask = (G_net > R_net) & (G_net > intensity_threshold)
    labeled_array, num_features = ndimage.label(ag_dominant_mask)
    
    ag_regions = []
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        if len(indices) >= 2:
            ag_regions.append({
                'start_px': int(indices[0]),
                'end_px': int(indices[-1]),
                'length_px': int(len(indices)),
                'position': 'left' if indices[0] < len(R_profile) / 2 else 'right'
            })
    
    def get_width_and_bounds(mask):
        if not np.any(mask):
            return 0, 0, 0
        start = int(np.argmax(mask))
        end = int(len(mask) - 1 - np.argmax(mask[::-1]))
        return (end - start), start, end
    
    D_total_px, start_t, end_t = get_width_and_bounds(particle_mask)
    D_core_px, start_c, end_c = get_width_and_bounds(cu_dominant)
    
    D_total_nm = D_total_px * nm_per_px
    D_core_nm = D_core_px * nm_per_px
    
    # Shell thickness with Ag validation
    if not ag_detectable or not ag_dominant_edges:
        delta_nm = 0.0
        structure_type = "Discontinuous / No detectable Ag shell"
    elif D_total_px > 0:
        delta_px = max(0, (D_total_px - D_core_px) / 2)
        delta_nm = delta_px * nm_per_px
    else:
        delta_nm = 0.0
        structure_type = "No particle detected"
    
    # Structure classification
    if D_total_px == 0:
        structure_type = "No particle detected"
    elif not ag_detectable:
        structure_type = "Pure Cu (No Ag detected)"
    elif D_core_px < (0.2 * D_total_px):
        structure_type = "Homogeneous Ag (No distinct core)"
    elif delta_nm < 1.0:
        structure_type = "Discontinuous / Ultra-thin shell"
    else:
        structure_type = "Valid Core-Shell"
    
    # Mole fraction proxy (AUC)
    valid_signal = (R_net > intensity_threshold) | (G_net > intensity_threshold)
    area_Cu = np.sum(R_net[valid_signal])
    area_Ag = np.sum(G_net[valid_signal])
    total_area = area_Cu + area_Ag
    ag_frac = area_Ag / total_area if total_area > 0 else 0.0
    cu_frac = area_Cu / total_area if total_area > 0 else 0.0
    
    return {
        'R_profile': R_profile, 'G_profile': G_profile,
        'R_net': R_net, 'G_net': G_net,
        'particle_mask': particle_mask, 'cu_dominant': cu_dominant,
        'D_total_px': D_total_px, 'D_core_px': D_core_px,
        'D_total_nm': D_total_nm, 'D_core_nm': D_core_nm,
        'delta_nm': delta_nm, 'structure_type': structure_type,
        'start_t': start_t, 'end_t': end_t,
        'start_c': start_c, 'end_c': end_c,
        'ag_frac': ag_frac, 'cu_frac': cu_frac,
        'total_area': total_area,
        'red_pixels': red_pixels, 'green_pixels': green_pixels,
        'ag_detectable': ag_detectable,
        'ag_dominant_edges': ag_dominant_edges,
        'ag_regions': ag_regions,
        'labeled_array': labeled_array,
        'num_features': num_features
    }


def detect_molar_ratio(filename):
    """Auto-detect Cu:Ag molar ratio from filename using regex patterns."""
    patterns = [
        r'Cu[;:]?Ag[=:]?(\d+)[;:](\d+)',
        r'(\d+)[;:](\d+).*?(?:Cu|Ag)',
        r'(?:ratio|molar)[_\s]?(\d+)[;:](\d+)',
        r'(\d+)[_\-]?(?:to|_)(\d+)',
        r'Fig\w*[_\s]?(\d+)[_\-]?(\d+)',
        r'[_\s](\d+)[_\-](\d+)[_\s]',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return f"{match.group(1)}:{match.group(2)}"
    return None


def discover_images(directory="."):
    """
    Auto-discover all image files in the app directory.
    Returns a sorted list of (filepath, detected_ratio, label) tuples.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for f in sorted(os.listdir(directory)):
        ext = Path(f).suffix.lower()
        if ext in image_extensions:
            filepath = os.path.join(directory, f)
            ratio = detect_molar_ratio(f)
            # If no ratio detected, try to infer from common naming
            if not ratio:
                # Check for single digits that might be ratios
                digits = re.findall(r'(\d)', f)
                if len(digits) >= 2:
                    ratio = f"{digits[0]}:{digits[1]}"
                else:
                    ratio = "?:?"
            image_files.append((filepath, ratio, f))
    
    return image_files


# =============================================================================
# SIDEBAR: SETTINGS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("---")
    st.header("🔧 Calibration")
    nm_per_px = st.number_input(
        "nm per pixel", value=0.35, step=0.05, min_value=0.01,
        help="From TEM scale bar. E.g., if 20 nm = 57 px, enter 0.35."
    )
    
    st.markdown("---")
    st.header("🎨 Color Detection")
    red_thresh = st.slider(
        "Red (Cu) dominance threshold", min_value=5, max_value=60,
        value=20, step=5,
        help="How much R must exceed G and B to count as a Cu pixel."
    )
    green_thresh = st.slider(
        "Green (Ag) dominance threshold", min_value=5, max_value=60,
        value=20, step=5,
        help="How much G must exceed R and B to count as an Ag pixel."
    )
    min_intensity = st.slider(
        "Minimum color intensity", min_value=30, max_value=200,
        value=80, step=10,
        help="Exclude faint anti-aliasing artifacts."
    )
    text_thresh = st.slider(
        "Text exclusion threshold", min_value=10, max_value=100,
        value=50, step=5,
        help="Pixels darker than this in ALL channels are treated as text/axes."
    )
    
    st.markdown("---")
    st.header("📐 Geometric Analysis")
    baseline = st.number_input(
        "Background baseline intensity", value=10, step=5, min_value=0
    )
    intensity_threshold = st.number_input(
        "Signal detection threshold", value=15, step=5, min_value=0
    )
    
    st.markdown("---")
    st.header("📁 Manual Upload (Optional)")
    st.markdown("If auto-discovery fails, upload images manually:")
    manual_files = st.file_uploader(
        "Upload cropped EDS images", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True, key="manual_upload"
    )


# =============================================================================
# AUTO-DISCOVER AND PROCESS ALL IMAGES
# =============================================================================
st.subheader("📁 Auto-Discovered Images")

# Try to auto-discover images
auto_images = discover_images(".")

# Use manual uploads if provided, otherwise use auto-discovered
if manual_files:
    st.info(f"Using {len(manual_files)} manually uploaded images.")
    image_list = []
    for f in manual_files:
        ratio = detect_molar_ratio(f.name) or "?:?"
        image_list.append((f, ratio, f.name))
elif auto_images:
    st.info(f"Auto-discovered {len(auto_images)} image(s) in app directory.")
    image_list = auto_images
else:
    st.warning("No images found. Please upload images manually using the sidebar.")
    image_list = []

# Store results for summary table
all_results = {}
batch_results = []

if image_list:
    # Create tabs for each image
    tabs = st.tabs([f"{ratio}" for _, ratio, _ in image_list])
    
    for idx, (image_source, detected_ratio, filename) in enumerate(image_list):
        with tabs[idx]:
            # Load image
            if isinstance(image_source, str):
                # Auto-discovered file path
                image = Image.open(image_source).convert("RGB")
            else:
                # Uploaded file object
                image = Image.open(image_source).convert("RGB")
            
            img_array = np.array(image)
            
            # Allow manual override of ratio
            col_info1, col_info2 = st.columns([1, 2])
            with col_info1:
                st.image(image, caption=f"{filename}", use_column_width=True)
            with col_info2:
                st.markdown(f"**Detected ratio:** `{detected_ratio}`")
                molar_ratio = st.text_input(
                    "Override Cu:Ag ratio if needed:",
                    value=detected_ratio,
                    key=f"ratio_override_{idx}"
                )
            
            # Run analysis
            results = extract_eds_geometric(
                img_array, nm_per_px=nm_per_px,
                red_threshold=red_thresh, green_threshold=green_thresh,
                intensity_min=min_intensity, baseline=baseline,
                intensity_threshold=intensity_threshold, text_threshold=text_thresh
            )
            all_results[molar_ratio] = results
            
            # Display metrics
            st.subheader("📊 Geometric Metrics")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Structure", results['structure_type'])
            m2.metric("Shell δ", f"{results['delta_nm']:.2f} nm")
            m3.metric("D_total", f"{results['D_total_nm']:.2f} nm")
            m4.metric("D_core", f"{results['D_core_nm']:.2f} nm")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Ag Detectable", "Yes" if results['ag_detectable'] else "No")
            c2.metric("Ag Edge Dom.", "Yes" if results['ag_dominant_edges'] else "No")
            c3.metric("Ag Regions", len(results['ag_regions']))
            
            m5, m6 = st.columns(2)
            m5.metric("Ag Mole Fraction", f"{results['ag_frac']:.1%}")
            m6.metric("Cu Mole Fraction", f"{results['cu_frac']:.1%}")
            
            # Profile plot
            fig, ax = plt.subplots(figsize=(12, 4))
            x = np.arange(len(results['R_profile']))
            ax.plot(x, results['R_profile'], 'r-', label='Cu Signal', alpha=0.8, linewidth=2)
            ax.plot(x, results['G_profile'], 'g-', label='Ag Signal', alpha=0.8, linewidth=2)
            
            if results['D_total_px'] > 0:
                ax.axvline(results['start_t'], color='blue', linestyle='--', linewidth=2)
                ax.axvline(results['end_t'], color='blue', linestyle='--', linewidth=2)
                ax.axvspan(results['start_t'], results['end_t'], color='blue', alpha=0.1,
                           label=f'D_total = {results["D_total_nm"]:.1f} nm')
            
            if results['D_core_px'] > 0:
                ax.axvline(results['start_c'], color='darkred', linestyle=':', linewidth=2)
                ax.axvline(results['end_c'], color='darkred', linestyle=':', linewidth=2)
                ax.axvspan(results['start_c'], results['end_c'], color='red', alpha=0.2,
                           label=f'D_core = {results["D_core_nm"]:.1f} nm')
            
            for r in results['ag_regions']:
                ax.axvspan(r['start_px'], r['end_px'], color='lime', alpha=0.25)
            
            ax.axhline(y=baseline + intensity_threshold, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Pixel Position")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"EDS Line Scan — Cu:Ag = {molar_ratio} — δ = {results['delta_nm']:.2f} nm")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Data export for this image
            df = pd.DataFrame({
                'Pixel_X': x,
                'Cu_Intensity': results['R_profile'],
                'Ag_Intensity': results['G_profile'],
                'Cu_Net': results['R_net'],
                'Ag_Net': results['G_net'],
                'Is_Particle': results['particle_mask'].astype(int),
                'Is_Core': results['cu_dominant'].astype(int)
            })
            safe_ratio = molar_ratio.replace(':', '_').replace('?', 'unknown')
            st.download_button(
                label=f"📥 Download CSV for {molar_ratio}",
                data=df.to_csv(index=False),
                file_name=f'eds_{safe_ratio}_data.csv',
                mime='text/csv',
                key=f"download_{idx}"
            )
            
            # Add to batch results
            batch_results.append({
                'Filename': filename,
                'Cu:Ag Ratio': molar_ratio,
                'Structure': results['structure_type'],
                'D_total (nm)': round(results['D_total_nm'], 2),
                'D_core (nm)': round(results['D_core_nm'], 2),
                'Shell δ (nm)': round(results['delta_nm'], 2),
                'Ag Detectable': "Yes" if results['ag_detectable'] else "No",
                'Ag Edge Dom.': "Yes" if results['ag_dominant_edges'] else "No",
                'Ag Regions': len(results['ag_regions']),
                'Ag Fraction': f"{results['ag_frac']:.1%}",
                'Cu Fraction': f"{results['cu_frac']:.1%}"
            })


# =============================================================================
# BATCH SUMMARY TABLE (Always visible if images processed)
# =============================================================================
if batch_results:
    st.markdown("---")
    st.subheader("📋 Complete Batch Summary")
    
    df_batch = pd.DataFrame(batch_results)
    st.dataframe(df_batch, use_container_width=True)
    
    st.download_button(
        label="📥 Download Complete Batch Summary (CSV)",
        data=df_batch.to_csv(index=False),
        file_name='batch_eds_analysis.csv',
        mime='text/csv'
    )
    
    # Comparison plot
    st.subheader("📈 Shell Thickness vs. Cu:Ag Ratio")
    
    # Sort by ratio for plotting
    def ratio_sort_key(row):
        try:
            parts = row['Cu:Ag Ratio'].split(':')
            return int(parts[0]) / int(parts[1])
        except:
            return 999
    
    df_sorted = df_batch.copy()
    df_sorted['sort_key'] = df_sorted.apply(ratio_sort_key, axis=1)
    df_sorted = df_sorted.sort_values('sort_key')
    
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 5))
    ratios = df_sorted['Cu:Ag Ratio'].tolist()
    deltas = df_sorted['Shell δ (nm)'].astype(float).tolist()
    
    colors = []
    for s in df_sorted['Structure']:
        if 'Valid Core-Shell' in s:
            colors.append('green')
        elif 'Homogeneous' in s:
            colors.append('orange')
        elif 'Discontinuous' in s:
            colors.append('red')
        else:
            colors.append('gray')
    
    bars = ax_cmp.bar(ratios, deltas, color=colors, alpha=0.7, edgecolor='black')
    ax_cmp.set_xlabel("Cu:Ag Molar Ratio")
    ax_cmp.set_ylabel("Ag Shell Thickness δ (nm)")
    ax_cmp.set_title("Ag Shell Thickness Across All Cu:Ag Ratios")
    ax_cmp.axhline(y=3, color='blue', linestyle='--', alpha=0.5, label='Target: 3 nm')
    ax_cmp.axhline(y=5, color='blue', linestyle='--', alpha=0.5, label='Target: 5 nm')
    ax_cmp.legend()
    ax_cmp.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, deltas):
        ax_cmp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig_cmp)


# =============================================================================
# FOOTER: METHODOLOGY
# =============================================================================
with st.expander("🧠 Methodology: Improved Geometric Measurement with Ag Validation"):
    st.markdown("""
    ### The Problem with Intensity Crossover
    
    Previous methods measured shell thickness as the width where `Ag intensity > Cu intensity`.
    This fails for extreme ratios:
    - **1:1 ratio**: Ag is dominant everywhere → measures particle radius, not a shell
    - **5:1 ratio**: Ag is too weak to cross the threshold → misses the shell entirely
    
    ### The Geometric Solution
    
    Shell thickness ($\\delta$) is calculated from spatial boundaries:
    
    $$\\delta = \\frac{D_{total} - D_{core}}{2}$$
    
    Where:
    - **$D_{total}$**: Full spatial width where combined signal (Cu + Ag) exceeds background.
    - **$D_{core}$**: Spatial width where **Cu dominates Ag** (not just exceeds a threshold).
      This is the critical improvement — Cu must be the stronger element in the core region.
    
    ### Ag Validation Checks
    
    Two independent checks prevent false shell detection:
    1. **Ag Detectable**: Is there any region where Ag signal exceeds the noise floor?
    2. **Ag Edge Dominant**: Does Ag dominate at the particle edges (characteristic of core-shell)?
    
    If **either check fails**, shell thickness is forced to **0 nm** and the structure
    is classified as discontinuous.
    
    ### Structure Classification
    
    | Condition | Classification | Typical Ratio |
    |-----------|----------------|---------------|
    | $D_{core} < 20\\%$ of $D_{total}$ | Homogeneous Ag (no core) | 1:1, 2:1 |
    | Ag not detectable OR not edge-dominant | Discontinuous / No shell | 5:1 |
    | $\\delta < 1.0$ nm | Ultra-thin shell | — |
    | Otherwise | Valid Core-Shell | 3:1, 4:1 |
    
    ### Color-Dominance Detection
    
    Unlike simple `np.max()` projection (which captures white background at RGB=255),
    this algorithm identifies pixels where one color channel **significantly exceeds** the others.
    This excludes white/gray background, black text/axes, and anti-aliasing artifacts.
    
    ### Mole Fraction Proxy
    
    Assuming linear detector response, the atomic fraction is approximated by the
    Area Under the Curve (AUC) ratio:
    
    $$\\text{Ag \\%} \\approx \\frac{\\int I_{Ag}(x) \\, dx}{\\int I_{Ag}(x) \\, dx + \\int I_{Cu}(x) \\, dx}$$
    
    *Note: This is an intensity proxy. True atomic percentages require ZAF correction
    from the EDS software, but this proxy is consistent for comparative analysis.*
    """)
