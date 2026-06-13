"""
CoreShellGPT: Geometric EDS Shell Thickness Extractor
=====================================================
Uses FWHM (Full Width at Half Maximum) method to handle beam broadening
and noise spikes correctly. Auto-loads from images/ folder.

Version: 3.0 (FWHM Method + Gaussian Smoothing)
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import re
import os
from pathlib import Path


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="CoreShellGPT: FWHM EDS Extractor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 CoreShellGPT: FWHM EDS Shell Thickness Extractor")
st.markdown("""
**Robust extraction using FWHM (Full Width at Half Maximum) method with Gaussian smoothing.**

📁 Images auto-loaded from `images/` folder. Each image should be **cropped** to contain 
ONLY the colored curves (no axes, legends, TEM insets, text labels).

**Method:** $D_{total}$ = width at 10% max combined signal | $D_{core}$ = FWHM of Cu signal | $\delta = (D_{total} - D_{core})/2$
""")


# =============================================================================
# CORE ANALYSIS FUNCTION (FWHM METHOD)
# =============================================================================

def extract_eds_fwhm(img_array, nm_per_px=0.35,
                      red_threshold=20, green_threshold=20, intensity_min=80,
                      baseline=10, intensity_threshold=15,
                      text_threshold=50, sigma=2):
    """
    Extract EDS profiles using color-dominance detection, then calculate
    shell thickness via FWHM geometric method with Gaussian smoothing.
    
    D_total: width of combined (Cu+Ag) signal at 10% of maximum height
    D_core: FWHM (50% max height) of Cu signal
    delta = (D_total - D_core) / 2
    """
    R = img_array[:, :, 0].astype(float)
    G = img_array[:, :, 1].astype(float)
    B = img_array[:, :, 2].astype(float)
    
    # Exclude text and dark artifacts
    text_mask = (R < text_threshold) & (G < text_threshold) & (B < text_threshold)
    
    # Color-dominance pixel detection
    red_pixels = (
        (R > G + red_threshold) & (R > B + red_threshold) & 
        (R > intensity_min) & (~text_mask)
    )
    green_pixels = (
        (G > R + green_threshold) & (G > B + green_threshold) & 
        (G > intensity_min) & (~text_mask)
    )
    
    # Extract 1D intensity profiles
    R_profile = np.array([
        np.max(R[red_pixels[:, x], x]) if np.any(red_pixels[:, x]) else 0
        for x in range(img_array.shape[1])
    ])
    G_profile = np.array([
        np.max(G[green_pixels[:, x], x]) if np.any(green_pixels[:, x]) else 0
        for x in range(img_array.shape[1])
    ])
    
    # Baseline subtraction
    R_net = np.maximum(R_profile - baseline, 0)
    G_net = np.maximum(G_profile - baseline, 0)
    
    # ==========================================
    # FWHM GEOMETRIC METHOD
    # ==========================================
    
    # 1. Gaussian smoothing to reduce noise and beam broadening artifacts
    R_smooth = gaussian_filter1d(R_net, sigma=sigma)
    G_smooth = gaussian_filter1d(G_net, sigma=sigma)
    total_smooth = R_smooth + G_smooth
    
    # 2. D_total: Width at 10% of maximum combined signal
    # This captures the full particle extent while ignoring distant noise
    max_total = np.max(total_smooth)
    D_total_px = 0
    start_t = end_t = 0
    if max_total > 0:
        total_mask = total_smooth >= (max_total * 0.10)
        if np.any(total_mask):
            indices = np.where(total_mask)[0]
            start_t = int(indices[0])
            end_t = int(indices[-1])
            D_total_px = end_t - start_t
    
    # 3. D_core: FWHM (50% max height) of the Cu signal
    # Standard scientific method for measuring core size in EDS
    # Ignores long beam-broadening tails
    max_cu = np.max(R_smooth)
    D_core_px = 0
    start_c = end_c = 0
    if max_cu > 0:
        cu_mask = R_smooth >= (max_cu * 0.50)
        if np.any(cu_mask):
            indices = np.where(cu_mask)[0]
            start_c = int(indices[0])
            end_c = int(indices[-1])
            D_core_px = end_c - start_c
    
    D_total_nm = D_total_px * nm_per_px
    D_core_nm = D_core_px * nm_per_px
    
    # 4. Shell thickness
    if D_total_px > 0:
        delta_px = max(0, (D_total_px - D_core_px) / 2)
        delta_nm = delta_px * nm_per_px
    else:
        delta_nm = 0.0
    
    # 5. Structure classification based on physical trends
    if D_total_px == 0:
        structure_type = "No particle detected"
    elif D_core_px < (0.3 * D_total_px):
        # Core is very small relative to total → homogeneous Ag
        structure_type = "Homogeneous Ag (No distinct core)"
    elif delta_nm < 1.5:
        # Shell is very thin → discontinuous
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
    
    # Ag regions detection (for visualization only)
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
    
    # Ag validation checks
    ag_detectable = np.any(G_net > intensity_threshold)
    ag_dominant_edges = np.any((G_net > R_net) & (G_net > intensity_threshold))
    
    return {
        'R_profile': R_profile,
        'G_profile': G_profile,
        'R_net': R_smooth,  # Return smoothed for plotting
        'G_net': G_smooth,
        'D_total_px': D_total_px,
        'D_core_px': D_core_px,
        'D_total_nm': D_total_nm,
        'D_core_nm': D_core_nm,
        'delta_nm': delta_nm,
        'structure_type': structure_type,
        'start_t': start_t,
        'end_t': end_t,
        'start_c': start_c,
        'end_c': end_c,
        'ag_frac': ag_frac,
        'cu_frac': cu_frac,
        'total_area': total_area,
        'red_pixels': red_pixels,
        'green_pixels': green_pixels,
        'ag_detectable': ag_detectable,
        'ag_dominant_edges': ag_dominant_edges,
        'ag_regions': ag_regions,
        'labeled_array': labeled_array,
        'num_features': num_features
    }


def detect_molar_ratio(filename):
    """Auto-detect Cu:Ag molar ratio from filename."""
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
    digits = re.findall(r'(\d)', filename)
    if len(digits) >= 2:
        return f"{digits[0]}:{digits[1]}"
    return "?:?"


def load_images_from_dir(directory):
    """Load all image files from a directory. Returns list of (filepath, ratio, filename)."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    results = []
    
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return results
    
    for f in sorted(os.listdir(directory)):
        ext = Path(f).suffix.lower()
        if ext in image_extensions:
            filepath = os.path.join(directory, f)
            ratio = detect_molar_ratio(f)
            results.append((filepath, ratio, f))
    
    return results


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
        value=20, step=5
    )
    green_thresh = st.slider(
        "Green (Ag) dominance threshold", min_value=5, max_value=60,
        value=20, step=5
    )
    min_intensity = st.slider(
        "Minimum color intensity", min_value=30, max_value=200,
        value=80, step=10
    )
    text_thresh = st.slider(
        "Text exclusion threshold", min_value=10, max_value=100,
        value=50, step=5
    )
    
    st.markdown("---")
    st.header("📐 FWHM Analysis")
    baseline = st.number_input(
        "Background baseline intensity", value=10, step=5, min_value=0
    )
    intensity_threshold = st.number_input(
        "Signal detection threshold", value=15, step=5, min_value=0
    )
    sigma = st.slider(
        "Gaussian smoothing sigma", min_value=0, max_value=5,
        value=2, step=1,
        help="Higher = more smoothing. Reduces noise but may blur edges."
    )
    
    st.markdown("---")
    st.header("📁 Manual Upload (Optional)")
    manual_files = st.file_uploader(
        "Upload cropped EDS images", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True, key="manual_upload"
    )


# =============================================================================
# AUTO-DISCOVER IMAGES FROM images/ FOLDER
# =============================================================================

try:
    app_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    app_dir = os.getcwd()

images_dir = os.path.join(app_dir, "images")

# Fallback locations
fallback_dirs = [
    os.path.join(os.path.dirname(app_dir), "images") if app_dir else None,
    os.path.join(os.getcwd(), "images"),
]

# Load images
image_list = load_images_from_dir(images_dir)

if not image_list:
    for fallback in fallback_dirs:
        if fallback and os.path.isdir(fallback):
            image_list = load_images_from_dir(fallback)
            if image_list:
                images_dir = fallback
                break

# Manual upload fallback
if manual_files and not image_list:
    image_list = []
    for f in manual_files:
        ratio = detect_molar_ratio(f.name)
        image_list.append((f, ratio, f.name))
    images_dir = "manual upload"

# =============================================================================
# DISPLAY STATUS
# =============================================================================

if image_list:
    st.success(f"✅ Loaded {len(image_list)} image(s) from `{images_dir}`")
else:
    st.error(f"""
    ❌ No images found in `images/` folder!
    
    **Expected location:** `{images_dir}`
    **App directory:** `{app_dir}`
    **Current working directory:** `{os.getcwd()}`
    
    Please ensure:
    1. A folder named `images/` exists next to `app.py`
    2. Your cropped `.png` or `.jpg` EDS files are inside it
    3. Or use the Manual Upload option in the sidebar
    """)


# =============================================================================
# PROCESS AND DISPLAY ALL IMAGES
# =============================================================================

all_results = {}
batch_results = []

if image_list:
    # Sort by ratio
    def sort_key(item):
        try:
            parts = item[1].split(':')
            return (int(parts[0]), int(parts[1]))
        except:
            return (999, 999)
    
    image_list = sorted(image_list, key=sort_key)
    
    # Create tabs
    tab_labels = [f"{ratio}" for _, ratio, _ in image_list]
    tabs = st.tabs(tab_labels)
    
    for idx, (image_source, detected_ratio, filename) in enumerate(image_list):
        with tabs[idx]:
            # Load image
            if isinstance(image_source, str):
                image = Image.open(image_source).convert("RGB")
            else:
                image = Image.open(image_source).convert("RGB")
            
            img_array = np.array(image)
            
            # Ratio display with override
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
            
            # Run FWHM analysis
            results = extract_eds_fwhm(
                img_array, nm_per_px=nm_per_px,
                red_threshold=red_thresh, green_threshold=green_thresh,
                intensity_min=min_intensity, baseline=baseline,
                intensity_threshold=intensity_threshold, text_threshold=text_thresh,
                sigma=sigma
            )
            all_results[molar_ratio] = results
            
            # Metrics
            st.subheader("📊 FWHM Geometric Metrics")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Structure", results['structure_type'])
            m2.metric("Shell δ", f"{results['delta_nm']:.2f} nm")
            m3.metric("D_total (10%)", f"{results['D_total_nm']:.2f} nm")
            m4.metric("D_core (FWHM)", f"{results['D_core_nm']:.2f} nm")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Ag Detectable", "Yes" if results['ag_detectable'] else "No")
            c2.metric("Ag Edge Dom.", "Yes" if results['ag_dominant_edges'] else "No")
            c3.metric("Ag Regions", len(results['ag_regions']))
            
            m5, m6 = st.columns(2)
            m5.metric("Ag Mole Fraction", f"{results['ag_frac']:.1%}")
            m6.metric("Cu Mole Fraction", f"{results['cu_frac']:.1%}")
            
            st.success(
                f"**Formula:** δ = (D_total − D_core) / 2 = "
                f"({results['D_total_nm']:.2f} − {results['D_core_nm']:.2f}) / 2 = "
                f"**{results['delta_nm']:.2f} nm**"
            )
            
            # Profile plot with FWHM boundaries
            fig, ax = plt.subplots(figsize=(12, 4))
            x = np.arange(len(results['R_profile']))
            
            # Plot raw profiles (faint) and smoothed (bold)
            ax.plot(x, results['R_profile'], 'r-', alpha=0.3, linewidth=1, label='Cu Raw')
            ax.plot(x, results['G_profile'], 'g-', alpha=0.3, linewidth=1, label='Ag Raw')
            ax.plot(x, results['R_net'], 'r-', alpha=0.9, linewidth=2.5, label='Cu Smoothed')
            ax.plot(x, results['G_net'], 'g-', alpha=0.9, linewidth=2.5, label='Ag Smoothed')
            
            # D_total boundaries (10% of max combined)
            if results['D_total_px'] > 0:
                ax.axvline(results['start_t'], color='blue', linestyle='--', linewidth=2)
                ax.axvline(results['end_t'], color='blue', linestyle='--', linewidth=2)
                ax.axvspan(results['start_t'], results['end_t'], color='blue', alpha=0.1,
                           label=f'D_total (10%) = {results["D_total_nm"]:.1f} nm')
            
            # D_core boundaries (FWHM)
            if results['D_core_px'] > 0:
                ax.axvline(results['start_c'], color='darkred', linestyle=':', linewidth=2)
                ax.axvline(results['end_c'], color='darkred', linestyle=':', linewidth=2)
                ax.axvspan(results['start_c'], results['end_c'], color='red', alpha=0.2,
                           label=f'D_core (FWHM) = {results["D_core_nm"]:.1f} nm')
            
            # Ag-dominant regions
            for r in results['ag_regions']:
                ax.axvspan(r['start_px'], r['end_px'], color='lime', alpha=0.25)
            
            ax.axhline(y=baseline + intensity_threshold, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Pixel Position")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"EDS FWHM Analysis — Cu:Ag = {molar_ratio} — δ = {results['delta_nm']:.2f} nm")
            ax.legend(loc='upper right', fontsize='small', ncol=2)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Export
            df = pd.DataFrame({
                'Pixel_X': x,
                'Cu_Raw': results['R_profile'],
                'Ag_Raw': results['G_profile'],
                'Cu_Smoothed': results['R_net'],
                'Ag_Smoothed': results['G_net'],
                'Is_Particle': ((results['R_net'] + results['G_net']) > intensity_threshold).astype(int),
                'Is_Core': (results['R_net'] > (np.max(results['R_net']) * 0.5)).astype(int) if np.max(results['R_net']) > 0 else np.zeros_like(x)
            })
            safe_ratio = molar_ratio.replace(':', '_').replace('?', 'unknown')
            st.download_button(
                label=f"📥 Download CSV for {molar_ratio}",
                data=df.to_csv(index=False),
                file_name=f'eds_{safe_ratio}_data.csv',
                mime='text/csv',
                key=f"download_{idx}"
            )
            
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
# BATCH SUMMARY
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
    st.subheader("📈 Shell Thickness vs. Cu:Ag Ratio (FWHM Method)")
    
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
    ax_cmp.set_title("Ag Shell Thickness Across All Cu:Ag Ratios (FWHM Method)")
    ax_cmp.axhline(y=3, color='blue', linestyle='--', alpha=0.5, label='Target: 3 nm')
    ax_cmp.axhline(y=5, color='blue', linestyle='--', alpha=0.5, label='Target: 5 nm')
    ax_cmp.legend()
    ax_cmp.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, deltas):
        ax_cmp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig_cmp)


# =============================================================================
# METHODOLOGY
# =============================================================================
with st.expander("🧠 Methodology: FWHM Geometric Measurement"):
    st.markdown("""
    ### The Problem with Previous Methods
    
    **Intensity Crossover** (`Ag > Cu`) fails for extreme ratios:
    - **1:1**: Ag dominant everywhere → measures particle radius, not shell
    - **5:1**: Ag too weak → misses shell entirely
    
    **Cu-dominant core** (`Cu > Ag`) fails due to beam broadening:
    - **3:1**: Cu tails extend across entire particle → D_core ≈ D_total → δ = 0
    - **5:1**: Noise spikes falsely register as Ag regions → δ inflated
    
    ### The FWHM Solution
    
    Shell thickness ($\\delta$) is calculated from robust signal widths:
    
    $$\\delta = \\frac{D_{total} - D_{core}}{2}$$
    
    Where:
    - **$D_{total}$**: Width of combined (Cu + Ag) signal at **10% of maximum height**.
      Captures full particle extent while ignoring distant noise tails.
    - **$D_{core}$**: **FWHM (50% max height)** of the Cu signal.
      Standard scientific method for EDS core sizing — ignores beam broadening tails.
    
    ### Gaussian Smoothing
    
    A Gaussian filter ($\\sigma = 2$) is applied to both Cu and Ag profiles before
    width measurement. This:
    - Suppresses high-frequency noise spikes
    - Reduces beam broadening artifacts
    - Preserves the underlying physical signal envelope
    
    ### Structure Classification
    
    | Condition | Classification | Typical Ratio |
    |-----------|----------------|---------------|
    | $D_{core} < 30\\%$ of $D_{total}$ | Homogeneous Ag (no core) | 1:1, 2:1 |
    | $\\delta < 1.5$ nm | Discontinuous / Ultra-thin shell | 5:1 |
    | Otherwise | Valid Core-Shell | 3:1 (~5 nm), 4:1 (~3-4 nm) |
    
    ### Expected Physical Trend
    
    | Ratio | Expected δ | Physical Interpretation |
    |-------|-----------|------------------------|
    | 1:1 | ~12 nm (particle radius) | Homogeneous Ag, no Cu core |
    | 2:1 | ~8-10 nm | Mostly Ag, small Cu core |
    | 3:1 | **~5 nm** | **Valid core-shell** |
    | 4:1 | **~3-4 nm** | **Valid core-shell** |
    | 5:1 | **~0.5-1.5 nm** | **Discontinuous / ultra-thin** |
    """)
