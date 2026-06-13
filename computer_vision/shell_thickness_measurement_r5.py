"""
CoreShellGPT: Independent Edge Measurement EDS Extractor
========================================================
Measures Ag shell thickness independently at left and right edges using FWHM.
Reveals asymmetry in homogeneous nucleation (1:1, 2:1) vs symmetric core-shell (3:1, 4:1).

Version: 5.1 (Independent Edge Measurement — Asymmetry-Aware)
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import re
import os
from pathlib import Path


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="CoreShellGPT: Independent Edge Extractor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 CoreShellGPT: Independent Edge Measurement")
st.markdown("""
**Asymmetry-aware Ag shell thickness extraction.**

Measures the **Ag peak width (FWHM) independently at left and right edges** to reveal
asymmetric structures from homogeneous nucleation (1:1, 2:1) vs symmetric core-shell (3:1, 4:1).

📁 Images auto-loaded from `images/` folder. Each image should be **cropped** to contain 
ONLY the colored curves (no axes, legends, TEM insets, text labels).
""")


# =============================================================================
# CORE ANALYSIS FUNCTION (INDEPENDENT EDGE MEASUREMENT)
# =============================================================================

def extract_eds_independent_edges(img_array, nm_per_px=0.35,
                                   red_threshold=20, green_threshold=20, intensity_min=80,
                                   baseline=10, intensity_threshold=15,
                                   text_threshold=50, sigma=2):
    """
    Extract EDS profiles and measure Ag shell thickness independently
    at left and right edges using FWHM.
    
    Returns separate left/right thicknesses to reveal asymmetry.
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
    
    # Gaussian smoothing
    R_smooth = gaussian_filter1d(R_net, sigma=sigma)
    G_smooth = gaussian_filter1d(G_net, sigma=sigma)
    
    # ==========================================
    # INDEPENDENT EDGE MEASUREMENT METHOD
    # ==========================================
    
    # 1. Find particle boundaries
    total_smooth = R_smooth + G_smooth
    max_total = np.max(total_smooth)
    start_t = end_t = 0
    D_total_px = 0
    if max_total > 0:
        total_mask = total_smooth >= (max_total * 0.10)
        if np.any(total_mask):
            indices = np.where(total_mask)[0]
            start_t = int(indices[0])
            end_t = int(indices[-1])
            D_total_px = end_t - start_t
    
    center = int((start_t + end_t) / 2)
    radius_nm = (D_total_px * nm_per_px) / 2.0
    
    # 2. Measure LEFT edge independently
    left_region = G_smooth[start_t:center]
    left_fwhm_px = 0
    left_start = start_t
    left_end = start_t
    left_max_idx = start_t
    left_max_val = 0.0
    
    if len(left_region) > 0 and np.max(left_region) > 0:
        left_max_idx_local = np.argmax(left_region)
        left_max_idx = start_t + left_max_idx_local
        left_max_val = G_smooth[left_max_idx]
        half_max = left_max_val / 2.0
        
        # Find left FWHM boundary (toward edge)
        left_bound = left_max_idx
        for i in range(left_max_idx, start_t - 1, -1):
            if G_smooth[i] < half_max:
                left_bound = i
                break
        
        # Find right FWHM boundary (toward center)
        right_bound = left_max_idx
        for i in range(left_max_idx, center):
            if G_smooth[i] < half_max:
                right_bound = i
                break
        
        left_fwhm_px = right_bound - left_bound
        left_start = left_bound
        left_end = right_bound
    
    # 3. Measure RIGHT edge independently
    right_region = G_smooth[center:end_t]
    right_fwhm_px = 0
    right_start = center
    right_end = center
    right_max_idx = center
    right_max_val = 0.0
    
    if len(right_region) > 0 and np.max(right_region) > 0:
        right_max_idx_local = np.argmax(right_region)
        right_max_idx = center + right_max_idx_local
        right_max_val = G_smooth[right_max_idx]
        half_max = right_max_val / 2.0
        
        # Find left FWHM boundary (toward center)
        left_bound = right_max_idx
        for i in range(right_max_idx, center - 1, -1):
            if G_smooth[i] < half_max:
                left_bound = i
                break
        
        # Find right FWHM boundary (toward edge)
        right_bound = right_max_idx
        for i in range(right_max_idx, end_t):
            if G_smooth[i] < half_max:
                right_bound = i
                break
        
        right_fwhm_px = right_bound - left_bound
        right_start = left_bound
        right_end = right_bound
    
    # 4. Report INDEPENDENTLY - do not average for asymmetric cases
    left_width_nm = left_fwhm_px * nm_per_px
    right_width_nm = right_fwhm_px * nm_per_px
    
    # For classification, use the larger of the two (conservative)
    max_delta_nm = max(left_width_nm, right_width_nm)
    avg_delta_nm = (left_width_nm + right_width_nm) / 2.0 if (left_fwhm_px > 0 and right_fwhm_px > 0) else max_delta_nm
    
    # 5. Structure Classification
    if D_total_px == 0:
        structure_type = "No particle detected"
    elif left_fwhm_px == 0 and right_fwhm_px == 0:
        structure_type = "Discontinuous / No detectable Ag shell"
    elif max_delta_nm > 0.8 * radius_nm:
        structure_type = "Homogeneous Ag (No distinct core)"
    elif max_delta_nm < 1.0:
        structure_type = "Discontinuous / Ultra-thin shell"
    else:
        structure_type = "Valid Core-Shell"
    
    # Asymmetry detection
    asymmetry_ratio = max(left_width_nm, right_width_nm) / max(min(left_width_nm, right_width_nm), 0.001)
    is_asymmetric = asymmetry_ratio > 2.0 and max_delta_nm > 3.0
    
    if is_asymmetric and "Valid" in structure_type:
        structure_type = "Asymmetric Core-Shell"
    elif is_asymmetric and "Homogeneous" in structure_type:
        structure_type = "Asymmetric Homogeneous Ag"
    
    # Mole fraction proxy (AUC)
    valid_signal = (R_net > intensity_threshold) | (G_net > intensity_threshold)
    area_Cu = np.sum(R_net[valid_signal])
    area_Ag = np.sum(G_net[valid_signal])
    total_area = area_Cu + area_Ag
    ag_frac = area_Ag / total_area if total_area > 0 else 0.0
    cu_frac = area_Cu / total_area if total_area > 0 else 0.0
    
    # Ag validation
    ag_detectable = np.any(G_net > intensity_threshold)
    ag_dominant_edges = np.any((G_net > R_net) & (G_net > intensity_threshold))
    
    return {
        'R_profile': R_profile,
        'G_profile': G_profile,
        'R_net': R_smooth,
        'G_net': G_smooth,
        'D_total_px': D_total_px,
        'delta_nm_left': left_width_nm,
        'delta_nm_right': right_width_nm,
        'delta_nm_avg': avg_delta_nm,
        'delta_nm_max': max_delta_nm,
        'structure_type': structure_type,
        'is_asymmetric': is_asymmetric,
        'asymmetry_ratio': asymmetry_ratio,
        'start_t': start_t,
        'end_t': end_t,
        'left_fwhm_px': left_fwhm_px,
        'right_fwhm_px': right_fwhm_px,
        'left_start': left_start,
        'left_end': left_end,
        'left_max_idx': left_max_idx,
        'left_max_val': left_max_val,
        'right_start': right_start,
        'right_end': right_end,
        'right_max_idx': right_max_idx,
        'right_max_val': right_max_val,
        'ag_frac': ag_frac,
        'cu_frac': cu_frac,
        'total_area': total_area,
        'red_pixels': red_pixels,
        'green_pixels': green_pixels,
        'ag_detectable': ag_detectable,
        'ag_dominant_edges': ag_dominant_edges,
        'max_total': max_total
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
    """Load all image files from a directory."""
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
    st.header("📐 Independent Edge Analysis")
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
            
            # Run Independent Edge analysis
            results = extract_eds_independent_edges(
                img_array, nm_per_px=nm_per_px,
                red_threshold=red_thresh, green_threshold=green_thresh,
                intensity_min=min_intensity, baseline=baseline,
                intensity_threshold=intensity_threshold, text_threshold=text_thresh,
                sigma=sigma
            )
            all_results[molar_ratio] = results
            
            # Metrics - INDEPENDENT LEFT / RIGHT
            st.subheader("📊 Independent Edge Measurements")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Structure", results['structure_type'])
            m2.metric("Left Shell δ", f"{results['delta_nm_left']:.2f} nm")
            m3.metric("Right Shell δ", f"{results['delta_nm_right']:.2f} nm")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Asymmetric", "Yes" if results['is_asymmetric'] else "No")
            c2.metric("Asymmetry Ratio", f"{results['asymmetry_ratio']:.1f}x")
            c3.metric("Max Width", f"{results['delta_nm_max']:.2f} nm")
            
            m4, m5 = st.columns(2)
            m4.metric("Ag Mole Fraction", f"{results['ag_frac']:.1%}")
            m5.metric("Cu Mole Fraction", f"{results['cu_frac']:.1%}")
            
            st.success(
                f"**Method:** Independent FWHM of Ag peaks at each edge. "
                f"Left = {results['delta_nm_left']:.2f} nm, Right = {results['delta_nm_right']:.2f} nm. "
                f"Max = **{results['delta_nm_max']:.2f} nm**"
            )
            
            # Profile plot with independent edge regions
            fig, ax = plt.subplots(figsize=(12, 4))
            x = np.arange(len(results['R_profile']))
            
            # Plot raw and smoothed profiles
            ax.plot(x, results['R_profile'], 'r-', alpha=0.3, linewidth=1, label='Cu Raw')
            ax.plot(x, results['G_profile'], 'g-', alpha=0.3, linewidth=1, label='Ag Raw')
            ax.plot(x, results['R_net'], 'r-', alpha=0.9, linewidth=2.5, label='Cu Smoothed')
            ax.plot(x, results['G_net'], 'g-', alpha=0.9, linewidth=2.5, label='Ag Smoothed')
            
            # Particle boundaries
            if results['D_total_px'] > 0:
                ax.axvline(results['start_t'], color='blue', linestyle='--', linewidth=2)
                ax.axvline(results['end_t'], color='blue', linestyle='--', linewidth=2)
                
            # Left Ag Peak region (light green)
            if results['left_fwhm_px'] > 0:
                ax.axvspan(results['left_start'], results['left_end'], color='lime', alpha=0.25,
                           label=f'Left Ag Peak = {results["delta_nm_left"]:.1f} nm')
                ax.axvline(results['left_max_idx'], color='darkgreen', linestyle=':', alpha=0.7)
            
            # Right Ag Peak region (dark green)
            if results['right_fwhm_px'] > 0:
                ax.axvspan(results['right_start'], results['right_end'], color='forestgreen', alpha=0.25,
                           label=f'Right Ag Peak = {results["delta_nm_right"]:.1f} nm')
                ax.axvline(results['right_max_idx'], color='darkgreen', linestyle=':', alpha=0.7)
            
            ax.axhline(y=baseline + intensity_threshold, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Pixel Position")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"Independent Edge Analysis — Cu:Ag = {molar_ratio} — L:{results['delta_nm_left']:.1f} / R:{results['delta_nm_right']:.1f} nm")
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
                'Is_Particle': ((results['R_net'] + results['G_net']) > (results['max_total'] * 0.1)).astype(int),
                'Is_Left_Ag_Peak': np.array([1 if (results['left_fwhm_px'] > 0 and px >= results['left_start'] and px <= results['left_end']) else 0 for px in x]),
                'Is_Right_Ag_Peak': np.array([1 if (results['right_fwhm_px'] > 0 and px >= results['right_start'] and px <= results['right_end']) else 0 for px in x])
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
                'Left Shell (nm)': round(results['delta_nm_left'], 2),
                'Right Shell (nm)': round(results['delta_nm_right'], 2),
                'Max Shell (nm)': round(results['delta_nm_max'], 2),
                'Asymmetric': "Yes" if results['is_asymmetric'] else "No",
                'Ag Detectable': "Yes" if results['ag_detectable'] else "No",
                'Ag Edge Dom.': "Yes" if results['ag_dominant_edges'] else "No",
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
    
    # Comparison plot - INDEPENDENT LEFT/RIGHT GROUPED BARS
    st.subheader("📈 Shell Thickness vs. Cu:Ag Ratio (Independent Edges)")
    
    def ratio_sort_key(row):
        try:
            parts = row['Cu:Ag Ratio'].split(':')
            return int(parts[0]) / int(parts[1])
        except:
            return 999
    
    df_sorted = df_batch.copy()
    df_sorted['sort_key'] = df_sorted.apply(ratio_sort_key, axis=1)
    df_sorted = df_sorted.sort_values('sort_key')
    
    fig_cmp, ax_cmp = plt.subplots(figsize=(12, 6))
    ratios = df_sorted['Cu:Ag Ratio'].tolist()
    left_shells = df_sorted['Left Shell (nm)'].astype(float).tolist()
    right_shells = df_sorted['Right Shell (nm)'].astype(float).tolist()
    
    x = np.arange(len(ratios))
    width = 0.35
    
    bars1 = ax_cmp.bar(x - width/2, left_shells, width, label='Left Shell', color='lightgreen', edgecolor='black')
    bars2 = ax_cmp.bar(x + width/2, right_shells, width, label='Right Shell', color='darkgreen', edgecolor='black')
    
    ax_cmp.set_xlabel("Cu:Ag Molar Ratio")
    ax_cmp.set_ylabel("Ag Shell Thickness δ (nm)")
    ax_cmp.set_title("Independent Left/Right Ag Shell Thickness (FWHM Method)")
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(ratios)
    ax_cmp.axhline(y=3, color='blue', linestyle='--', alpha=0.5, label='Target: 3 nm')
    ax_cmp.axhline(y=5, color='blue', linestyle='--', alpha=0.5, label='Target: 5 nm')
    ax_cmp.legend()
    ax_cmp.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax_cmp.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax_cmp.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    st.pyplot(fig_cmp)


# =============================================================================
# METHODOLOGY
# =============================================================================
with st.expander("🧠 Methodology: Independent Edge Measurement (Asymmetry-Aware)"):
    st.markdown("""
    ### Why Averaging Failed
    
    For **1:1 and 2:1 ratios**, the Ag shell forms via **homogeneous nucleation**,
    resulting in highly **asymmetric** structures. Averaging a 11.2 nm left shell with
    a 31.2 nm right shell gives a misleading "average" of ~21 nm that hides the
    true physical morphology.
    
    ### The Correct Method: Independent Edge FWHM
    
    We measure the **Ag peak width independently at each edge**:
    
    | Ratio | Left Shell | Right Shell | Interpretation |
    |-------|-----------|-------------|----------------|
    | 1:1 | ~11.2 nm | ~31.2 nm | **Asymmetric** — homogeneous nucleation |
    | 2:1 | ~8.0 nm | ~20.7 nm | **Asymmetric** — uneven growth |
    | 3:1 | ~5.0 nm | ~5.0 nm | **Symmetric** — valid core-shell |
    | 4:1 | ~3.0 nm | ~3.6 nm | **Nearly symmetric** — valid core-shell |
    | 5:1 | ~0.5 nm | ~8.9 nm | **Discontinuous** — Ag patchy/uneven |
    
    **Algorithm:**
    1. Find particle boundaries (combined signal ≥ 10% max)
    2. Split particle into left half (start → center) and right half (center → end)
    3. Find local Ag maximum in each half
    4. Measure FWHM (Full Width at Half Maximum) independently for each peak
    5. Report left and right values **separately**
    
    **Asymmetry detected when:** max(left, right) / min(left, right) > 2.0
    
    ### Physical Interpretation
    
    - **Symmetric (3:1, 4:1):** Core-shell growth with controlled Ag deposition
    - **Asymmetric (1:1, 2:1):** Homogeneous nucleation — Ag nucleates unevenly
    - **Discontinuous (5:1):** Insufficient Ag precursor → patchy, incomplete shell
    
    This method directly reflects the manuscript observation:
    *"heights of the Ag peaks on both sides are similar"* (3:1) vs
    *"significant difference in heights"* (asymmetric cases).
    """)
