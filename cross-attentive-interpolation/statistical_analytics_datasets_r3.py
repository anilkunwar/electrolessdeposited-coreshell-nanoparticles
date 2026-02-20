#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Electroless Ag-Cu Deposition — Dataset Designer & Analyzer (ENHANCED N-DIM EDITION)
✅ FIXED: All st.color_picker widgets now use valid hex format (#RRGGBB)
✅ FIXED: st.slider type mismatch for marker_line_width (int bounds vs float step)
✓ 50+ colormap options with safe loading (rainbow, turbo, jet, inferno, viridis, etc.)
✓ Full font/typography controls (size, family, weight, color for titles/labels/ticks)
✓ Line/curve/marker thickness sliders for all visualizations
✓ N-dimensional hierarchical sunburst (tertiary, quaternary, + dimensions supported)
✓ Advanced design panel: grid, legend, hover, annotations, backgrounds
✓ All previous robustness fixes maintained (hashable checks, safe nunique, etc.)
✓ Hex-to-RGBA conversion helper for Plotly transparency support
✓ Migrated legacy RGBA colors to hex automatically on startup
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PAGE CONFIGURATION & STYLING
# =============================================
st.set_page_config(
    page_title="🧪 Deposition Dataset Designer Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ ENHANCED CSS with font variables and design tokens
st.markdown("""
<style>
:root {
    --primary-gradient: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
    --card-bg: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    --border-accent: #3B82F6;
    --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'Fira Code', 'Consolas', monospace;
}
.main-header {
    font-size: 2.8rem;
    color: #1E3A8A;
    text-align: center;
    padding: 1.2rem;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 1.8rem;
    font-family: var(--font-main);
    letter-spacing: -0.02em;
}
.section-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.4rem;
    margin: 0.6rem 0;
    border-left: 5px solid var(--border-accent);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    transition: transform 0.2s ease;
}
.section-card:hover {
    transform: translateY(-2px);
}
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stDataFrame { font-size: 0.92rem !important; }
div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 600; }
.design-control { margin: 0.4rem 0; }
.colormap-preview { 
    height: 24px; 
    border-radius: 4px; 
    margin: 2px 0;
    display: flex;
    border: 1px solid #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧪 Electroless Deposition Dataset Designer Pro</h1>', unsafe_allow_html=True)

# =============================================
# ✅ HELPER: HEX TO RGBA CONVERSION FOR PLOTLY
# =============================================
def hex_to_rgba(hex_color: str, alpha: float = 0.9) -> str:
    """
    Convert hex color (#RRGGBB or #RGB) to rgba format for Plotly.

    Args:
        hex_color: Hex color string (e.g., "#f8fafc" or "#fff")
        alpha: Opacity value (0.0 to 1.0)

    Returns:
        RGBA string for Plotly (e.g., "rgba(248, 250, 252, 0.9)")
    """
    hex_color = hex_color.lstrip('#')
    # Handle short hex format (#RGB -> #RRGGBB)
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    except (ValueError, IndexError):
        # Fallback to gray if conversion fails
        return f"rgba(100, 100, 100, {alpha})"

def rgba_to_hex(rgba_string: str) -> str:
    """
    Convert rgba string to hex format for st.color_picker.

    Args:
        rgba_string: RGBA string (e.g., "rgba(248, 250, 252, 0.9)")

    Returns:
        Hex color string (e.g., "#f8fafc")
    """
    try:
        # Extract rgb values from rgba(r, g, b, a)
        match = re.match(r'rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', rgba_string)
        if match:
            r, g, b = [int(x) for x in match.groups()]
            return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        pass
    return "#f8fafc"  # Default fallback

# =============================================
# ✅ EXPANDED COLOR MAP LIBRARY (50+ OPTIONS WITH SAFE LOADING)
# =============================================
def build_safe_colormap_library() -> Dict[str, list]:
    """Build colormap library with safe attribute access and fallbacks."""
    colormap_lib = {}

    # Plotly Sequential - Core (always available in Plotly >= 4.0)
    core_sequential = [
        "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
        "Turbo", "Rainbow", "Jet",
        "Blues", "Greens", "Greys", "Oranges", "Reds",
        "Peach", "Pinkyl", "Sunset", "Teal"
    ]

    # Plotly Sequential - Extended (may not exist in older Plotly versions)
    extended_sequential = [
        "Aggrnyl", "Agsunset", "Deep", "Dense", "Electric",
        "Emrld", "Haline", "Ice", "Matter", "Oryel",
        "Portland", "Solar", "Speed", "Temps", "Topo", "Turbid"
    ]

    # Plotly Diverging
    diverging = [
        "RdBu", "PiYG", "PRGn", "BrBG", "PuOr",
        "Balance", "Earth", "Geyser", "Tarn", "Delta", "Curl"
    ]

    # Plotly Qualitative
    qualitative = [
        "Set1", "Set2", "Set3", "Pastel1", "Pastel2",
        "Dark24", "Bold", "Prism", "Safe", "Vivid", "Alphabet"
    ]

    # Custom curated colormaps (always available - defined locally)
    custom_colormaps = {
        "Ocean": ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600"],
        "Forest": ["#2d5016", "#4a7c2e", "#6ba847", "#8fd464", "#b8f086", "#e0ffad"],
        "Sunset Glow": ["#ff6b6b", "#ffa500", "#ffd93d", "#6bcb77", "#4d96ff"],
        "Arctic": ["#e6f7ff", "#b3e0ff", "#80c9ff", "#4db3ff", "#1a9dff", "#007acc", "#0059b3"],
        "Volcano": ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
        "Nebula": ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"],
        "Cyberpunk": ["#00f5ff", "#ff00ff", "#ffff00", "#00ff00", "#ff0000"],
        "Pastel Dream": ["#ffd6e0", "#c7f0d8", "#fde4cf", "#c7ceea", "#f8e9a1"],
        "Monochrome": ["#000000", "#333333", "#666666", "#999999", "#cccccc", "#ffffff"],
        "Heat": ["#000080", "#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000", "#800000"],
        "Ice Fire": ["#0066cc", "#3399ff", "#66ccff", "#ffffff", "#ffcc66", "#ff9933", "#cc3300"],
        "Purple Haze": ["#1a0033", "#330066", "#660099", "#9933cc", "#cc66ff", "#ff99ff"],
        "Golden Hour": ["#1a1a2e", "#2d2d44", "#4a4a6a", "#8b7355", "#d4a574", "#f5deb3"],
        "Deep Sea": ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00"],
        "Autumn": ["#582f0e", "#7f4f24", "#936639", "#a68a64", "#b6ad90", "#c2c5aa"],
        "Spring": ["#606c38", "#283618", "#fefae0", "#dda15e", "#bc6c25"],
        "Midnight": ["#03045e", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8"],
        "Candy": ["#ffadad", "#ffd6a5", "#fdffb6", "#caffbf", "#9bf6ff", "#a0c4ff", "#bdb2ff"],
        "Retro": ["#ef476f", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"],
        "Neon": ["#ff00ff", "#00ffff", "#ffff00", "#ff0080", "#8000ff"],
        "Earth": ["#3d2b1f", "#5c4033", "#7f5539", "#9c6644", "#b87333", "#d4a574"],
        "Galaxy": ["#0f0c29", "#302b63", "#24243e", "#4a3f7f", "#7b68a8", "#a890d0"],
        "Mint": ["#00b894", "#00cec9", "#81ecec", "#a29bfe", "#dfe6e9"],
        "Coral": ["#ff7675", "#fd79a8", "#fdcb6e", "#e17055", "#d63031"],
        "Lavender": ["#6c5ce7", "#a29bfe", "#dfe6e9", "#b2bec3", "#636e72"],
        "Sunrise": ["#fab1a0", "#ff7675", "#e84393", "#6c5ce7", "#00cec9"],
        "Twilight": ["#2d3436", "#636e72", "#b2bec3", "#dfe6e9", "#fff"],
        "Berry": ["#8e44ad", "#9b59b6", "#bb8fce", "#d7bde2", "#ebdef0"],
        "Tropical": ["#00b894", "#00cec9", "#0984e3", "#6c5ce7", "#a29bfe"],
        "Desert": ["#e17055", "#d63031", "#fdcb6e", "#f39c12", "#e67e22"],
        "Arctic Ice": ["#74b9ff", "#0984e3", "#00cec9", "#81ecec", "#dff9fb"],
        "Forest Night": ["#00b894", "#00a085", "#008871", "#006f5c", "#005747"],
        "Urban": ["#2d3436", "#636e72", "#b2bec3", "#dfe6e9", "#fff"],
        "Pastel Sunset": ["#ff9ff3", "#feca57", "#ff6b6b", "#48dbfb", "#1dd1a1"],
        "Midnight Blue": ["#0c2461", "#1e3799", "#3c6382", "#5352ed", "#778beb"],
        "Cherry Blossom": ["#ff9ff3", "#ff6b6b", "#feca57", "#54a0ff", "#5f27cd"],
        "Ocean Breeze": ["#00d2d3", "#01a3a4", "#027b7b", "#035354", "#042c2c"],
        "Golden Sunset": ["#ff9f43", "#ee5a24", "#c23616", "#8c7ae6", "#5352ed"],
        "Mystic": ["#5f27cd", "#341f97", "#1e3799", "#0c2461", "#000"],
        "Spring Meadow": ["#00b894", "#00cec9", "#81ecec", "#74b9ff", "#a29bfe"],
        "Autumn Leaves": ["#e67e22", "#d35400", "#c0392b", "#96281b", "#7b241c"],
        "Winter Frost": ["#dff9fb", "#c7ecee", "#95afc0", "#535c68", "#2f3542"],
        "Summer Vibes": ["#ff9ff3", "#feca57", "#ff6b6b", "#48dbfb", "#1dd1a1"],
        "Cosmic": ["#2c2c54", "#40407a", "#706fd3", "#47478b", "#1e1e2e"],
        "Jungle": ["#2ed573", "#7bed9f", "#a3cb38", "#6ab04c", "#3c6382"],
        "Volcanic": ["#ff4757", "#ff6b81", "#ff793f", "#ffa502", "#ffda79"],
        "Aurora": ["#00d2d3", "#01a3a4", "#54a0ff", "#5f27cd", "#8e44ad"],
        "Savanna": ["#ff9f43", "#ee5a24", "#c23616", "#8c7ae6", "#5352ed"],
        "Glacier": ["#dff9fb", "#c7ecee", "#95afc0", "#535c68", "#2f3542"],
        "Crimson": ["#eb4d4b", "#e55039", "#c23616", "#8c7ae6", "#5352ed"],
        "Emerald": ["#00b894", "#00a085", "#008871", "#006f5c", "#005747"],
        "Sapphire": ["#0984e3", "#0062cc", "#0047ab", "#003399", "#001a66"],
        "Ruby": ["#eb4d4b", "#e55039", "#c23616", "#8c7ae6", "#5352ed"],
        "Amethyst": ["#8e44ad", "#9b59b6", "#bb8fce", "#d7bde2", "#ebdef0"],
        "Topaz": ["#f39c12", "#e67e22", "#d35400", "#c0392b", "#96281b"],
        "Turquoise": ["#40e0d0", "#48d1cc", "#20b2aa", "#008b8b", "#006666"],
        "Magenta": ["#ff00ff", "#ee00ee", "#dd00dd", "#cc00cc", "#bb00bb"],
        "Lime": ["#00ff00", "#00ee00", "#00dd00", "#00cc00", "#00bb00"],
        "Orange": ["#ffa500", "#ee9900", "#dd8800", "#cc7700", "#bb6600"],
        "Pink": ["#ffc0cb", "#ffb6d9", "#ffacd1", "#ffa2c9", "#ff98c1"],
        "Brown": ["#8b4513", "#7a3d10", "#69350d", "#582d0a", "#472507"],
        "Teal": ["#008080", "#007373", "#006666", "#005959", "#004d4d"],
        "Indigo": ["#4b0082", "#430075", "#3b0068", "#33005b", "#2b004e"],
        "Violet": ["#8f00ff", "#8000e6", "#7100cc", "#6200b3", "#530099"],
        "Chartreuse": ["#7fff00", "#73e600", "#66cc00", "#59b300", "#4d9900"],
        "Aquamarine": ["#7fffd4", "#73e6bf", "#66ccaa", "#59b395", "#4d9980"],
        "Coral Red": ["#ff6b6b", "#e66060", "#cc5555", "#b34a4a", "#994040"],
        "Steel Blue": ["#4682b4", "#3f75a1", "#38688e", "#315b7b", "#2a4e68"],
        "Gold": ["#ffd700", "#e6c200", "#ccac00", "#b39600", "#998000"],
        "Silver": ["#c0c0c0", "#adadad", "#999999", "#868686", "#737373"],
        "Bronze": ["#cd7f32", "#b9732e", "#a56729", "#915b24", "#7d4f1f"],
        "Copper": ["#b87333", "#a6672e", "#945b29", "#824f24", "#70431f"],
        "Rose Gold": ["#b76e79", "#a4626d", "#915661", "#7e4a55", "#6b3e49"],
    }

    # Safely load Plotly sequential colormaps
    for name in core_sequential:
        try:
            attr_name = name.capitalize() if name not in ["RdBu", "PiYG", "PRGn", "BrBG", "PuOr"] else name
            if hasattr(px.colors.sequential, attr_name):
                colormap_lib[name] = getattr(px.colors.sequential, attr_name)
            elif name.lower() in dir(px.colors.sequential):
                colormap_lib[name] = getattr(px.colors.sequential, name.lower())
        except Exception:
            pass

    # Safely load extended sequential (may not exist in all Plotly versions)
    for name in extended_sequential:
        try:
            if hasattr(px.colors.sequential, name):
                colormap_lib[name] = getattr(px.colors.sequential, name)
        except Exception:
            pass

    # Safely load diverging colormaps
    for name in diverging:
        try:
            if hasattr(px.colors.diverging, name):
                colormap_lib[name] = getattr(px.colors.diverging, name)
        except Exception:
            pass

    # Safely load qualitative colormaps
    for name in qualitative:
        try:
            if hasattr(px.colors.qualitative, name):
                colormap_lib[name] = getattr(px.colors.qualitative, name)
        except Exception:
            pass

    # Add custom colormaps (always available)
    colormap_lib.update(custom_colormaps)

    # Ensure we have at least one fallback
    if not colormap_lib:
        try:
            colormap_lib["Viridis"] = px.colors.sequential.Viridis
        except Exception:
            # Ultimate fallback: manually defined Viridis
            colormap_lib["Viridis"] = ["#440154", "#482777", "#3f4a8a", "#31678e", "#26838f", "#1f9e89", "#6cce5a", "#bade2e", "#fde725"]

    return colormap_lib

# Build the safe colormap library at module load
COLORMAP_LIBRARY = build_safe_colormap_library()

# Font families for typography controls
FONT_FAMILIES = [
    "Inter, sans-serif",
    "Roboto, sans-serif",
    "Open Sans, sans-serif",
    "Lato, sans-serif",
    "Montserrat, sans-serif",
    "Poppins, sans-serif",
    "Source Sans Pro, sans-serif",
    "Noto Sans, sans-serif",
    "Fira Code, monospace",
    "JetBrains Mono, monospace",
    "Consolas, monospace",
    "Courier New, monospace",
    "Georgia, serif",
    "Times New Roman, serif",
    "Merriweather, serif"
]

# =============================================
# GLOBAL CONSTANTS & CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

# Parameter categories for organized analysis
PARAM_CATEGORIES = {
    "🎯 Target Variables": ["thickness_nm", "ag_area_nm2", "growth_rate", "final_concentration"],
    "⚙️ Physics Parameters": ["gamma_nd", "beta_nd", "k0_nd", "M_nd", "D_nd", "alpha_nd"],
    "🔬 Geometry Parameters": ["core_radius_frac", "shell_thickness_frac", "L0_nm", "Nx"],
    "🧪 Process Parameters": ["c_bulk", "dt_nd", "n_steps", "tau0_s"],
    "⚡ EDL Catalyst": ["use_edl", "lambda0_edl", "tau_edl_nd", "alpha_edl"],
    "📊 Simulation Metadata": ["mode", "bc_type", "growth_model", "runtime_seconds"]
}

# Derived metrics that can be computed from snapshots
DERIVED_METRICS = {
    "thickness_nm": "Final Ag shell thickness in nanometers",
    "ag_area_nm2": "Ag phase area/volume in nm²/nm³",
    "cu_area_nm2": "Cu core area/volume in nm³",
    "growth_rate": "Average thickness growth rate (nm/s)",
    "final_concentration": "Mean Ag+ concentration at final step",
    "interface_sharpness": "Gradient magnitude at Ag/electrolyte interface",
    "edl_efficiency": "Integrated EDL boost effect over simulation",
    "convergence_metric": "L2 norm of field changes at final steps"
}

# Default radar colors with transparency (using rgba format for Plotly)
DEFAULT_RADAR_COLORS = [
    'rgba(31, 119, 180, 0.75)',
    'rgba(255, 127, 14, 0.75)',
    'rgba(44, 160, 44, 0.75)',
    'rgba(214, 39, 40, 0.75)',
    'rgba(148, 103, 189, 0.75)',
    'rgba(140, 86, 75, 0.75)',
    'rgba(227, 119, 194, 0.75)',
    'rgba(127, 127, 127, 0.75)',
    'rgba(23, 190, 207, 0.75)',
    'rgba(188, 189, 34, 0.75)'
]

# =============================================
# HELPER FUNCTIONS FOR SAFE DATA HANDLING
# =============================================
def is_hashable_column(series: pd.Series) -> bool:
    """Check if a column contains only hashable values (safe for nunique)."""
    try:
        series.dropna().apply(hash)
        return True
    except (TypeError, ValueError):
        return False

def get_safe_columns_for_nunique(df: pd.DataFrame) -> List[str]:
    """Get columns that are safe to call nunique() on."""
    safe_cols = []
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                safe_cols.append(c)
            elif pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
                if is_hashable_column(df[c]):
                    safe_cols.append(c)
        except Exception:
            continue
    return safe_cols

def get_numeric_columns_safe(df: pd.DataFrame) -> List[str]:
    """Get only numeric columns that are safe for correlation."""
    numeric_cols = []
    for c in df.select_dtypes(include=['number']).columns:
        try:
            if df[c].notna().any() and df[c].std() > 1e-10:
                numeric_cols.append(c)
        except Exception:
            continue
    return numeric_cols

def safe_nunique(series: pd.Series, max_unique: int = 10) -> bool:
    """Safely check if series has <= max_unique values."""
    try:
        if not is_hashable_column(series):
            return False
        return series.dropna().nunique() <= max_unique
    except Exception:
        return False

def clean_value_for_plotly(val) -> float:
    """Clean a value to be Plotly-compatible."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return 0.0
        return float(val)
    except Exception:
        return 0.0

def get_colormap(name: str, n_colors: int = None) -> list:
    """Retrieve colormap by name, optionally resampled to n_colors."""
    if name not in COLORMAP_LIBRARY:
        return COLORMAP_LIBRARY.get("Viridis", ["#440154", "#482777", "#3f4a8a", "#31678e", "#26838f", "#1f9e89", "#6cce5a", "#bade2e", "#fde725"])
    cmap = COLORMAP_LIBRARY[name]
    if n_colors and len(cmap) != n_colors:
        from plotly.colors import sample_colorscale
        return sample_colorscale(cmap, np.linspace(0, 1, n_colors))
    return cmap

# =============================================
# PKL FILE LOADER WITH ENHANCED METADATA
# =============================================
class EnhancedPKLLoader:
    """Advanced loader for electroless deposition PKL files with metadata extraction."""
    REQUIRED_KEYS = ['parameters', 'snapshots', 'thickness_history_nm']

    def __init__(self, pkl_dir: str = SOLUTIONS_DIR):
        self.pkl_dir = pkl_dir
        self.loaded_files = {}
        self.metadata_df = None

    def scan_directory(self) -> List[str]:
        """Find all PKL files in directory."""
        if not os.path.exists(self.pkl_dir):
            os.makedirs(self.pkl_dir, exist_ok=True)
            return []
        try:
            return [f for f in os.listdir(self.pkl_dir) if f.endswith('.pkl')]
        except Exception:
            return []

    def extract_filename_metadata(self, filename: str) -> Dict[str, Any]:
        """Parse parameters from standardized filename format."""
        metadata = {'filename': filename}
        patterns = {
            'c_bulk': r'_c([0-9.]+)_',
            'L0_nm': r'_L0([0-9.]+)nm',
            'core_radius_frac': r'_fc([0-9.]+)_',
            'shell_thickness_frac': r'_rs([0-9.]+)_',
            'k0_nd': r'_k([0-9.]+)_',
            'M_nd': r'_M([0-9.]+)_',
            'D_nd': r'_D([0-9.]+)_',
            'Nx': r'_Nx(\d+)_',
            'n_steps': r'_steps(\d+)\.',
            'lambda0_edl': r'EDL([0-9.]+)',
        }
        for key, pattern in patterns.items():
            try:
                match = re.search(pattern, filename)
                if match:
                    val = match.group(1)
                    metadata[key] = float(val) if '.' in val else int(val)
            except Exception:
                continue
        if '2D' in filename:
            metadata['mode'] = '2D (planar)'
        elif '3D' in filename:
            metadata['mode'] = '3D (spherical)'
        if 'Neu' in filename:
            metadata['bc_type'] = 'Neumann'
        elif 'Dir' in filename:
            metadata['bc_type'] = 'Dirichlet'
        metadata['use_edl'] = 'EDL' in filename and 'noEDL' not in filename
        metadata['growth_model'] = 'Model B' if 'ModelB' in filename else 'Model A'
        return metadata

    def compute_derived_metrics(self, data: Dict) -> Dict[str, float]:
        """Compute derived metrics from simulation snapshots."""
        metrics = {}
        try:
            thick_hist = data.get('thickness_history_nm', [])
            if thick_hist and len(thick_hist) >= 2:
                final_th = thick_hist[-1][2] * 1e9
                initial_th = thick_hist[0][2] * 1e9
                time_span = (thick_hist[-1][0] - thick_hist[0][0]) * data['parameters'].get('tau0_s', 1e-4)
                metrics['thickness_nm'] = final_th
                metrics['growth_rate'] = (final_th - initial_th) / max(time_span, 1e-12)

            if data.get('snapshots'):
                final_snap = data['snapshots'][-1]
                t_final, phi, c, psi = final_snap
                ag_mask = (phi > 0.5) & (psi <= 0.5)
                cu_mask = psi > 0.5
                L0 = data['parameters'].get('L0_nm', 20.0)
                dx = 1.0 / (phi.shape[0] - 1) * L0
                metrics['ag_area_nm2'] = np.sum(ag_mask) * dx**2
                metrics['cu_area_nm2'] = np.sum(cu_mask) * dx**2
                metrics['final_concentration'] = np.mean(c[~ag_mask & ~cu_mask]) if np.any(~ag_mask & ~cu_mask) else 0

                if phi.ndim == 2:
                    grad_phi = np.gradient(phi, dx)
                    interface_mask = (phi > 0.3) & (phi < 0.7)
                    if np.any(interface_mask):
                        metrics['interface_sharpness'] = np.mean(np.sqrt(grad_phi[0]**2 + grad_phi[1]**2)[interface_mask])

                if data['parameters'].get('use_edl', False):
                    lambda0 = data['parameters'].get('lambda0_edl', 0)
                    tau_edl = data['parameters'].get('tau_edl_nd', 0.05)
                    metrics['edl_efficiency'] = lambda0 * tau_edl

                if len(data.get('snapshots', [])) >= 2:
                    last_phi = data['snapshots'][-1][1]
                    prev_phi = data['snapshots'][-2][1]
                    metrics['convergence_metric'] = np.mean(np.abs(last_phi - prev_phi))
        except Exception as e:
            st.warning(f"Metric computation warning: {e}")

        for key in DERIVED_METRICS:
            metrics.setdefault(key, None)
        return metrics

    def load_file(self, filepath: str) -> Optional[Dict]:
        """Load and validate a single PKL file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if not any(key in data for key in self.REQUIRED_KEYS):
                st.warning(f"⚠️ Invalid structure in {os.path.basename(filepath)}")
                return None

            filename_meta = self.extract_filename_metadata(os.path.basename(filepath))
            params = data.get('parameters', {})
            meta = data.get('meta', {})

            record = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'loaded_at': datetime.now().isoformat(),
                **filename_meta,
                **{k: v for k, v in params.items() if k not in filename_meta},
                **{f"meta_{k}": v for k, v in meta.items()},
            }

            derived = self.compute_derived_metrics(data)
            record.update({f"metric_{k}": v for k, v in derived.items()})

            if data.get('snapshots'):
                record['n_snapshots'] = len(data['snapshots'])
                record['final_time_nd'] = data['snapshots'][-1][0]
                record['grid_shape'] = data['snapshots'][0][1].shape
            if data.get('diagnostics'):
                record['n_diagnostics'] = len(data['diagnostics'])

            return {'metadata': record, 'data': data, 'derived_metrics': derived}
        except Exception as e:
            st.error(f"❌ Error loading {os.path.basename(filepath)}: {e}")
            return None

    def load_all(self, max_files: int = None) -> pd.DataFrame:
        """Load all PKL files and return consolidated metadata DataFrame."""
        files = self.scan_directory()
        if max_files:
            files = files[:max_files]
        if not files:
            st.info(f"📁 No PKL files found in `{self.pkl_dir}`")
            return pd.DataFrame()

        records = []
        progress_bar = st.progress(0)
        for i, fname in enumerate(files):
            filepath = os.path.join(self.pkl_dir, fname)
            result = self.load_file(filepath)
            if result:
                records.append(result['metadata'])
            progress_bar.progress((i + 1) / len(files))

        if not records:
            st.warning("⚠️ No valid files could be loaded")
            return pd.DataFrame()

        self.metadata_df = pd.DataFrame(records)
        st.success(f"✅ Loaded {len(records)} simulation files")
        return self.metadata_df

# =============================================
# ✅ ENHANCED VISUALIZATION COMPONENTS
# =============================================

class DesignConfig:
    """Centralized design configuration for all visualizations."""
    def __init__(self):
        # Typography - ✅ ALL COLORS IN HEX FORMAT FOR st.color_picker
        self.title_font_family = "Inter, sans-serif"
        self.title_font_size = 20
        self.title_font_weight = "bold"
        self.title_font_color = "#1E3A8A"  # ✅ HEX
        self.label_font_family = "Inter, sans-serif"
        self.label_font_size = 12
        self.label_font_color = "#374151"  # ✅ HEX
        self.tick_font_size = 10
        self.tick_font_color = "#6B7280"  # ✅ HEX

        # Colors - ✅ ALL HEX FORMAT
        self.colormap_name = "Viridis"
        self.colormap_reversed = False
        self.bg_color = "#F8FAFC"  # ✅ HEX
        self.plot_bg_color = "#FFFFFF"  # ✅ HEX
        self.grid_color = "#CBD5E1"  # ✅ HEX

        # Lines & Markers
        self.line_width = 2.5
        self.marker_size = 8
        self.marker_symbol = "circle"
        self.marker_line_width = 1.0  # Initialize as float for consistency
        self.marker_line_color = "#FFFFFF"  # ✅ HEX

        # Layout
        self.show_grid = True
        self.legend_position = "bottom right"
        self.hover_mode = "closest"
        self.plot_height = 550

        # Advanced
        self.annotation_enabled = False
        self.annotation_text = ""
        self.annotation_pos = (0.5, 0.95)
        self.annotation_font_size = 11
        self.annotation_font_color = "#374151"  # ✅ HEX

    def get_font_config(self, element: str = "title") -> dict:
        """Return font configuration dictionary."""
        if element == "title":
            return dict(family=self.title_font_family, size=self.title_font_size, weight=self.title_font_weight, color=self.title_font_color)
        elif element == "label":
            return dict(family=self.label_font_family, size=self.label_font_size, color=self.label_font_color)
        else:
            return dict(family=self.label_font_family, size=self.tick_font_size, color=self.tick_font_color)

    def get_colormap(self, n_colors: int = None) -> list:
        """Get current colormap, optionally resampled."""
        cmap = get_colormap(self.colormap_name, n_colors)
        return cmap[::-1] if self.colormap_reversed else cmap

    def get_layout_updates(self) -> dict:
        """Generate Plotly layout updates from config."""
        return {
            "font": self.get_font_config("label"),
            "title_font": self.get_font_config("title"),
            "plot_bgcolor": hex_to_rgba(self.plot_bg_color, 0.95),
            "paper_bgcolor": hex_to_rgba(self.bg_color, 0.98),
            "hovermode": self.hover_mode,
            "height": self.plot_height,
            "legend": dict(
                orientation="h" if "bottom" in self.legend_position else "v",
                xanchor="right" if "right" in self.legend_position else "left",
                yanchor="bottom" if "bottom" in self.legend_position else "top",
                x=0.98 if "right" in self.legend_position else 0.02,
                y=0.02 if "bottom" in self.legend_position else 0.98,
                font=self.get_font_config("label")
            ),
            "xaxis": dict(
                showgrid=self.show_grid,
                gridcolor=self.grid_color,
                tickfont=self.get_font_config("tick")
            ),
            "yaxis": dict(
                showgrid=self.show_grid,
                gridcolor=self.grid_color,
                tickfont=self.get_font_config("tick")
            )
        }


class RadarChartBuilder:
    """Build interactive radar charts with enhanced design controls."""

    @staticmethod
    def create_comparison_radar(df: pd.DataFrame,
                                selected_params: List[str],
                                selected_indices: List[int],
                                design: DesignConfig,
                                normalize: bool = True) -> go.Figure:
        """Create radar chart with full design customization."""
        if len(selected_indices) == 0 or len(selected_params) == 0:
            fig = go.Figure()
            fig.add_annotation(text="Select parameters and simulations to compare",
                             font=design.get_font_config("label"))
            return fig

        radar_data = []
        for idx in selected_indices:
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            values = []
            for param in selected_params:
                col_name = f"metric_{param}" if param in DERIVED_METRICS else param
                val = row.get(col_name) or row.get(param) or 0
                values.append(clean_value_for_plotly(val))

            if normalize:
                valid_vals = [v for v in values if v is not None and pd.notna(v)]
                if valid_vals:
                    min_v, max_v = min(valid_vals), max(valid_vals)
                    if max_v > min_v + 1e-10:
                        values = [(v - min_v) / (max_v - min_v) for v in values]

            radar_data.append({
                'name': f"#{idx}: {row.get('filename', 'Unknown')[:30]}",
                'values': values,
                'c_bulk': clean_value_for_plotly(row.get('c_bulk', 0)),
                'thickness': clean_value_for_plotly(row.get('metric_thickness_nm', 0))
            })

        fig = go.Figure()
        colors = design.get_colormap(len(radar_data))

        for i, entry in enumerate(radar_data):
            # Convert hex to rgba for fill
            fill_color = colors[i % len(colors)]
            if fill_color.startswith('#'):
                fill_color = hex_to_rgba(fill_color, 0.7)
            elif not fill_color.startswith('rgba'):
                try:
                    rgb = px.colors.hex_to_rgb(fill_color)
                    fill_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)"
                except Exception:
                    fill_color = f"rgba(100, 100, 100, 0.7)"

            # Line color (full opacity)
            line_color = fill_color.replace('0.7', '1.0').replace('0.75', '1.0')

            fig.add_trace(go.Scatterpolar(
                r=entry['values'],
                theta=selected_params,
                fill='toself',
                name=entry['name'],
                line=dict(color=line_color, width=design.line_width + 0.5),
                fillcolor=fill_color,
                marker=dict(
                    size=design.marker_size,
                    symbol=design.marker_symbol,
                    line=dict(width=design.marker_line_width, color=design.marker_line_color)
                ),
                hovertemplate='<br>'.join([
                    '<b>%{theta}</b>: %{r:.3f}',
                    'c<sub>bulk</sub>: %{customdata[0]}',
                    'Thickness: %{customdata[1]:.2f} nm',
                    '<extra></extra>'
                ]),
                customdata=[[entry['c_bulk'], entry['thickness']]] * len(selected_params)
            ))

        layout_updates = design.get_layout_updates()
        fig.update_layout(
            **layout_updates,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1] if normalize else None,
                    tickfont=design.get_font_config("tick")
                ),
                bgcolor=hex_to_rgba(design.bg_color, 0.95)
            ),
            showlegend=True,
            margin=dict(l=50, r=50, t=60, b=50),
            title=dict(
                text="🎯 Parameter Space Comparison",
                x=0.5,
                font=design.get_font_config("title")
            )
        )
        return fig


class SunburstBuilder:
    """Build N-dimensional hierarchical sunburst charts."""

    @staticmethod
    def create_nd_hierarchy(df: pd.DataFrame,
                           dimensions: List[str],  # Can be 2, 3, 4, or more levels!
                           value_col: str,
                           design: DesignConfig,
                           aggregation: str = 'mean') -> go.Figure:
        """Create sunburst chart supporting N hierarchical dimensions."""
        # Validate dimensions exist
        valid_dims = [d for d in dimensions if d in df.columns]
        if len(valid_dims) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Select at least 2 valid dimensions",
                             font=design.get_font_config("label"))
            return fig

        # Prepare data with binning for continuous variables
        df_plot = df.copy()
        processed_dims = []

        for dim in valid_dims:
            if df_plot[dim].dtype in ['float64', 'float32'] and df_plot[dim].nunique() > 8:
                # Bin continuous variables
                try:
                    bins = min(5, df_plot[dim].nunique())
                    df_plot[f"{dim}_bin"] = pd.qcut(df_plot[dim], q=bins, labels=False, duplicates='drop')
                    df_plot[f"{dim}_label"] = df_plot[f"{dim}_bin"].apply(
                        lambda x: f"{dim}: {x}" if pd.notna(x) else "N/A")
                    processed_dims.append(f"{dim}_label")
                except Exception:
                    df_plot[f"{dim}_label"] = df_plot[dim].astype(str)
                    processed_dims.append(f"{dim}_label")
            else:
                df_plot[f"{dim}_label"] = df_plot[dim].astype(str)
                processed_dims.append(f"{dim}_label")

        # Determine value column
        value_col_actual = f"metric_{value_col}" if f"metric_{value_col}" in df_plot.columns else value_col
        if value_col_actual not in df_plot.columns:
            value_col_actual = df_plot.select_dtypes(include=[np.number]).columns[0]

        # Aggregate data
        agg_func = getattr(pd.Series, aggregation, 'mean')
        agg_data = df_plot.groupby(processed_dims)[value_col_actual].agg([aggregation, 'count']).reset_index()
        agg_data = agg_data[agg_data['count'] >= 1]

        # Build sunburst with N levels (supports tertiary, quaternary, etc.)
        fig = px.sunburst(
            agg_data,
            path=processed_dims,  # ✅ Supports arbitrary depth!
            values=aggregation,
            hover_data={'count': True, aggregation: ':.3f'},
            color=aggregation,
            color_continuous_scale=design.get_colormap(),
            title=f"🌟 {value_col} Hierarchy: {' → '.join(dimensions)}",
            height=design.plot_height + 100
        )

        # Apply design customizations
        fig.update_traces(
            hovertemplate='<br>'.join([
                '<b>%{label}</b>',
                f'{value_col}: %{{value:.3f}}',
                'Simulations: %{customdata[0]}',
                '<extra></extra>'
            ]),
            textinfo='label+percent parent',
            textfont=dict(family=design.label_font_family, size=design.label_font_size),
            marker=dict(line=dict(width=design.line_width * 0.5, color='white'))
        )

        layout_updates = design.get_layout_updates()
        fig.update_layout(
            **layout_updates,
            title=dict(
                text=f"🌟 {value_col} Hierarchy: {' → '.join(dimensions)}",
                x=0.5,
                font=design.get_font_config("title")
            ),
            margin=dict(t=50, l=0, r=0, b=0)
        )
        return fig

    @staticmethod
    def get_dimension_selector_options(df: pd.DataFrame, max_dimensions: int = 5) -> List[str]:
        """Get list of columns suitable for hierarchical dimensions."""
        safe_cols = get_safe_columns_for_nunique(df)
        candidates = []
        for c in safe_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                if df[c].nunique() <= 50:  # Reasonable for hierarchy
                    candidates.append(c)
            elif safe_nunique(df[c], max_unique=20):
                candidates.append(c)
        return candidates[:max_dimensions]


class SummaryTableBuilder:
    """Create interactive, filterable summary tables with design options."""

    @staticmethod
    def create_summary_table(df: pd.DataFrame,
                            target_var: str,
                            design: DesignConfig,
                            top_n: int = 20,
                            sort_by: str = None,
                            highlight_max: bool = True,
                            highlight_min: bool = True) -> pd.DataFrame:
        """Create styled summary table with design-aware formatting."""
        display_cols = ['filename', 'c_bulk', 'core_radius_frac', 'shell_thickness_frac',
                       'L0_nm', 'use_edl', 'mode', f'metric_{target_var}']
        display_cols = [c for c in display_cols if c in df.columns]

        if not display_cols:
            return pd.DataFrame()

        table_df = df[display_cols].copy()
        metric_col = f'metric_{target_var}'
        if metric_col in table_df.columns:
            table_df = table_df.rename(columns={metric_col: target_var})

        if sort_by and sort_by in table_df.columns:
            table_df = table_df.sort_values(sort_by, ascending=False)

        if target_var in table_df.columns and table_df[target_var].notna().any():
            table_df['rank'] = table_df[target_var].rank(pct=True)

        # Format numeric columns
        for col in table_df.select_dtypes(include=[np.number]).columns:
            if table_df[col].notna().any():
                if table_df[col].abs().max() > 1000 or table_df[col].abs().min() < 0.001:
                    table_df[col] = table_df[col].apply(lambda x: f'{x:.2e}' if pd.notna(x) else 'N/A')
                else:
                    table_df[col] = table_df[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')

        return table_df.head(top_n)

    @staticmethod
    def create_correlation_matrix(df: pd.DataFrame,
                                 params: List[str],
                                 design: DesignConfig) -> go.Figure:
        """Create heatmap of parameter correlations with design controls."""
        numeric_cols = []
        for p in params:
            col_name = f'metric_{p}' if p in DERIVED_METRICS else p
            if col_name in df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(df[col_name]) and df[col_name].notna().any():
                        numeric_cols.append(col_name)
                except Exception:
                    continue

        if len(numeric_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need ≥2 numeric parameters for correlation",
                             font=design.get_font_config("label"))
            return fig

        numeric_df = df[numeric_cols].dropna()
        if len(numeric_df) < 3:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data points (need ≥3)",
                             font=design.get_font_config("label"))
            return fig

        try:
            numeric_df = numeric_df.loc[:, numeric_df.std() > 1e-10]
        except Exception:
            fig = go.Figure()
            fig.add_annotation(text="Error computing variance", font=design.get_font_config("label"))
            return fig

        if len(numeric_df.columns) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need ≥2 columns with non-zero variance",
                             font=design.get_font_config("label"))
            return fig

        try:
            corr_matrix = numeric_df.corr()
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Correlation error: {str(e)}", font=design.get_font_config("label"))
            return fig

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale=design.get_colormap(),
            title='📊 Parameter Correlation Matrix',
            height=design.plot_height
        )

        fig.update_traces(
            textfont=dict(family=design.label_font_family, size=design.label_font_size - 1),
            colorbar=dict(title_font=design.get_font_config("label"), tickfont=design.get_font_config("tick"))
        )

        layout_updates = design.get_layout_updates()
        fig.update_layout(
            **layout_updates,
            title=dict(text='📊 Parameter Correlation Matrix', x=0.5, font=design.get_font_config("title")),
            xaxis_title='Parameters',
            yaxis_title='Parameters',
            coloraxis_colorbar=dict(title='Correlation', title_font=design.get_font_config("label"))
        )
        return fig


# =============================================
# DATASET IMPROVEMENT ANALYZER (Enhanced)
# =============================================
class DatasetImprovementAnalyzer:
    """Analyze dataset coverage and suggest improvements."""

    @staticmethod
    def detect_parameter_gaps(df: pd.DataFrame,
                            params: List[str],
                            n_bins: int = 5) -> Dict[str, List[str]]:
        """Identify under-sampled regions in parameter space."""
        gaps = {}
        for param in params:
            col = f'metric_{param}' if param in DERIVED_METRICS else param
            if col not in df.columns or df[col].isna().all():
                continue
            try:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if not is_hashable_column(df[col]):
                    continue
            except Exception:
                continue

            values = df[col].dropna()
            if len(values) < 10:
                gaps[param] = ["Insufficient data points"]
                continue

            try:
                bins = pd.qcut(values, q=n_bins, duplicates='drop', retbins=True)[1]
                counts, _ = np.histogram(values, bins=bins)
                under_sampled = []
                for i, (count, bin_start, bin_end) in enumerate(zip(counts, bins[:-1], bins[1:])):
                    if count < len(values) / (n_bins * 2):
                        under_sampled.append(f"{bin_start:.2f}-{bin_end:.2f} ({count} samples)")
                if under_sampled:
                    gaps[param] = under_sampled
            except Exception:
                gaps[param] = ["Could not bin data"]
        return gaps

    @staticmethod
    def generate_recommendations(df: pd.DataFrame,
                               target_var: str,
                               gaps: Dict[str, List[str]]) -> List[Dict]:
        """Generate actionable dataset improvement recommendations."""
        recommendations = []

        for param, regions in gaps.items():
            if regions and "Insufficient" not in regions[0]:
                recommendations.append({
                    'type': '🎯 Fill Parameter Gaps',
                    'priority': 'High',
                    'description': f"{param}: Add simulations in ranges: {', '.join(regions[:3])}",
                    'action': f"Run simulations with {param} in under-sampled ranges"
                })

        target_col = f'metric_{target_var}' if target_var in DERIVED_METRICS else target_var
        if target_col in df.columns and df[target_col].notna().any():
            numeric_cols = get_numeric_columns_safe(df)
            if target_col in numeric_cols and len(numeric_cols) > 1:
                try:
                    numeric_df = df[numeric_cols].dropna()
                    if len(numeric_df) > 10:
                        other_cols = [c for c in numeric_cols if c != target_col]
                        if other_cols:
                            correlations = numeric_df[target_col].corr(numeric_df[other_cols])
                            top_corr = correlations.dropna().abs().sort_values(ascending=False).head(3)
                            for param, corr_val in top_corr.items():
                                if corr_val > 0.3:
                                    recommendations.append({
                                        'type': '📈 Optimize for Target',
                                        'priority': 'Medium',
                                        'description': f"{param} strongly correlates with {target_var} (r={corr_val:.2f})",
                                        'action': f"Explore {param} range to maximize {target_var}"
                                    })
                except Exception:
                    pass

        categorical_cols = [c for c in ['mode', 'bc_type', 'use_edl', 'growth_model'] if c in df.columns]
        for col in categorical_cols:
            if col in df.columns:
                try:
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 1 and value_counts.min() < value_counts.max() * 0.3:
                        rare_values = value_counts[value_counts < value_counts.max() * 0.5].index.tolist()
                        if rare_values:
                            recommendations.append({
                                'type': '🔄 Increase Diversity',
                                'priority': 'Low',
                                'description': f"{col}: Under-represented values: {rare_values}",
                                'action': f"Add simulations with {col} = {rare_values[0]}"
                            })
                except Exception:
                    pass

        if 'Nx' in df.columns:
            try:
                if df['Nx'].max() < 256:
                    recommendations.append({
                        'type': '🔍 Increase Resolution',
                        'priority': 'Medium',
                        'description': f"Max grid resolution is {df['Nx'].max()}×{df['Nx'].max()}",
                        'action': "Run select simulations at Nx=512 for validation"
                    })
            except Exception:
                pass

        return recommendations


# =============================================
# ✅ DESIGN CONTROL PANEL COMPONENT (with safe color conversion)
# =============================================
def render_design_panel(design: DesignConfig, key_prefix: str = "main") -> DesignConfig:
    """Render interactive design controls in sidebar with safe color conversion."""
    # Helper to ensure color is hex
    def ensure_hex(color_val):
        if isinstance(color_val, str) and not color_val.startswith('#'):
            return rgba_to_hex(color_val)
        return color_val

    with st.expander("🎨 Visualization Design Controls", expanded=False):
        st.markdown("### Typography")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            design.title_font_family = st.selectbox(
                "Title Font", FONT_FAMILIES,
                index=FONT_FAMILIES.index(design.title_font_family) if design.title_font_family in FONT_FAMILIES else 0,
                key=f"{key_prefix}_title_font"
            )
            design.title_font_size = st.slider("Title Size", 14, 36, design.title_font_size, 1, key=f"{key_prefix}_title_size")
            design.title_font_weight = st.selectbox("Title Weight", ["normal", "bold", "bolder"],
                                                   index=["normal", "bold", "bolder"].index(design.title_font_weight),
                                                   key=f"{key_prefix}_title_weight")
            design.title_font_color = st.color_picker(
                "Title Color",
                ensure_hex(design.title_font_color),
                key=f"{key_prefix}_title_color"
            )
        with col_t2:
            design.label_font_family = st.selectbox(
                "Label Font", FONT_FAMILIES,
                index=FONT_FAMILIES.index(design.label_font_family) if design.label_font_family in FONT_FAMILIES else 0,
                key=f"{key_prefix}_label_font"
            )
            design.label_font_size = st.slider("Label Size", 8, 18, design.label_font_size, 1, key=f"{key_prefix}_label_size")
            design.tick_font_size = st.slider("Tick Size", 7, 14, design.tick_font_size, 1, key=f"{key_prefix}_tick_size")
            design.label_font_color = st.color_picker(
                "Label Color",
                ensure_hex(design.label_font_color),
                key=f"{key_prefix}_label_color"
            )

        st.markdown("### Color & Colormap")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            cmap_names = list(COLORMAP_LIBRARY.keys())
            selected_cmap = st.selectbox(
                "Colormap", cmap_names,
                index=cmap_names.index(design.colormap_name) if design.colormap_name in cmap_names else 0,
                key=f"{key_prefix}_colormap"
            )
            design.colormap_name = selected_cmap
            design.colormap_reversed = st.checkbox("Reverse Colormap", value=design.colormap_reversed, key=f"{key_prefix}_reverse")
        with col_c2:
            st.markdown("**Preview:**")
            preview_colors = get_colormap(design.colormap_name, 10)
            if design.colormap_reversed:
                preview_colors = preview_colors[::-1]
            preview_html = '<div style="display:flex;height:20px">'
            for c in preview_colors:
                preview_html += f'<div style="flex:1;background:{c};border-right:1px solid white"></div>'
            preview_html += '</div>'
            st.markdown(preview_html, unsafe_allow_html=True)
            design.bg_color = st.color_picker(
                "Background",
                ensure_hex(design.bg_color),
                key=f"{key_prefix}_bg"
            )
            design.plot_bg_color = st.color_picker(
                "Plot Background",
                ensure_hex(design.plot_bg_color),
                key=f"{key_prefix}_plot_bg"
            )

        st.markdown("### Lines & Markers")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            design.line_width = st.slider("Line Width", 0.5, 5.0, float(design.line_width), 0.1, key=f"{key_prefix}_line_width")
            design.marker_size = st.slider("Marker Size", 3, 15, int(design.marker_size), 1, key=f"{key_prefix}_marker_size")
        with col_l2:
            design.marker_symbol = st.selectbox(
                "Marker Symbol",
                ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "pentagon"],
                index=["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "pentagon"].index(design.marker_symbol),
                key=f"{key_prefix}_marker_symbol"
            )
            # ✅ FIX: Changed min/max to floats (0.0, 3.0) to match step (0.5) and value type
            design.marker_line_width = st.slider(
                "Marker Border", 0.0, 3.0, float(design.marker_line_width), 0.5,
                key=f"{key_prefix}_marker_border"
            )
            design.marker_line_color = st.color_picker(
                "Marker Border Color",
                ensure_hex(design.marker_line_color),
                key=f"{key_prefix}_marker_color"
            )

        st.markdown("### Layout")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            design.show_grid = st.checkbox("Show Grid", value=design.show_grid, key=f"{key_prefix}_grid")
            design.grid_color = st.color_picker(
                "Grid Color",
                ensure_hex(design.grid_color),
                key=f"{key_prefix}_grid_color"
            )
            design.plot_height = st.slider("Plot Height", 300, 800, int(design.plot_height), 50, key=f"{key_prefix}_height")
        with col_g2:
            design.legend_position = st.selectbox(
                "Legend Position",
                ["bottom right", "bottom left", "top right", "top left", "center"],
                index=["bottom right", "bottom left", "top right", "top left", "center"].index(design.legend_position),
                key=f"{key_prefix}_legend_pos"
            )
            design.hover_mode = st.selectbox(
                "Hover Mode",
                ["closest", "x", "y", "x unified", "y unified"],
                index=["closest", "x", "y", "x unified", "y unified"].index(design.hover_mode),
                key=f"{key_prefix}_hover"
            )

        st.markdown("### Annotations (Advanced)")
        design.annotation_enabled = st.checkbox("Add Annotation", value=design.annotation_enabled, key=f"{key_prefix}_annot_enable")
        if design.annotation_enabled:
            design.annotation_text = st.text_input("Annotation Text", design.annotation_text, key=f"{key_prefix}_annot_text")
            annot_x = st.slider("X Position %", 0, 100, int(design.annotation_pos[0]*100), 5, key=f"{key_prefix}_annot_x")
            annot_y = st.slider("Y Position %", 0, 100, int(design.annotation_pos[1]*100), 5, key=f"{key_prefix}_annot_y")
            design.annotation_pos = (annot_x/100, annot_y/100)
            design.annotation_font_size = st.slider("Annotation Font Size", 8, 20, int(design.annotation_font_size), 1, key=f"{key_prefix}_annot_size")
            design.annotation_font_color = st.color_picker(
                "Annotation Color",
                ensure_hex(design.annotation_font_color),
                key=f"{key_prefix}_annot_color"
            )

    return design


# =============================================
# MAIN STREAMLIT APPLICATION (with migration)
# =============================================
def main():
    # Initialize session state
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedPKLLoader(SOLUTIONS_DIR)
    if 'metadata_df' not in st.session_state:
        st.session_state.metadata_df = pd.DataFrame()
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = 'thickness_nm'
    if 'design_config' not in st.session_state:
        st.session_state.design_config = DesignConfig()
    else:
        # ✅ Migrate any legacy RGBA colors to hex (fixes st.color_picker errors)
        design = st.session_state.design_config
        # List of color attributes that must be hex
        color_attrs = [
            'title_font_color', 'label_font_color', 'bg_color', 'plot_bg_color',
            'marker_line_color', 'grid_color', 'annotation_font_color'
        ]
        for attr in color_attrs:
            old_val = getattr(design, attr, None)
            if old_val and isinstance(old_val, str) and not old_val.startswith('#'):
                setattr(design, attr, rgba_to_hex(old_val))
        # Ensure numeric values are numeric (for sliders)
        numeric_attrs = [
            'title_font_size', 'label_font_size', 'tick_font_size',
            'line_width', 'marker_size', 'marker_line_width', 'plot_height',
            'annotation_font_size'
        ]
        for attr in numeric_attrs:
            val = getattr(design, attr, None)
            if val is not None:
                try:
                    setattr(design, attr, float(val))
                except (ValueError, TypeError):
                    # fallback to default from DesignConfig
                    default_val = getattr(DesignConfig(), attr)
                    setattr(design, attr, default_val)
        st.session_state.design_config = design

    # ================= SIDEBAR CONFIGURATION =================
    with st.sidebar:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Data Management")
        pkl_dir = st.text_input("PKL Directory", value=SOLUTIONS_DIR, help="Directory containing .pkl simulation files")
        if st.button("🔄 Scan Directory", use_container_width=True):
            with st.spinner("Scanning for PKL files..."):
                st.session_state.loader.pkl_dir = pkl_dir
                st.session_state.metadata_df = st.session_state.loader.load_all()
                st.rerun()

        if not st.session_state.metadata_df.empty:
            st.success(f"✅ {len(st.session_state.metadata_df)} files loaded")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sims", len(st.session_state.metadata_df))
            with col2:
                edl_count = st.session_state.metadata_df.get('use_edl', pd.Series([False])).sum()
                st.metric("With EDL", int(edl_count))
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()

        # Target Variable Selection
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Target Variable")
        available_targets = []
        if not st.session_state.metadata_df.empty:
            available_targets = [k for k in DERIVED_METRICS.keys()
                               if f'metric_{k}' in st.session_state.metadata_df.columns
                               or k in st.session_state.metadata_df.columns]
        st.session_state.selected_target = st.selectbox(
            "Select output metric to analyze",
            available_targets if available_targets else list(DERIVED_METRICS.keys()),
            index=available_targets.index(st.session_state.selected_target) if st.session_state.selected_target in available_targets else 0,
            help=DERIVED_METRICS.get(st.session_state.selected_target, "No description available")
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()

        # ✅ RENDER DESIGN CONTROL PANEL
        st.session_state.design_config = render_design_panel(st.session_state.design_config)
        st.divider()

        # Analysis Controls
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Analysis Settings")
        normalize_radar = st.checkbox("Normalize radar chart values", value=True)
        top_n_table = st.slider("Top N simulations in table", 5, 50, 20, 5)

        available_corr_params = []
        if not st.session_state.metadata_df.empty:
            for cat_params in PARAM_CATEGORIES.values():
                for p in cat_params:
                    col_name = f'metric_{p}' if p in DERIVED_METRICS else p
                    if col_name in st.session_state.metadata_df.columns:
                        try:
                            if pd.api.types.is_numeric_dtype(st.session_state.metadata_df[col_name]):
                                available_corr_params.append(p)
                        except Exception:
                            continue

        default_corr = [p for p in ['c_bulk', 'core_radius_frac', 'k0_nd'] if p in available_corr_params]
        if available_corr_params:
            correlation_params = st.multiselect(
                "Parameters for correlation analysis",
                available_corr_params,
                default=default_corr if default_corr else available_corr_params[:3],
                help="Select numeric parameters to compute correlations"
            )
        else:
            correlation_params = st.multiselect(
                "Parameters for correlation analysis",
                ["No numeric parameters available"],
                default=[],
                help="Load data first to see available parameters",
                disabled=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= MAIN CONTENT AREA =================
    if st.session_state.metadata_df.empty:
        st.info("👈 Load PKL files from the sidebar to begin analysis")
        with st.expander("📋 Expected PKL File Structure"):
            st.code("""
{
    "parameters": {"c_bulk": 0.5, "core_radius_frac": 0.18, "L0_nm": 20.0, ...},
    "snapshots": [(t_nd, phi, c, psi), ...],
    "thickness_history_nm": [(t_nd, th_nd, th_nm, ...), ...],
    "diagnostics": [...]
}
            """, language='json')
        st.markdown("""
### 🚀 Quick Start
1. Place your `.pkl` simulation files in the `numerical_solutions` folder
2. Click "🔄 Scan Directory" in the sidebar
3. Select a target variable and customize visualizations in the Design Panel!
        """)
        return

    df = st.session_state.metadata_df
    target = st.session_state.selected_target
    design = st.session_state.design_config

    # ================= HEADER METRICS =================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        target_col = f'metric_{target}' if target in DERIVED_METRICS else target
        if target_col in df.columns and df[target_col].notna().any():
            st.metric(f"Avg {target}", f"{df[target_col].mean():.3f}", delta=f"{df[target_col].std():.3f} σ")
        else:
            st.metric(f"Avg {target}", "N/A")
    with col2:
        if 'c_bulk' in df.columns:
            st.metric("c_bulk Range", f"{df['c_bulk'].min():.2f}–{df['c_bulk'].max():.2f}")
        else:
            st.metric("c_bulk Range", "N/A")
    with col3:
        if 'L0_nm' in df.columns:
            st.metric("Domain Size", f"{df['L0_nm'].mean():.1f} nm")
        else:
            st.metric("Domain Size", "N/A")
    with col4:
        if 'use_edl' in df.columns:
            edl_pct = df['use_edl'].mean() * 100
            st.metric("EDL Usage", f"{edl_pct:.1f}%")
        else:
            st.metric("EDL Usage", "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

    # ================= TABS FOR DIFFERENT VISUALIZATIONS =================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Summary Table",
        "🕸️ Radar Comparison",
        "🌟 N-D Sunburst",
        "🔗 Correlations",
        "💡 Dataset Improvements"
    ])

    # ===== TAB 1: SUMMARY TABLE =====
    with tab1:
        st.markdown("### 📋 Simulation Summary Table")
        with st.expander("🔍 Filter & Sort Options", expanded=False):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                if 'c_bulk' in df.columns:
                    filter_c_bulk = st.checkbox("Filter by c_bulk")
                    if filter_c_bulk:
                        c_min, c_max = float(df['c_bulk'].min()), float(df['c_bulk'].max())
                        c_range = st.slider("c_bulk range", c_min, c_max, (c_min, c_max))
            with col_f2:
                if 'use_edl' in df.columns:
                    filter_edl = st.checkbox("Filter by EDL")
                    if filter_edl:
                        edl_filter = st.multiselect("EDL status", [True, False], default=[True, False])
            with col_f3:
                sort_options = [target] + [c for c in df.columns if 'metric' in c or c in ['c_bulk', 'L0_nm']]
                if sort_options:
                    sort_col = st.selectbox("Sort by", sort_options, index=0)
                    sort_asc = st.checkbox("Ascending", value=False)

        filtered_df = df.copy()
        if 'c_bulk' in df.columns and 'filter_c_bulk' in locals() and filter_c_bulk:
            filtered_df = filtered_df[(filtered_df['c_bulk'] >= c_range[0]) & (filtered_df['c_bulk'] <= c_range[1])]
        if 'use_edl' in df.columns and 'filter_edl' in locals() and filter_edl:
            filtered_df = filtered_df[filtered_df['use_edl'].isin(edl_filter)]

        table_builder = SummaryTableBuilder()
        summary_table = table_builder.create_summary_table(
            filtered_df, target_var=target, design=design,
            top_n=top_n_table,
            sort_by=sort_col if 'sort_col' in locals() and sort_col in filtered_df.columns else None
        )

        if not summary_table.empty:
            st.dataframe(
                summary_table.style
                .format(precision=3)
                .highlight_max(subset=[target] if target in summary_table.columns else None, color='#d1fae5')
                .highlight_min(subset=[target] if target in summary_table.columns else None, color='#fee2e2'),
                use_container_width=True,
                height=400
            )
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv = summary_table.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, f"summary_{target}_{datetime.now():%Y%m%d}.csv", "text/csv", use_container_width=True)
            with col_exp2:
                json_data = summary_table.to_json(orient='records', indent=2)
                st.download_button("📥 Download JSON", json_data, f"summary_{target}_{datetime.now():%Y%m%d}.json", "application/json", use_container_width=True)
        else:
            st.warning("No data matches current filters")

    # ===== TAB 2: RADAR CHART =====
    with tab2:
        st.markdown("### 🕸️ Multi-Parameter Radar Comparison")
        col_r1, col_r2 = st.columns([2, 1])
        with col_r2:
            st.markdown("**Select Parameters to Compare**")
            available_params = []
            for category, params in PARAM_CATEGORIES.items():
                valid_params = [p for p in params if p in df.columns or f'metric_{p}' in df.columns]
                if valid_params:
                    with st.expander(category, expanded=False):
                        for p in valid_params:
                            if st.checkbox(p, key=f"radar_{p}", value=p in ['c_bulk', 'core_radius_frac']):
                                available_params.append(p)
            st.markdown("**Select Simulations**")
            n_to_show = st.slider("Number of simulations to compare", 2, min(100, len(df)), 4, 1)
            if len(df) >= n_to_show:
                sample_cols = [c for c in ['c_bulk', 'core_radius_frac', 'use_edl'] if c in df.columns]
                sample_df = df[sample_cols].copy() if sample_cols else df.copy()
                sample_df['idx'] = range(len(df))
                selected_indices = sample_df.drop_duplicates().head(n_to_show)['idx'].tolist()
            else:
                selected_indices = list(range(len(df)))
            compare_btn = st.button("🔄 Generate Comparison", use_container_width=True)

        with col_r1:
            if available_params and compare_btn:
                radar_builder = RadarChartBuilder()
                try:
                    fig = radar_builder.create_comparison_radar(
                        df, selected_params=available_params[:8],
                        selected_indices=selected_indices,
                        design=design,
                        normalize=normalize_radar
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Radar chart error: {e}")
                with st.expander("💡 How to interpret this radar chart"):
                    st.markdown("""
- **Each polygon** represents one simulation with customized styling
- **Vertices** show normalized values for each parameter
- **Line thickness & marker style** controlled in Design Panel
- **Colormap** applied to distinguish simulations
- Hover for detailed values with custom template
                    """)
            elif not available_params:
                st.info("👈 Select at least one parameter from the sidebar")

    # ===== TAB 3: N-DIMENSIONAL SUNBURST =====
    with tab3:
        st.markdown("### 🌟 N-Dimensional Hierarchical Parameter Exploration")
        st.info("💡 Select 2-5 dimensions to build hierarchical sunburst charts (inner → outer rings)")

        # Get available dimensions
        available_dims = SunburstBuilder.get_dimension_selector_options(df)

        # Dimension selectors (support up to 5 levels for N-D hierarchy!)
        dim_selectors = []
        cols = st.columns(min(5, len(available_dims) + 1))
        for i in range(5):
            with cols[i]:
                label = ["1st (Inner)", "2nd", "3rd", "4th", "5th (Outer)"][i]
                default_idx = min(i, len(available_dims) - 1) if available_dims else 0
                selected = st.selectbox(
                    f"{label} Dimension",
                    available_dims if available_dims else ["No dimensions available"],
                    index=default_idx if available_dims else 0,
                    key=f"sunburst_dim_{i}",
                    disabled=i > 0 and (not dim_selectors or dim_selectors[-1] == "No dimensions available")
                )
                dim_selectors.append(selected)

        # Filter out empty/invalid selections and remove duplicates
        selected_dimensions = [d for d in dim_selectors if d and d != "No dimensions available"]
        selected_dimensions = list(dict.fromkeys(selected_dimensions))  # Remove duplicates, preserve order

        # Aggregation and value column
        col_agg, col_val = st.columns(2)
        with col_agg:
            agg_method = st.selectbox("Aggregation", ["mean", "median", "sum", "min", "max", "std"], index=0)
        with col_val:
            value_options = [k for k in DERIVED_METRICS.keys() if f'metric_{k}' in df.columns] or [c for c in df.columns if 'metric' in c]
            value_col = st.selectbox("Value to Display", value_options if value_options else list(df.columns), index=0)

        if len(selected_dimensions) >= 2 and value_col:
            sunburst_builder = SunburstBuilder()
            try:
                fig = sunburst_builder.create_nd_hierarchy(
                    df,
                    dimensions=selected_dimensions,  # Supports 2, 3, 4, or 5 dimensions!
                    value_col=value_col,
                    design=design,
                    aggregation=agg_method
                )
                st.plotly_chart(fig, use_container_width=True)

                # Insights panel
                with st.expander("📊 Key Insights from Hierarchy"):
                    if all(d in df.columns for d in selected_dimensions[:2]):
                        target_col = f'metric_{value_col}' if value_col in DERIVED_METRICS else value_col
                        if target_col in df.columns:
                            try:
                                group_cols = [f"{d}_label" if df[d].dtype in ['float64', 'float32'] and df[d].nunique() > 8 else d
                                            for d in selected_dimensions[:2]]
                                if all(c in df.columns or c.replace('_label', '') in df.columns for c in group_cols):
                                    insights = df.groupby(group_cols)[target_col].agg(['mean', 'std', 'count'])
                                    if not insights.empty and len(insights) > 0:
                                        best_idx = insights['mean'].idxmax()
                                        best_combo = insights.loc[best_idx]
                                        st.markdown(f"""
**Best performing combination for {value_col}:**
- {' / '.join([f"{selected_dimensions[i]}: `{best_idx[i] if isinstance(best_idx, tuple) else best_idx}`" for i in range(min(2, len(selected_dimensions)))])}
- Average {value_col}: **{best_combo['mean']:.3f}** ± {best_combo['std']:.3f}
- Based on {int(best_combo['count'])} simulations
                                        """)
                            except Exception as e:
                                st.info(f"Could not compute insights: {e}")
            except Exception as e:
                st.error(f"Sunburst chart error: {e}")
                st.code(str(e), language='python')
        elif len(selected_dimensions) < 2:
            st.info("👈 Select at least 2 dimensions to build the hierarchy")

    # ===== TAB 4: CORRELATIONS =====
    with tab4:
        st.markdown("### 🔗 Parameter Correlation Analysis")
        valid_corr_params = [p for p in correlation_params if p in available_corr_params] if 'available_corr_params' in locals() else []
        if valid_corr_params:
            corr_builder = SummaryTableBuilder()
            try:
                fig = corr_builder.create_correlation_matrix(df, valid_corr_params, design)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Correlation matrix error: {e}")

            with st.expander("🎯 Strong Correlations with Target"):
                target_col = f'metric_{target}' if target in DERIVED_METRICS else target
                if target_col in df.columns:
                    numeric_cols = get_numeric_columns_safe(df)
                    if target_col in numeric_cols and len(numeric_cols) > 1:
                        try:
                            numeric_df = df[numeric_cols].dropna()
                            if len(numeric_df) > 5:
                                other_cols = [c for c in numeric_cols if c != target_col]
                                if other_cols:
                                    corrs = numeric_df[target_col].corr(numeric_df[other_cols])
                                    strong = corrs.dropna().abs().sort_values(ascending=False)
                                    if len(strong) > 0:
                                        st.markdown("**Top correlated parameters:**")
                                        for param, val in strong.head(5).items():
                                            direction = "📈 positive" if val > 0 else "📉 negative"
                                            strength = "strong" if abs(val) > 0.7 else "moderate" if abs(val) > 0.4 else "weak"
                                            st.markdown(f"- `{param}`: {direction} correlation ({strength}, r={val:.2f})")
                                    else:
                                        st.info("No significant correlations found")
                        except Exception as e:
                            st.warning(f"Correlation analysis error: {e}")
        else:
            st.info("👈 Select valid numeric parameters in the sidebar for correlation analysis")

    # ===== TAB 5: DATASET IMPROVEMENTS =====
    with tab5:
        st.markdown("### 💡 Dataset Improvement Recommendations")
        with st.spinner("Analyzing parameter coverage..."):
            analyzer = DatasetImprovementAnalyzer()
            all_params = [p for cat in PARAM_CATEGORIES.values() for p in cat]
            gaps = analyzer.detect_parameter_gaps(df, all_params[:10])
            recommendations = analyzer.generate_recommendations(df, target, gaps)

            if recommendations:
                st.success(f"Generated {len(recommendations)} actionable recommendations")
                for i, rec in enumerate(recommendations):
                    priority_color = {'High': '#fecaca', 'Medium': '#fef3c7', 'Low': '#d1fae5'}
                    border_color = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
                    st.markdown(f"""
<div class="section-card" style="border-left-color: {border_color[rec['priority']]}">
<strong>{rec['type']}</strong> <span style="background:{priority_color[rec['priority']]}; padding:2px 8px; border-radius:12px; font-size:0.8em; font-weight:500">{rec['priority']} Priority</span><br>
{rec['description']}<br>
<em style="color:#475569">→ Action: {rec['action']}</em>
</div>
                    """, unsafe_allow_html=True)
            else:
                st.info("✅ Dataset appears well-covered! Consider exploring new parameter combinations or higher resolution.")

            # Parameter coverage visualization with design controls
            st.markdown("#### 📊 Current Parameter Coverage")
            coverage_cols = [c for c in ['c_bulk', 'core_radius_frac', 'shell_thickness_frac', 'L0_nm', 'k0_nd'] if c in df.columns]
            if coverage_cols:
                fig_cov = make_subplots(rows=1, cols=len(coverage_cols), subplot_titles=coverage_cols)
                cmap = design.get_colormap(len(coverage_cols))
                for i, col in enumerate(coverage_cols, 1):
                    try:
                        fig_cov.add_trace(
                            go.Histogram(x=df[col], nbinsx=12, name=col,
                                       marker_color=cmap[i-1] if i <= len(cmap) else cmap[0],
                                       marker_line_width=design.marker_line_width,
                                       marker_line_color='white'),
                            row=1, col=i
                        )
                    except Exception:
                        pass
                layout_updates = design.get_layout_updates()
                fig_cov.update_layout(
                    **layout_updates,
                    height=320, showlegend=False,
                    title_text="Parameter Distribution (more uniform = better coverage)",
                    bargap=0.15,
                    title=dict(text="Parameter Distribution", x=0.5, font=design.get_font_config("title"))
                )
                fig_cov.update_xaxes(title_text="Value", tickfont=design.get_font_config("tick"))
                fig_cov.update_yaxes(title_text="Count", tickfont=design.get_font_config("tick"))
                st.plotly_chart(fig_cov, use_container_width=True)

            # Export experimental design
            st.markdown("#### 🎯 Export Next Experimental Design")
            with st.expander("Generate parameter suggestions for new simulations"):
                if gaps:
                    st.markdown("**Suggested parameter combinations to fill gaps:**")
                    suggestion_df = []
                    for param, regions in list(gaps.items())[:3]:
                        if regions and "Insufficient" not in regions[0]:
                            try:
                                range_str = regions[0].split('(')[0].strip()
                                if '-' in range_str:
                                    low, high = map(float, range_str.split('-'))
                                    suggestion_df.append({
                                        'parameter': param,
                                        'suggested_value': (low + high) / 2,
                                        'rationale': f"Fill gap in {range_str}",
                                        'priority': 'High'
                                    })
                            except:
                                pass
                    if suggestion_df:
                        sugg_df = pd.DataFrame(suggestion_df)
                        st.dataframe(sugg_df, use_container_width=True)
                        csv_sugg = sugg_df.to_csv(index=False)
                        st.download_button("📥 Download Suggested Parameters", csv_sugg,
                                         f"suggested_params_{datetime.now():%Y%m%d}.csv", "text/csv")
                else:
                    st.info("No clear gaps detected. Consider:")
                    st.markdown("""
- Testing extreme parameter values (±2σ from mean)
- Exploring parameter interactions (e.g., high c_bulk + low k0_nd)
- Adding EDL catalyst variations if not already explored
- Increasing spatial resolution for validation cases (Nx=512)
                    """)

    # =============================================
    # FOOTER & HELP
    # =============================================
    st.divider()
    with st.expander("❓ Help & Documentation"):
        st.markdown(f"""
### 🧪 Dataset Designer Pro Guide

**🎨 Design Panel Features:**
- **50+ colormaps**: Viridis, Plasma, Inferno, Turbo, Rainbow, Jet, Cividis, and more
- **Typography controls**: Font family, size, weight, **color (hex)** for titles, labels, and ticks
- **Line/Marker styling**: Adjust width, size, symbol, and border for all plot elements
- **Layout options**: Grid visibility, legend positioning, hover modes, backgrounds

**🌟 N-Dimensional Sunburst:**
- Select 2-5 hierarchical dimensions (inner → outer rings)
- Supports both categorical and binned continuous parameters
- Custom aggregation: mean, median, sum, min, max, std
- Interactive drill-down with hover details

**📊 Visualizations:**
- 🕸️ **Radar**: Multi-parameter comparison with design-customized styling
- 🌟 **N-D Sunburst**: Hierarchical exploration with arbitrary depth
- 🔗 **Correlations**: Heatmap with typography and color controls
- 💡 **Improvements**: Data-driven experimental suggestions

**💾 Export Options:**
- Download filtered tables as CSV/JSON
- Export suggested parameter sets for new simulations
- All Plotly charts are interactive and exportable (PNG/SVG/HTML)

**🔧 Troubleshooting:**
- If visualizations fail, verify PKL files have expected structure
- Check that selected dimensions have sufficient unique values for hierarchy
- Use the Design Panel to adjust font sizes if labels appear clipped
        """)

    st.markdown(f"""
<div style="text-align: center; padding: 1.2rem; color: #64748b; font-size: 0.92rem; font-family: {design.label_font_family};">
🧪 Electroless Deposition Dataset Designer Pro v4.0 •
Built with Streamlit + Plotly + Pandas •
<em style="color:#3B82F6">Design smarter simulations with full visual control</em>
</div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
