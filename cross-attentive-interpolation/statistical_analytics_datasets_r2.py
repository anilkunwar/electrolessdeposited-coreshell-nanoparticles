#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Electroless Ag-Cu Deposition — Dataset Designer & Analyzer (FULLY ROBUST EDITION)
✓ Fixed Plotly fillcolor validation error
✓ Fixed multiselect default value error
✓ Fixed nunique() crash on unhashable objects
✓ Fixed correlation calculation on invalid data
✓ Simple directory setup (numerical_solutions)
✓ Radar charts, sunburst hierarchies, summary tables
✓ Dataset gap detection & experimental design recommendations
✓ Export functionality for next-generation simulations
★ Enhanced visualizations: 50+ colormaps, adjustable fonts/line thickness
★ N‑dimensional sunburst (tertiary+ levels)
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
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PAGE CONFIGURATION & STYLING
# =============================================
st.set_page_config(
    page_title="🧪 Deposition Dataset Designer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visualization
st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem; 
        color: #1E3A8A; 
        text-align: center; 
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; 
        font-weight: 800; 
        margin-bottom: 1.5rem; 
    }
    .section-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .stDataFrame { font-size: 0.9rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧪 Electroless Deposition Dataset Designer</h1>', unsafe_allow_html=True)

# =============================================
# GLOBAL CONSTANTS & CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ✅ SIMPLE DIRECTORY PATTERN (like your working interpolation code)
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
    "cu_area_nm2": "Cu core area/volume in nm²/nm³", 
    "growth_rate": "Average thickness growth rate (nm/s)",
    "final_concentration": "Mean Ag+ concentration at final step",
    "interface_sharpness": "Gradient magnitude at Ag/electrolyte interface",
    "edl_efficiency": "Integrated EDL boost effect over simulation",
    "convergence_metric": "L2 norm of field changes at final steps"
}

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
            # Numeric columns are always safe
            if pd.api.types.is_numeric_dtype(df[c]):
                safe_cols.append(c)
            # Object/category columns need hashability check
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
            # Check column has valid numeric data
            if df[c].notna().any() and df[c].std() > 0:
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
        # Pattern matching for common parameter formats
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
        
        # Categorical parameters
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
            # Thickness metrics
            thick_hist = data.get('thickness_history_nm', [])
            if thick_hist and len(thick_hist) >= 2:
                final_th = thick_hist[-1][2] * 1e9  # Convert to nm
                initial_th = thick_hist[0][2] * 1e9
                time_span = (thick_hist[-1][0] - thick_hist[0][0]) * data['parameters'].get('tau0_s', 1e-4)
                metrics['thickness_nm'] = final_th
                metrics['growth_rate'] = (final_th - initial_th) / max(time_span, 1e-12)
            
            # Phase statistics from final snapshot
            if data.get('snapshots'):
                final_snap = data['snapshots'][-1]
                t_final, phi, c, psi = final_snap
                # Simple phase area calculation (2D assumption)
                ag_mask = (phi > 0.5) & (psi <= 0.5)
                cu_mask = psi > 0.5
                L0 = data['parameters'].get('L0_nm', 20.0)
                dx = 1.0 / (phi.shape[0] - 1) * L0  # nm per pixel
                metrics['ag_area_nm2'] = np.sum(ag_mask) * dx**2
                metrics['cu_area_nm2'] = np.sum(cu_mask) * dx**2
                metrics['final_concentration'] = np.mean(c[~ag_mask & ~cu_mask]) if np.any(~ag_mask & ~cu_mask) else 0
                
                # Interface sharpness (gradient magnitude)
                if phi.ndim == 2:
                    grad_phi = np.gradient(phi, dx)
                    interface_mask = (phi > 0.3) & (phi < 0.7)
                    if np.any(interface_mask):
                        metrics['interface_sharpness'] = np.mean(np.sqrt(grad_phi[0]**2 + grad_phi[1]**2)[interface_mask])
            
            # EDL efficiency
            if data['parameters'].get('use_edl', False):
                lambda0 = data['parameters'].get('lambda0_edl', 0)
                tau_edl = data['parameters'].get('tau_edl_nd', 0.05)
                # Simplified: integrated boost factor
                metrics['edl_efficiency'] = lambda0 * tau_edl
            
            # Convergence metric (change in fields at end)
            if len(data.get('snapshots', [])) >= 2:
                last_phi = data['snapshots'][-1][1]
                prev_phi = data['snapshots'][-2][1]
                metrics['convergence_metric'] = np.mean(np.abs(last_phi - prev_phi))
                
        except Exception as e:
            st.warning(f"Metric computation warning: {e}")
            # Set defaults for missing metrics
            for key in DERIVED_METRICS:
                metrics.setdefault(key, None)
        return metrics
    
    def load_file(self, filepath: str) -> Optional[Dict]:
        """Load and validate a single PKL file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Validate structure
            if not any(key in data for key in self.REQUIRED_KEYS):
                st.warning(f"⚠️ Invalid structure in {os.path.basename(filepath)}")
                return None
            
            # Merge metadata from filename and file contents
            filename_meta = self.extract_filename_metadata(os.path.basename(filepath))
            
            # Extract parameters with fallbacks
            params = data.get('parameters', {})
            meta = data.get('meta', {})
            
            # Build comprehensive metadata record
            record = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'loaded_at': datetime.now().isoformat(),
                **filename_meta,  # Filename-parsed params
                **{k: v for k, v in params.items() if k not in filename_meta},  # File params
                **{f"meta_{k}": v for k, v in meta.items()},  # Meta prefix
            }
            
            # Compute derived metrics
            derived = self.compute_derived_metrics(data)
            record.update({f"metric_{k}": v for k, v in derived.items()})
            
            # Store lightweight snapshot summary (not full arrays)
            if data.get('snapshots'):
                record['n_snapshots'] = len(data['snapshots'])
                record['final_time_nd'] = data['snapshots'][-1][0]
                record['grid_shape'] = data['snapshots'][0][1].shape
            
            # Store diagnostics summary
            if data.get('diagnostics'):
                record['n_diagnostics'] = len(data['diagnostics'])
            
            return {
                'metadata': record,
                'data': data,  # Full data for detailed analysis
                'derived_metrics': derived
            }
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
# VISUALIZATION COMPONENTS (Enhanced)
# =============================================

class RadarChartBuilder:
    """Build interactive radar charts for multi-parameter comparison."""
    
    @staticmethod
    def create_comparison_radar(df: pd.DataFrame, 
                               selected_params: List[str],
                               selected_indices: List[int],
                               normalize: bool = True,
                               line_width: float = 2.0,
                               font_size: int = 12,
                               font_family: str = "Arial") -> go.Figure:
        """Create radar chart comparing selected simulations across parameters."""
        if len(selected_indices) == 0 or len(selected_params) == 0:
            return go.Figure().add_annotation(text="Select parameters and simulations to compare")
        
        # Extract and optionally normalize values
        radar_data = []
        for idx in selected_indices:
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            values = []
            for param in selected_params:
                # Try derived metrics first, then direct columns
                col_name = f"metric_{param}" if param in DERIVED_METRICS else param
                val = row.get(col_name) or row.get(param) or 0
                # ✅ FIXED: Clean value for Plotly compatibility
                values.append(clean_value_for_plotly(val))
            
            # Normalize to [0, 1] if requested
            if normalize:
                valid_vals = [v for v in values if v is not None and pd.notna(v)]
                if valid_vals:
                    min_v, max_v = min(valid_vals), max(valid_vals)
                    if max_v > min_v:
                        values = [(v - min_v) / (max_v - min_v) for v in values]
            
            radar_data.append({
                'name': f"#{idx}: {row.get('filename', 'Unknown')[:30]}",
                'values': values,
                'c_bulk': clean_value_for_plotly(row.get('c_bulk', 0)),
                'thickness': clean_value_for_plotly(row.get('metric_thickness_nm', 0))
            })
        
        # Use a fixed set of rgba colors (or could be made configurable)
        radar_colors = [
            'rgba(31, 119, 180, 0.7)',
            'rgba(255, 127, 14, 0.7)',
            'rgba(44, 160, 44, 0.7)',
            'rgba(214, 39, 40, 0.7)',
            'rgba(148, 103, 189, 0.7)',
            'rgba(140, 86, 75, 0.7)',
            'rgba(227, 119, 194, 0.7)',
            'rgba(127, 127, 127, 0.7)'
        ]
        
        # Build Plotly figure
        fig = go.Figure()
        for i, entry in enumerate(radar_data):
            fillcolor = radar_colors[i % len(radar_colors)]
            linecolor = fillcolor.replace('0.7', '1.0')  # Full opacity for line
            
            fig.add_trace(go.Scatterpolar(
                r=entry['values'],
                theta=selected_params,
                fill='toself',
                name=entry['name'],
                line=dict(color=linecolor, width=line_width),
                fillcolor=fillcolor,
                hovertemplate='<br>'.join([
                    '%{theta}: %{r:.3f}',
                    'c_bulk: ' + str(entry['c_bulk']),
                    'Thickness: %{customdata:.2f} nm',
                    '<extra></extra>'
                ]),
                customdata=[entry['thickness']] * len(selected_params)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1] if normalize else None),
                bgcolor='rgba(248, 250, 252, 0.8)'
            ),
            showlegend=True,
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
            title=dict(text="🎯 Parameter Space Comparison", x=0.5),
            font=dict(size=font_size, family=font_family)
        )
        return fig


class SunburstBuilder:
    """Build hierarchical sunburst charts for parameter exploration (N‑dimensional)."""
    
    @staticmethod
    def create_parameter_hierarchy(df: pd.DataFrame,
                                  path_cols: List[str],
                                  value_col: str,
                                  colorscale: str = 'Viridis',
                                  font_size: int = 12,
                                  font_family: str = "Arial") -> go.Figure:
        """Create sunburst chart showing N‑level parameter relationships."""
        if not path_cols:
            return go.Figure().add_annotation(text="Select at least one dimension for the hierarchy")
        
        # Create binned categories for continuous variables
        df_plot = df.copy()
        binned_cols = []
        for col in path_cols:
            if col not in df_plot.columns:
                continue
            if pd.api.types.is_numeric_dtype(df_plot[col]) and df_plot[col].nunique() > 5:
                # Bin continuous variable
                try:
                    df_plot[f"{col}_bin"] = pd.qcut(df_plot[col], q=4, labels=False, duplicates='drop').astype(str)
                    binned_cols.append(f"{col}_bin")
                except Exception:
                    # If qcut fails (e.g., not enough unique), fallback to string
                    df_plot[f"{col}_bin"] = df_plot[col].astype(str)
                    binned_cols.append(f"{col}_bin")
            else:
                # Already categorical or low cardinality
                df_plot[f"{col}_bin"] = df_plot[col].astype(str)
                binned_cols.append(f"{col}_bin")
        
        # Use the binned columns for the path
        path = binned_cols
        
        value_col_actual = f"metric_{value_col}" if f"metric_{value_col}" in df_plot.columns else value_col
        if value_col_actual not in df_plot.columns:
            value_col_actual = df_plot.columns[0]  # Fallback
        
        # Aggregate: group by all path levels and compute mean of value
        # We'll use pandas groupby then pass to px.sunburst via custom aggregation
        # But px.sunburst can aggregate if we pass the whole dataframe with path and values
        # However, to avoid huge number of rows, we can pre-aggregate, but px.sunburst can handle raw data.
        # Let's pass the raw df and let px.sunburst aggregate (by default it sums, but we want mean)
        # We can use `values` column and `aggfunc`? Actually px.sunburst has `values` and `color` but aggregation is sum by default.
        # To get mean, we need to pre-aggregate.
        # So we'll pre-aggregate using groupby + mean.
        
        # Group by all path columns, compute mean and count
        agg_df = df_plot.groupby(path)[value_col_actual].agg(['mean', 'count']).reset_index()
        agg_df.columns = path + ['mean', 'count']
        
        # Remove groups with insufficient data (optional)
        agg_df = agg_df[agg_df['count'] >= 1]
        
        if agg_df.empty:
            return go.Figure().add_annotation(text="No data after aggregation")
        
        # Create sunburst
        fig = px.sunburst(
            agg_df,
            path=path,
            values='mean',
            color='mean',
            color_continuous_scale=colorscale,
            title=f"🌟 {value_col} by {' → '.join(path_cols)}",
            height=600
        )
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>' +
                         f'{value_col}: %{{value:.3f}}<br>' +
                         'Simulations: %{count}<extra></extra>',
            textinfo='label+percent parent'
        )
        fig.update_layout(
            font=dict(size=font_size, family=font_family)
        )
        return fig


class SummaryTableBuilder:
    """Create interactive, filterable summary tables."""
    
    @staticmethod
    def create_summary_table(df: pd.DataFrame,
                           target_var: str,
                           top_n: int = 20,
                           sort_by: str = None) -> pd.DataFrame:
        """Create styled summary table with key metrics."""
        # Select relevant columns
        display_cols = ['filename', 'c_bulk', 'core_radius_frac', 'shell_thickness_frac', 
                       'L0_nm', 'use_edl', 'mode', f'metric_{target_var}']
        display_cols = [c for c in display_cols if c in df.columns]
        
        if not display_cols:
            return pd.DataFrame()
        
        table_df = df[display_cols].copy()
        
        # Rename metric column for display
        metric_col = f'metric_{target_var}'
        if metric_col in table_df.columns:
            table_df = table_df.rename(columns={metric_col: target_var})
        
        # Sort by target variable if specified
        if sort_by and sort_by in table_df.columns:
            table_df = table_df.sort_values(sort_by, ascending=False)
        
        # Add ranking
        if target_var in table_df.columns and table_df[target_var].notna().any():
            table_df['rank'] = table_df[target_var].rank(pct=True)
        
        # Format numeric columns
        for col in table_df.select_dtypes(include=[np.number]).columns:
            if table_df[col].notna().any():
                if table_df[col].max() > 1000 or table_df[col].min() < 0.001:
                    table_df[col] = table_df[col].apply(lambda x: f'{x:.2e}' if pd.notna(x) else 'N/A')
                else:
                    table_df[col] = table_df[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
        
        return table_df.head(top_n)
    
    @staticmethod
    def create_correlation_matrix(df: pd.DataFrame, 
                                 params: List[str],
                                 colorscale: str = 'RdBu',
                                 font_size: int = 12,
                                 font_family: str = "Arial") -> go.Figure:
        """Create heatmap of parameter correlations."""
        # ✅ SAFE: Select only numeric columns that exist and have valid data
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
            return go.Figure().add_annotation(text="Need ≥2 numeric parameters for correlation")
        
        # ✅ SAFE: Create clean numeric dataframe
        numeric_df = df[numeric_cols].dropna()
        
        if len(numeric_df) < 3:
            return go.Figure().add_annotation(text="Insufficient data points for correlation (need ≥3)")
        
        # ✅ SAFE: Check for zero variance columns
        try:
            numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
        except Exception:
            return go.Figure().add_annotation(text="Error computing variance")
        
        if len(numeric_df.columns) < 2:
            return go.Figure().add_annotation(text="Need ≥2 columns with non-zero variance")
        
        # Compute correlation
        try:
            corr_matrix = numeric_df.corr()
        except Exception as e:
            return go.Figure().add_annotation(text=f"Correlation error: {str(e)}")
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale=colorscale,
            title='📊 Parameter Correlation Matrix',
            height=500
        )
        fig.update_layout(
            xaxis_title='Parameters',
            yaxis_title='Parameters',
            coloraxis_colorbar=dict(title='Correlation'),
            font=dict(size=font_size, family=font_family)
        )
        return fig

# =============================================
# DATASET IMPROVEMENT ANALYZER (unchanged)
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
            
            # ✅ SAFE: Check column is numeric and hashable
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
            
            # Create bins and count samples
            try:
                bins = pd.qcut(values, q=n_bins, duplicates='drop', retbins=True)[1]
                counts, _ = np.histogram(values, bins=bins)
                
                # Identify bins with few samples
                under_sampled = []
                for i, (count, bin_start, bin_end) in enumerate(zip(counts, bins[:-1], bins[1:])):
                    if count < len(values) / (n_bins * 2):  # Less than half expected
                        under_sampled.append(f"{bin_start:.2f}-{bin_end:.2f} ({count} samples)")
                
                if under_sampled:
                    gaps[param] = under_sampled
                    
            except Exception:
                gaps[param] = ["Could not bin data (check for constant values)"]
        
        return gaps
    
    @staticmethod
    def generate_recommendations(df: pd.DataFrame,
                               target_var: str,
                               gaps: Dict[str, List[str]]) -> List[Dict]:
        """Generate actionable dataset improvement recommendations."""
        recommendations = []
        
        # 1. Gap-filling recommendations
        for param, regions in gaps.items():
            if regions and "Insufficient" not in regions[0]:
                recommendations.append({
                    'type': '🎯 Fill Parameter Gaps',
                    'priority': 'High',
                    'description': f"{param}: Add simulations in ranges: {', '.join(regions[:3])}",
                    'action': f"Run simulations with {param} in under-sampled ranges"
                })
        
        # 2. Target variable optimization
        target_col = f'metric_{target_var}' if target_var in DERIVED_METRICS else target_var
        if target_col in df.columns and df[target_col].notna().any():
            # ✅ SAFE: Get only valid numeric columns
            numeric_cols = get_numeric_columns_safe(df)
            if target_col in numeric_cols and len(numeric_cols) > 1:
                try:
                    numeric_df = df[numeric_cols].dropna()
                    if len(numeric_df) > 10:
                        correlations = numeric_df[target_col].corr(numeric_df.drop(columns=[target_col], errors='ignore'))
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
        
        # 3. Diversity recommendations
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
        
        # 4. Resolution recommendations
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
# MAIN STREAMLIT APPLICATION
# =============================================

def get_all_colorscales() -> Dict[str, List[str]]:
    """Return a dictionary of available color scales (names and scales)."""
    # Use plotly's built-in color scales
    scales = {}
    # Sequential
    for name in px.colors.sequential.__dict__:
        if not name.startswith('_') and name.isupper():
            scales[name] = getattr(px.colors.sequential, name)
    # Diverging
    for name in px.colors.diverging.__dict__:
        if not name.startswith('_') and name.isupper():
            scales[name] = getattr(px.colors.diverging, name)
    # Cyclical
    for name in px.colors.cyclical.__dict__:
        if not name.startswith('_') and name.isupper():
            scales[name] = getattr(px.colors.cyclical, name)
    # Also add some commonly used names (if missing)
    additional = ['rainbow', 'turbo', 'jet', 'inferno', 'magma', 'plasma', 'viridis']
    for name in additional:
        if name not in scales:
            # Plotly may have them under different casing; try to get from px.colors.sequential
            try:
                scales[name.capitalize()] = getattr(px.colors.sequential, name.capitalize())
            except AttributeError:
                pass
    # Sort alphabetically
    sorted_scales = {k: scales[k] for k in sorted(scales.keys())}
    return sorted_scales

def main():
    # Initialize session state
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedPKLLoader(SOLUTIONS_DIR)
    if 'metadata_df' not in st.session_state:
        st.session_state.metadata_df = pd.DataFrame()
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = 'thickness_nm'
    if 'colorscale' not in st.session_state:
        st.session_state.colorscale = 'Viridis'
    if 'font_size' not in st.session_state:
        st.session_state.font_size = 12
    if 'font_family' not in st.session_state:
        st.session_state.font_family = 'Arial'
    if 'line_width' not in st.session_state:
        st.session_state.line_width = 2.0
    
    # ================= SIDEBAR CONFIGURATION =================
    with st.sidebar:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Data Management")
        
        # ✅ SIMPLE DIRECTORY INPUT (like your working interpolation code)
        pkl_dir = st.text_input("PKL Directory", value=SOLUTIONS_DIR,
                               help="Directory containing .pkl simulation files")
        
        if st.button("🔄 Scan Directory", use_container_width=True):
            with st.spinner("Scanning for PKL files..."):
                st.session_state.loader.pkl_dir = pkl_dir
                st.session_state.metadata_df = st.session_state.loader.load_all()
                st.rerun()
        
        if not st.session_state.metadata_df.empty:
            st.success(f"✅ {len(st.session_state.metadata_df)} files loaded")
            
            # Quick stats
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
        
        # ✅ Only show targets that exist in loaded data
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
        
        # ================= ENHANCED VISUALIZATION SETTINGS =================
        with st.expander("🎨 Visualization Settings", expanded=False):
            st.markdown("**Color Scales**")
            # Get all color scales
            all_scales = get_all_colorscales()
            scale_names = list(all_scales.keys())
            default_idx = scale_names.index('Viridis') if 'Viridis' in scale_names else 0
            selected_scale = st.selectbox(
                "Continuous colormap",
                scale_names,
                index=default_idx,
                help="Choose a color scale for heatmaps, sunbursts, etc."
            )
            st.session_state.colorscale = selected_scale
            
            st.markdown("**Font Settings**")
            st.session_state.font_size = st.slider("Font size", 8, 24, 12, 1)
            st.session_state.font_family = st.selectbox(
                "Font family",
                ["Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana"],
                index=0
            )
            
            st.markdown("**Line Thickness**")
            st.session_state.line_width = st.slider("Line width (radar, etc.)", 0.5, 5.0, 2.0, 0.5)
        
        st.divider()
        
        # Analysis Controls
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Analysis Settings")
        
        normalize_radar = st.checkbox("Normalize radar chart values", value=True)
        top_n_table = st.slider("Top N simulations in table", 5, 50, 20, 5)
        
        # ✅ SAFE & FIXED: Only use parameters that exist in the dataframe for correlation
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
        
        # ✅ FIXED: Only set defaults that exist in available options
        default_corr = [p for p in ['c_bulk', 'core_radius_frac', 'k0_nd'] if p in available_corr_params]
        
        # ✅ FIXED: Handle empty options gracefully
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
        
        # Show example of expected file structure
        with st.expander("📋 Expected PKL File Structure"):
            st.code("""
{
    "parameters": {
        "c_bulk": 0.5,
        "core_radius_frac": 0.18,
        "L0_nm": 20.0,
        "k0_nd": 0.4,
        ...
    },
    "snapshots": [(t_nd, phi, c, psi), ...],
    "thickness_history_nm": [(t_nd, th_nd, th_nm, ...), ...],
    "diagnostics": [...]
}
            """, language='json')
        
        st.markdown("""
        ### 🚀 Quick Start
        1. Place your `.pkl` simulation files in the `numerical_solutions` folder
        2. Click "🔄 Scan Directory" in the sidebar
        3. Select a target variable and explore your dataset!
        """)
        return
    
    df = st.session_state.metadata_df
    target = st.session_state.selected_target
    colorscale = st.session_state.colorscale
    font_size = st.session_state.font_size
    font_family = st.session_state.font_family
    line_width = st.session_state.line_width
    
    # ================= HEADER METRICS =================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        target_col = f'metric_{target}' if target in DERIVED_METRICS else target
        if target_col in df.columns and df[target_col].notna().any():
            st.metric(f"Avg {target}", f"{df[target_col].mean():.3f}", 
                     delta=f"{df[target_col].std():.3f} σ")
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
        "🌟 Sunburst Hierarchy (N‑D)",
        "🔗 Correlations", 
        "💡 Dataset Improvements"
    ])
    
    # ===== TAB 1: SUMMARY TABLE =====
    with tab1:
        st.markdown("### 📋 Simulation Summary Table")
        
        # Filters
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
        
        # Apply filters
        filtered_df = df.copy()
        if 'c_bulk' in df.columns and 'filter_c_bulk' in locals() and filter_c_bulk:
            filtered_df = filtered_df[(filtered_df['c_bulk'] >= c_range[0]) & 
                                    (filtered_df['c_bulk'] <= c_range[1])]
        if 'use_edl' in df.columns and 'filter_edl' in locals() and filter_edl:
            filtered_df = filtered_df[filtered_df['use_edl'].isin(edl_filter)]
        
        # Create and display table
        table_builder = SummaryTableBuilder()
        summary_table = table_builder.create_summary_table(
            filtered_df, 
            target_var=target, 
            top_n=top_n_table,
            sort_by=sort_col if 'sort_col' in locals() and sort_col in filtered_df.columns else None
        )
        
        if not summary_table.empty:
            # Display with styling
            st.dataframe(
                summary_table.style
                .format(precision=3)
                .highlight_max(subset=[target] if target in summary_table.columns else None, color='#d1fae5')
                .highlight_min(subset=[target] if target in summary_table.columns else None, color='#fee2e2'),
                use_container_width=True,
                height=400
            )
            
            # Export options
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv = summary_table.to_csv(index=False)
                st.download_button("📥 Download CSV", csv,
                    f"summary_{target}_{datetime.now():%Y%m%d}.csv", "text/csv", use_container_width=True)
            with col_exp2:
                json_data = summary_table.to_json(orient='records', indent=2)
                st.download_button("📥 Download JSON", json_data,
                    f"summary_{target}_{datetime.now():%Y%m%d}.json", "application/json", use_container_width=True)
        else:
            st.warning("No data matches current filters")
    
    # ===== TAB 2: RADAR CHART =====
    with tab2:
        st.markdown("### 🕸️ Multi-Parameter Radar Comparison")
        
        col_r1, col_r2 = st.columns([2, 1])
        
        with col_r2:
            st.markdown("**Select Parameters to Compare**")
            # Group parameters by category
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
            
            # Smart selection: pick diverse examples
            if len(df) >= n_to_show:
                sample_df = df[['c_bulk', 'core_radius_frac', 'use_edl']].copy() if all(c in df.columns for c in ['c_bulk', 'core_radius_frac', 'use_edl']) else df.copy()
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
                        df, 
                        selected_params=available_params[:8],  # Limit for readability
                        selected_indices=selected_indices,
                        normalize=normalize_radar,
                        line_width=line_width,
                        font_size=font_size,
                        font_family=font_family
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Radar chart error: {e}")
                
                # Interpretation help
                with st.expander("💡 How to interpret this radar chart"):
                    st.markdown("""
                    - **Each polygon** represents one simulation
                    - **Vertices** show normalized values for each parameter
                    - **Larger area** = higher values across parameters (if normalized)
                    - **Overlap** indicates similar parameter profiles
                    - Hover over vertices to see actual values
                    """)
            elif not available_params:
                st.info("👈 Select at least one parameter from the sidebar")
    
    # ===== TAB 3: SUNBURST HIERARCHY (N‑D) =====
    with tab3:
        st.markdown("### 🌟 N‑Dimensional Hierarchical Parameter Exploration")
        
        # ✅ SAFE: Only get columns that are safe for nunique
        safe_cols = get_safe_columns_for_nunique(df)
        numeric_or_cat_cols = [
            c for c in safe_cols 
            if pd.api.types.is_numeric_dtype(df[c]) or safe_nunique(df[c], 10)
        ]
        
        # Multi-select for hierarchy levels (order matters)
        st.markdown("**Select hierarchy dimensions (in order from inner to outer ring)**")
        selected_path = st.multiselect(
            "Hierarchy levels",
            numeric_or_cat_cols if numeric_or_cat_cols else list(df.columns),
            default=[c for c in ['c_bulk', 'core_radius_frac'] if c in numeric_or_cat_cols],
            help="Choose columns to build the sunburst hierarchy. First selected = inner ring."
        )
        
        if selected_path:
            sunburst_builder = SunburstBuilder()
            try:
                fig = sunburst_builder.create_parameter_hierarchy(
                    df, 
                    path_cols=selected_path, 
                    value_col=target,
                    colorscale=colorscale,
                    font_size=font_size,
                    font_family=font_family
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Sunburst chart error: {e}")
            
            # Key insights from sunburst
            with st.expander("📊 Key Insights from Hierarchy"):
                # Simple aggregation for insights
                if all(col in df.columns for col in selected_path):
                    target_col = f'metric_{target}' if target in DERIVED_METRICS else target
                    if target_col in df.columns:
                        try:
                            # Group by the deepest level (last in path) and show stats
                            deepest = selected_path[-1]
                            # For continuous, we might have binned version, but we use original for insight
                            # This is just a quick insight, may be approximate
                            insights = df.groupby(deepest)[target_col].agg(['mean', 'std', 'count'])
                            if not insights.empty:
                                best_val = insights['mean'].idxmax()
                                st.markdown(f"""
                                **Best performing segment for {target}:**
                                - {deepest}: `{best_val}`
                                - Average {target}: **{insights.loc[best_val, 'mean']:.3f}** ± {insights.loc[best_val, 'std']:.3f}
                                - Based on {int(insights.loc[best_val, 'count'])} simulations
                                """)
                        except Exception:
                            st.info("Could not compute insights for this combination")
        else:
            st.info("👈 Select at least one dimension for the hierarchy")
    
    # ===== TAB 4: CORRELATIONS =====
    with tab4:
        st.markdown("### 🔗 Parameter Correlation Analysis")
        
        # ✅ SAFE: Only proceed if we have valid correlation parameters
        valid_corr_params = [p for p in correlation_params if p in available_corr_params] if 'available_corr_params' in locals() else []
        
        if valid_corr_params:
            corr_builder = SummaryTableBuilder()
            try:
                fig = corr_builder.create_correlation_matrix(
                    df, 
                    valid_corr_params,
                    colorscale=colorscale,
                    font_size=font_size,
                    font_family=font_family
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Correlation matrix error: {e}")
            
            # Highlight strong correlations
            with st.expander("🎯 Strong Correlations with Target"):
                target_col = f'metric_{target}' if target in DERIVED_METRICS else target
                if target_col in df.columns:
                    # ✅ SAFE: Get only valid numeric columns
                    numeric_cols = get_numeric_columns_safe(df)
                    if target_col in numeric_cols and len(numeric_cols) > 1:
                        try:
                            numeric_df = df[numeric_cols].dropna()
                            if len(numeric_df) > 5:
                                corrs = numeric_df[target_col].corr(numeric_df.drop(columns=[target_col], errors='ignore'))
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
                        st.info("Need more numeric columns for correlation analysis")
        else:
            st.info("👈 Select valid numeric parameters in the sidebar for correlation analysis")
    
    # ===== TAB 5: DATASET IMPROVEMENTS =====
    with tab5:
        st.markdown("### 💡 Dataset Improvement Recommendations")
        
        with st.spinner("Analyzing parameter coverage..."):
            # Detect gaps
            analyzer = DatasetImprovementAnalyzer()
            all_params = [p for cat in PARAM_CATEGORIES.values() for p in cat]
            gaps = analyzer.detect_parameter_gaps(df, all_params[:10])  # Limit for performance
            # Generate recommendations
            recommendations = analyzer.generate_recommendations(df, target, gaps)
        
        if recommendations:
            st.success(f"Generated {len(recommendations)} actionable recommendations")
            
            # Display as cards
            for i, rec in enumerate(recommendations):
                priority_color = {'High': '#fecaca', 'Medium': '#fef3c7', 'Low': '#d1fae5'}
                st.markdown(f"""
                <div class="section-card" style="border-left-color: {'#ef4444' if rec['priority']=='High' else '#f59e0b' if rec['priority']=='Medium' else '#10b981'}">
                    <strong>{rec['type']}</strong> <span style="background:{priority_color[rec['priority']]}; padding:2px 8px; border-radius:12px; font-size:0.8em">{rec['priority']} Priority</span><br>
                    {rec['description']}<br>
                    <em>→ Action: {rec['action']}</em>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ Dataset appears well-covered! Consider exploring new parameter combinations or higher resolution.")
        
        # Parameter coverage visualization
        st.markdown("#### 📊 Current Parameter Coverage")
        
        coverage_cols = [c for c in ['c_bulk', 'core_radius_frac', 'shell_thickness_frac', 'L0_nm', 'k0_nd'] if c in df.columns]
        if coverage_cols:
            fig_cov = make_subplots(rows=1, cols=len(coverage_cols), subplot_titles=coverage_cols)
            
            for i, col in enumerate(coverage_cols, 1):
                try:
                    # Use the selected colorscale for histograms
                    fig_cov.add_trace(
                        go.Histogram(x=df[col], nbinsx=10, name=col, marker_color=colorscale),
                        row=1, col=i
                    )
                except Exception:
                    pass
            
            fig_cov.update_layout(
                height=300, 
                showlegend=False, 
                title_text="Parameter Distribution (more uniform = better coverage)",
                bargap=0.1,
                font=dict(size=font_size, family=font_family)
            )
            fig_cov.update_xaxes(title_text="Value")
            fig_cov.update_yaxes(title_text="Count")
            
            st.plotly_chart(fig_cov, use_container_width=True)
        
        # Export experimental design
        st.markdown("#### 🎯 Export Next Experimental Design")
        
        with st.expander("Generate parameter suggestions for new simulations"):
            # Simple grid suggestion based on gaps
            if gaps:
                st.markdown("**Suggested parameter combinations to fill gaps:**")
                
                suggestion_df = []
                for param, regions in list(gaps.items())[:3]:  # Top 3 gapped params
                    if regions and "Insufficient" not in regions[0]:
                        # Parse first region as example
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
                    
                    # Export
                    csv_sugg = sugg_df.to_csv(index=False)
                    st.download_button("📥 Download Suggested Parameters", csv_sugg,
                        f"suggested_params_{datetime.now():%Y%m%d}.csv", "text/csv")
            else:
                st.info("No clear gaps detected. Consider:")
                st.markdown("""
                - Testing extreme parameter values
                - Exploring parameter interactions (e.g., high c_bulk + low k0_nd)
                - Adding EDL catalyst variations if not already explored
                - Increasing spatial resolution for validation cases
                """)

# =============================================
# FOOTER & HELP
# =============================================
st.divider()
with st.expander("❓ Help & Documentation"):
    st.markdown("""
    ### 🧪 Dataset Designer Guide
    
    **📁 Loading Data:**
    - Place PKL files in `numerical_solutions` directory (or specify custom path)
    - Files should contain: `parameters`, `snapshots`, `thickness_history_nm`
    - Click "🔄 Scan Directory" to load
    
    **🎯 Target Variables:**
    - `thickness_nm`: Final Ag shell thickness (primary output)
    - `growth_rate`: Average deposition rate (nm/s)
    - `ag_area_nm2`: Deposited silver area/volume
    - `interface_sharpness`: Morphology quality metric
    
    **🎨 Visualization Enhancements:**
    - Choose from **50+ color scales** in the sidebar (expand "Visualization Settings")
    - Adjust **font size and family** globally
    - Set **line thickness** for radar charts and other line plots
    
    **📊 N‑D Sunburst:**
    - Select **multiple hierarchy levels** (order matters: inner ring first)
    - Continuous variables are automatically binned to create meaningful categories
    - Hover to see average target value and number of simulations
    
    **💾 Export Options:**
    - Download filtered tables as CSV/JSON
    - Export suggested parameter sets for new simulations
    - All charts are interactive and exportable via Plotly
    
    **🔧 Troubleshooting:**
    - If multiselect shows "No parameters available", load data first
    - If visualizations fail, check PKL files have expected structure
    - Check file naming convention matches expected pattern for auto-parsing
    """)

st.markdown("""
<div style="text-align: center; padding: 1rem; color: #64748b; font-size: 0.9rem;">
🧪 Electroless Deposition Dataset Designer v4.0 (Enhanced Visuals & N‑D Sunburst) • 
Built with Streamlit + Plotly + Pandas • 
<em>Design smarter simulations, faster</em>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
