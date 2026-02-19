#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELECTROLESS Ag-Cu CORE-SHELL INTERPOLATOR
* Interpolates fields: c (concentration), phi (shell), psi (core)
* Postprocessed: shell thickness (nm), material proxy = max(phi, psi) + psi, potential = -alpha * c
* Features: fc (core/L), rs (Δr/r_core), c_bulk, L0_nm
* Transformer with gated attention + parameter Gaussian kernel
* Spatially localized Gaussian: Post-interpolation filter on fields
* Reads PKL from 'electroless_pkl_solutions'
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm, ListedColormap
from matplotlib.cm import get_cmap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import pickle
import torch
import torch.nn as nn
from datetime import datetime
from io import BytesIO
import warnings
import json
import zipfile
import itertools
from typing import List, Dict, Any, Optional, Tuple, Union
import seaborn as sns
from scipy.ndimage import zoom, gaussian_filter
import re
import warnings
warnings.filterwarnings('ignore')
# =============================================
# GLOBAL STYLING CONFIGURATION
# =============================================
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'image.cmap': 'viridis'
})
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "electroless_pkl_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
COLORMAP_OPTIONS = {  # Same as original
    'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'hot', 'afmhot', 'gist_heat',
                  'copper', 'summer', 'Wistia', 'spring', 'autumn', 'winter', 'bone', 'gray', 'pink',
                  'gist_gray', 'gist_yarg', 'binary', 'gist_earth', 'terrain', 'ocean', 'gist_stern', 'gnuplot',
                  'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral',
                  'gist_ncar', 'hsv'],
    'Diverging': ['RdBu', 'RdYlBu', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'BrBG', 'PiYG', 'PRGn', 'PuOr',
                 'RdGy', 'RdYlGn', 'Spectral_r', 'coolwarm_r', 'bwr_r', 'seismic_r'],
    'Qualitative': ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'tab20b', 'tab20c', 'Pastel1', 'Pastel2',
                   'Paired', 'Accent', 'Dark2'],
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted',
                            'turbo'],
    'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu', 'RdBu_r', 'Spectral',
                            'coolwarm', 'bwr', 'seismic', 'BrBG']
}
# =============================================
# DEPOSITION PARAMETERS ENHANCEMENT
# =============================================
class DepositionParameters:
    """Parameters for core-shell deposition with normalization scales"""
    PARAM_SCALES = {
        'fc': 0.5,  # core/L max ~0.45
        'rs': 1.0,  # Δr/r_core max ~0.6
        'c_bulk': 1.0,  # max 1.0
        'L0_nm': 100.0  # typical scale
    }
    THEORETICAL_BASIS = {
        'fc': {
            'description': 'Core fraction relative to domain',
            'reference': 'Phase-field models for core-shell growth'
        },
        'rs': {
            'description': 'Shell thickness ratio',
            'reference': 'Electroless deposition theory'
        },
        # Add for others
    }
   
    @staticmethod
    def get_normalized_param(param_name: str, value: float) -> float:
        scale = DepositionParameters.PARAM_SCALES.get(param_name, 1.0)
        return value / scale
       
    @staticmethod
    def get_theoretical_info(param_name: str) -> Dict:
        return DepositionParameters.THEORETICAL_BASIS.get(param_name, {})
# =============================================
# DEPOSITION PHYSICS PARAMETERS
# =============================================
class DepositionPhysics:
    """Physics for deposition with proxies"""
    MATERIAL_PROPERTIES = {
        'AgCu': {
            'alpha_nd': 2.0,  # default, override from pkl
            # Add more as needed
        },
    }
   
    @staticmethod
    def get_material_properties(material='AgCu'):
        return DepositionPhysics.MATERIAL_PROPERTIES.get(material, DepositionPhysics.MATERIAL_PROPERTIES['AgCu'])
       
    @staticmethod
    def compute_material_proxy(phi, psi):
        return np.maximum(phi, psi) + psi
   
    @staticmethod
    def compute_potential_proxy(c, alpha_nd=2.0):
        return -alpha_nd * c
   
    @staticmethod
    def compute_shell_thickness(history):
        if history and 'thickness_history_nm' in history:
            return history['thickness_history_nm'][-1][-1]  # last thickness nm
        return 0.0
   
    @staticmethod
    def apply_spatial_regularization(field, sigma=1.0):
        return gaussian_filter(field, sigma=sigma)
# =============================================
# ENHANCED SOLUTION LOADER
# =============================================
class EnhancedSolutionLoader:
    """Enhanced loader for deposition PKLs"""
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
       
    def _ensure_directory(self):
        os.makedirs(self.solutions_dir, exist_ok=True)
           
    def scan_solutions(self) -> List[Dict[str, Any]]:
        all_files = []
        for ext in ['*.pkl', '*.pickle']:
            import glob
            pattern = os.path.join(self.solutions_dir, ext)
            files = glob.glob(pattern)
            all_files.extend(files)
       
        all_files.sort(key=os.path.getmtime, reverse=True)
       
        file_info = []
        for file_path in all_files:
            try:
                info = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'format': 'pkl'
                }
                file_info.append(info)
            except:
                continue
               
        return file_info
   
    def read_simulation_file(self, file_path, format_type='auto'):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
           
            standardized = self._standardize_data(data, file_path)
            return standardized
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
   
    def _standardize_data(self, data, file_path):
        standardized = {
            'params': {},
            'history': {},
            'metadata': {
                'filename': os.path.basename(file_path),
                'loaded_at': datetime.now().isoformat(),
                'physics_processed': False
            }
        }
       
        try:
            # Extract from pkl structure
            if 'meta' in data and 'parameters' in data:
                standardized['params'] = {
                    'c_bulk': data['meta']['c_bulk'],
                    'L0_nm': data['parameters']['L0_nm'],
                    'fc': data['parameters']['core_radius_frac'],
                    'rs': data['parameters']['shell_thickness_frac'],
                    'alpha_nd': data['parameters']['alpha_nd'],
                    'use_edl': data['meta']['use_edl'],
                    'mode': data['meta']['mode'],
                    'bc_type': data['meta']['bc_type']
                }
               
            # Fallback: Parse filename if missing
            filename = os.path.basename(file_path)
            match = re.match(r".*_c(\d+\.\d+)_L0(\d+\.\d+)nm_fc(\d+\.\d+)_rs(\d+\.\d+)_.*", filename)
            if match:
                standardized['params'].update({
                    'c_bulk': float(match.group(1)),
                    'L0_nm': float(match.group(2)),
                    'fc': float(match.group(3)),
                    'rs': float(match.group(4))
                })
               
            # History: Use last snapshot for fields
            if 'snapshots' in data and data['snapshots']:
                last_snap = data['snapshots'][-1]
                standardized['history'] = {
                    'c': last_snap[2],  # c
                    'phi': last_snap[1],  # phi
                    'psi': last_snap[3]   # psi
                }
                standardized['shell_thickness_nm'] = DepositionPhysics.compute_shell_thickness(data)
               
            if 'metadata' in data:
                standardized['metadata'].update(data['metadata'])
                   
            self._convert_tensors(standardized)
        except Exception as e:
            st.error(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
           
        return standardized
   
    def _convert_tensors(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.cpu().numpy()
                elif isinstance(value, (dict, list)):
                    self._convert_tensors(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, torch.Tensor):
                    data[i] = item.cpu().numpy()
                elif isinstance(item, (dict, list)):
                    self._convert_tensors(item)
   
    def load_all_solutions(self, use_cache=True, max_files=None):
        solutions = []
        file_info = self.scan_solutions()
       
        if max_files:
            file_info = file_info[:max_files]
           
        if not file_info:
            return solutions
           
        for file_info_item in file_info:
            cache_key = file_info_item['filename']
            if use_cache and cache_key in self.cache:
                solutions.append(self.cache[cache_key])
                continue
               
            solution = self.read_simulation_file(file_info_item['path'])
            if solution:
                self.cache[cache_key] = solution
                solutions.append(solution)
               
        return solutions
# =============================================
# POSITIONAL ENCODING FOR TRANSFORMER
# =============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
       
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)
# =============================================
# TRANSFORMER PARAMETER INTERPOLATOR WITH GATED ATTENTION
# =============================================
class TransformerParameterInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                param_sigma=0.1, temperature=1.0, locality_weight_factor=0.5):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.param_sigma = param_sigma
        self.temperature = temperature
        self.locality_weight_factor = locality_weight_factor
       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(15, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.gate_linear = nn.Linear(d_model, 1)  # For gated fusion
       
    def set_param_parameters(self, param_sigma=None, locality_weight_factor=None):
        if param_sigma is not None:
            self.param_sigma = param_sigma
        if locality_weight_factor is not None:
            self.locality_weight_factor = locality_weight_factor
           
    def compute_parameter_bracketing_kernel(self, source_params, target_params):
        param_weights = []
        param_mask = []
        param_distances = []
       
        target_fc = target_params.get('fc', 0.18)
        target_rs = target_params.get('rs', 0.2)
        target_c_bulk = target_params.get('c_bulk', 1.0)
        target_L0_nm = target_params.get('L0_nm', 20.0)
        target_use_edl = target_params.get('use_edl', False)
       
        for src in source_params:
            src_fc = src.get('fc', 0.18)
            src_rs = src.get('rs', 0.2)
            src_c_bulk = src.get('c_bulk', 1.0)
            src_L0_nm = src.get('L0_nm', 20.0)
           
            # Normalized differences
            diff_fc = (src_fc - target_fc) / DepositionParameters.PARAM_SCALES['fc']
            diff_rs = (src_rs - target_rs) / DepositionParameters.PARAM_SCALES['rs']
            diff_c = (src_c_bulk - target_c_bulk) / DepositionParameters.PARAM_SCALES['c_bulk']
            diff_L0 = (src_L0_nm - target_L0_nm) / DepositionParameters.PARAM_SCALES['L0_nm']
           
            param_dist = diff_fc**2 + diff_rs**2 + diff_c**2 + diff_L0**2
            param_distances.append(param_dist)
           
            if src.get('use_edl') == target_use_edl:
                param_mask.append(1.0)
            else:
                param_mask.append(1e-6)
               
            weight = np.exp(-0.5 * param_dist / self.param_sigma ** 2)
            param_weights.append(weight)
           
        return np.array(param_weights), np.array(param_mask), np.array(param_distances)
   
    def encode_parameters(self, params_list, target_params):
        encoded = []
        target_c_bulk = target_params.get('c_bulk', 1.0)
        for params in params_list:
            features = []
           
            # Normalized parameters
            features.append(DepositionParameters.get_normalized_param('fc', params.get('fc', 0.18)))
            features.append(DepositionParameters.get_normalized_param('rs', params.get('rs', 0.2)))
            features.append(DepositionParameters.get_normalized_param('c_bulk', params.get('c_bulk', 1.0)))
            features.append(DepositionParameters.get_normalized_param('L0_nm', params.get('L0_nm', 20.0)))
            features.append(params.get('alpha_nd', 2.0) / 10.0)  # normalize
           
            # One-hot for use_edl
            features.append(1.0 if params.get('use_edl', False) else 0.0)
            features.append(0.0 if params.get('use_edl', False) else 1.0)
           
            # One-hot for mode
            modes = ['2D', '3D']
            mode = params.get('mode', '2D')
            for m in modes:
                features.append(1.0 if m in mode else 0.0)
               
            # Proximity to target c_bulk
            diff_c = abs(params.get('c_bulk', 1.0) - target_c_bulk)
            features.append(np.exp(-diff_c / 0.5))
           
            # Pad to 15
            while len(features) < 15:
                features.append(0.0)
            encoded.append(features[:15])
           
        return torch.FloatTensor(encoded)
       
    def interpolate_spatial_fields(self, sources, target_params):
        if not sources:
            return None
           
        try:
            source_params = []
            source_fields = []
            source_indices = []
           
            for i, src in enumerate(sources):
                if 'params' not in src or 'history' not in src:
                    continue
               
                source_params.append(src['params'])
                source_indices.append(i)
               
                history = src['history']
                if history:
                    fields = {
                        'c': history['c'],
                        'phi': history['phi'],
                        'psi': history['psi'],
                        'source_index': i,
                        'source_params': src['params']
                    }
                    source_fields.append(fields)
               
            if not source_params or not source_fields:
                st.error("No valid sources found.")
                return None
               
            # Resize to common shape
            common_shape = (256, 256)  # Assume 2D, Nx=256
            resized_fields = []
            for fields in source_fields:
                resized = {}
                for key, field in fields.items():
                    if key in ['c', 'phi', 'psi'] and field.shape != common_shape:
                        factors = [t/s for t, s in zip(common_shape, field.shape)]
                        resized[key] = zoom(field, factors, order=1)
                    else:
                        resized[key] = field
                resized_fields.append(resized)
            source_fields = resized_fields
               
            # Encode
            source_features = self.encode_parameters(source_params, target_params)
            target_features = self.encode_parameters([target_params], target_params)
           
            # Pad features if needed
            if source_features.shape[1] < 15:
                padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                source_features = torch.cat([source_features, padding], dim=1)
            if target_features.shape[1] < 15:
                padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                target_features = torch.cat([target_features, padding], dim=1)
           
            # Parameter kernel
            param_kernel, param_mask, param_distances = self.compute_parameter_bracketing_kernel(
                source_params, target_params
            )
           
            # Transformer
            all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
            proj_features = self.input_proj(all_features)
            proj_features = self.pos_encoder(proj_features)
            transformer_output = self.transformer(proj_features)
           
            target_rep = transformer_output[:, 0, :]
            source_reps = transformer_output[:, 1:, :]
           
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1)
            attn_scores = attn_scores / np.sqrt(self.d_model)
            attn_scores = attn_scores / self.temperature
           
            param_kernel_tensor = torch.FloatTensor(param_kernel).unsqueeze(0)
            param_mask_tensor = torch.FloatTensor(param_mask).unsqueeze(0)
           
            # Gated fusion
            gate = torch.sigmoid(self.gate_linear(target_rep))  # [1, 1]
            biased_scores = gate * attn_scores + (1 - gate) * param_kernel_tensor * param_mask_tensor
           
            final_attention_weights = torch.softmax(biased_scores, dim=-1).squeeze().detach().cpu().numpy()
           
            # Interpolate fields
            interpolated_fields = {}
            shape = source_fields[0]['c'].shape
           
            for component in ['c', 'phi', 'psi']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    interpolated += final_attention_weights[i] * fields[component]
                # Apply spatial Gaussian regularization
                interpolated = DepositionPhysics.apply_spatial_regularization(interpolated, sigma=1.0)
                interpolated_fields[component] = interpolated
               
            # Derived fields
            alpha_nd = target_params.get('alpha_nd', 2.0)
            interpolated_fields['material_proxy'] = DepositionPhysics.compute_material_proxy(
                interpolated_fields['phi'], interpolated_fields['psi']
            )
            interpolated_fields['potential_proxy'] = DepositionPhysics.compute_potential_proxy(
                interpolated_fields['c'], alpha_nd
            )
            interpolated_fields['shell_thickness_nm'] = np.average([src['shell_thickness_nm'] * final_attention_weights[i] for i, src in enumerate(sources)])
               
            # Statistics (example)
            statistics = {
                'c': {'mean': float(np.mean(interpolated_fields['c']))},
                # Add more
            }
           
            return {
                'fields': interpolated_fields,
                'weights': {
                    'combined': final_attention_weights.tolist(),
                    'param_kernel': param_kernel.tolist(),
                    'param_mask': param_mask.tolist()
                },
                'statistics': statistics,
                'target_params': target_params,
                'shape': shape,
                'num_sources': len(source_fields),
                'source_distances': param_distances,
                'source_indices': source_indices
            }
           
        except Exception as e:
            st.error(f"Error during interpolation: {str(e)}")
            return None
# =============================================
# ENHANCED HEAT MAP VISUALIZER WITH VARIABLE DOMAIN
# =============================================
class HeatMapVisualizer:
    """Visualizer for deposition fields with variable L0_nm"""
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS  # From global
       
    def create_stress_heatmap(self, field, title="Field Heat Map",
                            cmap_name='viridis', figsize=(12, 10),
                            colorbar_label="Value", vmin=None, vmax=None,
                            L0_nm=20.0, show_stats=True, target_params=None):
        fig, ax = plt.subplots(figsize=figsize)
       
        cmap = plt.get_cmap(cmap_name)
           
        if vmin is None:
            vmin = np.nanmin(field)
        if vmax is None:
            vmax = np.nanmax(field)
           
        extent = [0, L0_nm, 0, L0_nm]
        im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=extent, aspect='equal', interpolation='bilinear', origin='lower')
       
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=16)
       
        title_str = f"{title}\nDomain: {L0_nm} nm × {L0_nm} nm"
        if target_params:
            title_str += f"\nfc={target_params['fc']:.3f}, rs={target_params['rs']:.3f}, c_bulk={target_params['c_bulk']:.3f}"
        ax.set_title(title_str, fontsize=20)
        ax.set_xlabel("X Position (nm)", fontsize=16)
        ax.set_ylabel("Y Position (nm)", fontsize=16)
        ax.grid(True, alpha=0.2)
       
        if show_stats:
            stats_text = f"Max: {vmax:.3f}\nMin: {vmin:.3f}\nMean: {np.nanmean(field):.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.9))
       
        plt.tight_layout()
        return fig
       
    # Add other methods similarly, passing L0_nm
    # e.g., create_interactive_heatmap, etc., update extent
# =============================================
# RESULTS MANAGER FOR EXPORT
# =============================================
class ResultsManager:
    def prepare_export_data(self, interpolation_result, visualization_params):
        # Similar, update for new fields
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'interpolation_method': 'transformer_gated_parameter',
                'visualization_params': visualization_params
            },
            'result': {
                'target_params': interpolation_result['target_params'],
                'shape': interpolation_result['shape'],
                'statistics': interpolation_result['statistics'],
                'weights': interpolation_result['weights'],
                'num_sources': interpolation_result.get('num_sources', 0)
            }
        }
       
        for field_name, field_data in interpolation_result['fields'].items():
            export_data['result'][f'{field_name}_data'] = field_data.tolist()
           
        return export_data
   
    def export_to_json(self, export_data, filename=None):
        # Same as original
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            params = export_data['result']['target_params']
            filename = f"deposition_interpolation_fc{params['fc']}_rs{params['rs']}_{timestamp}.json"
           
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
   
    def _json_serializer(self, obj):
        # Same
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)
# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Electroless Ag-Cu Interpolator", layout="wide")
   
    # Initialize
    if 'solutions' not in st.session_state: st.session_state.solutions = []
    if 'loader' not in st.session_state: st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        st.session_state.transformer_interpolator = TransformerParameterInterpolator(param_sigma=0.1)
    if 'heatmap_visualizer' not in st.session_state: st.session_state.heatmap_visualizer = HeatMapVisualizer()
    if 'results_manager' not in st.session_state: st.session_state.results_manager = ResultsManager()
    if 'interpolation_result' not in st.session_state: st.session_state.interpolation_result = None
   
    # Sidebar
    with st.sidebar:
        st.markdown("#### 📁 Data Management")
        if st.button("📥 Load Solutions"):
            st.session_state.solutions = st.session_state.loader.load_all_solutions()
            st.success(f"Loaded {len(st.session_state.solutions)} solutions")
       
        st.markdown("#### 🎯 Target Parameters")
        target_fc = st.slider("core/L (fc)", 0.05, 0.45, 0.07, 0.01)
        target_rs = st.slider("Δr/r_core (rs)", 0.01, 0.6, 0.1, 0.01)
        target_c_bulk = st.slider("c_bulk (C_Ag/C_Cu)", 0.1, 1.0, 1.0, 0.05)
        target_L0_nm = st.number_input("Domain L0 (nm)", 10.0, 100.0, 60.0)
        target_use_edl = st.checkbox("Use EDL", False)
        target_alpha_nd = st.slider("alpha_nd", 0.0, 10.0, 2.0, 0.1)
       
        st.markdown("#### ⚛️ Interpolation")
        param_sigma = st.slider("Param Kernel Sigma", 0.01, 1.0, 0.1, 0.01)
        locality_weight_factor = st.slider("Locality Factor", 0.0, 1.0, 0.5, 0.1)
       
        if st.button("🧠 Perform Interpolation", type="primary"):
            if not st.session_state.solutions:
                st.error("Load solutions first!")
            else:
                target_params = {
                    'fc': target_fc, 'rs': target_rs, 'c_bulk': target_c_bulk,
                    'L0_nm': target_L0_nm, 'use_edl': target_use_edl, 'alpha_nd': target_alpha_nd
                }
                result = st.session_state.transformer_interpolator.interpolate_spatial_fields(st.session_state.solutions, target_params)
                if result:
                    st.session_state.interpolation_result = result
                    st.success("Interpolation successful.")
   
    # Tabs if result
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        viz_tab, export_tab = st.tabs(["🎨 Visualization", "💾 Export"])
       
        with viz_tab:
            component = st.selectbox("Component", ['c', 'phi', 'psi', 'material_proxy', 'potential_proxy'])
            cmap_name = st.selectbox("Colormap", COLORMAP_OPTIONS['Sequential'])
           
            if component in result['fields']:
                field = result['fields'][component]
                L0_nm = result['target_params']['L0_nm']
                fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                    field, title=component, cmap_name=cmap_name,
                    L0_nm=L0_nm, target_params=result['target_params']
                )
                st.pyplot(fig)
           
        with export_tab:
            if st.button("📊 Export to JSON"):
                export_data = st.session_state.results_manager.prepare_export_data(result, {})
                json_str, filename = st.session_state.results_manager.export_to_json(export_data)
                st.download_button("⬇️ Download JSON", json_str, filename, "application/json")

if __name__ == "__main__":
    main()
