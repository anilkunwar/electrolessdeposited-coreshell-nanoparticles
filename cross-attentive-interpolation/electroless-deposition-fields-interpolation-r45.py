#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INTELLIGENT CORE‑SHELL DESIGNER – FULLY INTEGRATED VERSION
=============================================================
- Natural language interface (regex‑based, with optional GPT‑2 / Qwen parsing)
- Real physics‑based interpolation using hybrid weights (α·β·γ·Attention)
- All enhanced visualizations: Sankey, chord, radar, 3D thickness, ground‑truth comparison, etc.
- Temporal caching, animation streaming, and memory‑efficient key frames
- **Enhanced time interpolation (progress‑based)** and **phi shell tracking diagnostics**
- FIX: Strictly increasing thickness for spline construction (solves "x must be strictly increasing" error)
- **FIXED**: GPT‑2 / Qwen integration – robust JSON extraction, better prompts, fallback mechanisms
- **FIXED**: Shell completeness detection – median radial profile, coverage capping, softer threshold
- **NEW**: Hybrid (regex + GPT‑2/Qwen) parameter extraction with confidence merging and ensemble voting
- **NEW**: Model selection dropdown (GPT‑2, Qwen2‑0.5B‑Instruct, Qwen2.5‑0.5B‑Instruct)
- **FIXED**: Session state conflict with widget key – separate loaded backend from widget value
- **NEW**: Dynamic completeness analysis – updates when time slider moves (rule‑based and LLM)
- **FIXED**: UnhashableParamError in LLM caching – manual cache via session_state (LRU)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from io import BytesIO
import warnings
import json
import re
import tempfile
import weakref
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple, Callable
import time
from scipy.ndimage import zoom, gaussian_filter, binary_closing, generate_binary_structure, binary_dilation, binary_erosion
from scipy.interpolate import interp1d, CubicSpline, RegularGridInterpolator
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from math import pi, cos, sin
import itertools
import threading
import shutil

# NEW IMPORTS FOR GPT‑2 AND QWEN
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("transformers not installed. GPT‑2 / Qwen features will be disabled.")

warnings.filterwarnings('ignore')

# =============================================
# GLOBAL CONFIGURATION
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
    'image.cmap': 'viridis',
    'animation.html': 'html5'
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
TEMP_ANIMATION_DIR = os.path.join(SCRIPT_DIR, "temp_animations")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_ANIMATION_DIR, exist_ok=True)

# =============================================
# ENHANCED COLORMAP OPTIONS (50+ colormaps)
# =============================================
COLORMAP_OPTIONS = {
    'Perceptually Uniform Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    'Sequential (Matplotlib)': ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'],
    'Sequential (Matplotlib 2)': ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                                  'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                                  'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'],
    'Diverging': ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn',
                  'Spectral', 'coolwarm', 'bwr', 'seismic'],
    'Cyclic': ['twilight', 'twilight_shifted', 'hsv'],
    'Miscellaneous': ['jet', 'turbo', 'rainbow', 'gist_rainbow', 'gist_ncar', 'nipy_spectral',
                      'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                      'gist_earth', 'terrain', 'ocean', 'gist_water', 'flag', 'prism'],
    'Qualitative': ['tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3',
                    'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2'],
    'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu']
}

# =============================================
# EXACT PHASE-FIELD MATERIAL COLORS
# =============================================
MATERIAL_COLORS_EXACT = {
    'electrolyte': (0.894, 0.102, 0.110, 1.0),
    'Ag': (1.000, 0.498, 0.000, 1.0),
    'Cu': (0.600, 0.600, 0.600, 1.0)
}

MATERIAL_COLORMAP_MATPLOTLIB = ListedColormap([
    MATERIAL_COLORS_EXACT['electrolyte'][:3],
    MATERIAL_COLORS_EXACT['Ag'][:3],
    MATERIAL_COLORS_EXACT['Cu'][:3]
], name='phase_field_materials')

MATERIAL_BOUNDARY_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)

MATERIAL_COLORSCALE_PLOTLY = [
    [0.0, f"rgb({int(0.894*255)},{int(0.102*255)},{int(0.110*255)})"],
    [0.333, f"rgb({int(0.894*255)},{int(0.102*255)},{int(0.110*255)})"],
    [0.334, f"rgb({int(1.000*255)},{int(0.498*255)},{int(0.000*255)})"],
    [0.666, f"rgb({int(1.000*255)},{int(0.498*255)},{int(0.000*255)})"],
    [0.667, f"rgb({int(0.600*255)},{int(0.600*255)},{int(0.600*255)})"],
    [1.0, f"rgb({int(0.600*255)},{int(0.600*255)},{int(0.600*255)})"]
]

# =============================================
# DEPOSITION PARAMETERS (normalisation)
# =============================================
class DepositionParameters:
    """Normalises and stores core‑shell deposition parameters."""
    
    RANGES = {
        'fc': (0.05, 0.45),
        'rs': (0.01, 0.6),
        'c_bulk': (0.1, 1.0),
        'L0_nm': (10.0, 100.0)
    }
    
    @staticmethod
    def normalize(value: float, param_name: str) -> float:
        low, high = DepositionParameters.RANGES[param_name]
        if param_name == 'c_bulk':
            log_low = np.log10(low + 1e-6)
            log_high = np.log10(high + 1e-6)
            log_val = np.log10(value + 1e-6)
            return (log_val - log_low) / (log_high - log_low)
        else:
            return (value - low) / (high - low)
    
    @staticmethod
    def denormalize(norm_value: float, param_name: str) -> float:
        low, high = DepositionParameters.RANGES[param_name]
        if param_name == 'c_bulk':
            log_low = np.log10(low + 1e-6)
            log_high = np.log10(high + 1e-6)
            log_val = norm_value * (log_high - log_low) + log_low
            return 10**log_val
        else:
            return norm_value * (high - low) + low

# =============================================
# DEPOSITION PHYSICS (derived quantities)
# =============================================
class DepositionPhysics:
    """Computes derived quantities for core‑shell deposition."""
    
    @staticmethod
    def material_proxy(phi: np.ndarray, psi: np.ndarray, method: str = "max(phi, psi) + psi") -> np.ndarray:
        if method == "max(phi, psi) + psi":
            return np.where(psi > 0.5, 2.0, np.where(phi > 0.5, 1.0, 0.0))
        elif method == "phi + 2*psi":
            return phi + 2.0 * psi
        elif method == "phi*(1-psi) + 2*psi":
            return phi * (1.0 - psi) + 2.0 * psi
        else:
            raise ValueError(f"Unknown material proxy method: {method}")
    
    @staticmethod
    def potential_proxy(c: np.ndarray, alpha_nd: float) -> np.ndarray:
        return -alpha_nd * c
    
    @staticmethod
    def shell_thickness(phi: np.ndarray, psi: np.ndarray, core_radius_frac: float,
                       threshold: float = 0.5, dx: float = 1.0) -> float:
        """Use the exact visual proxy so thickness matches the plot."""
        proxy = DepositionPhysics.material_proxy(phi, psi)          # same as visual!
        ny, nx = proxy.shape
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dist = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)                 # normalized distance
        shell_mask = (proxy == 1.0)                                 # Ag shell
        if np.any(shell_mask):
            max_dist = np.max(dist[shell_mask])
            thickness_nd = max_dist - core_radius_frac
            return max(0.0, thickness_nd)
        return 0.0
    
    @staticmethod
    def phase_stats(phi, psi, dx, dy, L0, threshold=0.5):
        ag_mask = (phi > threshold) & (psi <= 0.5)
        cu_mask = psi > 0.5
        electrolyte_mask = ~(ag_mask | cu_mask)
        cell_area_nd = dx * dy
        electrolyte_area_nd = np.sum(electrolyte_mask) * cell_area_nd
        ag_area_nd = np.sum(ag_mask) * cell_area_nd
        cu_area_nd = np.sum(cu_mask) * cell_area_nd
        return {
            "Electrolyte": (electrolyte_area_nd, electrolyte_area_nd * (L0**2)),
            "Ag": (ag_area_nd, ag_area_nd * (L0**2)),
            "Cu": (cu_area_nd, cu_area_nd * (L0**2))
        }
    
    @staticmethod
    def compute_growth_rate(thickness_history: List[Dict], time_idx: int) -> float:
        if time_idx == 0 or time_idx >= len(thickness_history):
            return 0.0
        dt = thickness_history[time_idx]['t_nd'] - thickness_history[time_idx-1]['t_nd']
        if dt == 0:
            return 0.0
        dth = thickness_history[time_idx]['th_nm'] - thickness_history[time_idx-1]['th_nm']
        return dth / dt if dt > 0 else 0.0
    
    @staticmethod
    def compute_radial_profile(field, L0, center_frac=0.5, n_bins=100, use_median=False):  # MODIFIED: added use_median
        H, W = field.shape
        x = np.linspace(0, L0, W)
        y = np.linspace(0, L0, H)
        X, Y = np.meshgrid(x, y, indexing='xy')
        center_x, center_y = center_frac * L0, center_frac * L0
        R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        r_max = np.sqrt(2) * L0 / 2
        r_edges = np.linspace(0, r_max, n_bins + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        if use_median:
            profile = np.array([
                np.median(field[(R >= r_edges[i]) & (R < r_edges[i+1])])
                if np.any((R >= r_edges[i]) & (R < r_edges[i+1])) else 0.0
                for i in range(n_bins)
            ])
        else:
            profile = np.array([
                field[(R >= r_edges[i]) & (R < r_edges[i+1])].mean()
                if np.any((R >= r_edges[i]) & (R < r_edges[i+1])) else 0.0
                for i in range(n_bins)
            ])
        return r_centers, profile

# =============================================
# POSITIONAL ENCODING (for Transformer)
# =============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                           (-np.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)

# =============================================
# ENHANCED CORE‑SHELL INTERPOLATOR (Hybrid weights)
# =============================================
class CoreShellInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                 param_sigma=None, temperature=1.0,
                 gating_mode="Hierarchical: L0 → fc → rs → c_bulk",
                 lambda_shape=0.5, sigma_shape=0.15):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        if param_sigma is None:
            param_sigma = [0.15, 0.15, 0.15, 0.15]
        self.param_sigma = param_sigma
        
        self.temperature = temperature
        self.gating_mode = gating_mode
        self.lambda_shape = lambda_shape
        self.sigma_shape = sigma_shape
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(12, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
    
    def set_parameter_sigma(self, param_sigma):
        self.param_sigma = param_sigma
    
    def set_gating_mode(self, gating_mode):
        self.gating_mode = gating_mode
    
    def set_shape_params(self, lambda_shape, sigma_shape):
        self.lambda_shape = lambda_shape
        self.sigma_shape = sigma_shape
    
    def filter_sources_hierarchy(self, sources: List[Dict], target_params: Dict,
                              require_categorical_match: bool = False) -> Tuple[List[Dict], Dict]:
        valid_sources = []
        excluded_reasons = {'categorical': 0, 'kept': 0}
        
        target_mode = target_params.get('mode', '2D (planar)')
        target_bc = target_params.get('bc_type', 'Neu')
        target_edl = target_params.get('use_edl', False)
        
        for src in sources:
            params = src.get('params', {})
            
            if require_categorical_match:
                if params.get('mode') != target_mode:
                    excluded_reasons['categorical'] += 1
                    continue
                if params.get('bc_type') != target_bc:
                    excluded_reasons['categorical'] += 1
                    continue
                if params.get('use_edl') != target_edl:
                    excluded_reasons['categorical'] += 1
                    continue
            
            valid_sources.append(src)
            excluded_reasons['kept'] += 1
        
        if not valid_sources and sources:
            st.warning("⚠️ No sources passed filters. Using nearest neighbor fallback.")
            distances = []
            for src in sources:
                p = src['params']
                d = sum((target_params.get(k, 0) - p.get(k, 0))**2 
                       for k in ['fc', 'rs', 'L0_nm'])
                distances.append(d)
            valid_sources = [sources[np.argmin(distances)]]
        
        return valid_sources, excluded_reasons
    
    def compute_alpha(self, source_params: List[Dict], target_L0: float,
                     preference_tiers: Dict = None) -> np.ndarray:
        if preference_tiers is None:
            preference_tiers = {
                'preferred': (5.0, 1.0),
                'acceptable': (15.0, 0.75),
                'marginal': (30.0, 0.45),
                'poor': (50.0, 0.15),
                'exclude': (np.inf, 0.01)
            }
        
        alphas = []
        for src in source_params:
            src_L0 = src.get('L0_nm', 20.0)
            delta = abs(target_L0 - src_L0)
            
            if delta <= preference_tiers['preferred'][0]:
                weight = preference_tiers['preferred'][1]
            elif delta <= preference_tiers['acceptable'][0]:
                t = (delta - 5.0) / (15.0 - 5.0)
                weight = preference_tiers['preferred'][1] - t * (
                    preference_tiers['preferred'][1] - preference_tiers['acceptable'][1])
            elif delta <= preference_tiers['marginal'][0]:
                t = (delta - 15.0) / (30.0 - 15.0)
                weight = preference_tiers['acceptable'][1] - t * (
                    preference_tiers['acceptable'][1] - preference_tiers['marginal'][1])
            elif delta <= preference_tiers['poor'][0]:
                t = (delta - 30.0) / (50.0 - 30.0)
                weight = preference_tiers['marginal'][1] - t * (
                    preference_tiers['marginal'][1] - preference_tiers['poor'][1])
            else:
                weight = preference_tiers['exclude'][1]
            
            alphas.append(weight)
        
        return np.array(alphas)
    
    def compute_beta(self, source_params: List[Dict], target_params: Dict) -> Tuple[np.ndarray, Dict]:
        weights = {'fc': 2.0, 'rs': 1.5, 'c_bulk': 3.0}
        betas = []
        individual_weights = {'fc': [], 'rs': [], 'c_bulk': []}
        
        for src in source_params:
            sq_sum = 0.0
            src_indiv_weights = {}
            
            for i, (pname, w) in enumerate(weights.items()):
                norm_src = DepositionParameters.normalize(src.get(pname, 0.5), pname)
                norm_tar = DepositionParameters.normalize(target_params.get(pname, 0.5), pname)
                diff = norm_src - norm_tar
                sigma_idx = ['fc', 'rs', 'c_bulk'].index(pname)
                sigma = self.param_sigma[sigma_idx]
                
                indiv_weight = np.exp(-0.5 * (diff / sigma) ** 2)
                src_indiv_weights[pname] = indiv_weight
                
                sq_sum += w * (diff / sigma) ** 2
            
            beta = np.exp(-0.5 * sq_sum)
            betas.append(beta)
            
            for pname in weights.keys():
                individual_weights[pname].append(src_indiv_weights[pname])
        
        return np.array(betas), individual_weights
    
    def compute_gamma(self, source_fields: List[Dict], source_params: List[Dict],
                     target_params: Dict, time_norm: float, beta_weights: np.ndarray) -> np.ndarray:
        n_sources = len(source_fields)
        if n_sources == 0:
            return np.array([])
        
        profiles = []
        radii_list = []
        
        for i, src in enumerate(source_params):
            L0 = src.get('L0_nm', 20.0)
            field = source_fields[i]['phi']
            r_centers, profile = DepositionPhysics.compute_radial_profile(field, L0, n_bins=100)
            profiles.append(profile)
            radii_list.append(r_centers)
        
        max_radius = max([r[-1] for r in radii_list])
        r_common = np.linspace(0, max_radius, 100)
        
        profiles_interp = []
        for i in range(n_sources):
            prof_interp = np.interp(r_common, radii_list[i], profiles[i], left=0, right=0)
            profiles_interp.append(prof_interp)
        profiles_interp = np.array(profiles_interp)
        
        beta_norm = beta_weights / (np.sum(beta_weights) + 1e-12)
        ref_profile = np.sum(profiles_interp * beta_norm[:, None], axis=0)
        
        mse = np.mean((profiles_interp - ref_profile) ** 2, axis=1)
        gamma = np.exp(-mse / self.sigma_shape)
        
        return gamma
    
    def encode_parameters(self, params_list: List[Dict]) -> torch.Tensor:
        features = []
        for p in params_list:
            feat = []
            for name in ['fc', 'rs', 'c_bulk', 'L0_nm']:
                val = p.get(name, 0.5)
                norm_val = DepositionParameters.normalize(val, name)
                feat.append(norm_val)
            feat.append(1.0 if p.get('bc_type', 'Neu') == 'Dir' else 0.0)
            feat.append(1.0 if p.get('use_edl', False) else 0.0)
            feat.append(1.0 if p.get('mode', '2D (planar)') != '2D (planar)' else 0.0)
            feat.append(1.0 if 'B' in p.get('growth_model', 'Model A') else 0.0)
            while len(feat) < 12:
                feat.append(0.0)
            features.append(feat[:12])
        return torch.FloatTensor(features)
    
    def _get_fields_at_time(self, source: Dict, time_norm: float, target_shape: Tuple[int, int]):
        history = source.get('history', [])
        if not history:
            return {'phi': np.zeros(target_shape), 'c': np.zeros(target_shape), 'psi': np.zeros(target_shape)}
        
        t_max = 1.0
        if source.get('thickness_history'):
            t_max = source['thickness_history'][-1]['t_nd']
        else:
            t_max = history[-1]['t_nd']
        
        t_target = time_norm * t_max
        
        if len(history) == 1:
            snap = history[0]
            phi = self._ensure_2d(snap['phi'])
            c = self._ensure_2d(snap['c'])
            psi = self._ensure_2d(snap['psi'])
        else:
            t_vals = np.array([s['t_nd'] for s in history])
            if t_target <= t_vals[0]:
                snap = history[0]
                phi = self._ensure_2d(snap['phi'])
                c = self._ensure_2d(snap['c'])
                psi = self._ensure_2d(snap['psi'])
            elif t_target >= t_vals[-1]:
                snap = history[-1]
                phi = self._ensure_2d(snap['phi'])
                c = self._ensure_2d(snap['c'])
                psi = self._ensure_2d(snap['psi'])
            else:
                idx = np.searchsorted(t_vals, t_target) - 1
                idx = max(0, min(idx, len(history)-2))
                t1, t2 = t_vals[idx], t_vals[idx+1]
                snap1, snap2 = history[idx], history[idx+1]
                alpha = (t_target - t1) / (t2 - t1) if t2 > t1 else 0.0
                
                phi1 = self._ensure_2d(snap1['phi'])
                phi2 = self._ensure_2d(snap2['phi'])
                c1 = self._ensure_2d(snap1['c'])
                c2 = self._ensure_2d(snap2['c'])
                psi1 = self._ensure_2d(snap1['psi'])
                psi2 = self._ensure_2d(snap2['psi'])
                
                phi = (1 - alpha) * phi1 + alpha * phi2
                c = (1 - alpha) * c1 + alpha * c2
                psi = (1 - alpha) * psi1 + alpha * psi2
        
        if phi.shape != target_shape:
            factors = (target_shape[0]/phi.shape[0], target_shape[1]/phi.shape[1])
            phi = zoom(phi, factors, order=1)
            c = zoom(c, factors, order=1)
            psi = zoom(psi, factors, order=1)
        
        return {'phi': phi, 'c': c, 'psi': psi}
    
    def _ensure_2d(self, arr):
        if arr is None:
            return np.zeros((1,1))
        if torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        if arr.ndim == 3:
            mid = arr.shape[0] // 2
            return arr[mid, :, :]
        return arr
    
    def interpolate_fields(self, sources: List[Dict], target_params: Dict,
                          target_shape: Tuple[int, int] = (256, 256),
                          n_time_points: int = 100,
                          time_norm: Optional[float] = None,
                          require_categorical_match: bool = False,
                          recompute_thickness: bool = True):
        if not sources:
            return None
        
        filtered_sources, filter_stats = self.filter_sources_hierarchy(
            sources, target_params, require_categorical_match=require_categorical_match
        )
        active_sources = filtered_sources if filtered_sources else sources
        
        source_params = []
        source_fields = []
        source_thickness = []
        source_tau0 = []
        source_t_max_nd = []
        
        for src in active_sources:
            if 'params' not in src or 'history' not in src or len(src['history']) == 0:
                continue
            
            params = src['params'].copy()
            params.setdefault('fc', params.get('core_radius_frac', 0.18))
            params.setdefault('rs', params.get('shell_thickness_frac', 0.2))
            params.setdefault('c_bulk', params.get('c_bulk', 1.0))
            params.setdefault('L0_nm', params.get('L0_nm', 20.0))
            params.setdefault('bc_type', params.get('bc_type', 'Neu'))
            params.setdefault('use_edl', params.get('use_edl', False))
            params.setdefault('mode', params.get('mode', '2D (planar)'))
            params.setdefault('growth_model', params.get('growth_model', 'Model A'))
            params.setdefault('tau0_s', params.get('tau0_s', 1e-4))
            
            source_params.append(params)
            
            if time_norm is None:
                t_req = 1.0
            else:
                t_req = time_norm
            
            fields = self._get_fields_at_time(src, t_req, target_shape)
            source_fields.append(fields)
            
            # Recompute thickness curve from raw fields
            common_t_norm = np.linspace(0, 1, n_time_points)
            th_vals = []
            t_vals_nd = []
            t_max_nd = src['history'][-1]['t_nd'] if src.get('history') else 1.0
            
            for t_norm in common_t_norm:
                fields_t = self._get_fields_at_time(src, t_norm, target_shape)
                th_nd = DepositionPhysics.shell_thickness(
                    fields_t['phi'], fields_t['psi'],
                    params.get('fc', 0.18)
                )
                th_nm = th_nd * params.get('L0_nm', 20.0)
                th_vals.append(th_nm)
                t_vals_nd.append(t_norm * t_max_nd)
            
            # ---------- FIX: ensure strictly increasing thickness ----------
            th_vals = np.array(th_vals, dtype=float)
            if len(th_vals) > 1:
                # Force non-decreasing (cumulative max) and then strictly increasing (+ tiny epsilon)
                th_vals = np.maximum.accumulate(th_vals)
                th_vals += np.arange(len(th_vals)) * 1e-14
            t_vals_nd = np.array(t_vals_nd)   # also ensure array
            
            source_thickness.append({
                't_norm': common_t_norm,
                'th_nm': th_vals,
                't_nd': t_vals_nd,
                't_max': t_max_nd
            })
            source_t_max_nd.append(t_max_nd)
            source_tau0.append(params['tau0_s'])
        
        if not source_params:
            st.error("No valid source fields.")
            return None
        
        target_L0 = target_params.get('L0_nm', 20.0)
        
        alpha = self.compute_alpha(source_params, target_L0)
        beta, individual_param_weights = self.compute_beta(source_params, target_params)
        
        beta_norm = beta / (np.sum(beta) + 1e-12)
        gamma = self.compute_gamma(source_fields, source_params, target_params, 
                                  t_req if t_req is not None else 1.0, beta_norm)
        
        refinement_factor = alpha * beta * (1.0 + self.lambda_shape * gamma)
        
        source_features = self.encode_parameters(source_params)
        target_features = self.encode_parameters([target_params])
        all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
        
        proj = self.input_proj(all_features)
        proj = self.pos_encoder(proj)
        transformer_out = self.transformer(proj)
        
        target_rep = transformer_out[:, 0, :]
        source_reps = transformer_out[:, 1:, :]
        
        attn_scores = torch.matmul(target_rep.unsqueeze(1), 
                                  source_reps.transpose(1,2)).squeeze(1)
        attn_scores = attn_scores / np.sqrt(self.d_model) / self.temperature
        
        final_scores = attn_scores * torch.FloatTensor(refinement_factor).unsqueeze(0)
        final_weights = torch.softmax(final_scores, dim=-1).squeeze().detach().cpu().numpy()
        
        if np.isscalar(final_weights):
            final_weights = np.array([final_weights])
        elif final_weights.ndim == 0:
            final_weights = np.array([final_weights.item()])
        elif final_weights.ndim > 1:
            final_weights = final_weights.flatten()
        
        attn_np = attn_scores.squeeze().detach().cpu().numpy()
        if np.isscalar(attn_np):
            attn_np = np.array([attn_np])
        elif attn_np.ndim == 0:
            attn_np = np.array([attn_np.item()])
        elif attn_np.ndim > 1:
            attn_np = attn_np.flatten()
        
        if len(final_weights) != len(source_fields):
            st.warning(f"Weight length mismatch: {len(final_weights)} vs {len(source_fields)}. Truncating/padding.")
            if len(final_weights) > len(source_fields):
                final_weights = final_weights[:len(source_fields)]
            else:
                final_weights = np.pad(final_weights, (0, len(source_fields)-len(final_weights)),
                                      'constant', constant_values=0)
        
        eps = 1e-10
        entropy = -np.sum(final_weights * np.log(final_weights + eps))
        max_weight = np.max(final_weights)
        effective_sources = np.sum(final_weights > 0.01)
        
        interp = {'phi': np.zeros(target_shape),
                 'c': np.zeros(target_shape),
                 'psi': np.zeros(target_shape)}
        
        for i, fld in enumerate(source_fields):
            interp['phi'] += final_weights[i] * fld['phi']
            interp['c'] += final_weights[i] * fld['c']
            interp['psi'] += final_weights[i] * fld['psi']
        
        interp['phi'] = gaussian_filter(interp['phi'], sigma=1.0)
        interp['c'] = gaussian_filter(interp['c'], sigma=1.0)
        interp['psi'] = gaussian_filter(interp['psi'], sigma=1.0)
        
        common_t_norm = np.linspace(0, 1, n_time_points)
        thickness_curves = []
        for i, thick in enumerate(source_thickness):
            if len(thick['t_norm']) > 1:
                f = interp1d(thick['t_norm'], thick['th_nm'],
                           kind='linear', bounds_error=False,
                           fill_value=(thick['th_nm'][0], thick['th_nm'][-1]))
                th_interp = f(common_t_norm)
            else:
                th_interp = np.full_like(common_t_norm,
                                        thick['th_nm'][0] if len(thick['th_nm']) > 0 else 0.0)
            thickness_curves.append(th_interp)
        
        thickness_interp = np.zeros_like(common_t_norm)
        for i, curve in enumerate(thickness_curves):
            thickness_interp += final_weights[i] * curve
        
        avg_tau0 = np.average(source_tau0, weights=final_weights)
        avg_t_max_nd = np.average(source_t_max_nd, weights=final_weights)
        if target_params.get('tau0_s') is not None:
            avg_tau0 = target_params['tau0_s']
        
        common_t_real = common_t_norm * avg_t_max_nd * avg_tau0
        
        if time_norm is not None:
            t_real = time_norm * avg_t_max_nd * avg_tau0
        else:
            t_real = avg_t_max_nd * avg_tau0
        
        material = DepositionPhysics.material_proxy(interp['phi'], interp['psi'])
        alpha_phys = target_params.get('alpha_nd', 2.0)
        potential = DepositionPhysics.potential_proxy(interp['c'], alpha_phys)
        
        fc = target_params.get('fc', target_params.get('core_radius_frac', 0.18))
        dx = 1.0 / (target_shape[0] - 1)
        thickness_nd = DepositionPhysics.shell_thickness(interp['phi'], interp['psi'], fc, dx=dx)
        L0 = target_params.get('L0_nm', 20.0) * 1e-9
        thickness_nm = thickness_nd * L0 * 1e9
        
        stats = DepositionPhysics.phase_stats(interp['phi'], interp['psi'], dx, dx, L0)
        
        growth_rate = 0.0
        if time_norm is not None and len(thickness_curves) > 0:
            idx = int(time_norm * (len(common_t_norm) - 1))
            if idx > 0:
                dt_norm = common_t_norm[idx] - common_t_norm[idx-1]
                dt_real = dt_norm * avg_t_max_nd * avg_tau0
                dth = thickness_interp[idx] - thickness_interp[idx-1]
                growth_rate = dth / dt_real if dt_real > 0 else 0.0
        
        sources_data = []
        for i, (src_params, alpha_w, beta_w, gamma_w, indiv_weights, combined_w, attn_w) in enumerate(zip(
            source_params, alpha, beta, gamma,
            [dict(fc=individual_param_weights['fc'][i],
                 rs=individual_param_weights['rs'][i],
                 c_bulk=individual_param_weights['c_bulk'][i]) for i in range(len(source_params))],
            final_weights, attn_np
        )):
            sources_data.append({
                'source_index': i,
                'L0_nm': src_params.get('L0_nm', 20.0),
                'fc': src_params.get('fc', 0.18),
                'rs': src_params.get('rs', 0.2),
                'c_bulk': src_params.get('c_bulk', 0.5),
                'l0_weight': float(alpha_w),
                'fc_weight': float(indiv_weights['fc']),
                'rs_weight': float(indiv_weights['rs']),
                'c_bulk_weight': float(indiv_weights['c_bulk']),
                'beta_weight': float(beta_w),
                'gamma_weight': float(gamma_w),
                'attention_weight': float(attn_w),
                'physics_refinement': float(alpha_w * beta_w * (1.0 + self.lambda_shape * gamma_w)),
                'combined_weight': float(combined_w)
            })
        
        result = {
            'fields': interp,
            'derived': {
                'material': material,
                'potential': potential,
                'thickness_nm': thickness_nm,
                'growth_rate': growth_rate,
                'phase_stats': stats,
                'thickness_time': {
                    't_norm': common_t_norm.tolist(),
                    't_real_s': common_t_real.tolist(),
                    'th_nm': thickness_interp.tolist()
                }
            },
            'weights': {
                'combined': final_weights.tolist(),
                'alpha': alpha.tolist(),
                'beta': beta.tolist(),
                'gamma': gamma.tolist(),
                'individual_params': individual_param_weights,
                'refinement_factor': refinement_factor.tolist(),
                'attention': attn_np.tolist(),
                'entropy': float(entropy),
                'max_weight': float(max_weight),
                'effective_sources': int(effective_sources)
            },
            'sources_data': sources_data,
            'target_params': target_params,
            'shape': target_shape,
            'num_sources': len(source_fields),
            'source_params': source_params,
            'time_norm': t_req,
            'time_real_s': t_real,
            'avg_tau0': avg_tau0,
            'avg_t_max_nd': avg_t_max_nd,
            'filter_stats': filter_stats
        }
        
        return result

# =============================================
# TEMPORAL CACHE SYSTEM
# =============================================
@dataclass
class TemporalCacheEntry:
    """Lightweight container for cached temporal data."""
    time_norm: float
    time_real_s: float
    fields: Optional[Dict[str, np.ndarray]] = None
    thickness_nm: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_size_mb(self) -> float:
        if self.fields is None:
            return 0.001
        total_bytes = sum(arr.nbytes for arr in self.fields.values())
        return total_bytes / (1024 * 1024)

# =============================================
# TEMPORAL FIELD MANAGER (with progress‑based interpolation)
# =============================================
class TemporalFieldManager:
    """
    Three-tier temporal management system with hierarchical source filtering.
    Uses progress (thickness/max_th) for interpolation to capture non‑linear growth.
    """
    
    def __init__(self, interpolator, sources: List[Dict], target_params: Dict,
                 n_key_frames: int = 10, lru_size: int = 3,
                 require_categorical_match: bool = False):
        self.interpolator = interpolator
        self.target_params = target_params
        self.n_key_frames = n_key_frames
        self.lru_size = lru_size
        self.require_categorical_match = require_categorical_match
        
        self.sources, self.filter_stats = interpolator.filter_sources_hierarchy(
            sources, target_params, require_categorical_match=require_categorical_match
        )
        self._use_fallback = False
        
        if self.filter_stats:
            kept = self.filter_stats.get('kept', 0)
            total = len(sources)
            if kept < total:
                st.info(f"🛡️ Hard Masking: {kept}/{total} sources compatible. "
                       f"(Excluded: {self.filter_stats.get('categorical', 0)} cat)")
            if kept == 0:
                st.warning("⚠️ No compatible sources found. Using nearest neighbor fallback.")
                self._use_fallback = True
            else:
                st.success(f"✅ {total} sources compatible.")
        
        if not self.sources:
            self.sources = sources
            self._use_fallback = True
        
        self.avg_tau0 = None
        self.avg_t_max_nd = None
        self.thickness_time: Optional[Dict] = None
        self.weights: Optional[Dict] = None
        self.sources_data: Optional[List] = None
        self._compute_thickness_curve()
        
        # Progress‑related attributes (NEW)
        self.progress_splines: List[Optional[CubicSpline]] = []   # map thickness progress → t_norm
        self.max_thickness_per_source: List[float] = []           # max thickness of each source
        
        self.key_times: np.ndarray = np.linspace(0, 1, n_key_frames)
        self.key_frames: Dict[float, Dict[str, np.ndarray]] = {}
        self.key_thickness: Dict[float, float] = {}
        self.key_time_real: Dict[float, float] = {}
        self._precompute_key_frames()
        
        self.lru_cache: OrderedDict[float, TemporalCacheEntry] = OrderedDict()
        self.animation_temp_dir: Optional[str] = None
        self.animation_frame_paths: List[str] = []
    
    def _compute_thickness_curve(self):
        res = self.interpolator.interpolate_fields(
            self.sources, self.target_params, target_shape=(256, 256),
            n_time_points=100, time_norm=None, recompute_thickness=True
        )
        if res:
            self.thickness_time = res['derived']['thickness_time']
            self.weights = res['weights']
            self.sources_data = res.get('sources_data', [])
            self.avg_tau0 = res.get('avg_tau0', 1e-4)
            self.avg_t_max_nd = res.get('avg_t_max_nd', 1.0)
        else:
            self.thickness_time = {'t_norm': [0, 1], 'th_nm': [0, 0], 't_real_s': [0, 0]}
            self.weights = {'combined': [1.0], 'attention': [0.0], 'entropy': 0.0}
            self.sources_data = []
            self.avg_tau0 = 1e-4
            self.avg_t_max_nd = 1.0
        
        # Build progress splines for each source (NEW)
        self.progress_splines = []
        self.max_thickness_per_source = []
        for src in self.sources:
            params = src['params']
            # Compute thickness curve for this source
            common_t_norm = np.linspace(0, 1, 100)
            th_vals = []
            for t_norm in common_t_norm:
                fields_t = self.interpolator._get_fields_at_time(src, t_norm, (256,256))
                th_nd = DepositionPhysics.shell_thickness(
                    fields_t['phi'], fields_t['psi'],
                    params.get('fc', 0.18)
                )
                th_nm = th_nd * params.get('L0_nm', 20.0)
                th_vals.append(th_nm)
            
            # ---------- FIX: ensure strictly increasing thickness ----------
            th_vals = np.array(th_vals, dtype=float)
            if len(th_vals) > 1:
                # Force non-decreasing (cumulative max) and then strictly increasing (+ tiny epsilon)
                th_vals = np.maximum.accumulate(th_vals)
                th_vals += np.arange(len(th_vals)) * 1e-14

            max_th = th_vals[-1] if len(th_vals) > 0 else 1.0
            self.max_thickness_per_source.append(max_th)
            if max_th > 1e-9:
                progress = th_vals / max_th
                # Spline: progress -> t_norm
                if len(np.unique(progress)) >= 2:
                    spl = CubicSpline(progress, common_t_norm, extrapolate=True, bc_type='natural')
                else:
                    spl = None
            else:
                spl = None
            self.progress_splines.append(spl)
    
    def _precompute_key_frames(self):
        st.info(f"Pre-computing {self.n_key_frames} key frames using progress‑based time...")
        progress_bar = st.progress(0)
        for i, t in enumerate(self.key_times):
            # t is target normalized time (0..1). Convert to progress using the interpolated thickness curve.
            if self.thickness_time and len(self.thickness_time['th_nm']) > 1:
                # interpolate thickness at t
                th_t = np.interp(t, self.thickness_time['t_norm'], self.thickness_time['th_nm'])
                # target progress = th_t / max_thickness (where max is last thickness)
                max_th = self.thickness_time['th_nm'][-1] if self.thickness_time['th_nm'] else 1.0
                progress_target = th_t / max_th if max_th > 0 else t
            else:
                progress_target = t
            # Now we need fields at progress_target. But our key frames are stored at original t_norm.
            # Instead, we compute fields at t (the normalized time) – we keep original key frames for consistency.
            # However, when retrieving, we will map requested time to progress and then blend key frames.
            # For pre-computation, we still compute at t_norm = t (this is okay, because key frames are evenly spaced in t_norm).
            # The actual interpolation logic in get_fields will use progress to blend.
            res = self.interpolator.interpolate_fields(
                self.sources, self.target_params, target_shape=(256, 256),
                n_time_points=100, time_norm=t, recompute_thickness=True
            )
            if res:
                self.key_frames[t] = res['fields']
                self.key_thickness[t] = res['derived']['thickness_nm']
                self.key_time_real[t] = res.get('time_real_s', 0.0)
            progress_bar.progress((i + 1) / self.n_key_frames)
        progress_bar.empty()
        st.success(f"Key frames ready. Memory: ~{self._estimate_key_frame_memory():.1f} MB")
    
    def _estimate_key_frame_memory(self) -> float:
        if not self.key_frames:
            return 0.0
        sample_frame = next(iter(self.key_frames.values()))
        bytes_per_frame = sum(arr.nbytes for arr in sample_frame.values())
        return (bytes_per_frame * len(self.key_frames)) / (1024 * 1024)
    
    def get_fields(self, time_norm: float, use_interpolation: bool = True) -> Dict[str, np.ndarray]:
        """Return fields at given normalized time, using progress‑based blending."""
        t_key = round(time_norm, 4)
        time_real = time_norm * self.avg_t_max_nd * self.avg_tau0 if self.avg_t_max_nd else 0.0
        
        # Check LRU cache
        if t_key in self.lru_cache:
            entry = self.lru_cache.pop(t_key)
            self.lru_cache[t_key] = entry
            return entry.fields
        
        # Map requested time_norm to progress (NEW)
        if self.thickness_time and len(self.thickness_time['th_nm']) > 1:
            th_t = np.interp(time_norm, self.thickness_time['t_norm'], self.thickness_time['th_nm'])
            max_th = self.thickness_time['th_nm'][-1] if self.thickness_time['th_nm'] else 1.0
            progress_target = th_t / max_th if max_th > 0 else time_norm
        else:
            progress_target = time_norm
        
        # Find key frames that bound the progress_target (using the stored t_norm of key frames)
        key_times_arr = np.array(list(self.key_frames.keys()))
        # We need to map each key frame's t_norm to its thickness progress.
        key_progress = []
        for t in key_times_arr:
            th_t = np.interp(t, self.thickness_time['t_norm'], self.thickness_time['th_nm'])
            max_th = self.thickness_time['th_nm'][-1]
            key_progress.append(th_t / max_th if max_th > 0 else t)
        key_progress = np.array(key_progress)
        
        # Find interpolation indices in progress space
        idx = np.searchsorted(key_progress, progress_target)
        
        if idx == 0:
            # before first key frame
            t0 = key_times_arr[0]
            fields = self.key_frames[t0]
            self._add_to_lru(t_key, fields, self.key_thickness[t0], time_real)
            return fields
        elif idx >= len(key_progress):
            # after last key frame
            t1 = key_times_arr[-1]
            fields = self.key_frames[t1]
            self._add_to_lru(t_key, fields, self.key_thickness[t1], time_real)
            return fields
        
        # linear interpolation in progress space
        t0, t1 = key_times_arr[idx-1], key_times_arr[idx]
        p0, p1 = key_progress[idx-1], key_progress[idx]
        if p1 - p0 > 0:
            alpha = (progress_target - p0) / (p1 - p0)
        else:
            alpha = 0.0
        f0, f1 = self.key_frames[t0], self.key_frames[t1]
        th0, th1 = self.key_thickness[t0], self.key_thickness[t1]
        
        interp_fields = {}
        for key in f0:
            interp_fields[key] = (1 - alpha) * f0[key] + alpha * f1[key]
        interp_thickness = (1 - alpha) * th0 + alpha * th1
        
        self._add_to_lru(t_key, interp_fields, interp_thickness, time_real)
        return interp_fields
    
    def _add_to_lru(self, time_norm: float, fields: Dict[str, np.ndarray],
                   thickness_nm: float, time_real_s: float):
        if time_norm in self.lru_cache:
            del self.lru_cache[time_norm]
        while len(self.lru_cache) >= self.lru_size:
            oldest_key = next(iter(self.lru_cache))
            del self.lru_cache[oldest_key]
        self.lru_cache[time_norm] = TemporalCacheEntry(
            time_norm=time_norm,
            time_real_s=time_real_s,
            fields=fields,
            thickness_nm=thickness_nm
        )
    
    def get_thickness_at_time(self, time_norm: float) -> float:
        if self.thickness_time is None:
            return 0.0
        t_arr = np.array(self.thickness_time['t_norm'])
        th_arr = np.array(self.thickness_time['th_nm'])
        if time_norm <= t_arr[0]:
            return th_arr[0]
        if time_norm >= t_arr[-1]:
            return th_arr[-1]
        return np.interp(time_norm, t_arr, th_arr)
    
    def get_time_real(self, time_norm: float) -> float:
        return time_norm * self.avg_t_max_nd * self.avg_tau0 if self.avg_t_max_nd else 0.0
    
    def prepare_animation_streaming(self, n_frames: int = 50) -> List[str]:
        import tempfile
        self.animation_temp_dir = tempfile.mkdtemp(dir=TEMP_ANIMATION_DIR)
        self.animation_frame_paths = []
        times = np.linspace(0, 1, n_frames)
        st.info(f"Pre-rendering {n_frames} animation frames to disk...")
        progress = st.progress(0)
        for i, t in enumerate(times):
            fields = self.get_fields(t, use_interpolation=True)
            time_real = self.get_time_real(t)
            frame_path = os.path.join(self.animation_temp_dir, f"frame_{i:04d}.npz")
            np.savez_compressed(frame_path,
                               phi=fields['phi'], c=fields['c'], psi=fields['psi'],
                               time_norm=t, time_real_s=time_real)
            self.animation_frame_paths.append(frame_path)
            progress.progress((i + 1) / n_frames)
        progress.empty()
        return self.animation_frame_paths
    
    def get_animation_frame(self, frame_idx: int) -> Optional[Dict[str, np.ndarray]]:
        if not self.animation_frame_paths or frame_idx >= len(self.animation_frame_paths):
            return None
        data = np.load(self.animation_frame_paths[frame_idx])
        return {
            'phi': data['phi'], 'c': data['c'], 'psi': data['psi'],
            'time_norm': float(data['time_norm']),
            'time_real_s': float(data['time_real_s'])
        }
    
    def cleanup_animation(self):
        if self.animation_temp_dir and os.path.exists(self.animation_temp_dir):
            shutil.rmtree(self.animation_temp_dir)
        self.animation_temp_dir = None
        self.animation_frame_paths = []
    
    def get_memory_stats(self) -> Dict[str, float]:
        lru_memory = sum(entry.get_size_mb() for entry in self.lru_cache.values())
        key_memory = self._estimate_key_frame_memory()
        return {
            'lru_cache_mb': lru_memory,
            'key_frames_mb': key_memory,
            'total_mb': lru_memory + key_memory,
            'lru_entries': len(self.lru_cache),
            'key_frame_entries': len(self.key_frames)
        }
    
    def clear_lru_cache(self):
        self.lru_cache.clear()
        st.sidebar.info("LRU cache cleared.")
    
    def recompute_key_frames(self):
        st.sidebar.info("Recomputing key frames...")
        self.key_frames.clear()
        self.key_thickness.clear()
        self.key_time_real.clear()
        self._precompute_key_frames()
        st.sidebar.success("Key frames recomputed.")

# =============================================
# ROBUST SOLUTION LOADER
# =============================================
class EnhancedSolutionLoader:
    """Loads PKL files from numerical_solutions, parsing filenames as fallback."""
    
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
    
    def _ensure_directory(self):
        os.makedirs(self.solutions_dir, exist_ok=True)
    
    def scan_solutions(self) -> List[Dict[str, Any]]:
        import glob
        all_files = []
        for ext in ['*.pkl', '*.pickle']:
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
    
    def parse_filename(self, filename: str) -> Dict[str, any]:
        params = {}
        mode_match = re.search(r'_(2D|3D)_', filename)
        if mode_match:
            params['mode'] = '2D (planar)' if mode_match.group(1) == '2D' else '3D (spherical)'
        
        c_match = re.search(r'_c([0-9.]+)_', filename)
        if c_match:
            params['c_bulk'] = float(c_match.group(1))
        
        L_match = re.search(r'_L0([0-9.]+)nm', filename)
        if L_match:
            params['L0_nm'] = float(L_match.group(1))
        
        fc_match = re.search(r'_fc([0-9.]+)_', filename)
        if fc_match:
            params['fc'] = float(fc_match.group(1))
        
        rs_match = re.search(r'_rs([0-9.]+)_', filename)
        if rs_match:
            params['rs'] = float(rs_match.group(1))
        
        if 'Neu' in filename:
            params['bc_type'] = 'Neu'
        elif 'Dir' in filename:
            params['bc_type'] = 'Dir'
        
        if 'noEDL' in filename:
            params['use_edl'] = False
        elif 'EDL' in filename:
            params['use_edl'] = True
            edl_match = re.search(r'EDL([0-9.]+)', filename)
            if edl_match:
                params['lambda0_edl'] = float(edl_match.group(1))
        
        k_match = re.search(r'_k([0-9.]+)_', filename)
        if k_match:
            params['k0_nd'] = float(k_match.group(1))
        
        M_match = re.search(r'_M([0-9.]+)_', filename)
        if M_match:
            params['M_nd'] = float(M_match.group(1))
        
        D_match = re.search(r'_D([0-9.]+)_', filename)
        if D_match:
            params['D_nd'] = float(D_match.group(1))
        
        Nx_match = re.search(r'_Nx(\d+)_', filename)
        if Nx_match:
            params['Nx'] = int(Nx_match.group(1))
        
        steps_match = re.search(r'_steps(\d+)\.', filename)
        if steps_match:
            params['n_steps'] = int(steps_match.group(1))
        
        tau_match = re.search(r'_tau0([0-9.eE+-]+)s', filename)
        if tau_match:
            params['tau0_s'] = float(tau_match.group(1))
        
        return params
    
    def _ensure_2d(self, arr):
        if arr is None:
            return np.zeros((1, 1))
        if torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        if arr.ndim == 3:
            mid = arr.shape[0] // 2
            return arr[mid, :, :]
        elif arr.ndim == 1:
            n = int(np.sqrt(arr.size))
            return arr[:n*n].reshape(n, n)
        else:
            return arr
    
    def _convert_tensors(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.cpu().numpy()
                elif isinstance(value, (dict, list)):
                    self._convert_tensors(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if torch.is_tensor(item):
                    data[i] = item.cpu().numpy()
                elif isinstance(item, (dict, list)):
                    self._convert_tensors(item)
    
    def read_simulation_file(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            standardized = {
                'params': {},
                'history': [],
                'thickness_history': [],
                'metadata': {
                    'filename': os.path.basename(file_path),
                    'loaded_at': datetime.now().isoformat(),
                }
            }
            
            if isinstance(data, dict):
                if 'parameters' in data and isinstance(data['parameters'], dict):
                    standardized['params'].update(data['parameters'])
                if 'meta' in data and isinstance(data['meta'], dict):
                    standardized['params'].update(data['meta'])
                
                standardized['coords_nd'] = data.get('coords_nd', None)
                standardized['diagnostics'] = data.get('diagnostics', [])
                
                if 'thickness_history_nm' in data:
                    thick_list = []
                    for entry in data['thickness_history_nm']:
                        if len(entry) >= 3:
                            thick_list.append({
                                't_nd': entry[0],
                                'th_nd': entry[1],
                                'th_nm': entry[2]
                            })
                    standardized['thickness_history'] = thick_list
                
                if 'snapshots' in data and isinstance(data['snapshots'], list):
                    snap_list = []
                    for snap in data['snapshots']:
                        if isinstance(snap, tuple) and len(snap) == 4:
                            t, phi, c, psi = snap
                            snap_dict = {
                                't_nd': t,
                                'phi': self._ensure_2d(phi),
                                'c': self._ensure_2d(c),
                                'psi': self._ensure_2d(psi)
                            }
                            snap_list.append(snap_dict)
                        elif isinstance(snap, dict):
                            snap_dict = {
                                't_nd': snap.get('t_nd', 0),
                                'phi': self._ensure_2d(snap.get('phi', np.zeros((1,1)))),
                                'c': self._ensure_2d(snap.get('c', np.zeros((1,1)))),
                                'psi': self._ensure_2d(snap.get('psi', np.zeros((1,1))))
                            }
                            snap_list.append(snap_dict)
                    standardized['history'] = snap_list
                
                if not standardized['params']:
                    parsed = self.parse_filename(os.path.basename(file_path))
                    standardized['params'].update(parsed)
                    st.sidebar.info(f"Parsed parameters from filename: {os.path.basename(file_path)}")
                
                params = standardized['params']
                params.setdefault('fc', params.get('core_radius_frac', 0.18))
                params.setdefault('rs', params.get('shell_thickness_frac', 0.2))
                params.setdefault('c_bulk', params.get('c_bulk', 1.0))
                params.setdefault('L0_nm', params.get('L0_nm', 20.0))
                params.setdefault('bc_type', params.get('bc_type', 'Neu'))
                params.setdefault('use_edl', params.get('use_edl', False))
                params.setdefault('mode', params.get('mode', '2D (planar)'))
                params.setdefault('growth_model', params.get('growth_model', 'Model A'))
                params.setdefault('alpha_nd', params.get('alpha_nd', 2.0))
                params.setdefault('tau0_s', params.get('tau0_s', 1e-4))
                
                if not standardized['history']:
                    st.sidebar.warning(f"No snapshots in {os.path.basename(file_path)}")
                    return None
            
            self._convert_tensors(standardized)
            return standardized
        except Exception as e:
            st.sidebar.error(f"Error loading {os.path.basename(file_path)}: {e}")
            return None
    
    def load_all_solutions(self, use_cache=True, max_files=None):
        solutions = []
        file_info = self.scan_solutions()
        if max_files:
            file_info = file_info[:max_files]
        
        if not file_info:
            st.sidebar.warning("No PKL files found in numerical_solutions directory.")
            return solutions
        
        for item in file_info:
            cache_key = item['filename']
            if use_cache and cache_key in self.cache:
                solutions.append(self.cache[cache_key])
                continue
            
            sol = self.read_simulation_file(item['path'])
            if sol:
                self.cache[cache_key] = sol
                solutions.append(sol)
        
        st.sidebar.success(f"Loaded {len(solutions)} solutions.")
        return solutions

# =============================================
# INTELLIGENT DESIGNER MODULES (NLP INTERFACE)
# =============================================

class NLParser:
    """
    Extract deposition parameters from natural language input using regex or LLM (GPT‑2 / Qwen).
    """
    def __init__(self):
        self.defaults = {
            'fc': 0.18,
            'rs': 0.2,
            'c_bulk': 0.5,
            'L0_nm': 60.0,
            'time': None,
            'bc_type': 'Neu',
            'use_edl': True,
            'mode': '2D (planar)',
            'alpha_nd': 2.0,
            'tau0_s': 1e-4
        }
        self.patterns = {
            'L0_nm': [
                r'L0\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)?',
                r'(?:domain|box|length|size)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)',
                r'(\d+(?:\.\d+)?)\s*nm\s*(?:domain|box|length)',
            ],
            'fc': [
                r'fc\s*[=:]\s*(\d+(?:\.\d+)?)',
                r'core\s*(?:fraction|ratio|radius)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)',
                r'core\s*[=:]\s*(\d+(?:\.\d+)?)',
            ],
            'rs': [
                r'rs\s*[=:]\s*(\d+(?:\.\d+)?)',
                r'shell\s*(?:thickness\s*)?(?:fraction|ratio)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)',
                r'shell\s*[=:]\s*(\d+(?:\.\d+)?)',
            ],
            'c_bulk': [
                r'c[_-]?bulk\s*[=:]\s*(\d+(?:\.\d+)?)',
                r'concentration\s*(?:ratio|fraction)?\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)',
                r'c\s*[=:]\s*(\d+(?:\.\d+)?)(?!\s*nm)',
            ],
            'time': [
                r'time\s*[=:]\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:s|sec|seconds?)?',
                r'(?:at|for)\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:s|sec|seconds?)',
                r't\s*[=:]\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:s|sec)?',
            ],
            'bc_type': [
                r'bc[_-]?type\s*[=:]\s*(Neu|Dir|neumann|dirichlet)',
                r'boundary\s*(?:condition|type)?\s*(?:is|=|:)?\s*(Neumann|Dirichlet|neu|dir)',
            ],
            'use_edl': [
                r'use[_-]?edl\s*[=:]\s*(True|False|true|false|1|0|yes|no)',
                r'EDL\s*(?:enabled|disabled|on|off|true|false)',
            ],
            'mode': [
                r'mode\s*[=:]\s*([23]D\s*\([^)]+\)|[23]D)',
                r'(2D|3D)\s*(?:planar|spherical)?',
            ],
            'alpha_nd': [
                r'alpha[_-]?nd\s*[=:]\s*(\d+(?:\.\d+)?)',
                r'alpha\s*[=:]\s*(\d+(?:\.\d+)?)',
                r'coupling\s*(?:constant|parameter)?\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)',
            ],
        }

    # NEW: Improved LLM-based parsing with robust JSON extraction and fallback
    def parse_with_llm(self, text: str, tokenizer, model, regex_params=None, temperature=None) -> dict:
        """Use LLM (GPT‑2 or Qwen) to extract parameters from natural language.
        If temperature is None, uses 0.0 for Qwen, 0.1 for GPT‑2.
        """
        if not text:
            return self.defaults.copy()
        
        backend = st.session_state.get('llm_backend_loaded', 'GPT-2 (default, fastest startup)')
        if temperature is None:
            temperature = 0.0 if "Qwen" in backend else 0.1
        
        # Build a prompt with few-shot examples
        system = "You are an expert extractor. Always reply with valid JSON only. Clip every value to its exact range."
        examples = """
Examples:
- "L0=50 nm, fc=0.2, c_bulk=0.3" → {"L0_nm": 50.0, "fc": 0.2, "rs": 0.2, "c_bulk": 0.3, "time": null, "bc_type": "Neu", "use_edl": true, "mode": "2D (planar)", "alpha_nd": 2.0, "tau0_s": 0.0001}
- "Design a core-shell with L0=60, fc=0.25, time=1e-3 s" → {"L0_nm": 60.0, "fc": 0.25, "rs": 0.2, "c_bulk": 0.5, "time": 0.001, "bc_type": "Neu", "use_edl": true, "mode": "2D (planar)", "alpha_nd": 2.0, "tau0_s": 0.0001}
- "Domain 70nm, core fraction 0.22, concentration 0.4, run for 0.001 seconds" → {"L0_nm": 70.0, "fc": 0.22, "rs": 0.2, "c_bulk": 0.4, "time": 0.001, "bc_type": "Neu", "use_edl": true, "mode": "2D (planar)", "alpha_nd": 2.0, "tau0_s": 0.0001}
- "I need a thin shell: L0=40 nm, core 0.2, conc 0.15" → {"L0_nm": 40.0, "fc": 0.2, "rs": 0.2, "c_bulk": 0.15, "time": null, "bc_type": "Neu", "use_edl": true, "mode": "2D (planar)", "alpha_nd": 2.0, "tau0_s": 0.0001}
"""
        defaults_json = json.dumps(self.defaults, default=str)
        regex_hint = ""
        if regex_params is not None:
            regex_hint = f"\nPreliminary regex extraction (use as a hint, but prioritize the text): {json.dumps(regex_params, default=str)}"
        
        user = f"""{examples}
{regex_hint}
Text: "{text}"

Output ONLY the JSON object with keys: L0_nm (float, domain size in nm, range 10-100), fc (float, core fraction, range 0.05-0.45), rs (float, shell ratio, range 0.01-0.6), c_bulk (float, concentration, range 0.1-1.0), time (float or null, seconds), bc_type (str, either "Neu" or "Dir"), use_edl (bool, true/false), mode (str, either "2D (planar)" or "3D (spherical)"), alpha_nd (float, coupling constant, typically 2.0), tau0_s (float, time scale, typically 1e-4).

Defaults: {defaults_json}

JSON:"""

        # Use chat template for Qwen models
        if "Qwen" in backend:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system}\n\n{user}\nJSON:\n"

        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=300,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Robust JSON extraction
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            match = re.search(json_pattern, generated, re.DOTALL)
            if not match:
                # Try to find anything that looks like a JSON dict
                match = re.search(r'\{.*\}', generated, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Attempt to parse JSON
                params = json.loads(json_str)
                # Ensure all required keys exist
                for key in self.defaults:
                    if key not in params:
                        params[key] = self.defaults[key]
                # Clip values to valid ranges
                for p in ['fc', 'rs', 'c_bulk', 'L0_nm']:
                    if p in params and isinstance(params[p], (int, float)):
                        low, high = DepositionParameters.RANGES[p]
                        params[p] = np.clip(params[p], low, high)
                # Post-validation: if regex_params provided, check for mismatches
                if regex_params is not None:
                    for key in ['L0_nm', 'fc', 'rs', 'c_bulk']:
                        if key in regex_params and key in params:
                            # if regex extracted a value and LLM's differs significantly, warn and use regex
                            if abs(params[key] - regex_params[key]) > 1e-3:
                                st.warning(f"LLM {key}={params[key]} mismatches regex {regex_params[key]}. Using regex for this parameter.")
                                params[key] = regex_params[key]
                return params
            else:
                # Fallback: attempt to extract individual parameters using regex
                st.warning("LLM did not produce a valid JSON. Falling back to regex extraction.")
                return self._fallback_extract(generated)
        except Exception as e:
            st.warning(f"LLM parsing failed: {e}. Falling back to regex.")
            return self.parse(text)  # fallback to regex

    # Ensemble method – run multiple generations and combine by averaging/consensus
    def parse_with_llm_ensemble(self, text: str, tokenizer, model, n_runs=3, regex_params=None, temperature=0.2) -> dict:
        """Run LLM multiple times and combine results (average for floats, mode for categorical)."""
        all_params = []
        for _ in range(n_runs):
            params = self.parse_with_llm(text, tokenizer, model, regex_params, temperature=temperature)
            all_params.append(params)
        
        # Combine
        combined = {}
        for key in self.defaults:
            values = [p[key] for p in all_params]
            if isinstance(self.defaults[key], bool):
                # mode for boolean
                combined[key] = max(set(values), key=values.count)
            elif isinstance(self.defaults[key], str):
                # mode for string
                combined[key] = max(set(values), key=values.count)
            elif key == 'time' and self.defaults[key] is None:
                # time can be None or float; take most common, if tie, average?
                non_none = [v for v in values if v is not None]
                if non_none:
                    combined[key] = np.mean(non_none)
                else:
                    combined[key] = None
            else:
                # average for numeric
                combined[key] = np.mean([v for v in values if v is not None])
        return combined

    # Hybrid parse – combines regex and LLM with confidence
    def hybrid_parse(self, text: str, tokenizer, model, use_ensemble=False, ensemble_runs=3) -> dict:
        """Run regex first, then LLM (optionally ensemble), and merge based on confidence."""
        # Step 1: regex
        regex_params = self.parse(text)
        regex_conf = {}
        for key in self.defaults:
            # high confidence if value differs from default, low if default (may indicate no match)
            regex_conf[key] = 1.0 if regex_params[key] != self.defaults[key] else 0.0
        
        # Step 2: LLM if available
        if tokenizer is not None and model is not None:
            if use_ensemble:
                llm_params = self.parse_with_llm_ensemble(text, tokenizer, model, n_runs=ensemble_runs, regex_params=regex_params)
            else:
                llm_params = self.parse_with_llm(text, tokenizer, model, regex_params=regex_params)
            # Confidence for LLM: moderate if extracted, lower if default
            llm_conf = {}
            for key in self.defaults:
                llm_conf[key] = 0.7 if llm_params[key] != self.defaults[key] else 0.3
        else:
            llm_params = self.defaults.copy()
            llm_conf = {k: 0.0 for k in self.defaults}
        
        # Step 3: Merge with confidence
        final_params = {}
        for key in self.defaults:
            if regex_conf[key] > llm_conf[key]:
                final_params[key] = regex_params[key]
            elif llm_conf[key] > regex_conf[key]:
                final_params[key] = llm_params[key]
            else:
                # tie: prefer regex if it matched, else average if numeric
                if regex_conf[key] > 0:
                    final_params[key] = regex_params[key]
                elif isinstance(regex_params[key], (int, float)) and isinstance(llm_params[key], (int, float)):
                    final_params[key] = (regex_params[key] + llm_params[key]) / 2
                else:
                    final_params[key] = regex_params[key]  # fallback
        
        # Clip to ranges
        for p in ['fc', 'rs', 'c_bulk', 'L0_nm']:
            low, high = DepositionParameters.RANGES[p]
            final_params[p] = np.clip(final_params[p], low, high)
        
        return final_params

    def _fallback_extract(self, generated: str) -> dict:
        """Fallback: use regex to extract parameters from LLM's output if JSON fails."""
        params = self.defaults.copy()
        # Try to find numbers associated with known parameter names
        patterns = {
            'L0_nm': r'L0[:_]?\s*(\d+(?:\.\d+)?)',
            'fc': r'fc[:_]?\s*(\d+(?:\.\d+)?)',
            'rs': r'rs[:_]?\s*(\d+(?:\.\d+)?)',
            'c_bulk': r'c[_-]?bulk[:_]?\s*(\d+(?:\.\d+)?)',
            'time': r'time[:_]?\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)',
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, generated, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    if key in ['L0_nm', 'fc', 'rs', 'c_bulk']:
                        low, high = DepositionParameters.RANGES[key]
                        params[key] = np.clip(val, low, high)
                    else:
                        params[key] = val
                except:
                    pass
        # Handle boolean and categorical manually (not easily extracted via regex)
        return params

    def parse(self, text: str) -> dict:
        if not text or not isinstance(text, str):
            return self.defaults.copy()
        
        params = self.defaults.copy()
        text_lower = text.lower()
        
        for param_name, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value_str = match.group(1)
                    try:
                        if param_name == 'use_edl':
                            params[param_name] = value_str.lower() in ['true', '1', 'yes', 'on']
                        elif param_name == 'bc_type':
                            val = value_str.capitalize()
                            params[param_name] = 'Neu' if val.startswith('Neu') else 'Dir'
                        elif param_name == 'mode':
                            if '3d' in value_str.lower():
                                params[param_name] = '3D (spherical)'
                            else:
                                params[param_name] = '2D (planar)'
                        elif param_name == 'time':
                            if value_str:
                                params[param_name] = float(value_str)
                        else:
                            params[param_name] = float(value_str)
                        break
                    except (ValueError, TypeError):
                        continue
        
        for p in ['fc', 'rs', 'c_bulk', 'L0_nm']:
            low, high = DepositionParameters.RANGES[p]
            if not (low <= params[p] <= high):
                old_val = params[p]
                params[p] = np.clip(params[p], low, high)
                st.warning(f"Parameter {p}={old_val} out of range [{low}, {high}]; clipped to {params[p]}.")
        
        return params
    
    def get_explanation(self, params: dict, original_text: str) -> str:
        lines = ["### Parsed Parameters from Natural Language Input", ""]
        lines.append(f"**Original input:** _{original_text}_")
        lines.append("")
        lines.append("| Parameter | Value | Status |")
        lines.append("|-----------|-------|--------|")
        
        for key, val in params.items():
            if key == 'time' and val is None:
                status = "Default (full evolution)"
                val_str = "Full"
            else:
                status = "Extracted" if val != self.defaults[key] else "Default"
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            
            lines.append(f"| {key} | {val_str} | {status} |")
        
        return "\n".join(lines)


class RelevanceScorer:
    """Compute semantic relevance using SciBERT or fallback keyword matching."""
    _instance = None
    _model = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, use_scibert: bool = True):
        self.use_scibert = use_scibert
        self._embedding_cache = {}
        
        if use_scibert and RelevanceScorer._model is None:
            try:
                with RelevanceScorer._lock:
                    if RelevanceScorer._model is None:
                        with st.spinner("Loading SciBERT model for semantic analysis..."):
                            from sentence_transformers import SentenceTransformer
                            RelevanceScorer._model = SentenceTransformer(
                                'allenai/scibert_scivocab_uncased',
                                device='cpu'
                            )
                            st.success("SciBERT loaded successfully!")
                self.model = RelevanceScorer._model
            except ImportError:
                st.warning("sentence-transformers not installed. Using fallback relevance scoring.")
                self.use_scibert = False
            except Exception as e:
                st.warning(f"Could not load SciBERT: {e}. Using fallback.")
                self.use_scibert = False
    
    def encode_source(self, src_params: dict) -> str:
        return (
            f"Deposition simulation with domain length {src_params.get('L0_nm', 20):.1f} nm, "
            f"core fraction {src_params.get('fc', 0.18):.3f}, "
            f"shell ratio {src_params.get('rs', 0.2):.3f}, "
            f"bulk concentration ratio {src_params.get('c_bulk', 0.5):.2f}, "
            f"operating in {src_params.get('mode', '2D')} mode "
            f"with {src_params.get('bc_type', 'Neu')} boundary conditions "
            f"and EDL {'enabled' if src_params.get('use_edl', True) else 'disabled'}."
        )
    
    def score(self, query: str, sources: List[Dict], weights: np.ndarray) -> float:
        if not sources or len(weights) == 0:
            return 0.0
        
        if self.use_scibert and self.model is not None:
            try:
                query_hash = hashlib.md5(query.encode()).hexdigest()
                if query_hash not in self._embedding_cache:
                    query_emb = self.model.encode(query, convert_to_tensor=False)
                    self._embedding_cache[query_hash] = query_emb
                else:
                    query_emb = self._embedding_cache[query_hash]
                
                src_texts = [self.encode_source(s.get('params', {})) for s in sources]
                src_embs = self.model.encode(src_texts, convert_to_tensor=False)
                
                query_norm = np.linalg.norm(query_emb)
                src_norms = np.linalg.norm(src_embs, axis=1)
                
                valid_mask = src_norms > 1e-8
                if not np.any(valid_mask):
                    return float(np.max(weights))
                
                similarities = np.zeros(len(sources))
                similarities[valid_mask] = (
                    np.dot(src_embs[valid_mask], query_emb) / 
                    (src_norms[valid_mask] * query_norm + 1e-12)
                )
                
                weighted_score = np.average(similarities, weights=weights)
                normalized_score = (weighted_score + 1) / 2
                return float(np.clip(normalized_score, 0.0, 1.0))
                
            except Exception as e:
                st.warning(f"SciBERT scoring failed: {e}. Using fallback.")
                return float(np.max(weights)) if len(weights) > 0 else 0.0
        else:
            return float(np.max(weights)) if len(weights) > 0 else 0.0
    
    def get_confidence_level(self, score: float) -> Tuple[str, str]:
        if score >= 0.8:
            return "High confidence", "green"
        elif score >= 0.5:
            return "Moderate confidence", "blue"
        elif score >= 0.3:
            return "Low confidence", "orange"
        else:
            return "Very low confidence - consider adjusting parameters", "red"


class CompletionAnalyzer:
    """Determine shell completion and minimal thickness, plus shell‑tracking diagnostics."""
    
    @staticmethod
    def compute_completion(manager, target_params: Dict, 
                          max_time_norm: Optional[float] = None,
                          tolerance: float = 0.3,
                          use_median: bool = True,
                          completeness_threshold: float = 0.95) -> Tuple[Optional[float], Optional[float], bool]:
        """Compute completion time, minimal thickness, and whether shell is complete.
        If max_time_norm is provided, only consider key frames up to that time.
        """
        key_times_norm = list(manager.key_frames.keys()) if hasattr(manager, 'key_frames') else []
        if not key_times_norm:
            return None, None, False
        
        # Filter times up to max_time_norm if provided
        if max_time_norm is not None:
            key_times_norm = [t for t in key_times_norm if t <= max_time_norm]
            if not key_times_norm:
                # No frames up to this time; return incomplete
                final_th = manager.get_thickness_at_time(max_time_norm) if hasattr(manager, 'get_thickness_at_time') else 0.0
                return None, final_th, False
        
        core_radius_nm = target_params.get('fc', 0.18) * target_params.get('L0_nm', 60.0) / 2
        t_complete = None
        dr_min = None
        
        sorted_times = sorted(key_times_norm)
        L0 = target_params.get('L0_nm', 60.0)
        
        for t_norm in sorted_times:
            fields = manager.key_frames.get(t_norm)
            if fields is None:
                continue
            
            proxy = DepositionPhysics.material_proxy(fields.get('phi', np.zeros((1,1))), 
                                                      fields.get('psi', np.zeros((1,1))))
            
            r, prof = DepositionPhysics.compute_radial_profile(proxy, L0, n_bins=100, use_median=use_median)
            
            core_idx = np.argmin(np.abs(r - core_radius_nm))
            if core_idx >= len(prof):
                continue
            
            profile_from_core = prof[core_idx:]
            if len(profile_from_core) == 0:
                continue
            
            # Softer check: require that at least completeness_threshold fraction of shell bins are above threshold
            shell_bins = profile_from_core[profile_from_core < 1.5]  # exclude core region
            if len(shell_bins) > 0:
                frac_above = np.mean(shell_bins >= 1.0 - tolerance)
                is_continuous = frac_above >= completeness_threshold
            else:
                is_continuous = False
            
            if is_continuous and t_complete is None:
                ag_region = profile_from_core < 1.5
                if np.any(ag_region):
                    first_cu_idx = np.argmax(ag_region)
                    dr_est = r[min(core_idx + first_cu_idx, len(r)-1)] - core_radius_nm
                else:
                    dr_est = r[-1] - core_radius_nm
                
                t_complete = manager.get_time_real(t_norm)
                dr_min = max(0.0, dr_est)
                break
        
        # Check final state (last key frame considered)
        if sorted_times:
            final_t = sorted_times[-1]
            final_fields = manager.key_frames.get(final_t)
            if final_fields is not None:
                final_proxy = DepositionPhysics.material_proxy(final_fields['phi'], final_fields['psi'])
                r_final, prof_final = DepositionPhysics.compute_radial_profile(final_proxy, L0, n_bins=100, use_median=use_median)
                core_idx_final = np.argmin(np.abs(r_final - core_radius_nm))
                shell_bins_final = prof_final[core_idx_final:][prof_final[core_idx_final:] < 1.5]
                if len(shell_bins_final) > 0:
                    frac_above_final = np.mean(shell_bins_final >= 1.0 - tolerance)
                    is_complete_at_end = frac_above_final >= completeness_threshold
                else:
                    is_complete_at_end = False
            else:
                is_complete_at_end = False
        else:
            is_complete_at_end = False
        
        if t_complete is None:
            final_thickness = manager.get_thickness_at_time(sorted_times[-1]) if hasattr(manager, 'get_thickness_at_time') and sorted_times else 0.0
            return None, final_thickness, False
        
        return t_complete, dr_min, is_complete_at_end

    @staticmethod
    def compute_shell_quality(phi: np.ndarray, psi: np.ndarray, core_radius_nm: float,
                              L0_nm: float) -> Dict[str, float]:
        """
        Compute shell‑tracking diagnostics:
          - Intrusion: mean phi inside core (psi > 0.5)
          - Coverage: fraction of core boundary pixels adjacent to Ag shell
          - Uniformity: standard deviation of radial thickness (proxy‑based)
        """
        # Intrusion
        intrusion = np.mean(phi[psi > 0.5]) if np.any(psi > 0.5) else 0.0
        
        # Coverage of core boundary
        core_mask = psi > 0.5
        if np.any(core_mask):
            # Erode core to get boundary
            struct = generate_binary_structure(2, 1)
            eroded = binary_erosion(core_mask, struct)
            boundary = core_mask & (~eroded)
            # Dilate boundary to find adjacent shell
            dilated_boundary = binary_dilation(boundary, struct)
            adjacent_shell = (DepositionPhysics.material_proxy(phi, psi) == 1.0) & dilated_boundary
            coverage = np.sum(adjacent_shell) / np.sum(boundary) if np.sum(boundary) > 0 else 0.0
            # FIX: cap coverage at 1.0 to avoid dilation artifacts
            coverage = min(1.0, coverage)
        else:
            coverage = 0.0
        
        # Uniformity: radial thickness variance
        proxy = DepositionPhysics.material_proxy(phi, psi)
        r_centers, profile = DepositionPhysics.compute_radial_profile(proxy, L0_nm, n_bins=100)
        # thickness at each radius: distance from core radius
        core_radius_idx = np.argmin(np.abs(r_centers - core_radius_nm))
        # shell region: proxy == 1.0 (Ag)
        shell_mask = proxy == 1.0
        if np.any(shell_mask):
            # radial distances of shell pixels
            H, W = shell_mask.shape
            x = np.linspace(0, L0_nm, W)
            y = np.linspace(0, L0_nm, H)
            X, Y = np.meshgrid(x, y, indexing='xy')
            R = np.sqrt((X - L0_nm/2)**2 + (Y - L0_nm/2)**2)
            thickness_vals = R[shell_mask] - core_radius_nm
            uniformity = np.std(thickness_vals) if len(thickness_vals) > 1 else 0.0
        else:
            uniformity = 0.0
        
        return {
            'intrusion': float(intrusion),
            'coverage': float(coverage),
            'uniformity': float(uniformity)
        }

    # ------------------------------------------------------------------
    # MODIFIED: Improved LLM completeness inference with robust JSON extraction,
    # example-driven prompt, simplified generation parameters, and strong fallback.
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json_robust(generated: str) -> Optional[Dict]:
        """
        Extract a JSON object from generated text, attempt repair of common errors.
        Returns parsed dict or None.
        """
        # Try to find a JSON-like block: first non-greedy capture of {...}
        json_pattern = r'\{.*?\}'
        match = re.search(json_pattern, generated, re.DOTALL | re.MULTILINE)
        if not match:
            return None
        json_str = match.group(0)
        
        # Repair: insert missing commas after true/false/null before a key
        # e.g., "true" "t_complete" -> "true", "t_complete"
        json_str = re.sub(r'(true|false|null)\s*(")', r'\1,\2', json_str)
        # Repair: remove trailing commas before } or ]
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def llm_infer_completeness(desc: str, tokenizer, model) -> Tuple[bool, Optional[float], Optional[float], str]:
        """Use LLM (GPT‑2 or Qwen) to infer completeness from a textual description of shell metrics.
        Improved with example-driven prompt, deterministic generation, and robust extraction."""
        backend = st.session_state.get('llm_backend_loaded', 'GPT-2 (default, fastest startup)')
        # Use temperature=0.0 and do_sample=False for deterministic output (better JSON)
        temperature = 0.0
        do_sample = False

        # Build prompt with a concrete example (like CODE 1)
        system = "You are a materials scientist. Output ONLY a JSON object."
        examples = """
Example: {"complete": true, "t_complete": 0.001, "dr_min": 5.2, "explanation": "High coverage, low intrusion, uniform thickness."}
"""
        user = f"""{examples}
Data: {desc}
JSON:"""

        if "Qwen" in backend:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system}\n\n{user}\n"

        try:
            inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Step 1: try robust JSON extraction
            result = CompletionAnalyzer._extract_json_robust(generated)
            if result is not None:
                # Ensure required keys exist (fallback to defaults if missing)
                complete = result.get('complete', False)
                t_complete = result.get('t_complete', None)
                dr_min = result.get('dr_min', None)
                explanation = result.get('explanation', 'No explanation provided.')
                return complete, t_complete, dr_min, explanation

            # Step 2: fallback – regex for "complete": true/false
            complete_match = re.search(r'"complete"\s*:\s*(true|false)', generated, re.IGNORECASE)
            if complete_match:
                complete = complete_match.group(1).lower() == 'true'
                # Try to extract t_complete and dr_min if present
                t_match = re.search(r'"t_complete"\s*:\s*(\d+\.?\d*(?:e[+-]?\d+)?|null)', generated, re.IGNORECASE)
                dr_match = re.search(r'"dr_min"\s*:\s*(\d+\.?\d*(?:e[+-]?\d+)?|null)', generated, re.IGNORECASE)
                exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', generated, re.IGNORECASE)
                t_complete = float(t_match.group(1)) if t_match and t_match.group(1) != 'null' else None
                dr_min = float(dr_match.group(1)) if dr_match and dr_match.group(1) != 'null' else None
                explanation = exp_match.group(1) if exp_match else "LLM: Inferred completeness (keyword fallback)."
                return complete, t_complete, dr_min, explanation

            # Step 3: final fallback – look for the word "complete" in context
            if 'complete' in generated.lower():
                if 'true' in generated.lower() or 'yes' in generated.lower():
                    return True, None, None, "LLM: Detected 'complete' with positive keyword."
                else:
                    return False, None, None, "LLM: Detected 'complete' but no positive indicator."
            return False, None, None, "LLM: Could not parse output."

        except Exception as e:
            return False, None, None, f"LLM inference failed: {e}"

    # MODIFIED: generate_recommendations now accepts llm_complete and adds a discrepancy note.
    @staticmethod
    def generate_recommendations(params: dict, relevance: float, 
                                  t_complete: Optional[float], 
                                  dr_min: Optional[float],
                                  is_complete: bool,
                                  shell_quality: Optional[Dict] = None,
                                  llm_explanation: Optional[str] = None,
                                  llm_complete: Optional[bool] = None) -> List[str]:
        suggestions = []
        if relevance < 0.5:
            suggestions.append(
                "⚠️ **Low relevance**: The requested parameters are far from available simulation data. "
                "Consider parameters closer to existing sources (L0: 40-80 nm, fc: 0.15-0.25, c_bulk: 0.2-0.6)."
            )
        if not is_complete:
            if t_complete is None:
                suggestions.append(
                    "❌ **Incomplete shell**: Current parameters may not yield a complete Ag shell. "
                    "Try: (1) Lower c_bulk (0.2-0.4) to promote complete coverage, "
                    "(2) Increase domain size L0, or (3) Longer deposition time."
                )
            else:
                suggestions.append(
                    f"⏱️ **Time insufficient**: Shell completes at {t_complete:.2e}s. "
                    f"Increase simulation time to at least this value."
                )
        if dr_min is not None:
            if dr_min > 15:
                suggestions.append(
                    f"📏 **Thick shell**: Minimal thickness is {dr_min:.1f} nm. "
                    "For thinner shells, consider: (1) Lower c_bulk, (2) Smaller rs target, "
                    "or (3) Shorter deposition time."
                )
            elif dr_min < 2:
                suggestions.append(
                    f"📏 **Thin shell**: Minimal thickness is only {dr_min:.1f} nm. "
                    "This may be suitable for ultra-thin applications but verify continuity."
                )
        if params.get('c_bulk', 0.5) > 0.7:
            suggestions.append(
                "🧪 **High concentration**: c_bulk > 0.7 may lead to irregular growth. "
                "Consider 0.3-0.6 for more uniform shells."
            )
        if params.get('fc', 0.18) > 0.35:
            suggestions.append(
                "⚠️ **Large core**: fc > 0.35 leaves limited space for shell. "
                "Verify rs target is achievable with available domain space."
            )
        # Add shell quality suggestions
        if shell_quality:
            if shell_quality['intrusion'] > 0.1:
                suggestions.append(
                    f"⚠️ **Shell intrusion**: Mean φ inside core = {shell_quality['intrusion']:.3f} > 0.1. "
                    "Indicates Ag forming inside core region – check simulation or reduce reaction rate."
                )
            if shell_quality['coverage'] < 0.95:
                suggestions.append(
                    f"❌ **Low coverage**: Only {shell_quality['coverage']*100:.1f}% of core boundary is covered. "
                    "Shell may have gaps – try increasing c_bulk or time."
                )
            if shell_quality['uniformity'] > 5.0:
                suggestions.append(
                    f"📉 **Non‑uniform shell**: Thickness std dev = {shell_quality['uniformity']:.2f} nm. "
                    "Consider adjusting parameters for more uniform growth."
                )
        # Add LLM explanation if available
        if llm_explanation:
            suggestions.append(f"🧠 **LLM insight**: {llm_explanation}")
        # If LLM completeness differs from rule-based, add a note
        if llm_complete is not None and llm_complete != is_complete:
            suggestions.append(
                f"⚠️ **LLM vs. rule‑based discrepancy**: LLM says the shell is {'complete' if llm_complete else 'incomplete'}, "
                f"while the analytical check says {'complete' if is_complete else 'incomplete'}. "
                f"This may indicate borderline behavior – review the radial profile manually."
            )
        if not suggestions:
            suggestions.append(
                "✅ **Design looks promising!** No major issues detected. "
                "Proceed with detailed simulation and validation."
            )
        return suggestions

# =============================================
# ENHANCED HEATMAP VISUALIZER
# =============================================
class HeatMapVisualizer:
    # Mapping from Matplotlib colormap names to Plotly-compatible names
    MATPLOTLIB_TO_PLOTLY = {
        'viridis': 'Viridis', 'plasma': 'Plasma', 'inferno': 'Inferno',
        'magma': 'Magma', 'cividis': 'Cividis', 'Greys': 'Greys',
        'Blues': 'Blues', 'Reds': 'Reds', 'Greens': 'Greens',
        'Oranges': 'Oranges', 'Purples': 'Purples', 'YlOrBr': 'YlOrBr',
        'YlOrRd': 'YlOrRd', 'OrRd': 'OrRd', 'PuRd': 'PuRd',
        'RdPu': 'RdPu', 'BuPu': 'BuPu', 'GnBu': 'GnBu',
        'PuBu': 'PuBu', 'YlGnBu': 'YlGnBu', 'PuBuGn': 'PuBuGn',
        'BuGn': 'BuGn', 'YlGn': 'YlGn', 'binary': 'Greys',  # approximate
        'gist_yarg': 'Greys_r', 'gist_gray': 'Gray', 'gray': 'Gray',
        'bone': 'bone', 'pink': 'pink', 'spring': 'spring',
        'summer': 'summer', 'autumn': 'autumn', 'winter': 'winter',
        'cool': 'cool', 'Wistia': 'Wistia', 'hot': 'hot',
        'afmhot': 'hot', 'gist_heat': 'hot', 'copper': 'copper',
        'PiYG': 'PiYG', 'PRGn': 'PRGn', 'BrBG': 'BrBG',
        'PuOr': 'PuOr', 'RdGy': 'RdGy', 'RdBu': 'RdBu',
        'RdYlBu': 'RdYlBu', 'RdYlGn': 'RdYlGn', 'Spectral': 'Spectral',
        'coolwarm': 'RdBu', 'bwr': 'RdBu', 'seismic': 'RdBu',
        'twilight': 'twilight', 'twilight_shifted': 'twilight',
        'hsv': 'HSV', 'jet': 'Jet', 'turbo': 'Turbo',
        'rainbow': 'Rainbow', 'gist_rainbow': 'Rainbow',
        'gist_ncar': 'nipy_spectral', 'nipy_spectral': 'nipy_spectral',
        'gist_stern': 'gnuplot', 'gnuplot': 'gnuplot', 'gnuplot2': 'gnuplot2',
        'CMRmap': 'CMRmap', 'cubehelix': 'Cubehelix', 'brg': 'brg',
        'gist_earth': 'Earth', 'terrain': 'Terrain', 'ocean': 'Ocean',
        'gist_water': 'Blues', 'flag': 'flag', 'prism': 'Prism',
        'tab10': 'Set1',  # approximate
        'tab20': 'Set3',  # approximate
        'tab20b': 'Set2', # approximate
        'tab20c': 'Set2', # approximate
        'Set1': 'Set1', 'Set2': 'Set2', 'Set3': 'Set3',
        'Pastel1': 'Pastel1', 'Pastel2': 'Pastel2',
        'Paired': 'Paired', 'Accent': 'Accent', 'Dark2': 'Dark2'
    }

    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
    
    def _get_extent(self, L0_nm):
        return [0, L0_nm, 0, L0_nm]
    
    def _is_material_proxy(self, field_data, colorbar_label, title):
        unique_vals = np.unique(field_data)
        # Robust check: if values are exactly 0, 1, 2 (allowing for floats)
        valid_material_values = {0.0, 1.0, 2.0}
        is_discrete = set(unique_vals).issubset(valid_material_values) and len(unique_vals) <= 3
        # Optional keyword check as a fallback, but data values are primary
        return is_discrete
    
    def create_field_heatmap(self, field_data, title, cmap_name='viridis',
                           L0_nm=20.0, figsize=(10,8), colorbar_label="",
                           vmin=None, vmax=None, target_params=None, time_real_s=None):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        extent = self._get_extent(L0_nm)
        
        is_material = self._is_material_proxy(field_data, colorbar_label, title)
        
        if is_material:
            im = ax.imshow(field_data, cmap=MATERIAL_COLORMAP_MATPLOTLIB,
                          norm=MATERIAL_BOUNDARY_NORM,
                          extent=extent, aspect='equal', origin='lower')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['Electrolyte', 'Ag', 'Cu'], fontsize=12)
            cbar.set_label('Material Phase', fontsize=14, fontweight='bold')
        else:
            im = ax.imshow(field_data, cmap=cmap_name, vmin=vmin, vmax=vmax,
                          extent=extent, aspect='equal', origin='lower')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if colorbar_label:
                cbar.set_label(colorbar_label, fontsize=14, fontweight='bold')
        
        ax.set_xlabel('X (nm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (nm)', fontsize=14, fontweight='bold')
        
        title_str = title
        if target_params:
            fc = target_params.get('fc', 0); rs = target_params.get('rs', 0)
            cb = target_params.get('c_bulk', 0)
            title_str += f"\nfc={fc:.3f}, rs={rs:.3f}, c_bulk={cb:.2f}, L0={L0_nm} nm"
        if time_real_s is not None:
            title_str += f"\nt = {time_real_s:.3e} s"
        
        ax.set_title(title_str, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def create_interactive_heatmap(self, field_data, title, cmap_name='viridis',
                                 L0_nm=20.0, width=800, height=700,
                                 target_params=None, time_real_s=None):
        ny, nx = field_data.shape
        x = np.linspace(0, L0_nm, nx)
        y = np.linspace(0, L0_nm, ny)
        
        is_material = self._is_material_proxy(field_data, "", title)
        
        if is_material:
            hover = [[f"X={x[j]:.2f} nm, Y={y[i]:.2f} nm<br>Phase={int(field_data[i,j])}"
                     for j in range(nx)] for i in range(ny)]
            fig = go.Figure(data=go.Heatmap(
                z=field_data, x=x, y=y, colorscale=MATERIAL_COLORSCALE_PLOTLY,
                hoverinfo='text', text=hover,
                colorbar=dict(
                    title=dict(text="Material Phase", font=dict(size=14)),
                    tickvals=[0, 1, 2],
                    ticktext=['Electrolyte', 'Ag', 'Cu']
                ),
                zmin=0, zmax=2
            ))
        else:
            hover = [[f"X={x[j]:.2f} nm, Y={y[i]:.2f} nm<br>Value={field_data[i,j]:.4f}"
                     for j in range(nx)] for i in range(ny)]
            # Map Matplotlib colormap name to a Plotly-compatible one
            plotly_cmap = self.MATPLOTLIB_TO_PLOTLY.get(cmap_name, 'Viridis')
            fig = go.Figure(data=go.Heatmap(
                z=field_data, x=x, y=y, colorscale=plotly_cmap,
                hoverinfo='text', text=hover,
                colorbar=dict(title=dict(text="Value", font=dict(size=14)))
            ))
        
        title_str = title
        if target_params:
            fc = target_params.get('fc', 0); rs = target_params.get('rs', 0)
            cb = target_params.get('c_bulk', 0)
            title_str += f"<br>fc={fc:.3f}, rs={rs:.3f}, c_bulk={cb:.2f}, L0={L0_nm} nm"
        if time_real_s is not None:
            title_str += f"<br>t = {time_real_s:.3e} s"
        
        fig.update_layout(
            title=dict(text=title_str, font=dict(size=20), x=0.5),
            xaxis=dict(title="X (nm)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y (nm)"),
            width=width, height=height
        )
        return fig
    
    def create_thickness_plot(self, thickness_time, source_curves=None, weights=None,
                           title="Shell Thickness Evolution", figsize=(10,6),
                           current_time_norm=None, current_time_real=None,
                           show_growth_rate=False):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        
        if 't_real_s' in thickness_time:
            t_plot = np.array(thickness_time['t_real_s'])
            ax.set_xlabel("Time (s)")
        else:
            t_plot = np.array(thickness_time['t_norm'])
            ax.set_xlabel("Normalized Time")
        
        th_nm = np.array(thickness_time['th_nm'])
        ax.plot(t_plot, th_nm, 'b-', linewidth=3, label='Interpolated')
        
        if show_growth_rate and len(t_plot) > 1:
            growth_rate = np.gradient(th_nm, t_plot)
            ax2 = ax.twinx()
            ax2.plot(t_plot, growth_rate, 'g--', linewidth=2, alpha=0.7, label='Growth rate')
            ax2.set_ylabel('Growth Rate (nm/s)', fontsize=12, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.grid(False)
        
        if source_curves is not None and weights is not None:
            for i, (src_t, src_th) in enumerate(source_curves):
                alpha = min(weights[i] * 5, 0.8)
                ax.plot(src_t, src_th, '--', linewidth=1, alpha=alpha,
                       label=f'Source {i+1} (w={weights[i]:.3f})')
        
        if current_time_norm is not None:
            if 't_real_s' in thickness_time:
                current_th = np.interp(current_time_norm,
                                      np.array(thickness_time['t_norm']), th_nm)
                current_t_plot = np.interp(current_time_norm,
                                          np.array(thickness_time['t_norm']), t_plot)
            else:
                current_t_plot = current_time_norm
                current_th = np.interp(current_time_norm,
                                      np.array(thickness_time['t_norm']), th_nm)
            
            ax.axvline(current_t_plot, color='r', linestyle='--',
                      linewidth=2, alpha=0.7)
            ax.plot(current_t_plot, current_th, 'ro', markersize=8,
                   label=f'Current: t={current_t_plot:.2e}, h={current_th:.2f} nm')
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        return fig
    
    def create_temporal_comparison_plot(self, fields_list, times_list, field_key='phi',
                                      cmap_name='viridis', L0_nm=20.0, n_cols=3):
        n_frames = len(fields_list)
        n_rows = (n_frames + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=200)
        
        if n_frames == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        extent = self._get_extent(L0_nm)
        is_material = "material" in field_key.lower()
        
        if is_material:
            cmap = MATERIAL_COLORMAP_MATPLOTLIB
            norm = MATERIAL_BOUNDARY_NORM
            vmin, vmax = 0, 2
        else:
            cmap = cmap_name
            norm = None
            all_values = [f[field_key] for f in fields_list]
            vmin = min(np.min(v) for v in all_values)
            vmax = max(np.max(v) for v in all_values)
        
        for i, (fields, t) in enumerate(zip(fields_list, times_list)):
            ax = axes[i]
            if norm is not None:
                im = ax.imshow(fields[field_key], cmap=cmap, norm=norm,
                              extent=extent, aspect='equal', origin='lower')
            else:
                im = ax.imshow(fields[field_key], cmap=cmap, vmin=vmin, vmax=vmax,
                              extent=extent, aspect='equal', origin='lower')
            ax.set_title(f't = {t:.3e} s', fontsize=12)
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        cbar = plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        if is_material:
            cbar.set_ticks([0, 1, 2])
            cbar.set_ticklabels(['Electrolyte', 'Ag', 'Cu'])
        else:
            cbar.set_label(field_key)
        
        plt.suptitle(f'Temporal Evolution: {field_key}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

# =============================================
# HYBRID WEIGHT VISUALIZER
# =============================================
class HybridWeightVisualizer:
    """Creates Sankey, chord, radar, and breakdown diagrams for weight analysis."""
    
    def __init__(self):
        self.color_scheme = {
            'L0': '#FF6B6B',
            'fc': '#4ECDC4',
            'rs': '#95E1D3',
            'c_bulk': '#FFD93D',
            'Attention': '#9D4EDD',
            'Spatial': '#36A2EB',
            'Combined': '#9966FF',
            'Query': '#FF6B6B'
        }
        
        self.font_config = {
            'family': 'Arial, sans-serif',
            'size_title': 24,
            'size_labels': 18,
            'size_ticks': 14,
            'color': '#2C3E50'
        }
    
    def get_colormap(self, cmap_name, n_colors=10):
        try:
            cmap = plt.get_cmap(cmap_name)
            return [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
                    for r, g, b, _ in [cmap(i/n_colors) for i in range(n_colors)]]
        except:
            cmap = plt.get_cmap('viridis')
            return [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
                    for r, g, b, _ in [cmap(i/n_colors) for i in range(n_colors)]]
    
    def create_enhanced_sankey_diagram(self, sources_data, target_params, param_sigmas):
        labels = ['Target']
        
        for source in sources_data:
            idx = source['source_index']
            l0 = source['L0_nm']
            fc = source['fc']
            labels.append(f"S{idx}\nL0={l0:.0f}nm\nfc={fc:.2f}")
        
        component_start = len(labels)
        labels.extend(['L0 Weight', 'fc Weight', 'rs Weight', 'c_bulk Weight', 
                      'Attention', 'Physics Refinement', 'Combined Weight'])
        
        source_indices = []
        target_indices = []
        values = []
        colors = []
        
        color_palette = self.get_colormap('viridis', len(sources_data) + 7)
        
        node_colors = ["#FF6B6B"]
        combined_weights = [s['combined_weight'] for s in sources_data]
        for w in combined_weights:
            node_colors.append(f'rgba(153,102,255,{w:.2f})')
        node_colors.extend(["#FF6B6B", "#4ECDC4", "#95E1D3", "#FFD93D", "#9D4EDD", "#36A2EB", "#9966FF"])
        
        for i, source in enumerate(sources_data):
            source_idx = i + 1
            source_color = color_palette[i % len(color_palette)]
            
            source_indices.append(source_idx)
            target_indices.append(component_start)
            values.append(source['l0_weight'] * 100)
            colors.append(f'rgba(255, 107, 107, 0.8)')
            
            source_indices.append(source_idx)
            target_indices.append(component_start + 1)
            values.append(source['fc_weight'] * 100)
            colors.append(f'rgba(78, 205, 196, 0.8)')
            
            source_indices.append(source_idx)
            target_indices.append(component_start + 2)
            values.append(source['rs_weight'] * 100)
            colors.append(f'rgba(149, 225, 211, 0.8)')
            
            source_indices.append(source_idx)
            target_indices.append(component_start + 3)
            values.append(source['c_bulk_weight'] * 100)
            colors.append(f'rgba(255, 217, 61, 0.8)')
            
            source_indices.append(source_idx)
            target_indices.append(component_start + 4)
            values.append(source['attention_weight'] * 100)
            colors.append(f'rgba(157, 78, 221, 0.8)')
            
            source_indices.append(source_idx)
            target_indices.append(component_start + 5)
            values.append(source['physics_refinement'] * 100)
            colors.append(f'rgba(54, 162, 235, 0.8)')
            
            source_indices.append(source_idx)
            target_indices.append(component_start + 6)
            values.append(source['combined_weight'] * 100)
            colors.append(f'rgba(153, 102, 255, 0.8)')
        
        for comp_idx in range(7):
            source_indices.append(component_start + comp_idx)
            target_indices.append(0)
            comp_value = sum(v for s, t, v in zip(source_indices, target_indices, values)
                           if t == component_start + comp_idx)
            values.append(comp_value * 0.5)
            colors.append(f'rgba(153, 102, 255, 0.6)')
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=25,
                thickness=30,
                line=dict(color="black", width=2),
                label=labels,
                color=node_colors,
                hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<extra></extra>'
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value:.2f}<extra></extra>',
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            hoverinfo='all'
        )])
        
        fig.update_layout(
            title=dict(
                text=f'<b>SANKEY DIAGRAM: HYBRID WEIGHT COMPONENT FLOW</b><br>' + 
                     f'Target: L0={target_params.get("L0_nm", 20):.0f}nm, fc={target_params.get("fc", 0.18):.2f}',
                font=dict(family=self.font_config['family'], size=self.font_config['size_title'],
                         color=self.font_config['color']),
                x=0.5, y=0.95, xanchor='center', yanchor='top',
                pad=dict(b=20)
            ),
            font=dict(family=self.font_config['family'], size=self.font_config['size_labels'],
                     color=self.font_config['color']),
            width=1400,
            height=900,
            plot_bgcolor='rgba(240, 240, 245, 0.9)',
            paper_bgcolor='white',
            margin=dict(t=100, l=50, r=50, b=50),
            hoverlabel=dict(
                font=dict(family=self.font_config['family'], size=self.font_config['size_labels'],
                         color='white'),
                bgcolor='rgba(44, 62, 80, 0.9)',
                bordercolor='white'
            )
        )
        
        return fig
    
    def create_enhanced_chord_diagram(self, sources_data, target_params):
        n_sources = len(sources_data)
        center_x, center_y = 0, 0
        radius = 1.5
        
        source_positions = []
        for i in range(n_sources):
            angle = 2 * pi * i / n_sources
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            source_positions.append((x, y))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[center_x], y=[center_y],
            mode='markers+text',
            name='Target',
            marker=dict(size=50, color=self.color_scheme['Query'],
                       symbol='star', line=dict(width=3, color='white')),
            text=[f'Target\nL0={target_params.get("L0_nm", 20):.0f}nm\nfc={target_params.get("fc", 0.18):.2f}'],
            textposition="middle center",
            textfont=dict(size=16, color='white', family=self.font_config['family']),
            hoverinfo='text',
            hovertemplate='<b>Target</b><br>L0=%{text}<extra></extra>'
        ))
        
        source_x = []
        source_y = []
        source_colors = []
        source_sizes = []
        combined_weights = [s['combined_weight'] for s in sources_data]
        
        for i, source in enumerate(sources_data):
            x, y = source_positions[i]
            source_x.append(x)
            source_y.append(y)
            node_size = 20 + source['combined_weight'] * 60
            source_sizes.append(node_size)
            source_colors.append(self.color_scheme['Combined'])
        
        fig.add_trace(go.Scatter(
            x=source_x, y=source_y,
            mode='markers+text',
            name='Sources',
            marker=dict(size=source_sizes, color=source_colors,
                       line=dict(width=2, color='white')),
            text=[f"S{i}" for i in range(n_sources)],
            textposition="top center",
            textfont=dict(size=12, color='white', family=self.font_config['family']),
            hoverinfo='text',
            hovertemplate='<b>Source %{text}</b><br>Combined weight: %{marker.size:.1f}<extra></extra>'
        ))
        
        for i, source in enumerate(sources_data):
            sx, sy = source_positions[i]
            cx = (sx + center_x) / 2
            cy = (sy + center_y) / 2
            dx = center_x - sx
            dy = center_y - sy
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                nx = -dy / length * 0.3
                ny = dx / length * 0.3
            else:
                nx, ny = 0, 0
            
            t = np.linspace(0, 1, 50)
            
            combined_curve_x = (1-t)**2 * sx + 2*(1-t)*t * (cx - nx*1.0) + t**2 * center_x
            combined_curve_y = (1-t)**2 * sy + 2*(1-t)*t * (cy - ny*1.0) + t**2 * center_y
            combined_width = max(2, source['combined_weight'] * 20)
            weight_norm = source['combined_weight'] / max(combined_weights) if max(combined_weights) > 0 else 0.5
            line_color = px.colors.sample_colorscale('viridis', weight_norm)[0]
            
            fig.add_trace(go.Scatter(
                x=combined_curve_x, y=combined_curve_y,
                mode='lines',
                name=f'Source {i} - Combined',
                line=dict(width=combined_width, color=line_color, dash='solid'),
                hoverinfo='text',
                hovertext=f"Combined Weight: {source['combined_weight']:.3f}",
                showlegend=False
            ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>ENHANCED CHORD DIAGRAM: HYBRID WEIGHT VISUALIZATION</b><br>' + 
                     f'Target: L0={target_params.get("L0_nm", 20):.0f}nm, fc={target_params.get("fc", 0.18):.2f}',
                font=dict(family=self.font_config['family'], size=self.font_config['size_title'],
                         color=self.font_config['color']),
                x=0.5, y=0.95,
                pad=dict(b=20)
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                       font=dict(family=self.font_config['family'], size=self.font_config['size_labels'],
                                color=self.font_config['color']),
                       bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black', borderwidth=1,
                       itemwidth=50, tracegroupgap=10),
            width=1200,
            height=1000,
            plot_bgcolor='rgba(240, 240, 245, 0.9)',
            paper_bgcolor='white',
            hovermode='closest',
            margin=dict(l=100, r=100, t=150, b=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
            hoverlabel=dict(
                font=dict(family=self.font_config['family'], size=self.font_config['size_labels'],
                         color='white'),
                bgcolor='rgba(44, 62, 80, 0.9)',
                bordercolor='white'
            )
        )
        
        return fig
    
    def create_parameter_radar_charts(self, sources_data, target_params, param_sigmas):
        param_keys = ['L0_nm', 'fc', 'rs', 'c_bulk']
        param_names = ['L0 (nm)', 'fc', 'rs', 'c_bulk']
        figs = []

        for pkey, pname in zip(param_keys, param_names):
            sorted_idx = np.argsort([s[pkey] for s in sources_data])
            sorted_sources = [sources_data[i] for i in sorted_idx]

            l0_weights = [s['l0_weight'] for s in sorted_sources]
            fc_weights = [s['fc_weight'] for s in sorted_sources]
            rs_weights = [s['rs_weight'] for s in sorted_sources]
            c_weights = [s['c_bulk_weight'] for s in sorted_sources]
            attn_weights = [s['attention_weight'] for s in sorted_sources]
            combined_weights = [s['combined_weight'] for s in sorted_sources]

            max_w = max(combined_weights) if combined_weights else 1
            l0_norm = [w / max_w for w in l0_weights]
            fc_norm = [w / max_w for w in fc_weights]
            rs_norm = [w / max_w for w in rs_weights]
            c_norm = [w / max_w for w in c_weights]
            attn_norm = [w / max_w for w in attn_weights]
            combined_norm = [w / max_w for w in combined_weights]

            p_vals = np.array([s[pkey] for s in sorted_sources])
            min_p, max_p = p_vals.min(), p_vals.max()
            if max_p == min_p:
                angles = np.linspace(0, 360, len(p_vals), endpoint=False)
            else:
                angles = 360 * (p_vals - min_p) / (max_p - min_p)

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=l0_norm, theta=angles, mode='lines+markers',
                name='α (L0)', line=dict(color=self.color_scheme['L0'], width=3),
                marker=dict(size=8, color=self.color_scheme['L0']),
                hovertemplate='<b>L0 Weight</b>: %{r:.3f}<br>Parameter: %{theta:.1f}°<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=fc_norm, theta=angles, mode='lines+markers',
                name='β_fc', line=dict(color=self.color_scheme['fc'], width=3, dash='dot'),
                marker=dict(size=8, color=self.color_scheme['fc']),
                hovertemplate='<b>fc Weight</b>: %{r:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=rs_norm, theta=angles, mode='lines+markers',
                name='β_rs', line=dict(color=self.color_scheme['rs'], width=3, dash='dash'),
                marker=dict(size=8, color=self.color_scheme['rs']),
                hovertemplate='<b>rs Weight</b>: %{r:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=c_norm, theta=angles, mode='lines+markers',
                name='β_c', line=dict(color=self.color_scheme['c_bulk'], width=3, dash='longdash'),
                marker=dict(size=8, color=self.color_scheme['c_bulk']),
                hovertemplate='<b>c_bulk Weight</b>: %{r:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=attn_norm, theta=angles, mode='lines+markers',
                name='Attention', line=dict(color=self.color_scheme['Attention'], width=3, dash='dot'),
                marker=dict(size=8, color=self.color_scheme['Attention']),
                hovertemplate='<b>Attention</b>: %{r:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=combined_norm, theta=angles, mode='lines+markers',
                name='Hybrid Weight', line=dict(color=self.color_scheme['Combined'], width=4),
                marker=dict(size=10, color=self.color_scheme['Combined']),
                hovertemplate='<b>Hybrid Weight</b>: %{r:.3f}<extra></extra>'
            ))

            n_ticks = min(8, len(angles))
            tick_indices = np.linspace(0, len(angles)-1, n_ticks, dtype=int)
            tick_vals = angles[tick_indices]
            tick_text = [f'{p_vals[i]:.2f}' for i in tick_indices]

            fig.update_layout(
                title=dict(
                    text=f'Radar by {pname}',
                    x=0.5,
                    font=dict(family=self.font_config['family'], size=18, color=self.font_config['color']),
                    pad=dict(t=20, b=20)
                ),
                polar=dict(
                    radialaxis=dict(range=[0, 1.05], tickfont=dict(size=12, family=self.font_config['family']),
                                   gridwidth=2, linecolor='gray', linewidth=1),
                    angularaxis=dict(
                        tickmode='array',
                        tickvals=tick_vals,
                        ticktext=tick_text,
                        tickfont=dict(size=10, family=self.font_config['family'], color=self.font_config['color']),
                        tickangle=45,
                        gridcolor='lightgray',
                        linecolor='gray',
                        gridwidth=2,
                        direction='clockwise',
                        rotation=90
                    ),
                    bgcolor='rgba(240, 240, 245, 0.5)'
                ),
                legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5,
                           font=dict(family=self.font_config['family'], size=12, color=self.font_config['color']),
                           itemwidth=60, tracegroupgap=15),
                width=600,
                height=500,
                margin=dict(l=60, r=60, t=80, b=60)
            )
            figs.append(fig)

        return figs

    def create_weight_formula_breakdown(self, sources_data, target_params, param_sigmas):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Weight Component Breakdown</b>',
                '<b>Cumulative Weight Distribution</b>',
                '<b>Parameter Weight Analysis</b>',
                '<b>Hybrid Weight Distribution</b>'
            ),
            vertical_spacing=0.2,
            horizontal_spacing=0.15
        )
        
        sources_data = sorted(sources_data, key=lambda x: x['source_index'])
        source_indices = [s['source_index'] for s in sources_data]
        
        l0_values = [s['l0_weight'] for s in sources_data]
        fc_values = [s['fc_weight'] for s in sources_data]
        rs_values = [s['rs_weight'] for s in sources_data]
        c_bulk_values = [s['c_bulk_weight'] for s in sources_data]
        attention_values = [s['attention_weight'] for s in sources_data]
        combined_values = [s['combined_weight'] for s in sources_data]
        
        fig.add_trace(go.Bar(
            x=source_indices, y=l0_values, name='L0 Weight',
            marker_color=self.color_scheme['L0'],
            text=[f'{v:.3f}' for v in l0_values], textposition='outside', textfont=dict(size=10, family=self.font_config['family']),
            hovertemplate='<b>Source %{x}</b><br>L0 Weight: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=source_indices, y=fc_values, name='fc Weight',
            marker_color=self.color_scheme['fc'],
            text=[f'{v:.3f}' for v in fc_values], textposition='outside', textfont=dict(size=10, family=self.font_config['family'])
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=source_indices, y=attention_values, name='Attention',
            marker_color=self.color_scheme['Attention'],
            text=[f'{v:.3f}' for v in attention_values], textposition='outside', textfont=dict(size=10, family=self.font_config['family'])
        ), row=1, col=1)
        
        sorted_weights = np.sort(combined_values)[::-1]
        cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights) if np.sum(sorted_weights) > 0 else np.zeros_like(sorted_weights)
        x_vals = np.arange(1, len(cumulative) + 1)
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=cumulative, mode='lines+markers', name='Cumulative Weight',
            line=dict(color=self.color_scheme['Combined'], width=4), marker=dict(size=8),
            fill='tozeroy', fillcolor='rgba(153, 102, 255, 0.2)',
            hovertemplate='Top %{x} sources: %{y:.1%}<extra></extra>'
        ), row=1, col=2)
        
        if len(cumulative) > 0 and np.sum(sorted_weights) > 0:
            threshold_idx = np.where(cumulative >= 0.9)[0]
            if len(threshold_idx) > 0:
                fig.add_hline(y=0.9, line_dash="dash", line_color="red",
                             annotation_text="90% threshold", row=1, col=2)
                fig.add_vline(x=threshold_idx[0]+1, line_dash="dash", line_color="red", row=1, col=2)
        
        param_means = [np.mean(l0_values), np.mean(fc_values), np.mean(rs_values), np.mean(c_bulk_values)]
        param_names = ['L0', 'fc', 'rs', 'c_bulk']
        param_colors = [self.color_scheme['L0'], self.color_scheme['fc'], 
                       self.color_scheme['rs'], self.color_scheme['c_bulk']]
        
        fig.add_trace(go.Bar(
            x=param_names, y=param_means, name='Mean Weight by Parameter',
            marker_color=param_colors,
            text=[f'{v:.3f}' for v in param_means], textposition='auto',
            textfont=dict(family=self.font_config['family']),
            hovertemplate='<b>%{x}</b><br>Mean weight: %{y:.3f}<extra></extra>'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=source_indices, y=combined_values, mode='markers+lines',
            name='Combined Weight',
            line=dict(color=self.color_scheme['Combined'], width=3),
            marker=dict(size=15, color=self.color_scheme['Combined'],
                       symbol='circle', line=dict(width=2, color='white')),
            text=[f"S{i}: Combined={w:.3f}" for i, w in zip(source_indices, combined_values)],
            textfont=dict(family=self.font_config['family']),
            hovertemplate='<b>Source %{x}</b><br>Combined Weight: %{y:.3f}<extra></extra>'
        ), row=2, col=2)
        
        fig.update_layout(
            barmode='stack',
            title=dict(
                text=f'<b>HYBRID WEIGHT FORMULA ANALYSIS</b><br>' + 
                     f'wᵢ = α(L0) × β(params) × γ(shape) × Attention<br>' + 
                     f'Target: L0={target_params.get("L0_nm", 20):.0f}nm, fc={target_params.get("fc", 0.18):.2f}',
                font=dict(family=self.font_config['family'], size=20,
                         color=self.font_config['color']),
                x=0.5, y=0.98,
                pad=dict(t=20, b=20)
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                       font=dict(family=self.font_config['family'], size=12, color=self.font_config['color']),
                       itemwidth=50, tracegroupgap=10),
            width=1400,
            height=1000,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest',
            margin=dict(l=100, r=100, t=150, b=100)
        )
        
        fig.update_xaxes(title_text="Source Index", row=1, col=1, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_yaxes(title_text="Weight Component Value", row=1, col=1, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_xaxes(title_text="Number of Top Sources", row=1, col=2, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_yaxes(title_text="Cumulative Weight", row=1, col=2, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_xaxes(title_text="Parameter", row=2, col=1, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_yaxes(title_text="Mean Weight", row=2, col=1, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_xaxes(title_text="Source Index", row=2, col=2, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_yaxes(title_text="Weight Value", row=2, col=2, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        
        return fig

# =============================================
# ENHANCED MULTI-PREDICTION COMPARISON VISUALIZER
# =============================================
class MultiPredictionVisualizer:
    """Generates comparison plots for multiple saved predictions."""
    
    @staticmethod
    def thickness_evolution_plot(predictions, labels):
        fig = go.Figure()
        for pred, label in zip(predictions, labels):
            thick_time = pred['derived']['thickness_time']
            t = thick_time['t_real_s'] if 't_real_s' in thick_time else thick_time['t_norm']
            th = thick_time['th_nm']
            params = pred['target_params']
            legend = f"{label}: L0={params.get('L0_nm',20):.0f} fc={params.get('fc',0.18):.2f} rs={params.get('rs',0.2):.2f} c={params.get('c_bulk',0.5):.2f}"
            fig.add_trace(go.Scatter(x=t, y=th, mode='lines', name=legend))
        fig.update_layout(title='Thickness Evolution Comparison',
                         xaxis_title='Time (s)' if 't_real_s' in thick_time else 'Normalized Time',
                         yaxis_title='Thickness (nm)',
                         hovermode='x unified')
        return fig

    @staticmethod
    def thickness_evolution_matplotlib(predictions, labels):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        for pred, label in zip(predictions, labels):
            thick_time = pred['derived']['thickness_time']
            t = thick_time['t_real_s'] if 't_real_s' in thick_time else thick_time['t_norm']
            th = thick_time['th_nm']
            params = pred['target_params']
            legend = f"{label}: L0={params.get('L0_nm',20):.0f} fc={params.get('fc',0.18):.2f}"
            ax.plot(t, th, linewidth=2, label=legend)
        ax.set_xlabel('Time (s)' if 't_real_s' in thick_time else 'Normalized Time', fontsize=14)
        ax.set_ylabel('Thickness (nm)', fontsize=14)
        ax.set_title('Thickness Evolution Comparison', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def thickness_evolution_3d(predictions, labels, param_y='fc', param_color='growth_rate', cmap='turbo'):
        fig = go.Figure()
        sample = predictions[0]['derived']['thickness_time']
        if 't_real_s' in sample:
            xlabel = 'Time (s)'
        else:
            xlabel = 'Normalized Time'

        for pred, label in zip(predictions, labels):
            thick_time = pred['derived']['thickness_time']
            t = thick_time['t_real_s'] if 't_real_s' in thick_time else thick_time['t_norm']
            th = thick_time['th_nm']
            y_val = pred['target_params'].get(param_y, 0)
            if param_color == 'entropy':
                color_val = pred['weights'].get('entropy', 0)
            else:
                color_val = pred['derived'].get(param_color, 0)
            fig.add_trace(go.Scatter3d(
                x=t,
                y=[y_val]*len(t),
                z=th,
                mode='lines',
                line=dict(width=4, color=color_val, colorscale=cmap, colorbar=dict(title=param_color)),
                name=f"{param_y}={y_val:.2f}",
                hovertext=f"{label}<br>{param_color}={color_val:.3f}",
                hoverinfo='text'
            ))
        fig.update_layout(
            title=f'3D Thickness Evolution (Y: {param_y}, Color: {param_color})',
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=param_y.capitalize(),
                zaxis_title='Thickness (nm)'
            ),
            width=900,
            height=700
        )
        return fig
    
    @staticmethod
    def parameter_map(predictions, param_x, param_y, metric='thickness_nm', plot_type='scatter', cmap='viridis'):
        df = pd.DataFrame([
            {
                param_x: p['target_params'].get(param_x, np.nan),
                param_y: p['target_params'].get(param_y, np.nan),
                'metric': p['derived'].get(metric, p['weights'].get(metric, 0)),
                'label': f"{p['target_params'].get('L0_nm',20):.0f}_{p['target_params'].get('fc',0.18):.2f}"
            }
            for p in predictions
        ]).dropna()
        if len(df) < 3:
            return None
        if plot_type == 'contour' and len(df) > 5:
            fig = px.density_contour(df, x=param_x, y=param_y, z='metric',
                                     color_continuous_scale=cmap,
                                     title=f'{metric} Density Contour: {param_x} vs {param_y}')
            fig.update_traces(contours_coloring="fill", contours_showlabels=True)
        else:
            fig = px.scatter(df, x=param_x, y=param_y, size='metric', color='metric',
                            color_continuous_scale=cmap,
                            hover_data=['label'],
                            title=f'{metric} Map: {param_x} vs {param_y}')
        return fig
    
    @staticmethod
    def radar_comparison(predictions, labels):
        metrics = ['thickness_nm', 'growth_rate', 'entropy']
        fig = go.Figure()
        for pred, label in zip(predictions, labels):
            values = []
            for m in metrics:
                if m == 'entropy':
                    val = pred['weights'].get('entropy', 0)      # FIXED: safe get
                else:
                    val = pred['derived'].get(m, 0)
                values.append(val)
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=label
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title='Multi-Prediction Metrics Comparison'
        )
        return fig
    
    @staticmethod
    def weight_sunburst(predictions, labels, cmap='viridis'):
        ids = ['all']
        labels_all = ['All Predictions']
        parents = ['']
        values = [0]
        colors_all = ['lightgrey']

        for i, (pred, label) in enumerate(zip(predictions, labels)):
            pred_id = f"pred_{i}"
            ids.append(pred_id)
            labels_all.append(label[:20])
            parents.append('all')
            values.append(1)
            colors_all.append('lightblue')

            weights = pred['weights']
            comp_names = ['alpha', 'beta', 'gamma', 'attention']
            comp_labels = ['α (L0)', 'β (params)', 'γ (shape)', 'Attention']
            for comp, comp_label in zip(comp_names, comp_labels):
                comp_id = f"{pred_id}_{comp}"
                ids.append(comp_id)
                labels_all.append(comp_label)
                parents.append(pred_id)
                comp_vals = weights.get(comp, [0])
                if isinstance(comp_vals, list) and len(comp_vals) > 0:
                    val = np.mean(comp_vals)
                else:
                    val = 0
                values.append(val)
                norm_val = val / (max(values[-1], 1e-6))
                cmap_obj = plt.get_cmap(cmap)
                rgba = cmap_obj(norm_val)
                color_str = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})'
                colors_all.append(color_str)

        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels_all,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(colors=colors_all)
        ))
        fig.update_layout(title='Weight Component Sunburst (colored by weight magnitude)')
        return fig

    @staticmethod
    def parameter_scatter_3d(predictions, labels, x_param, y_param, z_param,
                             color_metric='thickness_nm', size_metric='growth_rate', cmap='viridis'):
        df = pd.DataFrame([
            {
                'label': lbl,
                x_param: p['target_params'].get(x_param, 0),
                y_param: p['target_params'].get(y_param, 0),
                z_param: p['target_params'].get(z_param, 0),
                'color': p['derived'].get(color_metric, p['weights'].get(color_metric, 0)),
                'size': p['derived'].get(size_metric, p['weights'].get(size_metric, 1))
            }
            for p, lbl in zip(predictions, labels)
        ]).dropna()
        if len(df) < 2:
            return None
        size_min, size_max = df['size'].min(), df['size'].max()
        if size_max > size_min:
            df['size_scaled'] = 10 + 40 * (df['size'] - size_min) / (size_max - size_min)
        else:
            df['size_scaled'] = 25

        fig = go.Figure()
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[row[x_param]], y=[row[y_param]], z=[row[z_param]],
                mode='markers',
                marker=dict(
                    size=row['size_scaled'],
                    color=row['color'],
                    colorscale=cmap,
                    colorbar=dict(title=color_metric),
                    showscale=True
                ),
                name=row['label'],
                text=f"{row['label']}<br>{x_param}={row[x_param]:.2f}<br>{y_param}={row[y_param]:.2f}<br>{z_param}={row[z_param]:.2f}<br>{color_metric}={row['color']:.3f}<br>{size_metric}={row['size']:.3f}",
                hoverinfo='text'
            ))
        fig.update_layout(
            title=f'3D Parameter Scatter (color={color_metric}, size={size_metric})',
            scene=dict(
                xaxis_title=x_param,
                yaxis_title=y_param,
                zaxis_title=z_param
            ),
            width=900, height=700,
            showlegend=False
        )
        return fig

    @staticmethod
    def parallel_coordinates(predictions, labels, dimensions):
        df = pd.DataFrame()
        for dim in dimensions:
            df[dim['label']] = dim['values']
        df['pred_id'] = pd.factorize(labels)[0]
        df['prediction'] = labels

        fig = px.parallel_coordinates(
            df,
            color='pred_id',
            dimensions=[d['label'] for d in dimensions],
            color_continuous_scale=px.colors.qualitative.Plotly,
            labels={'pred_id': 'Prediction ID'},
            title='Parallel Coordinates: Parameters and Metrics'
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_layout(width=1000, height=600)
        return fig

    @staticmethod
    def pairplot(predictions, labels, columns):
        data = []
        for p, lbl in zip(predictions, labels):
            row = {'label': lbl}
            for col in columns:
                if col in p['target_params']:
                    row[col] = p['target_params'][col]
                elif col in p['derived']:
                    row[col] = p['derived'][col]
                elif col in p['weights']:
                    if col == 'entropy':
                        row[col] = p.get('weights', {}).get('entropy', 0)   # FIXED: safe get
                    else:
                        row[col] = np.mean(p.get('weights', {}).get(col, [0]))
                else:
                    row[col] = 0
            data.append(row)
        df = pd.DataFrame(data)
        fig = px.scatter_matrix(
            df, dimensions=columns, color='label',
            title='Pair Plot of Parameters and Metrics',
            labels={c: c for c in columns}
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=True, showlowerhalf=True)
        fig.update_layout(width=1200, height=1200)
        return fig

# =============================================
# RESULTS MANAGER
# =============================================
class ResultsManager:
    def __init__(self):
        pass
    
    def prepare_export_data(self, interpolation_result, visualization_params):
        res = interpolation_result.copy()
        export = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'interpolation_method': 'core_shell_hybrid_weight_transformer',
                'visualization_params': visualization_params
            },
            'result': {
                'target_params': res['target_params'],
                'shape': res['shape'],
                'num_sources': res['num_sources'],
                'weights': res['weights'],
                'sources_data': res.get('sources_data', []),
                'time_norm': res.get('time_norm', 1.0),
                'time_real_s': res.get('time_real_s', 0.0),
                'growth_rate': res['derived'].get('growth_rate', 0.0)
            }
        }
        
        for fname, arr in res['fields'].items():
            export['result'][f'{fname}_data'] = arr.tolist()
        
        for dname, val in res['derived'].items():
            if isinstance(val, np.ndarray):
                export['result'][f'{dname}_data'] = val.tolist()
            elif isinstance(val, dict) and 'th_nm' in val:
                export['result'][dname] = val
            else:
                export['result'][dname] = val
        
        return export
    
    def export_to_json(self, export_data, filename=None):
        if filename is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            p = export_data['result']['target_params']
            fc = p.get('fc', 0); rs = p.get('rs', 0)
            cb = p.get('c_bulk', 0)
            t = export_data['result'].get('time_real_s', 0)
            filename = f"temporal_interp_fc{fc:.3f}_rs{rs:.3f}_c{cb:.2f}_t{t:.3e}s_{ts}.json"
        
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
    def export_to_csv(self, interpolation_result, filename=None):
        if filename is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            p = interpolation_result['target_params']
            fc = p.get('fc', 0); rs = p.get('rs', 0)
            cb = p.get('c_bulk', 0)
            t = interpolation_result.get('time_real_s', 0)
            filename = f"fields_fc{fc:.3f}_rs{rs:.3f}_c{cb:.2f}_t{t:.3e}s_{ts}.csv"
        
        shape = interpolation_result['shape']
        L0 = interpolation_result['target_params'].get('L0_nm', 20.0)
        x = np.linspace(0, L0, shape[1]); y = np.linspace(0, L0, shape[0])
        X, Y = np.meshgrid(x, y)
        
        data = {'x_nm': X.flatten(), 'y_nm': Y.flatten(),
               'time_norm': interpolation_result.get('time_norm', 0),
               'time_real_s': interpolation_result.get('time_real_s', 0)}
        
        for fname, arr in interpolation_result['fields'].items():
            data[fname] = arr.flatten()
        
        for dname, val in interpolation_result['derived'].items():
            if isinstance(val, np.ndarray):
                data[dname] = val.flatten()
        
        df = pd.DataFrame(data)
        csv_str = df.to_csv(index=False)
        return csv_str, filename
    
    def _json_serializer(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, datetime): return obj.isoformat()
        elif isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
        else: return str(obj)

# =============================================
# ERROR COMPUTATION WITH PHYSICAL COORDINATE ALIGNMENT
# =============================================
def create_common_physical_grid(L0_list, target_resolution_nm=0.2):
    L_ref = np.ceil(max(L0_list) / 10) * 10
    n_pixels = int(np.ceil(L_ref / target_resolution_nm))
    n_pixels = max(n_pixels, 256)
    x_ref = np.linspace(0, L_ref, n_pixels)
    y_ref = np.linspace(0, L_ref, n_pixels)
    return L_ref, x_ref, y_ref, (n_pixels, n_pixels)

def resample_to_physical_grid(field, L0_original, x_ref, y_ref, method='linear'):
    H, W = field.shape
    x_orig = np.linspace(0, L0_original, W)
    y_orig = np.linspace(0, L0_original, H)
    interpolator = RegularGridInterpolator(
        (y_orig, x_orig), field,
        method=method, bounds_error=False, fill_value=0.0
    )
    X_ref, Y_ref = np.meshgrid(x_ref, y_ref, indexing='xy')
    points = np.stack([Y_ref.ravel(), X_ref.ravel()], axis=1)
    field_resampled = interpolator(points).reshape(Y_ref.shape)
    return field_resampled

def compare_fields_physical(gt_field, gt_L0, interp_field, interp_L0,
                           target_resolution_nm=0.2, compare_region='overlap'):
    L_ref, x_ref, y_ref, shape_ref = create_common_physical_grid(
        [gt_L0, interp_L0], target_resolution_nm
    )
    gt_resampled = resample_to_physical_grid(gt_field, gt_L0, x_ref, y_ref)
    interp_resampled = resample_to_physical_grid(interp_field, interp_L0, x_ref, y_ref)
    
    if compare_region == 'overlap':
        gt_mask = np.zeros(shape_ref, dtype=bool)
        interp_mask = np.zeros(shape_ref, dtype=bool)
        gt_H, gt_W = gt_field.shape
        interp_H, interp_W = interp_field.shape
        gt_x_max_idx = int(np.round(gt_L0 / target_resolution_nm))
        gt_y_max_idx = int(np.round(gt_L0 / target_resolution_nm))
        interp_x_max_idx = int(np.round(interp_L0 / target_resolution_nm))
        interp_y_max_idx = int(np.round(interp_L0 / target_resolution_nm))
        gt_mask[:gt_y_max_idx, :gt_x_max_idx] = True
        interp_mask[:interp_y_max_idx, :interp_x_max_idx] = True
        valid_mask = gt_mask & interp_mask
        if np.sum(valid_mask) < 100:
            valid_mask = np.ones_like(valid_mask)
    else:
        valid_mask = np.ones(shape_ref, dtype=bool)
    
    gt_valid = gt_resampled[valid_mask]
    interp_valid = interp_resampled[valid_mask]
    
    mse = np.mean((gt_valid - interp_valid) ** 2)
    mae = np.mean(np.abs(gt_valid - interp_valid))
    max_err = np.max(np.abs(gt_valid - interp_valid))
    
    if np.sum(valid_mask) > 1000:
        y_idx, x_idx = np.where(valid_mask)
        y_min, y_max = y_idx.min(), y_idx.max()
        x_min, x_max = x_idx.min(), x_idx.max()
        ssim_val = ssim(
            gt_resampled[y_min:y_max, x_min:x_max],
            interp_resampled[y_min:y_max, x_min:x_max],
            data_range=max(gt_resampled.max() - gt_resampled.min(), 1e-6)
        )
    else:
        ssim_val = np.nan
    
    return {
        'gt_aligned': gt_resampled,
        'interp_aligned': interp_resampled,
        'valid_mask': valid_mask,
        'L_ref': L_ref,
        'shape_ref': shape_ref,
        'metrics': {
            'MSE': mse,
            'MAE': mae,
            'Max Error': max_err,
            'SSIM': ssim_val,
            'valid_pixels': int(np.sum(valid_mask))
        }
    }

def compute_errors(gt_field, interp_field):
    flat_gt = gt_field.flatten()
    flat_interp = interp_field.flatten()
    mse = mean_squared_error(flat_gt, flat_interp)
    mae = mean_absolute_error(flat_gt, flat_interp)
    max_err = np.max(np.abs(gt_field - interp_field))
    data_range = max(gt_field.max() - gt_field.min(),
                    interp_field.max() - interp_field.min(), 1e-6)
    if data_range == 0:
        ssim_val = 1.0 if np.allclose(gt_field, interp_field) else 0.0
    else:
        ssim_val = ssim(gt_field, interp_field, data_range=data_range)
    return {'MSE': mse, 'MAE': mae, 'Max Error': max_err, 'SSIM': ssim_val}

# =============================================
# HELPER FOR SCIENTIFIC NOTATION OF SMALL NUMBERS
# =============================================
def format_small_number(val: float, threshold: float = 0.001, decimals: int = 3) -> str:
    """Return scientific notation if |val| < threshold, else fixed-point."""
    if abs(val) < threshold:
        return f"{val:.3e}"
    else:
        return f"{val:.{decimals}f}"

# ----------------------------------------------------------------------
# Callback for template buttons (fixes Streamlit crash)
# ----------------------------------------------------------------------
def set_template(text: str):
    st.session_state.designer_input = text

# =============================================
# UNIFIED LLM LOADER (with caching)
# =============================================
@st.cache_resource(show_spinner="Loading selected LLM (cached forever)...")
def load_llm(backend: str):
    if not TRANSFORMERS_AVAILABLE:
        return None, None, backend
    if "GPT-2" in backend:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        mod = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        name = "Qwen/Qwen2-0.5B-Instruct" if "Qwen2-0.5B" in backend else "Qwen/Qwen2.5-0.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    mod.eval()
    return tok, mod, backend

# =============================================
# INITIALIZE SESSION STATE (with manual LLM cache)
# =============================================
def initialize_session_state():
    defaults = {
        'solutions': [],
        'loader': None,
        'interpolator': None,
        'visualizer': None,
        'weight_visualizer': None,
        'multi_visualizer': None,
        'results_manager': None,
        'temporal_manager': None,
        'current_time': 1.0,
        'last_target_hash': None,
        'saved_predictions': [],
        'design_history': [],
        'nlp_parser': None,
        'relevance_scorer': None,
        'completion_analyzer': None,
        'designer_input': "",
        # NEW: LLM model and tokenizer (unified)
        'llm_tokenizer': None,
        'llm_model': None,
        # FIXED: separate key for loaded backend (widget key is 'llm_backend')
        'llm_backend_loaded': "GPT-2 (default, fastest startup)",
        'llm_available': TRANSFORMERS_AVAILABLE,
        # ADDED: manual LLM cache (OrderedDict for LRU)
        'llm_cache': OrderedDict(),
        'llm_cache_maxsize': 20,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if st.session_state.loader is None:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if st.session_state.interpolator is None:
        st.session_state.interpolator = CoreShellInterpolator()
    if st.session_state.visualizer is None:
        st.session_state.visualizer = HeatMapVisualizer()
    if st.session_state.weight_visualizer is None:
        st.session_state.weight_visualizer = HybridWeightVisualizer()
    if st.session_state.multi_visualizer is None:
        st.session_state.multi_visualizer = MultiPredictionVisualizer()
    if st.session_state.results_manager is None:
        st.session_state.results_manager = ResultsManager()
    if st.session_state.nlp_parser is None:
        st.session_state.nlp_parser = NLParser()
    if st.session_state.completion_analyzer is None:
        st.session_state.completion_analyzer = CompletionAnalyzer()

# =============================================
# RENDER INTELLIGENT DESIGNER TAB (with manual LLM cache)
# =============================================
def render_intelligent_designer_tab():
    st.markdown('<h2 class="section-header">🤖 Intelligent Designer</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #F0F9FF; border-left: 5px solid #3B82F6; padding: 1.2rem; 
                border-radius: 0.6rem; margin: 1.2rem 0; font-size: 1.1rem;">
    <strong>Design Goal:</strong> Describe your desired core‑shell nanoparticle in natural language.
    The system extracts parameters, estimates feasibility, and predicts shell formation using real physics‑based interpolation.
    <br><br>
    <em>Example inputs:</em>
    <ul>
    <li>"Design a core-shell with L0=50 nm, fc=0.2, c_bulk=0.3, time=1e-3 s"</li>
    <li>"I need a complete Ag shell at L0=40 nm, fc=0.25, c_bulk=0.1"</li>
    <li>"Optimize for minimal thickness with rs=0.2, fc=0.18, L0=60 nm"</li>
    <li>"Domain 70nm, core fraction 0.22, concentration 0.4, run for 0.001 seconds"</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        user_input = st.text_area(
            "Enter your design request:",
            height=120,
            placeholder="e.g., Design a core-shell with L0=50 nm, fc=0.2, c_bulk=0.3, time=1e-3 s",
            key="designer_input"
        )
    with col_input2:
        st.markdown("**Quick Templates:**")
        if st.button("🔬 Thin Shell", use_container_width=True,
                     on_click=set_template,
                     args=("Thin Ag shell with L0=40nm, fc=0.2, c_bulk=0.15, time=5e-4s",)):
            pass
        if st.button("📏 Thick Shell", use_container_width=True,
                     on_click=set_template,
                     args=("Thick Ag shell with L0=80nm, fc=0.15, c_bulk=0.8, time=2e-3s",)):
            pass
        if st.button("⚡ Fast Growth", use_container_width=True,
                     on_click=set_template,
                     args=("Fast deposition with L0=50nm, fc=0.25, c_bulk=0.6, high concentration",)):
            pass
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        run_design = st.button("🚀 Run Designer", type="primary", use_container_width=True)
    with col_btn2:
        use_scibert = st.checkbox("Use SciBERT (if available)", value=True,
                                  help="Enables semantic relevance scoring")
    with col_btn3:
        if st.session_state.saved_predictions:
            if st.button("📊 Compare All Saved Designs", use_container_width=True):
                st.session_state.active_tab = "Multi-Prediction Comparison"
                st.rerun()
    
    # NEW: Model selection dropdown – widget key is 'llm_backend'
    model_choice = st.selectbox(
        "🧠 LLM Backend for Parsing & Completeness",
        options=["GPT-2 (default, fastest startup)", 
                 "Qwen2-0.5B-Instruct (best JSON)", 
                 "Qwen2.5-0.5B-Instruct (newest, recommended)"],
        index=0,  # GPT-2 is default
        key="llm_backend"
    )
    
    # Unified loader – store loaded backend in separate key
    if st.session_state.llm_available:
        tokenizer, model, active_backend = load_llm(model_choice)
        st.session_state.llm_tokenizer = tokenizer
        st.session_state.llm_model = model
        st.session_state.llm_backend_loaded = active_backend  # FIXED: separate key
        # Show current backend in UI
        st.caption(f"Current backend: **{st.session_state.llm_backend_loaded}** — Qwen models give cleaner JSON")
    
    # LLM toggles (use loaded backend)
    llm_available = st.session_state.llm_available and st.session_state.llm_tokenizer is not None
    col_gpt1, col_gpt2, col_gpt3 = st.columns(3)
    with col_gpt1:
        use_llm_parse = st.checkbox("Use LLM for parameter extraction (hybrid)", 
                                      value=False, disabled=not llm_available,
                                      help="Combine regex and LLM with confidence")
    with col_gpt2:
        use_llm_complete = st.checkbox("Use LLM for completeness inference (experimental)",
                                         value=False, disabled=not llm_available,
                                         help="LLM will analyze shell quality and provide explanations")
    with col_gpt3:
        use_llm_ensemble = st.checkbox("Use LLM ensemble (slower)", 
                                         value=False, disabled=not llm_available,
                                         help="Run multiple generations and average")
        if use_llm_ensemble:
            ensemble_runs = st.number_input("Ensemble runs", min_value=2, max_value=10, value=3, step=1)
        else:
            ensemble_runs = 3
    
    if run_design and user_input:
        with st.spinner("🔍 Parsing natural language input..."):
            parser = st.session_state.nlp_parser
            # MODIFIED: Use hybrid_parse if LLM is enabled
            if use_llm_parse and st.session_state.llm_tokenizer is not None:
                target_design = parser.hybrid_parse(
                    user_input, 
                    st.session_state.llm_tokenizer, 
                    st.session_state.llm_model,
                    use_ensemble=use_llm_ensemble,
                    ensemble_runs=ensemble_runs
                )
            else:
                target_design = parser.parse(user_input)
            target_design['rs'] = 0.2  # as per spec
            
            design_record = {
                'timestamp': datetime.now().isoformat(),
                'input': user_input,
                'params': target_design.copy()
            }
            st.session_state.design_history.append(design_record)
        
        explanation = parser.get_explanation(target_design, user_input)
        st.markdown(explanation)
        
        st.markdown("#### 📊 Parameter Visualization")
        cols = st.columns(5)
        param_icons = {'L0_nm': '📏', 'fc': '🔵', 'rs': '🟠', 'c_bulk': '🧪', 'time': '⏱️'}
        param_units = {'L0_nm': 'nm', 'fc': '', 'rs': '', 'c_bulk': '', 'time': 's'}
        
        for i, (key, val) in enumerate(target_design.items()):
            if key in ['bc_type', 'use_edl', 'mode', 'alpha_nd', 'tau0_s']:
                continue
            with cols[i % 5]:
                icon = param_icons.get(key, '•')
                unit = param_units.get(key, '')
                val_str = f"{val:.2e} {unit}" if isinstance(val, float) and (val < 0.01 or val > 1000) else f"{val} {unit}"
                if val is None:
                    val_str = "Full evolution"
                st.metric(f"{icon} {key}", val_str)
        
        if not st.session_state.solutions:
            st.error("⚠️ No simulation solutions loaded. Please load solutions in the sidebar first.")
            st.info("👈 Go to the sidebar and click 'Load Solutions' to import your numerical simulation data.")
            return
        
        # Clean up previous animation temp files before creating new manager
        if st.session_state.get('temporal_manager') is not None:
            st.session_state.temporal_manager.cleanup_animation()
        
        with st.spinner("⚙️ Initializing simulation environment..."):
            try:
                design_manager = TemporalFieldManager(
                    st.session_state.interpolator,
                    st.session_state.solutions,
                    target_design,
                    n_key_frames=5,
                    lru_size=2,
                    require_categorical_match=False
                )
                st.session_state.temporal_manager = design_manager  # store for other tabs
            except Exception as e:
                st.error(f"Failed to initialize simulation: {e}")
                return
        
        with st.spinner("🧠 Computing semantic relevance..."):
            if st.session_state.relevance_scorer is None:
                st.session_state.relevance_scorer = RelevanceScorer(use_scibert=use_scibert)
            
            scorer = st.session_state.relevance_scorer
            weights = np.array(design_manager.weights.get('combined', [1.0]))
            relevance = scorer.score(user_input, st.session_state.solutions, weights)
            confidence_text, confidence_color = scorer.get_confidence_level(relevance)
        
        # ------------------------------------------------------------------
        # SLIDER AND DYNAMIC CONTENT (completeness recomputed on slider change)
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("#### ⏱️ Select Any Time Point")
        
        # Determine default normalized time from user input
        times_norm = list(design_manager.key_frames.keys())
        times_real = [design_manager.get_time_real(t) for t in times_norm]
        if target_design['time'] is not None:
            target_time_real = target_design['time']
            default_idx = np.argmin(np.abs(np.array(times_real) - target_time_real))
            default_norm = times_norm[default_idx]
        else:
            default_norm = 1.0
        
        slider_col1, slider_col2 = st.columns([3, 1])
        with slider_col1:
            selected_time_norm = st.slider(
                f"Normalized time (0.0 = start → 1.0 = end) | Real time: {times_real[0]:.2e}s – {times_real[-1]:.2e}s",
                0.0, 1.0,
                value=float(default_norm),
                step=0.001,
                format="%.4f"
            )
        with slider_col2:
            t_sel_real = design_manager.get_time_real(selected_time_norm)
            st.markdown(f"**t = {t_sel_real:.3e} s**")
        
        # Fetch fields for selected time
        fields_sel = design_manager.get_fields(selected_time_norm, use_interpolation=True)
        
        # ==================================================================
        # DYNAMIC COMPLETENESS ANALYSIS (based on selected_time_norm)
        # ==================================================================
        analyzer = st.session_state.completion_analyzer
        core_radius_nm = target_design.get('fc', 0.18) * target_design.get('L0_nm', 60.0) / 2
        
        # Recompute shell quality using current fields
        shell_quality = analyzer.compute_shell_quality(
            fields_sel['phi'], fields_sel['psi'], core_radius_nm, target_design.get('L0_nm', 60.0)
        )
        
        # Recompute completeness up to selected time
        t_complete, dr_min, is_complete = analyzer.compute_completion(
            design_manager, target_design, 
            max_time_norm=selected_time_norm,
            tolerance=0.3, use_median=True, completeness_threshold=0.95
        )
        
        # LLM completeness inference (if enabled) – now with manual cache
        llm_explanation = None
        llm_complete = None
        if use_llm_complete and st.session_state.llm_tokenizer is not None:
            with st.spinner("🤖 LLM analyzing shell completeness..."):
                # Build description from current fields
                r, prof = DepositionPhysics.compute_radial_profile(
                    DepositionPhysics.material_proxy(fields_sel['phi'], fields_sel['psi']),
                    target_design.get('L0_nm', 60.0), use_median=True
                )
                core_idx = np.argmin(np.abs(r - core_radius_nm))
                profile_str = ', '.join([f"{x:.2f}" for x in prof[core_idx:core_idx+10]])  # first 10 bins
                desc = f"""
                - Core radius: {core_radius_nm:.2f} nm
                - Radial profile (first 10 bins after core): {profile_str}
                - Coverage: {shell_quality['coverage']*100:.1f}%
                - Uniformity std: {shell_quality['uniformity']:.2f} nm
                - Intrusion: {shell_quality['intrusion']:.3f}
                """
                # Compute hash for caching
                desc_hash = hashlib.md5(desc.encode()).hexdigest()
                
                # Check manual cache
                if desc_hash in st.session_state.llm_cache:
                    llm_complete, llm_t, llm_dr, llm_explanation = st.session_state.llm_cache[desc_hash]
                else:
                    # Call LLM (uncached)
                    llm_complete, llm_t, llm_dr, llm_explanation = CompletionAnalyzer.llm_infer_completeness(
                        desc, st.session_state.llm_tokenizer, st.session_state.llm_model
                    )
                    # Store in cache with LRU eviction
                    cache = st.session_state.llm_cache
                    cache[desc_hash] = (llm_complete, llm_t, llm_dr, llm_explanation)
                    if len(cache) > st.session_state.llm_cache_maxsize:
                        # pop the oldest item (OrderedDict maintains insertion order)
                        cache.popitem(last=False)
        
        # ==================================================================
        # DISPLAY DYNAMIC RESULTS
        # ==================================================================
        st.markdown("#### 🎯 Design Analysis Results")
        res_cols = st.columns(4)
        with res_cols[0]:
            st.metric("Relevance Score", f"{relevance:.3f}",
                     help="0-1 scale: semantic match between query and available data")
            st.markdown(f"<span style='color:{confidence_color};font-weight:bold;'>{confidence_text}</span>", 
                       unsafe_allow_html=True)
        with res_cols[1]:
            if dr_min is not None:
                st.metric("Min. Thickness", f"{dr_min:.2f} nm")
            else:
                st.metric("Min. Thickness", "N/A")
        with res_cols[2]:
            if t_complete is not None:
                st.metric("Completion Time", f"{t_complete:.2e} s")
            else:
                st.metric("Completion Time", "Incomplete")
        with res_cols[3]:
            if is_complete:
                st.success("✅ Complete")
            else:
                if t_complete:
                    st.warning("⏳ Pending")
                else:
                    st.error("❌ Failed")
        
        st.markdown("#### 🛡️ Shell Quality Metrics")
        qcols = st.columns(3)
        with qcols[0]:
            st.metric("Intrusion (mean φ in core)", f"{shell_quality['intrusion']:.3f}",
                      help="Should be <0.1 to avoid Ag inside core")
        with qcols[1]:
            st.metric("Coverage (% boundary covered)", f"{shell_quality['coverage']*100:.1f}%",
                      help=">95% indicates continuous shell")
        with qcols[2]:
            st.metric("Uniformity (std dev)", f"{shell_quality['uniformity']:.2f} nm",
                      help="Lower is more uniform")
        
        if use_llm_complete and llm_explanation is not None:
            st.markdown("#### 🤖 LLM Completeness Insight")
            st.info(f"**LLM says:** {'Complete' if llm_complete else 'Incomplete'} | "
                    f"Est. completion time: {llm_t if llm_t else 'N/A'} s | "
                    f"Min thickness: {llm_dr if llm_dr else 'N/A'} nm")
            st.caption(f"Explanation: {llm_explanation}")
        
        # ------------------------------------------------------------------
        # VISUALIZATIONS (use fields_sel)
        # ------------------------------------------------------------------
        if fields_sel:
            proxy_sel = DepositionPhysics.material_proxy(
                fields_sel.get('phi', np.zeros((256, 256))),
                fields_sel.get('psi', np.zeros((256, 256)))
            )
            
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                fig_mat = st.session_state.visualizer.create_field_heatmap(
                    proxy_sel,
                    title=f"Material Proxy (t={t_sel_real:.2e}s)",
                    cmap_name='Set1',
                    L0_nm=target_design['L0_nm'],
                    target_params=target_design,
                    time_real_s=t_sel_real
                )
                st.pyplot(fig_mat)
            
            with viz_col2:
                fig_inter = st.session_state.visualizer.create_interactive_heatmap(
                    proxy_sel,
                    title=f"Interactive View (t={t_sel_real:.2e}s)",
                    cmap_name='Set1',
                    L0_nm=target_design['L0_nm'],
                    target_params=target_design,
                    time_real_s=t_sel_real
                )
                st.plotly_chart(fig_inter, use_container_width=True)
        
        st.markdown("#### 💡 Optimization Recommendations")
        # MODIFIED: pass llm_complete to generate_recommendations
        recommendations = analyzer.generate_recommendations(
            target_design, relevance, t_complete, dr_min, is_complete, shell_quality, llm_explanation, llm_complete
        )
        for rec in recommendations:
            st.markdown(rec)
        
        st.markdown("---")
        save_cols = st.columns([1, 3])
        with save_cols[0]:
            design_name = st.text_input("Design name (optional):", 
                                       value=f"Design_{len(st.session_state.saved_predictions)+1}")
        with save_cols[1]:
            if st.button("💾 Save This Design for Comparison", use_container_width=True):
                with st.spinner("Saving prediction..."):
                    t_norm_requested = target_design['time'] / design_manager.get_time_real(1.0) if target_design['time'] else 1.0
                    t_norm_requested = np.clip(t_norm_requested, 0, 1)
                    
                    res_design = st.session_state.interpolator.interpolate_fields(
                        st.session_state.solutions,
                        target_design,
                        target_shape=(256, 256),
                        n_time_points=100,
                        time_norm=t_norm_requested,
                        recompute_thickness=True
                    )
                    
                    if res_design:
                        res_design['design_name'] = design_name
                        res_design['input_text'] = user_input
                        res_design['relevance_score'] = relevance
                        st.session_state.saved_predictions.append(res_design)
                        st.success(f"✅ Design '{design_name}' saved! Total saved: {len(st.session_state.saved_predictions)}")
                    else:
                        st.error("❌ Failed to save design.")
        
        st.markdown("#### 📈 Predicted Thickness Evolution")
        thick_time = design_manager.thickness_time
        if thick_time and 'th_nm' in thick_time:
            fig_thick = go.Figure()
            fig_thick.add_trace(go.Scatter(
                x=thick_time.get('t_real_s', thick_time['t_norm']),
                y=thick_time['th_nm'],
                mode='lines',
                name='Interpolated',
                line=dict(color='blue', width=3)
            ))
            
            if t_complete is not None:
                idx_complete = np.argmin(np.abs(np.array(thick_time.get('t_real_s', thick_time['t_norm'])) - t_complete))
                fig_thick.add_vline(x=t_complete, line_dash="dash", line_color="green",
                                   annotation_text="Completion")
                fig_thick.add_trace(go.Scatter(
                    x=[t_complete],
                    y=[thick_time['th_nm'][idx_complete]],
                    mode='markers',
                    marker=dict(size=12, color='green', symbol='star'),
                    name='Completion Point'
                ))
            
            if target_design['time'] is not None:
                fig_thick.add_vline(x=target_design['time'], line_dash="dot", line_color="red",
                                   annotation_text="Requested")
            
            fig_thick.update_layout(
                title='Shell Thickness vs. Time',
                xaxis_title='Time (s)',
                yaxis_title='Thickness (nm)',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_thick, use_container_width=True)

# =============================================
# MAIN APP (unchanged from previous correct version)
# =============================================
def main():
    st.set_page_config(
        page_title="Intelligent Core-Shell Designer with Hybrid‑Weight Interpolation",
        layout="wide",
        page_icon="🧪",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.2rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 2.0rem;
        color: #374151;
        font-weight: 800;
        border-left: 6px solid #3B82F6;
        padding-left: 1.2rem;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    .memory-stats {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1.0rem;
        border-radius: 0.4rem;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    .weight-formula {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 5px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧪 CoreShellGPT:Intelligent Core‑Shell Designer with Real Physics Interpolation</h1>',
               unsafe_allow_html=True)
    
    initialize_session_state()
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.markdown("### 📁 Data Management")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if st.button("📥 Load Solutions", use_container_width=True):
                with st.spinner("Loading simulation data..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                    if st.session_state.get('temporal_manager') is not None:
                        st.session_state.temporal_manager.clear_lru_cache()
        with col2:
            if st.button("🧹 Clear All", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.temporal_manager = None
                st.session_state.last_target_hash = None
                st.session_state.saved_predictions = []
                if os.path.exists(TEMP_ANIMATION_DIR):
                    shutil.rmtree(TEMP_ANIMATION_DIR)
                os.makedirs(TEMP_ANIMATION_DIR, exist_ok=True)
                st.success("All cleared")
        with col3:
            if st.session_state.get('temporal_manager') is not None:
                if st.button("🗑️ Clear Cache", use_container_width=True, help="Clears LRU cache only"):
                    st.session_state.temporal_manager.clear_lru_cache()
                    st.rerun()
        
        if st.session_state.solutions:
            st.success(f"✅ {len(st.session_state.solutions)} solutions loaded")
        
        st.markdown("---")
        st.markdown("### 🧠 Interpolation Hyperparameters")
        sigma_fc = st.slider("σ (fc)", 0.05, 0.3, 0.15, 0.01)
        sigma_rs = st.slider("σ (rs)", 0.05, 0.3, 0.15, 0.01)
        sigma_c = st.slider("σ (c_bulk)", 0.05, 0.3, 0.15, 0.01)
        sigma_L = st.slider("σ (L0_nm)", 0.05, 0.3, 0.15, 0.01)
        temperature = st.slider("Attention temperature", 0.1, 10.0, 1.0, 0.1)
        gating_mode = st.selectbox(
            "Gating Mode",
            ["Hierarchical: L0 → fc → rs → c_bulk",
             "Hierarchical-Parallel: L0 → (fc, rs, c_bulk)",
             "Joint Multiplicative",
             "No Gating"],
            index=0
        )
        lambda_shape = st.slider("λ (shape boost)", 0.0, 1.0, 0.5, 0.05)
        sigma_shape = st.slider("σ_shape (radial similarity)", 0.05, 0.5, 0.15, 0.01)
        require_categorical_match = st.checkbox("Require exact categorical match", value=False)
        n_key_frames = st.slider("Key frames", 1, 20, 5, 1)
        lru_cache_size = st.slider("LRU cache size", 1, 5, 2, 1)
        
        st.session_state.interpolator.set_parameter_sigma([sigma_fc, sigma_rs, sigma_c, sigma_L])
        st.session_state.interpolator.temperature = temperature
        st.session_state.interpolator.set_gating_mode(gating_mode)
        st.session_state.interpolator.set_shape_params(lambda_shape, sigma_shape)
        
        st.markdown("---")
        st.markdown("### 🗑️ Cache Management")
        if st.session_state.get('temporal_manager') is not None:
            col_clear1, col_clear2 = st.columns(2)
            with col_clear1:
                if st.button("Clear LRU Cache", use_container_width=True):
                    st.session_state.temporal_manager.clear_lru_cache()
            with col_clear2:
                if st.button("Recompute Key Frames", use_container_width=True):
                    with st.spinner("Recomputing..."):
                        st.session_state.temporal_manager.recompute_key_frames()
            
            mem_stats = st.session_state.temporal_manager.get_memory_stats()
            st.markdown(f"""
            <div class="memory-stats">
            <strong>📊 Cache Memory</strong><br>
            Key frames: {mem_stats['key_frames_mb']:.2f} MB<br>
            LRU cache: {mem_stats['lru_cache_mb']:.2f} MB<br>
            Total: {mem_stats['total_mb']:.2f} MB
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No active temporal manager. Run a design first.")
    
    tabs = st.tabs([
        "🤖 Intelligent Designer",
        "📊 Field Visualization",
        "📈 Thickness Evolution",
        "🎬 Animation",
        "🧪 Derived Quantities",
        "⚖️ Hybrid Weight Analysis",
        "💾 Export",
        "🔍 Ground Truth Comparison",
        "📊 Multi-Prediction Comparison"
    ])
    
    with tabs[0]:
        render_intelligent_designer_tab()
    
    mgr = st.session_state.get('temporal_manager', None)
    
    with tabs[1]:
        st.markdown('<h2 class="section-header">📊 Field Visualization</h2>', unsafe_allow_html=True)
        if mgr is None:
            st.info("Please run a design in the Intelligent Designer tab first.")
        else:
            target = mgr.target_params
            current_time_norm = st.session_state.current_time
            current_time_real = mgr.get_time_real(current_time_norm)
            fields = mgr.get_fields(current_time_norm, use_interpolation=True)
            
            field_choice = st.selectbox("Select field",
                                       ['c (concentration)', 'phi (shell)', 'psi (core)',
                                        'material proxy'],
                                       key='viz_field')
            field_map = {'c (concentration)': 'c', 'phi (shell)': 'phi',
                        'psi (core)': 'psi', 'material proxy': 'material'}
            field_key = field_map[field_choice]
            
            if field_key == 'material':
                field_data = DepositionPhysics.material_proxy(fields['phi'], fields['psi'])
            else:
                field_data = fields[field_key]
            
            cmap_cat = st.selectbox("Colormap category", list(COLORMAP_OPTIONS.keys()), index=0, key='viz_cmap_cat')
            cmap = st.selectbox("Colormap", COLORMAP_OPTIONS[cmap_cat], index=0, key='viz_cmap')
            
            fig = st.session_state.visualizer.create_field_heatmap(
                field_data,
                title=f"Interpolated {field_choice}",
                cmap_name=cmap,
                L0_nm=target.get('L0_nm', 60.0),
                target_params=target,
                time_real_s=current_time_real,
                colorbar_label="Material" if field_key == 'material' else field_choice.split()[0]
            )
            st.pyplot(fig)
            
            if st.checkbox("Show interactive heatmap"):
                fig_inter = st.session_state.visualizer.create_interactive_heatmap(
                    field_data,
                    title=f"Interactive {field_choice}",
                    cmap_name=cmap,
                    L0_nm=target.get('L0_nm', 60.0),
                    target_params=target,
                    time_real_s=current_time_real
                )
                st.plotly_chart(fig_inter, use_container_width=True)
    
    with tabs[2]:
        st.markdown('<h2 class="section-header">📈 Thickness Evolution</h2>', unsafe_allow_html=True)
        if mgr is None:
            st.info("Please run a design in the Intelligent Designer tab first.")
        else:
            thickness_time = mgr.thickness_time
            show_growth = st.checkbox("Show growth rate", value=False)
            fig_th = st.session_state.visualizer.create_thickness_plot(
                thickness_time,
                title=f"Shell Thickness Evolution",
                current_time_norm=st.session_state.current_time,
                current_time_real=mgr.get_time_real(st.session_state.current_time),
                show_growth_rate=show_growth
            )
            st.pyplot(fig_th)
    
    with tabs[3]:
        st.markdown('<h2 class="section-header">🎬 Animation</h2>', unsafe_allow_html=True)
        if mgr is None:
            st.info("Please run a design in the Intelligent Designer tab first.")
        else:
            anim_method = st.radio("Animation method",
                                  ["Real-time interpolation", "Pre-rendered (smooth)"])
            if anim_method == "Real-time interpolation":
                fps = st.slider("FPS", 1, 30, 10)
                n_frames = st.slider("Frames", 10, 100, 30)
                if st.button("▶️ Play Animation", use_container_width=True):
                    placeholder = st.empty()
                    times = np.linspace(0, 1, n_frames)
                    for t_norm in times:
                        fields = mgr.get_fields(t_norm, use_interpolation=True)
                        t_real = mgr.get_time_real(t_norm)
                        field_data = DepositionPhysics.material_proxy(fields['phi'], fields['psi'])
                        fig = st.session_state.visualizer.create_field_heatmap(
                            field_data,
                            title=f"t = {t_real:.3e} s",
                            cmap_name='Set1',
                            L0_nm=mgr.target_params.get('L0_nm', 60.0),
                            target_params=mgr.target_params,
                            time_real_s=t_real,
                            colorbar_label="Material"
                        )
                        placeholder.pyplot(fig)
                        time.sleep(1/fps)
                    st.success("Animation complete")
            else:
                n_frames = st.slider("Pre-render frames", 20, 100, 50)
                if st.button("🎥 Pre-render Animation", use_container_width=True):
                    with st.spinner(f"Rendering {n_frames} frames to disk..."):
                        mgr.prepare_animation_streaming(n_frames)
                        st.success(f"Pre-rendered {n_frames} frames")
                if mgr.animation_frame_paths:
                    fps = st.slider("Playback FPS", 1, 30, 15)
                    if st.button("▶️ Play Pre-rendered", use_container_width=True):
                        placeholder = st.empty()
                        for i, _ in enumerate(mgr.animation_frame_paths):
                            frame = mgr.get_animation_frame(i)
                            if frame:
                                fields = {'phi': frame['phi'], 'c': frame['c'], 'psi': frame['psi']}
                                t_real = frame['time_real_s']
                                field_data = DepositionPhysics.material_proxy(fields['phi'], fields['psi'])
                                fig = st.session_state.visualizer.create_field_heatmap(
                                    field_data,
                                    title=f"t = {t_real:.3e} s [Pre-rendered]",
                                    cmap_name='Set1',
                                    L0_nm=mgr.target_params.get('L0_nm', 60.0),
                                    target_params=mgr.target_params,
                                    time_real_s=t_real,
                                    colorbar_label="Material"
                                )
                                placeholder.pyplot(fig)
                                time.sleep(1/fps)
                        st.success("Playback complete")
                    if st.button("🗑️ Clean Pre-rendered", use_container_width=True):
                        mgr.cleanup_animation()
                        st.success("Cleaned up")
    
    with tabs[4]:
        st.markdown('<h2 class="section-header">🧪 Derived Quantities</h2>', unsafe_allow_html=True)
        if mgr is None:
            st.info("Please run a design in the Intelligent Designer tab first.")
        else:
            res = st.session_state.interpolator.interpolate_fields(
                st.session_state.solutions, mgr.target_params, target_shape=(256,256),
                n_time_points=100, time_norm=st.session_state.current_time,
                require_categorical_match=st.session_state.get('require_categorical_match', False),
                recompute_thickness=True
            )
            if res:
                col1, col2, col3 = st.columns(3)
                with col1:
                    val = res['derived']['thickness_nm']
                    st.metric("Shell thickness (nm)", format_small_number(val))
                with col2:
                    val = res['derived'].get('growth_rate', 0)
                    st.metric("Growth rate (nm/s)", format_small_number(val))
                with col3:
                    st.metric("Sources used", res['num_sources'])
                
                st.subheader("Phase Statistics")
                stats = res['derived']['phase_stats']
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Electrolyte", f"{stats['Electrolyte'][0]:.4f} nd²",
                             help=f"Real: {stats['Electrolyte'][1]*1e18:.2f} nm²")
                with cols[1]:
                    st.metric("Ag shell", f"{stats['Ag'][0]:.4f} nd²",
                             help=f"Real: {stats['Ag'][1]*1e18:.2f} nm²")
                with cols[2]:
                    st.metric("Cu core", f"{stats['Cu'][0]:.4f} nd²",
                             help=f"Real: {stats['Cu'][1]*1e18:.2f} nm²")
                
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    st.subheader("Material Proxy")
                    fig_mat = st.session_state.visualizer.create_field_heatmap(
                        res['derived']['material'], "Material Proxy",
                        cmap_name='Set1', L0_nm=mgr.target_params.get('L0_nm',60.0),
                        target_params=mgr.target_params,
                        colorbar_label="Material", vmin=0, vmax=2,
                        time_real_s=mgr.get_time_real(st.session_state.current_time)
                    )
                    st.pyplot(fig_mat)
                with col_viz2:
                    st.subheader("Potential Proxy")
                    fig_pot = st.session_state.visualizer.create_field_heatmap(
                        res['derived']['potential'], "Potential Proxy",
                        cmap_name='RdBu_r', L0_nm=mgr.target_params.get('L0_nm',60.0),
                        target_params=mgr.target_params,
                        colorbar_label="-α·c",
                        time_real_s=mgr.get_time_real(st.session_state.current_time)
                    )
                    st.pyplot(fig_pot)
    
    with tabs[5]:
        st.markdown('<h2 class="section-header">⚖️ Hybrid Weight Analysis</h2>', unsafe_allow_html=True)
        if mgr is None:
            st.info("Please run a design in the Intelligent Designer tab first.")
        else:
            sources_data = mgr.sources_data if mgr.sources_data else []
            if sources_data:
                weight_tabs = st.tabs(["Sankey", "Chord", "Radar", "Breakdown"])
                with weight_tabs[0]:
                    fig = st.session_state.weight_visualizer.create_enhanced_sankey_diagram(
                        sources_data, mgr.target_params, [sigma_fc, sigma_rs, sigma_c, sigma_L]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with weight_tabs[1]:
                    fig = st.session_state.weight_visualizer.create_enhanced_chord_diagram(
                        sources_data, mgr.target_params
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with weight_tabs[2]:
                    radars = st.session_state.weight_visualizer.create_parameter_radar_charts(
                        sources_data, mgr.target_params, [sigma_fc, sigma_rs, sigma_c, sigma_L]
                    )
                    col1, col2 = st.columns(2)
                    for i, radar in enumerate(radars):
                        with col1 if i%2==0 else col2:
                            st.plotly_chart(radar, use_container_width=True)
                with weight_tabs[3]:
                    fig = st.session_state.weight_visualizer.create_weight_formula_breakdown(
                        sources_data, mgr.target_params, [sigma_fc, sigma_rs, sigma_c, sigma_L]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 📋 Source Weight Table")
                df_weights = pd.DataFrame(sources_data)
                st.dataframe(df_weights.style.format({
                    'L0_nm': '{:.0f}',
                    'fc': '{:.3f}',
                    'rs': '{:.3f}',
                    'c_bulk': '{:.2f}',
                    'l0_weight': '{:.4f}',
                    'fc_weight': '{:.4f}',
                    'rs_weight': '{:.4f}',
                    'c_bulk_weight': '{:.4f}',
                    'attention_weight': '{:.4f}',
                    'combined_weight': '{:.4f}'
                }).background_gradient(subset=['combined_weight'], cmap='viridis'), use_container_width=True)
            else:
                st.info("No source weight data available.")
    
    with tabs[6]:
        st.markdown('<h2 class="section-header">💾 Export</h2>', unsafe_allow_html=True)
        if mgr is None:
            st.info("Please run a design in the Intelligent Designer tab first.")
        else:
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📊 Export Current State (JSON)", use_container_width=True):
                    res = st.session_state.interpolator.interpolate_fields(
                        st.session_state.solutions, mgr.target_params, target_shape=(256,256),
                        n_time_points=100, time_norm=st.session_state.current_time,
                        require_categorical_match=st.session_state.get('require_categorical_match', False),
                        recompute_thickness=True
                    )
                    if res:
                        export_data = st.session_state.results_manager.prepare_export_data(
                            res, {'time_norm': st.session_state.current_time,
                                 'time_real_s': mgr.get_time_real(st.session_state.current_time)}
                        )
                        json_str, fname = st.session_state.results_manager.export_to_json(export_data)
                        st.download_button("⬇️ Download JSON", json_str, fname, "application/json")
            with col_exp2:
                if st.button("📈 Export Current State (CSV)", use_container_width=True):
                    res = st.session_state.interpolator.interpolate_fields(
                        st.session_state.solutions, mgr.target_params, target_shape=(256,256),
                        n_time_points=100, time_norm=st.session_state.current_time,
                        require_categorical_match=st.session_state.get('require_categorical_match', False),
                        recompute_thickness=True
                    )
                    if res:
                        csv_str, fname = st.session_state.results_manager.export_to_csv(res)
                        st.download_button("⬇️ Download CSV", csv_str, fname, "text/csv")
    
    with tabs[7]:
        st.markdown('<h2 class="section-header">🔍 Ground Truth Comparison</h2>', unsafe_allow_html=True)
        if mgr is None:
            st.info("Please run a design in the Intelligent Designer tab first.")
        elif not st.session_state.solutions:
            st.warning("No solutions loaded.")
        else:
            target = mgr.target_params
            sources_data = mgr.sources_data if mgr.sources_data else []
            if sources_data:
                weights_combined = mgr.weights['combined']
                source_list = list(enumerate(mgr.sources))
                source_list.sort(key=lambda x: weights_combined[x[0]], reverse=True)
                gt_options = []
                gt_indices = []
                for pos, src in source_list:
                    params = src['params']
                    weight = weights_combined[pos]
                    gt_options.append(f"Source {pos} (hybrid={weight:.4f}): L0={params.get('L0_nm',20):.1f}nm, fc={params.get('fc',0.18):.3f}, rs={params.get('rs',0.2):.3f}, c_bulk={params.get('c_bulk',0.5):.2f}")
                    gt_indices.append(pos)
            else:
                st.info("No source weight data available.")
                gt_options = []
            
            if gt_options:
                selected_gt_label = st.selectbox("Select Ground Truth Simulation", gt_options)
                selected_idx = gt_indices[gt_options.index(selected_gt_label)]
                gt_sol = st.session_state.solutions[selected_idx]
                gt_params = gt_sol['params']
                
                with st.spinner("Computing interpolation at ground truth parameters..."):
                    interp_res = st.session_state.interpolator.interpolate_fields(
                        st.session_state.solutions, gt_params, target_shape=(256,256),
                        n_time_points=100, time_norm=st.session_state.current_time,
                        require_categorical_match=st.session_state.get('require_categorical_match', False),
                        recompute_thickness=True
                    )
                    
                    if interp_res:
                        gt_mgr = TemporalFieldManager(st.session_state.interpolator,
                                                     [gt_sol], gt_params,
                                                     n_key_frames=5, lru_size=1,
                                                     require_categorical_match=False)
                        gt_fields = gt_mgr.get_fields(st.session_state.current_time)
                        
                        target_shape = interp_res['shape']
                        for key in gt_fields:
                            if gt_fields[key].shape != target_shape:
                                factors = (target_shape[0]/gt_fields[key].shape[0],
                                          target_shape[1]/gt_fields[key].shape[1])
                                gt_fields[key] = zoom(gt_fields[key], factors, order=1)
                        
                        interp_material = DepositionPhysics.material_proxy(
                            interp_res['fields']['phi'], interp_res['fields']['psi'])
                        gt_material = DepositionPhysics.material_proxy(
                            gt_fields['phi'], gt_fields['psi'])
                        
                        compare_fields = {
                            'phi (shell)': ('phi', 'viridis'),
                            'c (concentration)': ('c', 'plasma'),
                            'psi (core)': ('psi', 'inferno'),
                            'material proxy': ('material', 'Set1')
                        }
                        
                        use_physical_alignment = st.checkbox("✅ Use Physical Coordinate Alignment", value=True)
                        
                        for field_title, (field_key, cmap_choice) in compare_fields.items():
                            st.subheader(field_title)
                            
                            if field_key == 'material':
                                interp_data = interp_material
                                gt_data = gt_material
                            else:
                                interp_data = interp_res['fields'][field_key]
                                gt_data = gt_fields[field_key]
                            
                            if use_physical_alignment:
                                gt_L0 = gt_params.get('L0_nm', 20.0)
                                interp_L0 = gt_params.get('L0_nm', 20.0)
                                comparison = compare_fields_physical(
                                    gt_data, gt_L0, interp_data, interp_L0,
                                    target_resolution_nm=0.2,
                                    compare_region='overlap'
                                )
                                gt_aligned = comparison['gt_aligned']
                                interp_aligned = comparison['interp_aligned']
                                errors = comparison['metrics']
                                st.info(f"📐 Aligned to {comparison['L_ref']:.1f}nm grid, "
                                     f"{errors['valid_pixels']} valid pixels")
                            else:
                                gt_aligned = gt_data
                                interp_aligned = interp_data
                                errors = compute_errors(gt_data, interp_data)
                            
                            col_err1, col_err2, col_err3, col_err4 = st.columns(4)
                            col_err1.metric("MSE", f"{errors['MSE']:.4e}")
                            col_err2.metric("MAE", f"{errors['MAE']:.4e}")
                            col_err3.metric("Max Error", f"{errors['Max Error']:.4e}")
                            col_err4.metric("SSIM", f"{errors['SSIM']:.3f}" if not np.isnan(errors['SSIM']) else "N/A")
                            
                            col_plot1, col_plot2, col_plot3 = st.columns(3)
                            with col_plot1:
                                st.markdown("**Ground Truth**")
                                fig_gt = st.session_state.visualizer.create_field_heatmap(
                                    gt_aligned, "Ground Truth", cmap_name=cmap_choice,
                                    L0_nm=comparison['L_ref'] if use_physical_alignment else gt_params.get('L0_nm',20.0)
                                )
                                st.pyplot(fig_gt)
                            with col_plot2:
                                st.markdown("**Interpolated**")
                                fig_interp = st.session_state.visualizer.create_field_heatmap(
                                    interp_aligned, "Interpolated", cmap_name=cmap_choice,
                                    L0_nm=comparison['L_ref'] if use_physical_alignment else gt_params.get('L0_nm',20.0)
                                )
                                st.pyplot(fig_interp)
                            with col_plot3:
                                st.markdown("**Difference (GT - Interp)**")
                                diff = gt_aligned - interp_aligned
                                fig_diff = st.session_state.visualizer.create_field_heatmap(
                                    diff, "Difference", cmap_name='RdBu_r',
                                    L0_nm=comparison['L_ref'] if use_physical_alignment else gt_params.get('L0_nm',20.0)
                                )
                                st.pyplot(fig_diff)
                        
                        del gt_mgr
                    else:
                        st.error("Failed to compute interpolation for comparison.")
    
    with tabs[8]:
        st.markdown('<h2 class="section-header">📊 Multi-Prediction Comparison</h2>', unsafe_allow_html=True)
        if not st.session_state.saved_predictions:
            st.info("No saved predictions yet. Save some designs from the Intelligent Designer tab.")
        else:
            st.write(f"Total saved predictions: {len(st.session_state.saved_predictions)}")
            pred_labels_short = []
            for i, pred in enumerate(st.session_state.saved_predictions):
                p = pred['target_params']
                short_label = f"P{i}: L0={p.get('L0_nm',20):.0f} fc={p.get('fc',0.18):.2f}"
                pred_labels_short.append(short_label)
            
            selected_indices = st.multiselect("Select predictions to compare", 
                                              options=range(len(pred_labels_short)),
                                              format_func=lambda i: pred_labels_short[i],
                                              default=list(range(min(3, len(pred_labels_short)))))
            
            if selected_indices:
                selected_preds = [st.session_state.saved_predictions[i] for i in selected_indices]
                selected_labels = [pred_labels_short[i] for i in selected_indices]
                
                plot_type = st.radio("Plot type", 
                                     ["Thickness Evolution", "Thickness Evolution (Matplotlib)",
                                      "3D Thickness Evolution", "Parameter Map",
                                      "Radar Metrics", "Weight Sunburst",
                                      "3D Parameter Scatter", "Parallel Coordinates", "Pair Plot"])
                
                if plot_type == "Thickness Evolution":
                    fig = st.session_state.multi_visualizer.thickness_evolution_plot(selected_preds, selected_labels)
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Thickness Evolution (Matplotlib)":
                    fig = st.session_state.multi_visualizer.thickness_evolution_matplotlib(selected_preds, selected_labels)
                    st.pyplot(fig)
                elif plot_type == "3D Thickness Evolution":
                    param_y = st.selectbox("Y-axis parameter", ['L0_nm', 'fc', 'rs', 'c_bulk'], index=0)
                    param_color = st.selectbox("Color metric", ['growth_rate', 'thickness_nm', 'entropy', 'max_weight'], index=0)
                    cmap_cat_3d = st.selectbox("Colormap category", list(COLORMAP_OPTIONS.keys()), index=0, key='cmap_cat_3d')
                    cmap_3d = st.selectbox("Colormap", COLORMAP_OPTIONS[cmap_cat_3d], index=0, key='cmap_3d')
                    fig = st.session_state.multi_visualizer.thickness_evolution_3d(
                        selected_preds, selected_labels, param_y=param_y, param_color=param_color, cmap=cmap_3d)
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Parameter Map":
                    param_x = st.selectbox("X parameter", ['L0_nm', 'fc', 'rs', 'c_bulk'], key='pmap_x')
                    param_y = st.selectbox("Y parameter", ['fc', 'rs', 'c_bulk', 'L0_nm'], index=1, key='pmap_y')
                    metric = st.selectbox("Metric", ['thickness_nm', 'growth_rate', 'entropy'], key='pmap_metric')
                    plot_type_map = st.radio("Map type", ['scatter', 'contour'], horizontal=True)
                    cmap_cat_map = st.selectbox("Colormap category", list(COLORMAP_OPTIONS.keys()), index=0, key='cmap_cat_map')
                    cmap_map = st.selectbox("Colormap", COLORMAP_OPTIONS[cmap_cat_map], index=0, key='cmap_map')
                    fig = st.session_state.multi_visualizer.parameter_map(
                        selected_preds, param_x, param_y, metric, plot_type=plot_type_map, cmap=cmap_map)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough distinct parameter points.")
                elif plot_type == "Radar Metrics":
                    fig = st.session_state.multi_visualizer.radar_comparison(selected_preds, selected_labels)
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Weight Sunburst":
                    cmap_cat_sun = st.selectbox("Colormap category", list(COLORMAP_OPTIONS.keys()), index=0, key='cmap_cat_sun')
                    cmap_sun = st.selectbox("Colormap", COLORMAP_OPTIONS[cmap_cat_sun], index=0, key='cmap_sun')
                    fig = st.session_state.multi_visualizer.weight_sunburst(selected_preds, selected_labels, cmap=cmap_sun)
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "3D Parameter Scatter":
                    param_options = ['L0_nm', 'fc', 'rs', 'c_bulk']
                    metric_options = ['thickness_nm', 'growth_rate', 'entropy', 'max_weight']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_param = st.selectbox("X axis", param_options, index=0)
                    with col2:
                        y_param = st.selectbox("Y axis", param_options, index=1)
                    with col3:
                        z_param = st.selectbox("Z axis", param_options, index=2)
                    color_metric = st.selectbox("Color by", metric_options, index=0)
                    size_metric = st.selectbox("Size by", metric_options, index=1)
                    cmap_cat_scat = st.selectbox("Colormap category", list(COLORMAP_OPTIONS.keys()), index=0, key='cmap_cat_scat')
                    cmap_scat = st.selectbox("Colormap", COLORMAP_OPTIONS[cmap_cat_scat], index=0, key='cmap_scat')
                    fig = st.session_state.multi_visualizer.parameter_scatter_3d(
                        selected_preds, selected_labels, x_param, y_param, z_param,
                        color_metric=color_metric, size_metric=size_metric, cmap=cmap_scat)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data for 3D scatter.")
                elif plot_type == "Parallel Coordinates":
                    all_dims = ['L0_nm', 'fc', 'rs', 'c_bulk', 'thickness_nm', 'growth_rate', 'entropy']
                    selected_dims = st.multiselect("Select dimensions", all_dims, default=all_dims[:4])
                    dims = []
                    for dim in selected_dims:
                        values = []
                        for p in selected_preds:
                            if dim in p['target_params']:
                                values.append(p['target_params'][dim])
                            elif dim in p['derived']:
                                values.append(p['derived'][dim])
                            elif dim == 'entropy':
                                values.append(p.get('weights', {}).get('entropy', 0))
                            else:
                                values.append(0)
                        dims.append({'label': dim, 'values': values})
                    fig = st.session_state.multi_visualizer.parallel_coordinates(selected_preds, selected_labels, dims)
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Pair Plot":
                    all_cols = ['L0_nm', 'fc', 'rs', 'c_bulk', 'thickness_nm', 'growth_rate', 'entropy']
                    selected_cols = st.multiselect("Select columns", all_cols, default=all_cols[:3])
                    fig = st.session_state.multi_visualizer.pairplot(selected_preds, selected_labels, selected_cols)
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
