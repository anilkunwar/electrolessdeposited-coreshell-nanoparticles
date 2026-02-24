#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer-Inspired Interpolation for Electroless Ag Shell Deposition on Cu Core
FULL TEMPORAL SUPPORT + PHYSICS‑EMBEDDED KERNEL + RADIAL MORPHING
OPTIMIZED VERSION: Pre-morph caching + self-similar scaling + forced Numba JIT

Key optimizations:
- Radial morph happens ONCE per source (not per key frame)
- Thickness scaling uses cheap radial stretch instead of full morph
- Numba JIT is forced with contiguous float32 arrays (no silent fallback)
- Three-tier temporal caching: thickness curve + sparse key frames + LRU field cache
- Streaming animation with disk-backed frames for long sequences
- Real physical time (seconds) from source PKL τ₀
- Physics kernel based on absolute dimensionless differences
- Discrete material colorbar with EXACT phase-field colors
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
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import interp1d, CubicSpline, griddata
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import shutil
warnings.filterwarnings('ignore')

# =============================================
# NUMBA JIT COMPILATION FOR PERFORMANCE
# =============================================
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

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

COLORMAP_OPTIONS = {
    'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'hot', 'afmhot'],
    'Diverging': ['RdBu', 'RdYlBu', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'BrBG'],
    'Qualitative': ['tab10', 'tab20', 'Set1', 'Set2', 'Set3'],
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu']
}

# =============================================
# EXACT PHASE-FIELD MATERIAL COLORS (matching phase field generator)
# =============================================
MATERIAL_COLORS_EXACT = {
    'electrolyte': (0.894, 0.102, 0.110, 1.0),  # Red
    'Ag':          (1.000, 0.498, 0.000, 1.0),  # Orange
    'Cu':          (0.600, 0.600, 0.600, 1.0)   # Gray
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
        ny, nx = phi.shape
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dist = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
        mask = (phi > threshold) & (psi <= 0.5)
        if np.any(mask):
            max_dist = np.max(dist[mask])
            thickness = max_dist - core_radius_frac
            return max(0.0, thickness)
        else:
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

# =============================================
# NUMBA-OPTIMIZED PHYSICS KERNEL FUNCTIONS
# =============================================
@jit(nopython=True, cache=True)
def compute_dimensionless_groups_jit(L0_m, D_nd, gamma_nd, M_nd, k0_nd, c_bulk, fc, tau0_s, t_max_nd):
    """JIT-compiled version of dimensionless groups computation."""
    r_core = fc * L0_m
    Fo = D_nd * t_max_nd / (L0_m**2)
    kappa_star = gamma_nd * M_nd / (L0_m * k0_nd * c_bulk)
    Da = k0_nd * c_bulk * L0_m**2 / D_nd
    r_star = r_core
    log_c = np.log(c_bulk + 1e-12)
    return Fo, kappa_star, Da, r_star, log_c, L0_m, tau0_s

@jit(nopython=True, cache=True)
def physics_kernel_jit(src_Fo, src_kappa, src_Da, src_r_star, src_log_c,
                       tgt_Fo, tgt_kappa, tgt_Da, tgt_r_star, tgt_log_c, tgt_L0, sigma):
    """JIT-compiled physics kernel computation."""
    w_Fo = 1.0
    w_kappa = 0.5
    w_Da = 0.5
    w_r = 2.0
    w_logc = 1.0
    d2 = (w_Fo * (src_Fo - tgt_Fo)**2 +
          w_kappa * (src_kappa - tgt_kappa)**2 +
          w_Da * (src_Da - tgt_Da)**2 +
          w_r * ((src_r_star - tgt_r_star) / tgt_L0)**2 +
          w_logc * (src_log_c - tgt_log_c)**2)
    return np.exp(-0.5 * d2 / sigma**2)

@jit(nopython=True, cache=True, parallel=True)
def radial_morph_jit(src_phi, src_psi, src_c, src_fc, tgt_fc, shape_out):
    """JIT-compiled radial morphing with parallel execution."""
    ny, nx = shape_out
    src_ny, src_nx = src_phi.shape
    warped_phi = np.zeros((ny, nx))
    warped_psi = np.zeros((ny, nx))
    warped_c = np.zeros((ny, nx))
    scale = tgt_fc / (src_fc + 1e-8)
    
    for i in prange(ny):
        for j in range(nx):
            x_tgt = j / (nx - 1)
            y_tgt = i / (ny - 1)
            dist = np.sqrt((x_tgt - 0.5)**2 + (y_tgt - 0.5)**2)
            r_src = dist / scale
            r_src = np.clip(r_src, 0, 1)
            angle = np.arctan2(y_tgt - 0.5, x_tgt - 0.5)
            x_src = 0.5 + r_src * np.cos(angle)
            y_src = 0.5 + r_src * np.sin(angle)
            x_src = np.clip(x_src, 0, 1)
            y_src = np.clip(y_src, 0, 1)
            
            src_x_idx = x_src * (src_nx - 1)
            src_y_idx = y_src * (src_ny - 1)
            x0 = int(np.floor(src_x_idx))
            x1 = min(x0 + 1, src_nx - 1)
            y0 = int(np.floor(src_y_idx))
            y1 = min(y0 + 1, src_ny - 1)
            dx = src_x_idx - x0
            dy = src_y_idx - y0
            
            warped_phi[i, j] = (1-dx)*(1-dy)*src_phi[y0, x0] + dx*(1-dy)*src_phi[y0, x1] + \
                               (1-dx)*dy*src_phi[y1, x0] + dx*dy*src_phi[y1, x1]
            warped_psi[i, j] = (1-dx)*(1-dy)*src_psi[y0, x0] + dx*(1-dy)*src_psi[y0, x1] + \
                               (1-dx)*dy*src_psi[y1, x0] + dx*dy*src_psi[y1, x1]
            warped_c[i, j] = (1-dx)*(1-dy)*src_c[y0, x0] + dx*(1-dy)*src_c[y0, x1] + \
                             (1-dx)*dy*src_c[y1, x0] + dx*dy*src_c[y1, x1]
    
    return warped_phi, warped_psi, warped_c

@jit(nopython=True, cache=True, parallel=True)
def radial_scale_field_jit(field: np.ndarray, center: Tuple[float, float], 
                           scale_factor: float, shape_out: Tuple[int, int]) -> np.ndarray:
    """
    Cheap radial scaling of an already-morphed field.
    Only stretches/compresses radially from center based on thickness ratio.
    """
    ny, nx = shape_out
    out = np.zeros((ny, nx), dtype=np.float32)
    inv_scale = 1.0 / (scale_factor + 1e-8)
    cy, cx = center
    
    for i in prange(ny):
        for j in range(nx):
            dy = (i / (ny - 1)) - cy
            dx = (j / (nx - 1)) - cx
            r = np.sqrt(dx*dx + dy*dy)
            r_src = r * inv_scale
            if r_src > 1.0:
                continue
            angle = np.arctan2(dy, dx)
            y_src = cy + r_src * np.sin(angle)
            x_src = cx + r_src * np.cos(angle)
            
            if 0 <= x_src < 1.0 and 0 <= y_src < 1.0:
                src_ny, src_nx = field.shape
                src_x = x_src * (src_nx - 1)
                src_y = y_src * (src_ny - 1)
                x0 = int(np.floor(src_x)); x1 = min(x0 + 1, src_nx - 1)
                y0 = int(np.floor(src_y)); y1 = min(y0 + 1, src_ny - 1)
                fx = src_x - x0; fy = src_y - y0
                
                out[i, j] = ((1-fx)*(1-fy)*field[y0, x0] + 
                            fx*(1-fy)*field[y0, x1] + 
                            (1-fx)*fy*field[y1, x0] + 
                            fx*fy*field[y1, x1])
    return out

@jit(nopython=True, cache=True)
def compute_material_proxy_jit(phi, psi):
    """JIT-compiled material proxy computation."""
    ny, nx = phi.shape
    material = np.zeros((ny, nx), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            if psi[i, j] > 0.5:
                material[i, j] = 2.0
            elif phi[i, j] > 0.5:
                material[i, j] = 1.0
            else:
                material[i, j] = 0.0
    return material

@jit(nopython=True, cache=True)
def interpolate_thickness_jit(common_t_real, thickness_curves_t, thickness_curves_th, weights, n_curves):
    """JIT-compiled thickness interpolation."""
    n_points = len(common_t_real)
    thickness_interp = np.zeros(n_points)
    for i in range(n_points):
        t = common_t_real[i]
        for c in range(n_curves):
            t_arr = thickness_curves_t[c]
            th_arr = thickness_curves_th[c]
            n_t = len(t_arr)
            if t <= t_arr[0]:
                th_val = th_arr[0]
            elif t >= t_arr[-1]:
                th_val = th_arr[-1]
            else:
                for k in range(n_t - 1):
                    if t_arr[k] <= t <= t_arr[k+1]:
                        alpha = (t - t_arr[k]) / (t_arr[k+1] - t_arr[k] + 1e-12)
                        th_val = (1 - alpha) * th_arr[k] + alpha * th_arr[k+1]
                        break
                else:
                    th_val = th_arr[-1]
            thickness_interp[i] += weights[c] * th_val
    return thickness_interp

# =============================================
# ORIGINAL PYTHON FUNCTIONS (fallback if numba not available)
# =============================================
def compute_dimensionless_groups(params: Dict, t_real: float = None) -> Dict:
    """Compute dimensionless numbers that govern the phase-field evolution."""
    L0 = params.get('L0_nm', 20.0) * 1e-9
    D = params.get('D_nd', 0.05)
    gamma = params.get('gamma_nd', 0.02)
    M = params.get('M_nd', 0.2)
    k0 = params.get('k0_nd', 0.4)
    c_bulk = params.get('c_bulk', 1.0)
    fc = params.get('fc', params.get('core_radius_frac', 0.18))
    rs = params.get('rs', params.get('shell_thickness_frac', 0.2))
    tau0 = params.get('tau0_s', 1e-4)
    r_core = fc * L0
    
    if t_real is not None:
        t_nd = t_real / tau0
        Fo = D * t_nd / (L0**2)
    else:
        t_max_nd = params.get('t_max_nd', 1.0)
        Fo = D * t_max_nd / (L0**2)
    
    kappa_star = gamma * M / (L0 * k0 * c_bulk)
    Da = k0 * c_bulk * L0**2 / D
    r_star = r_core
    log_c = np.log(c_bulk + 1e-12)
    
    return {
        'Fo': Fo,
        'kappa_star': kappa_star,
        'Da': Da,
        'r_star': r_star,
        'log_c': log_c,
        'L0': L0,
        'tau0': tau0
    }

def physics_kernel(src_groups: Dict, tgt_groups: Dict, sigma: float = 0.3) -> float:
    """Gaussian kernel on dimensionless groups."""
    w_Fo = 1.0
    w_kappa = 0.5
    w_Da = 0.5
    w_r = 2.0
    w_logc = 1.0
    d2 = (w_Fo * (src_groups['Fo'] - tgt_groups['Fo'])**2 +
          w_kappa * (src_groups['kappa_star'] - tgt_groups['kappa_star'])**2 +
          w_Da * (src_groups['Da'] - tgt_groups['Da'])**2 +
          w_r * ((src_groups['r_star'] - tgt_groups['r_star']) / tgt_groups['L0'])**2 +
          w_logc * (src_groups['log_c'] - tgt_groups['log_c'])**2)
    return np.exp(-0.5 * d2 / sigma**2)

def radial_morph(source_fields: Dict[str, np.ndarray],
                 src_fc: float, src_L0_nm: float,
                 tgt_fc: float, tgt_L0_nm: float,
                 shape_out: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Warp a 2D source field to match target core radius and domain size.
    HARDENED VERSION: Forces Numba, ensures contiguous float32 arrays, no silent fallback.
    """
    src_phi = np.ascontiguousarray(source_fields['phi'].astype(np.float32))
    src_psi = np.ascontiguousarray(source_fields['psi'].astype(np.float32))
    src_c = np.ascontiguousarray(source_fields['c'].astype(np.float32))
    
    if NUMBA_AVAILABLE:
        try:
            warped_phi, warped_psi, warped_c = radial_morph_jit(
                src_phi, src_psi, src_c,
                src_fc, tgt_fc, shape_out
            )
            return {'phi': warped_phi, 'psi': warped_psi, 'c': warped_c}
        except Exception as e:
            st.error(f"Numba radial_morph_jit failed: {e}. Ensure arrays are contiguous float32.")
            st.stop()
    else:
        st.error("Numba is REQUIRED for acceptable performance. Install with: pip install numba")
        st.stop()
    return None

# =============================================
# POSITIONAL ENCODING (unchanged)
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
# UPDATED CoreShellInterpolator with physics kernel and radial morphing
# =============================================
class CoreShellInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                 param_sigma=None, temperature=1.0, gating_mode="PhysicsKernel"):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        if param_sigma is None:
            param_sigma = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
        self.param_sigma = param_sigma
        self.temperature = temperature
        self.gating_mode = gating_mode
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(16, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
    
    def set_parameter_sigma(self, sigma_list: List[float]):
        if not isinstance(sigma_list, list) or len(sigma_list) < 4:
            raise ValueError("sigma_list must be a list with at least 4 values")
        self.param_sigma = sigma_list
        return self
    
    def encode_parameters_physics(self, params_list: List[Dict], t_real: float = None) -> torch.Tensor:
        features = []
        for p in params_list:
            fc_norm = DepositionParameters.normalize(p.get('fc', 0.18), 'fc')
            rs_norm = DepositionParameters.normalize(p.get('rs', 0.2), 'rs')
            c_norm = DepositionParameters.normalize(p.get('c_bulk', 1.0), 'c_bulk')
            L0_norm = DepositionParameters.normalize(p.get('L0_nm', 20.0), 'L0_nm')
            groups = compute_dimensionless_groups(p, t_real)
            Fo_norm = np.clip(groups['Fo'] / 10.0, 0, 1)
            kappa_norm = np.clip(groups['kappa_star'] / 5.0, 0, 1)
            Da_norm = np.clip(groups['Da'] / 100.0, 0, 1)
            r_norm = fc_norm
            dirichlet = 1.0 if p.get('bc_type', 'Neu') == 'Dir' else 0.0
            edl = 1.0 if p.get('use_edl', False) else 0.0
            mode3d = 1.0 if p.get('mode', '2D (planar)') != '2D (planar)' else 0.0
            modelB = 1.0 if 'B' in p.get('growth_model', 'Model A') else 0.0
            feat = [fc_norm, rs_norm, c_norm, L0_norm,
                    Fo_norm, kappa_norm, Da_norm, r_norm,
                    dirichlet, edl, mode3d, modelB]
            while len(feat) < 16:
                feat.append(0.0)
            features.append(feat[:16])
        return torch.FloatTensor(features)
    
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
                          time_norm: Optional[float] = None):
        """Main interpolation routine – now with physics kernel + radial morphing."""
        if not sources:
            return None
        
        tgt_groups_char = compute_dimensionless_groups(target_params)
        warped_fields = []
        source_weights_phys = []
        source_params_list = []
        source_t_real_arrays = []
        thickness_curves = []
        t_max_vals = []
        
        for src in sources:
            src_params = src.get('params', {}).copy()
            src_params.setdefault('fc', src_params.get('core_radius_frac', 0.18))
            src_params.setdefault('rs', src_params.get('shell_thickness_frac', 0.2))
            src_params.setdefault('c_bulk', src_params.get('c_bulk', 1.0))
            src_params.setdefault('L0_nm', src_params.get('L0_nm', 20.0))
            src_params.setdefault('bc_type', src_params.get('bc_type', 'Neu'))
            src_params.setdefault('use_edl', src_params.get('use_edl', False))
            src_params.setdefault('mode', src_params.get('mode', '2D (planar)'))
            src_params.setdefault('growth_model', src_params.get('growth_model', 'Model A'))
            src_params.setdefault('tau0_s', src_params.get('tau0_s', 1e-4))
            
            src_groups_char = compute_dimensionless_groups(src_params)
            cat_mask = 1.0
            if src_params['bc_type'] != target_params.get('bc_type', 'Neu'):
                cat_mask *= 1e-8
            if src_params['use_edl'] != target_params.get('use_edl', False):
                cat_mask *= 1e-8
            if src_params['mode'] != target_params.get('mode', '2D (planar)'):
                cat_mask *= 1e-8
            
            w_phys = physics_kernel(src_groups_char, tgt_groups_char, sigma=0.3) * cat_mask
            
            history = src.get('history', [])
            if not history:
                continue
            final_snap = history[-1]
            phi_src = self._ensure_2d(final_snap['phi'])
            psi_src = self._ensure_2d(final_snap['psi'])
            c_src = self._ensure_2d(final_snap['c'])
            src_fields = {'phi': phi_src, 'psi': psi_src, 'c': c_src}
            
            warped = radial_morph(src_fields,
                                 src_params['fc'], src_params['L0_nm'],
                                 target_params['fc'], target_params['L0_nm'],
                                 target_shape)
            warped_fields.append(warped)
            source_weights_phys.append(w_phys)
            source_params_list.append(src_params)
            
            if 'history' in src and len(src['history']) > 1:
                t_real_vals = [snap['t_nd'] * src_params['tau0_s'] for snap in src['history']]
                source_t_real_arrays.append(t_real_vals)
                if t_real_vals[-1] > 0:
                    t_max_vals.append(t_real_vals[-1])
            else:
                source_t_real_arrays.append([0.0])
            
            thick_hist = src.get('thickness_history', [])
            if thick_hist:
                t_th = [th['t_nd'] * src_params['tau0_s'] for th in thick_hist]
                th_nm = [th['th_nm'] for th in thick_hist]
                thickness_curves.append((t_th, th_nm))
            else:
                thickness_curves.append(([0.0], [0.0]))
        
        if not warped_fields:
            st.error("No valid source fields after warping.")
            return None
        
        w_phys_arr = np.array(source_weights_phys)
        w_phys_soft = np.exp(w_phys_arr / self.temperature)
        w_phys_soft /= np.sum(w_phys_soft) + 1e-12
        
        src_feats = self.encode_parameters_physics(source_params_list)
        tgt_feats = self.encode_parameters_physics([target_params])
        all_feats = torch.cat([tgt_feats, src_feats], dim=0).unsqueeze(0)
        proj = self.input_proj(all_feats)
        proj = self.pos_encoder(proj)
        transformer_out = self.transformer(proj)
        tgt_rep = transformer_out[:, 0, :]
        src_reps = transformer_out[:, 1:, :]
        attn_scores = torch.matmul(tgt_rep.unsqueeze(1), src_reps.transpose(1,2)).squeeze(1)
        attn_scores = attn_scores / np.sqrt(self.d_model) / self.temperature
        attn_weights = torch.softmax(attn_scores, dim=-1).squeeze().detach().cpu().numpy()
        
        final_weights = w_phys_soft * attn_weights
        final_weights /= np.sum(final_weights) + 1e-12
        
        if time_norm is not None:
            if t_max_vals:
                avg_t_max = np.average(t_max_vals, weights=final_weights)
                t_target_real = time_norm * avg_t_max
            else:
                t_target_real = time_norm * target_params.get('tau0_s', 1e-4)
        else:
            t_target_real = None
        
        interp_fields = {'phi': np.zeros(target_shape),
                        'c': np.zeros(target_shape),
                        'psi': np.zeros(target_shape)}
        
        for i, warped in enumerate(warped_fields):
            if t_target_real is not None and len(source_t_real_arrays[i]) > 1:
                th_src_t = np.interp(t_target_real, thickness_curves[i][0], thickness_curves[i][1])
                th_src_final = thickness_curves[i][1][-1]
                scale_factor = th_src_t / (th_src_final + 1e-12)
                interp_fields['phi'] += final_weights[i] * warped['phi'] * scale_factor
                interp_fields['c']   += final_weights[i] * warped['c']
                interp_fields['psi'] += final_weights[i] * warped['psi']
            else:
                for key in interp_fields:
                    interp_fields[key] += final_weights[i] * warped[key]
        
        interp_fields['phi'] = gaussian_filter(interp_fields['phi'], sigma=1.0)
        interp_fields['c']   = gaussian_filter(interp_fields['c'], sigma=1.0)
        interp_fields['psi'] = gaussian_filter(interp_fields['psi'], sigma=1.0)
        
        common_t_norm = np.linspace(0, 1, n_time_points)
        if t_max_vals:
            avg_t_max = np.average(t_max_vals, weights=final_weights)
            common_t_real = common_t_norm * avg_t_max
        else:
            common_t_real = common_t_norm * target_params.get('tau0_s', 1e-4)
        
        if NUMBA_AVAILABLE and len(thickness_curves) > 0:
            try:
                max_curves = len(thickness_curves)
                max_points = max(len(tc[0]) for tc in thickness_curves)
                thickness_curves_t = np.zeros((max_curves, max_points))
                thickness_curves_th = np.zeros((max_curves, max_points))
                for i, (t_th, th_nm) in enumerate(thickness_curves):
                    thickness_curves_t[i, :len(t_th)] = t_th
                    thickness_curves_th[i, :len(th_nm)] = th_nm
                thickness_interp = interpolate_thickness_jit(
                    common_t_real, thickness_curves_t, thickness_curves_th,
                    final_weights, len(thickness_curves)
                )
            except:
                thickness_interp = np.zeros_like(common_t_norm)
                for i, (t_th, th_nm) in enumerate(thickness_curves):
                    th_on_grid = np.interp(common_t_real, t_th, th_nm, left=th_nm[0], right=th_nm[-1])
                    thickness_interp += final_weights[i] * th_on_grid
        else:
            thickness_interp = np.zeros_like(common_t_norm)
            for i, (t_th, th_nm) in enumerate(thickness_curves):
                th_on_grid = np.interp(common_t_real, t_th, th_nm, left=th_nm[0], right=th_nm[-1])
                thickness_interp += final_weights[i] * th_on_grid
        
        fc_target = target_params.get('fc', 0.18)
        L0 = target_params.get('L0_nm', 20.0) * 1e-9
        dx_nd = 1.0 / (target_shape[0] - 1)
        
        if NUMBA_AVAILABLE:
            try:
                material = compute_material_proxy_jit(interp_fields['phi'], interp_fields['psi'])
            except:
                material = DepositionPhysics.material_proxy(interp_fields['phi'], interp_fields['psi'])
        else:
            material = DepositionPhysics.material_proxy(interp_fields['phi'], interp_fields['psi'])
        
        alpha = target_params.get('alpha_nd', 2.0)
        potential = DepositionPhysics.potential_proxy(interp_fields['c'], alpha)
        thickness_nm = DepositionPhysics.shell_thickness(interp_fields['phi'],
                                                        interp_fields['psi'],
                                                        fc_target, dx=dx_nd) * L0 * 1e9
        stats = DepositionPhysics.phase_stats(interp_fields['phi'], interp_fields['psi'],
                                             dx_nd, dx_nd, L0)
        growth_rate = 0.0
        if t_target_real is not None and len(common_t_real) > 1:
            idx = int(time_norm * (len(common_t_real) - 1))
            if idx > 0:
                dt = common_t_real[idx] - common_t_real[idx-1]
                dth = thickness_interp[idx] - thickness_interp[idx-1]
                growth_rate = dth / dt if dt > 0 else 0.0
        
        result = {
            'fields': interp_fields,
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
                'physics_kernel': w_phys_soft.tolist(),
                'attention': attn_weights.tolist(),
                'entropy': float(-np.sum(final_weights * np.log(final_weights + 1e-12)))
            },
            'target_params': target_params,
            'shape': target_shape,
            'num_sources': len(warped_fields),
            'source_params': source_params_list,
            'time_norm': time_norm if time_norm is not None else 1.0,
            'time_real_s': t_target_real if t_target_real is not None else 0.0,
            'avg_tau0': float(np.average([p.get('tau0_s',1e-4) for p in source_params_list], weights=final_weights)),
            'avg_t_max_nd': float(avg_t_max / target_params.get('tau0_s',1e-4) if t_max_vals else 1.0)
        }
        return result

# =============================================
# MEMORY-EFFICIENT TEMPORAL CACHE SYSTEM (OPTIMIZED)
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


class TemporalFieldManager:
    """
    OPTIMIZED: Three-tier temporal management with pre-morph caching + self-similar scaling.
    Key innovation: Radial morph happens ONCE per source, not per key frame.
    """
    
    def __init__(self, interpolator, sources: List[Dict], target_params: Dict,
                 n_key_frames: int = 5, lru_size: int = 3, target_shape: Tuple[int, int] = (256, 256)):
        self.interpolator = interpolator
        self.sources = sources
        self.target_params = target_params
        self.target_shape = target_shape
        self.n_key_frames = n_key_frames
        self.lru_size = lru_size
        
        # Pre-morphed fields cache (source_idx → morphed final fields)
        self.morphed_finals: Dict[int, Dict[str, np.ndarray]] = {}
        self.source_final_thickness: Dict[int, float] = {}
        self.source_params_list: List[Dict] = []
        
        # Temporal data
        self.avg_tau0 = None
        self.avg_t_max_nd = None
        self.thickness_time: Optional[Dict] = None
        self.weights: Optional[Dict] = None
        self.key_times: np.ndarray = np.linspace(0, 1, n_key_frames)
        
        # Caches
        self.key_fields: Dict[float, Dict[str, np.ndarray]] = {}
        self.key_thickness: Dict[float, float] = {}
        self.key_time_real: Dict[float, float] = {}
        self.lru_cache: OrderedDict[float, TemporalCacheEntry] = OrderedDict()
        
        # Animation
        self.animation_temp_dir: Optional[str] = None
        self.animation_frame_paths: List[str] = []
        
        # Precompute everything efficiently
        self._initialize()
    
    def _initialize(self):
        """One-time setup: pre-morph sources + compute thickness curve + key frames."""
        st.info("Initializing optimized temporal interpolation...")
        self._pre_morph_all_sources()
        self._compute_thickness_curve()
        self._precompute_key_frames_fast()
        st.success(f"✓ Ready: {len(self.sources)} sources pre-morphed, {self.n_key_frames} key frames cached")
    
    def _pre_morph_all_sources(self):
        """Morph each source's final snapshot to target geometry — ONCE."""
        if not self.sources:
            return
        
        st.info(f"Pre-morphing {len(self.sources)} sources to target geometry...")
        progress = st.progress(0)
        
        for idx, src in enumerate(self.sources):
            src_params = src.get('params', {}).copy()
            for key, default in [('fc', 0.18), ('rs', 0.2), ('c_bulk', 1.0), 
                               ('L0_nm', 20.0), ('tau0_s', 1e-4)]:
                src_params.setdefault(key, default)
            
            history = src.get('history', [])
            if not history:
                continue
            final_snap = history[-1]
            
            phi_f = np.ascontiguousarray(final_snap['phi'].astype(np.float32))
            psi_f = np.ascontiguousarray(final_snap['psi'].astype(np.float32))
            c_f = np.ascontiguousarray(final_snap['c'].astype(np.float32))
            
            warped = radial_morph(
                {'phi': phi_f, 'psi': psi_f, 'c': c_f},
                src_params['fc'], src_params['L0_nm'],
                self.target_params['fc'], self.target_params['L0_nm'],
                self.target_shape
            )
            
            self.morphed_finals[idx] = warped
            self.source_params_list.append(src_params)
            
            fc_src = src_params['fc']
            rs_src = src_params['rs']
            L0_src = src_params['L0_nm']
            self.source_final_thickness[idx] = rs_src * fc_src * L0_src
            
            progress.progress((idx + 1) / len(self.sources))
        
        progress.empty()
    
    def _compute_thickness_curve(self):
        """Compute blended thickness curve using pre-morphed sources."""
        if not self.morphed_finals:
            return
        
        thickness_curves = []
        for src in self.sources:
            thick_hist = src.get('thickness_history', [])
            if thick_hist and len(thick_hist) > 1:
                t_real = [th['t_nd'] * src['params'].get('tau0_s', 1e-4) for th in thick_hist]
                th_nm = [th['th_nm'] for th in thick_hist]
                thickness_curves.append((np.array(t_real), np.array(th_nm)))
            else:
                idx = list(self.morphed_finals.keys())[0] if self.morphed_finals else 0
                th_final = self.source_final_thickness.get(idx, 1.0)
                thickness_curves.append((np.array([0.0, 1.0]), np.array([0.0, th_final])))
        
        tgt_groups = compute_dimensionless_groups(self.target_params)
        w_phys = []
        for src_params in self.source_params_list:
            src_groups = compute_dimensionless_groups(src_params)
            cat_mask = 1.0
            if src_params.get('bc_type') != self.target_params.get('bc_type'):
                cat_mask *= 1e-8
            if src_params.get('use_edl') != self.target_params.get('use_edl'):
                cat_mask *= 1e-8
            w_phys.append(physics_kernel(src_groups, tgt_groups, sigma=0.3) * cat_mask)
        
        w_phys_arr = np.array(w_phys)
        w_phys_soft = np.exp(w_phys_arr / self.interpolator.temperature)
        w_phys_soft /= np.sum(w_phys_soft) + 1e-12
        
        if self.source_params_list:
            src_feats = self.interpolator.encode_parameters_physics(self.source_params_list)
            tgt_feats = self.interpolator.encode_parameters_physics([self.target_params])
            all_feats = torch.cat([tgt_feats, src_feats], dim=0).unsqueeze(0)
            proj = self.interpolator.input_proj(all_feats)
            proj = self.interpolator.pos_encoder(proj)
            transformer_out = self.interpolator.transformer(proj)
            tgt_rep = transformer_out[:, 0, :]
            src_reps = transformer_out[:, 1:, :]
            attn_scores = torch.matmul(tgt_rep.unsqueeze(1), src_reps.transpose(1,2)).squeeze(1)
            attn_scores = attn_scores / np.sqrt(self.interpolator.d_model) / self.interpolator.temperature
            attn_weights = torch.softmax(attn_scores, dim=-1).squeeze().detach().cpu().numpy()
        else:
            attn_weights = np.ones(len(w_phys_soft)) / len(w_phys_soft)
        
        final_weights = w_phys_soft * attn_weights
        final_weights /= np.sum(final_weights) + 1e-12
        self.weights = {
            'combined': final_weights.tolist(),
            'physics_kernel': w_phys_soft.tolist(),
            'attention': attn_weights.tolist(),
            'entropy': float(-np.sum(final_weights * np.log(final_weights + 1e-12)))
        }
        
        n_points = 100
        common_t_norm = np.linspace(0, 1, n_points)
        
        tau0_vals = [p.get('tau0_s', 1e-4) for p in self.source_params_list]
        self.avg_tau0 = float(np.average(tau0_vals, weights=final_weights))
        
        t_max_vals = []
        for src, params in zip(self.sources, self.source_params_list):
            hist = src.get('history', [])
            if hist:
                t_max_vals.append(hist[-1]['t_nd'] * params.get('tau0_s', 1e-4))
        if t_max_vals:
            self.avg_t_max_nd = float(np.average(t_max_vals, weights=final_weights))
        else:
            self.avg_t_max_nd = 1.0
        
        common_t_real = common_t_norm * self.avg_t_max_nd * self.avg_tau0
        
        thickness_interp = np.zeros(n_points)
        for i, (t_src, th_src) in enumerate(thickness_curves):
            th_on_grid = np.interp(common_t_real, t_src, th_src, left=th_src[0], right=th_src[-1])
            thickness_interp += final_weights[i] * th_on_grid
        
        self.thickness_time = {
            't_norm': common_t_norm.tolist(),
            't_real_s': common_t_real.tolist(),
            'th_nm': thickness_interp.tolist()
        }
    
    def _precompute_key_frames_fast(self):
        """Precompute key frames using FAST path: blend pre-morphed finals + scale by thickness."""
        st.info(f"Pre-computing {self.n_key_frames} key frames (fast path)...")
        progress = st.progress(0)
        
        for i, t_norm in enumerate(self.key_times):
            fields = self._get_fields_fast(t_norm)
            self.key_fields[t_norm] = fields
            self.key_thickness[t_norm] = self.get_thickness_at_time(t_norm)
            self.key_time_real[t_norm] = self.get_time_real(t_norm)
            progress.progress((i + 1) / self.n_key_frames)
        
        progress.empty()
    
    def _get_fields_fast(self, time_norm: float) -> Dict[str, np.ndarray]:
        """
        FAST PATH: Blend pre-morphed finals + radial scale by thickness ratio.
        Avoids re-morphing entirely.
        """
        phi_b = np.zeros(self.target_shape, dtype=np.float32)
        psi_b = np.zeros(self.target_shape, dtype=np.float32)
        c_b = np.zeros(self.target_shape, dtype=np.float32)
        
        weights = np.array(self.weights['combined']) if self.weights else None
        if weights is None or len(weights) != len(self.morphed_finals):
            weights = np.ones(len(self.morphed_finals)) / len(self.morphed_finals)
        
        for idx, morphed in self.morphed_finals.items():
            w = weights[idx] if idx < len(weights) else 1.0 / len(self.morphed_finals)
            phi_b += w * morphed['phi']
            psi_b += w * morphed['psi']
            c_b += w * morphed['c']
        
        h_current = self.get_thickness_at_time(time_norm)
        fc_tgt = self.target_params['fc']
        rs_tgt = self.target_params['rs']
        L0_tgt = self.target_params['L0_nm']
        h_target_final = rs_tgt * fc_tgt * L0_tgt
        
        scale = h_current / (h_target_final + 1e-9)
        scale = np.clip(scale, 0.1, 3.0)
        
        if NUMBA_AVAILABLE and abs(scale - 1.0) > 1e-3:
            try:
                phi_scaled = radial_scale_field_jit(phi_b, (0.5, 0.5), scale, self.target_shape)
            except:
                phi_scaled = phi_b
        else:
            phi_scaled = phi_b
        
        phi_scaled = gaussian_filter(phi_scaled, sigma=0.5)
        c_b = gaussian_filter(c_b, sigma=0.5)
        
        return {'phi': phi_scaled, 'psi': psi_b, 'c': c_b}
    
    def get_fields(self, time_norm: float, use_interpolation: bool = True) -> Dict[str, np.ndarray]:
        """Get fields at given normalized time with LRU caching."""
        t_key = round(time_norm, 4)
        time_real = self.get_time_real(time_norm)
        
        if t_key in self.lru_cache:
            entry = self.lru_cache.pop(t_key)
            self.lru_cache[t_key] = entry
            return entry.fields
        
        if t_key in self.key_fields:
            fields = self.key_fields[t_key]
            self._add_to_lru(t_key, fields, self.key_thickness.get(t_key, 0.0), time_real)
            return fields
        
        if use_interpolation and self.key_fields:
            key_times_arr = np.array(sorted(self.key_fields.keys()))
            idx = np.searchsorted(key_times_arr, t_key)
            
            if idx == 0:
                fields = self.key_fields[key_times_arr[0]]
            elif idx >= len(key_times_arr):
                fields = self.key_fields[key_times_arr[-1]]
            else:
                t0, t1 = key_times_arr[idx-1], key_times_arr[idx]
                alpha = (t_key - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
                f0, f1 = self.key_fields[t0], self.key_fields[t1]
                
                fields = {}
                for key in f0:
                    fields[key] = (1 - alpha) * f0[key] + alpha * f1[key]
            
            th_interp = self.get_thickness_at_time(time_norm)
            self._add_to_lru(t_key, fields, th_interp, time_real)
            return fields
        
        fields = self._get_fields_fast(time_norm)
        self._add_to_lru(t_key, fields, self.get_thickness_at_time(time_norm), time_real)
        return fields
    
    def _add_to_lru(self, time_norm: float, fields: Dict[str, np.ndarray], 
                    thickness_nm: float, time_real_s: float):
        """Add entry to LRU cache with size management."""
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
        """Interpolate thickness from precomputed curve."""
        if self.thickness_time is None:
            return 0.0
        t_arr = np.array(self.thickness_time['t_norm'])
        th_arr = np.array(self.thickness_time['th_nm'])
        return float(np.interp(time_norm, t_arr, th_arr))
    
    def get_time_real(self, time_norm: float) -> float:
        """Convert normalized time to real physical time."""
        return time_norm * self.avg_t_max_nd * self.avg_tau0 if self.avg_t_max_nd else 0.0
    
    def prepare_animation_streaming(self, n_frames: int = 50) -> List[str]:
        """Pre-render animation frames to disk using fast path."""
        import tempfile
        self.animation_temp_dir = tempfile.mkdtemp(dir=TEMP_ANIMATION_DIR)
        self.animation_frame_paths = []
        times = np.linspace(0, 1, n_frames)
        
        st.info(f"Pre-rendering {n_frames} animation frames (fast path)...")
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
        """Load pre-rendered frame from disk."""
        if not self.animation_frame_paths or frame_idx >= len(self.animation_frame_paths):
            return None
        data = np.load(self.animation_frame_paths[frame_idx])
        return {
            'phi': data['phi'], 'c': data['c'], 'psi': data['psi'],
            'time_norm': float(data['time_norm']),
            'time_real_s': float(data['time_real_s'])
        }
    
    def cleanup_animation(self):
        """Clean up temporary animation files."""
        if self.animation_temp_dir and os.path.exists(self.animation_temp_dir):
            shutil.rmtree(self.animation_temp_dir)
        self.animation_temp_dir = None
        self.animation_frame_paths = []
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Report memory usage."""
        lru_memory = sum(entry.get_size_mb() for entry in self.lru_cache.values())
        
        key_memory = 0.0
        if self.key_fields:
            sample = next(iter(self.key_fields.values()))
            bytes_per_frame = sum(arr.nbytes for arr in sample.values())
            key_memory = (bytes_per_frame * len(self.key_fields)) / (1024 * 1024)
        
        morph_memory = 0.0
        if self.morphed_finals:
            sample = next(iter(self.morphed_finals.values()))
            bytes_per_source = sum(arr.nbytes for arr in sample.values())
            morph_memory = (bytes_per_source * len(self.morphed_finals)) / (1024 * 1024)
        
        return {
            'lru_cache_mb': lru_memory,
            'key_frames_mb': key_memory,
            'pre_morphed_mb': morph_memory,
            'total_mb': lru_memory + key_memory + morph_memory,
            'lru_entries': len(self.lru_cache),
            'key_frame_entries': len(self.key_fields),
            'pre_morphed_sources': len(self.morphed_finals)
        }

# =============================================
# ROBUST SOLUTION LOADER (unchanged)
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
# ENHANCED HEATMAP VISUALIZER (unchanged)
# =============================================
class HeatMapVisualizer:
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
    
    def _get_extent(self, L0_nm):
        return [0, L0_nm, 0, L0_nm]
    
    def _is_material_proxy(self, field_data, colorbar_label, title):
        unique_vals = np.unique(field_data)
        is_discrete = np.all(np.isin(unique_vals, [0, 1, 2])) and len(unique_vals) <= 3
        has_material_keyword = any(kw in colorbar_label.lower() or kw in title.lower()
                                  for kw in ['material', 'proxy', 'phase', 'electrolyte', 'ag', 'cu'])
        return is_discrete and has_material_keyword
    
    def create_field_heatmap(self, field_data, title, cmap_name='viridis',
                           L0_nm=20.0, figsize=(10,8), colorbar_label="",
                           vmin=None, vmax=None, target_params=None, time_real_s=None):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        extent = self._get_extent(L0_nm)
        is_material = self._is_material_proxy(field_data, colorbar_label, title)
        if is_material:
            im = ax.imshow(field_data, cmap=MATERIAL_COLORMAP_MATPLOTLIB, norm=MATERIAL_BOUNDARY_NORM,
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
            fc = target_params.get('fc', 0); rs = target_params.get('rs', 0); cb = target_params.get('c_bulk', 0)
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
            fig = go.Figure(data=go.Heatmap(
                z=field_data, x=x, y=y, colorscale=cmap_name,
                hoverinfo='text', text=hover,
                colorbar=dict(title=dict(text="Value", font=dict(size=14)))
            ))
        title_str = title
        if target_params:
            fc = target_params.get('fc', 0); rs = target_params.get('rs', 0); cb = target_params.get('c_bulk', 0)
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
                ax.plot(src_t, src_th, '--', linewidth=1, alpha=alpha, label=f'Source {i+1} (w={weights[i]:.3f})')
        if current_time_norm is not None:
            if 't_real_s' in thickness_time:
                current_th = np.interp(current_time_norm, np.array(thickness_time['t_norm']), th_nm)
                current_t_plot = np.interp(current_time_norm, np.array(thickness_time['t_norm']), t_plot)
            else:
                current_t_plot = current_time_norm
                current_th = np.interp(current_time_norm, np.array(thickness_time['t_norm']), th_nm)
            ax.axvline(current_t_plot, color='r', linestyle='--', linewidth=2, alpha=0.7)
            ax.plot(current_t_plot, current_th, 'ro', markersize=8, label=f'Current: t={current_t_plot:.2e}, h={current_th:.2f} nm')
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
                im = ax.imshow(fields[field_key], cmap=cmap, norm=norm, extent=extent, aspect='equal', origin='lower')
            else:
                im = ax.imshow(fields[field_key], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect='equal', origin='lower')
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
# RESULTS MANAGER (unchanged)
# =============================================
class ResultsManager:
    def __init__(self):
        pass
    
    def prepare_export_data(self, interpolation_result, visualization_params):
        res = interpolation_result.copy()
        export = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'interpolation_method': 'core_shell_temporal_transformer_physics',
                'visualization_params': visualization_params
            },
            'result': {
                'target_params': res['target_params'],
                'shape': res['shape'],
                'num_sources': res['num_sources'],
                'weights': res['weights'],
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
            fc = p.get('fc', 0); rs = p.get('rs', 0); cb = p.get('c_bulk', 0); t = export_data['result'].get('time_real_s', 0)
            filename = f"temporal_interp_fc{fc:.3f}_rs{rs:.3f}_c{cb:.2f}_t{t:.3e}s_{ts}.json"
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
    def export_to_csv(self, interpolation_result, filename=None):
        if filename is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            p = interpolation_result['target_params']
            fc = p.get('fc', 0); rs = p.get('rs', 0); cb = p.get('c_bulk', 0); t = interpolation_result.get('time_real_s', 0)
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
# MAIN STREAMLIT APP (updated with Numba status + lower default key frames)
# =============================================
def main():
    st.set_page_config(page_title="Core‑Shell Deposition: Full Temporal Interpolation",
                      layout="wide", page_icon="🧪", initial_sidebar_state="expanded")
    
    # Display Numba optimization status
    if NUMBA_AVAILABLE:
        import numba
        st.sidebar.success(f"✅ Numba JIT: v{numba.__version__} | Acceleration active")
    else:
        st.sidebar.error("⚠️ Numba NOT installed — app will be VERY slow. Run: pip install numba")
    
    st.markdown("""
    <style>
    .main-header { font-size: 3.2rem; color: #1E3A8A; text-align: center; padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; font-weight: 900; margin-bottom: 1rem; }
    .section-header { font-size: 2.0rem; color: #374151; font-weight: 800;
        border-left: 6px solid #3B82F6; padding-left: 1.2rem; margin-top: 1.8rem; margin-bottom: 1.2rem; }
    .info-box { background-color: #F0F9FF; border-left: 5px solid #3B82F6; padding: 1.2rem;
        border-radius: 0.6rem; margin: 1.2rem 0; font-size: 1.1rem; }
    .memory-stats { background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 1.0rem;
        border-radius: 0.4rem; margin: 0.8rem 0; font-size: 0.9rem; }
    .color-legend { display: flex; gap: 1rem; margin: 0.5rem 0; }
    .color-item { display: flex; align-items: center; gap: 0.3rem; font-size: 0.9rem; }
    .color-box { width: 20px; height: 20px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧪 Core‑Shell Deposition: Full Temporal Interpolation</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#f8f9fa; padding:0.8rem; border-radius:0.4rem; margin:1rem 0;">
    <strong>Material Proxy Colors (max(φ,ψ)+ψ):</strong>
    <div class="color-legend">
    <div class="color-item"><div class="color-box" style="background:rgb(228,26,28)"></div>Electrolyte (φ≤0.5, ψ≤0.5)</div>
    <div class="color-item"><div class="color-box" style="background:rgb(255,127,0)"></div>Ag shell (φ>0.5, ψ≤0.5)</div>
    <div class="color-item"><div class="color-box" style="background:rgb(153,153,153)"></div>Cu core (ψ>0.5)</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = CoreShellInterpolator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = HeatMapVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    if 'temporal_manager' not in st.session_state:
        st.session_state.temporal_manager = None
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 1.0
    if 'last_target_hash' not in st.session_state:
        st.session_state.last_target_hash = None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        st.markdown("#### 📁 Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Solutions", use_container_width=True):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
        with col2:
            if st.button("🧹 Clear All", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.temporal_manager = None
                st.session_state.last_target_hash = None
                if os.path.exists(TEMP_ANIMATION_DIR):
                    shutil.rmtree(TEMP_ANIMATION_DIR)
                os.makedirs(TEMP_ANIMATION_DIR, exist_ok=True)
                st.success("All cleared")
        st.divider()
        
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        fc = st.slider("Core / L (fc)", 0.05, 0.45, 0.18, 0.01)
        rs = st.slider("Δr / r_core (rs)", 0.01, 0.6, 0.2, 0.01)
        c_bulk = st.slider("c_bulk (C_Ag / C_Cu)", 0.1, 1.0, 0.5, 0.05)
        L0_nm = st.number_input("Domain length L0 (nm)", 10.0, 100.0, 60.0, 5.0)
        bc_type = st.selectbox("BC type", ["Neu", "Dir"], index=0)
        use_edl = st.checkbox("Use EDL catalyst", value=False)
        mode = st.selectbox("Mode", ["2D (planar)", "3D (spherical)"], index=0)
        growth_model = st.selectbox("Growth model", ["Model A", "Model B"], index=0)
        alpha_nd = st.slider("α (coupling)", 0.0, 10.0, 2.0, 0.1)
        tau0_input = st.number_input("τ₀ (×10⁻⁴ s)", 1e-6, 1e6, 1.0) * 1e-4
        tau0_target = tau0_input
        st.divider()
        
        st.markdown('<h2 class="section-header">⚛️ Interpolation Settings</h2>', unsafe_allow_html=True)
        sigma_fc = st.slider("Kernel σ (fc)", 0.05, 0.3, 0.15, 0.01)
        sigma_rs = st.slider("Kernel σ (rs)", 0.05, 0.3, 0.15, 0.01)
        sigma_c = st.slider("Kernel σ (c_bulk)", 0.05, 0.3, 0.15, 0.01)
        sigma_L = st.slider("Kernel σ (L0_nm)", 0.05, 0.3, 0.15, 0.01)
        temperature = st.slider("Attention temperature", 0.1, 10.0, 1.0, 0.1)
        gating_mode = st.selectbox(
            "Composite Gating Mode",
            ["PhysicsKernel (recommended)", "Hierarchical", "Joint Multiplicative", "No Gating"],
            index=0,
            help="PhysicsKernel uses dimensionless groups; others are legacy."
        )
        # Lower default key frames for faster setup
        n_key_frames = st.slider("Key frames for temporal interpolation", 
                                1, 10, 5, 1,  # Changed max from 20 → 10, default 5
                                help="More frames = smoother animation but more memory")
        lru_cache_size = st.slider("Interactive cache size", 1, 5, 3, 1,
                                  help="Frames to keep in memory for slider responsiveness")
        
        # Compute target hash
        target = {
            'fc': fc, 'rs': rs, 'c_bulk': c_bulk, 'L0_nm': L0_nm,
            'bc_type': bc_type, 'use_edl': use_edl, 'mode': mode,
            'growth_model': growth_model, 'alpha_nd': alpha_nd,
            'tau0_s': tau0_target
        }
        target_hash = hashlib.md5(json.dumps(target, sort_keys=True).encode()).hexdigest()[:16]
        
        # Initialize temporal manager
        if target_hash != st.session_state.last_target_hash:
            if st.button("🧠 Initialize Temporal Interpolation", type="primary", use_container_width=True):
                if not st.session_state.solutions:
                    st.error("Please load solutions first!")
                else:
                    with st.spinner("Setting up temporal interpolation..."):
                        st.session_state.interpolator.set_parameter_sigma([sigma_fc, sigma_rs, sigma_c, sigma_L])
                        st.session_state.interpolator.temperature = temperature
                        st.session_state.interpolator.gating_mode = gating_mode
                        st.session_state.temporal_manager = TemporalFieldManager(
                            st.session_state.interpolator,
                            st.session_state.solutions,
                            target,
                            n_key_frames=n_key_frames,
                            lru_size=lru_cache_size
                        )
                        st.session_state.last_target_hash = target_hash
                        st.session_state.current_time = 1.0
                        st.success("Temporal interpolation ready!")
        
        # Memory stats display
        if st.session_state.temporal_manager:
            with st.expander("💾 Memory Statistics"):
                stats = st.session_state.temporal_manager.get_memory_stats()
                st.markdown(f"""
                <div class="memory-stats">
                <strong>Memory Usage:</strong><br>
                • Key frames: {stats['key_frame_entries']} frames ({stats['key_frames_mb']:.1f} MB)<br>
                • LRU cache: {stats['lru_entries']} frames ({stats['lru_cache_mb']:.1f} MB)<br>
                • Pre-morphed sources: {stats['pre_morphed_sources']} ({stats['pre_morphed_mb']:.1f} MB)<br>
                • <strong>Total: {stats['total_mb']:.1f} MB</strong>
                </div>
                """, unsafe_allow_html=True)
    
    # Main area
    if st.session_state.temporal_manager:
        mgr = st.session_state.temporal_manager
        st.markdown('<h2 class="section-header">⏱️ Temporal Control</h2>', unsafe_allow_html=True)
        col_time1, col_time2, col_time3 = st.columns([3, 1, 1])
        with col_time1:
            current_time_norm = st.slider("Normalized Time (0=start, 1=end)",
                                         0.0, 1.0,
                                         value=st.session_state.current_time,
                                         step=0.001,
                                         format="%.3f")
            st.session_state.current_time = current_time_norm
        with col_time2:
            if st.button("⏮️ Start", use_container_width=True):
                st.session_state.current_time = 0.0
                st.rerun()
        with col_time3:
            if st.button("⏭️ End", use_container_width=True):
                st.session_state.current_time = 1.0
                st.rerun()
        
        current_time_real = mgr.get_time_real(current_time_norm)
        current_thickness = mgr.get_thickness_at_time(current_time_norm)
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Current Thickness", f"{current_thickness:.3f} nm")
        with col_info2:
            st.empty()
        with col_info3:
            st.metric("Time", f"{current_time_real:.3e} s")
        
        tabs = st.tabs(["📊 Field Visualization", "📈 Thickness Evolution",
                       "🎬 Animation", "🧪 Derived Quantities", "⚖️ Weights", "💾 Export"])
        
        with tabs[0]:
            st.markdown('<h2 class="section-header">📊 Field Visualization</h2>', unsafe_allow_html=True)
            fields = mgr.get_fields(current_time_norm, use_interpolation=True)
            field_choice = st.selectbox("Select field",
                                       ['c (concentration)', 'phi (shell)', 'psi (core)', 'material proxy'],
                                       key='field_choice')
            field_map = {'c (concentration)': 'c', 'phi (shell)': 'phi', 'psi (core)': 'psi', 'material proxy': 'material'}
            field_key = field_map[field_choice]
            if field_key == 'material':
                field_data = DepositionPhysics.material_proxy(fields['phi'], fields['psi'])
            else:
                field_data = fields[field_key]
            cmap_cat = st.selectbox("Colormap category", list(COLORMAP_OPTIONS.keys()), index=0)
            cmap = st.selectbox("Colormap", COLORMAP_OPTIONS[cmap_cat], index=0)
            fig = st.session_state.visualizer.create_field_heatmap(
                field_data,
                title=f"Interpolated {field_choice}",
                cmap_name=cmap,
                L0_nm=L0_nm,
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
                    L0_nm=L0_nm,
                    target_params=target,
                    time_real_s=current_time_real
                )
                st.plotly_chart(fig_inter, use_container_width=True)
            if st.checkbox("Show temporal evolution grid"):
                n_compare = st.slider("Number of time points", 3, 9, 5)
                times_compare_norm = np.linspace(0, 1, n_compare)
                times_compare_real = [mgr.get_time_real(t) for t in times_compare_norm]
                fields_list = []
                for t in times_compare_norm:
                    f = mgr.get_fields(t, use_interpolation=True)
                    if field_key == 'material':
                        f['material'] = DepositionPhysics.material_proxy(f['phi'], f['psi'])
                    fields_list.append(f)
                fig_grid = st.session_state.visualizer.create_temporal_comparison_plot(
                    fields_list, times_compare_real, field_key='material' if field_key == 'material' else field_key,
                    cmap_name=cmap, L0_nm=L0_nm
                )
                st.pyplot(fig_grid)
        
        with tabs[1]:
            st.markdown('<h2 class="section-header">📈 Thickness Evolution</h2>', unsafe_allow_html=True)
            thickness_time = mgr.thickness_time
            show_growth = st.checkbox("Show growth rate", value=False)
            fig_th = st.session_state.visualizer.create_thickness_plot(
                thickness_time,
                title=f"Shell Thickness Evolution (fc={fc:.3f}, rs={rs:.3f}, c_bulk={c_bulk:.2f})",
                current_time_norm=current_time_norm,
                current_time_real=current_time_real,
                show_growth_rate=show_growth
            )
            st.pyplot(fig_th)
            st.markdown("#### Thickness Statistics")
            th_arr = np.array(thickness_time['th_nm'])
            t_arr = np.array(thickness_time['t_real_s']) if 't_real_s' in thickness_time else np.array(thickness_time['t_norm'])
            stats_cols = st.columns(4)
            with stats_cols[0]:
                st.metric("Final Thickness", f"{th_arr[-1]:.3f} nm")
            with stats_cols[1]:
                st.metric("Initial Growth Rate", f"{(th_arr[1]-th_arr[0])/(t_arr[1]-t_arr[0]):.3f} nm/s")
            with stats_cols[2]:
                avg_rate = (th_arr[-1] - th_arr[0]) / (t_arr[-1] - t_arr[0])
                st.metric("Avg Growth Rate", f"{avg_rate:.3f} nm/s")
            with stats_cols[3]:
                idx_50 = np.argmin(np.abs(th_arr - 0.5*th_arr[-1]))
                st.metric("Time to 50% thickness", f"{t_arr[idx_50]:.3e} s")
        
        with tabs[2]:
            st.markdown('<h2 class="section-header">🎬 Animation</h2>', unsafe_allow_html=True)
            anim_method = st.radio("Animation method",
                                  ["Real-time interpolation", "Pre-rendered (smooth)"],
                                  help="Real-time: compute on fly, lower memory. Pre-rendered: smoother but uses disk.")
            if anim_method == "Real-time interpolation":
                fps = st.slider("FPS", 1, 30, 10)
                n_frames = st.slider("Frames", 10, 100, 30)
                if st.button("▶️ Play Animation", use_container_width=True):
                    placeholder = st.empty()
                    times = np.linspace(0, 1, n_frames)
                    for t_norm in times:
                        fields = mgr.get_fields(t_norm, use_interpolation=True)
                        t_real = mgr.get_time_real(t_norm)
                        field_data = DepositionPhysics.material_proxy(fields['phi'], fields['psi']) if field_key == 'material' else fields[field_key]
                        fig = st.session_state.visualizer.create_field_heatmap(
                            field_data,
                            title=f"t = {t_real:.3e} s",
                            cmap_name=cmap,
                            L0_nm=L0_nm,
                            target_params=target,
                            time_real_s=t_real,
                            colorbar_label="Material" if field_key == 'material' else field_choice.split()[0]
                        )
                        placeholder.pyplot(fig)
                        time.sleep(1/fps)
                    st.success("Animation complete")
            else:  # Pre-rendered
                n_frames = st.slider("Pre-render frames", 20, 100, 50)
                if st.button("🎥 Pre-render Animation", use_container_width=True):
                    with st.spinner(f"Rendering {n_frames} frames to disk..."):
                        frame_paths = mgr.prepare_animation_streaming(n_frames)
                        st.success(f"Pre-rendered {len(frame_paths)} frames")
                if mgr.animation_frame_paths:
                    fps = st.slider("Playback FPS", 1, 30, 15)
                    if st.button("▶️ Play Pre-rendered", use_container_width=True):
                        placeholder = st.empty()
                        for i, frame_path in enumerate(mgr.animation_frame_paths):
                            data = np.load(frame_path)
                            fields = {'phi': data['phi'], 'c': data['c'], 'psi': data['psi']}
                            t_real = float(data['time_real_s'])
                            field_data = DepositionPhysics.material_proxy(fields['phi'], fields['psi']) if field_key == 'material' else fields[field_key]
                            fig = st.session_state.visualizer.create_field_heatmap(
                                field_data,
                                title=f"t = {t_real:.3e} s [Pre-rendered]",
                                cmap_name=cmap,
                                L0_nm=L0_nm,
                                target_params=target,
                                time_real_s=t_real,
                                colorbar_label="Material" if field_key == 'material' else field_choice.split()[0]
                            )
                            placeholder.pyplot(fig)
                            time.sleep(1/fps)
                        st.success("Playback complete")
                    if st.button("🗑️ Clean Pre-rendered", use_container_width=True):
                        mgr.cleanup_animation()
                        st.success("Cleaned up")
        
        with tabs[3]:
            st.markdown('<h2 class="section-header">🧪 Derived Quantities</h2>', unsafe_allow_html=True)
            res = st.session_state.interpolator.interpolate_fields(
                st.session_state.solutions, target, target_shape=(256,256),
                n_time_points=100, time_norm=current_time_norm
            )
            if res:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Shell thickness (nm)", f"{res['derived']['thickness_nm']:.3f}")
                with col2:
                    st.metric("Growth rate (nm/s)", f"{res['derived'].get('growth_rate', 0):.3f}")
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
                    st.markdown("*Threshold logic: Electrolyte=red (φ≤0.5,ψ≤0.5) | Ag=orange (φ>0.5,ψ≤0.5) | Cu=gray (ψ>0.5)*")
                    fig_mat = st.session_state.visualizer.create_field_heatmap(
                        res['derived']['material'], "Material Proxy",
                        cmap_name='Set1', L0_nm=L0_nm, target_params=target,
                        colorbar_label="Material", vmin=0, vmax=2,
                        time_real_s=current_time_real
                    )
                    st.pyplot(fig_mat)
                with col_viz2:
                    st.subheader("Potential Proxy")
                    fig_pot = st.session_state.visualizer.create_field_heatmap(
                        res['derived']['potential'], "Potential Proxy",
                        cmap_name='RdBu_r', L0_nm=L0_nm, target_params=target,
                        colorbar_label="-α·c",
                        time_real_s=current_time_real
                    )
                    st.pyplot(fig_pot)
        
        with tabs[4]:
            st.markdown('<h2 class="section-header">⚖️ Weights & Uncertainty</h2>', unsafe_allow_html=True)
            weights = mgr.weights
            df_weights = pd.DataFrame({
                'Source': range(len(weights['combined'])),
                'Combined': weights['combined'],
                'Physics Kernel': weights['physics_kernel'],
                'Attention': weights['attention']
            })
            st.dataframe(df_weights.style.format("{:.4f}"))
            entropy = weights.get('entropy', 0.0)
            st.metric("Weight Entropy (Uncertainty)", f"{entropy:.4f}",
                     help="Higher = more uncertain (sources contribute equally)")
            fig_w, ax = plt.subplots(figsize=(10,5))
            x = np.arange(len(weights['combined']))
            width = 0.25
            ax.bar(x - width, weights['physics_kernel'], width, label='Physics Kernel', alpha=0.7)
            ax.bar(x, weights['attention'], width, label='Attention', alpha=0.7)
            ax.bar(x + width, weights['combined'], width, label='Combined', alpha=0.7)
            ax.set_xlabel('Source Index')
            ax.set_ylabel('Weight')
            ax.set_title('Interpolation Weights')
            ax.legend()
            st.pyplot(fig_w)
        
        with tabs[5]:
            st.markdown('<h2 class="section-header">💾 Export Data</h2>', unsafe_allow_html=True)
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📊 Export Current State (JSON)", use_container_width=True):
                    res = st.session_state.interpolator.interpolate_fields(
                        st.session_state.solutions, target, target_shape=(256,256),
                        n_time_points=100, time_norm=current_time_norm
                    )
                    if res:
                        export_data = st.session_state.results_manager.prepare_export_data(
                            res, {'cmap': cmap, 'field': field_key, 'time_norm': current_time_norm, 'time_real_s': current_time_real}
                        )
                        json_str, fname = st.session_state.results_manager.export_to_json(export_data)
                        st.download_button("⬇️ Download JSON", json_str, fname, "application/json")
            with col_exp2:
                if st.button("📈 Export Current State (CSV)", use_container_width=True):
                    res = st.session_state.interpolator.interpolate_fields(
                        st.session_state.solutions, target, target_shape=(256,256),
                        n_time_points=100, time_norm=current_time_norm
                    )
                    if res:
                        csv_str, fname = st.session_state.results_manager.export_to_csv(res)
                        st.download_button("⬇️ Download CSV", csv_str, fname, "text/csv")
            st.markdown("#### Full Temporal Export")
            if st.button("📦 Export All Key Frames (ZIP)", use_container_width=True):
                import zipfile
                import io
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for t_norm in mgr.key_times:
                        res_t = st.session_state.interpolator.interpolate_fields(
                            st.session_state.solutions, target, target_shape=(256,256),
                            n_time_points=100, time_norm=t_norm
                        )
                        if res_t:
                            export_data = st.session_state.results_manager.prepare_export_data(
                                res_t, {'time_norm': t_norm, 'time_real_s': res_t.get('time_real_s',0)}
                            )
                            json_str, _ = st.session_state.results_manager.export_to_json(export_data)
                            zip_file.writestr(f"frame_t{t_norm:.4f}.json", json_str)
                st.download_button("⬇️ Download ZIP",
                                  zip_buffer.getvalue(),
                                  f"temporal_sequence_{target_hash}.zip",
                                  "application/zip")
    else:
        st.info("""
        👈 **Get Started:**
        1. Load solutions using the sidebar
        2. Set target parameters
        3. Click **"Initialize Temporal Interpolation"**
        The system will pre-compute key frames for smooth temporal exploration while keeping memory usage low (~15-20 MB).
        """)

if __name__ == "__main__":
    main()
