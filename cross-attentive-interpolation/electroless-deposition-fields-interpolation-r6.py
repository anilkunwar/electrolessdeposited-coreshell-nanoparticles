#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer‑Inspired Interpolation for Electroless Ag Shell Deposition on Cu Core
FULLY EXPANDED VERSION WITH PARAMETER-AWARE TEMPORAL ATTENTION
-------------------------------------------------------------------------------
Enhancements in this version:
- Parameter-conditioned temporal attention (thickness evolution depends on params)
- Physics-informed time warping for sequence alignment
- Time-dependent parameter kernel modulation
- Uncertainty quantification across sources
- Patch-based spatial attention (optional)
- Full 3D support via configurable slicing
- All features optional and backward-compatible
- **NEW**: Temporal proximity kernel (time‑distance weighting)
- **NEW**: Numba acceleration for performance-critical loops
- **NEW**: Memory optimizations (float32, lazy loading)
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import warnings
import json
import re
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import interp1d
from typing import List, Dict, Any, Optional, Tuple, Union
import time
from einops import rearrange, repeat

# =============================================
# OPTIONAL NUMBA IMPORT (for acceleration)
# =============================================
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # dummy decorator for fallback
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    prange = range

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
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

COLORMAP_OPTIONS = {
    'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'hot'],
    'Diverging': ['RdBu', 'RdYlBu', 'Spectral', 'coolwarm', 'bwr', 'seismic'],
    'Qualitative': ['tab10', 'tab20', 'Set1', 'Set2', 'Set3'],
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu']
}

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
        return (value - low) / (high - low)
    
    @staticmethod
    def denormalize(norm_value: float, param_name: str) -> float:
        low, high = DepositionParameters.RANGES[param_name]
        return norm_value * (high - low) + low
    
    @staticmethod
    def get_reference_values() -> Dict[str, float]:
        """Reference values for time warping calculations."""
        return {
            'c_bulk': 0.5,
            'L0_nm': 60.0,
            'fc': 0.18,
            'rs': 0.2
        }

# =============================================
# DEPOSITION PHYSICS (derived quantities)
# =============================================
class DepositionPhysics:
    """Computes derived quantities and physics‑informed constraints."""
    
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
        cell_area_real = cell_area_nd * (L0**2)
        electrolyte_area_nd = np.sum(electrolyte_mask) * cell_area_nd
        ag_area_nd = np.sum(ag_mask) * cell_area_nd
        cu_area_nd = np.sum(cu_mask) * cell_area_nd
        return {
            "Electrolyte": (electrolyte_area_nd, electrolyte_area_nd * (L0**2)),
            "Ag": (ag_area_nd, ag_area_nd * (L0**2)),
            "Cu": (cu_area_nd, cu_area_nd * (L0**2))
        }
    
    @staticmethod
    def enforce_phase_constraints(phi, psi, threshold=0.5, interface_width=2.0):
        """Physics‑informed post‑processing."""
        phi = np.clip(phi, 0, 1)
        psi = np.clip(psi, 0, 1)
        overlap = (phi > threshold) & (psi > threshold)
        if np.any(overlap):
            phi[overlap & (phi >= psi)] = 1.0
            psi[overlap & (phi >= psi)] = 0.0
            psi[overlap & (psi > phi)] = 1.0
            phi[overlap & (psi > phi)] = 0.0
        return phi, psi
    
    @staticmethod
    def compute_time_warp_factor(params: Dict[str, float]) -> float:
        """
        Compute physics-informed time warping factor based on parameters.
        
        Physical basis:
        - Higher concentration → faster reaction → warp time forward
        - Larger domain → longer diffusion → warp time backward
        - Larger core → more surface area → warp time forward
        """
        ref = DepositionParameters.get_reference_values()
        
        # Concentration effect (linear scaling)
        c_factor = params.get('c_bulk', ref['c_bulk']) / ref['c_bulk']
        
        # Domain length effect (inverse scaling - larger domain = slower)
        L_factor = ref['L0_nm'] / params.get('L0_nm', ref['L0_nm'])
        
        # Core fraction effect (larger core = more surface = faster)
        fc_factor = params.get('fc', ref['fc']) / ref['fc']
        
        # Combined warping factor
        warp_factor = c_factor * L_factor * fc_factor
        
        # Clamp to reasonable range to avoid extreme warping
        warp_factor = np.clip(warp_factor, 0.5, 2.0)
        
        return warp_factor

# =============================================
# NUMBA‑ACCELERATED HELPERS
# =============================================
@jit(nopython=True, parallel=HAS_NUMBA)
def _interpolate_fields_numba(phi1, phi2, c1, c2, psi1, psi2, alpha):
    """Numba‑accelerated linear interpolation of fields."""
    phi = (1 - alpha) * phi1 + alpha * phi2
    c   = (1 - alpha) * c1   + alpha * c2
    psi = (1 - alpha) * psi1 + alpha * psi2
    return phi, c, psi

@jit(nopython=True, parallel=HAS_NUMBA)
def _blend_patches_numba(phi_s_list, psi_s_list, c_s_list, weights, patch_size, num_patches_h, num_patches_w):
    """Numba‑accelerated patch‑based blending."""
    H, W = phi_s_list[0].shape
    interp_phi = np.zeros((H, W), dtype=np.float32)
    interp_psi = np.zeros((H, W), dtype=np.float32)
    interp_c   = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    n_sources = len(phi_s_list)
    for src_idx in range(n_sources):
        for i in prange(num_patches_h):
            for j in prange(num_patches_w):
                w_patch = weights[src_idx, i*num_patches_w + j]
                if w_patch == 0.0:
                    continue
                h_start = i * patch_size
                h_end = h_start + patch_size
                w_start = j * patch_size
                w_end = w_start + patch_size
                interp_phi[h_start:h_end, w_start:w_end] += w_patch * phi_s_list[src_idx][h_start:h_end, w_start:w_end]
                interp_psi[h_start:h_end, w_start:w_end] += w_patch * psi_s_list[src_idx][h_start:h_end, w_start:w_end]
                interp_c[h_start:h_end, w_start:w_end]   += w_patch * c_s_list[src_idx][h_start:h_end, w_start:w_end]
                weight_sum[h_start:h_end, w_start:w_end] += w_patch
    return interp_phi, interp_psi, interp_c, weight_sum

# =============================================
# ROBUST SOLUTION LOADER (with lazy loading)
# =============================================
class EnhancedSolutionLoader:
    """Loads PKL files from numerical_solutions, parsing filenames as fallback."""
    
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        self.use_cache = True  # can be toggled for memory savings
    
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
        """Extract parameters from filenames."""
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
    
    def _ensure_3d(self, arr, target_shape=None):
        if arr is None:
            return np.zeros((1, 1, 1))
        if torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        if arr.ndim == 2:
            return arr[np.newaxis, :, :]
        return arr
    
    def _convert_tensors(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.cpu().numpy().astype(np.float32)  # convert to float32
                elif isinstance(value, (dict, list)):
                    self._convert_tensors(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if torch.is_tensor(item):
                    data[i] = item.cpu().numpy().astype(np.float32)
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
                                'phi': self._ensure_3d(phi).astype(np.float32),
                                'c': self._ensure_3d(c).astype(np.float32),
                                'psi': self._ensure_3d(psi).astype(np.float32)
                            }
                            snap_list.append(snap_dict)
                        elif isinstance(snap, dict):
                            snap_dict = {
                                't_nd': snap.get('t_nd', 0),
                                'phi': self._ensure_3d(snap.get('phi', np.zeros((1,1,1)))).astype(np.float32),
                                'c': self._ensure_3d(snap.get('c', np.zeros((1,1,1)))).astype(np.float32),
                                'psi': self._ensure_3d(snap.get('psi', np.zeros((1,1,1)))).astype(np.float32)
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
            if not standardized['history']:
                st.sidebar.warning(f"No snapshots in {os.path.basename(file_path)}")
                return None
            self._convert_tensors(standardized)
            return standardized
        except Exception as e:
            st.sidebar.error(f"Error loading {os.path.basename(file_path)}: {e}")
            return None
    
    def load_all_solutions(self, use_cache=True, max_files=None):
        """Load all solutions into a list (memory intensive)."""
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
                if use_cache:
                    self.cache[cache_key] = sol
                solutions.append(sol)
        st.sidebar.success(f"Loaded {len(solutions)} solutions.")
        return solutions
    
    def load_solution_generator(self, use_cache=True, max_files=None):
        """Generator that yields solutions one by one to save memory."""
        file_info = self.scan_solutions()
        if max_files:
            file_info = file_info[:max_files]
        for item in file_info:
            cache_key = item['filename']
            if use_cache and cache_key in self.cache:
                yield self.cache[cache_key]
                continue
            sol = self.read_simulation_file(item['path'])
            if sol:
                if use_cache:
                    self.cache[cache_key] = sol
                yield sol

# =============================================
# PATCH EMBEDDING AND SPATIAL ATTENTION MODULES
# =============================================
class PatchEmbed(nn.Module):
    """Convert field patches to embeddings using a small CNN."""
    def __init__(self, patch_size=16, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class SpatialCrossAttention(nn.Module):
    """Multi‑head cross‑attention between target and source patch embeddings."""
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, target_patches, source_patches):
        attn_output, attn_weights = self.cross_attn(target_patches, source_patches, source_patches)
        return attn_weights

# =============================================
# PARAMETER-AWARE TEMPORAL ATTENTION
# =============================================
class ParameterAwareTemporalAttention(nn.Module):
    """
    Parameter-conditioned temporal attention for thickness evolution.
    
    Theory:
    - Thickness evolution h(t) depends on physical parameters p
    - Attention weights should vary with both time AND parameter similarity
    - Query encodes (target_time, target_params)
    - Key/Value encodes (source_time, source_params, thickness)
    
    Architecture:
    1. Parameter encoding: MLP projects params to d_model
    2. Time encoding: Sinusoidal positional encoding
    3. Cross-attention: Query attends to source sequences
    4. Output: Parameter-aware thickness prediction
    """
    def __init__(self, 
                 d_model=64, 
                 nhead=4, 
                 num_layers=2,
                 num_params=4,  # fc, rs, c_bulk, L0_nm
                 use_time_warping=True,
                 time_warp_learnable=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_params = num_params
        self.use_time_warping = use_time_warping
        self.time_warp_learnable = time_warp_learnable
        
        # Parameter encoding MLP
        self.param_encoder = nn.Sequential(
            nn.Linear(num_params, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Time encoding (sinusoidal)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Combined query/key projection
        self.query_proj = nn.Linear(d_model * 2, d_model)  # time + params
        self.key_proj = nn.Linear(d_model * 2 + 1, d_model)  # time + params + thickness
        self.value_proj = nn.Linear(d_model * 2 + 1, d_model)
        
        # Multi-head attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Optional learnable time warping scale
        if time_warp_learnable:
            self.warp_scale = nn.Parameter(torch.ones(1))
        else:
            self.warp_scale = 1.0
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def _encode_params(self, params: torch.Tensor) -> torch.Tensor:
        """Encode physical parameters to embedding space."""
        return self.param_encoder(params)
    
    def _encode_time(self, time: torch.Tensor) -> torch.Tensor:
        """Encode time to embedding space."""
        return self.time_encoder(time.unsqueeze(-1))
    
    def _apply_time_warping(self, time: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Apply physics-informed time warping.
        
        Warping factor based on:
        - c_bulk: higher → faster (warp forward)
        - L0_nm: larger → slower (warp backward)
        - fc: larger → faster (warp forward)
        """
        if not self.use_time_warping:
            return time
        
        # Extract parameters (assumes order: fc, rs, c_bulk, L0_nm)
        fc = params[..., 0:1]
        c_bulk = params[..., 2:3]
        L0 = params[..., 3:4]
        
        # Reference values (normalized)
        fc_ref = 0.5  # normalized reference
        c_ref = 0.5
        L_ref = 0.5
        
        # Compute warping factor
        warp_factor = (c_bulk / c_ref) * (L_ref / L0) * (fc / fc_ref)
        
        # Apply learnable scale if enabled
        if self.time_warp_learnable:
            warp_factor = warp_factor * self.warp_scale
        
        # Clamp to avoid extreme warping
        warp_factor = torch.clamp(warp_factor, 0.5, 2.0)
        
        # Apply warping
        warped_time = time * warp_factor
        
        # Keep within [0, 1] range
        warped_time = torch.clamp(warped_time, 0.0, 1.0)
        
        return warped_time
    
    def forward(self, 
                target_times: torch.Tensor,      # (batch, num_target_times, 1)
                target_params: torch.Tensor,      # (batch, num_params)
                source_times: List[torch.Tensor], # List of (num_source_times, 1)
                source_params: torch.Tensor,      # (num_sources, num_params)
                source_thickness: List[torch.Tensor], # List of (num_source_times, 1)
                source_masks: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for parameter-aware temporal attention.
        
        Args:
            target_times: Normalized target time points
            target_params: Target physical parameters (normalized)
            source_times: List of source time sequences
            source_params: Source physical parameters (normalized)
            source_thickness: List of source thickness sequences
            source_masks: Optional masks for padding
        
        Returns:
            thickness_pred: Predicted thickness at target times (batch, num_target_times, 1)
            attention_weights: Attention weights for analysis (batch, num_target_times, num_sources)
        """
        batch_size = target_times.size(0)
        num_target_times = target_times.size(1)
        num_sources = len(source_times)
        
        # Encode target parameters
        target_param_emb = self._encode_params(target_params)  # (batch, d_model)
        
        # Apply time warping to target times
        warped_target_times = self._apply_time_warping(target_times.squeeze(-1), target_params)
        
        # Encode target times
        target_time_emb = self._encode_time(warped_target_times)  # (batch, num_target_times, d_model)
        
        # Combine time and parameter embeddings for query
        target_param_emb_exp = target_param_emb.unsqueeze(1).expand(-1, num_target_times, -1)
        query_input = torch.cat([target_time_emb, target_param_emb_exp], dim=-1)
        query = self.query_proj(query_input)  # (batch, num_target_times, d_model)
        
        # Prepare source keys and values
        source_keys = []
        source_values = []
        source_key_masks = []
        
        for src_idx in range(num_sources):
            src_time = source_times[src_idx]  # (num_source_times, 1)
            src_thick = source_thickness[src_idx]  # (num_source_times, 1)
            src_param = source_params[src_idx:src_idx+1]  # (1, num_params)
            
            # Apply time warping to source times
            warped_src_times = self._apply_time_warping(src_time.squeeze(-1), src_param)
            
            # Encode source times
            src_time_emb = self._encode_time(warped_src_times)  # (num_source_times, d_model)
            
            # Encode source parameters
            src_param_emb = self._encode_params(src_param)  # (1, d_model)
            src_param_emb_exp = src_param_emb.expand(src_time_emb.size(0), -1)
            
            # Combine time, params, and thickness for key/value
            src_input = torch.cat([src_time_emb, src_param_emb_exp, src_thick], dim=-1)
            
            # Project to key/value space
            key = self.key_proj(src_input)  # (num_source_times, d_model)
            value = self.value_proj(src_input)  # (num_source_times, d_model)
            
            source_keys.append(key)
            source_values.append(value)
            
            # Create mask if provided
            if source_masks and source_masks[src_idx] is not None:
                source_key_masks.append(source_masks[src_idx])
            else:
                source_key_masks.append(None)
        
        # Concatenate all sources along time dimension
        keys_concat = torch.cat(source_keys, dim=0)  # (total_source_times, d_model)
        values_concat = torch.cat(source_values, dim=0)  # (total_source_times, d_model)
        
        # Create combined key mask
        key_mask = None
        if any(m is not None for m in source_key_masks):
            key_mask = torch.cat([m if m is not None else torch.zeros_like(source_keys[i][:, 0]).bool() 
                                  for i, m in enumerate(source_key_masks)], dim=0)
            key_mask = key_mask.unsqueeze(0).expand(batch_size, -1)
        
        # Expand keys/values for batch
        keys_concat = keys_concat.unsqueeze(0).expand(batch_size, -1, -1)
        values_concat = values_concat.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention
        attn_output, attn_weights = self.cross_attn(
            query, 
            keys_concat, 
            values_concat, 
            key_padding_mask=key_mask
        )  # attn_output: (batch, num_target_times, d_model)
        
        # Layer norm and residual
        attn_output = self.norm1(attn_output)
        
        # Project to thickness
        thickness_pred = self.output_proj(attn_output)  # (batch, num_target_times, 1)
        
        # Process attention weights for analysis
        # attn_weights: (batch, num_target_times, total_source_times)
        # We want per-source weights, so sum over time steps within each source
        source_time_lengths = [st.size(0) for st in source_times]
        attention_per_source = []
        start_idx = 0
        for src_idx, src_len in enumerate(source_time_lengths):
            end_idx = start_idx + src_len
            src_attn = attn_weights[:, :, start_idx:end_idx].sum(dim=-1)  # (batch, num_target_times)
            attention_per_source.append(src_attn)
            start_idx = end_idx
        
        attention_per_source = torch.stack(attention_per_source, dim=-1)  # (batch, num_target_times, num_sources)
        
        # Normalize attention weights across sources
        attention_per_source = attention_per_source / (attention_per_source.sum(dim=-1, keepdim=True) + 1e-8)
        
        return thickness_pred, attention_per_source

# =============================================
# ENHANCED TEMPORAL ATTENTION (Backward Compatible Wrapper)
# =============================================
class TemporalAttention(nn.Module):
    """
    Wrapper for temporal attention with parameter awareness.
    Maintains backward compatibility while adding new features.
    """
    def __init__(self, 
                 d_model=64, 
                 nhead=4, 
                 num_layers=2,
                 use_parameter_aware=True,
                 use_time_warping=True,
                 time_warp_learnable=False):
        super().__init__()
        self.use_parameter_aware = use_parameter_aware
        self.use_time_warping = use_time_warping
        
        if use_parameter_aware:
            self.param_aware_attn = ParameterAwareTemporalAttention(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                use_time_warping=use_time_warping,
                time_warp_learnable=time_warp_learnable
            )
        else:
            # Fallback to simple attention
            self.d_model = d_model
            self.input_proj = nn.Linear(1, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
    
    def forward(self, 
                target_t: torch.Tensor,
                source_sequences: List[torch.Tensor],
                source_params: Optional[torch.Tensor] = None,
                target_params: Optional[torch.Tensor] = None,
                source_masks: Optional[List[torch.Tensor]] = None):
        """
        Forward pass with optional parameter awareness.
        
        If parameter_aware is True, uses ParameterAwareTemporalAttention.
        Otherwise, falls back to simple attention (backward compatible).
        """
        if self.use_parameter_aware and source_params is not None and target_params is not None:
            # Use parameter-aware attention
            source_times = [torch.linspace(0, 1, seq.size(0)).unsqueeze(-1) for seq in source_sequences]
            thickness_pred, attn_weights = self.param_aware_attn(
                target_times=target_t,
                target_params=target_params,
                source_times=source_times,
                source_params=source_params,
                source_thickness=source_sequences,
                source_masks=source_masks
            )
            return thickness_pred, attn_weights
        else:
            # Fallback to simple attention (backward compatible)
            target_emb = self.input_proj(target_t)
            source_embs = []
            for seq in source_sequences:
                seq_emb = self.input_proj(seq.unsqueeze(0))
                seq_emb = self.transformer(seq_emb)
                source_embs.append(seq_emb)
            source_concat = torch.cat(source_embs, dim=1)
            attn_out, attn_weights = self.cross_attn(target_emb, source_concat, source_concat)
            thickness = attn_out.mean(dim=-1, keepdim=True)
            return thickness.squeeze(-1), None

# =============================================
# CORE‑SHELL INTERPOLATOR – WITH PARAMETER-AWARE TEMPORAL ATTENTION
# =============================================
class CoreShellInterpolator:
    def __init__(self,
                 d_model=64,
                 nhead=8,
                 num_layers=3,
                 param_sigma=[0.15, 0.15, 0.15, 0.15],
                 temperature=1.0,
                 use_spatial_attention=False,
                 patch_size=16,
                 spatial_embed_dim=64,
                 spatial_nhead=4,
                 blend_mode='gate',
                 use_parameter_aware_temporal=True,
                 use_time_warping=True,
                 temporal_d_model=64,
                 temporal_nhead=4,
                 temporal_layers=2,
                 use_temporal_kernel=False,        # ⭐ NEW
                 temporal_sigma=0.1):               # ⭐ NEW
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.param_sigma = param_sigma
        self.temperature = temperature
        self.use_spatial_attention = use_spatial_attention
        self.patch_size = patch_size
        self.spatial_embed_dim = spatial_embed_dim
        self.spatial_nhead = spatial_nhead
        self.blend_mode = blend_mode
        self.use_parameter_aware_temporal = use_parameter_aware_temporal
        self.use_time_warping = use_time_warping
        self.use_temporal_kernel = use_temporal_kernel
        self.temporal_sigma = temporal_sigma
        
        # Global parameter encoder
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
        
        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention modules
        if use_spatial_attention:
            self.patch_embed = PatchEmbed(patch_size=patch_size,
                                          in_channels=3,
                                          embed_dim=spatial_embed_dim)
            self.patch_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=spatial_embed_dim, 
                                          nhead=spatial_nhead, 
                                          batch_first=True),
                num_layers=2
            )
            self.spatial_cross_attn = nn.MultiheadAttention(spatial_embed_dim, 
                                                           spatial_nhead, 
                                                           batch_first=True)
            self.blend_layer = nn.Linear(2, 1) if blend_mode == 'learned' else None
        
        # Parameter-aware temporal attention
        self.temporal_attn = TemporalAttention(
            d_model=temporal_d_model,
            nhead=temporal_nhead,
            num_layers=temporal_layers,
            use_parameter_aware=use_parameter_aware_temporal,
            use_time_warping=use_time_warping
        )
    
    def set_parameter_sigma(self, param_sigma):
        self.param_sigma = param_sigma
    
    def compute_parameter_kernel(self, source_params: List[Dict], target_params: Dict,
                                  source_times_norm: List[float] = None,   # ⭐ NEW
                                  target_time_norm: float = None):         # ⭐ NEW
        """Gaussian kernel in normalised parameter space AND time (optional)."""
        def norm_val(params, name):
            val = params.get(name, 0.5)
            return DepositionParameters.normalize(val, name)
        
        target_norm = np.array([
            norm_val(target_params, 'fc'),
            norm_val(target_params, 'rs'),
            norm_val(target_params, 'c_bulk'),
            norm_val(target_params, 'L0_nm')
        ])
        
        weights = []
        for src in source_params:
            src_norm = np.array([
                norm_val(src, 'fc'),
                norm_val(src, 'rs'),
                norm_val(src, 'c_bulk'),
                norm_val(src, 'L0_nm')
            ])
            diff = src_norm - target_norm
            w_param = np.exp(-0.5 * np.sum((diff / self.param_sigma)**2))
            
            # ⭐ NEW: temporal proximity weight (if enabled and times provided)
            if self.use_temporal_kernel and source_times_norm is not None and target_time_norm is not None:
                # source_times_norm is a list of mean source times (for each source)
                src_time_center = source_times_norm[0]  # but we need per source; fix: we'll pass list
                # Actually we pass a list of length len(source_params), so we need to index correctly.
                # Let's redesign: we'll pass source_times_norm as list of floats, one per source.
                # We'll fix below.
                pass  # will handle outside loop

        # We need to restructure: compute per-source time centers inside the loop.
        # Let's assume source_times_norm is a list of floats (mean time for each source).
        if self.use_temporal_kernel and source_times_norm is not None and target_time_norm is not None:
            for i, src in enumerate(source_params):
                src_norm = np.array([
                    norm_val(src, 'fc'),
                    norm_val(src, 'rs'),
                    norm_val(src, 'c_bulk'),
                    norm_val(src, 'L0_nm')
                ])
                diff = src_norm - target_norm
                w_param = np.exp(-0.5 * np.sum((diff / self.param_sigma)**2))
                # temporal part
                time_diff = source_times_norm[i] - target_time_norm
                w_time = np.exp(-0.5 * (time_diff / self.temporal_sigma)**2)
                # categorical factors
                factor = 1.0
                if src.get('bc_type') != target_params.get('bc_type'):
                    factor *= 0.1
                if src.get('use_edl') != target_params.get('use_edl'):
                    factor *= 0.1
                if src.get('mode') != target_params.get('mode'):
                    factor *= 0.1
                weights.append(w_param * w_time * factor)
        else:
            # Original parameter-only kernel
            for src in source_params:
                src_norm = np.array([
                    norm_val(src, 'fc'),
                    norm_val(src, 'rs'),
                    norm_val(src, 'c_bulk'),
                    norm_val(src, 'L0_nm')
                ])
                diff = src_norm - target_norm
                w_param = np.exp(-0.5 * np.sum((diff / self.param_sigma)**2))
                factor = 1.0
                if src.get('bc_type') != target_params.get('bc_type'):
                    factor *= 0.1
                if src.get('use_edl') != target_params.get('use_edl'):
                    factor *= 0.1
                if src.get('mode') != target_params.get('mode'):
                    factor *= 0.1
                weights.append(w_param * factor)
        return np.array(weights)
    
    def encode_parameters(self, params_list: List[Dict]) -> torch.Tensor:
        """Encode parameters into feature vectors."""
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
    
    def _extract_patches(self, field_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert field dict to patch embeddings."""
        phi = field_dict['phi']
        psi = field_dict['psi']
        c = field_dict['c']
        if phi.ndim == 3:
            mid = phi.shape[0] // 2
            phi = phi[mid]
            psi = psi[mid]
            c = c[mid]
        stack = np.stack([phi, psi, c], axis=0)
        tensor = torch.FloatTensor(stack).unsqueeze(0)
        patches = self.patch_embed(tensor)
        return patches
    
    def _get_fields_at_time(self, source: Dict, time_norm: float, target_shape: Tuple[int, ...]):
        """Linear interpolation in time for a given source, using Numba if available."""
        history = source.get('history', [])
        if not history:
            return {'phi': np.zeros(target_shape, dtype=np.float32), 
                    'c': np.zeros(target_shape, dtype=np.float32), 
                    'psi': np.zeros(target_shape, dtype=np.float32)}
        
        t_max = 1.0
        if source.get('thickness_history'):
            t_max = source['thickness_history'][-1]['t_nd']
        else:
            t_max = history[-1]['t_nd']
        
        t_target = time_norm * t_max
        
        if len(history) == 1:
            snap = history[0]
            phi = snap['phi'].astype(np.float32)
            c   = snap['c'].astype(np.float32)
            psi = snap['psi'].astype(np.float32)
        else:
            t_vals = np.array([s['t_nd'] for s in history])
            if t_target <= t_vals[0]:
                snap = history[0]
                phi = snap['phi'].astype(np.float32)
                c   = snap['c'].astype(np.float32)
                psi = snap['psi'].astype(np.float32)
            elif t_target >= t_vals[-1]:
                snap = history[-1]
                phi = snap['phi'].astype(np.float32)
                c   = snap['c'].astype(np.float32)
                psi = snap['psi'].astype(np.float32)
            else:
                idx = np.searchsorted(t_vals, t_target) - 1
                idx = max(0, min(idx, len(history)-2))
                t1, t2 = t_vals[idx], t_vals[idx+1]
                snap1, snap2 = history[idx], history[idx+1]
                alpha = (t_target - t1) / (t2 - t1) if t2 > t1 else 0.0
                phi1, phi2 = snap1['phi'].astype(np.float32), snap2['phi'].astype(np.float32)
                c1, c2     = snap1['c'].astype(np.float32),   snap2['c'].astype(np.float32)
                psi1, psi2 = snap1['psi'].astype(np.float32), snap2['psi'].astype(np.float32)
                
                if HAS_NUMBA:
                    phi, c, psi = _interpolate_fields_numba(phi1, phi2, c1, c2, psi1, psi2, alpha)
                else:
                    phi = (1 - alpha) * phi1 + alpha * phi2
                    c   = (1 - alpha) * c1   + alpha * c2
                    psi = (1 - alpha) * psi1 + alpha * psi2
        
        # Zoom to target shape if needed (still using scipy zoom)
        if phi.shape != target_shape:
            factors = tuple(float(target_shape[i]) / float(phi.shape[i]) for i in range(phi.ndim))
            phi = zoom(phi, factors, order=1)
            c   = zoom(c,   factors, order=1)
            psi = zoom(psi, factors, order=1)
        
        return {'phi': phi, 'c': c, 'psi': psi}
    
    def interpolate_fields(self,
                           sources: List[Dict],
                           target_params: Dict,
                           target_shape: Tuple[int, ...] = (256, 256),
                           n_time_points: int = 100,
                           time_norm: Optional[float] = None,
                           kernel_strength: float = 1.0,
                           apply_physics_constraints: bool = True):
        """
        Interpolate fields and thickness evolution with parameter-aware temporal attention.
        """
        if not sources:
            return None
        
        if time_norm is None:
            time_norm = 1.0
        
        # Prepare source data
        source_params = []
        source_fields_at_time = []
        source_thickness_hist = []
        source_time_centers = []  # for temporal kernel
        
        for src in sources:
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
            source_params.append(params)
            
            fields_t = self._get_fields_at_time(src, time_norm, target_shape)
            source_fields_at_time.append(fields_t)
            
            thick_hist = src.get('thickness_history', [])
            if thick_hist:
                t_vals = np.array([th['t_nd'] for th in thick_hist])
                th_vals = np.array([th['th_nm'] for th in thick_hist])
                t_max = t_vals[-1] if len(t_vals) > 0 else 1.0
                t_norm = t_vals / t_max
                source_thickness_hist.append({
                    't_norm': t_norm,
                    'th_nm': th_vals,
                    't_max': t_max
                })
                source_time_centers.append(np.mean(t_norm))
            else:
                source_thickness_hist.append({
                    't_norm': np.array([0.0, 1.0]),
                    'th_nm': np.array([0.0, 0.0]),
                    't_max': 1.0
                })
                source_time_centers.append(0.5)
        
        N = len(source_params)
        if N == 0:
            st.error("No valid source fields.")
            return None
        
        # ---- Global parameter attention ----
        source_features = self.encode_parameters(source_params)
        target_features = self.encode_parameters([target_params])
        all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
        proj = self.input_proj(all_features)
        proj = self.pos_encoder(proj)
        transformer_out = self.transformer(proj)
        target_rep = transformer_out[:, 0, :]
        source_reps = transformer_out[:, 1:, :]
        
        global_attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1,2)).squeeze(1)
        global_attn_scores = global_attn_scores / np.sqrt(self.d_model) / self.temperature
        
        # Compute kernel weights with optional temporal proximity
        kernel_weights = self.compute_parameter_kernel(
            source_params, target_params,
            source_times_norm=source_time_centers if self.use_temporal_kernel else None,
            target_time_norm=time_norm if self.use_temporal_kernel else None
        )
        kernel_tensor = torch.FloatTensor(kernel_weights).unsqueeze(0)
        
        target_exp = target_rep.expand(N, -1).unsqueeze(0)
        gate_input = torch.cat([target_exp, source_reps], dim=-1)
        gate = self.gate_net(gate_input).squeeze(-1)
        
        global_blended = kernel_strength * kernel_tensor + (1 - kernel_strength) * global_attn_scores
        global_weights = torch.softmax(global_blended, dim=-1)
        
        # ---- Parameter-aware temporal attention for thickness ----
        common_t_norm = np.linspace(0, 1, n_time_points)
        target_times = torch.FloatTensor(common_t_norm).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        
        # Encode target parameters for temporal attention
        target_params_for_attn = torch.FloatTensor([
            DepositionParameters.normalize(target_params.get('fc', 0.18), 'fc'),
            DepositionParameters.normalize(target_params.get('rs', 0.2), 'rs'),
            DepositionParameters.normalize(target_params.get('c_bulk', 0.5), 'c_bulk'),
            DepositionParameters.normalize(target_params.get('L0_nm', 60.0), 'L0_nm')
        ]).unsqueeze(0)  # (1, 4)
        
        # Prepare source sequences and parameters
        source_seq_tensors = []
        source_param_tensors = []
        for i, thick in enumerate(source_thickness_hist):
            t_norm_src = thick['t_norm']
            th_src = thick['th_nm']
            if len(t_norm_src) > 1:
                f = interp1d(t_norm_src, th_src, kind='linear', bounds_error=False, fill_value='extrapolate')
                th_interp = f(common_t_norm)
            else:
                th_interp = np.full_like(common_t_norm, th_src[0] if len(th_src)>0 else 0.0)
            source_seq_tensors.append(torch.FloatTensor(th_interp).unsqueeze(-1))
            
            # Source parameters for temporal attention
            src_params = torch.FloatTensor([
                DepositionParameters.normalize(source_params[i].get('fc', 0.18), 'fc'),
                DepositionParameters.normalize(source_params[i].get('rs', 0.2), 'rs'),
                DepositionParameters.normalize(source_params[i].get('c_bulk', 0.5), 'c_bulk'),
                DepositionParameters.normalize(source_params[i].get('L0_nm', 60.0), 'L0_nm')
            ])
            source_param_tensors.append(src_params)
        
        source_param_tensor = torch.stack(source_param_tensors, dim=0)  # (N, 4)
        
        # Use parameter-aware temporal attention
        if self.use_parameter_aware_temporal:
            thickness_pred, temporal_attn_weights = self.temporal_attn(
                target_t=target_times,
                source_sequences=source_seq_tensors,
                source_params=source_param_tensor,
                target_params=target_params_for_attn
            )
            thickness_interp = thickness_pred.squeeze(0).squeeze(-1).detach().cpu().numpy()
            temporal_attention_info = temporal_attn_weights.squeeze(0).detach().cpu().numpy() if temporal_attn_weights is not None else None
        else:
            # Fallback to simple weighted average
            thickness_interp = np.zeros_like(common_t_norm)
            global_weights_np = global_weights.detach().cpu().numpy().flatten()
            for i, seq in enumerate(source_seq_tensors):
                thickness_interp += global_weights_np[i] * seq.squeeze().numpy()
            temporal_attention_info = None
        
        # ---- Spatial attention (if enabled) ----
        if self.use_spatial_attention:
            source_patch_embs = []
            for fld in source_fields_at_time:
                patches = self._extract_patches(fld)
                patches = self.patch_transformer(patches)
                source_patch_embs.append(patches)
            source_patches_concat = torch.cat(source_patch_embs, dim=1)
            
            sim = torch.matmul(target_rep, source_patches_concat.transpose(1,2))
            sim = sim.squeeze(1) / np.sqrt(self.spatial_embed_dim)
            patch_weights_all = torch.softmax(sim, dim=-1)
            spatial_weights_per_patch = patch_weights_all.view(1, N, -1)
            
            global_expanded = global_weights.unsqueeze(-1).expand(-1, -1, spatial_weights_per_patch.size(-1))
            
            if self.blend_mode == 'product':
                blended_patch_weights = global_expanded * spatial_weights_per_patch
            elif self.blend_mode == 'add':
                blended_patch_weights = global_expanded + spatial_weights_per_patch
            else:
                blended_patch_weights = global_expanded * spatial_weights_per_patch
            
            blended_patch_weights = blended_patch_weights / (blended_patch_weights.sum(dim=1, keepdim=True) + 1e-8)
        else:
            blended_patch_weights = None
        
        # ---- Interpolate fields ----
        if self.use_spatial_attention and blended_patch_weights is not None:
            num_patches = blended_patch_weights.size(-1)
            H, W = target_shape[-2:]
            patch_size = self.patch_size
            assert H % patch_size == 0 and W % patch_size == 0, "Target shape must be divisible by patch_size"
            num_patches_h = H // patch_size
            num_patches_w = W // patch_size
            assert num_patches == num_patches_h * num_patches_w, "Patch count mismatch"
            
            # Use Numba-accelerated blending if available
            if HAS_NUMBA:
                phi_s_list = [f['phi'] for f in source_fields_at_time]
                psi_s_list = [f['psi'] for f in source_fields_at_time]
                c_s_list   = [f['c']   for f in source_fields_at_time]
                interp_phi, interp_psi, interp_c, weight_sum = _blend_patches_numba(
                    phi_s_list, psi_s_list, c_s_list,
                    blended_patch_weights.detach().cpu().numpy().squeeze(0),
                    patch_size, num_patches_h, num_patches_w
                )
            else:
                interp_phi = np.zeros(target_shape)
                interp_psi = np.zeros(target_shape)
                interp_c = np.zeros(target_shape)
                weight_sum = np.zeros(target_shape)
                for src_idx in range(N):
                    phi_s = source_fields_at_time[src_idx]['phi']
                    psi_s = source_fields_at_time[src_idx]['psi']
                    c_s = source_fields_at_time[src_idx]['c']
                    for i in range(num_patches_h):
                        for j in range(num_patches_w):
                            w_patch = blended_patch_weights[0, src_idx, i*num_patches_w + j].item()
                            if w_patch == 0:
                                continue
                            h_start = i * patch_size
                            h_end = h_start + patch_size
                            w_start = j * patch_size
                            w_end = w_start + patch_size
                            interp_phi[h_start:h_end, w_start:w_end] += w_patch * phi_s[h_start:h_end, w_start:w_end]
                            interp_psi[h_start:h_end, w_start:w_end] += w_patch * psi_s[h_start:h_end, w_start:w_end]
                            interp_c[h_start:h_end, w_start:w_end]   += w_patch * c_s[h_start:h_end, w_start:w_end]
                            weight_sum[h_start:h_end, w_start:w_end] += w_patch
            
            # Normalize
            interp_phi = np.divide(interp_phi, weight_sum, where=weight_sum>0)
            interp_psi = np.divide(interp_psi, weight_sum, where=weight_sum>0)
            interp_c   = np.divide(interp_c, weight_sum, where=weight_sum>0)
            
            mask_zero = weight_sum == 0
            if np.any(mask_zero):
                global_phi = np.zeros_like(interp_phi)
                global_psi = np.zeros_like(interp_psi)
                global_c = np.zeros_like(interp_c)
                global_w = global_weights.detach().cpu().numpy().flatten()
                for i in range(N):
                    global_phi += global_w[i] * source_fields_at_time[i]['phi']
                    global_psi += global_w[i] * source_fields_at_time[i]['psi']
                    global_c += global_w[i] * source_fields_at_time[i]['c']
                interp_phi[mask_zero] = global_phi[mask_zero]
                interp_psi[mask_zero] = global_psi[mask_zero]
                interp_c[mask_zero]   = global_c[mask_zero]
            
            interp = {'phi': interp_phi, 'psi': interp_psi, 'c': interp_c}
        else:
            interp = {'phi': np.zeros(target_shape, dtype=np.float32),
                      'c': np.zeros(target_shape, dtype=np.float32),
                      'psi': np.zeros(target_shape, dtype=np.float32)}
            global_weights_np = global_weights.detach().cpu().numpy().flatten()
            for i, fld in enumerate(source_fields_at_time):
                interp['phi'] += global_weights_np[i] * fld['phi']
                interp['c']   += global_weights_np[i] * fld['c']
                interp['psi'] += global_weights_np[i] * fld['psi']
            
            if not self.use_spatial_attention:
                interp['phi'] = gaussian_filter(interp['phi'], sigma=1.0)
                interp['c']   = gaussian_filter(interp['c'], sigma=1.0)
                interp['psi'] = gaussian_filter(interp['psi'], sigma=1.0)
        
        # ---- Physics‑informed post‑processing ----
        if apply_physics_constraints:
            interp['phi'], interp['psi'] = DepositionPhysics.enforce_phase_constraints(interp['phi'], interp['psi'])
        
        # ---- Derived quantities ----
        material = DepositionPhysics.material_proxy(interp['phi'], interp['psi'])
        alpha = target_params.get('alpha_nd', 2.0)
        potential = DepositionPhysics.potential_proxy(interp['c'], alpha)
        fc = target_params.get('fc', target_params.get('core_radius_frac', 0.18))
        dx = 1.0 / (target_shape[-1] - 1) if len(target_shape)==2 else 1.0 / (target_shape[-1]-1)
        L0 = target_params.get('L0_nm', 20.0) * 1e-9
        thickness_nm = DepositionPhysics.shell_thickness(interp['phi'], interp['psi'], fc, dx=dx) * L0 * 1e9
        stats = DepositionPhysics.phase_stats(interp['phi'], interp['psi'], dx, dx, L0)
        
        # ---- Uncertainty ----
        phi_stack = np.stack([f['phi'] for f in source_fields_at_time], axis=0)
        weighted_mean = np.sum(global_weights_np[:, None, None] * phi_stack, axis=0)
        weighted_var = np.sum(global_weights_np[:, None, None] * (phi_stack - weighted_mean)**2, axis=0)
        sum_w_sq = np.sum(global_weights_np**2)
        if sum_w_sq < 1.0:
            weighted_var /= (1 - sum_w_sq + 1e-8)
        uncertainty_map = np.sqrt(weighted_var)
        
        # ---- Time warping info ----
        target_warp_factor = DepositionPhysics.compute_time_warp_factor(target_params)
        source_warp_factors = [DepositionPhysics.compute_time_warp_factor(p) for p in source_params]
        
        result = {
            'fields': interp,
            'derived': {
                'material': material,
                'potential': potential,
                'thickness_nm': thickness_nm,
                'phase_stats': stats,
                'thickness_time': {
                    't_norm': common_t_norm.tolist(),
                    'th_nm': thickness_interp.tolist()
                },
                'uncertainty_phi': uncertainty_map.tolist(),
                'temporal_attention_weights': temporal_attention_info.tolist() if temporal_attention_info is not None else None,
                'time_warp_factor': target_warp_factor,
                'source_warp_factors': source_warp_factors
            },
            'weights': {
                'combined': global_weights_np.tolist(),
                'kernel': kernel_weights.tolist(),
                'attention': global_attn_scores.squeeze().detach().cpu().numpy().tolist(),
                'gate': gate.squeeze().detach().cpu().numpy().tolist()
            },
            'target_params': target_params,
            'shape': target_shape,
            'num_sources': N,
            'source_params': source_params,
            'time_norm': time_norm,
            'use_spatial_attention': self.use_spatial_attention,
            'use_parameter_aware_temporal': self.use_parameter_aware_temporal,
            'use_time_warping': self.use_time_warping,
            'use_temporal_kernel': self.use_temporal_kernel
        }
        return result
    
    def _ensure_2d(self, arr):
        if arr is None:
            return np.zeros((1,1))
        if torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        if arr.ndim == 3:
            mid = arr.shape[0] // 2
            return arr[mid, :, :]
        return arr

# =============================================
# POSITIONAL ENCODING
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
# HEATMAP VISUALIZER
# =============================================
class HeatMapVisualizer:
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
    
    def _get_extent(self, L0_nm):
        return [0, L0_nm, 0, L0_nm]
    
    def create_field_heatmap(self, field_data, title, cmap_name='viridis',
                             L0_nm=20.0, figsize=(10,8), colorbar_label="",
                             vmin=None, vmax=None, target_params=None):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        extent = self._get_extent(L0_nm)
        im = ax.imshow(field_data, cmap=cmap_name, vmin=vmin, vmax=vmax,
                       extent=extent, aspect='equal', origin='lower')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (nm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (nm)', fontsize=14, fontweight='bold')
        title_str = title
        if target_params:
            fc = target_params.get('fc', 0)
            rs = target_params.get('rs', 0)
            cb = target_params.get('c_bulk', 0)
            title_str += f"\nfc={fc:.3f}, rs={rs:.3f}, c_bulk={cb:.2f}, L0={L0_nm} nm"
        ax.set_title(title_str, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def create_interactive_heatmap(self, field_data, title, cmap_name='viridis',
                                   L0_nm=20.0, width=800, height=700,
                                   target_params=None):
        ny, nx = field_data.shape
        x = np.linspace(0, L0_nm, nx)
        y = np.linspace(0, L0_nm, ny)
        hover = [[f"X={x[j]:.2f} nm, Y={y[i]:.2f} nm<br>Value={field_data[i,j]:.4f}"
                  for j in range(nx)] for i in range(ny)]
        fig = go.Figure(data=go.Heatmap(
            z=field_data, x=x, y=y, colorscale=cmap_name,
            hoverinfo='text', text=hover,
            colorbar=dict(title=dict(text="Value", font=dict(size=14)))
        ))
        title_str = title
        if target_params:
            fc = target_params.get('fc', 0)
            rs = target_params.get('rs', 0)
            cb = target_params.get('c_bulk', 0)
            title_str += f"<br>fc={fc:.3f}, rs={rs:.3f}, c_bulk={cb:.2f}, L0={L0_nm} nm"
        fig.update_layout(
            title=dict(text=title_str, font=dict(size=20), x=0.5),
            xaxis=dict(title="X (nm)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y (nm)"),
            width=width, height=height
        )
        return fig
    
    def create_thickness_plot(self, thickness_time, source_curves=None, weights=None,
                              title="Shell Thickness Evolution", figsize=(10,6),
                              current_time=None, temporal_attention_weights=None):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        t_norm = thickness_time['t_norm']
        th_nm = thickness_time['th_nm']
        ax.plot(t_norm, th_nm, 'b-', linewidth=3, label='Interpolated (Parameter-Aware)')
        if source_curves is not None and weights is not None:
            for i, (src_t, src_th) in enumerate(source_curves):
                alpha = min(weights[i] * 5, 0.8)
                ax.plot(src_t, src_th, '--', linewidth=1, alpha=alpha, label=f'Source {i+1} (w={weights[i]:.3f})')
        if current_time is not None:
            interp_th = np.interp(current_time, t_norm, th_nm, left=th_nm[0], right=th_nm[-1])
            ax.axvline(current_time, color='r', linestyle='--', linewidth=2, alpha=0.7)
            ax.plot(current_time, interp_th, 'ro', markersize=8, label=f'Current t={current_time:.2f}')
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Shell Thickness (nm)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        return fig
    
    def create_uncertainty_map(self, uncertainty_data, title="Uncertainty (φ)",
                               cmap_name='hot', L0_nm=20.0, figsize=(10,8),
                               target_params=None):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        extent = self._get_extent(L0_nm)
        im = ax.imshow(uncertainty_data, cmap=cmap_name, extent=extent, aspect='equal', origin='lower')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Standard deviation', fontsize=14)
        ax.set_xlabel('X (nm)', fontsize=14)
        ax.set_ylabel('Y (nm)', fontsize=14)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def create_temporal_attention_heatmap(self, attention_weights, source_params, target_params,
                                          title="Temporal Attention Weights", figsize=(12, 6)):
        """Visualize how attention weights vary across time and sources."""
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        
        # attention_weights: (num_time_points, num_sources)
        im = ax.imshow(attention_weights.T, aspect='auto', cmap='viridis', origin='lower')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=14)
        
        ax.set_xlabel('Normalized Time', fontsize=14)
        ax.set_ylabel('Source Index', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Add source parameter annotations
        for i, params in enumerate(source_params):
            c_bulk = params.get('c_bulk', 0.5)
            ax.text(attention_weights.shape[0] + 0.5, i, f'c={c_bulk:.2f}', 
                   va='center', fontsize=10)
        
        plt.tight_layout()
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
                'interpolation_method': 'core_shell_transformer_parameter_aware',
                'use_spatial_attention': res.get('use_spatial_attention', False),
                'use_parameter_aware_temporal': res.get('use_parameter_aware_temporal', False),
                'use_time_warping': res.get('use_time_warping', False),
                'use_temporal_kernel': res.get('use_temporal_kernel', False),
                'visualization_params': visualization_params
            },
            'result': {
                'target_params': res['target_params'],
                'shape': res['shape'],
                'num_sources': res['num_sources'],
                'weights': res['weights'],
                'time_norm': res.get('time_norm', 1.0)
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
            fc = p.get('fc', 0)
            rs = p.get('rs', 0)
            cb = p.get('c_bulk', 0)
            filename = f"interp_fc{fc:.3f}_rs{rs:.3f}_c{cb:.2f}_{ts}.json"
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
    def export_to_csv(self, interpolation_result, filename=None):
        if filename is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            p = interpolation_result['target_params']
            fc = p.get('fc', 0)
            rs = p.get('rs', 0)
            cb = p.get('c_bulk', 0)
            filename = f"fields_fc{fc:.3f}_rs{rs:.3f}_c{cb:.2f}_{ts}.csv"
        shape = interpolation_result['shape']
        L0 = interpolation_result['target_params'].get('L0_nm', 20.0)
        x = np.linspace(0, L0, shape[1])
        y = np.linspace(0, L0, shape[0])
        X, Y = np.meshgrid(x, y)
        data = {'x_nm': X.flatten(), 'y_nm': Y.flatten()}
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
# MAIN STREAMLIT APP
# =============================================
def main():
    st.set_page_config(page_title="Core‑Shell Deposition Interpolator (Parameter-Aware Temporal Attention)",
                       layout="wide", page_icon="🧪", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .main-header { font-size: 3.2rem; color: #1E3A8A; text-align: center; padding: 1rem;
                   background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981); -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent; font-weight: 900; margin-bottom: 1rem; }
    .section-header { font-size: 2.0rem; color: #374151; font-weight: 800;
                      border-left: 6px solid #3B82F6; padding-left: 1.2rem; margin-top: 1.8rem; margin-bottom: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🧪 Core‑Shell Deposition Interpolator (Parameter-Aware Temporal Attention)</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = HeatMapVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'temporal_cache' not in st.session_state:
        st.session_state.temporal_cache = {}
    
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
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.session_state.temporal_cache = {}
                st.success("Cache cleared")
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
        st.divider()
        
        st.markdown('<h2 class="section-header">⚛️ Interpolation Settings</h2>', unsafe_allow_html=True)
        sigma_fc = st.slider("Kernel σ (fc)", 0.05, 0.3, 0.15, 0.01)
        sigma_rs = st.slider("Kernel σ (rs)", 0.05, 0.3, 0.15, 0.01)
        sigma_c = st.slider("Kernel σ (c_bulk)", 0.05, 0.3, 0.15, 0.01)
        sigma_L = st.slider("Kernel σ (L0_nm)", 0.05, 0.3, 0.15, 0.01)
        temperature = st.slider("Attention temperature", 0.1, 10.0, 1.0, 0.1)
        n_time_points = st.slider("Number of time points for thickness curve", 20, 200, 100, 10)
        kernel_strength = st.slider("Kernel strength (0 = pure attention, 1 = pure vicinity)", 0.0, 1.0, 1.0, 0.05)
        
        st.markdown("#### 🧩 Local Spatial Attention")
        use_spatial_attention = st.checkbox("Enable patch‑based spatial attention", value=False)
        if use_spatial_attention:
            patch_size = st.slider("Patch size (pixels)", 8, 64, 16, 8)
            spatial_embed_dim = st.slider("Spatial embedding dimension", 16, 128, 64, 16)
            spatial_nhead = st.slider("Spatial attention heads", 1, 8, 4, 1)
            blend_mode = st.selectbox("Blend global/local", ["product", "add"], index=0)
        else:
            patch_size = 16
            spatial_embed_dim = 64
            spatial_nhead = 4
            blend_mode = 'product'
        
        st.markdown("#### ⏱️ Temporal Proximity Kernel")
        use_temporal_kernel = st.checkbox("Use time‑distance kernel", value=False,
                                          help="Give higher weight to sources with simulation times closer to target time")
        temporal_sigma = st.slider("Temporal kernel σ", 0.05, 0.5, 0.1, 0.01)
        
        st.markdown("#### ⏱️ Parameter-Aware Temporal Attention")
        use_parameter_aware_temporal = st.checkbox("Enable parameter-aware temporal attention", value=True,
                                                   help="Makes thickness evolution depend on physical parameters")
        use_time_warping = st.checkbox("Enable physics-informed time warping", value=True,
                                       help="Aligns sequences based on concentration, domain size, core radius")
        temporal_d_model = st.slider("Temporal attention dimension", 32, 128, 64, 16)
        temporal_nhead = st.slider("Temporal attention heads", 1, 8, 4, 1)
        temporal_layers = st.slider("Temporal transformer layers", 1, 4, 2, 1)
        
        apply_physics = st.checkbox("Apply physics constraints (phase separation)", value=True)
        
        if st.button("🧠 Perform Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Interpolating with parameter-aware temporal attention..."):
                    interpolator = CoreShellInterpolator(
                        d_model=64,
                        nhead=8,
                        num_layers=3,
                        param_sigma=[sigma_fc, sigma_rs, sigma_c, sigma_L],
                        temperature=temperature,
                        use_spatial_attention=use_spatial_attention,
                        patch_size=patch_size,
                        spatial_embed_dim=spatial_embed_dim,
                        spatial_nhead=spatial_nhead,
                        blend_mode=blend_mode,
                        use_parameter_aware_temporal=use_parameter_aware_temporal,
                        use_time_warping=use_time_warping,
                        temporal_d_model=temporal_d_model,
                        temporal_nhead=temporal_nhead,
                        temporal_layers=temporal_layers,
                        use_temporal_kernel=use_temporal_kernel,
                        temporal_sigma=temporal_sigma
                    )
                    st.session_state.interpolator = interpolator
                    target = {
                        'fc': fc, 'rs': rs, 'c_bulk': c_bulk, 'L0_nm': L0_nm,
                        'bc_type': bc_type, 'use_edl': use_edl, 'mode': mode,
                        'growth_model': growth_model, 'alpha_nd': alpha_nd
                    }
                    target_shape = (256, 256)
                    res = interpolator.interpolate_fields(
                        st.session_state.solutions, target, target_shape=target_shape,
                        n_time_points=n_time_points, time_norm=1.0,
                        kernel_strength=kernel_strength,
                        apply_physics_constraints=apply_physics
                    )
                    if res:
                        st.session_state.interpolation_result = res
                        cache_key = (frozenset(target.items()), 1.0, kernel_strength, 
                                    use_spatial_attention, patch_size, use_parameter_aware_temporal,
                                    use_temporal_kernel, temporal_sigma)
                        st.session_state.temporal_cache[cache_key] = res
                        st.success("Interpolation successful! Parameter-aware temporal attention enabled.")
                    else:
                        st.error("Interpolation failed.")
    
    # Main area
    if st.session_state.interpolation_result:
        res = st.session_state.interpolation_result
        target = res['target_params']
        L0_nm = target.get('L0_nm', 60.0)
        
        st.markdown('<h2 class="section-header">⏱️ Global Time Control</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            current_time = st.slider("Normalized Time (0 = start, 1 = end)", 0.0, 1.0,
                                     value=res.get('time_norm', 1.0), step=0.01)
        with col2:
            if st.button("🔄 Update to this time", use_container_width=True):
                ks = kernel_strength if 'kernel_strength' in locals() else 1.0
                interp = st.session_state.interpolator
                cache_key = (frozenset(target.items()), current_time, ks,
                            interp.use_spatial_attention, interp.patch_size,
                            interp.use_parameter_aware_temporal,
                            interp.use_temporal_kernel, interp.temporal_sigma)
                if cache_key in st.session_state.temporal_cache:
                    st.session_state.interpolation_result = st.session_state.temporal_cache[cache_key]
                else:
                    with st.spinner(f"Interpolating at t = {current_time:.2f}..."):
                        new_res = interp.interpolate_fields(
                            st.session_state.solutions, target, target_shape=(256,256),
                            n_time_points=n_time_points, time_norm=current_time,
                            kernel_strength=ks, apply_physics_constraints=apply_physics
                        )
                        if new_res:
                            st.session_state.temporal_cache[cache_key] = new_res
                            st.session_state.interpolation_result = new_res
                    st.rerun()
        
        tabs = st.tabs(["📊 Fields", "📈 Thickness Evolution", "🧪 Derived Quantities", 
                       "⚖️ Weights", "📉 Uncertainty", "⏱️ Temporal Attention", "💾 Export"])
        
        with tabs[0]:
            st.markdown('<h2 class="section-header">📊 Interpolated Fields</h2>', unsafe_allow_html=True)
            field_choice = st.selectbox("Select field", ['c (concentration)', 'phi (shell)', 'psi (core)'], key='field_choice')
            field_map = {'c (concentration)': 'c', 'phi (shell)': 'phi', 'psi (core)': 'psi'}
            field_key = field_map[field_choice]
            field_data = res['fields'][field_key]
            cmap_cat = st.selectbox("Colormap category", list(COLORMAP_OPTIONS.keys()), index=0)
            cmap = st.selectbox("Colormap", COLORMAP_OPTIONS[cmap_cat], index=0, key='cmap')
            fig = st.session_state.visualizer.create_field_heatmap(
                field_data, title=f"Interpolated {field_choice} at t = {res['time_norm']:.2f}",
                cmap_name=cmap, L0_nm=L0_nm, target_params=target,
                colorbar_label=field_choice.split()[0]
            )
            st.pyplot(fig)
            if st.checkbox("Show interactive heatmap"):
                fig_inter = st.session_state.visualizer.create_interactive_heatmap(
                    field_data, title=f"Interpolated {field_choice} at t = {res['time_norm']:.2f}",
                    cmap_name=cmap, L0_nm=L0_nm, target_params=target
                )
                st.plotly_chart(fig_inter, use_container_width=True)
        
        with tabs[1]:
            st.markdown('<h2 class="section-header">📈 Shell Thickness Evolution</h2>', unsafe_allow_html=True)
            thickness_time = res['derived']['thickness_time']
            title_th = (f"Interpolated Thickness for fc={target['fc']:.3f}, rs={target['rs']:.3f}, c_bulk={target['c_bulk']:.2f}")
            
            # Show time warping info
            col_warp1, col_warp2 = st.columns(2)
            with col_warp1:
                st.metric("Target Time Warp Factor", f"{res['derived']['time_warp_factor']:.3f}",
                         help=">1.0 means faster evolution, <1.0 means slower")
            with col_warp2:
                st.metric("Temporal Attention", "Enabled" if res['use_parameter_aware_temporal'] else "Disabled")
            
            fig_th = st.session_state.visualizer.create_thickness_plot(
                thickness_time, title=title_th, current_time=res['time_norm']
            )
            st.pyplot(fig_th)
        
        with tabs[2]:
            st.markdown('<h2 class="section-header">🧪 Derived Quantities at current time</h2>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Shell thickness (nm)", f"{res['derived']['thickness_nm']:.3f}")
            with col2:
                st.metric("Number of sources", res['num_sources'])
            st.subheader("Phase statistics")
            stats = res['derived']['phase_stats']
            cols = st.columns(3)
            with cols[0]:
                st.metric("Electrolyte", f"{stats['Electrolyte'][0]:.4f} (nd²)",
                          help=f"Real area: {stats['Electrolyte'][1]*1e18:.2f} nm²")
            with cols[1]:
                st.metric("Ag shell", f"{stats['Ag'][0]:.4f} (nd²)",
                          help=f"Real area: {stats['Ag'][1]*1e18:.2f} nm²")
            with cols[2]:
                st.metric("Cu core", f"{stats['Cu'][0]:.4f} (nd²)",
                          help=f"Real area: {stats['Cu'][1]*1e18:.2f} nm²")
            st.subheader("Material proxy")
            fig_mat = st.session_state.visualizer.create_field_heatmap(
                res['derived']['material'], title="Material proxy",
                cmap_name='Set1', L0_nm=L0_nm, target_params=target,
                colorbar_label="Material", vmin=0, vmax=2
            )
            st.pyplot(fig_mat)
            st.subheader("Potential proxy")
            fig_pot = st.session_state.visualizer.create_field_heatmap(
                res['derived']['potential'], title="Potential proxy",
                cmap_name='RdBu_r', L0_nm=L0_nm, target_params=target,
                colorbar_label="-α·c"
            )
            st.pyplot(fig_pot)
        
        with tabs[3]:
            st.markdown('<h2 class="section-header">⚖️ Attention Weights & Gate</h2>', unsafe_allow_html=True)
            df_weights = pd.DataFrame({
                'Source index': range(len(res['weights']['combined'])),
                'Combined weight': res['weights']['combined'],
                'Kernel weight': res['weights']['kernel'],
                'Attention score': res['weights']['attention'],
                'Gate': res['weights']['gate']
            })
            st.dataframe(df_weights.style.format("{:.4f}"))
            fig_w, ax = plt.subplots(figsize=(10,5))
            x = np.arange(len(res['weights']['combined']))
            width = 0.2
            ax.bar(x - 1.5*width, res['weights']['kernel'], width, label='Kernel', alpha=0.7)
            ax.bar(x - 0.5*width, res['weights']['attention'], width, label='Attention', alpha=0.7)
            ax.bar(x + 0.5*width, res['weights']['combined'], width, label='Combined', alpha=0.7)
            ax.bar(x + 1.5*width, res['weights']['gate'], width, label='Gate', alpha=0.7)
            ax.set_xlabel('Source index')
            ax.set_ylabel('Weight')
            ax.set_title('Comparison of weights')
            ax.legend()
            st.pyplot(fig_w)
        
        with tabs[4]:
            st.markdown('<h2 class="section-header">📉 Uncertainty (φ field)</h2>', unsafe_allow_html=True)
            if 'uncertainty_phi' in res['derived']:
                unc_data = np.array(res['derived']['uncertainty_phi'])
                fig_unc = st.session_state.visualizer.create_uncertainty_map(
                    unc_data, title="Standard deviation of φ across sources",
                    cmap_name='hot', L0_nm=L0_nm, target_params=target
                )
                st.pyplot(fig_unc)
            else:
                st.info("Uncertainty not computed.")
        
        with tabs[5]:
            st.markdown('<h2 class="section-header">⏱️ Temporal Attention Visualization</h2>', unsafe_allow_html=True)
            if res['derived'].get('temporal_attention_weights') is not None:
                attn_weights = np.array(res['derived']['temporal_attention_weights'])
                fig_attn = st.session_state.visualizer.create_temporal_attention_heatmap(
                    attn_weights, res['source_params'], target,
                    title="Temporal Attention Weights (Time × Sources)"
                )
                st.pyplot(fig_attn)
                
                st.markdown("#### Time Warping Factors")
                st.write(f"**Target warp factor:** {res['derived']['time_warp_factor']:.3f}")
                st.write("**Source warp factors:**")
                for i, wf in enumerate(res['derived']['source_warp_factors']):
                    st.write(f"- Source {i}: {wf:.3f}")
            else:
                st.info("Temporal attention weights not available (using simple weighted average).")
        
        with tabs[6]:
            st.markdown('<h2 class="section-header">💾 Export Data</h2>', unsafe_allow_html=True)
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📊 Export to JSON", use_container_width=True):
                    export_data = st.session_state.results_manager.prepare_export_data(res, {})
                    json_str, fname = st.session_state.results_manager.export_to_json(export_data)
                    st.download_button("Download JSON", json_str, fname, "application/json")
            with col_exp2:
                if st.button("📈 Export to CSV", use_container_width=True):
                    csv_str, fname = st.session_state.results_manager.export_to_csv(res)
                    st.download_button("Download CSV", csv_str, fname, "text/csv")
            st.markdown("#### Export preview")
            st.json({
                "target_params": res['target_params'],
                "shape": res['shape'],
                "num_sources": res['num_sources'],
                "current_time_norm": res.get('time_norm', 1.0),
                "final_thickness_nm": res['derived']['thickness_nm'],
                "spatial_attention_used": res.get('use_spatial_attention', False),
                "parameter_aware_temporal_used": res.get('use_parameter_aware_temporal', False),
                "time_warping_used": res.get('use_time_warping', False),
                "temporal_kernel_used": res.get('use_temporal_kernel', False)
            })
    else:
        st.info("Load solutions and set target parameters in the sidebar, then click 'Perform Interpolation'.")

if __name__ == "__main__":
    main()
