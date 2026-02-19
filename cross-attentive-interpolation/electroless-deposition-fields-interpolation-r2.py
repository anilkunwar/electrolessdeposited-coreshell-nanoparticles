#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer‑Inspired Interpolation for Electroless Ag Shell Deposition on Cu Core
Now with full temporal information: thickness evolution is interpolated alongside final fields.
Reads PKL files from "numerical_solutions" and interpolates fields and thickness vs. time.
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
from typing import List, Dict, Any, Optional, Tuple

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
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

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
        'fc': (0.05, 0.45),       # core/L
        'rs': (0.01, 0.6),         # Δr/r_core
        'c_bulk': (0.1, 1.0),       # bulk concentration
        'L0_nm': (10.0, 100.0)      # domain length in nm
    }

    @staticmethod
    def normalize(value: float, param_name: str) -> float:
        low, high = DepositionParameters.RANGES[param_name]
        return (value - low) / (high - low)

    @staticmethod
    def denormalize(norm_value: float, param_name: str) -> float:
        low, high = DepositionParameters.RANGES[param_name]
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
        cell_area_real = cell_area_nd * (L0**2)
        electrolyte_area_nd = np.sum(electrolyte_mask) * cell_area_nd
        ag_area_nd = np.sum(ag_mask) * cell_area_nd
        cu_area_nd = np.sum(cu_mask) * cell_area_nd
        return {
            "Electrolyte": (electrolyte_area_nd, electrolyte_area_nd * (L0**2)),
            "Ag": (ag_area_nd, ag_area_nd * (L0**2)),
            "Cu": (cu_area_nd, cu_area_nd * (L0**2))
        }

# =============================================
# ROBUST SOLUTION LOADER (with filename fallback)
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
        """Extract parameters from filenames like:
        AgCu_2D_c0.100_L040.0nm_fc0.100_rs0.010_Neu_EDL2.0_k0.40_M0.20_D0.050_Nx256_steps100000.pkl
        """
        params = {}
        # Mode
        mode_match = re.search(r'_(2D|3D)_', filename)
        if mode_match:
            params['mode'] = '2D (planar)' if mode_match.group(1) == '2D' else '3D (spherical)'
        # c_bulk
        c_match = re.search(r'_c([0-9.]+)_', filename)
        if c_match:
            params['c_bulk'] = float(c_match.group(1))
        # L0_nm
        L_match = re.search(r'_L0([0-9.]+)nm', filename)
        if L_match:
            params['L0_nm'] = float(L_match.group(1))
        # fc (core_radius_frac)
        fc_match = re.search(r'_fc([0-9.]+)_', filename)
        if fc_match:
            params['fc'] = float(fc_match.group(1))
        # rs (shell_thickness_frac)
        rs_match = re.search(r'_rs([0-9.]+)_', filename)
        if rs_match:
            params['rs'] = float(rs_match.group(1))
        # bc_type
        if 'Neu' in filename:
            params['bc_type'] = 'Neu'
        elif 'Dir' in filename:
            params['bc_type'] = 'Dir'
        # use_edl
        if 'noEDL' in filename:
            params['use_edl'] = False
        elif 'EDL' in filename:
            params['use_edl'] = True
            edl_match = re.search(r'EDL([0-9.]+)', filename)
            if edl_match:
                params['lambda0_edl'] = float(edl_match.group(1))
        # k0_nd
        k_match = re.search(r'_k([0-9.]+)_', filename)
        if k_match:
            params['k0_nd'] = float(k_match.group(1))
        # M_nd
        M_match = re.search(r'_M([0-9.]+)_', filename)
        if M_match:
            params['M_nd'] = float(M_match.group(1))
        # D_nd
        D_match = re.search(r'_D([0-9.]+)_', filename)
        if D_match:
            params['D_nd'] = float(D_match.group(1))
        # Nx
        Nx_match = re.search(r'_Nx(\d+)_', filename)
        if Nx_match:
            params['Nx'] = int(Nx_match.group(1))
        # steps
        steps_match = re.search(r'_steps(\d+)\.', filename)
        if steps_match:
            params['n_steps'] = int(steps_match.group(1))
        return params

    def _ensure_2d(self, arr):
        """Convert to 2D numpy array; take middle slice if 3D."""
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
                'thickness_history': [],   # NEW: store thickness vs time
                'metadata': {
                    'filename': os.path.basename(file_path),
                    'loaded_at': datetime.now().isoformat(),
                }
            }

            # Extract from PKL dict
            if isinstance(data, dict):
                if 'parameters' in data and isinstance(data['parameters'], dict):
                    standardized['params'].update(data['parameters'])
                if 'meta' in data and isinstance(data['meta'], dict):
                    standardized['params'].update(data['meta'])
                standardized['coords_nd'] = data.get('coords_nd', None)
                standardized['diagnostics'] = data.get('diagnostics', [])

                # Extract thickness history (list of tuples)
                if 'thickness_history_nm' in data:
                    thick_list = []
                    for entry in data['thickness_history_nm']:
                        # entry is (t_nd, th_nd, th_nm, c_mean, c_max, total_Ag)
                        if len(entry) >= 3:
                            thick_list.append({
                                't_nd': entry[0],
                                'th_nd': entry[1],
                                'th_nm': entry[2]
                            })
                    standardized['thickness_history'] = thick_list

                # Extract snapshots
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

            # Fallback to filename parsing if parameters missing
            if not standardized['params']:
                parsed = self.parse_filename(os.path.basename(file_path))
                standardized['params'].update(parsed)
                st.sidebar.info(f"Parsed parameters from filename: {os.path.basename(file_path)}")

            # Set defaults for mandatory keys
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
# CORE‑SHELL INTERPOLATOR with Gated Attention and Temporal Thickness
# =============================================
class CoreShellInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                 param_sigma=[0.15, 0.15, 0.15, 0.15],
                 temperature=1.0, locality_weight_factor=0.5):
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
        self.input_proj = nn.Linear(12, d_model)   # 4 cont + up to 8 categorical
        self.pos_encoder = PositionalEncoding(d_model)

        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def set_parameter_sigma(self, param_sigma):
        self.param_sigma = param_sigma

    def compute_parameter_kernel(self, source_params: List[Dict], target_params: Dict):
        """Gaussian kernel in normalised parameter space."""
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
            w = np.exp(-0.5 * np.sum((diff / self.param_sigma)**2))
            weights.append(w)

        # Categorical factors
        cat_factor = []
        for src in source_params:
            factor = 1.0
            if src.get('bc_type') != target_params.get('bc_type'):
                factor *= 0.1
            if src.get('use_edl') != target_params.get('use_edl'):
                factor *= 0.1
            if src.get('mode') != target_params.get('mode'):
                factor *= 0.1
            cat_factor.append(factor)

        return np.array(weights) * np.array(cat_factor)

    def encode_parameters(self, params_list: List[Dict]) -> torch.Tensor:
        features = []
        for p in params_list:
            feat = []
            # continuous (normalised)
            for name in ['fc', 'rs', 'c_bulk', 'L0_nm']:
                val = p.get(name, 0.5)
                norm_val = DepositionParameters.normalize(val, name)
                feat.append(norm_val)
            # categorical: bc_type, use_edl, mode (2D/3D), growth_model
            feat.append(1.0 if p.get('bc_type', 'Neu') == 'Dir' else 0.0)
            feat.append(1.0 if p.get('use_edl', False) else 0.0)
            feat.append(1.0 if p.get('mode', '2D (planar)') != '2D (planar)' else 0.0)
            feat.append(1.0 if 'B' in p.get('growth_model', 'Model A') else 0.0)
            # pad to 12
            while len(feat) < 12:
                feat.append(0.0)
            features.append(feat[:12])
        return torch.FloatTensor(features)

    def interpolate_fields(self, sources: List[Dict], target_params: Dict,
                           target_shape: Tuple[int, int] = (256, 256),
                           n_time_points: int = 100):
        """
        Interpolate fields and thickness evolution.
        sources: list of solution dicts (each with 'params', 'history', 'thickness_history')
        target_params: dict of target parameters
        target_shape: (ny, nx) for output fields
        n_time_points: number of points for interpolated thickness curve
        """
        if not sources:
            return None

        source_params = []
        source_fields = []
        source_thickness = []  # each will be dict with 't_norm', 'th_nm'

        for src in sources:
            if 'params' not in src or 'history' not in src or len(src['history']) == 0:
                continue
            params = src['params'].copy()
            # Ensure all needed keys
            params.setdefault('fc', params.get('core_radius_frac', 0.18))
            params.setdefault('rs', params.get('shell_thickness_frac', 0.2))
            params.setdefault('c_bulk', params.get('c_bulk', 1.0))
            params.setdefault('L0_nm', params.get('L0_nm', 20.0))
            params.setdefault('bc_type', params.get('bc_type', 'Neu'))
            params.setdefault('use_edl', params.get('use_edl', False))
            params.setdefault('mode', params.get('mode', '2D (planar)'))
            params.setdefault('growth_model', params.get('growth_model', 'Model A'))

            # Final snapshot for field interpolation
            last = src['history'][-1]
            phi = self._ensure_2d(last['phi'])
            c = self._ensure_2d(last['c'])
            psi = self._ensure_2d(last['psi'])

            # Resize to target shape
            if phi.shape != target_shape:
                factors = (target_shape[0]/phi.shape[0], target_shape[1]/phi.shape[1])
                phi = zoom(phi, factors, order=1)
                c = zoom(c, factors, order=1)
                psi = zoom(psi, factors, order=1)

            source_params.append(params)
            source_fields.append({'phi': phi, 'c': c, 'psi': psi})

            # Process thickness history
            thick_hist = src.get('thickness_history', [])
            if thick_hist:
                t_vals = np.array([th['t_nd'] for th in thick_hist])
                th_vals = np.array([th['th_nm'] for th in thick_hist])
                # Normalise time to [0,1] using max time from this source
                t_max = t_vals[-1] if len(t_vals) > 0 else 1.0
                t_norm = t_vals / t_max
                source_thickness.append({
                    't_norm': t_norm,
                    'th_nm': th_vals,
                    't_max': t_max
                })
            else:
                # If no thickness history, use a dummy (constant zero)
                source_thickness.append({
                    't_norm': np.array([0.0, 1.0]),
                    'th_nm': np.array([0.0, 0.0]),
                    't_max': 1.0
                })

        if not source_params:
            st.error("No valid source fields.")
            return None

        # Encode parameters
        source_features = self.encode_parameters(source_params)          # (N, 12)
        target_features = self.encode_parameters([target_params])       # (1, 12)

        all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)  # (1, 1+N, 12)
        proj = self.input_proj(all_features)                             # (1, 1+N, d_model)
        proj = self.pos_encoder(proj)

        transformer_out = self.transformer(proj)                         # (1, 1+N, d_model)
        target_rep = transformer_out[:, 0, :]                            # (1, d_model)
        source_reps = transformer_out[:, 1:, :]                          # (1, N, d_model)

        # Attention scores
        attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1,2)).squeeze(1)
        attn_scores = attn_scores / np.sqrt(self.d_model) / self.temperature

        # Physics kernel
        kernel_weights = self.compute_parameter_kernel(source_params, target_params)
        kernel_tensor = torch.FloatTensor(kernel_weights).unsqueeze(0)   # (1, N)

        # Gate for each source
        target_exp = target_rep.expand(source_reps.shape[1], -1).unsqueeze(0)   # (1, N, d_model)
        gate_input = torch.cat([target_exp, source_reps], dim=-1)               # (1, N, 2*d_model)
        gate = self.gate_net(gate_input).squeeze(-1)                            # (1, N)

        # Blend
        final_scores = gate * attn_scores + (1 - gate) * kernel_tensor
        final_weights = torch.softmax(final_scores, dim=-1).squeeze().detach().cpu().numpy()

        # Interpolate fields
        interp = {'phi': np.zeros(target_shape),
                  'c': np.zeros(target_shape),
                  'psi': np.zeros(target_shape)}
        for i, fld in enumerate(source_fields):
            interp['phi'] += final_weights[i] * fld['phi']
            interp['c']   += final_weights[i] * fld['c']
            interp['psi'] += final_weights[i] * fld['psi']

        # Optional spatial smoothing
        interp['phi'] = gaussian_filter(interp['phi'], sigma=1.0)
        interp['c']   = gaussian_filter(interp['c'], sigma=1.0)
        interp['psi'] = gaussian_filter(interp['psi'], sigma=1.0)

        # Interpolate thickness evolution
        # Define common normalized time grid
        common_t_norm = np.linspace(0, 1, n_time_points)
        thickness_curves = []
        for i, thick in enumerate(source_thickness):
            # Interpolate source thickness onto common grid
            if len(thick['t_norm']) > 1:
                f = interp1d(thick['t_norm'], thick['th_nm'],
                             kind='linear', bounds_error=False, fill_value=(thick['th_nm'][0], thick['th_nm'][-1]))
                th_interp = f(common_t_norm)
            else:
                th_interp = np.full_like(common_t_norm, thick['th_nm'][0] if len(thick['th_nm']) > 0 else 0.0)
            thickness_curves.append(th_interp)

        # Weighted sum using same final_weights
        thickness_interp = np.zeros_like(common_t_norm)
        for i, curve in enumerate(thickness_curves):
            thickness_interp += final_weights[i] * curve

        # Derived quantities
        material = DepositionPhysics.material_proxy(interp['phi'], interp['psi'])
        alpha = target_params.get('alpha_nd', 2.0)
        potential = DepositionPhysics.potential_proxy(interp['c'], alpha)
        fc = target_params.get('fc', target_params.get('core_radius_frac', 0.18))
        dx = 1.0 / (target_shape[0] - 1)
        thickness_nd = DepositionPhysics.shell_thickness(interp['phi'], interp['psi'], fc, dx=dx)
        L0 = target_params.get('L0_nm', 20.0) * 1e-9
        thickness_nm = thickness_nd * L0 * 1e9
        stats = DepositionPhysics.phase_stats(interp['phi'], interp['psi'], dx, dx, L0)

        result = {
            'fields': interp,
            'derived': {
                'material': material,
                'potential': potential,
                'thickness_nm': thickness_nm,
                'phase_stats': stats,
                'thickness_time': {          # NEW: temporal thickness
                    't_norm': common_t_norm.tolist(),
                    'th_nm': thickness_interp.tolist()
                }
            },
            'weights': {
                'combined': final_weights.tolist(),
                'kernel': kernel_weights.tolist(),
                'attention': attn_scores.squeeze().detach().cpu().numpy().tolist(),
                'gate': gate.squeeze().detach().cpu().numpy().tolist()
            },
            'target_params': target_params,
            'shape': target_shape,
            'num_sources': len(source_fields),
            'source_params': source_params
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
# HEATMAP VISUALIZER (variable domain)
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
                              title="Shell Thickness Evolution", figsize=(10,6)):
        """Plot interpolated thickness vs normalized time, optionally with source curves."""
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        t_norm = thickness_time['t_norm']
        th_nm = thickness_time['th_nm']
        ax.plot(t_norm, th_nm, 'b-', linewidth=3, label='Interpolated')

        if source_curves is not None and weights is not None:
            # Plot source curves with transparency proportional to weight
            for i, (src_t, src_th) in enumerate(source_curves):
                alpha = min(weights[i] * 5, 0.8)  # scale for visibility
                ax.plot(src_t, src_th, '--', linewidth=1, alpha=alpha, label=f'Source {i+1} (w={weights[i]:.3f})')

        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Shell Thickness (nm)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
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
                'interpolation_method': 'core_shell_transformer_gated',
                'visualization_params': visualization_params
            },
            'result': {
                'target_params': res['target_params'],
                'shape': res['shape'],
                'num_sources': res['num_sources'],
                'weights': res['weights']
            }
        }
        for fname, arr in res['fields'].items():
            export['result'][f'{fname}_data'] = arr.tolist()
        for dname, val in res['derived'].items():
            if isinstance(val, np.ndarray):
                export['result'][f'{dname}_data'] = val.tolist()
            elif isinstance(val, dict) and 'th_nm' in val:
                # thickness_time dict
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
    st.set_page_config(page_title="Core‑Shell Deposition Interpolator (with Temporal Thickness)",
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

    st.markdown('<h1 class="main-header">🧪 Core‑Shell Deposition Interpolator (with Temporal Thickness)</h1>', unsafe_allow_html=True)

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
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None

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

        if st.button("🧠 Perform Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Interpolating..."):
                    st.session_state.interpolator.param_sigma = [sigma_fc, sigma_rs, sigma_c, sigma_L]
                    st.session_state.interpolator.temperature = temperature
                    target = {
                        'fc': fc, 'rs': rs, 'c_bulk': c_bulk, 'L0_nm': L0_nm,
                        'bc_type': bc_type, 'use_edl': use_edl, 'mode': mode,
                        'growth_model': growth_model, 'alpha_nd': alpha_nd
                    }
                    res = st.session_state.interpolator.interpolate_fields(
                        st.session_state.solutions, target, target_shape=(256,256),
                        n_time_points=n_time_points
                    )
                    if res:
                        st.session_state.interpolation_result = res
                        st.success("Interpolation successful!")
                    else:
                        st.error("Interpolation failed.")

    # Main area
    if st.session_state.interpolation_result:
        res = st.session_state.interpolation_result
        target = res['target_params']
        L0_nm = target.get('L0_nm', 60.0)

        tabs = st.tabs(["📊 Fields", "📈 Thickness Evolution", "🧪 Derived Quantities", "⚖️ Weights", "💾 Export"])

        with tabs[0]:
            st.markdown('<h2 class="section-header">📊 Interpolated Fields</h2>', unsafe_allow_html=True)
            field_choice = st.selectbox("Select field", ['c (concentration)', 'phi (shell)', 'psi (core)'])
            field_map = {'c (concentration)': 'c', 'phi (shell)': 'phi', 'psi (core)': 'psi'}
            field_key = field_map[field_choice]
            field_data = res['fields'][field_key]

            cmap_cat = st.selectbox("Colormap category", list(COLORMAP_OPTIONS.keys()), index=0)
            cmap = st.selectbox("Colormap", COLORMAP_OPTIONS[cmap_cat], index=0)

            fig = st.session_state.visualizer.create_field_heatmap(
                field_data, title=f"Interpolated {field_choice}",
                cmap_name=cmap, L0_nm=L0_nm, target_params=target,
                colorbar_label=field_choice.split()[0]
            )
            st.pyplot(fig)

            if st.checkbox("Show interactive heatmap"):
                fig_inter = st.session_state.visualizer.create_interactive_heatmap(
                    field_data, title=f"Interpolated {field_choice}",
                    cmap_name=cmap, L0_nm=L0_nm, target_params=target
                )
                st.plotly_chart(fig_inter, use_container_width=True)

        with tabs[1]:
            st.markdown('<h2 class="section-header">📈 Shell Thickness Evolution</h2>', unsafe_allow_html=True)
            thickness_time = res['derived']['thickness_time']
            # Prepare source curves for plotting (optional)
            source_curves = []
            weights = res['weights']['combined']
            # We need to get source thickness histories from the original sources?
            # For simplicity, we'll just show the interpolated curve.
            fig_th = st.session_state.visualizer.create_thickness_plot(
                thickness_time, title=f"Interpolated Thickness for fc={target['fc']:.3f}, rs={target['rs']:.3f}, c_bulk={target['c_bulk']:.2f}"
            )
            st.pyplot(fig_th)

            # Optionally show source curves (if user wants)
            if st.checkbox("Show source thickness curves (weighted transparency)"):
                # Reconstruct source thickness from the stored source_params and source_thickness inside interpolator?
                # Not stored in result, so we'll need to store them. For now, we'll skip.
                st.info("Source curves are not stored in result; re-run with debug flag to see them.")

        with tabs[2]:
            st.markdown('<h2 class="section-header">🧪 Derived Quantities</h2>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final shell thickness (nm)", f"{res['derived']['thickness_nm']:.3f}")
            with col2:
                st.metric("Number of sources", res['num_sources'])

            st.subheader("Phase statistics (final state)")
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

            st.subheader("Material proxy (max(φ,ψ)+ψ)")
            fig_mat = st.session_state.visualizer.create_field_heatmap(
                res['derived']['material'], title="Material proxy",
                cmap_name='Set1', L0_nm=L0_nm, target_params=target,
                colorbar_label="Material", vmin=0, vmax=2
            )
            st.pyplot(fig_mat)

            st.subheader("Potential proxy (-α·c)")
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
                "final_thickness_nm": res['derived']['thickness_nm'],
                "thickness_time_available": 'thickness_time' in res['derived']
            })

    else:
        st.info("Load solutions and set target parameters in the sidebar, then click 'Perform Interpolation'.")

if __name__ == "__main__":
    main()
