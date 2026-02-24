```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer-Inspired Interpolation for Electroless Ag Shell Deposition on Cu Core
FULL TEMPORAL SUPPORT + MEMORY-EFFICIENT ARCHITECTURE + REAL-TIME UNITS
- Physics-aware feature encoding with log-scaled concentration
- Domain-specific kernel with weighted distances and hard categorical constraints
- Three-tier temporal caching: thickness curve (always) + sparse key frames (10) + LRU field cache (3)
- Streaming animation with disk-backed frames for long sequences
- Real-time temporal interpolation between key frames
- **Real physical time (seconds) from source PKL τ₀**
- **Gated attention based on absolute L0 difference** (prioritises sources with similar physical scale)
- **Hierarchical composite gating** (L0 → fc → rs → c_bulk) for improved parameter sensitivity
- **Discrete material colorbar with EXACT phase-field colors**:
  * Electrolyte (red):    RGBA(0.894, 0.102, 0.110, 1.0) → proxy=0: phi≤0.5 AND psi≤0.5
  * Ag shell (orange):    RGBA(1.000, 0.498, 0.000, 1.0) → proxy=1: phi>0.5 AND psi≤0.5
  * Cu core (gray):       RGBA(0.600, 0.600, 0.600, 1.0) → proxy=2: psi>0.5
- **THERMODYNAMICALLY CONSISTENT TIME SCALING**: Normalized time t_nd ∈ [0,1] maps to real time via τ₀ ONLY
  * Domain size L0 affects spatial resolution, NOT temporal dynamics speed
  * No artificial ω speedup factor that violates phase-field kinetics
- **GROUND TRUTH COMPARISON**: Added dashboard to compare interpolation against loaded PKL files with error metrics.
- **ENHANCED L0 PRIORITIZATION**:
  * Sharp L0‑vicinity kernel (Gaussian) with tunable sigma
  * Increased L0 weight in physics kernel from 0.5 → 5.0
  * More aggressive hierarchical L0 gating (hard cut‑off for large mismatches)
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
from scipy.interpolate import interp1d, CubicSpline
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib

# New imports for error metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim

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
# ERROR METRICS HELPER
# =============================================
def calculate_error_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """
    Computes pixel-wise error metrics between ground truth and prediction.
    Handles spatial dimension mismatch by raising an error (caller must resize).
    """
    if ground_truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: GT {ground_truth.shape} vs Pred {prediction.shape}")

    # Flatten for MSE/MAE
    gt_flat = ground_truth.flatten()
    pred_flat = prediction.flatten()

    mse = mean_squared_error(gt_flat, pred_flat)
    mae = mean_absolute_error(gt_flat, pred_flat)
    max_err = np.max(np.abs(ground_truth - prediction))

    # SSIM calculation
    # Determine data range. If both are constant (unlikely), default to 1.0
    data_range = ground_truth.max() - ground_truth.min()
    if data_range < 1e-6:
        data_range = prediction.max() - prediction.min()
    if data_range < 1e-6:
        data_range = 1.0

    try:
        # Use win_size based on smaller dimension to avoid errors on small images
        win_size = min(7, ground_truth.shape[0] // 4, ground_truth.shape[1] // 4)
        if win_size % 2 == 0: win_size -= 1
        if win_size < 3: win_size = 3
        
        ssim_val = ssim(ground_truth, prediction, data_range=data_range, win_size=win_size)
    except Exception as e:
        st.warning(f"SSIM calculation failed ({e}). Returning NaN.")
        ssim_val = np.nan

    return {
        "MSE": mse,
        "MAE": mae,
        "Max Error": max_err,
        "SSIM": ssim_val
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
# MEMORY-EFFICIENT TEMPORAL CACHE SYSTEM
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
    Three-tier temporal management system.
    KEY PRINCIPLE: Normalized time t_nd ∈ [0,1] maps to real time via τ₀ ONLY.
    Domain size L0 affects spatial resolution, NOT temporal dynamics speed.
    """
    def __init__(self, interpolator, sources: List[Dict], target_params: Dict,
                 n_key_frames: int = 10, lru_size: int = 3):
        self.interpolator = interpolator
        self.sources = sources
        self.target_params = target_params
        self.n_key_frames = n_key_frames
        self.lru_size = lru_size
        self.avg_tau0 = None
        self.avg_t_max_nd = None
        self.thickness_time: Optional[Dict] = None
        self.weights: Optional[Dict] = None
        self._compute_thickness_curve()
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
            n_time_points=100, time_norm=None
        )
        self.thickness_time = res['derived']['thickness_time']
        self.weights = res['weights']
        self.avg_tau0 = res.get('avg_tau0', 1e-4)
        self.avg_t_max_nd = res.get('avg_t_max_nd', 1.0)

    def _precompute_key_frames(self):
        st.info(f"Pre-computing {self.n_key_frames} key frames...")
        progress_bar = st.progress(0)
        for i, t in enumerate(self.key_times):
            res = self.interpolator.interpolate_fields(
                self.sources, self.target_params, target_shape=(256, 256),
                n_time_points=100, time_norm=t
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
        t_key = round(time_norm, 4)
        time_real = time_norm * self.avg_t_max_nd * self.avg_tau0 if self.avg_t_max_nd else 0.0

        if t_key in self.lru_cache:
            entry = self.lru_cache.pop(t_key)
            self.lru_cache[t_key] = entry
            return entry.fields

        if t_key in self.key_frames:
            fields = self.key_frames[t_key]
            self._add_to_lru(t_key, fields, self.key_thickness.get(t_key, 0.0), time_real)
            return fields

        if use_interpolation and self.key_frames:
            key_times_arr = np.array(list(self.key_frames.keys()))
            idx = np.searchsorted(key_times_arr, t_key)
            if idx == 0:
                fields = self.key_frames[key_times_arr[0]]
                self._add_to_lru(t_key, fields, self.key_thickness[key_times_arr[0]], time_real)
                return fields
            elif idx >= len(key_times_arr):
                fields = self.key_frames[key_times_arr[-1]]
                self._add_to_lru(t_key, fields, self.key_thickness[key_times_arr[-1]], time_real)
                return fields

            t0, t1 = key_times_arr[idx-1], key_times_arr[idx]
            alpha = (t_key - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
            f0, f1 = self.key_frames[t0], self.key_frames[t1]
            th0, th1 = self.key_thickness[t0], self.key_thickness[t1]

            interp_fields = {}
            for key in f0:
                interp_fields[key] = (1 - alpha) * f0[key] + alpha * f1[key]
            interp_thickness = (1 - alpha) * th0 + alpha * th1
            self._add_to_lru(t_key, interp_fields, interp_thickness, time_real)
            return interp_fields

        res = self.interpolator.interpolate_fields(
            self.sources, self.target_params, target_shape=(256, 256),
            n_time_points=100, time_norm=time_norm
        )
        if res:
            self._add_to_lru(t_key, res['fields'], res['derived']['thickness_nm'], time_real)
            return res['fields']

        nearest_t = min(self.key_frames.keys(), key=lambda x: abs(x - t_key))
        return self.key_frames[nearest_t]

    def _add_to_lru(self, time_norm: float, fields: Dict[str, np.ndarray], thickness_nm: float, time_real_s: float):
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
            import shutil
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
# ENHANCED CORE‑SHELL INTERPOLATOR WITH HIERARCHICAL GATED ATTENTION
# AND STRONG L0 PRIORITIZATION
# =============================================
class CoreShellInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                 param_sigma=None, temperature=1.0, gating_mode="Hierarchical: L0 → fc → rs → c_bulk",
                 l0_sigma_nm=5.0):   # <-- NEW: L0 vicinity kernel width (nm)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        if param_sigma is None:
            param_sigma = [0.15, 0.15, 0.15, 0.15]
        self.param_sigma = param_sigma
        self.temperature = temperature
        self.gating_mode = gating_mode
        self.l0_sigma_nm = l0_sigma_nm   # <-- NEW

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

    def set_l0_sigma_nm(self, l0_sigma_nm):   # <-- NEW setter
        self.l0_sigma_nm = l0_sigma_nm

    def compute_composite_gates(self, source_params: List[Dict], target_params: Dict) -> List[float]:
        """
        Compute composite gate factors based on hierarchical gating modes.
        Returns a list of multiplicative factors (one per source).
        """
        target_L0 = target_params.get('L0_nm', 20.0)
        target_fc = target_params.get('fc', 0.18)
        target_rs = target_params.get('rs', 0.2)
        target_c_bulk = target_params.get('c_bulk', 0.5)

        gates = []
        for src in source_params:
            src_L0 = src.get('L0_nm', 20.0)
            src_fc = src.get('fc', 0.18)
            src_rs = src.get('rs', 0.2)
            src_c_bulk = src.get('c_bulk', 0.5)

            delta_L0 = abs(target_L0 - src_L0)
            delta_fc = abs(target_fc - src_fc)
            delta_rs = abs(target_rs - src_rs)
            delta_c_bulk = abs(target_c_bulk - src_c_bulk)

            if self.gating_mode == "No Gating":
                gate = 1.0

            elif self.gating_mode == "Joint Multiplicative":
                gate = 1.0
                # L0 gate
                if delta_L0 < 5:
                    gate *= 0.95
                elif delta_L0 < 10:
                    gate *= 0.60
                elif delta_L0 < 15:
                    gate *= 0.40
                elif delta_L0 < 25:
                    gate *= 0.20
                else:
                    gate *= 0.05
                # fc gate
                if delta_fc < 0.05:
                    gate *= 0.95
                else:
                    gate *= 0.60
                # rs gate
                if delta_rs < 0.05:
                    gate *= 0.95
                else:
                    gate *= 0.60
                # c_bulk gate (log‑sensitive)
                if delta_c_bulk < 0.05:
                    gate *= 0.95
                else:
                    gate *= 0.60

            elif self.gating_mode == "Hierarchical: L0 → fc → rs → c_bulk":
                gate = 1.0
                # --- Root L0 gate (now more aggressive) ---
                if delta_L0 < 3:
                    gate *= 1.0
                elif delta_L0 < 8:
                    gate *= 0.8
                elif delta_L0 < 15:
                    gate *= 0.2
                else:
                    gate *= 0.0   # hard cut‑off (will be floored later)

                # Only apply sub‑gates if L0 mismatch is not too severe
                if gate > 0.5:
                    if delta_fc < 0.05:
                        gate *= 0.95
                    else:
                        gate *= 0.60
                    if delta_rs < 0.05:
                        gate *= 0.95
                    else:
                        gate *= 0.60
                    if delta_c_bulk < 0.05:
                        gate *= 0.95
                    else:
                        gate *= 0.60

            elif self.gating_mode == "Hierarchical-Parallel: L0 → (fc, rs, c_bulk)":
                gate = 1.0
                # Root L0 gate
                if delta_L0 < 3:
                    gate *= 1.0
                elif delta_L0 < 8:
                    gate *= 0.8
                elif delta_L0 < 15:
                    gate *= 0.2
                else:
                    gate *= 0.0

                # Parallel sub‑gates, independent
                if gate > 0.5:
                    fc_gate = 0.95 if delta_fc < 0.05 else 0.60
                    rs_gate = 0.95 if delta_rs < 0.05 else 0.60
                    c_gate = 0.95 if delta_c_bulk < 0.05 else 0.60
                    gate *= fc_gate * rs_gate * c_gate

            gates.append(max(gate, 1e-6))  # floor at 1e-6 to avoid complete zero

        return gates

    def compute_parameter_kernel(self, source_params: List[Dict], target_params: Dict):
        """Domain‑specific kernel with weighted Euclidean distance and hierarchical gates,
           plus a sharp L0‑vicinity kernel."""
        # --- Increased weight for L0 from 0.5 → 5.0 ---
        phys_weights = {'fc': 2.0, 'rs': 1.0, 'c_bulk': 3.0, 'L0_nm': 5.0}

        def norm_val(params, name):
            val = params.get(name, 0.5)
            return DepositionParameters.normalize(val, name)

        target_norm = np.array([
            norm_val(target_params, 'fc'),
            norm_val(target_params, 'rs'),
            norm_val(target_params, 'c_bulk'),
            norm_val(target_params, 'L0_nm')
        ])

        base_weights = []
        for src in source_params:
            src_norm = np.array([
                norm_val(src, 'fc'),
                norm_val(src, 'rs'),
                norm_val(src, 'c_bulk'),
                norm_val(src, 'L0_nm')
            ])
            diff = src_norm - target_norm
            weighted_sq = sum(phys_weights[p] * (d / self.param_sigma[i])**2
                              for i, (p, d) in enumerate(zip(['fc','rs','c_bulk','L0_nm'], diff)))
            w = np.exp(-0.5 * weighted_sq)
            base_weights.append(w)

        cat_factor = []
        for src in source_params:
            factor = 1.0
            if src.get('bc_type') != target_params.get('bc_type'): factor *= 1e-6
            if src.get('use_edl') != target_params.get('use_edl'): factor *= 1e-6
            if src.get('mode') != target_params.get('mode'): factor *= 1e-6
            cat_factor.append(factor)

        # Compute hierarchical composite gates
        composite_gates = self.compute_composite_gates(source_params, target_params)

        # --- NEW: Sharp L0 vicinity kernel (Gaussian based on absolute nm difference) ---
        target_L0 = target_params.get('L0_nm', 20.0)
        l0_kernel = []
        for src in source_params:
            delta_L0 = abs(target_L0 - src.get('L0_nm', 20.0))
            # Gaussian with sigma = self.l0_sigma_nm (slider)
            l0_kernel.append(np.exp(-0.5 * (delta_L0 / self.l0_sigma_nm)**2))
        l0_kernel = np.array(l0_kernel)

        # Combine all factors (multiply)
        final_weights = np.array(base_weights) * np.array(cat_factor) * np.array(composite_gates) * l0_kernel

        return final_weights

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
                c   = (1 - alpha) * c1   + alpha * c2
                psi = (1 - alpha) * psi1 + alpha * psi2

        if phi.shape != target_shape:
            factors = (target_shape[0]/phi.shape[0], target_shape[1]/phi.shape[1])
            phi = zoom(phi, factors, order=1)
            c   = zoom(c,   factors, order=1)
            psi = zoom(psi, factors, order=1)

        return {'phi': phi, 'c': c, 'psi': psi}

    def interpolate_fields(self, sources: List[Dict], target_params: Dict,
                           target_shape: Tuple[int, int] = (256, 256),
                           n_time_points: int = 100,
                           time_norm: Optional[float] = None):
        if not sources:
            return None

        source_params = []
        source_fields = []
        source_thickness = []
        source_tau0 = []
        source_t_max_nd = []

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
            params.setdefault('tau0_s', params.get('tau0_s', 1e-4))

            source_params.append(params)

            if time_norm is None:
                t_req = 1.0
            else:
                t_req = time_norm

            fields = self._get_fields_at_time(src, t_req, target_shape)
            source_fields.append(fields)

            thick_hist = src.get('thickness_history', [])
            if thick_hist:
                t_vals = np.array([th['t_nd'] for th in thick_hist])
                th_vals = np.array([th['th_nm'] for th in thick_hist])
                t_max = t_vals[-1] if len(t_vals) > 0 else 1.0
                t_norm_arr = t_vals / t_max
                source_thickness.append({
                    't_norm': t_norm_arr,
                    'th_nm': th_vals,
                    't_max': t_max
                })
                source_t_max_nd.append(t_max)
            else:
                source_thickness.append({
                    't_norm': np.array([0.0, 1.0]),
                    'th_nm': np.array([0.0, 0.0]),
                    't_max': 1.0
                })
                source_t_max_nd.append(1.0)

            source_tau0.append(params['tau0_s'])

        if not source_params:
            st.error("No valid source fields.")
            return None

        source_features = self.encode_parameters(source_params)
        target_features = self.encode_parameters([target_params])

        all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
        proj = self.input_proj(all_features)
        proj = self.pos_encoder(proj)

        transformer_out = self.transformer(proj)
        target_rep = transformer_out[:, 0, :]
        source_reps = transformer_out[:, 1:, :]

        attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1,2)).squeeze(1)
        attn_scores = attn_scores / np.sqrt(self.d_model) / self.temperature

        kernel_weights = self.compute_parameter_kernel(source_params, target_params)
        kernel_tensor = torch.FloatTensor(kernel_weights).unsqueeze(0)

        final_scores = attn_scores * kernel_tensor
        final_weights = torch.softmax(final_scores, dim=-1).squeeze().detach().cpu().numpy()

        eps = 1e-10
        entropy = -np.sum(final_weights * np.log(final_weights + eps))

        interp = {'phi': np.zeros(target_shape),
                  'c': np.zeros(target_shape),
                  'psi': np.zeros(target_shape)}
        for i, fld in enumerate(source_fields):
            interp['phi'] += final_weights[i] * fld['phi']
            interp['c']   += final_weights[i] * fld['c']
            interp['psi'] += final_weights[i] * fld['psi']

        interp['phi'] = gaussian_filter(interp['phi'], sigma=1.0)
        interp['c']   = gaussian_filter(interp['c'], sigma=1.0)
        interp['psi'] = gaussian_filter(interp['psi'], sigma=1.0)

        common_t_norm = np.linspace(0, 1, n_time_points)
        thickness_curves = []
        for i, thick in enumerate(source_thickness):
            if len(thick['t_norm']) > 1:
                f = interp1d(thick['t_norm'], thick['th_nm'],
                             kind='linear', bounds_error=False, fill_value=(thick['th_nm'][0], thick['th_nm'][-1]))
                th_interp = f(common_t_norm)
            else:
                th_interp = np.full_like(common_t_norm, thick['th_nm'][0] if len(thick['th_nm']) > 0 else 0.0)
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
        alpha = target_params.get('alpha_nd', 2.0)
        potential = DepositionPhysics.potential_proxy(interp['c'], alpha)
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
                'kernel': kernel_weights.tolist(),
                'attention': attn_scores.squeeze().detach().cpu().numpy().tolist(),
                'entropy': float(entropy)
            },
            'target_params': target_params,
            'shape': target_shape,
            'num_sources': len(source_fields),
            'source_params': source_params,
            'time_norm': t_req,
            'time_real_s': t_real,
            'avg_tau0': avg_tau0,
            'avg_t_max_nd': avg_t_max_nd
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
# ENHANCED HEATMAP VISUALIZER WITH EXACT PHASE-FIELD MATERIAL COLORS
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
                'interpolation_method': 'core_shell_temporal_transformer',
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
# MAIN STREAMLIT APP
# =============================================
def main():
    st.set_page_config(page_title="Core‑Shell Deposition: Full Temporal Interpolation",
                       layout="wide", page_icon="🧪", initial_sidebar_state="expanded")

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
                import shutil
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

        # NEW: L0 vicinity kernel sigma (nm)
        l0_sigma_nm = st.slider("L0 kernel σ (nm)", 1.0, 20.0, 5.0, 0.5,
                                 help="Width of Gaussian L0‑vicinity kernel. Smaller = stronger L0 prioritisation.")

        # Gating mode selection
        gating_mode = st.selectbox(
            "Composite Gating Mode",
            ["Hierarchical: L0 → fc → rs → c_bulk",
             "Hierarchical-Parallel: L0 → (fc, rs, c_bulk)",
             "Joint Multiplicative",
             "No Gating"],
            index=0,
            help="Hierarchical modes apply L0 gate first, then sub‑gates only if L0 is close. Joint multiplies all gates independently."
        )

        n_key_frames = st.slider("Key frames for temporal interpolation", 1, 20, 5, 1,
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
                        # Pass all new settings to the interpolator
                        st.session_state.interpolator.set_parameter_sigma([sigma_fc, sigma_rs, sigma_c, sigma_L])
                        st.session_state.interpolator.temperature = temperature
                        st.session_state.interpolator.set_gating_mode(gating_mode)
                        st.session_state.interpolator.set_l0_sigma_nm(l0_sigma_nm)   # <-- NEW

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
                       "🎬 Animation", "🧪 Derived Quantities", "⚖️ Weights", 
                       "💾 Export", "🔍 Ground Truth Comparison"])

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
                'Kernel': weights['kernel'],
                'Attention': weights['attention']
            })
            st.dataframe(df_weights.style.format("{:.4f}"))

            entropy = weights.get('entropy', 0.0)
            st.metric("Weight Entropy (Uncertainty)", f"{entropy:.4f}",
                     help="Higher = more uncertain (sources contribute equally)")

            fig_w, ax = plt.subplots(figsize=(10,5))
            x = np.arange(len(weights['combined']))
            width = 0.25
            ax.bar(x - width, weights['kernel'], width, label='Kernel (physics)', alpha=0.7)
            ax.bar(x, weights['attention'], width, label='Attention (learned)', alpha=0.7)
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

        # =============================================
        # NEW TAB: Ground Truth Comparison
        # =============================================
        with tabs[6]:
            st.markdown('<h2 class="section-header">🔍 Ground Truth Comparison</h2>', unsafe_allow_html=True)
            st.info("Compare the interpolated result against a selected Ground Truth PKL file at the current time step.")

            if not st.session_state.solutions:
                st.warning("No solutions loaded. Please load PKL files first to enable comparison.")
            else:
                # Filter solutions suitable for comparison (optional, but good for UX)
                # We allow all, but we can sort by parameter closeness if desired.
                # For simplicity, we list all loaded files.
                
                gt_options = [f"{i}: {sol['metadata']['filename']}" for i, sol in enumerate(st.session_state.solutions)]
                selected_gt_idx = st.selectbox("Select Ground Truth File", range(len(st.session_state.solutions)), format_func=lambda x: gt_options[x])
                
                gt_solution = st.session_state.solutions[selected_gt_idx]
                
                st.markdown(f"**Selected File:** `{gt_solution['metadata']['filename']}`")
                st.markdown(f"**Parameters:** fc={gt_solution['params'].get('fc', 0):.3f}, rs={gt_solution['params'].get('rs', 0):.3f}, c={gt_solution['params'].get('c_bulk', 0):.2f}, L0={gt_solution['params'].get('L0_nm', 0):.1f}nm")

                # Get interpolated fields at current time
                interp_fields = mgr.get_fields(current_time_norm, use_interpolation=True)
                interp_shape = interp_fields['phi'].shape

                # Retrieve Ground Truth fields at the same normalized time
                # Need to interpolate within the GT history if exact time doesn't exist
                gt_history = gt_solution.get('history', [])
                
                if not gt_history:
                    st.error("Selected ground truth file has no history/snapshots.")
                else:
                    # Find t_max for GT
                    gt_t_max = gt_solution.get('thickness_history', [{}])[-1].get('t_nd', 1.0)
                    if gt_t_max == 0: gt_t_max = 1.0
                    
                    target_t_gt = current_time_norm * gt_t_max
                    
                    # Extract closest frames
                    t_vals = np.array([s['t_nd'] for s in gt_history])
                    
                    if len(gt_history) == 1:
                        snap = gt_history[0]
                        gt_phi = snap['phi']
                        gt_c = snap['c']
                        gt_psi = snap['psi']
                    else:
                        # Interpolate linearly between snapshots
                        if target_t_gt <= t_vals[0]:
                            snap = gt_history[0]
                            gt_phi, gt_c, gt_psi = snap['phi'], snap['c'], snap['psi']
                        elif target_t_gt >= t_vals[-1]:
                            snap = gt_history[-1]
                            gt_phi, gt_c, gt_psi = snap['phi'], snap['c'], snap['psi']
                        else:
                            idx = np.searchsorted(t_vals, target_t_gt) - 1
                            idx = max(0, min(idx, len(gt_history)-2))
                            t1, t2 = t_vals[idx], t_vals[idx+1]
                            alpha = (target_t_gt - t1) / (t2 - t1) if t2 > t1 else 0.0
                            
                            s1, s2 = gt_history[idx], gt_history[idx+1]
                            gt_phi = (1 - alpha) * s1['phi'] + alpha * s2['phi']
                            gt_c   = (1 - alpha) * s1['c']   + alpha * s2['c']
                            gt_psi = (1 - alpha) * s1['psi'] + alpha * s2['psi']

                    # Ensure GT fields are 2D
                    def ensure_2d(arr):
                        if arr.ndim == 3: return arr[arr.shape[0]//2, :, :]
                        return arr
                    
                    gt_phi = ensure_2d(gt_phi)
                    gt_c = ensure_2d(gt_c)
                    gt_psi = ensure_2d(gt_psi)

                    # Resize GT fields to match Interpolation shape
                    if gt_phi.shape != interp_shape:
                        factors = (interp_shape[0]/gt_phi.shape[0], interp_shape[1]/gt_phi.shape[1])
                        gt_phi = zoom(gt_phi, factors, order=1)
                        gt_c   = zoom(gt_c,   factors, order=1)
                        gt_psi = zoom(gt_psi, factors, order=1)

                    # Compute Derived Fields for both
                    gt_material = DepositionPhysics.material_proxy(gt_phi, gt_psi)
                    interp_material = DepositionPhysics.material_proxy(interp_fields['phi'], interp_fields['psi'])

                    # Define fields to compare
                    comparison_set = {
                        'Phi (Shell Phase)': {'gt': gt_phi, 'interp': interp_fields['phi'], 'cmap': 'viridis'},
                        'Concentration (c)': {'gt': gt_c, 'interp': interp_fields['c'], 'cmap': 'plasma'},
                        'Psi (Core Phase)': {'gt': gt_psi, 'interp': interp_fields['psi'], 'cmap': 'inferno'},
                        'Material Proxy': {'gt': gt_material, 'interp': interp_material, 'cmap': 'Set1', 'is_material': True}
                    }

                    selected_field_name = st.selectbox("Select Field for Comparison", list(comparison_set.keys()))
                    field_data = comparison_set[selected_field_name]
                    
                    gt_f = field_data['gt']
                    interp_f = field_data['interp']
                    field_cmap = field_data['cmap']
                    is_mat = field_data.get('is_material', False)

                    # Compute Metrics
                    try:
                        metrics = calculate_error_metrics(gt_f, interp_f)
                    except ValueError as e:
                        st.error(f"Error computing metrics: {e}")
                        metrics = {}

                    if metrics:
                        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                        c_m1.metric("MSE", f"{metrics['MSE']:.2e}")
                        c_m2.metric("MAE", f"{metrics['MAE']:.2e}")
                        c_m3.metric("Max Error", f"{metrics['Max Error']:.2e}")
                        c_m4.metric("SSIM", f"{metrics['SSIM']:.4f}")

                    # Compute Difference Map
                    diff_map = gt_f - interp_f
                    
                    # Determine symmetric limits for difference plot
                    max_diff = np.max(np.abs(diff_map))
                    if max_diff == 0: max_diff = 1.0

                    # Visualizations
                    col_v1, col_v2, col_v3 = st.columns(3)

                    with col_v1:
                        st.markdown("**Ground Truth**")
                        fig_gt = st.session_state.visualizer.create_field_heatmap(
                            gt_f, f"GT: {selected_field_name}",
                            cmap_name=field_cmap, L0_nm=L0_nm,
                            target_params=gt_solution['params'],
                            time_real_s=target_t_gt * gt_solution['params'].get('tau0_s', 1e-4)
                        )
                        st.pyplot(fig_gt)

                    with col_v2:
                        st.markdown("**Interpolated**")
                        fig_int = st.session_state.visualizer.create_field_heatmap(
                            interp_f, f"Interp: {selected_field_name}",
                            cmap_name=field_cmap, L0_nm=L0_nm,
                            target_params=target,
                            time_real_s=current_time_real
                        )
                        st.pyplot(fig_int)

                    with col_v3:
                        st.markdown("**Difference (GT - Interp)**")
                        fig_diff = st.session_state.visualizer.create_field_heatmap(
                            diff_map, "Error Map",
                            cmap_name='RdBu', # Diverging colormap
                            L0_nm=L0_nm,
                            vmin=-max_diff, vmax=max_diff,
                            target_params=None,
                            colorbar_label="Difference"
                        )
                        st.pyplot(fig_diff)
                    
                    # Histogram of errors
                    st.subheader("Error Distribution")
                    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                    ax_hist.hist(diff_map.flatten(), bins=50, color='gray', alpha=0.7)
                    ax_hist.axvline(0, color='red', linestyle='--')
                    ax_hist.set_title(f"Distribution of Errors (Mean: {np.mean(diff_map):.2e})")
                    ax_hist.set_xlabel("Pixel Value Difference")
                    ax_hist.set_ylabel("Frequency")
                    st.pyplot(fig_hist)

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
```
