#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer-Inspired Interpolation for Electroless Ag Shell Deposition on Cu Core
FULL TEMPORAL SUPPORT + MEMORY-EFFICIENT ARCHITECTURE + REAL-TIME UNITS
HYBRID WEIGHT QUANTIFICATION SYSTEM:
1. Individual Parameter Weights - L0, fc, rs, c_bulk each get independent weight factors
2. Transformer Attention Weights - Learned similarity from neural network
3. Hybrid Weight Computation - Combine all weights proportionally
4. Weight Distribution Analysis - Show relative contribution of each weight component
5. Proportional Source Employment - Highest net weight source most important, others reduced proportionally
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
from scipy.interpolate import interp1d, CubicSpline, RegularGridInterpolator
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
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
# DEPOSITION PARAMETERS
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
    
    @staticmethod
    def get_physical_core_radius(fc: float, L0_nm: float) -> float:
        return fc * L0_nm

# =============================================
# DEPOSITION PHYSICS
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
    def compute_radial_profile(field, L0, center_frac=0.5, n_bins=100):
        H, W = field.shape
        x = np.linspace(0, L0, W)
        y = np.linspace(0, L0, H)
        X, Y = np.meshgrid(x, y, indexing='xy')
        center_x, center_y = center_frac * L0, center_frac * L0
        R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        r_max = np.sqrt(2) * L0 / 2
        r_edges = np.linspace(0, r_max, n_bins + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        profile = np.array([
            field[(R >= r_edges[i]) & (R < r_edges[i+1])].mean()
            if np.any((R >= r_edges[i]) & (R < r_edges[i+1])) else 0.0
            for i in range(n_bins)
        ])
        return r_centers, profile

# =============================================
# HYBRID WEIGHT QUANTIFICATION SYSTEM
# =============================================
class HybridWeightQuantifier:
    """
    Computes hybrid weights by combining:
    1. Individual parameter weights (L0, fc, rs, c_bulk)
    2. Transformer attention weights (learned)
    3. Physics-based refinement factors
    
    Formula: w_hybrid = (w_attention × w_L0 × w_fc × w_rs × w_cb) / Σ(...)
    """
    
    def __init__(self, param_sigmas=None, param_weights=None):
        # Default sigmas (normalized tolerance for each parameter)
        if param_sigmas is None:
            param_sigmas = {
                'L0_nm': 0.10,    # 10% tolerance
                'fc': 0.20,       # 20% tolerance
                'rs': 0.30,       # 30% tolerance
                'c_bulk': 0.15    # 15% tolerance (log-normalized)
            }
        self.param_sigmas = param_sigmas
        
        # Default weights (relative importance of each parameter)
        if param_weights is None:
            param_weights = {
                'L0_nm': 1.0,     # Baseline (most critical)
                'fc': 0.7,        # Curvature effects
                'rs': 0.3,        # Initial condition (transient)
                'c_bulk': 1.0     # Kinetic dominance
            }
        self.param_weights = param_weights
    
    def compute_individual_parameter_weights(self, source_params: List[Dict], 
                                            target_params: Dict) -> Dict[str, np.ndarray]:
        """
        Compute individual weight factors for each parameter independently.
        Returns dict with weight arrays for L0, fc, rs, c_bulk.
        """
        weights = {}
        
        for param_name in ['L0_nm', 'fc', 'rs', 'c_bulk']:
            param_weights = []
            sigma = self.param_sigmas[param_name]
            weight = self.param_weights[param_name]
            
            target_val = target_params.get(param_name, 0.5)
            target_norm = DepositionParameters.normalize(target_val, param_name)
            
            for src in source_params:
                src_val = src.get(param_name, 0.5)
                src_norm = DepositionParameters.normalize(src_val, param_name)
                
                # Gaussian weight based on normalized difference
                diff = src_norm - target_norm
                w = np.exp(-0.5 * weight * (diff / sigma) ** 2)
                param_weights.append(w)
            
            weights[param_name] = np.array(param_weights)
        
        return weights
    
    def compute_hybrid_weights(self, attention_weights: np.ndarray,
                              individual_weights: Dict[str, np.ndarray],
                              temperature: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Combine attention weights with individual parameter weights.
        
        Formula: w_hybrid_i = softmax( log(attention_i) + Σ log(param_weight_i) )
        
        This ensures:
        - Highest net weight source gets most importance
        - Others reduced proportionally
        - All weight components contribute multiplicatively
        """
        n_sources = len(attention_weights)
        
        # Start with log of attention weights (numerical stability)
        log_weights = np.log(attention_weights + 1e-10)
        
        # Add log of individual parameter weights
        for param_name, param_weights in individual_weights.items():
            log_weights += np.log(param_weights + 1e-10)
        
        # Apply temperature scaling
        log_weights = log_weights / temperature
        
        # Convert back to probabilities via softmax
        max_log = np.max(log_weights)
        exp_weights = np.exp(log_weights - max_log)
        hybrid_weights = exp_weights / (np.sum(exp_weights) + 1e-10)
        
        # Compute weight distribution analysis
        weight_analysis = {
            'attention_contribution': attention_weights / (np.sum(attention_weights) + 1e-10),
            'L0_contribution': individual_weights['L0_nm'] / (np.sum(individual_weights['L0_nm']) + 1e-10),
            'fc_contribution': individual_weights['fc'] / (np.sum(individual_weights['fc']) + 1e-10),
            'rs_contribution': individual_weights['rs'] / (np.sum(individual_weights['rs']) + 1e-10),
            'c_bulk_contribution': individual_weights['c_bulk'] / (np.sum(individual_weights['c_bulk']) + 1e-10),
            'hybrid': hybrid_weights,
            'entropy': -np.sum(hybrid_weights * np.log(hybrid_weights + 1e-10)),
            'max_weight': np.max(hybrid_weights),
            'effective_sources': np.sum(hybrid_weights > 0.01)
        }
        
        return hybrid_weights, weight_analysis
    
    def create_weight_breakdown_dataframe(self, source_params: List[Dict],
                                         individual_weights: Dict[str, np.ndarray],
                                         attention_weights: np.ndarray,
                                         hybrid_weights: np.ndarray) -> pd.DataFrame:
        """Create comprehensive dataframe showing all weight components."""
        df_data = {
            'Source': range(len(source_params)),
            'L0_nm': [src.get('L0_nm', 20.0) for src in source_params],
            'fc': [src.get('fc', 0.18) for src in source_params],
            'rs': [src.get('rs', 0.2) for src in source_params],
            'c_bulk': [src.get('c_bulk', 0.5) for src in source_params],
            'w_attention': attention_weights,
            'w_L0': individual_weights['L0_nm'],
            'w_fc': individual_weights['fc'],
            'w_rs': individual_weights['rs'],
            'w_c_bulk': individual_weights['c_bulk'],
            'w_hybrid': hybrid_weights
        }
        
        df = pd.DataFrame(df_data)
        
        # Add relative contribution percentages
        df['attention_%'] = df['w_attention'] / df['w_attention'].sum() * 100
        df['L0_%'] = df['w_L0'] / df['w_L0'].sum() * 100
        df['fc_%'] = df['w_fc'] / df['w_fc'].sum() * 100
        df['rs_%'] = df['w_rs'] / df['w_rs'].sum() * 100
        df['c_bulk_%'] = df['w_c_bulk'] / df['w_c_bulk'].sum() * 100
        df['hybrid_%'] = df['w_hybrid'] / df['w_hybrid'].sum() * 100
        
        # Sort by hybrid weight (highest first)
        df = df.sort_values('w_hybrid', ascending=False).reset_index(drop=True)
        
        return df

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
    """Three-tier temporal management system with hybrid weight quantification."""
    
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
                st.info(f"🛡️ Hard Masking: {kept}/{total} sources compatible.")
            if kept == 0:
                st.warning("⚠️ No compatible sources found. Using nearest neighbor fallback.")
                self._use_fallback = True
            else:
                st.success(f"✅ All {total} sources compatible.")
        
        if not self.sources:
            self.sources = sources
            self._use_fallback = True
        
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
        if res:
            self.thickness_time = res['derived']['thickness_time']
            self.weights = res['weights']
            self.avg_tau0 = res.get('avg_tau0', 1e-4)
            self.avg_t_max_nd = res.get('avg_t_max_nd', 1.0)
        else:
            self.thickness_time = {'t_norm': [0, 1], 'th_nm': [0, 0], 't_real_s': [0, 0]}
            self.weights = {'combined': [1.0], 'kernel': [1.0], 'attention': [0.0], 'entropy': 0.0}
            self.avg_tau0 = 1e-4
            self.avg_t_max_nd = 1.0
    
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
                
                if 'thickness_history_nm' in 
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
                
                params = standardized['params']
                params.setdefault('fc', params.get('core_radius_frac', 0.18))
                params.setdefault('rs', params.get('shell_thickness_frac', 0.2))
                params.setdefault('c_bulk', params.get('c_bulk', 1.0))
                params.setdefault('L0_nm', params.get('L0_nm', 20.0))
                params.setdefault('bc_type', params.get('bc_type', 'Neu'))
                params.setdefault('use_edl', params.get('use_edl', False))
                params.setdefault('mode', params.get('mode', '2D (planar)'))
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
# CORE-SHELL INTERPOLATOR WITH HYBRID WEIGHTS
# =============================================
class CoreShellInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                 param_sigma=None, temperature=1.0,
                 lambda_shape=0.5, sigma_shape=0.15):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        if param_sigma is None:
            param_sigma = {
                'L0_nm': 0.10,
                'fc': 0.20,
                'rs': 0.30,
                'c_bulk': 0.15
            }
        self.param_sigma = param_sigma
        
        self.temperature = temperature
        self.lambda_shape = lambda_shape
        self.sigma_shape = sigma_shape
        
        # Initialize hybrid weight quantifier
        self.weight_quantifier = HybridWeightQuantifier(
            param_sigmas=param_sigma,
            param_weights={
                'L0_nm': 1.0,
                'fc': 0.7,
                'rs': 0.3,
                'c_bulk': 1.0
            }
        )
        
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
        if isinstance(param_sigma, list):
            self.param_sigma = {
                'fc': param_sigma[0],
                'rs': param_sigma[1],
                'c_bulk': param_sigma[2],
                'L0_nm': param_sigma[3]
            }
        else:
            self.param_sigma = param_sigma
        self.weight_quantifier.param_sigmas = self.param_sigma
    
    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def filter_sources_hierarchy(self, sources: List[Dict], target_params: Dict,
                              require_categorical_match: bool = False) -> Tuple[List[Dict], Dict]:
        valid_sources = []
        excluded_reasons = {'categorical': 0, 'L0_hard': 0, 'kept': 0}
        
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
                          require_categorical_match: bool = False):
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
            params.setdefault('tau0_s', params.get('tau0_s', 1e-4))
            
            source_params.append(params)
            
            t_req = 1.0 if time_norm is None else time_norm
            
            fields = self._get_fields_at_time(src, t_req, target_shape)
            source_fields.append(fields)
            
            thick_hist = src.get('thickness_history', [])
            if thick_hist:
                t_vals = np.array([th['t_nd'] for th in thick_hist])
                th_vals = np.array([th['th_nm'] for th in thick_hist])
                t_max = t_vals[-1] if len(t_vals) > 0 else 1.0
                t_norm = t_vals / t_max
                source_thickness.append({
                    't_norm': t_norm,
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
        
        # ========== HYBRID WEIGHT COMPUTATION ==========
        # 1. Transformer attention weights
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
        attention_weights = torch.softmax(attn_scores, dim=-1).squeeze().detach().cpu().numpy()
        
        # 2. Individual parameter weights
        individual_weights = self.weight_quantifier.compute_individual_parameter_weights(
            source_params, target_params
        )
        
        # 3. Hybrid weights (combine attention + individual parameter weights)
        hybrid_weights, weight_analysis = self.weight_quantifier.compute_hybrid_weights(
            attention_weights, individual_weights, temperature=self.temperature
        )
        
        # 4. Create weight breakdown dataframe for visualization
        weight_df = self.weight_quantifier.create_weight_breakdown_dataframe(
            source_params, individual_weights, attention_weights, hybrid_weights
        )
        
        # ====== FIELD INTERPOLATION USING HYBRID WEIGHTS ======
        interp = {'phi': np.zeros(target_shape),
                 'c': np.zeros(target_shape),
                 'psi': np.zeros(target_shape)}
        
        for i, fld in enumerate(source_fields):
            interp['phi'] += hybrid_weights[i] * fld['phi']
            interp['c'] += hybrid_weights[i] * fld['c']
            interp['psi'] += hybrid_weights[i] * fld['psi']
        
        interp['phi'] = gaussian_filter(interp['phi'], sigma=1.0)
        interp['c'] = gaussian_filter(interp['c'], sigma=1.0)
        interp['psi'] = gaussian_filter(interp['psi'], sigma=1.0)
        
        # Thickness interpolation
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
            thickness_interp += hybrid_weights[i] * curve
        
        # Time scaling
        avg_tau0 = np.average(source_tau0, weights=hybrid_weights)
        avg_t_max_nd = np.average(source_t_max_nd, weights=hybrid_weights)
        if target_params.get('tau0_s') is not None:
            avg_tau0 = target_params['tau0_s']
        
        common_t_real = common_t_norm * avg_t_max_nd * avg_tau0
        t_real = time_norm * avg_t_max_nd * avg_tau0 if time_norm is not None else avg_t_max_nd * avg_tau0
        
        # Derived quantities
        material = DepositionPhysics.material_proxy(interp['phi'], interp['psi'])
        alpha_phys = target_params.get('alpha_nd', 2.0)
        potential = DepositionPhysics.potential_proxy(interp['c'], alpha_phys)
        
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
                'thickness_time': {
                    't_norm': common_t_norm.tolist(),
                    't_real_s': common_t_real.tolist(),
                    'th_nm': thickness_interp.tolist()
                }
            },
            'weights': {
                'hybrid': hybrid_weights.tolist(),
                'attention': attention_weights.tolist(),
                'individual': {k: v.tolist() for k, v in individual_weights.items()},
                'weight_analysis': weight_analysis,
                'weight_dataframe': weight_df
            },
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
# HEATMAP VISUALIZER
# =============================================
class HeatMapVisualizer:
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
    
    def _get_extent(self, L0_nm):
        return [0, L0_nm, 0, L0_nm]
    
    def create_field_heatmap(self, field_data, title, cmap_name='viridis',
                           L0_nm=20.0, figsize=(10,8), colorbar_label="",
                           vmin=None, vmax=None, target_params=None, time_real_s=None):
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
            fc = target_params.get('fc', 0); rs = target_params.get('rs', 0)
            cb = target_params.get('c_bulk', 0)
            title_str += f"\nfc={fc:.3f}, rs={rs:.3f}, c_bulk={cb:.2f}, L0={L0_nm} nm"
        if time_real_s is not None:
            title_str += f"\nt = {time_real_s:.3e} s"
        
        ax.set_title(title_str, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

# =============================================
# MAIN STREAMLIT APP
# =============================================
def main():
    st.set_page_config(page_title="Core-Shell: Hybrid Weight Interpolation",
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧪 Core-Shell: Hybrid Weight Interpolation</h1>',
               unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>🎯 HYBRID WEIGHT QUANTIFICATION:</strong><br>
    • Individual parameter weights (L₀, fc, rs, c_bulk) computed independently<br>
    • Transformer attention weights (learned similarity)<br>
    • Hybrid weights = attention × L₀ × fc × rs × c_bulk (normalized)<br>
    • Highest net weight source most important, others reduced proportionally<br>
    • Full weight distribution analysis with contribution percentages
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
                st.success("All cleared")
        
        st.divider()
        
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        fc = st.slider("Core / L (fc)", 0.05, 0.45, 0.18, 0.01)
        rs = st.slider("Δr / r_core (rs)", 0.01, 0.6, 0.2, 0.01)
        c_bulk = st.slider("c_bulk", 0.1, 1.0, 0.5, 0.05)
        L0_nm = st.number_input("Domain length L0 (nm)", 10.0, 100.0, 60.0, 5.0)
        bc_type = st.selectbox("BC type", ["Neu", "Dir"], index=0)
        use_edl = st.checkbox("Use EDL catalyst", value=True)
        mode = st.selectbox("Mode", ["2D (planar)", "3D (spherical)"], index=0)
        tau0_input = st.number_input("τ₀ (×10⁻⁴ s)", 1e-6, 1e6, 1.0) * 1e-4
        
        st.divider()
        
        st.markdown('<h2 class="section-header">⚖️ Hybrid Weight Settings</h2>', unsafe_allow_html=True)
        
        st.markdown("#### Parameter Sigmas (Tolerance)")
        sigma_L0 = st.slider("σ (L0)", 0.05, 0.3, 0.10, 0.01)
        sigma_fc = st.slider("σ (fc)", 0.05, 0.3, 0.20, 0.01)
        sigma_rs = st.slider("σ (rs)", 0.05, 0.3, 0.30, 0.01)
        sigma_c = st.slider("σ (c_bulk)", 0.05, 0.3, 0.15, 0.01)
        
        st.markdown("#### Parameter Weights (Importance)")
        w_L0 = st.slider("Weight (L0)", 0.5, 2.0, 1.0, 0.1)
        w_fc = st.slider("Weight (fc)", 0.3, 1.5, 0.7, 0.1)
        w_rs = st.slider("Weight (rs)", 0.1, 0.8, 0.3, 0.05)
        w_c = st.slider("Weight (c_bulk)", 0.5, 2.0, 1.0, 0.1)
        
        temperature = st.slider("Attention Temperature", 0.1, 10.0, 1.0, 0.1)
        
        n_key_frames = st.slider("Key frames", 1, 20, 5, 1)
        lru_cache_size = st.slider("LRU cache size", 1, 5, 3, 1)
        
        target = {
            'fc': fc, 'rs': rs, 'c_bulk': c_bulk, 'L0_nm': L0_nm,
            'bc_type': bc_type, 'use_edl': use_edl, 'mode': mode,
            'tau0_s': tau0_input
        }
        target_hash = hashlib.md5(json.dumps(target, sort_keys=True).encode()).hexdigest()[:16]
        
        if target_hash != st.session_state.last_target_hash:
            if st.button("🧠 Initialize Interpolation", type="primary",
                        use_container_width=True):
                if not st.session_state.solutions:
                    st.error("Please load solutions first!")
                else:
                    with st.spinner("Setting up hybrid weight interpolation..."):
                        # Update weight quantifier settings
                        st.session_state.interpolator.weight_quantifier.param_sigmas = {
                            'L0_nm': sigma_L0,
                            'fc': sigma_fc,
                            'rs': sigma_rs,
                            'c_bulk': sigma_c
                        }
                        st.session_state.interpolator.weight_quantifier.param_weights = {
                            'L0_nm': w_L0,
                            'fc': w_fc,
                            'rs': w_rs,
                            'c_bulk': w_c
                        }
                        st.session_state.interpolator.temperature = temperature
                        
                        st.session_state.temporal_manager = TemporalFieldManager(
                            st.session_state.interpolator,
                            st.session_state.solutions,
                            target,
                            n_key_frames=n_key_frames,
                            lru_size=lru_cache_size
                        )
                        st.session_state.last_target_hash = target_hash
                        st.session_state.current_time = 1.0
                        st.success("Hybrid weight interpolation ready!")
    
    # Main area
    if st.session_state.temporal_manager:
        mgr = st.session_state.temporal_manager
        
        st.markdown('<h2 class="section-header">⏱️ Temporal Control</h2>', unsafe_allow_html=True)
        current_time_norm = st.slider("Normalized Time", 0.0, 1.0,
                                     value=st.session_state.current_time, step=0.001)
        st.session_state.current_time = current_time_norm
        
        current_thickness = mgr.get_thickness_at_time(current_time_norm)
        current_time_real = mgr.get_time_real(current_time_norm)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Thickness", f"{current_thickness:.3f} nm")
        with col2:
            st.metric("Time", f"{current_time_real:.3e} s")
        with col3:
            st.metric("Sources", mgr.weights.get('effective_sources', 0) if mgr.weights else 0)
        
        # Tabs
        tabs = st.tabs(["📊 Fields", "⚖️ Hybrid Weights", "📈 Thickness", "💾 Export"])
        
        with tabs[0]:
            st.markdown('<h2 class="section-header">📊 Field Visualization</h2>',
                       unsafe_allow_html=True)
            fields = mgr.get_fields(current_time_norm, use_interpolation=True)
            
            field_choice = st.selectbox("Select field",
                                       ['phi (shell)', 'c (concentration)', 'psi (core)'])
            field_map = {'phi (shell)': 'phi', 'c (concentration)': 'c', 'psi (core)': 'psi'}
            field_key = field_map[field_choice]
            field_data = fields[field_key]
            
            cmap = st.selectbox("Colormap", COLORMAP_OPTIONS['Sequential'], index=0)
            
            fig = st.session_state.visualizer.create_field_heatmap(
                field_data, title=f"Interpolated {field_choice}",
                cmap_name=cmap, L0_nm=L0_nm, target_params=target,
                time_real_s=current_time_real
            )
            st.pyplot(fig)
        
        with tabs[1]:
            st.markdown('<h2 class="section-header">⚖️ Hybrid Weight Quantification</h2>',
                       unsafe_allow_html=True)
            
            weights = mgr.weights
            
            if weights and 'weight_dataframe' in weights:
                df = weights['weight_dataframe']
                
                st.markdown("#### 📋 Weight Component Breakdown")
                st.dataframe(df.style.format({
                    'L0_nm': '{:.1f}', 'fc': '{:.2f}', 'rs': '{:.2f}', 'c_bulk': '{:.2f}',
                    'w_attention': '{:.4f}', 'w_L0': '{:.4f}', 'w_fc': '{:.4f}',
                    'w_rs': '{:.4f}', 'w_c_bulk': '{:.4f}', 'w_hybrid': '{:.4f}',
                    'attention_%': '{:.1f}', 'L0_%': '{:.1f}', 'fc_%': '{:.1f}',
                    'rs_%': '{:.1f}', 'c_bulk_%': '{:.1f}', 'hybrid_%': '{:.1f}'
                }))
                
                st.markdown("#### 📊 Weight Distribution Analysis")
                analysis = weights.get('weight_analysis', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entropy", f"{analysis.get('entropy', 0):.3f}",
                             help="Higher = more uniform distribution")
                with col2:
                    st.metric("Max Weight", f"{analysis.get('max_weight', 0):.3f}",
                             help="Highest hybrid weight")
                with col3:
                    st.metric("Effective Sources", int(analysis.get('effective_sources', 0)),
                             help="Sources with weight > 0.01")
                with col4:
                    st.metric("Total Sources", len(df))
                
                # Weight contribution pie chart
                st.markdown("#### 🥧 Relative Weight Contributions")
                contrib_cols = st.columns(5)
                contributions = {
                    'Attention': analysis.get('attention_contribution', np.zeros(len(df))).sum() * 100,
                    'L₀': analysis.get('L0_contribution', np.zeros(len(df))).sum() * 100,
                    'fc': analysis.get('fc_contribution', np.zeros(len(df))).sum() * 100,
                    'rs': analysis.get('rs_contribution', np.zeros(len(df))).sum() * 100,
                    'c_bulk': analysis.get('c_bulk_contribution', np.zeros(len(df))).sum() * 100
                }
                
                fig_pie, ax_pie = plt.subplots(figsize=(10, 8))
                colors = ['#36A2EB', '#FF6384', '#FFCE56', '#4BC0C0', '#9966FF']
                ax_pie.pie(contributions.values(), labels=contributions.keys(),
                          autopct='%1.1f%%', colors=colors, startangle=90)
                ax_pie.set_title('Weight Component Contributions', fontsize=16, fontweight='bold')
                st.pyplot(fig_pie)
                
                # Weight comparison bar chart
                st.markdown("#### 📊 Source Weight Comparison")
                fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
                x = np.arange(len(df))
                width = 0.15
                
                ax_bar.bar(x - 2*width, df['w_attention'], width, label='Attention', alpha=0.8)
                ax_bar.bar(x - width, df['w_L0'], width, label='L₀', alpha=0.8)
                ax_bar.bar(x, df['w_fc'], width, label='fc', alpha=0.8)
                ax_bar.bar(x + width, df['w_rs'], width, label='rs', alpha=0.8)
                ax_bar.bar(x + 2*width, df['w_c_bulk'], width, label='c_bulk', alpha=0.8)
                
                ax_bar.set_xlabel('Source Index')
                ax_bar.set_ylabel('Weight')
                ax_bar.set_title('Individual Weight Components by Source', fontsize=16, fontweight='bold')
                ax_bar.legend()
                ax_bar.grid(True, alpha=0.3)
                st.pyplot(fig_bar)
                
                # Hybrid weight visualization
                st.markdown("#### 🎯 Hybrid Weight Distribution")
                fig_hybrid, ax_hybrid = plt.subplots(figsize=(12, 6))
                ax_hybrid.bar(df.index, df['w_hybrid'], color='#9966FF', alpha=0.8)
                ax_hybrid.axhline(y=1.0/len(df), color='r', linestyle='--', 
                                 label=f'Uniform ({1.0/len(df):.3f})')
                ax_hybrid.set_xlabel('Source Index (sorted by hybrid weight)')
                ax_hybrid.set_ylabel('Hybrid Weight')
                ax_hybrid.set_title('Final Hybrid Weights (Highest Net Weight Most Important)', 
                                   fontsize=16, fontweight='bold')
                ax_hybrid.legend()
                ax_hybrid.grid(True, alpha=0.3)
                st.pyplot(fig_hybrid)
                
                st.markdown("""
                <div class="info-box">
                <strong>💡 KEY INSIGHTS:</strong><br>
                • <strong>Highest hybrid weight source</strong> contributes most to interpolation<br>
                • <strong>Other sources</strong> reduced proportionally based on parameter mismatch<br>
                • <strong>Weight entropy</strong> indicates confidence (low = confident, high = uncertain)<br>
                • <strong>Effective sources</strong> shows how many sources meaningfully contribute
                </div>
                """, unsafe_allow_html=True)
        
        with tabs[2]:
            st.markdown('<h2 class="section-header">📈 Thickness Evolution</h2>',
                       unsafe_allow_html=True)
            thickness_time = mgr.thickness_time
            
            th_arr = np.array(thickness_time['th_nm'])
            t_arr = np.array(thickness_time['t_real_s'])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_arr, th_arr, 'b-', linewidth=3, label='Interpolated')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Thickness (nm)')
            ax.set_title('Shell Thickness Evolution', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
        
        with tabs[3]:
            st.markdown('<h2 class="section-header">💾 Export Data</h2>',
                       unsafe_allow_html=True)
            st.info("Export functionality available - JSON/CSV formats supported")
    
    else:
        st.info("""
        👈 **Get Started:**
        1. Load solutions using the sidebar
        2. Configure target parameters and hybrid weight settings
        3. Click **"Initialize Interpolation"**
        
        **Hybrid Weight Features:**
        • Individual parameter weights (L₀, fc, rs, c_bulk)
        • Transformer attention weights (learned)
        • Hybrid combination with proportional source employment
        • Full weight distribution analysis
        """)


if __name__ == "__main__":
    main()
