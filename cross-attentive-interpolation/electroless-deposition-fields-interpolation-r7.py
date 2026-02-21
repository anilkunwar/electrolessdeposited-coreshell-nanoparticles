#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer‑Inspired Interpolation for Electroless Ag Shell Deposition on Cu Core
FULL TEMPORAL SUPPORT + PHYSICS‑AWARE ENHANCEMENTS + MEMORY‑EFFICIENT STRATEGIES

Features:
- Physics‑aware feature encoding (log‑scaled concentration)
- Domain‑specific kernel with weighted distances and hard categorical constraints
- Kernel applied as multiplicative bias on attention (no learned gate)
- Uncertainty estimate (weight entropy)
- Temporal interpolation at any normalized time
- Memory-efficient temporal management (LRU cache, sparse key frames, streaming)
- Animation streaming with disk-based frame storage
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
import time
import tempfile
import shutil
from functools import lru_cache
from collections import OrderedDict
import gc
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
        # For concentration, apply log scaling first to capture exponential effects
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
# MEMORY-EFFICIENT TEMPORAL CACHE (LRU Implementation)
# =============================================
class TemporalFieldCache:
    """
    LRU cache that only stores recent field interpolations.
    Strategy 1: Lazy Field Loading with LRU Cache
    """
    def __init__(self, maxsize=3):  # Only keep 3 time points in memory
        self.maxsize = maxsize
        self._cache = OrderedDict()
        self._access_order = []
        self._memory_bytes = 0
        self._max_memory_bytes = 15 * 1024 * 1024  # 15 MB limit
    
    def _estimate_size(self, value):
        """Estimate memory size of cached value"""
        if isinstance(value, dict):
            size = 0
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    size += v.nbytes
                elif isinstance(v, dict):
                    size += self._estimate_size(v)
            return size
        return 0
    
    def get(self, key):
        if key in self._cache:
            # Move to front (most recent)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key, value):
        if key in self._cache:
            self._access_order.remove(key)
            old_size = self._estimate_size(self._cache[key])
            self._memory_bytes -= old_size
        elif len(self._cache) >= self.maxsize:
            # Evict oldest
            oldest = self._access_order.pop(0)
            old_size = self._estimate_size(self._cache[oldest])
            self._memory_bytes -= old_size
            del self._cache[oldest]
        
        self._cache[key] = value
        self._access_order.append(key)
        self._memory_bytes += self._estimate_size(value)
    
    def clear(self):
        self._cache.clear()
        self._access_order.clear()
        self._memory_bytes = 0
    
    def get_stats(self):
        return {
            'cached_entries': len(self._cache),
            'memory_bytes': self._memory_bytes,
            'memory_mb': self._memory_bytes / (1024 * 1024)
        }

# =============================================
# STREAMING ANIMATION (Memory-Mapped Field Storage)
# =============================================
class StreamingAnimation:
    """
    Pre-computes frames to disk, streams to UI without loading all into RAM.
    Strategy 3: Memory-Mapped Field Storage for Animation
    """
    def __init__(self, interpolator, sources, target, n_frames=50, target_shape=(256, 256)):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.frame_files = []
        self.n_frames = n_frames
        self.target_shape = target_shape
        self.interpolator = interpolator
        self.sources = sources
        self.target = target
        self._precompute_to_disk()
    
    def _precompute_to_disk(self):
        """Pre-compute to disk (not memory)"""
        times = np.linspace(0, 1, self.n_frames)
        for i, t in enumerate(times):
            try:
                res = self.interpolator.interpolate_fields(
                    self.sources, self.target, 
                    target_shape=self.target_shape, 
                    time_norm=t
                )
                # Save to temp file, keep only path
                frame_path = os.path.join(self.temp_dir.name, f"frame_{i:04d}.npz")
                np.savez_compressed(frame_path,
                    phi=res['fields']['phi'],
                    c=res['fields']['c'],
                    psi=res['fields']['psi'],
                    thickness_nm=res['derived']['thickness_nm'],
                    time_norm=t
                )
                self.frame_files.append(frame_path)
            except Exception as e:
                st.warning(f"Frame {i} failed: {e}")
                self.frame_files.append(None)
    
    def get_frame(self, idx):
        """Load single frame on demand"""
        if idx >= len(self.frame_files) or self.frame_files[idx] is None:
            return None
        data = np.load(self.frame_files[idx])
        return {
            'phi': data['phi'],
            'c': data['c'],
            'psi': data['psi'],
            'thickness_nm': float(data['thickness_nm']),
            'time_norm': float(data['time_norm'])
        }
    
    def cleanup(self):
        self.temp_dir.cleanup()
        self.frame_files = []
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass

# =============================================
# SPARSE TEMPORAL INTERPOLATOR
# =============================================
class SparseTemporalInterpolator:
    """
    Instead of computing 100 time points, compute key frames and interpolate between.
    Strategy 4: Sparse Temporal Sampling with Interpolation
    """
    def __init__(self, interpolator, key_frames=10):
        self.interpolator = interpolator
        self.key_times = np.linspace(0, 1, key_frames)
        self.key_results = {}  # Only store sparse key frames
        self.sources = None
        self.target = None
        self.target_shape = (256, 256)
    
    def precompute_key_frames(self, sources, target, target_shape=(256, 256)):
        """Compute only sparse key frames"""
        self.sources = sources
        self.target = target
        self.target_shape = target_shape
        self.key_results = {}
        
        for t in self.key_times:
            try:
                res = self.interpolator.interpolate_fields(
                    sources, target, target_shape=target_shape, time_norm=t
                )
                # Store lightweight version (fields only, no metadata duplication)
                self.key_results[t] = {
                    'fields': res['fields'],
                    'thickness_nm': res['derived']['thickness_nm'],
                    'time_norm': t
                }
            except Exception as e:
                st.warning(f"Key frame t={t:.2f} failed: {e}")
    
    def get_at_time(self, t_query):
        """Interpolate between nearest key frames"""
        if not self.key_results:
            return None
        
        # Find bracketing key frames
        idx = np.searchsorted(self.key_times, t_query)
        if idx == 0:
            return self.key_results[self.key_times[0]]
        if idx >= len(self.key_times):
            return self.key_results[self.key_times[-1]]
        
        t0, t1 = self.key_times[idx-1], self.key_times[idx]
        alpha = (t_query - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
        
        # Linear interpolation between key frames
        res0 = self.key_results[t0]
        res1 = self.key_results[t1]
        
        interp_fields = {}
        for key in res0['fields']:
            interp_fields[key] = (1-alpha) * res0['fields'][key] + alpha * res1['fields'][key]
        
        interp_thickness = (1-alpha) * res0['thickness_nm'] + alpha * res1['thickness_nm']
        
        return {
            'fields': interp_fields,
            'thickness_nm': interp_thickness,
            'time_norm': t_query
        }
    
    def get_stats(self):
        return {
            'key_frames': len(self.key_results),
            'key_times': self.key_times.tolist()
        }

# =============================================
# MEMORY-EFFICIENT TEMPORAL MANAGER (Tiered Approach)
# =============================================
class TemporalManager:
    """
    Tiered caching approach for optimal memory vs. responsiveness balance.
    Recommended Implementation combining all strategies.
    """
    def __init__(self, interpolator, sources, target, max_cache_size=3, n_key_frames=10):
        self.interpolator = interpolator
        self.sources = sources
        self.target = target
        self.max_cache_size = max_cache_size
        self.n_key_frames = n_key_frames
        
        # Tier 1: Always keep thickness curve (lightweight ~1KB)
        self.thickness_curve = None
        self.weights = None
        self._compute_thickness_curve()
        
        # Tier 2: LRU cache for full fields (3 entries max ~4.5MB)
        self.field_cache = TemporalFieldCache(maxsize=max_cache_size)
        
        # Tier 3: Sparse key frames for fast animation (10 entries ~15MB)
        self.sparse_interpolator = SparseTemporalInterpolator(interpolator, key_frames=n_key_frames)
        self.sparse_interpolator.precompute_key_frames(sources, target, target_shape=(256, 256))
        
        # Tier 4: Animation streamer (disk-based, ~1MB RAM)
        self.animation_streamer = None
        
        # Access tracking
        self.access_order = []
        self.hit_count = 0
        self.miss_count = 0
    
    def _compute_thickness_curve(self):
        """Compute once, store forever (cheap)"""
        try:
            res = self.interpolator.interpolate_fields(
                self.sources, self.target, time_norm=None, n_time_points=100
            )
            if res:
                self.thickness_curve = res['derived']['thickness_time']
                self.weights = res['weights']
        except Exception as e:
            st.warning(f"Thickness curve computation failed: {e}")
            self.thickness_curve = {'t_norm': [0, 1], 'th_nm': [0, 0]}
            self.weights = None
    
    def get_fields(self, t_query):
        """Get fields at any time with minimal computation"""
        # Check exact cache first (Tier 2)
        cached = self.field_cache.get(t_query)
        if cached is not None:
            self.hit_count += 1
            return cached
        
        self.miss_count += 1
        
        # Check if near key frame (within 0.05) (Tier 3)
        nearest_key = min(self.sparse_interpolator.key_times, key=lambda x: abs(x - t_query))
        if abs(nearest_key - t_query) < 0.05:
            fields = self.sparse_interpolator.key_results[nearest_key]['fields']
            self.field_cache.put(t_query, fields)
            return fields
        
        # Interpolate between key frames (Tier 3)
        interp_result = self.sparse_interpolator.get_at_time(t_query)
        if interp_result:
            fields = interp_result['fields']
            self.field_cache.put(t_query, fields)
            return fields
        
        # Fallback: compute on-demand (slowest)
        try:
            res = self.interpolator.interpolate_fields(
                self.sources, self.target, target_shape=(256, 256), time_norm=t_query
            )
            if res:
                fields = res['fields']
                self.field_cache.put(t_query, fields)
                return fields
        except Exception as e:
            st.warning(f"On-demand computation failed at t={t_query}: {e}")
        
        return None
    
    def get_thickness_curve(self):
        """Return pre-computed thickness evolution curve"""
        return self.thickness_curve
    
    def get_weights(self):
        """Return interpolation weights"""
        return self.weights
    
    def init_animation_streamer(self, n_frames=50):
        """Initialize disk-based animation streaming (Tier 4)"""
        if self.animation_streamer is None:
            self.animation_streamer = StreamingAnimation(
                self.interpolator, self.sources, self.target, n_frames=n_frames
            )
        return self.animation_streamer
    
    def cleanup_animation(self):
        """Clean up animation streamer"""
        if self.animation_streamer:
            self.animation_streamer.cleanup()
            self.animation_streamer = None
    
    def get_stats(self):
        return {
            'thickness_curve': self.thickness_curve is not None,
            'field_cache': self.field_cache.get_stats(),
            'sparse_interpolator': self.sparse_interpolator.get_stats(),
            'animation_streamer': self.animation_streamer is not None,
            'cache_hits': self.hit_count,
            'cache_misses': self.miss_count,
            'hit_rate': self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
        }
    
    def clear(self):
        """Clear all caches"""
        self.field_cache.clear()
        self.cleanup_animation()
        self.thickness_curve = None
        self.weights = None
        self.access_order = []
        self.hit_count = 0
        self.miss_count = 0
        gc.collect()

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
        """Extract parameters from filenames"""
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
# ENHANCED CORE‑SHELL INTERPOLATOR
# =============================================
class CoreShellInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                 param_sigma=None, temperature=1.0):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        if param_sigma is None:
            param_sigma = [0.15, 0.15, 0.15, 0.15]
        self.param_sigma = param_sigma
        self.temperature = temperature
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
    
    def compute_parameter_kernel(self, source_params: List[Dict], target_params: Dict):
        """Domain‑specific kernel with weighted Euclidean distance and hard categorical constraints."""
        phys_weights = {
            'fc': 2.0,
            'rs': 1.0,
            'c_bulk': 3.0,
            'L0_nm': 0.5
        }
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
            weighted_sq = sum(phys_weights[p] * (d / self.param_sigma[i])**2
                            for i, (p, d) in enumerate(zip(['fc','rs','c_bulk','L0_nm'], diff)))
            w = np.exp(-0.5 * weighted_sq)
            weights.append(w)
        cat_factor = []
        for src in source_params:
            factor = 1.0
            if src.get('bc_type') != target_params.get('bc_type'):
                factor *= 1e-6
            if src.get('use_edl') != target_params.get('use_edl'):
                factor *= 1e-6
            if src.get('mode') != target_params.get('mode'):
                factor *= 1e-6
            cat_factor.append(factor)
        return np.array(weights) * np.array(cat_factor)
    
    def encode_parameters(self, params_list: List[Dict]) -> torch.Tensor:
        """Physics‑aware encoding: concentration is log‑scaled."""
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
        """Extract or interpolate fields from source at normalised time."""
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
        """Interpolate fields and thickness evolution at a given normalised time."""
        if not sources:
            return None
        source_params = []
        source_fields = []
        source_thickness = []
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
                t_norm = t_vals / t_max
                source_thickness.append({
                    't_norm': t_norm,
                    'th_nm': th_vals,
                    't_max': t_max
                })
            else:
                source_thickness.append({
                    't_norm': np.array([0.0, 1.0]),
                    'th_nm': np.array([0.0, 0.0]),
                    't_max': 1.0
                })
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
                'thickness_time': {
                    't_norm': common_t_norm.tolist(),
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
            'time_norm': t_req
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
                             title="Shell Thickness Evolution", figsize=(10,6)):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        t_norm = thickness_time['t_norm']
        th_nm = thickness_time['th_nm']
        ax.plot(t_norm, th_nm, 'b-', linewidth=3, label='Interpolated')
        if source_curves is not None and weights is not None:
            for i, (src_t, src_th) in enumerate(source_curves):
                alpha = min(weights[i] * 5, 0.8)
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
                'interpolation_method': 'core_shell_transformer_physics_enhanced',
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
    st.set_page_config(page_title="Core‑Shell Deposition Interpolator (Physics‑Enhanced + Temporal)",
                      layout="wide", page_icon="🧪", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .main-header { font-size: 3.2rem; color: #1E3A8A; text-align: center; padding: 1rem;
    background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 900; margin-bottom: 1rem; }
    .section-header { font-size: 2.0rem; color: #374151; font-weight: 800;
    border-left: 6px solid #3B82F6; padding-left: 1.2rem; margin-top: 1.8rem; margin-bottom: 1.2rem; }
    .memory-stats { background: #f0f4f8; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🧪 Core‑Shell Deposition Interpolator (Physics‑Enhanced + Temporal)</h1>', unsafe_allow_html=True)
    
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
    if 'temporal_mgr' not in st.session_state:
        st.session_state.temporal_mgr = None
    if 'params_hash' not in st.session_state:
        st.session_state.params_hash = None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        st.markdown("#### 📁 Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Solutions", use_container_width=True):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                    st.session_state.temporal_mgr = None  # Reset temporal manager
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                if st.session_state.temporal_mgr:
                    st.session_state.temporal_mgr.clear()
                    st.session_state.temporal_mgr = None
                st.success("Cache cleared")
        st.divider()
        
        # Memory Management Settings
        st.markdown("#### 💾 Memory Settings")
        max_cache_size = st.slider("LRU Cache Size (frames)", 1, 10, 3, 1,
                                   help="Number of recent time points to keep in fast cache")
        n_key_frames = st.slider("Sparse Key Frames", 5, 20, 10, 1,
                                help="Number of pre-computed key frames for interpolation")
        enable_streaming = st.checkbox("Enable Disk Streaming for Animation", value=True,
                                       help="Use disk-based storage for animation frames (slower but memory-efficient)")
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
        
        # Check if parameters changed
        current_params_hash = hash((fc, rs, c_bulk, L0_nm, bc_type, use_edl, mode, growth_model))
        params_changed = (st.session_state.params_hash != current_params_hash)
        
        if st.button("🧠 Perform Initial Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Interpolating final state..."):
                    st.session_state.interpolator.set_parameter_sigma([sigma_fc, sigma_rs, sigma_c, sigma_L])
                    st.session_state.interpolator.temperature = temperature
                    target = {
                        'fc': fc, 'rs': rs, 'c_bulk': c_bulk, 'L0_nm': L0_nm,
                        'bc_type': bc_type, 'use_edl': use_edl, 'mode': mode,
                        'growth_model': growth_model, 'alpha_nd': alpha_nd
                    }
                    # Initialize TemporalManager with memory-efficient strategies
                    st.session_state.temporal_mgr = TemporalManager(
                        st.session_state.interpolator,
                        st.session_state.solutions,
                        target,
                        max_cache_size=max_cache_size,
                        n_key_frames=n_key_frames
                    )
                    # Get initial result at final time
                    res = st.session_state.temporal_mgr.get_fields(1.0)
                    if res:
                        # Reconstruct full result structure
                        st.session_state.interpolation_result = {
                            'fields': res,
                            'derived': {
                                'thickness_time': st.session_state.temporal_mgr.get_thickness_curve(),
                                'thickness_nm': 0.0,  # Will be computed
                                'phase_stats': {},
                                'material': np.zeros((256, 256)),
                                'potential': np.zeros((256, 256))
                            },
                            'weights': st.session_state.temporal_mgr.get_weights(),
                            'target_params': target,
                            'shape': (256, 256),
                            'num_sources': len(st.session_state.solutions),
                            'time_norm': 1.0
                        }
                        st.session_state.params_hash = current_params_hash
                        st.success("Interpolation successful! Use slider below to explore time.")
                    else:
                        st.error("Interpolation failed.")
        
        # Memory Statistics Display
        if st.session_state.temporal_mgr:
            st.divider()
            st.markdown("#### 📊 Memory Statistics")
            stats = st.session_state.temporal_mgr.get_stats()
            st.markdown(f"""
            <div class="memory-stats">
            <strong>Cache Hits:</strong> {stats['cache_hits']}<br>
            <strong>Cache Misses:</strong> {stats['cache_misses']}<br>
            <strong>Hit Rate:</strong> {stats['hit_rate']:.2%}<br>
            <strong>Field Cache:</strong> {stats['field_cache']['cached_entries']} entries ({stats['field_cache']['memory_mb']:.2f} MB)<br>
            <strong>Key Frames:</strong> {stats['sparse_interpolator']['key_frames']}<br>
            <strong>Thickness Curve:</strong> {'✓' if stats['thickness_curve'] else '✗'}<br>
            <strong>Animation Streaming:</strong> {'✓' if stats['animation_streamer'] else '✗'}
            </div>
            """, unsafe_allow_html=True)
    
    # Main area
    if st.session_state.interpolation_result:
        res = st.session_state.interpolation_result
        target = res['target_params']
        L0_nm = target.get('L0_nm', 60.0)
        tabs = st.tabs(["📊 Fields", "📈 Thickness Evolution", "🧪 Derived Quantities", "⚖️ Weights & Uncertainty", "💾 Export"])
        
        with tabs[0]:
            st.markdown('<h2 class="section-header">📊 Interpolated Fields</h2>', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                current_time = st.slider("⏱️ Normalized Time", 0.0, 1.0,
                                        value=res.get('time_norm', 1.0),
                                        step=0.01,
                                        help="0 = start of deposition, 1 = final state")
            with col2:
                if st.button("🔄 Update to this time", use_container_width=True):
                    with st.spinner(f"Interpolating at t = {current_time:.2f}..."):
                        fields = st.session_state.temporal_mgr.get_fields(current_time)
                        if fields:
                            res['fields'] = fields
                            res['time_norm'] = current_time
                            st.session_state.interpolation_result = res
                            st.rerun()
                        else:
                            st.error("Interpolation failed at this time.")
            
            # Animation with memory-efficient streaming
            if st.checkbox("🎬 Animate evolution"):
                fps = st.slider("Frames per second", 1, 30, 10)
                n_anim_frames = st.slider("Number of animation frames", 10, 100, 20)
                
                if enable_streaming and st.session_state.temporal_mgr.animation_streamer is None:
                    with st.spinner("Preparing animation frames (disk-based)..."):
                        st.session_state.temporal_mgr.init_animation_streamer(n_frames=n_anim_frames)
                
                placeholder = st.empty()
                field_choice = st.selectbox("Select field for animation", 
                                           ['c (concentration)', 'phi (shell)', 'psi (core)'],
                                           key='anim_field_choice')
                field_map = {'c (concentration)': 'c', 'phi (shell)': 'phi', 'psi (core)': 'psi'}
                field_key = field_map[field_choice]
                
                if st.session_state.temporal_mgr.animation_streamer and enable_streaming:
                    # Use disk-based streaming
                    for i in range(st.session_state.temporal_mgr.animation_streamer.n_frames):
                        frame = st.session_state.temporal_mgr.animation_streamer.get_frame(i)
                        if frame:
                            field_data = frame[field_key]
                            fig = st.session_state.visualizer.create_field_heatmap(
                                field_data, title=f"t = {frame['time_norm']:.2f}",
                                cmap_name=st.session_state.get('cmap', 'viridis'),
                                L0_nm=L0_nm, target_params=target,
                                colorbar_label=field_choice.split()[0]
                            )
                            placeholder.pyplot(fig)
                            plt.close(fig)
                            time.sleep(1/fps)
                    st.success("Animation finished (disk-streamed).")
                else:
                    # Use in-memory interpolation
                    times = np.linspace(0, 1, n_anim_frames)
                    for t in times:
                        fields = st.session_state.temporal_mgr.get_fields(t)
                        if fields:
                            field_data = fields[field_key]
                            fig = st.session_state.visualizer.create_field_heatmap(
                                field_data, title=f"t = {t:.2f}",
                                cmap_name=st.session_state.get('cmap', 'viridis'),
                                L0_nm=L0_nm, target_params=target,
                                colorbar_label=field_choice.split()[0]
                            )
                            placeholder.pyplot(fig)
                            plt.close(fig)
                            time.sleep(1/fps)
                    st.success("Animation finished (in-memory).")
            
            # Static field display
            field_choice = st.selectbox("Select field", 
                                       ['c (concentration)', 'phi (shell)', 'psi (core)'],
                                       key='field_choice')
            field_map = {'c (concentration)': 'c', 'phi (shell)': 'phi', 'psi (core)': 'psi'}
            field_key = field_map[field_choice]
            field_data = res['fields'][field_key]
            cmap_cat = st.selectbox("Colormap category", list(COLORMAP_OPTIONS.keys()), index=0)
            cmap = st.selectbox("Colormap", COLORMAP_OPTIONS[cmap_cat], index=0, key='cmap')
            fig = st.session_state.visualizer.create_field_heatmap(
                field_data, title=f"Interpolated {field_choice} at t = {res.get('time_norm', 1.0):.2f}",
                cmap_name=cmap, L0_nm=L0_nm, target_params=target,
                colorbar_label=field_choice.split()[0]
            )
            st.pyplot(fig)
            if st.checkbox("Show interactive heatmap"):
                fig_inter = st.session_state.visualizer.create_interactive_heatmap(
                    field_data, title=f"Interpolated {field_choice} at t = {res.get('time_norm', 1.0):.2f}",
                    cmap_name=cmap, L0_nm=L0_nm, target_params=target
                )
                st.plotly_chart(fig_inter, use_container_width=True)
        
        with tabs[1]:
            st.markdown('<h2 class="section-header">📈 Shell Thickness Evolution</h2>', unsafe_allow_html=True)
            thickness_time = res['derived']['thickness_time']
            fig_th = st.session_state.visualizer.create_thickness_plot(
                thickness_time, title=f"Interpolated Thickness for fc={target['fc']:.3f}, rs={target['rs']:.3f}, c_bulk={target['c_bulk']:.2f}"
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
                st.metric("Electrolyte", f"{stats.get('Electrolyte', (0, 0))[0]:.4f} (nd²)",
                         help=f"Real area: {stats.get('Electrolyte', (0, 0))[1]*1e18:.2f} nm²")
            with cols[1]:
                st.metric("Ag shell", f"{stats.get('Ag', (0, 0))[0]:.4f} (nd²)",
                         help=f"Real area: {stats.get('Ag', (0, 0))[1]*1e18:.2f} nm²")
            with cols[2]:
                st.metric("Cu core", f"{stats.get('Cu', (0, 0))[0]:.4f} (nd²)",
                         help=f"Real area: {stats.get('Cu', (0, 0))[1]*1e18:.2f} nm²")
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
            st.markdown('<h2 class="section-header">⚖️ Weights & Uncertainty</h2>', unsafe_allow_html=True)
            if res['weights']:
                df_weights = pd.DataFrame({
                    'Source index': range(len(res['weights']['combined'])),
                    'Combined weight': res['weights']['combined'],
                    'Kernel weight': res['weights']['kernel'],
                    'Attention score': res['weights']['attention']
                })
                st.dataframe(df_weights.style.format("{:.4f}"))
                entropy = res['weights'].get('entropy', 0.0)
                st.metric("Weight Entropy (uncertainty)", f"{entropy:.4f}",
                         help="Higher entropy means more sources contribute similarly → less confident prediction.")
                fig_w, ax = plt.subplots(figsize=(10,5))
                x = np.arange(len(res['weights']['combined']))
                width = 0.25
                ax.bar(x - width, res['weights']['kernel'], width, label='Kernel (physics prior)', alpha=0.7)
                ax.bar(x, res['weights']['attention'], width, label='Attention (learned)', alpha=0.7)
                ax.bar(x + width, res['weights']['combined'], width, label='Combined (final)', alpha=0.7)
                ax.set_xlabel('Source index')
                ax.set_ylabel('Weight')
                ax.set_title('Comparison of weights (kernel now a hard prior)')
                ax.legend()
                st.pyplot(fig_w)
            else:
                st.info("Weights not available. Re-run interpolation.")
        
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
                "current_time_norm": res.get('time_norm', 1.0),
                "final_thickness_nm": res['derived']['thickness_nm'],
                "uncertainty_entropy": res['weights'].get('entropy', 0.0) if res['weights'] else None
            })
    else:
        st.info("Load solutions and set target parameters in the sidebar, then click 'Perform Initial Interpolation'.")
    
    # Cleanup on session end
    if st.session_state.temporal_mgr:
        # Optional: cleanup animation streamer when not in use
        pass

if __name__ == "__main__":
    main()
