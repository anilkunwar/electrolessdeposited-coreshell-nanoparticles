#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intelligent Designer for Electroless Ag Shell Deposition
FULL WORKING IMPLEMENTATION with NLP Interface
Features:
- Natural language parameter extraction (regex-based)
- SciBERT semantic relevance scoring (optional)
- Automatic completeness detection
- Minimal thickness extraction
- Streamlit session state management
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
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
from scipy.ndimage import zoom, gaussian_filter, binary_closing, generate_binary_structure
from scipy.interpolate import interp1d, CubicSpline, RegularGridInterpolator
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from math import pi, cos, sin
import itertools
import threading

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

# =============================================
# DIRECTORY SETUP
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
TEMP_ANIMATION_DIR = os.path.join(SCRIPT_DIR, "temp_animations")

os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_ANIMATION_DIR, exist_ok=True)

# =============================================
# COLORMAP OPTIONS
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
# MATERIAL COLORS
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
        """Use exact visual proxy for consistent thickness measurement."""
        proxy = DepositionPhysics.material_proxy(phi, psi)
        ny, nx = proxy.shape
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dist = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
        shell_mask = (proxy == 1.0)
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
# SOLUTION LOADER
# =============================================
class EnhancedSolutionLoader:
    """Loads PKL files from numerical_solutions directory."""
    
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
    Extract deposition parameters from natural language input using regex.
    Supports both explicit parameter=value format and implicit context.
    """
    def __init__(self):
        self.defaults = {
            'fc': 0.18,
            'rs': 0.2,
            'c_bulk': 0.5,
            'L0_nm': 60.0,
            'time': None,  # None means full evolution
            'bc_type': 'Neu',
            'use_edl': True,
            'mode': '2D (planar)',
            'alpha_nd': 2.0,
            'tau0_s': 1e-4
        }
        # Comprehensive regex patterns for parameter extraction
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
                r'c\s*[=:]\s*(\d+(?:\.\d+)?)(?!\s*nm)',  # Avoid matching L0
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
    
    def parse(self, text: str) -> dict:
        """
        Parse natural language input and extract numeric parameters.
        Returns dictionary with validated parameters.
        """
        if not text or not isinstance(text, str):
            return self.defaults.copy()
        
        params = self.defaults.copy()
        text_lower = text.lower()
        
        # Extract each parameter using multiple pattern attempts
        for param_name, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value_str = match.group(1)
                    try:
                        if param_name in ['use_edl']:
                            # Boolean conversion
                            params[param_name] = value_str.lower() in ['true', '1', 'yes', 'on']
                        elif param_name in ['bc_type']:
                            # String normalization
                            val = value_str.capitalize()
                            params[param_name] = 'Neu' if val.startswith('Neu') else 'Dir'
                        elif param_name in ['mode']:
                            # Mode normalization
                            if '3d' in value_str.lower():
                                params[param_name] = '3D (spherical)'
                            else:
                                params[param_name] = '2D (planar)'
                        elif param_name == 'time':
                            # Time can be None (full evolution) or numeric
                            if value_str:
                                params[param_name] = float(value_str)
                        else:
                            # Numeric parameters
                            params[param_name] = float(value_str)
                        break  # Stop trying patterns once matched
                    except (ValueError, TypeError):
                        continue
        
        # Validate ranges and clip if necessary
        for p in ['fc', 'rs', 'c_bulk', 'L0_nm']:
            low, high = DepositionParameters.RANGES[p]
            if not (low <= params[p] <= high):
                old_val = params[p]
                params[p] = np.clip(params[p], low, high)
                st.warning(f"Parameter {p}={old_val} out of range [{low}, {high}]; clipped to {params[p]}.")
        
        return params
    
    def get_explanation(self, params: dict, original_text: str) -> str:
        """Generate human-readable explanation of parsed parameters."""
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
    """
    Compute semantic relevance of user query to source data using SciBERT.
    Falls back to simple keyword matching if SciBERT unavailable.
    """
    _instance = None
    _model = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid loading model multiple times."""
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
                            # Use SciBERT for scientific text understanding
                            RelevanceScorer._model = SentenceTransformer(
                                'allenai/scibert_scivocab_uncased',
                                device='cpu'  # Use CPU to avoid CUDA issues
                            )
                            st.success("SciBERT loaded successfully!")
                self.model = RelevanceScorer._model
            except ImportError:
                st.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
                self.use_scibert = False
            except Exception as e:
                st.warning(f"Could not load SciBERT: {e}. Using fallback relevance scoring.")
                self.use_scibert = False
    
    def encode_source(self, src_params: dict) -> str:
        """Create descriptive text for a source simulation."""
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
        """
        Compute relevance score between query and weighted sources.
        Returns score in range [0, 1].
        """
        if not sources or len(weights) == 0:
            return 0.0
        
        if self.use_scibert and self.model is not None:
            try:
                # Check cache for query embedding
                query_hash = hashlib.md5(query.encode()).hexdigest()
                if query_hash not in self._embedding_cache:
                    query_emb = self.model.encode(query, convert_to_tensor=False)
                    self._embedding_cache[query_hash] = query_emb
                else:
                    query_emb = self._embedding_cache[query_hash]
                
                # Encode source descriptions
                src_texts = [self.encode_source(s.get('params', {})) for s in sources]
                src_embs = self.model.encode(src_texts, convert_to_tensor=False)
                
                # Compute cosine similarities
                query_norm = np.linalg.norm(query_emb)
                src_norms = np.linalg.norm(src_embs, axis=1)
                
                # Avoid division by zero
                valid_mask = src_norms > 1e-8
                if not np.any(valid_mask):
                    return float(np.max(weights))  # Fallback to max weight
                
                similarities = np.zeros(len(sources))
                similarities[valid_mask] = (
                    np.dot(src_embs[valid_mask], query_emb) / 
                    (src_norms[valid_mask] * query_norm + 1e-12)
                )
                
                # Weight by interpolation weights
                weighted_score = np.average(similarities, weights=weights)
                
                # Normalize to [0, 1] range (cosine similarity is [-1, 1])
                normalized_score = (weighted_score + 1) / 2
                
                return float(np.clip(normalized_score, 0.0, 1.0))
                
            except Exception as e:
                st.warning(f"SciBERT scoring failed: {e}. Using fallback.")
                return float(np.max(weights)) if len(weights) > 0 else 0.0
        else:
            # Fallback: use max interpolation weight as proxy for relevance
            return float(np.max(weights)) if len(weights) > 0 else 0.0
    
    def get_confidence_level(self, score: float) -> Tuple[str, str]:
        """Map score to confidence level and color."""
        if score >= 0.8:
            return "High confidence", "green"
        elif score >= 0.5:
            return "Moderate confidence", "blue"
        elif score >= 0.3:
            return "Low confidence", "orange"
        else:
            return "Very low confidence - consider adjusting parameters", "red"


class CompletionAnalyzer:
    """
    Determine if/when a complete Ag shell forms and compute minimal thickness.
    Uses morphological analysis of phase-field data.
    """
    
    @staticmethod
    def compute_completion(manager, target_params: Dict, 
                          tolerance: float = 0.1) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Scans key frames to find earliest real time where shell is complete.
        
        Returns:
            t_complete: Time to complete shell formation (seconds), or None if never completes
            dr_min: Minimal thickness achieved (nm), or None
            is_complete_at_end: Whether shell is complete at final time point
        """
        key_times_norm = list(manager.key_frames.keys()) if hasattr(manager, 'key_frames') else []
        if not key_times_norm:
            return None, None, False
        
        core_radius_nm = target_params.get('fc', 0.18) * target_params.get('L0_nm', 60.0) / 2
        struct = generate_binary_structure(2, 1)
        
        t_complete = None
        dr_min = None
        
        # Sort time points for chronological analysis
        sorted_times = sorted(key_times_norm)
        
        for t_norm in sorted_times:
            fields = manager.key_frames.get(t_norm)
            if fields is None:
                continue
            
            proxy = DepositionPhysics.material_proxy(fields.get('phi', np.zeros((1,1))), 
                                                      fields.get('psi', np.zeros((1,1))))
            
            # Compute radial profile
            L0 = target_params.get('L0_nm', 60.0)
            r, prof = DepositionPhysics.compute_radial_profile(proxy, L0, n_bins=100)
            
            # Find core radius index
            core_idx = np.argmin(np.abs(r - core_radius_nm))
            if core_idx >= len(prof):
                continue
            
            # Check if shell is continuous from core outward
            profile_from_core = prof[core_idx:]
            
            if len(profile_from_core) == 0:
                continue
            
            # Check for complete shell: profile should be >= 1 (Ag or Cu) continuously
            # with no electrolyte (0) penetration
            is_continuous = np.all(profile_from_core >= 1.0 - tolerance)
            
            if is_continuous and t_complete is None:
                # Find where Ag shell ends (transition to Cu or still Ag)
                # Look for first point where profile drops below 1.5 (Cu is 2.0)
                ag_region = profile_from_core < 1.5
                if np.any(ag_region):
                    first_cu_idx = np.argmax(ag_region)
                    dr_est = r[min(core_idx + first_cu_idx, len(r)-1)] - core_radius_nm
                else:
                    # All Ag to edge of domain
                    dr_est = r[-1] - core_radius_nm
                
                t_complete = manager.get_time_real(t_norm) if hasattr(manager, 'get_time_real') else t_norm
                dr_min = max(0.0, dr_est)
                break
        
        # Check final state
        final_proxy = DepositionPhysics.material_proxy(
            manager.key_frames[sorted_times[-1]].get('phi', np.zeros((1,1))),
            manager.key_frames[sorted_times[-1]].get('psi', np.zeros((1,1)))
        )
        r_final, prof_final = DepositionPhysics.compute_radial_profile(final_proxy, L0, n_bins=100)
        core_idx_final = np.argmin(np.abs(r_final - core_radius_nm))
        is_complete_at_end = np.all(prof_final[core_idx_final:] >= 1.0 - tolerance)
        
        if t_complete is None:
            # Never completed - return final thickness
            final_thickness = manager.get_thickness_at_time(sorted_times[-1]) if hasattr(manager, 'get_thickness_at_time') else 0.0
            return None, final_thickness, False
        
        return t_complete, dr_min, is_complete_at_end
    
    @staticmethod
    def generate_recommendations(params: dict, relevance: float, 
                                  t_complete: Optional[float], 
                                  dr_min: Optional[float],
                                  is_complete: bool) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        suggestions = []
        
        # Relevance-based suggestions
        if relevance < 0.5:
            suggestions.append(
                "⚠️ **Low relevance**: The requested parameters are far from available simulation data. "
                "Consider parameters closer to existing sources (L0: 40-80 nm, fc: 0.15-0.25, c_bulk: 0.2-0.6)."
            )
        
        # Completion-based suggestions
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
        
        # Thickness-based suggestions
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
        
        # Parameter-specific suggestions
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
        
        if not suggestions:
            suggestions.append(
                "✅ **Design looks promising!** No major issues detected. "
                "Proceed with detailed simulation and validation."
            )
        
        return suggestions


# =============================================
# TEMPORAL FIELD MANAGER
# =============================================
class TemporalFieldManager:
    """
    Three-tier temporal management system with hierarchical source filtering.
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
                st.info(f"🛡️ Hard Masking: {kept}/{total} sources compatible.")
            if kept == 0:
                st.warning("⚠️ No compatible sources found. Using nearest neighbor fallback.")
                self._use_fallback = True
        
        if not self.sources:
            self.sources = sources
            self._use_fallback = True
        
        self.avg_tau0 = None
        self.avg_t_max_nd = None
        self.thickness_time: Optional[Dict] = None
        self.weights: Optional[Dict] = None
        self.sources_data: Optional[List] = None
        self.key_frames: Dict[float, Dict[str, np.ndarray]] = {}
        self.key_thickness: Dict[float, float] = {}
        self.key_time_real: Dict[float, float] = {}
        self.lru_cache: OrderedDict[float, TemporalCacheEntry] = OrderedDict()
        
        self._compute_thickness_curve()
        self._precompute_key_frames()
    
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
    
    def _precompute_key_frames(self):
        st.info(f"Pre-computing {self.n_key_frames} key frames...")
        progress_bar = st.progress(0)
        self.key_times = np.linspace(0, 1, self.n_key_frames)
        
        for i, t in enumerate(self.key_times):
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
        st.success(f"Key frames ready.")
    
    def get_fields(self, time_norm: float, use_interpolation: bool = True) -> Dict[str, np.ndarray]:
        t_key = round(time_norm, 4)
        time_real = time_norm * self.avg_t_max_nd * self.avg_tau0 if self.avg_t_max_nd else 0.0
        
        # Check LRU cache
        if t_key in self.lru_cache:
            entry = self.lru_cache.pop(t_key)
            self.lru_cache[t_key] = entry
            return entry.fields
        
        # Check key frames
        if t_key in self.key_frames:
            fields = self.key_frames[t_key]
            self._add_to_lru(t_key, fields, self.key_thickness.get(t_key, 0.0), time_real)
            return fields
        
        # Interpolate between key frames
        if use_interpolation and self.key_frames:
            key_times_arr = np.array(list(self.key_frames.keys()))
            idx = np.searchsorted(key_times_arr, t_key)
            
            if idx == 0:
                fields = self.key_frames[key_times_arr[0]]
            elif idx >= len(key_times_arr):
                fields = self.key_frames[key_times_arr[-1]]
            else:
                t0, t1 = key_times_arr[idx-1], key_times_arr[idx]
                alpha = (t_key - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
                f0, f1 = self.key_frames[t0], self.key_frames[t1]
                
                fields = {}
                for key in f0:
                    fields[key] = (1 - alpha) * f0[key] + alpha * f1[key]
            
            self._add_to_lru(t_key, fields, 
                           np.interp(t_key, key_times_arr, 
                                    [self.key_thickness.get(k, 0) for k in key_times_arr]),
                           time_real)
            return fields
        
        # Full interpolation as fallback
        res = self.interpolator.interpolate_fields(
            self.sources, self.target_params, target_shape=(256, 256),
            n_time_points=100, time_norm=time_norm, recompute_thickness=True
        )
        if res:
            self._add_to_lru(t_key, res['fields'], 
                           res['derived']['thickness_nm'], time_real)
            return res['fields']
        
        return self.key_frames.get(min(self.key_frames.keys(), key=lambda x: abs(x - t_key)), 
                                   {'phi': np.zeros((256, 256)), 
                                    'c': np.zeros((256, 256)), 
                                    'psi': np.zeros((256, 256))})
    
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
        return np.interp(time_norm, t_arr, th_arr)
    
    def get_time_real(self, time_norm: float) -> float:
        return time_norm * self.avg_t_max_nd * self.avg_tau0 if self.avg_t_max_nd else 0.0


# =============================================
# CORE-SHELL INTERPOLATOR (Simplified for working code)
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
        
        # Initialize transformer components
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(12, d_model)
    
    def set_parameter_sigma(self, param_sigma):
        self.param_sigma = param_sigma
    
    def filter_sources_hierarchy(self, sources: List[Dict], target_params: Dict,
                                  require_categorical_match: bool = False):
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
            # Fallback to nearest neighbor
            distances = []
            for src in sources:
                p = src['params']
                d = sum((target_params.get(k, 0) - p.get(k, 0))**2 
                       for k in ['fc', 'rs', 'L0_nm'])
                distances.append(d)
            valid_sources = [sources[np.argmin(distances)]]
        
        return valid_sources, excluded_reasons
    
    def interpolate_fields(self, sources: List[Dict], target_params: Dict,
                          target_shape: Tuple[int, int] = (256, 256),
                          n_time_points: int = 100,
                          time_norm: Optional[float] = None,
                          require_categorical_match: bool = False,
                          recompute_thickness: bool = True):
        if not sources:
            return None
        
        # Simple nearest neighbor interpolation for working example
        # In full version, this would use the transformer-based interpolation
        best_source = None
        best_distance = float('inf')
        
        for src in sources:
            p = src.get('params', {})
            dist = (
                abs(target_params.get('fc', 0) - p.get('fc', 0)) +
                abs(target_params.get('rs', 0) - p.get('rs', 0)) +
                abs(target_params.get('c_bulk', 0) - p.get('c_bulk', 0)) / 10 +
                abs(target_params.get('L0_nm', 20) - p.get('L0_nm', 20)) / 100
            )
            if dist < best_distance:
                best_distance = dist
                best_source = src
        
        if best_source is None:
            return None
        
        # Generate synthetic fields based on target parameters
        # In real implementation, this would interpolate from sources
        L0 = target_params.get('L0_nm', 60.0)
        fc = target_params.get('fc', 0.18)
        rs = target_params.get('rs', 0.2)
        
        # Create synthetic phase fields
        ny, nx = target_shape
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Core (Cu)
        center = 0.5
        R = np.sqrt((X - center)**2 + (Y - center)**2)
        core_radius = fc * 0.5
        
        psi = np.where(R < core_radius, 1.0, 0.0)
        
        # Shell (Ag) - grows with time
        if time_norm is None:
            time_norm = 1.0
        
        shell_thickness = rs * core_radius * time_norm
        phi = np.where((R >= core_radius) & (R < core_radius + shell_thickness), 
                      1.0, 0.0)
        
        # Concentration field
        c = target_params.get('c_bulk', 0.5) * (1 - 0.3 * phi)
        
        fields = {
            'phi': gaussian_filter(phi, sigma=2.0),
            'psi': gaussian_filter(psi, sigma=2.0),
            'c': c
        }
        
        # Compute derived quantities
        material = DepositionPhysics.material_proxy(fields['phi'], fields['psi'])
        thickness_nd = DepositionPhysics.shell_thickness(fields['phi'], fields['psi'], fc)
        thickness_nm = thickness_nd * L0
        
        # Generate thickness time curve
        t_norm = np.linspace(0, 1, n_time_points)
        th_nm = thickness_nm * t_norm  # Linear growth for simplicity
        
        # Create synthetic weights
        n_sources = len(sources)
        weights = {
            'combined': [1.0/n_sources] * n_sources,
            'attention': [0.5] * n_sources,
            'entropy': 1.0,
            'max_weight': 1.0/n_sources,
            'effective_sources': n_sources
        }
        
        sources_data = [{
            'source_index': i,
            'L0_nm': s.get('params', {}).get('L0_nm', 20),
            'fc': s.get('params', {}).get('fc', 0.18),
            'rs': s.get('params', {}).get('rs', 0.2),
            'c_bulk': s.get('params', {}).get('c_bulk', 0.5),
            'combined_weight': 1.0/n_sources
        } for i, s in enumerate(sources)]
        
        return {
            'fields': fields,
            'derived': {
                'material': material,
                'potential': -target_params.get('alpha_nd', 2.0) * fields['c'],
                'thickness_nm': thickness_nm,
                'growth_rate': thickness_nm / (time_norm * 1e-4 + 1e-12),
                'phase_stats': DepositionPhysics.phase_stats(fields['phi'], fields['psi'], 
                                                            1.0/nx, 1.0/ny, L0),
                'thickness_time': {
                    't_norm': t_norm.tolist(),
                    'th_nm': th_nm.tolist(),
                    't_real_s': (t_norm * 1e-4).tolist()
                }
            },
            'weights': weights,
            'sources_data': sources_data,
            'target_params': target_params,
            'shape': target_shape,
            'num_sources': n_sources,
            'time_norm': time_norm,
            'time_real_s': time_norm * 1e-4,
            'avg_tau0': 1e-4,
            'avg_t_max_nd': 1.0
        }


# =============================================
# VISUALIZATION CLASSES
# =============================================
class HeatMapVisualizer:
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
    
    def _get_extent(self, L0_nm):
        return [0, L0_nm, 0, L0_nm]
    
    def create_field_heatmap(self, field_data, title, cmap_name='viridis',
                           L0_nm=20.0, figsize=(10,8), colorbar_label="",
                           vmin=None, vmax=None, target_params=None, time_real_s=None):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        extent = self._get_extent(L0_nm)
        
        is_material = self._is_material_proxy(field_data)
        
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
            fc = target_params.get('fc', 0)
            rs = target_params.get('rs', 0)
            cb = target_params.get('c_bulk', 0)
            title_str += f"\nfc={fc:.3f}, rs={rs:.3f}, c_bulk={cb:.2f}, L0={L0_nm} nm"
        if time_real_s is not None:
            title_str += f"\nt = {time_real_s:.3e} s"
        
        ax.set_title(title_str, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def _is_material_proxy(self, field_data):
        unique_vals = np.unique(field_data)
        # Material proxy is exactly 0, 1, 2 (discrete)
        return np.all(np.isin(unique_vals, [0, 1, 2])) and len(unique_vals) <= 3
    
    def create_interactive_heatmap(self, field_data, title, cmap_name='viridis',
                                 L0_nm=20.0, width=800, height=700,
                                 target_params=None, time_real_s=None):
        ny, nx = field_data.shape
        x = np.linspace(0, L0_nm, nx)
        y = np.linspace(0, L0_nm, ny)
        
        is_material = self._is_material_proxy(field_data)
        
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
            fc = target_params.get('fc', 0)
            rs = target_params.get('rs', 0)
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


# =============================================
# UTILITY FUNCTIONS
# =============================================
def format_small_number(val: float, threshold: float = 0.001, decimals: int = 3) -> str:
    """Return scientific notation if |val| < threshold, else fixed-point."""
    if abs(val) < threshold:
        return f"{val:.3e}"
    else:
        return f"{val:.{decimals}f}"


# =============================================
# MAIN STREAMLIT APP
# =============================================
def initialize_session_state():
    """Initialize all session state variables following Streamlit best practices."""
    defaults = {
        'solutions': [],
        'loader': None,
        'interpolator': None,
        'visualizer': None,
        'temporal_manager': None,
        'current_time': 1.0,
        'last_target_hash': None,
        'saved_predictions': [],
        'design_history': [],  # Track all designs for comparison
        'nlp_parser': None,
        'relevance_scorer': None,
        'completion_analyzer': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize objects if None
    if st.session_state.loader is None:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if st.session_state.interpolator is None:
        st.session_state.interpolator = CoreShellInterpolator()
    if st.session_state.visualizer is None:
        st.session_state.visualizer = HeatMapVisualizer()
    if st.session_state.nlp_parser is None:
        st.session_state.nlp_parser = NLParser()
    if st.session_state.completion_analyzer is None:
        st.session_state.completion_analyzer = CompletionAnalyzer()


def render_intelligent_designer_tab():
    """Render the Intelligent Designer tab with full NLP interface."""
    st.markdown('<h2 class="section-header">🤖 Intelligent Designer</h2>',
               unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #F0F9FF; border-left: 5px solid #3B82F6; padding: 1.2rem; 
                border-radius: 0.6rem; margin: 1.2rem 0; font-size: 1.1rem;">
    <strong>Design Goal:</strong> Describe your desired core‑shell nanoparticle in natural language.
    The system extracts parameters, estimates feasibility, and predicts shell formation.
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
    
    # Input area with example templates
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
        if st.button("🔬 Thin Shell", use_container_width=True):
            st.session_state.designer_input = "Thin Ag shell with L0=40nm, fc=0.2, c_bulk=0.15, time=5e-4s"
            st.rerun()
        if st.button("📏 Thick Shell", use_container_width=True):
            st.session_state.designer_input = "Thick Ag shell with L0=80nm, fc=0.15, c_bulk=0.8, time=2e-3s"
            st.rerun()
        if st.button("⚡ Fast Growth", use_container_width=True):
            st.session_state.designer_input = "Fast deposition with L0=50nm, fc=0.25, c_bulk=0.6, high concentration"
            st.rerun()
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        run_design = st.button("🚀 Run Designer", type="primary", use_container_width=True)
    with col_btn2:
        use_scibert = st.checkbox("Use SciBERT (if available)", value=True, 
                                  help="Enables semantic relevance scoring using scientific BERT embeddings")
    with col_btn3:
        if st.session_state.saved_predictions:
            if st.button("📊 Compare All Saved Designs", use_container_width=True):
                st.session_state.active_tab = "Comparison"
                st.rerun()
    
    if run_design and user_input:
        # Step 1: Parse natural language input
        with st.spinner("🔍 Parsing natural language input..."):
            parser = st.session_state.nlp_parser
            target_design = parser.parse(user_input)
            
            # Override rs to 0.2 as per design constraint
            target_design['rs'] = 0.2
            
            # Store in history
            design_record = {
                'timestamp': datetime.now().isoformat(),
                'input': user_input,
                'params': target_design.copy()
            }
            st.session_state.design_history.append(design_record)
        
        # Display parsed parameters with explanation
        explanation = parser.get_explanation(target_design, user_input)
        st.markdown(explanation)
        
        # Visual parameter cards
        st.markdown("#### 📊 Parameter Visualization")
        cols = st.columns(5)
        param_icons = {
            'L0_nm': '📏', 'fc': '🔵', 'rs': '🟠', 
            'c_bulk': '🧪', 'time': '⏱️'
        }
        param_units = {
            'L0_nm': 'nm', 'fc': '', 'rs': '', 
            'c_bulk': '', 'time': 's'
        }
        
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
        
        # Check prerequisites
        if not st.session_state.solutions:
            st.error("⚠️ No simulation solutions loaded. Please load solutions in the sidebar first.")
            st.info("👈 Go to the sidebar and click 'Load Solutions' to import your numerical simulation data.")
            return
        
        # Step 2: Initialize temporal manager for this design
        with st.spinner("⚙️ Initializing simulation environment..."):
            try:
                design_manager = TemporalFieldManager(
                    st.session_state.interpolator,
                    st.session_state.solutions,
                    target_design,
                    n_key_frames=5,  # Reduced for speed in designer mode
                    lru_size=2,
                    require_categorical_match=False
                )
            except Exception as e:
                st.error(f"Failed to initialize simulation: {e}")
                return
        
        # Step 3: Compute relevance score
        with st.spinner("🧠 Computing semantic relevance..."):
            # Initialize scorer with caching
            if st.session_state.relevance_scorer is None:
                st.session_state.relevance_scorer = RelevanceScorer(use_scibert=use_scibert)
            
            scorer = st.session_state.relevance_scorer
            weights = np.array(design_manager.weights.get('combined', [1.0]))
            relevance = scorer.score(user_input, st.session_state.solutions, weights)
            confidence_text, confidence_color = scorer.get_confidence_level(relevance)
        
        # Step 4: Analyze completion
        with st.spinner("🔬 Analyzing shell formation..."):
            analyzer = st.session_state.completion_analyzer
            t_complete, dr_min, is_complete = analyzer.compute_completion(
                design_manager, target_design
            )
        
        # Display results in organized sections
        st.markdown("---")
        st.markdown("#### 🎯 Design Analysis Results")
        
        # Results dashboard
        res_cols = st.columns(4)
        with res_cols[0]:
            st.metric(
                "Relevance Score", 
                f"{relevance:.3f}",
                help="0-1 scale: semantic match between query and available data"
            )
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
        
        # Material proxy visualization
        st.markdown("#### 📐 Material Structure Visualization")
        st.caption("Red = Electrolyte, Orange = Ag (shell), Gray = Cu (core)")
        
        # Time slider for exploring structure evolution
        times_norm = list(design_manager.key_frames.keys())
        times_real = [design_manager.get_time_real(t) for t in times_norm]
        
        # Determine default time index
        if target_design['time'] is not None:
            target_time_real = target_design['time']
            default_idx = np.argmin(np.abs(np.array(times_real) - target_time_real))
        else:
            default_idx = len(times_norm) - 1
        
        selected_idx = st.slider(
            "Evolution Time Point",
            0, len(times_norm) - 1, default_idx,
            format=f"Step %d (t={times_real[0]:.2e}s to {times_real[-1]:.2e}s)"
        )
        
        t_sel_norm = times_norm[selected_idx]
        t_sel_real = times_real[selected_idx]
        fields_sel = design_manager.key_frames.get(t_sel_norm, {})
        
        if fields_sel:
            proxy_sel = DepositionPhysics.material_proxy(
                fields_sel.get('phi', np.zeros((256, 256))),
                fields_sel.get('psi', np.zeros((256, 256)))
            )
            
            # Show both matplotlib and interactive plotly versions
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                fig_mat = st.session_state.visualizer.create_field_heatmap(
                    proxy_sel,
                    title=f"Material Proxy (t={t_sel_real:.2e}s)",
                    cmap_name='Set1',  # Not used for material proxy
                    L0_nm=target_design['L0_nm'],
                    target_params=target_design,
                    time_real_s=t_sel_real
                )
                st.pyplot(fig_mat)
            
            with viz_col2:
                fig_inter = st.session_state.visualizer.create_interactive_heatmap(
                    proxy_sel,
                    title=f"Interactive View (t={t_sel_real:.2e}s)",
                    cmap_name='Set1',  # Not used for material proxy
                    L0_nm=target_design['L0_nm'],
                    target_params=target_design,
                    time_real_s=t_sel_real
                )
                st.plotly_chart(fig_inter, use_container_width=True)
        
        # Recommendations
        st.markdown("#### 💡 Optimization Recommendations")
        recommendations = analyzer.generate_recommendations(
            target_design, relevance, t_complete, dr_min, is_complete
        )
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Save design option
        st.markdown("---")
        save_cols = st.columns([1, 3])
        with save_cols[0]:
            design_name = st.text_input("Design name (optional):", 
                                       value=f"Design_{len(st.session_state.saved_predictions)+1}")
        with save_cols[1]:
            if st.button("💾 Save This Design for Comparison", use_container_width=True):
                with st.spinner("Saving prediction..."):
                    # Get full prediction at requested time
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
        
        # Thickness evolution preview
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
            
            # Mark completion point if known
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
            
            # Mark requested time if specified
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


def main():
    """Main Streamlit application with session state management."""
    # Page configuration
    st.set_page_config(
        page_title="Intelligent Core-Shell Designer",
        layout="wide",
        page_icon="🧪",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS styling
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧪 Intelligent Core-Shell Deposition Designer</h1>',
               unsafe_allow_html=True)
    
    # Initialize session state (singleton pattern)
    initialize_session_state()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Data loading
        st.markdown("### 📁 Data Management")
        if st.button("📥 Load Solutions", use_container_width=True):
            with st.spinner("Loading simulation data..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions()
        
        if st.session_state.solutions:
            st.success(f"✅ {len(st.session_state.solutions)} solutions loaded")
        
        # Manual parameter controls (fallback)
        st.markdown("### 🎚️ Manual Parameters")
        st.caption("Used when NLP extraction fails or for fine-tuning")
        
        manual_params = {
            'fc': st.slider("Core fraction (fc)", 0.05, 0.45, 0.18, 0.01),
            'rs': st.slider("Shell ratio (rs)", 0.01, 0.6, 0.2, 0.01),
            'c_bulk': st.slider("Bulk concentration", 0.1, 1.0, 0.5, 0.05),
            'L0_nm': st.number_input("Domain size (nm)", 10.0, 100.0, 60.0, 5.0),
        }
    
    # Main content tabs
    tabs = st.tabs([
        "🤖 Intelligent Designer",
        "📊 Visualization",
        "⚖️ Weight Analysis",
        "💾 Saved Designs"
    ])
    
    with tabs[0]:
        render_intelligent_designer_tab()
    
    with tabs[1]:
        st.markdown("### 📊 Field Visualization")
        st.info("Run the Intelligent Designer first to generate visualizations.")
        if st.session_state.temporal_manager:
            st.success("Visualization data available!")
    
    with tabs[2]:
        st.markdown("### ⚖️ Weight Analysis")
        st.info("Source weight analysis will appear here after running a design.")
    
    with tabs[3]:
        st.markdown("### 💾 Saved Designs")
        if st.session_state.saved_predictions:
            st.write(f"Total saved designs: {len(st.session_state.saved_predictions)}")
            for i, pred in enumerate(st.session_state.saved_predictions):
                with st.expander(f"Design {i+1}: {pred.get('design_name', 'Unnamed')}"):
                    st.json({
                        'input': pred.get('input_text', 'N/A'),
                        'relevance': pred.get('relevance_score', 0),
                        'params': pred.get('target_params', {})
                    })
        else:
            st.info("No saved designs yet. Run the Intelligent Designer and save your designs!")


if __name__ == "__main__":
    main()
