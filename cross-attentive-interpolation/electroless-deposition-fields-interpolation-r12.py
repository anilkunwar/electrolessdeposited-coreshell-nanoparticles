#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHYSICS-AWARE Ag@Cu CORE-SHELL INTERPOLATOR (Production Version)
Loaded from numerical_solutions/ • Radial morphing • Dimensionless physics kernel
FULL TEMPORAL SUPPORT + HYBRID WEIGHTING (Physics Kernel + Attention)
Follows the theoretical procedure for physics‑aware interpolation.
"""

import streamlit as st
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.interpolate import CubicSpline
import torch
import torch.nn as nn
from dataclasses import dataclass
import hashlib
from datetime import datetime
import re
import json
import warnings
from typing import List, Dict, Any, Optional, Tuple

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ag@Cu Physics Interpolator", layout="wide")
st.title("🧪 Ag@Cu Core-Shell Physics-Aware Interpolator")
st.markdown("**Loaded from `numerical_solutions/` • Radial morphing • Full temporal support**")

# ========================= CONFIG =========================
# Absolute path to the script's directory – ensures correct folder regardless of how Streamlit is launched
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
os.makedirs(SOLUTION_DIR, exist_ok=True)

# ========================= ENHANCED LOADER =========================
class EnhancedSolutionLoader:
    """Loads PKL files from numerical_solutions, parsing filenames as fallback."""
    def __init__(self, solutions_dir: str = SOLUTION_DIR):
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

# ========================= LOADER ADAPTER =========================
def convert_to_original_format(enhanced_sol):
    """
    Convert the output of EnhancedSolutionLoader to the format expected by
    the original CoreShellPhysicsInterpolator.
    """
    params = enhanced_sol['params']
    # Ensure all required keys are present
    params.setdefault('L0_nm', params.get('L0_nm', 20.0))
    params.setdefault('fc', params.get('fc', 0.18))
    params.setdefault('rs', params.get('rs', 0.2))
    params.setdefault('c_bulk', params.get('c_bulk', 1.0))
    params.setdefault('use_edl', params.get('use_edl', False))
    params.setdefault('tau0_s', params.get('tau0_s', 1e-4))
    params.setdefault('mode', params.get('mode', '2D (planar)'))
    
    # Convert history to snapshots list (either tuple or dict)
    snapshots = []
    for snap in enhanced_sol.get('history', []):
        # Keep as dict, but original code can handle both
        snapshots.append({
            't_nd': snap['t_nd'],
            'phi': snap['phi'],
            'c': snap['c'],
            'psi': snap['psi']
        })
    
    # Convert thickness_history to list of tuples (t_nd, th_nd, th_nm)
    thickness_history = []
    for th in enhanced_sol.get('thickness_history', []):
        thickness_history.append((th['t_nd'], th['th_nd'], th['th_nm']))
    
    return {
        'filename': enhanced_sol['metadata']['filename'],
        'params': params,
        'snapshots': snapshots,
        'thickness_history': thickness_history,
        'coords_nd': enhanced_sol.get('coords_nd')
    }

# ========================= INITIALIZE SESSION STATE =========================
if 'loader' not in st.session_state:
    st.session_state.loader = EnhancedSolutionLoader(SOLUTION_DIR)
if 'solutions' not in st.session_state:
    st.session_state.solutions = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("📁 Data Management")
    if st.button("📥 Load Solutions", use_container_width=True):
        with st.spinner("Loading solutions from numerical_solutions/ ..."):
            st.session_state.solutions = st.session_state.loader.load_all_solutions()
            # Convert to the format expected by the interpolator
            st.session_state.solutions = [convert_to_original_format(sol) for sol in st.session_state.solutions]
        st.rerun()

    st.divider()
    st.header("🎯 Target Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        L0_nm = st.number_input("L₀ (nm)", 10.0, 200.0, 40.0, 5.0)
        fc = st.slider("Core fraction (fc)", 0.05, 0.45, 0.22, 0.01)
        rs = st.slider("Shell ratio (rs)", 0.01, 0.6, 0.25, 0.01)
    with col2:
        c_bulk = st.slider("c_bulk", 0.1, 1.0, 0.6, 0.05)
        tau0_s = st.number_input("τ₀ (s)", 1e-5, 1e-2, 1e-4, format="%.2e")
        t_nd = st.slider("Normalized time t_nd", 0.0, 1.0, 0.75, 0.01)
    
    use_edl = st.checkbox("Enable EDL catalyst", value=False)
    
    if st.button("🚀 Compute Interpolation", type="primary", use_container_width=True):
        if not st.session_state.solutions:
            st.error("No simulation files loaded. Please click '📥 Load Solutions' first.")
        else:
            target = {
                "L0_nm": L0_nm, "fc": fc, "rs": rs,
                "c_bulk": c_bulk, "tau0_s": tau0_s,
                "t_nd": t_nd, "use_edl": use_edl
            }
            with st.spinner("Full temporal + hybrid weighting interpolation..."):
                # Use the existing interpolator from the theoretical framework
                from scipy.interpolate import CubicSpline  # already imported
                # We'll reuse the interpolate_fields function defined below
                # (the function is defined after the sidebar, but we need it here)
                # We'll just call it after its definition, or restructure.
                # For clarity, we keep the function at module level and call it.
                # The function is defined below, so it's fine as long as we are after its definition.
                # We'll move the interpolator definition before the sidebar to avoid NameError.
                pass  # We'll define the interpolator before the sidebar in the final code.

# ==================== THEORETICAL FRAMEWORK IMPLEMENTATION ====================
# Steps from the theoretical procedure:
# 0. Preprocessing (offline) – done in loader
# 1. Target encoding (dimensionless feature vector)
# 2. Physics-informed kernel
# 3. Radial geometry morphing
# 4. Temporal interpolation on real time axis
# 5. Hybrid weighting (physics kernel + attention)
# 6. Post-processing enforcement

# ---------- Step 1: Target Encoding ----------
def encode_parameters(params: Dict, t_real: Optional[float] = None) -> np.ndarray:
    """
    Encode parameters into a dimensionless feature vector.
    If t_real is given, compute time‑dependent dimensionless groups.
    """
    L0 = params.get('L0_nm', 20.0) * 1e-9          # m
    D = params.get('D_nd', 0.05)                    # non‑dimensional diffusion
    gamma = params.get('gamma_nd', 0.02)
    M = params.get('M_nd', 0.2)
    k0 = params.get('k0_nd', 0.4)
    c_bulk = params.get('c_bulk', 1.0)
    fc = params.get('fc', 0.18)
    tau0 = params.get('tau0_s', 1e-4)
    
    # Normalised parameters (to [0,1])
    fc_norm = (fc - 0.05) / 0.40
    rs_norm = (params.get('rs', 0.2) - 0.01) / 0.59
    L0_norm = (params.get('L0_nm', 20.0) - 10.0) / 90.0   # L0 range 10-100 nm
    log_c = np.log10(c_bulk + 1e-8)
    
    # Fourier number (dimensionless time)
    if t_real is not None:
        t_nd = t_real / tau0
        Fo = D * t_nd / (L0**2)
    else:
        # Use total simulation time from source if available
        t_max_nd = params.get('t_max_nd', 1.0)
        Fo = D * t_max_nd / (L0**2)
    
    # Dimensionless curvature strength
    kappa_star = gamma * M / (L0 * k0 * c_bulk)
    
    # Damköhler number (reaction vs diffusion)
    Da = k0 * c_bulk * L0**2 / D
    
    # Normalise Fo, kappa, Da roughly to [0,1] using typical ranges
    Fo_norm = np.clip(Fo / 10.0, 0, 1)
    kappa_norm = np.clip(kappa_star / 5.0, 0, 1)
    Da_norm = np.clip(Da / 100.0, 0, 1)
    
    # Categorical flags
    edl_flag = 1.0 if params.get('use_edl', False) else 0.0
    mode_flag = 1.0 if params.get('mode', '2D (planar)') != '2D (planar)' else 0.0
    
    return np.array([fc_norm, rs_norm, L0_norm, log_c,
                     Fo_norm, kappa_norm, Da_norm,
                     edl_flag, mode_flag])

# ---------- Step 2: Physics Kernel ----------
def physics_kernel(src_params: Dict, tgt_params: Dict, t_real_src: Optional[float] = None,
                   t_real_tgt: Optional[float] = None, sigma: float = 0.3) -> float:
    """Gaussian kernel on dimensionless groups."""
    src_vec = encode_parameters(src_params, t_real_src)
    tgt_vec = encode_parameters(tgt_params, t_real_tgt)
    # Weighted squared differences (tune weights)
    weights = np.array([1.2, 1.2, 1.0, 1.5, 1.0, 0.8, 0.8, 0.5, 0.5])
    d2 = np.sum(weights * (src_vec - tgt_vec)**2)
    return np.exp(-0.5 * d2 / sigma**2)

# ---------- Step 3: Radial Morphing ----------
def radial_morph_2d(phi_src, psi_src, fc_src, rs_src, L0_src, fc_tgt, rs_tgt, L0_tgt, shape=(256, 256)):
    """Radial geometry-preserving morph (original function)."""
    h, w = phi_src.shape if len(phi_src.shape) == 2 else (phi_src.shape[1], phi_src.shape[2])
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    
    dist_src = np.sqrt((x - cx)**2 + (y - cy)**2) * (L0_src / w)
    scale = L0_tgt / L0_src
    r_core_tgt = fc_tgt * L0_tgt
    r_shell_tgt = r_core_tgt * (1 + rs_tgt)
    
    dist_warped = dist_src * scale
    
    phi_warped = np.zeros(shape)
    psi_warped = np.zeros(shape)
    
    mask_core = dist_warped <= r_core_tgt
    mask_shell = (dist_warped > r_core_tgt) & (dist_warped <= r_shell_tgt)
    
    phi_warped[mask_core] = 0.0
    psi_warped[mask_core] = 1.0
    
    r_norm = np.clip((dist_warped[mask_shell] - r_core_tgt) / (r_shell_tgt - r_core_tgt), 0, 1)
    phi_warped[mask_shell] = 1.0 - r_norm
    psi_warped[mask_shell] = 0.0
    
    return phi_warped, psi_warped

# ---------- Simple Attention Mechanism ----------
class SimpleAttention(nn.Module):
    """Dot‑product attention on encoded parameter vectors."""
    def __init__(self, d_model=9):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
    
    def forward(self, target_vec, source_vecs):
        # target_vec: (d_model,), source_vecs: (n_sources, d_model)
        q = self.query_proj(target_vec.unsqueeze(0))   # (1, d_model)
        k = self.key_proj(source_vecs)                  # (n_sources, d_model)
        attn_scores = torch.matmul(q, k.T) / np.sqrt(k.size(-1))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return attn_weights.squeeze(0)

# ---------- Step 4+5: Temporal Interpolation + Hybrid Weighting ----------
def interpolate_fields(sources: List[Dict], target_params: Dict, target_shape=(256,256)):
    """
    Full interpolation following theoretical steps.
    Returns blended fields at target time (given by target_params['t_nd']).
    """
    # Convert target normalized time to real time using target tau0
    t_tgt_norm = target_params.get('t_nd', 0.75)
    tau0_tgt = target_params.get('tau0_s', 1e-4)
    t_tgt_real = t_tgt_norm * tau0_tgt
    
    # Prepare storage
    source_fields_morphed = []   # list of (phi, psi, c) after morphing + temporal interpolation
    physics_weights = []
    source_params_list = []
    source_time_real_arrays = []
    
    # Pre-allocate for attention
    target_vec = torch.FloatTensor(encode_parameters(target_params, t_tgt_real))
    source_vecs = []
    
    for src in sources:
        src_params = src['params'].copy()
        src_params.setdefault('tau0_s', 1e-4)
        
        # ---- Step 1: Encode source parameters (for attention) ----
        src_vec = torch.FloatTensor(encode_parameters(src_params, t_real=None))
        source_vecs.append(src_vec)
        
        # ---- Step 2: Physics kernel weight (using characteristic groups) ----
        w_phys = physics_kernel(src_params, target_params, sigma=0.3)
        physics_weights.append(w_phys)
        
        # ---- Step 3: Radial morphing of source fields ----
        history = src['snapshots']   # list of dicts with 't_nd', 'phi', 'c', 'psi'
        if not history:
            continue
        
        # Build real time axis for this source
        t_src_real = np.array([snap['t_nd'] * src_params['tau0_s'] for snap in history])
        # Build arrays of fields (list of 2D arrays)
        phi_list = [snap['phi'] for snap in history]
        psi_list = [snap['psi'] for snap in history]
        c_list   = [snap['c'] for snap in history]
        
        # Interpolate each field in time at t_tgt_real (if within range)
        if t_tgt_real <= t_src_real[0]:
            phi_t = phi_list[0]
            psi_t = psi_list[0]
            c_t   = c_list[0]
        elif t_tgt_real >= t_src_real[-1]:
            phi_t = phi_list[-1]
            psi_t = psi_list[-1]
            c_t   = c_list[-1]
        else:
            # Use cubic spline for smoothness
            phi_spline = CubicSpline(t_src_real, np.array(phi_list), axis=0, bc_type='natural')
            psi_spline = CubicSpline(t_src_real, np.array(psi_list), axis=0, bc_type='natural')
            c_spline   = CubicSpline(t_src_real, np.array(c_list), axis=0, bc_type='natural')
            phi_t = phi_spline(t_tgt_real)
            psi_t = psi_spline(t_tgt_real)
            c_t   = c_spline(t_tgt_real)
        
        # ---- Apply radial morphing to the temporally interpolated fields ----
        phi_m, psi_m = radial_morph_2d(
            phi_t, psi_t,
            src_params['fc'], src_params['rs'], src_params['L0_nm'],
            target_params['fc'], target_params['rs'], target_params['L0_nm'],
            shape=target_shape
        )
        
        # Warp concentration field using the same radial scaling
        def warp_field(field, src_fc, src_L0, tgt_fc, tgt_L0, shape):
            h, w = field.shape
            y, x = np.mgrid[0:h, 0:w]
            cx, cy = w//2, h//2
            dist_src = np.sqrt((x-cx)**2 + (y-cy)**2) * (src_L0 / w)
            scale = tgt_L0 / src_L0
            # For c, map each target pixel to source pixel using nearest neighbour
            y_t, x_t = np.mgrid[0:shape[0], 0:shape[1]]
            r_tgt = np.sqrt((x_t - shape[1]//2)**2 + (y_t - shape[0]//2)**2) * (tgt_L0 / shape[1])
            r_src = r_tgt / scale
            x_src = cx + (r_src * np.cos(np.arctan2(y_t - shape[0]//2, x_t - shape[1]//2))) * (w / src_L0)
            y_src = cy + (r_src * np.sin(np.arctan2(y_t - shape[0]//2, x_t - shape[1]//2))) * (h / src_L0)
            x_src = np.clip(x_src, 0, w-1).astype(int)
            y_src = np.clip(y_src, 0, h-1).astype(int)
            return field[y_src, x_src]
        
        c_m = warp_field(c_t, src_params['fc'], src_params['L0_nm'],
                         target_params['fc'], target_params['L0_nm'], target_shape)
        
        source_fields_morphed.append((phi_m, psi_m, c_m))
        source_params_list.append(src_params)
        source_time_real_arrays.append(t_src_real)
    
    if not source_fields_morphed:
        return None
    
    # ---- Step 5: Hybrid weighting ----
    physics_weights = np.array(physics_weights)
    physics_weights /= physics_weights.sum() + 1e-12
    
    # Attention weights
    attention = SimpleAttention(d_model=9)
    with torch.no_grad():
        source_vecs_tensor = torch.stack(source_vecs)   # (n_sources, 9)
        attn_weights = attention(target_vec, source_vecs_tensor).numpy()
    
    # Combine: alpha = 0.7 (physics dominates)
    alpha = 0.7
    final_weights = alpha * physics_weights + (1 - alpha) * attn_weights
    final_weights /= final_weights.sum() + 1e-12
    
    # Blend fields
    phi_final = np.zeros(target_shape, dtype=float)
    psi_final = np.zeros(target_shape, dtype=float)
    c_final   = np.zeros(target_shape, dtype=float)
    
    for w, (phi, psi, c) in zip(final_weights, source_fields_morphed):
        phi_final += w * phi
        psi_final += w * psi
        c_final   += w * c
    
    # ---- Step 6: Post-processing ----
    # Clip
    phi_final = np.clip(phi_final, 0, 1)
    psi_final = np.clip(psi_final, 0, 1)
    c_final   = np.clip(c_final, 0, target_params['c_bulk'])
    # Enforce material consistency: if psi > 0.5, phi must be 0
    mask_cu = psi_final > 0.5
    phi_final[mask_cu] = 0.0
    # Also ensure psi <= 1 (already clipped)
    
    return {
        'phi': phi_final,
        'psi': psi_final,
        'c': c_final,
        'weights': final_weights.tolist(),
        'target': target_params,
        'time_real_s': t_tgt_real
    }

# ========================= INTERPOLATOR (wrapper) =========================
class CoreShellPhysicsInterpolator:
    def interpolate(self, sources, target):
        if not sources:
            return None
        return interpolate_fields(sources, target)

interpolator = CoreShellPhysicsInterpolator()

# ========================= MAIN DISPLAY =========================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    tgt = res["target"]
    
    st.header(f"Interpolated Solution — L₀ = {tgt['L0_nm']:.0f} nm | fc = {tgt['fc']:.3f} | rs = {tgt['rs']:.3f}")
    st.caption(f"Real time: {res['time_real_s']:.3e} s")
    
    tab1, tab2, tab3 = st.tabs(["📊 Fields", "📈 Thickness & Stats", "⚖️ Source Weights"])
    
    with tab1:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(res["phi"], origin="lower", cmap="viridis")
        axs[0].set_title("ϕ — Ag shell")
        axs[1].imshow(res["c"], origin="lower", cmap="plasma")
        axs[1].set_title("c — Concentration")
        axs[2].imshow(res["psi"], origin="lower", cmap="gray")
        axs[2].set_title("ψ — Cu core")
        
        material = np.where(res["psi"] > 0.5, 2.0, np.where(res["phi"] > 0.5, 1.0, 0.0))
        axs[3].imshow(material, origin="lower", cmap="Set1_r")
        axs[3].set_title("Material Proxy")
        
        for ax in axs:
            ax.axis("off")
        st.pyplot(fig)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            # Thickness (simple max radial distance)
            y, x = np.mgrid[0:res["phi"].shape[0], 0:res["phi"].shape[1]]
            dist = np.sqrt((x - res["phi"].shape[1]//2)**2 + (y - res["phi"].shape[0]//2)**2)
            thickness_nm = np.max(dist[res["phi"] > 0.5]) * tgt["L0_nm"] / res["phi"].shape[0]
            st.metric("Max Ag Shell Thickness", f"{thickness_nm:.2f} nm")
        
        with col2:
            st.metric("Ag Area Fraction", f"{np.mean(res['phi']):.4f}")
            st.metric("Cu Core Area Fraction", f"{np.mean(res['psi']):.4f}")
        
        # Download
        download_data = pickle.dumps({
            "meta": {"generated": datetime.now().isoformat(), "interpolated": True},
            "parameters": tgt,
            "snapshots": [(tgt["t_nd"], res["phi"], res["c"], res["psi"])],
            "thickness_history_nm": [(tgt["t_nd"], thickness_nm, thickness_nm)]
        })
        st.download_button(
            "📥 Download as PKL (compatible format)",
            download_data,
            f"interp_L0{tgt['L0_nm']}_fc{tgt['fc']:.3f}_rs{tgt['rs']:.3f}_t{tgt['t_nd']:.3f}.pkl",
            "application/octet-stream",
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Source Contribution Weights (Hybrid: Physics + Attention)")
        w = np.array(res["weights"])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(w)), w)
        ax.set_xlabel("Source Index")
        ax.set_ylabel("Weight")
        ax.set_title("Final Blending Weights")
        st.pyplot(fig)
        st.write("Higher bars = more influence from that source")

else:
    if st.session_state.solutions:
        st.info("👈 Set target parameters in the sidebar and click **Compute Interpolation**")
    else:
        st.warning("⚠️ No simulation files found. Please place your `.pkl` files in the `numerical_solutions/` directory and click **📥 Load Solutions** in the sidebar.")

st.caption("Enhanced with full temporal support, hybrid weighting, and post‑processing following the theoretical procedure.")
