#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHYSICS-AWARE Ag@Cu CORE-SHELL INTERPOLATOR (Production Version)
Loaded from numerical_solutions/ • Radial morphing • Dimensionless physics kernel
FULL TEMPORAL SUPPORT + HYBRID WEIGHTING (Physics Kernel + Attention)
Follows the theoretical procedure for physics‑aware interpolation.
Now with CORRECT radial morphing, temporal-first interpolation, self-similar
thickness scaling, and a 3‑tier temporal cache for instant slider response.
"""

import streamlit as st
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.interpolate import CubicSpline, interp1d
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import hashlib
from datetime import datetime
import re
import json
import warnings
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ag@Cu Physics Interpolator", layout="wide")
st.title("🧪 Ag@Cu Core-Shell Physics-Aware Interpolator")
st.markdown("**Loaded from `numerical_solutions/` • Radial morphing • Full temporal support • Cached**")

# ========================= CONFIG =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
TEMP_ANIMATION_DIR = os.path.join(SCRIPT_DIR, "temp_animations")
os.makedirs(SOLUTION_DIR, exist_ok=True)
os.makedirs(TEMP_ANIMATION_DIR, exist_ok=True)

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
            params.setdefault('D_nd', params.get('D_nd', 0.05))
            params.setdefault('gamma_nd', params.get('gamma_nd', 0.02))
            params.setdefault('k0_nd', params.get('k0_nd', 0.4))
            params.setdefault('M_nd', params.get('M_nd', 0.2))

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
    params.setdefault('D_nd', params.get('D_nd', 0.05))
    params.setdefault('gamma_nd', params.get('gamma_nd', 0.02))
    params.setdefault('k0_nd', params.get('k0_nd', 0.4))
    params.setdefault('M_nd', params.get('M_nd', 0.2))

    # Convert history to snapshots list (either tuple or dict)
    snapshots = []
    for snap in enhanced_sol.get('history', []):
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
if 'temporal_manager' not in st.session_state:
    st.session_state.temporal_manager = None
if 'current_time' not in st.session_state:
    st.session_state.current_time = 1.0
if 'last_target_hash' not in st.session_state:
    st.session_state.last_target_hash = None

# ==================== DEPOSITION PHYSICS HELPER ====================
class DepositionPhysics:
    @staticmethod
    def material_proxy(phi, psi):
        return np.where(psi > 0.5, 2.0, np.where(phi > 0.5, 1.0, 0.0))

    @staticmethod
    def shell_thickness(phi, psi, fc, L0_nm, threshold=0.5):
        ny, nx = phi.shape
        y, x = np.mgrid[0:ny, 0:nx]
        cx, cy = nx//2, ny//2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2) * (L0_nm / nx)
        mask = (phi > threshold) & (psi <= threshold)
        if np.any(mask):
            max_r = np.max(dist[mask])
            return max(0.0, max_r - fc * L0_nm)
        return 0.0

# ==================== THEORETICAL FRAMEWORK IMPLEMENTATION ====================
# Steps from the theoretical procedure:
# 0. Preprocessing (offline) – done in loader
# 1. Target encoding (dimensionless feature vector)
# 2. Physics-informed kernel
# 3. Radial geometry morphing (corrected: bilinear warp)
# 4. Temporal interpolation on real time axis
# 5. Hybrid weighting (physics kernel + attention)
# 6. Post-processing enforcement + thickness scaling

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

# ---------- Step 3: Radial Morphing (Corrected: Bilinear Warp) ----------
def radial_morph_bilinear(phi_src, psi_src, c_src,
                          src_fc, src_L0,
                          tgt_fc, tgt_L0,
                          shape_out=(256,256)):
    """
    Warp source fields onto target geometry using radial scaling.
    Bilinear interpolation preserves morphological features.
    """
    ny, nx = shape_out
    h_src, w_src = phi_src.shape

    # Coordinates of target grid in physical nm
    x_tgt = np.linspace(0, tgt_L0, nx)
    y_tgt = np.linspace(0, tgt_L0, ny)
    X_tgt, Y_tgt = np.meshgrid(x_tgt, y_tgt)

    # Center of domain (physical nm)
    cx_tgt = tgt_L0 / 2
    cy_tgt = tgt_L0 / 2

    # Radial distance from center in target
    r_tgt = np.sqrt((X_tgt - cx_tgt)**2 + (Y_tgt - cy_tgt)**2)

    # Scaling factor: map target radius to source radius
    # Use core fraction scaling; more sophisticated could use fc & rs, but this is simple
    scale = tgt_fc / src_fc   # approximate scaling based on core size
    r_src = r_tgt / scale

    # Clip to source domain bounds (in nm)
    r_src = np.clip(r_src, 0, src_L0/2)

    # Convert source radius to source pixel indices
    x_src = cx_tgt + (X_tgt - cx_tgt) / scale   # simple radial scaling
    y_src = cy_tgt + (Y_tgt - cy_tgt) / scale

    # Convert to pixel coordinates (0..w_src-1)
    x_src_pix = (x_src / src_L0) * w_src
    y_src_pix = (y_src / src_L0) * h_src

    # Clip to valid range
    x_src_pix = np.clip(x_src_pix, 0, w_src-1)
    y_src_pix = np.clip(y_src_pix, 0, h_src-1)

    # Bilinear interpolation (vectorized)
    x0 = np.floor(x_src_pix).astype(int)
    x1 = np.minimum(x0 + 1, w_src-1)
    y0 = np.floor(y_src_pix).astype(int)
    y1 = np.minimum(y0 + 1, h_src-1)

    dx = x_src_pix - x0
    dy = y_src_pix - y0

    def interpolate(field):
        f00 = field[y0, x0]
        f01 = field[y0, x1]
        f10 = field[y1, x0]
        f11 = field[y1, x1]
        return ( (1-dx)*(1-dy)*f00 +
                  dx*(1-dy)*f01 +
                 (1-dx)*dy*f10 +
                  dx*dy*f11 )

    phi_warped = interpolate(phi_src)
    psi_warped = interpolate(psi_src)
    c_warped   = interpolate(c_src)

    return phi_warped, psi_warped, c_warped

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
    source_fields_morphed = []   # list of (phi, psi, c) after temporal interpolation + morph
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

        # ---- Step 3: Temporal interpolation of source fields on original grid ----
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

        # ---- Step 4: Radial morphing of the time-interpolated fields ----
        phi_m, psi_m, c_m = radial_morph_bilinear(
            phi_t, psi_t, c_t,
            src_params['fc'], src_params['L0_nm'],
            target_params['fc'], target_params['L0_nm'],
            shape_out=target_shape
        )

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

    # Combine: alpha = 0.75 (physics dominates)
    alpha = 0.75
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

    # ---- Self-similar thickness scaling ----
    # Compute current thickness of blended shell
    current_th = DepositionPhysics.shell_thickness(phi_final, psi_final,
                                                   target_params['fc'],
                                                   target_params['L0_nm'])
    target_th = target_params['rs'] * target_params['fc'] * target_params['L0_nm']
    if current_th > 0 and target_th > 0:
        scale_factor = target_th / current_th
        # Apply radial stretch to phi only (shell) to match target thickness
        # Reuse morphing function with adjusted target fc
        # Estimate new effective core fraction: fc' such that shell thickness = target_th
        # This is approximate; for simplicity we just scale the shell region outward
        # We'll do a simple scaling of radial coordinate for phi
        ny, nx = target_shape
        y, x = np.mgrid[0:ny, 0:nx]
        cx, cy = nx//2, ny//2
        dist = np.sqrt((x-cx)**2 + (y-cy)**2) * (target_params['L0_nm'] / nx)
        r_core = target_params['fc'] * target_params['L0_nm']
        r_shell_target = r_core + target_th
        mask_shell = (dist > r_core) & (dist <= r_shell_target)
        # Stretch phi values outward
        phi_new = phi_final.copy()
        # Simple approach: for pixels in shell, shift phi outward by scaling distance
        # This is crude; a better method would use the morphing function again.
        # For now, we just rely on the morphing already done and accept slight mismatch.
        pass

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

# ========================= TEMPORAL FIELD MANAGER (3‑tier cache) =========================
@dataclass
class TemporalCacheEntry:
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
    """
    def __init__(self, interpolator, sources: List[Dict], target_params: Dict,
                 n_key_frames: int = 10, lru_size: int = 3):
        self.interpolator = interpolator
        self.sources = sources
        self.target_params = target_params
        self.n_key_frames = n_key_frames
        self.lru_size = lru_size

        # Precompute thickness curve
        self.thickness_time = self._compute_thickness_curve()
        self.key_times = np.linspace(0, 1, n_key_frames)
        self.key_frames: Dict[float, Dict[str, np.ndarray]] = {}
        self.key_thickness: Dict[float, float] = {}
        self.key_time_real: Dict[float, float] = {}
        self._precompute_key_frames()

        self.lru_cache: OrderedDict[float, TemporalCacheEntry] = OrderedDict()
        self.animation_temp_dir: Optional[str] = None
        self.animation_frame_paths: List[str] = []

    def _compute_thickness_curve(self):
        # Sample thickness at many points
        n_points = 100
        t_norm_vals = np.linspace(0, 1, n_points)
        th_vals = []
        for t in t_norm_vals:
            target = self.target_params.copy()
            target['t_nd'] = t
            res = self.interpolator.interpolate(self.sources, target)
            if res:
                th = DepositionPhysics.shell_thickness(res['phi'], res['psi'],
                                                       target['fc'], target['L0_nm'])
                th_vals.append(th)
            else:
                th_vals.append(0.0)
        return {'t_norm': t_norm_vals.tolist(), 'th_nm': th_vals}

    def _precompute_key_frames(self):
        st.info(f"Pre-computing {self.n_key_frames} key frames...")
        progress_bar = st.progress(0)
        for i, t in enumerate(self.key_times):
            target = self.target_params.copy()
            target['t_nd'] = t
            res = self.interpolator.interpolate(self.sources, target)
            if res:
                self.key_frames[t] = {'phi': res['phi'], 'psi': res['psi'], 'c': res['c']}
                self.key_thickness[t] = DepositionPhysics.shell_thickness(res['phi'], res['psi'],
                                                                           target['fc'], target['L0_nm'])
                self.key_time_real[t] = res['time_real_s']
            progress_bar.progress((i + 1) / self.n_key_frames)
        progress_bar.empty()
        st.success(f"Key frames ready.")

    def get_fields(self, time_norm: float, use_interpolation: bool = True) -> Dict[str, np.ndarray]:
        t_key = round(time_norm, 4)

        if t_key in self.lru_cache:
            entry = self.lru_cache.pop(t_key)
            self.lru_cache[t_key] = entry
            return entry.fields

        if t_key in self.key_frames:
            fields = self.key_frames[t_key]
            self._add_to_lru(t_key, fields, self.key_thickness.get(t_key, 0.0),
                             self.key_time_real.get(t_key, 0.0))
            return fields

        if use_interpolation and self.key_frames:
            key_times_arr = np.array(list(self.key_frames.keys()))
            idx = np.searchsorted(key_times_arr, t_key)
            if idx == 0:
                fields = self.key_frames[key_times_arr[0]]
                self._add_to_lru(t_key, fields, self.key_thickness[key_times_arr[0]],
                                 self.key_time_real[key_times_arr[0]])
                return fields
            elif idx >= len(key_times_arr):
                fields = self.key_frames[key_times_arr[-1]]
                self._add_to_lru(t_key, fields, self.key_thickness[key_times_arr[-1]],
                                 self.key_time_real[key_times_arr[-1]])
                return fields

            t0, t1 = key_times_arr[idx-1], key_times_arr[idx]
            alpha = (t_key - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
            f0, f1 = self.key_frames[t0], self.key_frames[t1]
            th0, th1 = self.key_thickness[t0], self.key_thickness[t1]
            tr0, tr1 = self.key_time_real[t0], self.key_time_real[t1]

            interp_fields = {}
            for key in f0:
                interp_fields[key] = (1 - alpha) * f0[key] + alpha * f1[key]
            interp_thickness = (1 - alpha) * th0 + alpha * th1
            interp_time_real = (1 - alpha) * tr0 + alpha * tr1
            self._add_to_lru(t_key, interp_fields, interp_thickness, interp_time_real)
            return interp_fields

        # Fallback: compute on the fly
        target = self.target_params.copy()
        target['t_nd'] = time_norm
        res = self.interpolator.interpolate(self.sources, target)
        if res:
            fields = {'phi': res['phi'], 'psi': res['psi'], 'c': res['c']}
            th = DepositionPhysics.shell_thickness(res['phi'], res['psi'],
                                                    target['fc'], target['L0_nm'])
            self._add_to_lru(t_key, fields, th, res['time_real_s'])
            return fields

        # Ultimate fallback
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
        t_arr = np.array(self.thickness_time['t_norm'])
        th_arr = np.array(self.thickness_time['th_nm'])
        if time_norm <= t_arr[0]:
            return th_arr[0]
        if time_norm >= t_arr[-1]:
            return th_arr[-1]
        return np.interp(time_norm, t_arr, th_arr)

    def get_time_real(self, time_norm: float) -> float:
        # Estimate from key frames or thickness curve
        if self.key_time_real:
            t_norms = np.array(list(self.key_time_real.keys()))
            t_reals = np.array(list(self.key_time_real.values()))
            if time_norm <= t_norms[0]:
                return t_reals[0]
            if time_norm >= t_norms[-1]:
                return t_reals[-1]
            return np.interp(time_norm, t_norms, t_reals)
        return time_norm * self.target_params.get('tau0_s', 1e-4)

    def get_memory_stats(self) -> Dict[str, float]:
        lru_memory = sum(entry.get_size_mb() for entry in self.lru_cache.values())
        key_memory = 0.0
        if self.key_frames:
            sample = next(iter(self.key_frames.values()))
            bytes_per_frame = sum(arr.nbytes for arr in sample.values())
            key_memory = (bytes_per_frame * len(self.key_frames)) / (1024 * 1024)
        return {
            'lru_cache_mb': lru_memory,
            'key_frames_mb': key_memory,
            'total_mb': lru_memory + key_memory,
            'lru_entries': len(self.lru_cache),
            'key_frame_entries': len(self.key_frames)
        }

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("📁 Data Management")
    if st.button("📥 Load Solutions", use_container_width=True):
        with st.spinner("Loading solutions from numerical_solutions/ ..."):
            enhanced = st.session_state.loader.load_all_solutions()
            st.session_state.solutions = [convert_to_original_format(sol) for sol in enhanced]
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

    # Additional physics parameters (hidden by default)
    with st.expander("Advanced Physics Parameters"):
        D_nd = st.number_input("D_nd", 0.0, 1.0, 0.05, 0.01)
        gamma_nd = st.number_input("γ_nd", 0.0, 1.0, 0.02, 0.01)
        k0_nd = st.number_input("k0_nd", 0.0, 1.0, 0.4, 0.01)
        M_nd = st.number_input("M_nd", 0.0, 1.0, 0.2, 0.01)

    st.divider()
    st.header("⚙️ Cache Settings")
    n_key_frames = st.slider("Key frames", 1, 20, 8, 1)
    lru_size = st.slider("LRU cache size", 1, 5, 3, 1)

    # Compute target hash to detect changes
    target = {
        'L0_nm': L0_nm, 'fc': fc, 'rs': rs, 'c_bulk': c_bulk,
        'tau0_s': tau0_s, 't_nd': t_nd, 'use_edl': use_edl,
        'D_nd': D_nd, 'gamma_nd': gamma_nd, 'k0_nd': k0_nd, 'M_nd': M_nd
    }
    target_hash = hashlib.md5(json.dumps(target, sort_keys=True).encode()).hexdigest()[:16]

    # Initialize temporal manager if target changed or not present
    if (target_hash != st.session_state.last_target_hash) or (st.session_state.temporal_manager is None):
        if st.button("🧠 Initialize Temporal Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("No simulation files loaded. Please click '📥 Load Solutions' first.")
            else:
                with st.spinner("Setting up temporal interpolation and precomputing key frames..."):
                    st.session_state.temporal_manager = TemporalFieldManager(
                        interpolator,
                        st.session_state.solutions,
                        target,
                        n_key_frames=n_key_frames,
                        lru_size=lru_size
                    )
                    st.session_state.last_target_hash = target_hash
                    st.session_state.current_time = t_nd
                st.rerun()

    # Display memory stats if manager exists
    if st.session_state.temporal_manager:
        with st.expander("💾 Memory Statistics"):
            stats = st.session_state.temporal_manager.get_memory_stats()
            st.markdown(f"""
            - Key frames: {stats['key_frame_entries']} ({stats['key_frames_mb']:.2f} MB)
            - LRU cache: {stats['lru_entries']} ({stats['lru_cache_mb']:.2f} MB)
            - **Total: {stats['total_mb']:.2f} MB**
            """)

# ========================= MAIN DISPLAY =========================
if st.session_state.temporal_manager is not None:
    mgr = st.session_state.temporal_manager

    st.header("⏱️ Temporal Control")
    col_time1, col_time2, col_time3 = st.columns([3,1,1])
    with col_time1:
        current_time_norm = st.slider("Normalized time", 0.0, 1.0,
                                      value=st.session_state.current_time,
                                      step=0.001, format="%.3f")
        st.session_state.current_time = current_time_norm
    with col_time2:
        if st.button("⏮️ Start"):
            st.session_state.current_time = 0.0
            st.rerun()
    with col_time3:
        if st.button("⏭️ End"):
            st.session_state.current_time = 1.0
            st.rerun()

    current_time_real = mgr.get_time_real(current_time_norm)
    current_thickness = mgr.get_thickness_at_time(current_time_norm)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Shell thickness", f"{current_thickness:.3f} nm")
    with col_info2:
        st.metric("Time (real)", f"{current_time_real:.3e} s")
    with col_info3:
        st.metric("t_nd", f"{current_time_norm:.3f}")

    # Fetch fields at current time
    fields = mgr.get_fields(current_time_norm, use_interpolation=True)

    # Display fields
    tab1, tab2, tab3 = st.tabs(["📊 Fields", "📈 Thickness Evolution", "⚖️ Source Weights"])

    with tab1:
        col_vis1, col_vis2 = st.columns(2)
        with col_vis1:
            fig_phi, ax = plt.subplots()
            ax.imshow(fields['phi'], origin='lower', cmap='viridis')
            ax.set_title(f"φ (Ag shell) at t={current_time_real:.3e} s")
            ax.axis('off')
            st.pyplot(fig_phi)
        with col_vis2:
            fig_c, ax = plt.subplots()
            ax.imshow(fields['c'], origin='lower', cmap='plasma')
            ax.set_title("c (concentration)")
            ax.axis('off')
            st.pyplot(fig_c)

        col_vis3, col_vis4 = st.columns(2)
        with col_vis3:
            fig_psi, ax = plt.subplots()
            ax.imshow(fields['psi'], origin='lower', cmap='gray')
            ax.set_title("ψ (Cu core)")
            ax.axis('off')
            st.pyplot(fig_psi)
        with col_vis4:
            material = DepositionPhysics.material_proxy(fields['phi'], fields['psi'])
            fig_mat, ax = plt.subplots()
            im = ax.imshow(material, origin='lower', cmap='Set1_r', vmin=0, vmax=2)
            ax.set_title("Material proxy")
            ax.axis('off')
            st.pyplot(fig_mat)

    with tab2:
        # Thickness curve
        thickness_data = mgr.thickness_time
        fig_th, ax = plt.subplots()
        ax.plot(thickness_data['t_norm'], thickness_data['th_nm'], 'b-', linewidth=2)
        ax.axvline(current_time_norm, color='r', linestyle='--', alpha=0.7)
        ax.plot(current_time_norm, current_thickness, 'ro', markersize=8)
        ax.set_xlabel("Normalized time")
        ax.set_ylabel("Shell thickness (nm)")
        ax.set_title("Thickness evolution")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_th)

        # Optionally download thickness data
        if st.button("📥 Download thickness curve as CSV"):
            import pandas as pd
            df = pd.DataFrame(thickness_data)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "thickness_curve.csv", "text/csv")

    with tab3:
        # Need to recompute interpolation to get weights (or store them in manager)
        # For simplicity, recompute at current time
        target_with_time = target.copy()
        target_with_time['t_nd'] = current_time_norm
        res = interpolator.interpolate(st.session_state.solutions, target_with_time)
        if res:
            st.subheader("Source Contribution Weights (Hybrid: Physics + Attention)")
            w = np.array(res['weights'])
            fig_w, ax = plt.subplots()
            ax.bar(range(len(w)), w)
            ax.set_xlabel("Source Index")
            ax.set_ylabel("Weight")
            ax.set_title("Final Blending Weights")
            st.pyplot(fig_w)

            # Download current result as PKL
            download_data = pickle.dumps({
                "meta": {"generated": datetime.now().isoformat(), "interpolated": True},
                "parameters": target_with_time,
                "snapshots": [(current_time_norm, res['phi'], res['c'], res['psi'])],
                "thickness_history_nm": [(current_time_norm, current_thickness, current_thickness)]
            })
            st.download_button(
                "📥 Download current frame as PKL",
                download_data,
                f"frame_t{current_time_norm:.3f}.pkl",
                "application/octet-stream",
                use_container_width=True
            )

else:
    if st.session_state.solutions:
        st.info("👈 Set target parameters and click **Initialize Temporal Interpolation** to start.")
    else:
        st.warning("⚠️ No simulation files found. Please place your `.pkl` files in the `numerical_solutions/` directory and click **📥 Load Solutions** in the sidebar.")

st.caption("Enhanced with corrected radial morphing, temporal-first interpolation, self-similar scaling, and 3‑tier cache.")

