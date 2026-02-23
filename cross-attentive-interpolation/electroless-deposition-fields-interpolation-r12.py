#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHYSICS-AWARE Ag@Cu CORE-SHELL INTERPOLATOR (Production Version)
Loaded from numerical_solutions/ • Radial morphing • Dimensionless physics kernel
Uses enhanced loader – warns instead of crashing if no files are found.
"""

import streamlit as st
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torch
import torch.nn as nn
from dataclasses import dataclass
import hashlib
from datetime import datetime
import re
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ag@Cu Physics Interpolator", layout="wide")
st.title("🧪 Ag@Cu Core-Shell Physics-Aware Interpolator")
st.markdown("**Loaded from `numerical_solutions/` • Radial morphing + dimensionless kernel**")

# ========================= CONFIG =========================
SOLUTION_DIR = "numerical_solutions"
os.makedirs(SOLUTION_DIR, exist_ok=True)

# ========================= ENHANCED LOADER (copied from second script) =========================
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

# ========================= LOADER =========================
loader = EnhancedSolutionLoader(SOLUTION_DIR)

@st.cache_data
def load_all_pkl():
    enhanced_solutions = loader.load_all_solutions(use_cache=True, max_files=None)
    # Convert each to the original format
    solutions = [convert_to_original_format(sol) for sol in enhanced_solutions]
    return solutions

solutions = load_all_pkl()

if not solutions:
    st.sidebar.warning("No simulation files found. Please place .pkl files in numerical_solutions/")
    # The app will continue, but the "Compute Interpolation" button will do nothing
else:
    st.sidebar.success(f"Loaded **{len(solutions)}** solutions from `numerical_solutions/`")

# ========================= PHYSICS KERNEL =========================
def compute_physics_kernel(src, tgt):
    """Dimensionless physics distance kernel"""
    # Fourier number (time scaling)
    Fo_s = 0.05 * src["params"]["tau0_s"] * src["snapshots"][-1][0] / (src["params"]["L0_nm"] ** 2)
    Fo_t = 0.05 * tgt["tau0_s"] * tgt["t_nd"] / (tgt["L0_nm"] ** 2)
    
    # Curvature number
    kappa_star_s = 0.02 * 0.2 / (src["params"]["L0_nm"] * 0.4 * src["params"]["c_bulk"])
    kappa_star_t = 0.02 * 0.2 / (tgt["L0_nm"] * 0.4 * tgt["c_bulk"])
    
    D2 = (
        ((Fo_s - Fo_t) / 0.25)**2 +
        ((kappa_star_s - kappa_star_t) / 0.6)**2 +
        ((src["params"]["fc"] - tgt["fc"]) / 0.08)**2 +
        ((src["params"]["rs"] - tgt["rs"]) / 0.12)**2 +
        (np.log10(max(src["params"]["c_bulk"], 1e-8) / max(tgt["c_bulk"], 1e-8)) / 0.5)**2
    )
    return np.exp(-0.5 * D2)

# ========================= RADIAL MORPHING =========================
def radial_morph_2d(phi_src, psi_src, fc_src, rs_src, L0_src, fc_tgt, rs_tgt, L0_tgt, shape=(256, 256)):
    """Radial geometry-preserving morph"""
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

# ========================= INTERPOLATOR =========================
class CoreShellPhysicsInterpolator:
    def interpolate(self, sources, target):
        if not sources:
            return None
            
        weights = []
        morphed = []
        
        for src in sources:
            w = compute_physics_kernel(src, target)
            weights.append(w)
            
            # Take last snapshot (or interpolate temporally if needed)
            last_snap = src["snapshots"][-1]
            # last_snap can be tuple or dict; handle both
            if isinstance(last_snap, (list, tuple)):
                phi_src = last_snap[1]
                psi_src = last_snap[3]
            else:
                phi_src = last_snap.get("phi", last_snap)
                psi_src = last_snap.get("psi", last_snap)
            
            phi_m, psi_m = radial_morph_2d(
                phi_src, psi_src,
                src["params"]["fc"], src["params"]["rs"], src["params"]["L0_nm"],
                target["fc"], target["rs"], target["L0_nm"]
            )
            morphed.append((phi_m, psi_m))
        
        weights = np.array(weights)
        weights /= weights.sum() + 1e-12
        
        # Blend
        phi_final = np.zeros_like(morphed[0][0])
        psi_final = np.zeros_like(morphed[0][1])
        
        for w, (p, s) in zip(weights, morphed):
            phi_final += w * p
            psi_final += w * s
        
        phi_final = np.clip(phi_final, 0, 1)
        psi_final = np.clip(psi_final, 0, 1)
        c_final = target["c_bulk"] * (1 - phi_final) * (1 - psi_final)
        
        return {
            "phi": phi_final,
            "psi": psi_final,
            "c": c_final,
            "weights": weights.tolist(),
            "target": target
        }

interpolator = CoreShellPhysicsInterpolator()

# ========================= SIDEBAR =========================
with st.sidebar:
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
        if not solutions:
            st.error("No simulation files loaded. Please add .pkl files to numerical_solutions/ and refresh.")
        else:
            target = {
                "L0_nm": L0_nm, "fc": fc, "rs": rs,
                "c_bulk": c_bulk, "tau0_s": tau0_s,
                "t_nd": t_nd, "use_edl": use_edl
            }
            with st.spinner("Radial morphing + physics kernel interpolation..."):
                result = interpolator.interpolate(solutions, target)
                if result:
                    st.session_state.last_result = result
                    st.success("✅ Interpolation successful!")

# ========================= MAIN DISPLAY =========================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    tgt = res["target"]
    
    st.header(f"Interpolated Solution — L₀ = {tgt['L0_nm']:.0f} nm | fc = {tgt['fc']:.3f} | rs = {tgt['rs']:.3f}")
    
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
        st.subheader("Source Contribution Weights")
        w = np.array(res["weights"])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(w)), w)
        ax.set_xlabel("Source Index")
        ax.set_ylabel("Weight")
        ax.set_title("Physics Kernel + Attention Weights")
        st.pyplot(fig)
        st.write("Higher bars = more influence from that source")

else:
    if solutions:
        st.info("👈 Set target parameters in the sidebar and click **Compute Interpolation**")
    else:
        st.warning("⚠️ No simulation files found. Please place your `.pkl` files in the `numerical_solutions/` directory and refresh the page.")

st.caption("Modified & Expanded for `numerical_solutions/` directory • Physics kernel + radial morphing • Enhanced loader")
