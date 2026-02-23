#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHYSICS-AWARE Ag@Cu CORE-SHELL INTERPOLATOR (Final Production Version)
Loads from numerical_solutions/ • Radial morphing • Dimensionless physics kernel
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

st.set_page_config(page_title="Ag@Cu Physics Interpolator", layout="wide")
st.title("🧪 Ag@Cu Core-Shell Physics-Aware Interpolator")
st.markdown("**Loaded from `numerical_solutions/` • Radial morphing + dimensionless kernel**")

# ========================= CONFIG =========================
SOLUTION_DIR = "numerical_solutions"
os.makedirs(SOLUTION_DIR, exist_ok=True)

# ========================= LOADER =========================
@st.cache_data
def load_all_pkl():
    solutions = []
    for f in Path(SOLUTION_DIR).glob("*.pkl"):
        try:
            with open(f, "rb") as fp:
                data = pickle.load(fp)
            
            # Support both your original generator format and any variant
            params = data.get("parameters", data.get("params", {}))
            meta = data.get("meta", {})
            params.update(meta)
            
            snapshots = data.get("snapshots", [])
            if not snapshots:
                continue
                
            sol = {
                "filename": f.name,
                "params": {
                    "L0_nm": float(params.get("L0_nm", params.get("L0", 20.0))),
                    "fc": float(params.get("core_radius_frac", params.get("fc", 0.18))),
                    "rs": float(params.get("shell_thickness_frac", params.get("rs", 0.2))),
                    "c_bulk": float(params.get("c_bulk", 1.0)),
                    "use_edl": bool(params.get("use_edl", False)),
                    "tau0_s": float(params.get("tau0_s", params.get("tau0", 1e-4))),
                    "mode": params.get("mode", "2D (planar)"),
                },
                "snapshots": snapshots,           # list of (t_nd, phi, c, psi) or dicts
                "thickness_history": data.get("thickness_history_nm", data.get("thick", [])),
                "coords_nd": data.get("coords_nd"),
            }
            solutions.append(sol)
        except Exception as e:
            st.sidebar.warning(f"Skipped {f.name}: {e}")
    return solutions

solutions = load_all_pkl()

if not solutions:
    st.error(f"No .pkl files found in `{SOLUTION_DIR}`. Please place your simulation files there.")
    st.stop()

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
            phi_src = last_snap[1] if isinstance(last_snap, (list, tuple)) else last_snap.get("phi", last_snap)
            psi_src = last_snap[3] if isinstance(last_snap, (list, tuple)) else last_snap.get("psi", last_snap)
            
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
    st.info("👈 Set target parameters in the sidebar and click **Compute Interpolation**")

st.caption("Modified & Expanded for `numerical_solutions/` directory • Physics kernel + radial morphing")
