#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer-Inspired Interpolation for Electroless Ag Shell Deposition on Cu Core
FULL TEMPORAL SUPPORT + HYBRID WEIGHT QUANTIFICATION + PHYSICS-AWARE INTERPOLATION

KEY ENHANCEMENTS IN THIS VERSION:
1. HYBRID WEIGHT COMPUTATION - Combines individual parameter weights + attention + physics refinement
2. WEIGHT VISUALIZATION - Sankey, Chord, Radar, and Breakdown diagrams for weight analysis
3. PROPORTIONAL SOURCE WEIGHTING - Sources weighted by hybrid weight magnitude
4. INDIVIDUAL PARAMETER WEIGHTS - L0, fc, rs, c_bulk each get separate weight components
5. PHYSICS REFINEMENT FACTORS - Alpha (L0), Beta (params), Gamma (shape) refinement
6. TRANSFORMER ATTENTION - Learned attention scores combined with physics priors
7. WEIGHT DIAGNOSTICS - Entropy, max weight, effective sources for uncertainty quantification
8. BUG FIX - Fixed UnboundLocalError in _get_fields_at_time by ensuring phi, c, psi always initialized
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
from math import cos, sin, pi
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
# HYBRID WEIGHT VISUALIZER (CORRECTED)
# =============================================
class HybridWeightVisualizer:
    """
    Comprehensive weight visualization system for core-shell interpolation.
    Creates Sankey, Chord, Radar, and Breakdown diagrams for weight analysis.
    """
    
    def __init__(self):
        self.color_scheme = {
            'L0': '#FF6B6B',
            'fc': '#4ECDC4',
            'rs': '#95E1D3',
            'c_bulk': '#FFD93D',
            'Attention': '#9D4EDD',
            'Spatial': '#36A2EB',
            'Combined': '#9966FF',
            'Query': '#FF6B6B'
        }
        
        self.font_config = {
            'family': 'Arial, sans-serif',
            'size_title': 24,
            'size_labels': 18,
            'size_ticks': 14,
            'color': '#2C3E50'
            # weight removed – use HTML <b> tags in title strings instead
        }
    
    def get_colormap(self, cmap_name, n_colors=10):
        try:
            cmap = plt.get_cmap(cmap_name)
            return [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
                    for r, g, b, _ in [cmap(i/n_colors) for i in range(n_colors)]]
        except:
            cmap = plt.get_cmap('viridis')
            return [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
                    for r, g, b, _ in [cmap(i/n_colors) for i in range(n_colors)]]
    
    def create_enhanced_sankey_diagram(self, sources_data, target_params, param_sigmas):
        labels = ['Target']
        
        for source in sources_data:
            idx = source['source_index']
            l0 = source['L0_nm']
            fc = source['fc']
            labels.append(f"S{idx}\nL0={l0:.0f}nm\nfc={fc:.2f}")
        
        component_start = len(labels)
        labels.extend(['L0 Weight', 'fc Weight', 'rs Weight', 'c_bulk Weight', 
                      'Attention', 'Physics Refinement', 'Combined Weight'])
        
        source_indices = []
        target_indices = []
        values = []
        colors = []
        
        color_palette = self.get_colormap('viridis', len(sources_data) + 7)
        
        # Node colors: target, sources, then components
        node_colors = ["#FF6B6B"]  # target
        combined_weights = [s['combined_weight'] for s in sources_data]
        for w in combined_weights:
            node_colors.append(f'rgba(153,102,255,{w:.2f})')  # source nodes opacity = weight
        node_colors.extend(["#FF6B6B", "#4ECDC4", "#95E1D3", "#FFD93D", "#9D4EDD", "#36A2EB", "#9966FF"])
        
        for i, source in enumerate(sources_data):
            source_idx = i + 1
            source_color = color_palette[i % len(color_palette)]
            
            # L0 weight
            source_indices.append(source_idx)
            target_indices.append(component_start)
            values.append(source['l0_weight'] * 100)
            colors.append(f'rgba(255, 107, 107, 0.8)')
            
            # fc weight
            source_indices.append(source_idx)
            target_indices.append(component_start + 1)
            values.append(source['fc_weight'] * 100)
            colors.append(f'rgba(78, 205, 196, 0.8)')
            
            # rs weight
            source_indices.append(source_idx)
            target_indices.append(component_start + 2)
            values.append(source['rs_weight'] * 100)
            colors.append(f'rgba(149, 225, 211, 0.8)')
            
            # c_bulk weight
            source_indices.append(source_idx)
            target_indices.append(component_start + 3)
            values.append(source['c_bulk_weight'] * 100)
            colors.append(f'rgba(255, 217, 61, 0.8)')
            
            # Attention
            source_indices.append(source_idx)
            target_indices.append(component_start + 4)
            values.append(source['attention_weight'] * 100)
            colors.append(f'rgba(157, 78, 221, 0.8)')
            
            # Physics Refinement
            source_indices.append(source_idx)
            target_indices.append(component_start + 5)
            values.append(source['physics_refinement'] * 100)
            colors.append(f'rgba(54, 162, 235, 0.8)')
            
            # Combined Weight
            source_indices.append(source_idx)
            target_indices.append(component_start + 6)
            values.append(source['combined_weight'] * 100)
            colors.append(f'rgba(153, 102, 255, 0.8)')
        
        for comp_idx in range(7):
            source_indices.append(component_start + comp_idx)
            target_indices.append(0)
            comp_value = sum(v for s, t, v in zip(source_indices, target_indices, values)
                           if t == component_start + comp_idx)
            values.append(comp_value * 0.5)
            colors.append(f'rgba(153, 102, 255, 0.6)')
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=25,
                thickness=30,
                line=dict(color="black", width=2),
                label=labels,
                color=node_colors,
                hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<extra></extra>'
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value:.2f}<extra></extra>',
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            hoverinfo='all'
        )])
        
        fig.update_layout(
            title=dict(
                text=f'<b>SANKEY DIAGRAM: HYBRID WEIGHT COMPONENT FLOW</b><br>' + 
                     f'Target: L0={target_params.get("L0_nm", 20):.0f}nm, fc={target_params.get("fc", 0.18):.2f}',
                font=dict(family=self.font_config['family'], size=self.font_config['size_title'],
                         color=self.font_config['color']),  # removed weight
                x=0.5, y=0.95, xanchor='center', yanchor='top',
                pad=dict(b=20)
            ),
            font=dict(family=self.font_config['family'], size=self.font_config['size_labels'],
                     color=self.font_config['color']),  # removed weight
            width=1400,
            height=900,
            plot_bgcolor='rgba(240, 240, 245, 0.9)',
            paper_bgcolor='white',
            margin=dict(t=100, l=50, r=50, b=50),
            hoverlabel=dict(
                font=dict(family=self.font_config['family'], size=self.font_config['size_labels'],
                         color='white'),  # removed weight
                bgcolor='rgba(44, 62, 80, 0.9)',
                bordercolor='white',
                pad=dict(l=10, r=10, t=5, b=5)
            )
        )
        
        return fig
    
    def create_enhanced_chord_diagram(self, sources_data, target_params):
        n_sources = len(sources_data)
        center_x, center_y = 0, 0
        radius = 1.5
        
        source_positions = []
        for i in range(n_sources):
            angle = 2 * pi * i / n_sources
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            source_positions.append((x, y))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[center_x], y=[center_y],
            mode='markers+text',
            name='Target',
            marker=dict(size=50, color=self.color_scheme['Query'],
                       symbol='star', line=dict(width=3, color='white')),
            text=[f'Target\nL0={target_params.get("L0_nm", 20):.0f}nm\nfc={target_params.get("fc", 0.18):.2f}'],
            textposition="middle center",
            textfont=dict(size=16, color='white', family=self.font_config['family']),
            hoverinfo='text',
            hovertemplate='<b>Target</b><br>L0=%{text}<extra></extra>'
        ))
        
        source_x = []
        source_y = []
        source_colors = []
        source_sizes = []
        combined_weights = [s['combined_weight'] for s in sources_data]
        
        for i, source in enumerate(sources_data):
            x, y = source_positions[i]
            source_x.append(x)
            source_y.append(y)
            node_size = 20 + source['combined_weight'] * 60
            source_sizes.append(node_size)
            source_colors.append(self.color_scheme['Combined'])
        
        fig.add_trace(go.Scatter(
            x=source_x, y=source_y,
            mode='markers+text',
            name='Sources',
            marker=dict(size=source_sizes, color=source_colors,
                       line=dict(width=2, color='white')),
            text=[f"S{i}" for i in range(n_sources)],
            textposition="top center",
            textfont=dict(size=12, color='white', family=self.font_config['family']),
            hoverinfo='text',
            hovertemplate='<b>Source %{text}</b><br>Combined weight: %{marker.size:.1f}<extra></extra>'
        ))
        
        # Draw curved links with thickness scaled by combined weight
        for i, source in enumerate(sources_data):
            sx, sy = source_positions[i]
            cx = (sx + center_x) / 2
            cy = (sy + center_y) / 2
            dx = center_x - sx
            dy = center_y - sy
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                nx = -dy / length * 0.3
                ny = dx / length * 0.3
            else:
                nx, ny = 0, 0
            
            t = np.linspace(0, 1, 50)
            
            combined_curve_x = (1-t)**2 * sx + 2*(1-t)*t * (cx - nx*1.0) + t**2 * center_x
            combined_curve_y = (1-t)**2 * sy + 2*(1-t)*t * (cy - ny*1.0) + t**2 * center_y
            combined_width = max(2, source['combined_weight'] * 20)
            # Color based on weight (using colormap)
            weight_norm = source['combined_weight'] / max(combined_weights) if max(combined_weights) > 0 else 0.5
            line_color = px.colors.sample_colorscale('viridis', weight_norm)[0]
            
            fig.add_trace(go.Scatter(
                x=combined_curve_x, y=combined_curve_y,
                mode='lines',
                name=f'Source {i} - Combined',
                line=dict(width=combined_width, color=line_color, dash='solid'),
                hoverinfo='text',
                hovertext=f"Combined Weight: {source['combined_weight']:.3f}",
                showlegend=False
            ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>ENHANCED CHORD DIAGRAM: HYBRID WEIGHT VISUALIZATION</b><br>' + 
                     f'Target: L0={target_params.get("L0_nm", 20):.0f}nm, fc={target_params.get("fc", 0.18):.2f}',
                font=dict(family=self.font_config['family'], size=self.font_config['size_title'],
                         color=self.font_config['color']),  # removed weight
                x=0.5, y=0.95,
                pad=dict(b=20)
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                       font=dict(family=self.font_config['family'], size=self.font_config['size_labels'],
                                color=self.font_config['color']),  # removed weight
                       bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black', borderwidth=1,
                       itemwidth=50, tracegroupgap=10),
            width=1200,
            height=1000,
            plot_bgcolor='rgba(240, 240, 245, 0.9)',
            paper_bgcolor='white',
            hovermode='closest',
            margin=dict(l=100, r=100, t=150, b=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5])
        )
        
        return fig
    
    # NEW: Parameter‑based radar charts
    def create_parameter_radar_charts(self, sources_data, target_params, param_sigmas):
        """
        Returns a list of four Plotly figures, each with angles based on
        a different source parameter (L0_nm, fc, rs, c_bulk).
        """
        param_keys = ['L0_nm', 'fc', 'rs', 'c_bulk']
        param_names = ['L0 (nm)', 'fc', 'rs', 'c_bulk']
        figs = []

        for pkey, pname in zip(param_keys, param_names):
            # Sort sources by this parameter to get smooth angular progression
            sorted_idx = np.argsort([s[pkey] for s in sources_data])
            sorted_sources = [sources_data[i] for i in sorted_idx]

            # Extract weights in sorted order
            l0_weights = [s['l0_weight'] for s in sorted_sources]
            fc_weights = [s['fc_weight'] for s in sorted_sources]
            rs_weights = [s['rs_weight'] for s in sorted_sources]
            c_weights = [s['c_bulk_weight'] for s in sorted_sources]
            attn_weights = [s['attention_weight'] for s in sorted_sources]
            combined_weights = [s['combined_weight'] for s in sorted_sources]

            # Normalize weights for radial axis (scale to 0-1)
            max_w = max(combined_weights) if combined_weights else 1
            l0_norm = [w / max_w for w in l0_weights]
            fc_norm = [w / max_w for w in fc_weights]
            rs_norm = [w / max_w for w in rs_weights]
            c_norm = [w / max_w for w in c_weights]
            attn_norm = [w / max_w for w in attn_weights]
            combined_norm = [w / max_w for w in combined_weights]

            # Map sorted parameter values to angles (0 to 360)
            p_vals = np.array([s[pkey] for s in sorted_sources])
            min_p, max_p = p_vals.min(), p_vals.max()
            if max_p == min_p:
                angles = np.linspace(0, 360, len(p_vals), endpoint=False)  # fallback
            else:
                angles = 360 * (p_vals - min_p) / (max_p - min_p)

            # Build figure
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=l0_norm, theta=angles, mode='lines+markers',
                name='α (L0)', line=dict(color=self.color_scheme['L0'], width=3),
                marker=dict(size=8, color=self.color_scheme['L0']),
                hovertemplate='<b>L0 Weight</b>: %{r:.3f}<br>Parameter: %{theta:.1f}°<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=fc_norm, theta=angles, mode='lines+markers',
                name='β_fc', line=dict(color=self.color_scheme['fc'], width=3, dash='dot'),
                marker=dict(size=8, color=self.color_scheme['fc']),
                hovertemplate='<b>fc Weight</b>: %{r:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=rs_norm, theta=angles, mode='lines+markers',
                name='β_rs', line=dict(color=self.color_scheme['rs'], width=3, dash='dash'),
                marker=dict(size=8, color=self.color_scheme['rs']),
                hovertemplate='<b>rs Weight</b>: %{r:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=c_norm, theta=angles, mode='lines+markers',
                name='β_c', line=dict(color=self.color_scheme['c_bulk'], width=3, dash='longdash'),
                marker=dict(size=8, color=self.color_scheme['c_bulk']),
                hovertemplate='<b>c_bulk Weight</b>: %{r:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=attn_norm, theta=angles, mode='lines+markers',
                name='Attention', line=dict(color=self.color_scheme['Attention'], width=3, dash='dot'),
                marker=dict(size=8, color=self.color_scheme['Attention']),
                hovertemplate='<b>Attention</b>: %{r:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatterpolar(
                r=combined_norm, theta=angles, mode='lines+markers',
                name='Hybrid Weight', line=dict(color=self.color_scheme['Combined'], width=4),
                marker=dict(size=10, color=self.color_scheme['Combined']),
                hovertemplate='<b>Hybrid Weight</b>: %{r:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title=dict(
                    text=f'Radar by {pname}',
                    x=0.5,
                    font=dict(family=self.font_config['family'], size=18, color=self.font_config['color']),
                    pad=dict(t=20, b=20)
                ),
                polar=dict(
                    radialaxis=dict(range=[0, 1.05], tickfont=dict(size=12, family=self.font_config['family']),
                                   gridwidth=2, linecolor='gray', linewidth=1),
                    angularaxis=dict(
                        showticklabels=False,           # remove degree ticks
                        gridcolor='lightgray',
                        linecolor='gray',
                        gridwidth=2
                    ),
                    bgcolor='rgba(240, 240, 245, 0.5)'
                ),
                legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5,
                           font=dict(family=self.font_config['family'], size=12, color=self.font_config['color']),
                           itemwidth=60, tracegroupgap=15),
                width=600,
                height=500,
                margin=dict(l=60, r=60, t=80, b=60)
            )
            figs.append(fig)

        return figs

    def create_weight_formula_breakdown(self, sources_data, target_params, param_sigmas):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Weight Component Breakdown</b>',
                '<b>Cumulative Weight Distribution</b>',
                '<b>Parameter Weight Analysis</b>',
                '<b>Hybrid Weight Distribution</b>'
            ),
            vertical_spacing=0.2,    # increased spacing
            horizontal_spacing=0.15   # increased spacing
        )
        
        sources_data = sorted(sources_data, key=lambda x: x['source_index'])
        source_indices = [s['source_index'] for s in sources_data]
        
        l0_values = [s['l0_weight'] for s in sources_data]
        fc_values = [s['fc_weight'] for s in sources_data]
        rs_values = [s['rs_weight'] for s in sources_data]
        c_bulk_values = [s['c_bulk_weight'] for s in sources_data]
        attention_values = [s['attention_weight'] for s in sources_data]
        combined_values = [s['combined_weight'] for s in sources_data]
        
        fig.add_trace(go.Bar(
            x=source_indices, y=l0_values, name='L0 Weight',
            marker_color=self.color_scheme['L0'],
            text=[f'{v:.3f}' for v in l0_values], textposition='outside', textfont=dict(size=10, family=self.font_config['family']),
            hovertemplate='<b>Source %{x}</b><br>L0 Weight: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=source_indices, y=fc_values, name='fc Weight',
            marker_color=self.color_scheme['fc'],
            text=[f'{v:.3f}' for v in fc_values], textposition='outside', textfont=dict(size=10, family=self.font_config['family'])
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=source_indices, y=attention_values, name='Attention',
            marker_color=self.color_scheme['Attention'],
            text=[f'{v:.3f}' for v in attention_values], textposition='outside', textfont=dict(size=10, family=self.font_config['family'])
        ), row=1, col=1)
        
        sorted_weights = np.sort(combined_values)[::-1]
        cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights) if np.sum(sorted_weights) > 0 else np.zeros_like(sorted_weights)
        x_vals = np.arange(1, len(cumulative) + 1)
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=cumulative, mode='lines+markers', name='Cumulative Weight',
            line=dict(color=self.color_scheme['Combined'], width=4), marker=dict(size=8),
            fill='tozeroy', fillcolor='rgba(153, 102, 255, 0.2)',
            hovertemplate='Top %{x} sources: %{y:.1%}<extra></extra>'
        ), row=1, col=2)
        
        if len(cumulative) > 0 and np.sum(sorted_weights) > 0:
            threshold_idx = np.where(cumulative >= 0.9)[0]
            if len(threshold_idx) > 0:
                fig.add_hline(y=0.9, line_dash="dash", line_color="red",
                             annotation_text="90% threshold", row=1, col=2)
                fig.add_vline(x=threshold_idx[0]+1, line_dash="dash", line_color="red", row=1, col=2)
        
        param_means = [np.mean(l0_values), np.mean(fc_values), np.mean(rs_values), np.mean(c_bulk_values)]
        param_names = ['L0', 'fc', 'rs', 'c_bulk']
        param_colors = [self.color_scheme['L0'], self.color_scheme['fc'], 
                       self.color_scheme['rs'], self.color_scheme['c_bulk']]
        
        fig.add_trace(go.Bar(
            x=param_names, y=param_means, name='Mean Weight by Parameter',
            marker_color=param_colors,
            text=[f'{v:.3f}' for v in param_means], textposition='auto',
            textfont=dict(family=self.font_config['family']),
            hovertemplate='<b>%{x}</b><br>Mean weight: %{y:.3f}<extra></extra>'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=source_indices, y=combined_values, mode='markers+lines',
            name='Combined Weight',
            line=dict(color=self.color_scheme['Combined'], width=3),
            marker=dict(size=15, color=self.color_scheme['Combined'],
                       symbol='circle', line=dict(width=2, color='white')),
            text=[f"S{i}: Combined={w:.3f}" for i, w in zip(source_indices, combined_values)],
            textfont=dict(family=self.font_config['family']),
            hovertemplate='<b>Source %{x}</b><br>Combined Weight: %{y:.3f}<extra></extra>'
        ), row=2, col=2)
        
        fig.update_layout(
            barmode='stack',
            title=dict(
                text=f'<b>HYBRID WEIGHT FORMULA ANALYSIS</b><br>' + 
                     f'wᵢ = α(L0) × β(params) × γ(shape) × Attention<br>' + 
                     f'Target: L0={target_params.get("L0_nm", 20):.0f}nm, fc={target_params.get("fc", 0.18):.2f}',
                font=dict(family=self.font_config['family'], size=20,
                         color=self.font_config['color']),  # removed weight
                x=0.5, y=0.98,
                pad=dict(t=20, b=20)
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                       font=dict(family=self.font_config['family'], size=12, color=self.font_config['color']),
                       itemwidth=50, tracegroupgap=10),
            width=1400,
            height=1000,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest',
            margin=dict(l=100, r=100, t=150, b=100)
        )
        
        fig.update_xaxes(title_text="Source Index", row=1, col=1, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_yaxes(title_text="Weight Component Value", row=1, col=1, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_xaxes(title_text="Number of Top Sources", row=1, col=2, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_yaxes(title_text="Cumulative Weight", row=1, col=2, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_xaxes(title_text="Parameter", row=2, col=1, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_yaxes(title_text="Mean Weight", row=2, col=1, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_xaxes(title_text="Source Index", row=2, col=2, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        fig.update_yaxes(title_text="Weight Value", row=2, col=2, title_font=dict(family=self.font_config['family'], size=14, color=self.font_config['color']))
        
        return fig

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
    """Three-tier temporal management system with hierarchical source filtering."""
    
    def __init__(self, interpolator, sources: List[Dict], target_params: Dict,
                 n_key_frames: int = 10, lru_size: int = 3,
                 require_categorical_match: bool = False):
        try:
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
        except Exception as e:
            st.error(f"❌ TemporalFieldManager initialization failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            raise
    
    def _compute_thickness_curve(self):
        try:
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
        except Exception as e:
            st.error(f"❌ Thickness curve computation failed: {str(e)}")
            self.thickness_time = {'t_norm': [0, 1], 'th_nm': [0, 0], 't_real_s': [0, 0]}
            self.weights = {'combined': [1.0], 'kernel': [1.0], 'attention': [0.0], 'entropy': 0.0}
            self.avg_tau0 = 1e-4
            self.avg_t_max_nd = 1.0
    
    def _precompute_key_frames(self):
        try:
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
        except Exception as e:
            st.error(f"❌ Key frame precomputation failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            raise
    
    def _estimate_key_frame_memory(self) -> float:
        if not self.key_frames:
            return 0.0
        sample_frame = next(iter(self.key_frames.values()))
        bytes_per_frame = sum(arr.nbytes for arr in sample_frame.values())
        return (bytes_per_frame * len(self.key_frames)) / (1024 * 1024)
    
    def get_fields(self, time_norm: float, use_interpolation: bool = True) -> Dict[str, np.ndarray]:
        try:
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
        except Exception as e:
            st.error(f"❌ get_fields failed at time_norm={time_norm}: {str(e)}")
            return {'phi': np.zeros((256, 256)), 'c': np.zeros((256, 256)), 'psi': np.zeros((256, 256))}
    
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
# ENHANCED CORE‑SHELL INTERPOLATOR WITH HYBRID WEIGHTS
# =============================================
class CoreShellInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                 param_sigma=None, temperature=1.0,
                 gating_mode="Hierarchical: L0 → fc → rs → c_bulk",
                 lambda_shape=0.5, sigma_shape=0.15):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        if param_sigma is None:
            param_sigma = [0.15, 0.15, 0.15, 0.15]
        self.param_sigma = param_sigma
        
        self.temperature = temperature
        self.gating_mode = gating_mode
        self.lambda_shape = lambda_shape
        self.sigma_shape = sigma_shape
        
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
    
    def set_shape_params(self, lambda_shape, sigma_shape):
        self.lambda_shape = lambda_shape
        self.sigma_shape = sigma_shape
    
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
    
    def compute_alpha(self, source_params: List[Dict], target_L0: float,
                     preference_tiers: Dict = None) -> np.ndarray:
        if preference_tiers is None:
            preference_tiers = {
                'preferred': (5.0, 1.0),
                'acceptable': (15.0, 0.75),
                'marginal': (30.0, 0.45),
                'poor': (50.0, 0.15),
                'exclude': (np.inf, 0.01)
            }
        
        alphas = []
        for src in source_params:
            src_L0 = src.get('L0_nm', 20.0)
            delta = abs(target_L0 - src_L0)
            
            if delta <= preference_tiers['preferred'][0]:
                weight = preference_tiers['preferred'][1]
            elif delta <= preference_tiers['acceptable'][0]:
                t = (delta - 5.0) / (15.0 - 5.0)
                weight = preference_tiers['preferred'][1] - t * (
                    preference_tiers['preferred'][1] - preference_tiers['acceptable'][1])
            elif delta <= preference_tiers['marginal'][0]:
                t = (delta - 15.0) / (30.0 - 15.0)
                weight = preference_tiers['acceptable'][1] - t * (
                    preference_tiers['acceptable'][1] - preference_tiers['marginal'][1])
            elif delta <= preference_tiers['poor'][0]:
                t = (delta - 30.0) / (50.0 - 30.0)
                weight = preference_tiers['marginal'][1] - t * (
                    preference_tiers['marginal'][1] - preference_tiers['poor'][1])
            else:
                weight = preference_tiers['exclude'][1]
            
            alphas.append(weight)
        
        return np.array(alphas)
    
    def compute_beta(self, source_params: List[Dict], target_params: Dict) -> Tuple[np.ndarray, Dict]:
        weights = {'fc': 2.0, 'rs': 1.5, 'c_bulk': 3.0}
        betas = []
        individual_weights = {'fc': [], 'rs': [], 'c_bulk': []}
        
        for src in source_params:
            sq_sum = 0.0
            src_indiv_weights = {}
            
            for i, (pname, w) in enumerate(weights.items()):
                norm_src = DepositionParameters.normalize(src.get(pname, 0.5), pname)
                norm_tar = DepositionParameters.normalize(target_params.get(pname, 0.5), pname)
                diff = norm_src - norm_tar
                sigma_idx = ['fc', 'rs', 'c_bulk'].index(pname)
                sigma = self.param_sigma[sigma_idx]
                
                indiv_weight = np.exp(-0.5 * (diff / sigma) ** 2)
                src_indiv_weights[pname] = indiv_weight
                
                sq_sum += w * (diff / sigma) ** 2
            
            beta = np.exp(-0.5 * sq_sum)
            betas.append(beta)
            
            for pname in weights.keys():
                individual_weights[pname].append(src_indiv_weights[pname])
        
        return np.array(betas), individual_weights
    
    def compute_gamma(self, source_fields: List[Dict], source_params: List[Dict],
                     target_params: Dict, time_norm: float, beta_weights: np.ndarray) -> np.ndarray:
        n_sources = len(source_fields)
        if n_sources == 0:
            return np.array([])
        
        profiles = []
        radii_list = []
        
        for i, src in enumerate(source_params):
            L0 = src.get('L0_nm', 20.0)
            field = source_fields[i]['phi']
            r_centers, profile = DepositionPhysics.compute_radial_profile(field, L0, n_bins=100)
            profiles.append(profile)
            radii_list.append(r_centers)
        
        max_radius = max([r[-1] for r in radii_list])
        r_common = np.linspace(0, max_radius, 100)
        
        profiles_interp = []
        for i in range(n_sources):
            prof_interp = np.interp(r_common, radii_list[i], profiles[i], left=0, right=0)
            profiles_interp.append(prof_interp)
        profiles_interp = np.array(profiles_interp)
        
        beta_norm = beta_weights / (np.sum(beta_weights) + 1e-12)
        ref_profile = np.sum(profiles_interp * beta_norm[:, None], axis=0)
        
        mse = np.mean((profiles_interp - ref_profile) ** 2, axis=1)
        gamma = np.exp(-mse / self.sigma_shape)
        
        return gamma
    
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
        """
        FIXED: Ensure phi, c, psi are always initialized before use.
        This fixes the UnboundLocalError.
        """
        # Initialize with default zeros to prevent UnboundLocalError
        phi = np.zeros(target_shape)
        c = np.zeros(target_shape)
        psi = np.zeros(target_shape)
        
        history = source.get('history', [])
        if not history:
            return {'phi': phi, 'c': c, 'psi': psi}
        
        t_max = 1.0
        if source.get('thickness_history'):
            t_max = source['thickness_history'][-1]['t_nd']
        else:
            t_max = history[-1]['t_nd']
        
        t_target = time_norm * t_max
        
        try:
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
        except Exception as e:
            st.warning(f"Warning in _get_fields_at_time: {str(e)}")
            phi = np.zeros(target_shape)
            c = np.zeros(target_shape)
            psi = np.zeros(target_shape)
        
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
        
        if not require_categorical_match and len(active_sources) > 0:
            l0_values = [src['params'].get('L0_nm', 20.0) for src in active_sources]
            target_L0 = target_params.get('L0_nm', 20.0)
            deltas = [abs(L0 - target_L0) for L0 in l0_values]
            st.info(f"📊 L0 Distribution: target={target_L0}nm, "
                   f"sources: min={min(l0_values):.1f}nm, max={max(l0_values):.1f}nm, "
                   f"closest ΔL0={min(deltas):.1f}nm")
        
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
        
        target_L0 = target_params.get('L0_nm', 20.0)
        
        alpha = self.compute_alpha(source_params, target_L0)
        beta, individual_param_weights = self.compute_beta(source_params, target_params)
        
        beta_norm = beta / (np.sum(beta) + 1e-12)
        gamma = self.compute_gamma(source_fields, source_params, target_params, 
                                  t_req if t_req is not None else 1.0, beta_norm)
        
        refinement_factor = alpha * beta * (1.0 + self.lambda_shape * gamma)
        
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
        
        final_scores = attn_scores * torch.FloatTensor(refinement_factor).unsqueeze(0)
        final_weights = torch.softmax(final_scores, dim=-1).squeeze().detach().cpu().numpy()
        
        if np.isscalar(final_weights):
            final_weights = np.array([final_weights])
        elif final_weights.ndim == 0:
            final_weights = np.array([final_weights.item()])
        elif final_weights.ndim > 1:
            final_weights = final_weights.flatten()
        
        if len(final_weights) != len(source_fields):
            st.warning(f"Weight length mismatch: {len(final_weights)} vs {len(source_fields)}.")
            if len(final_weights) > len(source_fields):
                final_weights = final_weights[:len(source_fields)]
            else:
                final_weights = np.pad(final_weights, (0, len(source_fields)-len(final_weights)),
                                      'constant', constant_values=0)
        
        eps = 1e-10
        entropy = -np.sum(final_weights * np.log(final_weights + eps))
        max_weight = np.max(final_weights)
        effective_sources = np.sum(final_weights > 0.01)
        
        if max_weight < 0.2:
            st.warning(f"⚠️ Moderate confidence: max weight={max_weight:.3f}, "
                      f"effective sources={effective_sources}.")
        
        interp = {'phi': np.zeros(target_shape),
                 'c': np.zeros(target_shape),
                 'psi': np.zeros(target_shape)}
        
        for i, fld in enumerate(source_fields):
            interp['phi'] += final_weights[i] * fld['phi']
            interp['c'] += final_weights[i] * fld['c']
            interp['psi'] += final_weights[i] * fld['psi']
        
        interp['phi'] = gaussian_filter(interp['phi'], sigma=1.0)
        interp['c'] = gaussian_filter(interp['c'], sigma=1.0)
        interp['psi'] = gaussian_filter(interp['psi'], sigma=1.0)
        
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
        alpha_phys = target_params.get('alpha_nd', 2.0)
        potential = DepositionPhysics.potential_proxy(interp['c'], alpha_phys)
        
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
        
        # Prepare sources data for visualization
        sources_data = []
        for i, (src_params, alpha_w, beta_w, gamma_w, indiv_weights, combined_w, attn_w) in enumerate(zip(
            source_params, alpha, beta, gamma,
            [dict(fc=individual_param_weights['fc'][i],
                 rs=individual_param_weights['rs'][i],
                 c_bulk=individual_param_weights['c_bulk'][i]) for i in range(len(source_params))],
            final_weights, attn_scores.squeeze().detach().cpu().numpy()
        )):
            sources_data.append({
                'source_index': i,
                'L0_nm': src_params.get('L0_nm', 20.0),
                'fc': src_params.get('fc', 0.18),
                'rs': src_params.get('rs', 0.2),
                'c_bulk': src_params.get('c_bulk', 0.5),
                'l0_weight': float(alpha_w),
                'fc_weight': float(indiv_weights['fc']),
                'rs_weight': float(indiv_weights['rs']),
                'c_bulk_weight': float(indiv_weights['c_bulk']),
                'beta_weight': float(beta_w),
                'gamma_weight': float(gamma_w),
                'attention_weight': float(attn_w),
                'physics_refinement': float(alpha_w * beta_w * (1.0 + self.lambda_shape * gamma_w)),
                'combined_weight': float(combined_w)
            })
        
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
                'alpha': alpha.tolist(),
                'beta': beta.tolist(),
                'gamma': gamma.tolist(),
                'individual_params': individual_param_weights,
                'refinement_factor': refinement_factor.tolist(),
                'attention': attn_scores.squeeze().detach().cpu().numpy().tolist(),
                'entropy': float(entropy),
                'max_weight': float(max_weight),
                'effective_sources': int(effective_sources)
            },
            'sources_data': sources_data,
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
# ENHANCED HEATMAP VISUALIZER
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
                'interpolation_method': 'core_shell_hybrid_weight_transformer',
                'visualization_params': visualization_params
            },
            'result': {
                'target_params': res['target_params'],
                'shape': res['shape'],
                'num_sources': res['num_sources'],
                'weights': res['weights'],
                'sources_data': res.get('sources_data', []),
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
            fc = p.get('fc', 0); rs = p.get('rs', 0)
            cb = p.get('c_bulk', 0)
            t = export_data['result'].get('time_real_s', 0)
            filename = f"temporal_interp_fc{fc:.3f}_rs{rs:.3f}_c{cb:.2f}_t{t:.3e}s_{ts}.json"
        
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
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
    st.set_page_config(page_title="Core‑Shell Deposition: Hybrid Weight Interpolation",
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
    .weight-formula { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 5px solid #3B82F6; padding: 1.5rem; border-radius: 10px;
        margin: 1.5rem 0; font-family: 'Courier New', monospace; font-size: 1.1rem;
        color: white; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧪 Core‑Shell Deposition: Hybrid Weight Interpolation</h1>',
               unsafe_allow_html=True)
    
    st.markdown("""
    <div class="weight-formula">
    <strong>HYBRID WEIGHT FORMULA:</strong><br><br>
    wᵢ = α(L0)ᵢ × β(params)ᵢ × γ(shape)ᵢ × Attentionᵢ<br><br>
    <strong>Components:</strong><br>
    • α(L0): L₀ proximity weight (tiered preference: 5/15/30/50nm)<br>
    • β(params): Parameter closeness (fc, rs, c_bulk) with individual weights<br>
    • γ(shape): Radial profile similarity boost<br>
    • Attention: Learned transformer attention scores<br><br>
    <strong>Result:</strong> Sources weighted proportionally by hybrid weight magnitude
    </div>
    """, unsafe_allow_html=True)
    
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = CoreShellInterpolator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = HeatMapVisualizer()
    if 'weight_visualizer' not in st.session_state:
        st.session_state.weight_visualizer = HybridWeightVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    if 'temporal_manager' not in st.session_state:
        st.session_state.temporal_manager = None
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 1.0
    if 'last_target_hash' not in st.session_state:
        st.session_state.last_target_hash = None
    
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Add beta explanation expander
        with st.expander("ℹ️ What is β (beta)?"):
            st.markdown("""
            **β (parameter closeness)**  
            Measures similarity in `fc`, `rs`, `c_bulk` using a weighted Gaussian kernel.  
            - Weighted combination: fc (2.0), rs (1.5), c_bulk (3.0) – c_bulk is most important.  
            - Kernel width controlled by sliders σ_fc, σ_rs, σ_c.  
            - Higher β → better parameter match.  
            """)
        
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
        use_edl = st.checkbox("Use EDL catalyst", value=True)
        mode = st.selectbox("Mode", ["2D (planar)", "3D (spherical)"], index=0)
        
        st.divider()
        
        st.markdown('<h2 class="section-header">⚛️ Interpolation Settings</h2>', unsafe_allow_html=True)
        sigma_fc = st.slider("Kernel σ (fc)", 0.05, 0.3, 0.15, 0.01)
        sigma_rs = st.slider("Kernel σ (rs)", 0.05, 0.3, 0.15, 0.01)
        sigma_c = st.slider("Kernel σ (c_bulk)", 0.05, 0.3, 0.15, 0.01)
        sigma_L = st.slider("Kernel σ (L0_nm)", 0.05, 0.3, 0.15, 0.01)
        temperature = st.slider("Attention temperature", 0.1, 10.0, 1.0, 0.1)
        
        st.markdown("#### 🌀 Shape-Aware Refinement")
        lambda_shape = st.slider("λ (shape boost weight)", 0.0, 1.0, 0.5, 0.05)
        sigma_shape = st.slider("σ_shape (radial similarity)", 0.05, 0.5, 0.15, 0.01)
        
        n_key_frames = st.slider("Key frames", 1, 20, 5, 1)
        lru_cache_size = st.slider("LRU cache size", 1, 5, 3, 1)
        
        target = {
            'fc': fc, 'rs': rs, 'c_bulk': c_bulk, 'L0_nm': L0_nm,
            'bc_type': bc_type, 'use_edl': use_edl, 'mode': mode
        }
        target_hash = hashlib.md5(json.dumps(target, sort_keys=True).encode()).hexdigest()[:16]
        
        if target_hash != st.session_state.last_target_hash:
            if st.button("🧠 Initialize Temporal Interpolation", type="primary",
                        use_container_width=True):
                if not st.session_state.solutions:
                    st.error("Please load solutions first!")
                else:
                    with st.spinner("Setting up temporal interpolation..."):
                        st.session_state.interpolator.set_parameter_sigma(
                            [sigma_fc, sigma_rs, sigma_c, sigma_L])
                        st.session_state.interpolator.temperature = temperature
                        st.session_state.interpolator.set_shape_params(lambda_shape, sigma_shape)
                        
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
        
        if st.session_state.temporal_manager:
            with st.expander("💾 Memory Statistics"):
                stats = st.session_state.temporal_manager.get_memory_stats()
                st.markdown(f"""
                <div class="memory-stats">
                <strong>Memory Usage:</strong><br>
                • Key frames: {stats['key_frame_entries']} ({stats['key_frames_mb']:.1f} MB)<br>
                • LRU cache: {stats['lru_entries']} ({stats['lru_cache_mb']:.1f} MB)<br>
                • <strong>Total: {stats['total_mb']:.1f} MB</strong>
                </div>
                """, unsafe_allow_html=True)
    
    if st.session_state.temporal_manager:
        mgr = st.session_state.temporal_manager
        
        st.markdown('<h2 class="section-header">⏱️ Temporal Control</h2>', unsafe_allow_html=True)
        col_time1, col_time2, col_time3 = st.columns([3, 1, 1])
        with col_time1:
            current_time_norm = st.slider("Normalized Time", 0.0, 1.0,
                                         value=st.session_state.current_time,
                                         step=0.001, format="%.3f")
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
        with col_info3:
            st.metric("Time", f"{current_time_real:.3e} s")
        
        tabs = st.tabs(["📊 Field Visualization", "⚖️ Hybrid Weight Analysis",
                       "📈 Thickness Evolution", "💾 Export"])
        
        with tabs[0]:
            st.markdown('<h2 class="section-header">📊 Field Visualization</h2>',
                       unsafe_allow_html=True)
            fields = mgr.get_fields(current_time_norm, use_interpolation=True)
            
            field_choice = st.selectbox("Select field",
                                       ['c (concentration)', 'phi (shell)', 'psi (core)',
                                        'material proxy'],
                                       key='field_choice')
            field_map = {'c (concentration)': 'c', 'phi (shell)': 'phi',
                        'psi (core)': 'psi', 'material proxy': 'material'}
            field_key = field_map[field_choice]
            
            if field_key == 'material':
                field_data = DepositionPhysics.material_proxy(fields['phi'], fields['psi'])
            else:
                field_data = fields[field_key]
            
            cmap = st.selectbox("Colormap", COLORMAP_OPTIONS['Sequential'], index=0)
            
            fig = st.session_state.visualizer.create_field_heatmap(
                field_data,
                title=f"Interpolated {field_choice}",
                cmap_name=cmap,
                L0_nm=L0_nm,
                target_params=target,
                time_real_s=current_time_real
            )
            st.pyplot(fig)
        
        with tabs[1]:
            st.markdown('<h2 class="section-header">⚖️ Hybrid Weight Analysis</h2>',
                       unsafe_allow_html=True)
            
            weights = mgr.weights
            
            st.markdown("""
            <div class="info-box">
            <strong>Weight Quantification:</strong> Each source receives multiple weight components
            that are combined into a hybrid weight. The source with highest net weight contributes
            most to the interpolation, others contribute proportionally.
            </div>
            """, unsafe_allow_html=True)
            
            weight_tabs = st.tabs(["📊 Sankey Diagram", "🔗 Chord Diagram",
                                  "🕸️ Parameter Radar Charts", "📈 Breakdown"])
            
            interp_res = st.session_state.interpolator.interpolate_fields(
                st.session_state.solutions, target, target_shape=(256,256),
                n_time_points=100, time_norm=current_time_norm
            )
            sources_data = interp_res.get('sources_data', []) if interp_res else []
            
            with weight_tabs[0]:
                st.markdown("#### 📊 Enhanced Sankey Diagram")
                if sources_data:
                    fig_sankey = st.session_state.weight_visualizer.create_enhanced_sankey_diagram(
                        sources_data, target, [sigma_fc, sigma_rs, sigma_c, sigma_L]
                    )
                    st.plotly_chart(fig_sankey, use_container_width=True)
            
            with weight_tabs[1]:
                st.markdown("#### 🔗 Enhanced Chord Diagram")
                if sources_data:
                    fig_chord = st.session_state.weight_visualizer.create_enhanced_chord_diagram(
                        sources_data, target
                    )
                    st.plotly_chart(fig_chord, use_container_width=True)
            
            with weight_tabs[2]:
                st.markdown("#### 🕸️ Parameter‑Based Radar Charts")
                if sources_data:
                    radar_figs = st.session_state.weight_visualizer.create_parameter_radar_charts(
                        sources_data, target, [sigma_fc, sigma_rs, sigma_c, sigma_L]
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(radar_figs[0], use_container_width=True)  # L0
                        st.plotly_chart(radar_figs[2], use_container_width=True)  # rs
                    with col2:
                        st.plotly_chart(radar_figs[1], use_container_width=True)  # fc
                        st.plotly_chart(radar_figs[3], use_container_width=True)  # c_bulk
                else:
                    st.info("No source data available.")
            
            with weight_tabs[3]:
                st.markdown("#### 📈 Weight Formula Breakdown")
                if sources_data:
                    fig_breakdown = st.session_state.weight_visualizer.create_weight_formula_breakdown(
                        sources_data, target, [sigma_fc, sigma_rs, sigma_c, sigma_L]
                    )
                    st.plotly_chart(fig_breakdown, use_container_width=True)
            
            st.markdown("#### 📋 Weight Diagnostics")
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                st.metric("Weight Entropy", f"{weights.get('entropy', 0):.4f}",
                         help="Higher = more uncertain")
            with col_w2:
                st.metric("Max Weight", f"{weights.get('max_weight', 0):.4f}",
                         help="Lower = less confident")
            with col_w3:
                st.metric("Effective Sources", weights.get('effective_sources', 0),
                         help="Sources with weight > 0.01")
            
            if sources_data:
                st.markdown("#### 📋 Source Weight Table (Expanded)")
                df_weights = pd.DataFrame(sources_data)
                columns_to_show = [
                    'source_index', 'L0_nm', 'fc', 'rs', 'c_bulk',
                    'l0_weight', 'fc_weight', 'rs_weight', 'c_bulk_weight',
                    'beta_weight', 'gamma_weight', 'physics_refinement',
                    'attention_weight', 'combined_weight'
                ]
                df_display = df_weights[columns_to_show].copy()
                df_display.rename(columns={
                    'l0_weight': 'α (L0)',
                    'fc_weight': 'β_fc',
                    'rs_weight': 'β_rs',
                    'c_bulk_weight': 'β_c',
                    'beta_weight': 'β_total',
                    'gamma_weight': 'γ (shape)',
                    'physics_refinement': 'α·β·(1+λγ)',
                    'attention_weight': 'Attention',
                    'combined_weight': 'Hybrid Weight'
                }, inplace=True)
                
                styled = df_display.style.format({
                    'L0_nm': '{:.0f}',
                    'fc': '{:.3f}',
                    'rs': '{:.3f}',
                    'c_bulk': '{:.2f}',
                    'α (L0)': '{:.4f}',
                    'β_fc': '{:.4f}',
                    'β_rs': '{:.4f}',
                    'β_c': '{:.4f}',
                    'β_total': '{:.4f}',
                    'γ (shape)': '{:.4f}',
                    'α·β·(1+λγ)': '{:.4f}',
                    'Attention': '{:.4f}',
                    'Hybrid Weight': '{:.4f}'
                }).background_gradient(subset=['Hybrid Weight'], cmap='viridis')
                
                st.dataframe(styled, use_container_width=True)
        
        with tabs[2]:
            st.markdown('<h2 class="section-header">📈 Thickness Evolution</h2>',
                       unsafe_allow_html=True)
            thickness_time = mgr.thickness_time
            
            fig_th = plt.figure(figsize=(10, 6), dpi=300)
            t_arr = np.array(thickness_time['t_real_s']) if 't_real_s' in thickness_time else np.array(thickness_time['t_norm'])
            th_arr = np.array(thickness_time['th_nm'])
            plt.plot(t_arr, th_arr, 'b-', linewidth=3, label='Interpolated')
            plt.xlabel("Time (s)" if 't_real_s' in thickness_time else "Normalized Time")
            plt.ylabel("Thickness (nm)")
            plt.title(f"Shell Thickness Evolution")
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig_th)
        
        with tabs[3]:
            st.markdown('<h2 class="section-header">💾 Export Data</h2>',
                       unsafe_allow_html=True)
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📊 Export Current State (JSON)", use_container_width=True):
                    res = st.session_state.interpolator.interpolate_fields(
                        st.session_state.solutions, target, target_shape=(256,256),
                        n_time_points=100, time_norm=current_time_norm
                    )
                    if res:
                        export_data = st.session_state.results_manager.prepare_export_data(
                            res, {'time_norm': current_time_norm, 'time_real_s': current_time_real}
                        )
                        json_str, fname = st.session_state.results_manager.export_to_json(export_data)
                        st.download_button("⬇️ Download JSON", json_str, fname, "application/json")
    
    else:
        st.info("""
        👈 **Get Started:**
        1. Load solutions using the sidebar
        2. Set target parameters
        3. Click **"Initialize Temporal Interpolation"**
        
        **Hybrid Weight Features:**
        • Individual parameter weights (L0, fc, rs, c_bulk)
        • Physics refinement factors (α, β, γ)
        • Transformer attention scores
        • Combined hybrid weights for proportional source weighting
        • Comprehensive weight visualization (Sankey, Chord, Radar, Breakdown)
        """)


if __name__ == "__main__":
    main()
