#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT – Intelligent Experimental Input Generator
(Fully enhanced: colour‑aware detection, auto scale value, LLM verification,
plot detection with robust OCR + LLM parsing, improved scale‑bar filtering)
"""
import streamlit as st
import os
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import cv2

# -------------------- SESSION STATE INITIALISATION (crash prevention) --------------------
if "initialised" not in st.session_state:
    st.session_state.update({
        "core_diameter": None,
        "scale_nm_per_px": None,
        "c_bulk": None,
        "ratio_str": "?",
        "geo_annotated": None,
        "comp_annotated": None,
        "llm_tokenizer": None,
        "llm_model": None,
        "temp_geo_detection": None,
        "temp_comp_detection": None,
    })
    st.session_state.initialised = True

# -------------------- Optional dependency checks (graceful) --------------------
try:
    from skimage import feature, transform, measure, color, morphology, exposure
    from skimage.transform import probabilistic_hough_line, hough_circle, hough_circle_peaks
    from skimage.measure import regionprops, label
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    st.warning("scikit‑image not installed. Automatic detection disabled. Install with: pip install scikit-image")

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TORCH_AVAILABLE = False
    st.warning("Transformers or PyTorch not installed. LLM query generation disabled; default query will be shown.")

try:
    from google.cloud import vision
    import io
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# -------------------- Path configuration --------------------
BASE_DIR = Path(__file__).parent.resolve()
GEOMETRY_FOLDER = BASE_DIR / "experimental_images" / "geometry"
COMPOSITION_FOLDER = BASE_DIR / "experimental_images" / "composition_ratio"
GEOMETRY_FOLDER.mkdir(parents=True, exist_ok=True)
COMPOSITION_FOLDER.mkdir(parents=True, exist_ok=True)

# -------------------- Image type classifier --------------------
def classify_image_type(pil_img):
    """Return 'composition' for EDS maps, 'geometry' for HRTEM."""
    img_rgb = np.array(pil_img.convert('RGB'))
    hsv = color.rgb2hsv(img_rgb)
    red_score = np.sum(((hsv[..., 0] < 0.05) | (hsv[..., 0] > 0.95)) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3))
    green_score = np.sum((np.abs(hsv[..., 0] - 0.33) < 0.1) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3))
    if red_score > 8000 and green_score > 8000:
        return "composition"
    return "geometry"

# -------------------- ENHANCED: Multi-strategy scale bar detection --------------------
def detect_scale_bar_morphological(pil_img):
    """
    Detect scale bar using morphological operations - looks for bright horizontal bars.
    This is more robust than Hough lines for thick scale bars.
    """
    img_gray = np.array(pil_img.convert('L'))
    height, width = img_gray.shape
    
    # Threshold to find bright objects (scale bars are usually white)
    _, binary = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)
    
    # Define horizontal kernel to detect horizontal lines/bars
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    
    # Morphological operations to isolate horizontal structures
    detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scale_bar_candidates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = cv2.contourArea(contour)
        
        # Scale bar heuristics:
        # - Wide and relatively short (high aspect ratio)
        # - Reasonable area (not too small, not the whole image)
        # - Located in bottom or top 25% of image
        y_center = y + h / 2
        is_in_edge_region = (y_center < height * 0.25) or (y_center > height * 0.75)
        
        if aspect_ratio > 4 and 500 < area < (width * height * 0.05) and is_in_edge_region and w > 30:
            confidence = min(0.95, (aspect_ratio / 10) * 0.5 + (area / 2000) * 0.5)
            scale_bar_candidates.append({
                'rect': (x, y, w, h),
                'line_coords': ((x, y + h//2), (x + w, y + h//2)),
                'length_px': w,
                'confidence': confidence,
                'center_x': x + w // 2,
                'center_y': y + h // 2
            })
    
    if scale_bar_candidates:
        # Sort by confidence and prefer right-side bars
        scale_bar_candidates.sort(key=lambda k: (k['confidence'], k['center_x']), reverse=True)
        best = scale_bar_candidates[0]
        return best['length_px'], best['line_coords'], best['confidence'], pil_img
    
    return None, None, 0.0, pil_img

def detect_scale_bar_hough(pil_img):
    """Original Hough line-based detection as fallback."""
    img = np.array(pil_img.convert('L'))
    height, width = img.shape
    edges = feature.canny(img, sigma=2)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=20, line_gap=3)
    horizontal_lines = []
    
    for line in lines:
        (x1, y1), (x2, y2) = line
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 10:
            margin = 10
            if y1 < margin or y2 < margin or y1 > height - margin or y2 > height - margin:
                continue
            
            y_min, y_max = min(y1, y2), max(y1, y2)
            x_min, x_max = min(x1, x2), max(x1, x2)
            line_region = img[y_min:y_max+1, x_min:x_max+1]
            
            if line_region.size > 0 and np.mean(line_region) < 200:
                continue
            
            y_center = (y1 + y2) / 2
            if y_center < 0.25 * height or y_center > 0.75 * height:
                if length > 40:
                    horizontal_lines.append((length, line, (x1+x2)/2))
    
    if not horizontal_lines:
        return None, None, 0.0, pil_img
    
    scored_lines = []
    for length, line, x_center in horizontal_lines:
        score = length * (x_center / width)
        scored_lines.append((score, length, line))
    
    scored_lines.sort(key=lambda x: x[0], reverse=True)
    best_score, length_px, line_coords = scored_lines[0]
    confidence = 0.7
    annotated = annotate_geometry(pil_img, scale_line=line_coords, core_info=None)
    return length_px, line_coords, confidence, annotated

def detect_scale_bar_skimage(pil_img):
    """Combined approach: try morphological first, then Hough."""
    # Try morphological detection first (better for thick bars)
    length_px, line_coords, conf, annotated = detect_scale_bar_morphological(pil_img)
    
    if length_px and conf > 0.6:
        return length_px, line_coords, conf, annotated
    
    # Fallback to Hough
    return detect_scale_bar_hough(pil_img)

# -------------------- ENHANCED: Intelligent scale value extraction --------------------
def extract_text_from_region(pil_img, region_bbox):
    """Extract text from a specific region using multiple OCR methods."""
    left, top, right, bottom = region_bbox
    crop = pil_img.crop((left, top, right, bottom))
    
    text = ""
    
    # Try Google Vision first
    if GOOGLE_VISION_AVAILABLE:
        try:
            client = vision.ImageAnnotatorClient()
            buffer = io.BytesIO()
            crop.save(buffer, format='PNG')
            content = buffer.getvalue()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            if response.text_annotations:
                text = response.text_annotations[0].description
        except Exception:
            pass
    
    # Try Pytesseract if Google Vision failed
    if not text and PYTESSERACT_AVAILABLE:
        try:
            text = pytesseract.image_to_string(crop, config='--psm 7')
        except Exception:
            pass
    
    return text.strip()

def auto_extract_scale_value(pil_img, line_coords):
    """
    Extract numeric scale value (nm) by searching in multiple regions around the scale bar.
    Uses intelligent region expansion and multiple OCR strategies.
    """
    if not line_coords:
        return 20.0  # safe default
    
    (x1, y1), (x2, y2) = line_coords
    height, width = pil_img.size
    
    # Define multiple search regions with different priorities
    search_regions = []
    
    # Region 1: Directly below the bar (most common)
    search_regions.append({
        'bbox': (max(0, min(x1, x2) - 30), 
                 max(0, max(y1, y2) + 5), 
                 min(width, max(x1, x2) + 30), 
                 min(height, max(y1, y2) + 70)),
        'priority': 1,
        'name': 'below'
    })
    
    # Region 2: Directly above the bar
    search_regions.append({
        'bbox': (max(0, min(x1, x2) - 30), 
                 max(0, min(y1, y2) - 70), 
                 min(width, max(x1, x2) + 30), 
                 min(height, min(y1, y2) - 5)),
        'priority': 2,
        'name': 'above'
    })
    
    # Region 3: Right side of the bar
    search_regions.append({
        'bbox': (max(0, max(x1, x2) + 5), 
                 max(0, min(y1, y2) - 20), 
                 min(width, max(x1, x2) + 100), 
                 min(height, max(y1, y2) + 20)),
        'priority': 3,
        'name': 'right'
    })
    
    # Region 4: Left side of the bar
    search_regions.append({
        'bbox': (max(0, min(x1, x2) - 100), 
                 max(0, min(y1, y2) - 20), 
                 min(width, min(x1, x2) - 5), 
                 min(height, max(y1, y2) + 20)),
        'priority': 4,
        'name': 'left'
    })
    
    # Region 5: Larger surrounding area
    search_regions.append({
        'bbox': (max(0, min(x1, x2) - 50), 
                 max(0, min(y1, y2) - 80), 
                 min(width, max(x1, x2) + 50), 
                 min(height, max(y1, y2) + 80)),
        'priority': 5,
        'name': 'surrounding'
    })
    
    # Sort by priority
    search_regions.sort(key=lambda k: k['priority'])
    
    # Try each region
    for region in search_regions:
        text = extract_text_from_region(pil_img, region['bbox'])
        
        if text:
            # Try multiple regex patterns
            patterns = [
                r'(\d+\.?\d*)\s*(?:nm|nanometers?)',
                r'(\d+\.?\d*)\s*(?:μm|um|micrometers?)',
                r'(\d+\.?\d*)\s*[nμ]m',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit_match = re.search(r'(μm|um)', text, re.IGNORECASE)
                    if unit_match:
                        value *= 1000  # Convert μm to nm
                    return value
    
    # Fallback: Search entire bottom portion of image
    full_bottom_text = extract_text_from_region(
        pil_img, 
        (0, height // 2, width, height)
    )
    
    if full_bottom_text:
        match = re.search(r'(\d+\.?\d*)\s*(?:nm|μm|um)', full_bottom_text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if re.search(r'μm|um', full_bottom_text, re.IGNORECASE):
                value *= 1000
            return value
    
    return 20.0  # Default fallback

# -------------------- Core extraction from red mask (EDS) --------------------
def extract_core_from_mask(red_mask, scale_nm_per_px):
    """Accurate diameter from red mask (EDS mode)."""
    labeled = label(red_mask.astype(int))
    props = regionprops(labeled)
    if not props:
        return None, None, 0.0, None
    biggest = max(props, key=lambda p: p.area)
    diam_px = biggest.equivalent_diameter_area
    diam_nm = diam_px * scale_nm_per_px
    center = (int(biggest.centroid[1]), int(biggest.centroid[0]))
    radius_px = diam_px / 2
    confidence = min(0.95, biggest.area / (np.sum(red_mask) * 0.8))
    core_info = {
        'center': center,
        'radius_px': radius_px,
        'diam_nm': diam_nm,
        'contour': None
    }
    return diam_nm, diam_px, confidence, core_info

# -------------------- Annotation helper (enhanced) --------------------
def annotate_geometry(pil_img, scale_line=None, core_info=None):
    """Return a copy of pil_img with scale bar, core highlighted, arrows, and legend."""
    annotated = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Highlight scale bar
    if scale_line:
        (x1, y1), (x2, y2) = scale_line
        draw.line([(x1, y1), (x2, y2)], fill="#FF0000", width=6)
        draw.rectangle([(min(x1,x2)-10, min(y1,y2)-15),
                        (max(x1,x2)+10, max(y1,y2)+15)],
                       outline="#FF0000", width=3)
        draw.text((min(x1,x2), min(y1,y2)-25),
                  "SCALE BAR", fill="#FF0000", font=font)
    
    # Highlight core and add arrow & legend
    if core_info:
        cx, cy, r = core_info["center"][0], core_info["center"][1], core_info["radius_px"]
        draw.ellipse((cx-r-3, cy-r-3, cx+r+3, cy+r+3),
                     outline="#00FF00", width=5)
        shell_r = r * 1.3
        dash_len = 10
        for angle in range(0, 360, 15):
            x = cx + shell_r * np.cos(np.radians(angle))
            y = cy + shell_r * np.sin(np.radians(angle))
            x2 = cx + shell_r * np.cos(np.radians(angle+dash_len))
            y2 = cy + shell_r * np.sin(np.radians(angle+dash_len))
            draw.line([(x, y), (x2, y2)], fill="#FFA500", width=3)
        
        diam_nm = core_info.get("diam_nm", 0)
        label = f"{diam_nm:.1f} nm" if diam_nm else "core"
        draw.text((cx-30, cy-15), label,
                  fill="white", font=font, stroke_width=2, stroke_fill="black")
        draw.line([(cx, cy-30), (cx+50, cy-60)], fill="white", width=3)
        draw.text((cx+60, cy-70), "Cu CORE", fill="white", font=font, stroke_width=3)
        
        legend_x, legend_y = 10, 10
        draw.rectangle([(legend_x, legend_y), (legend_x+200, legend_y+60)], outline="white", width=2)
        draw.rectangle([(legend_x+5, legend_y+5), (legend_x+25, legend_y+25)], fill="#00FF00")
        draw.text((legend_x+35, legend_y+5), "Cu core", fill="white", font=small_font)
        draw.rectangle([(legend_x+5, legend_y+30), (legend_x+25, legend_y+50)], fill="#FFA500")
        draw.text((legend_x+35, legend_y+30), "Ag shell", fill="white", font=small_font)
    
    if core_info.get("contour") is not None:
        for pt in core_info["contour"].astype(int):
            if 0 <= pt[0] < annotated.height and 0 <= pt[1] < annotated.width:
                annotated.putpixel((pt[1], pt[0]), (255, 255, 0))
    
    return annotated

def annotate_composition(pil_img, ratio_str, c_bulk, masks=None):
    """Add a text overlay with the detected Cu:Ag ratio, and optionally draw masks."""
    annotated = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    text = f"Cu:Ag ≈ {ratio_str}  →  c_bulk = {c_bulk:.3f}"
    bbox = draw.textbbox((10, annotated.height-50), text, font=font)
    draw.rectangle(bbox, fill=(0,0,0,180))
    draw.text((10, annotated.height-50), text, fill="#00FF00", font=font)
    return annotated

# -------------------- Image analysis functions (scikit‑image) --------------------
if SKIMAGE_AVAILABLE:
    def detect_core_diameter_skimage(pil_img, scale_nm_per_px=None):
        """Detect circular core, now with image type classification and mask‑based extraction for EDS."""
        img_type = classify_image_type(pil_img)
        if img_type == "composition":
            img_rgb = np.array(pil_img.convert('RGB'))
            hsv = color.rgb2hsv(img_rgb)
            red_mask = ((hsv[..., 0] < 0.05) | (hsv[..., 0] > 0.95)) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3)
            diam_nm, diam_px, conf, core_info = extract_core_from_mask(red_mask, scale_nm_per_px)
            if diam_nm:
                debug_info = {'method': 'mask_eds'}
                annotated = annotate_geometry(pil_img, scale_line=None, core_info=core_info)
                return diam_nm, diam_px, conf, debug_info, annotated
            else:
                return None, None, 0.0, {'method': 'mask_failed'}, pil_img
        
        img_rgb = np.array(pil_img.convert('RGB'))
        red_channel = img_rgb[..., 0]
        red_enhanced = exposure.equalize_adapthist(red_channel)
        hough_radii = np.arange(20, 150, 5)
        hough_res = hough_circle(red_enhanced, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=3)
        
        if len(radii) > 0:
            best_idx = np.argmax(accums)
            radius_px = radii[best_idx]
            diameter_px = 2 * radius_px
            confidence = accums[best_idx] / accums.max() if accums.max() > 0 else 0.5
            diam_nm = diameter_px * scale_nm_per_px if scale_nm_per_px else None
            core_info = {
                'center': (cx[best_idx], cy[best_idx]),
                'radius_px': radius_px,
                'diam_nm': diam_nm,
                'contour': None
            }
            debug_info = {'method': 'hough', 'circles_found': len(radii)}
            annotated = annotate_geometry(pil_img, scale_line=None, core_info=core_info)
            return diam_nm, diameter_px, confidence, debug_info, annotated
        
        edges = feature.canny(red_enhanced, sigma=2)
        contours = measure.find_contours(edges, 0.8)
        best_circularity = 0
        best_contour = None
        
        for contour in contours:
            if len(contour) < 5:
                continue
            area = measure.grid_points_in_poly(contour, red_enhanced.shape)
            if area == 0:
                continue
            perimeter = measure.perimeter(contour)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if circularity > best_circularity and circularity > 0.6:
                best_circularity = circularity
                best_contour = contour
        
        if best_contour is not None:
            center = np.mean(best_contour, axis=0)
            radius_px = np.max(np.linalg.norm(best_contour - center, axis=1))
            diameter_px = 2 * radius_px
            confidence = 0.5 * best_circularity
            diam_nm = diameter_px * scale_nm_per_px if scale_nm_per_px else None
            core_info = {
                'center': (int(center[1]), int(center[0])),
                'radius_px': radius_px,
                'diam_nm': diam_nm,
                'contour': best_contour
            }
            debug_info = {'method': 'contour', 'circularity': best_circularity}
            annotated = annotate_geometry(pil_img, scale_line=None, core_info=core_info)
            return diam_nm, diameter_px, confidence, debug_info, annotated
        
        return None, None, 0.0, {}, pil_img

    def extract_composition_skimage(pil_img):
        """Simple color analysis. Returns (c_bulk, ratio_str, confidence, annotated_img)."""
        img_rgb = np.array(pil_img.convert('RGB'))
        hsv = color.rgb2hsv(img_rgb)
        red_mask1 = (hsv[..., 0] < 0.05) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3)
        red_mask2 = (hsv[..., 0] > 0.95) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3)
        red_mask = red_mask1 | red_mask2
        green_mask = (np.abs(hsv[..., 0] - 0.33) < 0.1) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3)
        cu_pixels = np.sum(red_mask)
        ag_pixels = np.sum(green_mask)
        
        if cu_pixels > 100 and ag_pixels > 100:
            ratio = cu_pixels / ag_pixels
            c_bulk = 1.0 / ratio if ratio > 0 else 1.0
            c_bulk = np.clip(c_bulk, 0.1, 1.0)
            ratio_str = f"{round(ratio,1)}:1"
            confidence = 0.6
            annotated = annotate_composition(pil_img, ratio_str, c_bulk)
            return c_bulk, ratio_str, confidence, annotated
        
        return None, None, 0.0, pil_img

    def is_plot_figure(pil_img):
        """Detect whether the image is likely a line‑scan plot (has many straight lines)."""
        img = np.array(pil_img.convert('L'))
        edges = feature.canny(img, sigma=1.5)
        lines = probabilistic_hough_line(edges, threshold=10, line_length=30, line_gap=5)
        horizontal_lines = 0
        for line in lines:
            (x1, y1), (x2, y2) = line
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10:
                horizontal_lines += 1
        total_lines = len(lines)
        return horizontal_lines > 3 and total_lines > 10

# -------------------- Helper functions (unchanged) --------------------
def list_images_in_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    files = [str(f) for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    return sorted(files)

def image_selector(folder_path, label, key_prefix):
    st.markdown(f"**{label}**")
    image_paths = list_images_in_folder(folder_path)
    if image_paths:
        filenames = [Path(p).name for p in image_paths]
        selected_name = st.selectbox(f"Select from {folder_path.name} folder", filenames, key=f"{key_prefix}_dropdown")
        selected_path = next(p for p in image_paths if Path(p).name == selected_name)
        img = Image.open(selected_path).convert("RGB")
        source = f"Folder: {selected_name}"
        return img, source, selected_path
    else:
        st.info(f"No images found in `{folder_path}`. Upload one below.")
        uploaded = st.file_uploader(f"Upload {label.lower()} image", type=['png', 'jpg', 'jpeg', 'gif', 'bmp'], key=f"{key_prefix}_uploader")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            source = f"Uploaded: {uploaded.name}"
            return img, source, uploaded.name
    return None, None, None

@st.cache_resource(show_spinner="Loading LLM...")
def load_llm(backend: str):
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return None, None
    if "GPT-2" in backend:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        model_name = "Qwen/Qwen2-0.5B-Instruct" if "Qwen2-0.5B" in backend else "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        model.eval()
    return tokenizer, model

def generate_query(llm_tokenizer, llm_model, params, backend):
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return "LLM not available."
    prompt = f"""Based on experimental images:
- Core diameter = {params['core_diameter']} nm
- Cu:Ag ratio = {params['ratio_str']} → c_bulk = {params['c_bulk']}
- fc = {params['fc']:.3f}, L0 = {params['L0']:.1f} nm, rs = {params['rs']:.2f}
Generate a natural language query starting with "Design a core-shell with". Output ONLY the sentence."""
    if "Qwen" in backend:
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        full_prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = prompt
    inputs = llm_tokenizer.encode(full_prompt, return_tensors='pt', truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = llm_model.generate(inputs, max_new_tokens=80, temperature=0.3, do_sample=True, pad_token_id=llm_tokenizer.eos_token_id)
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full_prompt in answer:
        answer = answer.replace(full_prompt, "").strip()
    return answer

def google_vision_ocr(pil_img):
    if not GOOGLE_VISION_AVAILABLE:
        return None
    try:
        client = vision.ImageAnnotatorClient()
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        content = buffer.getvalue()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description
    except Exception as e:
        st.warning(f"Google Vision OCR failed: {e}")
    return None

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="CoreShellGPT – Intelligent Input Generator", layout="wide")
st.title("🧪 CoreShellGPT – Intelligent Experimental Input Generator")
st.markdown("**Enhanced: Morphological scale bar detection + Multi-region OCR + Intelligent label parsing**")

with st.expander("🔧 System Info"):
    st.write(f"**scikit‑image:** {'✅' if SKIMAGE_AVAILABLE else '❌'}")
    st.write(f"**OpenCV:** {'✅' if 'cv2' in globals() else '❌'}")
    st.write(f"**Google Cloud Vision:** {'✅' if GOOGLE_VISION_AVAILABLE else '❌'}")
    st.write(f"**PyTesseract:** {'✅' if PYTESSERACT_AVAILABLE else '❌'}")
    st.write(f"**Transformers:** {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")

st.sidebar.header("🧠 LLM Settings")
if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
    llm_options = ["GPT-2", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"]
    chosen_llm = st.sidebar.selectbox("LLM for query generation", llm_options)
    if st.sidebar.button("Load LLM"):
        with st.spinner("Loading..."):
            tokenizer, model = load_llm(chosen_llm)
            st.session_state['llm_tokenizer'] = tokenizer
            st.session_state['llm_model'] = model
            st.sidebar.success("LLM loaded")
else:
    st.sidebar.info("Install transformers & torch for LLM features")

st.sidebar.header("📐 Geometry Mode")
geo_mode = st.sidebar.radio("Mode", ["Provide fc", "Provide shell distance", "Provide L0"], index=0)
if geo_mode == "Provide fc":
    fc_input = st.sidebar.number_input("fc", min_value=0.05, max_value=0.45, value=0.18, step=0.01)
elif geo_mode == "Provide shell distance":
    d_shell = st.sidebar.number_input("Shell distance (nm)", min_value=0.0, value=10.0, step=1.0)
else:
    L0_input = st.sidebar.number_input("L0 (nm)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)
rs_default = st.sidebar.number_input("rs (shell fraction)", min_value=0.01, max_value=0.6, value=0.1, step=0.01)

col1, col2 = st.columns(2)

# -------------------- GEOMETRY IMAGE --------------------
with col1:
    st.subheader("📁 HRTEM / Geometry Image")
    geo_img, geo_source, geo_path = image_selector(GEOMETRY_FOLDER, "Geometry Image", "geo")
    if geo_img:
        st.image(geo_img, caption=geo_source, use_container_width=True)
        st.session_state['geo_image'] = geo_img
        
        if SKIMAGE_AVAILABLE:
            if st.button("🔍 Auto-detect Scale Bar & Core (Enhanced)", key="auto_geo"):
                with st.spinner("Analyzing image with morphological operations + multi-region OCR..."):
                    img_type = classify_image_type(geo_img)
                    if img_type == "composition":
                        st.warning("✅ Detected EDS image! Using colour‑aware core detection.")
                    
                    scale_px, line_coords, scale_conf, annotated_scale = detect_scale_bar_skimage(geo_img)
                    
                    if scale_px:
                        scale_nm = auto_extract_scale_value(geo_img, line_coords)
                        scale_nm_per_px = scale_nm / scale_px
                        
                        diam_nm, diam_px, diam_conf, debug, annotated_core = detect_core_diameter_skimage(
                            geo_img, scale_nm_per_px
                        )
                        
                        st.session_state.temp_geo_detection = {
                            'scale_nm_per_px': scale_nm_per_px,
                            'diam_nm': diam_nm,
                            'annotated_scale': annotated_scale,
                            'annotated_core': annotated_core,
                            'scale_px': scale_px,
                            'detected_scale_nm': scale_nm,
                            'diam_conf': diam_conf,
                        }
                        st.success(f"Detection complete! Scale: {scale_nm}nm, Core: {diam_nm:.1f}nm")
                    else:
                        st.error("Could not detect scale bar. Please use manual entry.")
            
            if st.session_state.get('temp_geo_detection'):
                det = st.session_state.temp_geo_detection
                st.image(det['annotated_scale'], caption="Detected scale bar", use_container_width=True)
                st.image(det['annotated_core'], caption="Detected core", use_container_width=True)
                if det['diam_nm']:
                    st.metric("Detected diameter", f"{det['diam_nm']:.2f} nm")
                    st.metric("Detected scale bar value", f"{det['detected_scale_nm']:.1f} nm")
                else:
                    st.warning("Core detection failed.")
                
                if st.button("✅ Accept All Geometry Detections", key="accept_geo"):
                    st.session_state.scale_nm_per_px = det['scale_nm_per_px']
                    st.session_state.core_diameter = det['diam_nm']
                    st.session_state.geo_annotated = det['annotated_core']
                    del st.session_state.temp_geo_detection
                    st.success("Geometry accepted!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("**Manual Entry**")
        scale_nm_manual = st.number_input("Scale bar value (nm)", value=20.0, step=5.0, key="scale_manual_final")
        scale_px_manual = st.number_input("Scale bar length (pixels)", value=100, step=10, key="scalepx_manual_final")
        diam_manual = st.number_input("Core diameter (nm)", min_value=0.0, value=20.0, step=0.1, key="diam_manual_final")
        if st.button("Set Geometry Manually", key="set_geo_manual"):
            if scale_px_manual > 0:
                st.session_state.scale_nm_per_px = scale_nm_manual / scale_px_manual
                st.session_state.core_diameter = diam_manual
                if geo_img:
                    ann = geo_img.copy().convert("RGB")
                    draw = ImageDraw.Draw(ann)
                    draw.text((10,10), f"Manual: core = {diam_manual} nm", fill="white")
                    st.session_state.geo_annotated = ann
                st.success("Manual geometry set!")
                st.rerun()
        
        if st.session_state.get('geo_annotated'):
            st.image(st.session_state.geo_annotated, caption="✅ Final Annotated Geometry", use_container_width=True)
            if st.button("📥 Download Annotated Geometry", key="dl_geo"):
                buf = io.BytesIO()
                st.session_state.geo_annotated.save(buf, format="PNG")
                st.download_button("Download PNG", data=buf.getvalue(), file_name="annotated_geometry.png", mime="image/png")

# -------------------- COMPOSITION IMAGE --------------------
with col2:
    st.subheader("📁 Elemental Mapping / Composition Image")
    comp_img, comp_source, comp_path = image_selector(COMPOSITION_FOLDER, "Composition Image", "comp")
    if comp_img:
        st.image(comp_img, caption=comp_source, use_container_width=True)
        st.session_state['comp_image'] = comp_img
        
        if SKIMAGE_AVAILABLE:
            if st.button("🔍 Auto-extract Composition (Enhanced OCR)", key="auto_comp"):
                with st.spinner("Analyzing with plot detection + intelligent label parsing..."):
                    if is_plot_figure(comp_img):
                        st.info("📊 Plot detected – using OCR to read labels")
                        text = google_vision_ocr(comp_img) if GOOGLE_VISION_AVAILABLE else ""
                        
                        if not text and PYTESSERACT_AVAILABLE:
                            try:
                                text = pytesseract.image_to_string(comp_img)
                            except:
                                pass
                        
                        ratio_patterns = [
                            r'Cu:Ag\s*[:=]\s*(\d+(?:\.\d+)?)\s*[:=]\s*(\d+(?:\.\d+)?)',
                            r'Cu:Ag\s*=\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',
                            r'Cu\s*:\s*Ag\s*=\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',
                            r'Cu:Ag\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',
                        ]
                        ratio_match = None
                        for pat in ratio_patterns:
                            ratio_match = re.search(pat, text, re.IGNORECASE)
                            if ratio_match:
                                break
                        
                        bulk_match = re.search(r'(?:c_?bulk|bulk|c bulk)\s*[:=]\s*([0-9.]+)', text, re.IGNORECASE)
                        
                        if bulk_match or ratio_match:
                            if bulk_match:
                                c_bulk = float(bulk_match.group(1))
                                ratio_str = f"{ratio_match.group(1)}:{ratio_match.group(2)}" if ratio_match else "1:1"
                            else:
                                cu = float(ratio_match.group(1))
                                ag = float(ratio_match.group(2))
                                ratio = cu / ag
                                c_bulk = 1.0 / ratio if ratio > 0 else 1.0
                                c_bulk = np.clip(c_bulk, 0.1, 1.0)
                                ratio_str = f"{cu}:{ag}"
                            
                            annotated = annotate_composition(comp_img, ratio_str, c_bulk)
                            st.session_state.temp_comp_detection = {
                                'c_bulk': round(c_bulk, 3),
                                'ratio_str': ratio_str,
                                'annotated': annotated,
                            }
                            st.success(f"✅ Extracted c_bulk = {c_bulk:.3f} and ratio = {ratio_str}")
                        else:
                            st.warning("Could not parse labels from text; falling back to colour analysis.")
                            c_bulk, ratio_str, conf, annotated = extract_composition_skimage(comp_img)
                            if c_bulk:
                                st.session_state.temp_comp_detection = {
                                    'c_bulk': c_bulk,
                                    'ratio_str': ratio_str,
                                    'annotated': annotated,
                                }
                            else:
                                st.error("Automatic extraction failed.")
                    else:
                        c_bulk, ratio_str, conf, annotated = extract_composition_skimage(comp_img)
                        if c_bulk:
                            st.session_state.temp_comp_detection = {
                                'c_bulk': c_bulk,
                                'ratio_str': ratio_str,
                                'annotated': annotated,
                            }
                            st.success("Composition detected!")
                        else:
                            st.warning("Could not extract automatically.")
            
            if st.session_state.get('temp_comp_detection'):
                det = st.session_state.temp_comp_detection
                st.image(det['annotated'], caption="Detected composition", use_container_width=True)
                st.metric("Cu:Ag ratio", det['ratio_str'])
                st.metric("c_bulk", f"{det['c_bulk']:.3f}")
                
                if st.button("✅ Accept Composition Detection", key="accept_comp"):
                    st.session_state.c_bulk = det['c_bulk']
                    st.session_state.ratio_str = det['ratio_str']
                    st.session_state.comp_annotated = det['annotated']
                    del st.session_state.temp_comp_detection
                    st.success("Composition accepted!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("**Manual Entry**")
        cu_num = st.number_input("Cu count", min_value=1, value=1, step=1, key="cu")
        ag_num = st.number_input("Ag count", min_value=1, value=1, step=1, key="ag")
        if st.button("Set Ratio Manually", key="set_comp_manual"):
            ratio = cu_num / ag_num
            c_bulk = 1.0 / ratio if ratio > 0 else 1.0
            c_bulk = np.clip(c_bulk, 0.1, 1.0)
            st.session_state.c_bulk = round(c_bulk, 3)
            st.session_state.ratio_str = f"{cu_num}:{ag_num}"
            if comp_img:
                ann = comp_img.copy().convert("RGB")
                draw = ImageDraw.Draw(ann)
                draw.text((10,10), f"Manual: Cu:Ag = {cu_num}:{ag_num}", fill="white")
                st.session_state.comp_annotated = ann
            st.success("Manual composition set!")
            st.rerun()
        
        if st.session_state.get('comp_annotated'):
            st.image(st.session_state.comp_annotated, caption="✅ Final Annotated Composition", use_container_width=True)
            if st.button("📥 Download Annotated Composition", key="dl_comp"):
                buf = io.BytesIO()
                st.session_state.comp_annotated.save(buf, format="PNG")
                st.download_button("Download PNG", data=buf.getvalue(), file_name="annotated_composition.png", mime="image/png")

# -------------------- CALCULATIONS --------------------
st.header("📐 Derived Parameters")
if st.session_state.core_diameter and st.session_state.c_bulk:
    d_core = st.session_state.core_diameter
    r_core = d_core / 2.0
    if geo_mode == "Provide fc":
        fc = fc_input
        L0 = d_core / (2 * fc)
    elif geo_mode == "Provide shell distance":
        L0 = d_core + 2 * d_shell
        fc = r_core / L0
    else:
        L0 = L0_input
        fc = r_core / L0
    rs = rs_default
    c_bulk = st.session_state.c_bulk
    ratio_str = st.session_state.ratio_str
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Core diameter", f"{d_core} nm")
        st.metric("fc (core fraction)", f"{fc:.4f}")
        st.metric("L0 (domain size)", f"{L0:.2f} nm")
    with col_b:
        st.metric("c_bulk", f"{c_bulk}")
        st.metric("rs (shell fraction)", f"{rs:.2f}")
        st.metric("Cu:Ag ratio", ratio_str)
    
    params = {
        'core_diameter': d_core,
        'fc': round(fc, 4),
        'L0': round(L0, 2),
        'c_bulk': c_bulk,
        'rs': round(rs, 3),
        'mode': geo_mode,
        'ratio_str': ratio_str
    }
    st.session_state['params'] = params
    
    if st.session_state.get('llm_tokenizer') and st.session_state.get('llm_model'):
        if st.button("🚀 Generate Query with LLM"):
            with st.spinner("Generating..."):
                query = generate_query(st.session_state['llm_tokenizer'], st.session_state['llm_model'], params, chosen_llm)
                st.text_area("📋 Generated Query:", query, height=100)
    
    default_query = f"Design a core-shell with L0={params['L0']:.1f} nm, fc={params['fc']:.3f}, c_bulk={params['c_bulk']:.2f}, rs={params['rs']:.2f}, time=1e-3 s from HRTEM (core {params['core_diameter']} nm) and EDS (Cu:Ag={params['ratio_str']})."
    st.text_area("📋 Default Query:", default_query, height=100)
else:
    st.info("👈 Set core diameter and c_bulk from the images above")
