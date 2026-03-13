#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT – Intelligent Experimental Input Generator
(Enhanced: robust scale bar detection + label OCR, plot recognition, intelligent parsing)
"""

import streamlit as st
import os
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io

# -------------------- SESSION STATE INITIALISATION --------------------
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

# -------------------- Optional dependency checks --------------------
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
    st.warning("Transformers or PyTorch not installed. LLM features disabled.")

try:
    from google.cloud import vision
    import io
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

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

# -------------------- Core detection for geometry images (Hough + contour) --------------------
def detect_core_diameter_skimage(pil_img, scale_nm_per_px=None):
    """Detect circular core for HRTEM images (uses red channel + CLAHE)."""
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

    # Fallback: contour analysis
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

# -------------------- Enhanced scale bar detection with label OCR --------------------
def parse_scale_value_from_text(text):
    """Extract numeric value followed by 'nm' from text."""
    if not text:
        return None
    patterns = [
        r'(\d+\.?\d*)\s*nm',
        r'(\d+\.?\d*)\s*nanometer',
        r'(\d+\.?\d*)\s*nanometers',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None

def extract_text_from_region(pil_img, bbox):
    """Run Google Vision OCR on a cropped region."""
    left, top, right, bottom = bbox
    crop = pil_img.crop((left, top, right, bottom))
    if not GOOGLE_VISION_AVAILABLE:
        return ""
    try:
        client = vision.ImageAnnotatorClient()
        buffer = io.BytesIO()
        crop.save(buffer, format='PNG')
        content = buffer.getvalue()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        if response.text_annotations:
            return response.text_annotations[0].description
    except Exception:
        pass
    return ""

def search_scale_label_near_line(pil_img, line_coords):
    """
    Search for scale bar label (number + nm) in regions around the line.
    Returns the numeric value if found, else None.
    """
    (x1, y1), (x2, y2) = line_coords
    img_width, img_height = pil_img.size

    # Define search regions (priority order)
    search_regions = [
        # Below line
        (max(0, min(x1, x2) - 20),
         max(y1, y2) + 5,
         min(img_width, max(x1, x2) + 20),
         min(img_height, max(y1, y2) + 50)),
        # Above line
        (max(0, min(x1, x2) - 20),
         max(0, min(y1, y2) - 50),
         min(img_width, max(x1, x2) + 20),
         min(y1, y2) - 5),
        # Right side
        (max(0, max(x1, x2) - 30),
         max(0, min(y1, y2) - 30),
         min(img_width, max(x1, x2) + 40),
         min(img_height, max(y1, y2) + 30)),
        # Left side
        (max(0, min(x1, x2) - 40),
         max(0, min(y1, y2) - 30),
         min(img_width, min(x1, x2) - 5),
         min(img_height, max(y1, y2) + 30)),
    ]

    for bbox in search_regions:
        left, top, right, bottom = bbox
        if right <= left or bottom <= top:
            continue
        text = extract_text_from_region(pil_img, bbox)
        if text:
            value = parse_scale_value_from_text(text)
            if value:
                return value
    return None

def detect_scale_bar_with_label(pil_img):
    """
    Enhanced scale bar detection that:
    1. Finds horizontal lines (with brightness, border, and position filters)
    2. Searches for nearby text containing a number + 'nm'
    3. Returns line parameters and the extracted scale value
    Always returns 5 values: length_px, line_coords, confidence, annotated_img, scale_nm
    """
    img_gray = np.array(pil_img.convert('L'))
    height, width = img_gray.shape
    edges = feature.canny(img_gray, sigma=2)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=20, line_gap=3)

    candidate_bars = []

    for line in lines:
        (x1, y1), (x2, y2) = line
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # Filter for near‑horizontal lines
        if angle < 10 and length > 30:
            # Exclude lines touching the border
            margin = 10
            if y1 < margin or y2 < margin or y1 > height - margin or y2 > height - margin:
                continue
            # Brightness check (scale bars are white)
            y_min, y_max = max(0, min(y1, y2) - 5), min(height, max(y1, y2) + 5)
            x_min, x_max = max(0, min(x1, x2) - 5), min(width, max(x1, x2) + 5)
            line_region = img_gray[y_min:y_max, x_min:x_max]
            if line_region.size > 0 and np.mean(line_region) < 200:
                continue
            # Position filter: prefer top or bottom quarter
            y_center = (y1 + y2) / 2
            if y_center < 0.25 * height or y_center > 0.75 * height:
                # Search for label near this line
                scale_nm = search_scale_label_near_line(pil_img, line)
                if scale_nm:
                    # Score: longer lines and more rightward are preferred
                    score = length * ((x1 + x2) / (2 * width))
                    candidate_bars.append({
                        'line': line,
                        'length_px': length,
                        'label_nm': scale_nm,
                        'confidence': min(0.9, length / 200),
                        'score': score
                    })

    if not candidate_bars:
        # No candidate found – return five values (none)
        return None, None, 0.0, pil_img, None

    # Pick the candidate with highest score
    candidate_bars.sort(key=lambda x: x['score'], reverse=True)
    best = candidate_bars[0]

    annotated = annotate_geometry(pil_img, scale_line=best['line'], core_info=None)
    return best['length_px'], best['line'], best['confidence'], annotated, best['label_nm']

# -------------------- Annotation helpers --------------------
def annotate_geometry(pil_img, scale_line=None, core_info=None):
    """Return annotated image with scale bar and core highlighted."""
    annotated = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = font

    if scale_line:
        (x1, y1), (x2, y2) = scale_line
        draw.line([(x1, y1), (x2, y2)], fill="#FF0000", width=6)
        draw.rectangle([(min(x1,x2)-10, min(y1,y2)-15),
                        (max(x1,x2)+10, max(y1,y2)+15)],
                       outline="#FF0000", width=3)
        draw.text((min(x1,x2), min(y1,y2)-25),
                  "SCALE BAR", fill="#FF0000", font=font)

    if core_info:
        cx, cy, r = core_info["center"][0], core_info["center"][1], core_info["radius_px"]
        draw.ellipse((cx-r-3, cy-r-3, cx+r+3, cy+r+3),
                     outline="#00FF00", width=5)
        diam_nm = core_info.get("diam_nm", 0)
        label = f"{diam_nm:.1f} nm" if diam_nm else "core"
        draw.text((cx-30, cy-15), label,
                  fill="white", font=font, stroke_width=2, stroke_fill="black")

    return annotated

def annotate_composition(pil_img, ratio_str, c_bulk):
    """Add text overlay with detected composition."""
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

# -------------------- Plot detection and composition parsing --------------------
def is_plot_figure(pil_img):
    """Detect if image is a line‑scan plot (many horizontal lines)."""
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

def parse_composition_from_text(text):
    """Intelligently parse composition ratio and c_bulk from OCR text."""
    if not text:
        return None, None, "Empty text"

    text = text.replace('\n', ' ').replace('\t', ' ')

    # Ratio patterns
    ratio_patterns = [
        r'Cu\s*:\s*Ag\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[:=]\s*(\d+(?:\.\d+)?)',
        r'Cu:Ag\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',
        r'ratio\s*[:=]?\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*(?:Cu:Ag|ratio)',
    ]

    # c_bulk patterns
    bulk_patterns = [
        r'c[_\s]?bulk\s*[:=]\s*([0-9.]+)',
        r'bulk\s*[:=]\s*([0-9.]+)',
        r'c\s*[:=]\s*([0-9.]+)\s*bulk',
    ]

    ratio_match = None
    bulk_match = None

    for pattern in ratio_patterns:
        ratio_match = re.search(pattern, text, re.IGNORECASE)
        if ratio_match:
            break

    for pattern in bulk_patterns:
        bulk_match = re.search(pattern, text, re.IGNORECASE)
        if bulk_match:
            break

    if bulk_match:
        c_bulk = float(bulk_match.group(1))
        c_bulk = np.clip(c_bulk, 0.0, 1.0)

        if ratio_match:
            cu_val = float(ratio_match.group(1))
            ag_val = float(ratio_match.group(2))
            ratio_str = f"{cu_val}:{ag_val}"
        else:
            # Estimate ratio from c_bulk
            if c_bulk > 0:
                ratio = 1.0 / c_bulk
                ratio_str = f"{ratio:.1f}:1"
            else:
                ratio_str = "1:1"

        return c_bulk, ratio_str, "Success"

    elif ratio_match:
        cu_val = float(ratio_match.group(1))
        ag_val = float(ratio_match.group(2))

        if ag_val > 0:
            ratio = cu_val / ag_val
            c_bulk = 1.0 / ratio if ratio > 0 else 1.0
        else:
            c_bulk = 1.0

        c_bulk = np.clip(c_bulk, 0.0, 1.0)
        ratio_str = f"{cu_val}:{ag_val}"

        return c_bulk, ratio_str, "Success"

    return None, None, "No match found"

def detect_plot_and_extract_labels(pil_img):
    """Run OCR on the whole image and parse composition from text."""
    if GOOGLE_VISION_AVAILABLE:
        try:
            client = vision.ImageAnnotatorClient()
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            image = vision.Image(content=buffer.getvalue())
            response = client.text_detection(image=image)
            if response.text_annotations:
                text = response.text_annotations[0].description
                return parse_composition_from_text(text)
        except Exception as e:
            st.warning(f"OCR failed: {e}")

    return None, None, "No text detected"

# -------------------- Colour‑based composition extraction (for EDS maps) --------------------
def extract_composition_skimage(pil_img):
    """Color‑based analysis for EDS maps (red = Cu, green = Ag)."""
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
        ratio_str = f"{round(ratio, 1)}:1"
        confidence = 0.6
        annotated = annotate_composition(pil_img, ratio_str, c_bulk)
        return c_bulk, ratio_str, confidence, annotated

    return None, None, 0.0, pil_img

# -------------------- Helper functions --------------------
def list_images_in_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    files = [str(f) for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions]
    return sorted(files)

def image_selector(folder_path, label, key_prefix):
    st.markdown(f"**{label}**")
    image_paths = list_images_in_folder(folder_path)

    if image_paths:
        filenames = [Path(p).name for p in image_paths]
        selected_name = st.selectbox(
            f"Select from {folder_path.name} folder",
            filenames,
            key=f"{key_prefix}_dropdown"
        )
        selected_path = next(p for p in image_paths if Path(p).name == selected_name)
        img = Image.open(selected_path).convert("RGB")
        source = f"Folder: {selected_name}"
        return img, source, selected_path
    else:
        st.info(f"No images found in `{folder_path}`. Upload one below.")
        uploaded = st.file_uploader(
            f"Upload {label.lower()} image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            key=f"{key_prefix}_uploader"
        )
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

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="CoreShellGPT – Enhanced Detection", layout="wide")
st.title("🧪 CoreShellGPT – Intelligent Input Generator")
st.markdown("**Enhanced scale bar & label detection with intelligent parsing**")

with st.expander("🔧 System Info"):
    st.write(f"**scikit‑image:** {'✅' if SKIMAGE_AVAILABLE else '❌'}")
    st.write(f"**Google Cloud Vision:** {'✅' if GOOGLE_VISION_AVAILABLE else '❌'}")
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
            if st.button("🔍 Auto-detect Scale Bar (Enhanced)", key="auto_geo"):
                with st.spinner("Analyzing image with enhanced detection..."):
                    # Use enhanced scale bar detection (returns 5 values)
                    scale_px, line_coords, scale_conf, annotated_scale, scale_value = detect_scale_bar_with_label(geo_img)

                    if scale_px and scale_value:
                        st.success(f"✅ Detected scale bar: {scale_value} nm ({scale_px:.1f} px)")
                        scale_nm_per_px = scale_value / scale_px

                        # Detect core (colour‑aware)
                        img_type = classify_image_type(geo_img)
                        if img_type == "composition":
                            # EDS image – use red mask
                            img_rgb = np.array(geo_img.convert('RGB'))
                            hsv = color.rgb2hsv(img_rgb)
                            red_mask = ((hsv[..., 0] < 0.05) | (hsv[..., 0] > 0.95)) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3)
                            diam_nm, diam_px, diam_conf, core_info = extract_core_from_mask(red_mask, scale_nm_per_px)
                            if core_info:
                                # Create annotated core image
                                annotated_core = annotate_geometry(geo_img, scale_line=None, core_info=core_info)
                            else:
                                annotated_core = geo_img
                        else:
                            # HRTEM – use Hough on red channel
                            diam_nm, diam_px, diam_conf, debug, annotated_core = detect_core_diameter_skimage(geo_img, scale_nm_per_px)

                        if diam_nm:
                            st.session_state.temp_geo_detection = {
                                'scale_nm_per_px': scale_nm_per_px,
                                'diam_nm': diam_nm,
                                'annotated_scale': annotated_scale,
                                'annotated_core': annotated_core,
                                'scale_px': scale_px,
                                'scale_value': scale_value,
                            }
                            st.image(annotated_scale, caption="Detected scale bar", use_container_width=True)
                            st.image(annotated_core, caption="Detected core", use_container_width=True)
                            st.metric("Detected diameter", f"{diam_nm:.2f} nm")
                        else:
                            st.warning("Scale bar detected but core detection failed – please enter core diameter manually.")
                            # Still store scale
                            st.session_state.temp_geo_detection = {
                                'scale_nm_per_px': scale_nm_per_px,
                                'annotated_scale': annotated_scale,
                                'scale_px': scale_px,
                                'scale_value': scale_value,
                            }
                    else:
                        st.error("Could not detect scale bar. Please use manual entry.")

            if st.session_state.get('temp_geo_detection'):
                det = st.session_state.temp_geo_detection
                if 'annotated_scale' in det:
                    st.image(det['annotated_scale'], caption="Detected scale bar", use_container_width=True)
                if 'annotated_core' in det:
                    st.image(det['annotated_core'], caption="Detected core", use_container_width=True)
                if det.get('diam_nm'):
                    st.metric("Detected diameter", f"{det['diam_nm']:.2f} nm")
                st.metric("Detected scale bar value", f"{det.get('scale_value', 20):.1f} nm")

                if st.button("✅ Accept All Geometry Detections", key="accept_geo"):
                    st.session_state.scale_nm_per_px = det['scale_nm_per_px']
                    st.session_state.core_diameter = det.get('diam_nm')
                    st.session_state.geo_annotated = det.get('annotated_core', det['annotated_scale'])
                    del st.session_state.temp_geo_detection
                    st.success("Geometry accepted!")
                    st.rerun()

        st.markdown("---")
        st.markdown("**Manual Entry**")
        scale_nm_manual = st.number_input("Scale bar value (nm)", value=20.0, step=5.0, key="scale_manual")
        scale_px_manual = st.number_input("Scale bar length (pixels)", value=100, step=10, key="scalepx_manual")
        diam_manual = st.number_input("Core diameter (nm)", min_value=0.0, value=20.0, step=0.1, key="diam_manual")

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
            if st.button("🔍 Auto-extract Composition (Enhanced)", key="auto_comp"):
                with st.spinner("Analyzing with intelligent label detection..."):
                    # Check if it's a plot
                    if is_plot_figure(comp_img):
                        st.info("📊 Plot detected – using OCR and parsing")
                        c_bulk, ratio_str, status = detect_plot_and_extract_labels(comp_img)

                        if c_bulk and ratio_str:
                            st.success(f"✅ Extracted: c_bulk = {c_bulk:.3f}, ratio = {ratio_str}")
                            annotated = annotate_composition(comp_img, ratio_str, c_bulk)
                            st.session_state.temp_comp_detection = {
                                'c_bulk': round(c_bulk, 3),
                                'ratio_str': ratio_str,
                                'annotated': annotated,
                            }
                            st.image(annotated, caption="Detected composition", use_container_width=True)
                        else:
                            st.warning(f"Could not extract from plot: {status}. Falling back to colour analysis.")
                            c_bulk, ratio_str, conf, annotated = extract_composition_skimage(comp_img)
                            if c_bulk:
                                st.session_state.temp_comp_detection = {
                                    'c_bulk': c_bulk,
                                    'ratio_str': ratio_str,
                                    'annotated': annotated,
                                }
                    else:
                        # EDS map – use colour analysis
                        c_bulk, ratio_str, conf, annotated = extract_composition_skimage(comp_img)
                        if c_bulk:
                            st.session_state.temp_comp_detection = {
                                'c_bulk': c_bulk,
                                'ratio_str': ratio_str,
                                'annotated': annotated,
                            }
                            st.success("Composition detected via colour analysis!")
                            st.image(annotated, caption="Detected composition", use_container_width=True)
                        else:
                            st.error("Automatic extraction failed. Please enter manually.")

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
        cu_num = st.number_input("Cu count", min_value=1, value=1, step=1, key="cu_manual")
        ag_num = st.number_input("Ag count", min_value=1, value=1, step=1, key="ag_manual")

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

if st.session_state.get('scale_nm_per_px') and st.session_state.get('core_diameter') is not None and st.session_state.get('c_bulk') is not None:
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
                prompt = f"""Based on experimental parameters:
                - Core diameter = {params['core_diameter']} nm
                - Cu:Ag ratio = {params['ratio_str']} → c_bulk = {params['c_bulk']}
                - fc = {params['fc']:.3f}, L0 = {params['L0']:.1f} nm, rs = {params['rs']:.2f}
                Generate a natural language query starting with 'Design a core-shell with'. Output ONLY the sentence."""

                tokenizer = st.session_state.llm_tokenizer
                model = st.session_state.llm_model

                if "Qwen" in chosen_llm:
                    messages = [{"role": "system", "content": "You are a helpful assistant."},
                               {"role": "user", "content": prompt}]
                    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    full_prompt = prompt

                inputs = tokenizer.encode(full_prompt, return_tensors='pt', truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')

                with torch.no_grad():
                    outputs = model.generate(inputs, max_new_tokens=80, temperature=0.3, do_sample=True,
                                           pad_token_id=tokenizer.eos_token_id)

                query = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if full_prompt in query:
                    query = query.replace(full_prompt, "").strip()

                st.text_area("📋 Generated Query:", query, height=100)

    default_query = f"Design a core-shell with L0={params['L0']:.1f} nm, fc={params['fc']:.3f}, c_bulk={params['c_bulk']:.2f}, rs={params['rs']:.2f}, time=1e-3 s from HRTEM (core {params['core_diameter']} nm) and EDS (Cu:Ag={params['ratio_str']})."
    st.text_area("📋 Default Query:", default_query, height=100)

else:
    st.info("👈 Set core diameter and composition from the images above")
