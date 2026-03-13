#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT – Intelligent Experimental Input Generator
(ENHANCED: Robust scale-bar blob detection, multi-region OCR, 
 corner-based label scanning, and improved regex parsing)
"""
import streamlit as st
import os
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import cv2 # Added for robust morphological operations

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
    st.warning("scikit‑image not installed. Install with: pip install scikit-image")

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
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

# Try importing pytesseract as a local fallback for OCR
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
    # Simple heuristic: EDS maps usually have distinct red/green channels on black
    r = img_rgb[:,:,0]
    g = img_rgb[:,:,1]
    # If significant red and green exist but blue is low, likely EDS
    if np.mean(r) > 30 and np.mean(g) > 30 and np.mean(img_rgb[:,:,2]) < 30:
        return "composition"
    return "geometry"

# -------------------- ENHANCED: Smart OCR Region Search --------------------
def smart_ocr_search(pil_img, target_patterns=None):
    """
    Scans specific regions (corners) for text patterns instead of whole image 
    to reduce noise and improve speed/accuracy.
    """
    if not GOOGLE_VISION_AVAILABLE and not PYTESSERACT_AVAILABLE:
        return ""
    
    w, h = pil_img.size
    # Define regions of interest (ROIs) where labels usually are
    # Bottom-Right, Top-Right, Bottom-Left, Top-Left
    rois = [
        (int(w*0.5), int(h*0.5), w, h), # Bottom Right
        (int(w*0.5), 0, w, int(h*0.5)), # Top Right
        (0, int(h*0.5), int(w*0.5), h), # Bottom Left
        (0, 0, int(w*0.5), int(h*0.5))  # Top Left
    ]
    
    full_text = ""
    
    for i, (x, y, r_x, r_y) in enumerate(rois):
        crop = pil_img.crop((x, y, r_x, r_y))
        text = ""
        
        if GOOGLE_VISION_AVAILABLE:
            try:
                client = vision.ImageAnnotatorClient()
                buffer = io.BytesIO()
                crop.save(buffer, format='PNG')
                image = vision.Image(content=buffer.getvalue())
                response = client.text_detection(image=image)
                if response.text_annotations:
                    text = response.text_annotations[0].description
            except Exception:
                pass
        elif PYTESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(crop, config='--psm 6')
            except Exception:
                pass
        
        if text:
            full_text += "\n" + text
            
    return full_text

# -------------------- ENHANCED: Scale Bar Detection (Morphological) --------------------
def detect_scale_bar_skimage(pil_img):
    """
    Detect horizontal scale bar using Morphological Operations (more robust than Hough Lines).
    Looks for bright white horizontal rectangles.
    """
    img = np.array(pil_img.convert('L'))
    height, width = img.shape
    
    # 1. Threshold to find bright objects (scale bars are usually white #FFFFFF)
    # Use a high threshold to ignore noise
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological Closing to connect broken parts of the bar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)) # Horizontal kernel
    closed = cv2.morphologyOps(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        
        # Heuristics for a scale bar:
        # - Wide and short (high aspect ratio)
        # - Reasonable area (not too small, not the whole image)
        # - Located near edges (bottom or top 20%)
        y_center = y + h/2
        is_near_edge = (y_center < height * 0.25) or (y_center > height * 0.75)
        
        if aspect_ratio > 5 and 1000 < area < (width * height * 0.1) and is_near_edge:
            candidates.append((area, (x, y, x+w, y+h))) # Store area and rect
    
    if not candidates:
        # Fallback to original Hough Line method if morphological fails
        return _fallback_hough_detection(pil_img)

    # Sort by area (largest likely the scale bar)
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_rect = candidates[0][1]
    
    # Convert rect to line coordinates (center line of the bar)
    x1, y1, x2, y2 = best_rect
    line_coords = ((x1, y1 + (y2-y1)//2), (x2, y2 - (y2-y1)//2))
    length_px = x2 - x1
    
    confidence = 0.85
    annotated = annotate_geometry(pil_img, scale_line=line_coords, core_info=None)
    
    return length_px, line_coords, confidence, annotated

def _fallback_hough_detection(pil_img):
    """Original Hough Line logic as fallback."""
    img = np.array(pil_img.convert('L'))
    height, width = img.shape
    edges = feature.canny(img, sigma=2)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=20, line_gap=3)
    horizontal_lines = []
    for line in lines:
        (x1, y1), (x2, y2) = line
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 10 and length > 40:
            y_center = (y1 + y2) / 2
            if y_center < 0.25 * height or y_center > 0.75 * height:
                horizontal_lines.append((length, ((x1,y1),(x2,y2))))
    
    if not horizontal_lines:
        return None, None, 0.0, pil_img
        
    horizontal_lines.sort(key=lambda x: x[0], reverse=True)
    length_px, line_coords = horizontal_lines[0]
    annotated = annotate_geometry(pil_img, scale_line=line_coords, core_info=None)
    return length_px, line_coords, 0.7, annotated

# -------------------- ENHANCED: Auto Scale Value Extraction --------------------
def auto_extract_scale_value(pil_img, line_coords):
    """
    Extract numeric scale value (nm) by searching regions AROUND the detected line,
    not just below it.
    """
    if not line_coords:
        return 20.0
        
    (x1, y1), (x2, y2) = line_coords
    y_bar = (y1 + y2) / 2
    x_min, x_max = min(x1, x2), max(x1, x2)
    
    # Define search regions relative to the bar
    # 1. Directly Above
    # 2. Directly Below
    # 3. Immediate Vicinity (Bounding Box)
    
    regions = [
        (x_min - 50, y_bar - 80, x_max + 50, y_bar - 10), # Above
        (x_min - 50, y_bar + 10, x_max + 50, y_bar + 80), # Below
        (x_min - 50, y_bar - 80, x_max + 50, y_bar + 80)  # Surrounding
    ]
    
    found_value = None
    
    for (left, top, right, bottom) in regions:
        # Clamp coordinates
        left = max(0, int(left))
        top = max(0, int(top))
        right = min(pil_img.width, int(right))
        bottom = min(pil_img.height, int(bottom))
        
        if right <= left or bottom <= top:
            continue
            
        crop = pil_img.crop((left, top, right, bottom))
        text = ""
        
        if GOOGLE_VISION_AVAILABLE:
            try:
                client = vision.ImageAnnotatorClient()
                buffer = io.BytesIO()
                crop.save(buffer, format='PNG')
                image = vision.Image(content=buffer.getvalue())
                response = client.text_detection(image=image)
                if response.text_annotations:
                    text = response.text_annotations[0].description
            except Exception:
                pass
        elif PYTESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(crop, config='--psm 7') # Single line mode
            except Exception:
                pass
        
        if text:
            # Robust Regex for "20 nm", "50nm", "100 nm", "1 um"
            match = re.search(r'(\d+\.?\d*)\s*(nm|um|μm)', text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                unit = match.group(2).lower()
                if unit in ['um', 'μm']:
                    val *= 1000 # Convert to nm
                found_value = val
                break # Found it, stop searching
    
    # Fallback: Global search for "nm" if local failed
    if found_value is None:
        global_text = smart_ocr_search(pil_img)
        match = re.search(r'(\d+\.?\d*)\s*(nm|um|μm)', global_text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            unit = match.group(2).lower()
            if unit in ['um', 'μm']:
                val *= 1000
            found_value = val

    return found_value if found_value else 20.0

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

# -------------------- Annotation helper --------------------
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

    if scale_line:
        (x1, y1), (x2, y2) = scale_line
        draw.line([(x1, y1), (x2, y2)], fill="#FF0000", width=6)
        draw.rectangle([(min(x1,x2)-10, min(y1,y2)-15),
                        (max(x1,x2)+10, max(y1,y2)+15)],
                        outline="#FF0000", width=3)
        draw.text((min(x1,x2), min(y1,y2)-25), "SCALE BAR", fill="#FF0000", font=font)

    if core_info:
        cx, cy, r = core_info["center"][0], core_info["center"][1], core_info["radius_px"]
        draw.ellipse((cx-r-3, cy-r-3, cx+r+3, cy+r+3), outline="#00FF00", width=5)
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
        draw.text((cx-30, cy-15), label, fill="white", font=font, stroke_width=2, stroke_fill="black")
        draw.line([(cx, cy-30), (cx+50, cy-60)], fill="white", width=3)
        draw.text((cx+60, cy-70), "Cu CORE", fill="white", font=font, stroke_width=3)
        
        legend_x, legend_y = 10, 10
        draw.rectangle([(legend_x, legend_y), (legend_x+200, legend_y+60)], outline="white", width=2)
        draw.rectangle([(legend_x+5, legend_y+5), (legend_x+25, legend_y+25)], fill="#00FF00")
        draw.text((legend_x+35, legend_y+5), "Cu core", fill="white", font=small_font)
        draw.rectangle([(legend_x+5, legend_y+30), (legend_x+25, legend_y+50)], fill="#FFA500")
        draw.text((legend_x+35, legend_y+30), "Ag shell", fill="white", font=small_font)

    return annotated

def annotate_composition(pil_img, ratio_str, c_bulk, masks=None):
    """Add a text overlay with the detected Cu:Ag ratio."""
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

# -------------------- Image analysis functions --------------------
if SKIMAGE_AVAILABLE:
    def detect_core_diameter_skimage(pil_img, scale_nm_per_px=None):
        img_type = classify_image_type(pil_img)
        if img_type == "composition":
            img_rgb = np.array(pil_img.convert('RGB'))
            hsv = color.rgb2hsv(img_rgb)
            red_mask = ((hsv[..., 0] < 0.05) | (hsv[..., 0] > 0.95)) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3)
            diam_nm, diam_px, conf, core_info = extract_core_from_mask(red_mask, scale_nm_per_px)
            if diam_nm:
                annotated = annotate_geometry(pil_img, scale_line=None, core_info=core_info)
                return diam_nm, diam_px, conf, {'method': 'mask_eds'}, annotated
            else:
                return None, None, 0.0, {'method': 'mask_failed'}, pil_img
        
        # Geometry image logic (Hough/Contours) remains similar to original
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
            diam_nm = diameter_px * scale_nm_per_px if scale_nm_per_px else None
            core_info = {'center': (cx[best_idx], cy[best_idx]), 'radius_px': radius_px, 'diam_nm': diam_nm, 'contour': None}
            annotated = annotate_geometry(pil_img, scale_line=None, core_info=core_info)
            return diam_nm, diameter_px, 0.8, {'method': 'hough'}, annotated
            
        return None, None, 0.0, {}, pil_img

    def extract_composition_skimage(pil_img):
        """Enhanced: Tries OCR first for labels, falls back to color analysis."""
        
        # 1. Try Intelligent OCR for "Cu:Ag" or "c_bulk"
        text = smart_ocr_search(pil_img)
        
        # Regex patterns for ratios
        ratio_patterns = [
            r'Cu:Ag\s*[:=]\s*(\d+(?:\.\d+)?)\s*[:=]\s*(\d+(?:\.\d+)?)',
            r'Cu:Ag\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*Cu:Ag', # Reverse order
        ]
        
        for pat in ratio_patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                cu = float(match.group(1))
                ag = float(match.group(2))
                if ag > 0:
                    ratio = cu / ag
                    c_bulk = 1.0 / ratio if ratio > 0 else 1.0
                    c_bulk = np.clip(c_bulk, 0.1, 1.0)
                    ratio_str = f"{cu}:{ag}"
                    annotated = annotate_composition(pil_img, ratio_str, c_bulk)
                    return c_bulk, ratio_str, 0.9, annotated

        # 2. Fallback to Color Analysis
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
            annotated = annotate_composition(pil_img, ratio_str, c_bulk)
            return c_bulk, ratio_str, 0.6, annotated
            
        return None, None, 0.0, pil_img

# -------------------- Helper functions --------------------
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

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="CoreShellGPT – Intelligent Input Generator", layout="wide")
st.title("🧪 CoreShellGPT – Intelligent Input Generator")
st.markdown("**Enhanced Detection: Morphological Scale Bars + Smart Corner OCR + Ratio Parsing**")

with st.expander("🔧 System Info"):
    st.write(f"**scikit‑image:** {'✅' if SKIMAGE_AVAILABLE else '❌'}")
    st.write(f"**OpenCV:** {'✅' if 'cv2' in globals() else '❌'}")
    st.write(f"**Google Cloud Vision:** {'✅' if GOOGLE_VISION_AVAILABLE else '❌'}")
    st.write(f"**PyTesseract (Local OCR):** {'✅' if PYTESSERACT_AVAILABLE else '❌'}")
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
                with st.spinner("Analyzing image (Morphology + OCR)..."):
                    # 1. Detect Scale Bar
                    scale_px, line_coords, scale_conf, annotated_scale = detect_scale_bar_skimage(geo_img)
                    
                    if scale_px:
                        # 2. Extract Value using Smart OCR
                        scale_nm = auto_extract_scale_value(geo_img, line_coords)
                        scale_nm_per_px = scale_nm / scale_px
                        
                        # 3. Detect Core
                        diam_nm, diam_px, diam_conf, debug, annotated_core = detect_core_diameter_skimage(geo_img, scale_nm_per_px)
                        
                        st.session_state.temp_geo_detection = {
                            'scale_nm_per_px': scale_nm_per_px,
                            'diam_nm': diam_nm,
                            'annotated_scale': annotated_scale,
                            'annotated_core': annotated_core,
                            'scale_px': scale_px,
                            'detected_scale_nm': scale_nm
                        }
                        st.success(f"Detection complete! Scale: {scale_nm}nm, Core: {diam_nm:.1f}nm")
                    else:
                        st.error("Could not detect scale bar automatically. Please use manual entry.")

            if st.session_state.get('temp_geo_detection'):
                det = st.session_state.temp_geo_detection
                st.image(det['annotated_scale'], caption="Detected scale bar", use_container_width=True)
                st.image(det['annotated_core'], caption="Detected core", use_container_width=True)
                st.metric("Detected diameter", f"{det['diam_nm']:.2f} nm")
                
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
            if st.button("🔍 Auto-extract Composition (Smart OCR)", key="auto_comp"):
                with st.spinner("Scanning corners for labels..."):
                    c_bulk, ratio_str, conf, annotated = extract_composition_skimage(comp_img)
                    
                    if c_bulk is not None:
                        st.session_state.temp_comp_detection = {
                            'c_bulk': c_bulk,
                            'ratio_str': ratio_str,
                            'annotated': annotated,
                        }
                        st.success(f"✅ Extracted c_bulk = {c_bulk:.3f} and ratio = {ratio_str}")
                    else:
                        st.warning("Automatic extraction failed. Please enter manually.")

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
