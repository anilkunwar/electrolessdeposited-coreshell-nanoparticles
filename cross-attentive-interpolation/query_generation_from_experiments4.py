#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT – Intelligent Experimental Input Generator
(using scikit‑image, no system dependencies)
--------------------------------------------------------
- Automatic core diameter & scale bar detection via scikit‑image
- Optional Google Cloud Vision OCR (if API key available)
- Manual entry fallback for all values
- Derived parameters (L0, fc, c_bulk, rs) using exact formulas
- LLM query generation with GPT‑2 / Qwen (optional)
"""

import streamlit as st
import os
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

# -------------------- Optional dependency checks (graceful) --------------------
try:
    from skimage import filters, feature, transform, measure, color, morphology
    from skimage.transform import probabilistic_hough_line, hough_circle, hough_circle_peaks
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

# Google Cloud Vision (optional, requires API key)
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

# -------------------- Image analysis functions (scikit‑image) --------------------
if SKIMAGE_AVAILABLE:
    def detect_scale_bar_skimage(pil_img):
        """Detect horizontal scale bar using probabilistic Hough lines."""
        img = np.array(pil_img.convert('L'))  # grayscale
        edges = filters.canny(img, sigma=2)
        lines = probabilistic_hough_line(edges, threshold=10, line_length=20, line_gap=3)
        
        horizontal_lines = []
        for line in lines:
            (x1, y1), (x2, y2) = line
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 and 20 < length < 200:  # nearly horizontal, reasonable length
                horizontal_lines.append((length, line))
        
        if not horizontal_lines:
            return None, None, 0.0
        
        # Choose the longest line as scale bar
        longest = max(horizontal_lines, key=lambda x: x[0])
        length_px = longest[0]
        confidence = 0.7  # heuristic
        return length_px, None, confidence  # return length, no label yet (need OCR)

    def detect_core_diameter_skimage(pil_img, scale_nm_per_px=None):
        """Detect circular core using Hough circle transform."""
        img = np.array(pil_img.convert('L'))
        # Try a range of radii
        hough_radii = np.arange(20, 150, 5)
        hough_res = hough_circle(img, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=3)
        
        if len(radii) > 0:
            # Use the most prominent circle (highest accumulator)
            best_idx = np.argmax(accums)
            radius_px = radii[best_idx]
            diameter_px = 2 * radius_px
            confidence = accums[best_idx] / accums.max() if accums.max() > 0 else 0.5
            
            if scale_nm_per_px is not None:
                diameter_nm = diameter_px * scale_nm_per_px
            else:
                diameter_nm = None
            
            debug_info = {
                'method': 'hough',
                'circles_found': len(radii),
                'center': (cx[best_idx], cy[best_idx]),
                'radius_px': radius_px
            }
            return diameter_nm, diameter_px, confidence, debug_info
        
        # Fallback: contour analysis
        edges = filters.canny(img, sigma=2)
        contours = measure.find_contours(edges, 0.8)
        
        best_circularity = 0
        best_contour = None
        for contour in contours:
            if len(contour) < 5:
                continue
            area = measure.grid_points_in_poly(contour, (img.shape))
            if area == 0:
                continue
            perimeter = measure.perimeter(contour)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if circularity > best_circularity and circularity > 0.6:
                best_circularity = circularity
                best_contour = contour
        
        if best_contour is not None:
            # Fit minimal enclosing circle (using cv2 or custom? we can approximate)
            # For simplicity, use bounding circle from extreme points
            coords = best_contour
            center = np.mean(coords, axis=0)
            radius_px = np.max(np.linalg.norm(coords - center, axis=1))
            diameter_px = 2 * radius_px
            confidence = 0.5 * best_circularity
            debug_info = {'method': 'contour', 'circularity': best_circularity}
            if scale_nm_per_px:
                diameter_nm = diameter_px * scale_nm_per_px
            else:
                diameter_nm = None
            return diameter_nm, diameter_px, confidence, debug_info
        
        return None, None, 0.0, {}

    def extract_composition_skimage(pil_img):
        """Simple color-based analysis for Cu (red) and Ag (green)."""
        img_rgb = np.array(pil_img.convert('RGB'))
        hsv = color.rgb2hsv(img_rgb)
        
        # Red mask (hue near 0 or 1)
        red_mask1 = (hsv[..., 0] < 0.05) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3)
        red_mask2 = (hsv[..., 0] > 0.95) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3)
        red_mask = red_mask1 | red_mask2
        
        # Green mask (hue ~0.33)
        green_mask = (np.abs(hsv[..., 0] - 0.33) < 0.1) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3)
        
        cu_pixels = np.sum(red_mask)
        ag_pixels = np.sum(green_mask)
        
        if cu_pixels > 100 and ag_pixels > 100:
            ratio = cu_pixels / ag_pixels
            c_bulk = 1.0 / ratio if ratio > 0 else 1.0
            c_bulk = np.clip(c_bulk, 0.1, 1.0)
            ratio_str = f"{round(ratio,1)}:1"
            return round(c_bulk, 3), ratio_str, 0.6, "color_analysis"
        return None, None, 0.0, "failed"

# -------------------- Google Cloud Vision OCR (optional) --------------------
def google_vision_ocr(pil_img):
    """Use Google Vision API to read text labels (requires credentials)."""
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

# -------------------- Helper functions (always available) --------------------
def list_images_in_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    files = []
    for f in folder.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            if f.suffix.lower() in valid_extensions:
                files.append(str(f))
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

def generate_query(llm_tokenizer, llm_model, params, backend):
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return "LLM not available."
    prompt = f"""Based on experimental images:
- Core diameter = {params['core_diameter']} nm
- Cu:Ag ratio = {params['ratio_str']} → c_bulk = {params['c_bulk']}
- fc = {params['fc']:.3f}, L0 = {params['L0']:.1f} nm, rs = {params['rs']:.2f}

Generate a natural language query starting with "Design a core-shell with". Output ONLY the sentence."""

    if "Qwen" in backend:
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
        full_prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = prompt

    inputs = llm_tokenizer.encode(full_prompt, return_tensors='pt', truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = llm_model.generate(inputs, max_new_tokens=80, temperature=0.3, do_sample=True,
                                     pad_token_id=llm_tokenizer.eos_token_id)
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full_prompt in answer:
        answer = answer.replace(full_prompt, "").strip()
    return answer

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="CoreShellGPT – Intelligent Input Generator", layout="wide")
st.title("🧪 CoreShellGPT – Intelligent Experimental Input Generator")
st.markdown("**Automatic detection using scikit‑image (pure Python).**")

with st.expander("🔧 System Info"):
    st.write(f"**scikit‑image:** {'✅' if SKIMAGE_AVAILABLE else '❌'} (auto-detection will be hidden if missing)")
    st.write(f"**Google Cloud Vision:** {'✅' if GOOGLE_VISION_AVAILABLE else '❌'} (OCR if API key set)")
    st.write(f"**Transformers:** {'✅' if TRANSFORMERS_AVAILABLE else '❌'} (LLM disabled if missing)")

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
            if st.button("🔍 Auto-detect Scale Bar & Core (skimage)", key="auto_geo"):
                with st.spinner("Analyzing image..."):
                    # Step 1: detect scale bar
                    scale_px, _, scale_conf = detect_scale_bar_skimage(geo_img)

                    if scale_px:
                        st.success(f"✅ Scale bar detected: {scale_px:.1f} pixels")

                        # Ask user for scale value (OCR would come here)
                        scale_nm = st.number_input("Enter scale bar value (nm)", value=20.0, step=5.0, key="scale_nm_auto")
                        if scale_nm > 0:
                            scale_nm_per_px = scale_nm / scale_px
                            st.info(f"📏 Scale: {scale_nm_per_px:.4f} nm/pixel")
                            st.session_state['scale_nm_per_px'] = scale_nm_per_px

                            # Step 2: detect core
                            diam_nm, diam_px, diam_conf, debug = detect_core_diameter_skimage(geo_img, scale_nm_per_px)
                            if diam_nm:
                                st.success(f"✅ Core diameter: {diam_nm:.2f} nm ({diam_px:.1f} px)")
                                st.info(f"🎯 Detection confidence: {diam_conf:.1%}")
                                if debug.get('circles_found', 0) > 0:
                                    st.caption(f"Found {debug['circles_found']} circles using {debug['method']}")

                                if st.button("Use this diameter", key="accept_diam"):
                                    st.session_state['core_diameter'] = diam_nm
                                    st.rerun()
                            else:
                                st.warning("⚠️ Could not detect core automatically. Try manual entry below.")
                    else:
                        st.error("❌ Could not detect scale bar. Please enter manually.")
        else:
            st.info("Automatic detection disabled (scikit‑image not installed). Please enter manually.")

        st.markdown("---")
        st.markdown("**Manual Entry**")
        scale_nm_manual = st.number_input("Scale bar value (nm)", value=20.0, step=5.0, key="scale_manual")
        scale_px_manual = st.number_input("Scale bar length (pixels)", value=100, step=10, key="scalepx_manual")

        if st.button("Set scale manually", key="set_scale_manual"):
            if scale_px_manual > 0:
                st.session_state['scale_nm_per_px'] = scale_nm_manual / scale_px_manual
                st.success(f"Scale set: {scale_nm_manual/scale_px_manual:.4f} nm/pixel")

        if 'scale_nm_per_px' in st.session_state or True:
            diam_manual = st.number_input("Core diameter (nm)", min_value=0.0, value=20.0, step=0.1, key="diam_manual")
            if st.button("Set core diameter", key="set_diam_manual"):
                st.session_state['core_diameter'] = diam_manual
                st.success(f"Core diameter set to {diam_manual} nm")

# -------------------- COMPOSITION IMAGE --------------------
with col2:
    st.subheader("📁 Elemental Mapping / Composition Image")
    comp_img, comp_source, comp_path = image_selector(COMPOSITION_FOLDER, "Composition Image", "comp")

    if comp_img:
        st.image(comp_img, caption=comp_source, use_container_width=True)
        st.session_state['comp_image'] = comp_img

        if SKIMAGE_AVAILABLE:
            if st.button("🔍 Auto-extract Composition (color analysis)", key="auto_comp"):
                with st.spinner("Analyzing composition..."):
                    c_bulk, ratio_str, conf, method = extract_composition_skimage(comp_img)
                    if c_bulk is not None:
                        st.success(f"✅ Extracted: Cu:Ag ≈ {ratio_str}")
                        st.info(f"c_bulk = {c_bulk} (confidence: {conf:.1%}, method: {method})")

                        if st.button("Use this ratio", key="accept_comp"):
                            st.session_state['c_bulk'] = c_bulk
                            st.session_state['ratio_str'] = ratio_str
                            st.rerun()
                    else:
                        st.warning("⚠️ Could not extract automatically. Try manual entry or OCR.")
        else:
            st.info("Automatic color analysis disabled (scikit‑image not installed).")

        # Optional Google Vision OCR
        if GOOGLE_VISION_AVAILABLE:
            if st.button("📝 Read labels with Google Vision", key="ocr_comp"):
                with st.spinner("Running OCR..."):
                    text = google_vision_ocr(comp_img)
                    if text:
                        st.text_area("OCR result", text, height=100)
                        # Try to parse ratio
                        match = re.search(r'cu\s*:\s*ag\s*=\s*(\d+)\s*:\s*(\d+)', text, re.IGNORECASE)
                        if match:
                            cu = int(match.group(1))
                            ag = int(match.group(2))
                            if ag > 0:
                                ratio = cu / ag
                                c_bulk = 1.0 / ratio
                                c_bulk = np.clip(c_bulk, 0.1, 1.0)
                                st.success(f"Found Cu:Ag = {cu}:{ag} → c_bulk = {c_bulk}")
                                if st.button("Use this ratio (OCR)", key="accept_ocr"):
                                    st.session_state['c_bulk'] = round(c_bulk, 3)
                                    st.session_state['ratio_str'] = f"{cu}:{ag}"
                                    st.rerun()
                    else:
                        st.warning("No text detected.")

        st.markdown("---")
        st.markdown("**Manual Entry**")
        cu_num = st.number_input("Cu count", min_value=1, value=1, step=1, key="cu")
        ag_num = st.number_input("Ag count", min_value=1, value=1, step=1, key="ag")

        if st.button("Set ratio manually", key="set_comp_manual"):
            ratio = cu_num / ag_num
            c_bulk = 1.0 / ratio if ratio > 0 else 1.0
            c_bulk = np.clip(c_bulk, 0.1, 1.0)
            st.session_state['c_bulk'] = round(c_bulk, 3)
            st.session_state['ratio_str'] = f"{cu_num}:{ag_num}"
            st.success(f"Set: c_bulk = {c_bulk}")

# -------------------- CALCULATIONS --------------------
st.header("📐 Derived Parameters")

if 'core_diameter' in st.session_state and 'c_bulk' in st.session_state:
    d_core = st.session_state['core_diameter']
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
    c_bulk = st.session_state['c_bulk']
    ratio_str = st.session_state.get('ratio_str', '?')

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
                query = generate_query(
                    st.session_state['llm_tokenizer'],
                    st.session_state['llm_model'],
                    params,
                    chosen_llm
                )
                st.text_area("📋 Generated Query:", query, height=100)

    default_query = f"Design a core-shell with L0={params['L0']:.1f} nm, fc={params['fc']:.3f}, c_bulk={params['c_bulk']:.2f}, rs={params['rs']:.2f}, time=1e-3 s from HRTEM (core {params['core_diameter']} nm) and EDS (Cu:Ag={params['ratio_str']})."
    st.text_area("📋 Default Query:", default_query, height=100)

else:
    st.info("👈 Set core diameter and c_bulk from the images above")
