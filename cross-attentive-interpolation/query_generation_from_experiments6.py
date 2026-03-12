#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT – Intelligent Experimental Input Generator
(Enhanced with semantic reasoning, colour‑aware detection, and LLM verification)
"""

import streamlit as st
import os
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io

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
        "temp_geo_detection": None,       # store last detection results
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

# -------------------- Path configuration --------------------
BASE_DIR = Path(__file__).parent.resolve()
GEOMETRY_FOLDER = BASE_DIR / "experimental_images" / "geometry"
COMPOSITION_FOLDER = BASE_DIR / "experimental_images" / "composition_ratio"
GEOMETRY_FOLDER.mkdir(parents=True, exist_ok=True)
COMPOSITION_FOLDER.mkdir(parents=True, exist_ok=True)

# -------------------- NEW: Image type classifier --------------------
def classify_image_type(pil_img):
    """Return 'composition' for EDS maps, 'geometry' for HRTEM."""
    img_rgb = np.array(pil_img.convert('RGB'))
    hsv = color.rgb2hsv(img_rgb)
    red_score = np.sum(((hsv[..., 0] < 0.05) | (hsv[..., 0] > 0.95)) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3))
    green_score = np.sum((np.abs(hsv[..., 0] - 0.33) < 0.1) & (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.3))
    if red_score > 8000 and green_score > 8000:
        return "composition"
    return "geometry"

# -------------------- NEW: Auto scale value from OCR + LLM --------------------
def auto_extract_scale_value(pil_img, line_coords):
    """Extract numeric scale value (nm) from the region below the detected scale bar."""
    if not line_coords:
        return 20.0  # safe default
    (x1, y1), (x2, y2) = line_coords
    # Crop region below the bar (typical label position)
    left = min(x1, x2) - 20
    top = max(y1, y2) + 5
    right = max(x1, x2) + 20
    bottom = max(y1, y2) + 60
    crop = pil_img.crop((left, top, right, bottom))
    text = ""
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
    if text:
        match = re.search(r'(\d+\.?\d*)\s*nm', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    # LLM fallback if available
    if st.session_state.get('llm_tokenizer') and st.session_state.get('llm_model'):
        prompt = f"Extract only the number before 'nm' from this OCR text: {text}. Output ONLY the number."
        tokenizer = st.session_state.llm_tokenizer
        model = st.session_state.llm_model
        inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=128)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=10, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            return float(answer.strip())
        except:
            pass
    return 20.0

# -------------------- NEW: Core extraction from red mask (EDS) --------------------
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
        'contour': None  # we could compute contour if needed
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
        # thick red line over the bar
        draw.line([(x1, y1), (x2, y2)], fill="#FF0000", width=6)
        # bounding box around it
        draw.rectangle([(min(x1,x2)-10, min(y1,y2)-15),
                        (max(x1,x2)+10, max(y1,y2)+15)],
                       outline="#FF0000", width=3)
        draw.text((min(x1,x2), min(y1,y2)-25),
                  "SCALE BAR", fill="#FF0000", font=font)

    # Highlight core and add arrow & legend
    if core_info:
        cx, cy, r = core_info["center"][0], core_info["center"][1], core_info["radius_px"]
        # Draw outer circle (core)
        draw.ellipse((cx-r-3, cy-r-3, cx+r+3, cy+r+3),
                     outline="#00FF00", width=5)
        # Draw dashed orange shell (outer boundary if available)
        shell_r = r * 1.3  # estimate shell outer radius (for illustration)
        dash_len = 10
        for angle in range(0, 360, 15):
            x = cx + shell_r * np.cos(np.radians(angle))
            y = cy + shell_r * np.sin(np.radians(angle))
            x2 = cx + shell_r * np.cos(np.radians(angle+dash_len))
            y2 = cy + shell_r * np.sin(np.radians(angle+dash_len))
            draw.line([(x, y), (x2, y2)], fill="#FFA500", width=3)

        # Label with diameter
        diam_nm = core_info.get("diam_nm", 0)
        label = f"{diam_nm:.1f} nm" if diam_nm else "core"
        draw.text((cx-30, cy-15), label,
                  fill="white", font=font, stroke_width=2, stroke_fill="black")

        # Arrow pointing to core
        draw.line([(cx, cy-30), (cx+50, cy-60)], fill="white", width=3)
        draw.text((cx+60, cy-70), "Cu CORE", fill="white", font=font, stroke_width=3)

        # Legend box
        legend_x, legend_y = 10, 10
        draw.rectangle([(legend_x, legend_y), (legend_x+200, legend_y+60)], outline="white", width=2)
        draw.rectangle([(legend_x+5, legend_y+5), (legend_x+25, legend_y+25)], fill="#00FF00")
        draw.text((legend_x+35, legend_y+5), "Cu core", fill="white", font=small_font)
        draw.rectangle([(legend_x+5, legend_y+30), (legend_x+25, legend_y+50)], fill="#FFA500")
        draw.text((legend_x+35, legend_y+30), "Ag shell", fill="white", font=small_font)

        # If we have contour pixels (fallback), highlight them in yellow
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
    # Semi‑transparent background for readability
    bbox = draw.textbbox((10, annotated.height-50), text, font=font)
    draw.rectangle(bbox, fill=(0,0,0,180))
    draw.text((10, annotated.height-50), text, fill="#00FF00", font=font)
    return annotated

# -------------------- Image analysis functions (scikit‑image) --------------------
if SKIMAGE_AVAILABLE:
    def detect_scale_bar_skimage(pil_img):
        """Detect horizontal scale bar. Returns (length_px, line_coords, confidence, annotated_img)."""
        img = np.array(pil_img.convert('L'))
        edges = feature.canny(img, sigma=2)
        lines = probabilistic_hough_line(edges, threshold=10, line_length=20, line_gap=3)

        horizontal_lines = []
        for line in lines:
            (x1, y1), (x2, y2) = line
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 and 20 < length < 200:
                horizontal_lines.append((length, line))

        if not horizontal_lines:
            return None, None, 0.0, pil_img

        longest = max(horizontal_lines, key=lambda x: x[0])
        length_px, line_coords = longest
        confidence = 0.7
        # Create annotated image (just scale bar highlighted)
        annotated = annotate_geometry(pil_img, scale_line=line_coords, core_info=None)
        return length_px, line_coords, confidence, annotated

    # UPDATED: detect_core_diameter_skimage with colour‑aware routing and improvements
    def detect_core_diameter_skimage(pil_img, scale_nm_per_px=None):
        """Detect circular core, now with image type classification and mask‑based extraction for EDS."""
        # Step 1: classify image
        img_type = classify_image_type(pil_img)

        if img_type == "composition":
            # Use colour mask method
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

        # Geometry image – use improved Hough on red channel + CLAHE
        img_rgb = np.array(pil_img.convert('RGB'))
        red_channel = img_rgb[..., 0]  # red only (often better for core)
        # CLAHE for contrast enhancement
        red_enhanced = exposure.equalize_adapthist(red_channel)

        hough_radii = np.arange(20, 150, 5)
        hough_res = hough_circle(red_enhanced, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=3)

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
                'center': (int(center[1]), int(center[0])),  # (x,y) order for drawing
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
            # Build annotated image
            annotated = annotate_composition(pil_img, ratio_str, c_bulk)
            return c_bulk, ratio_str, confidence, annotated
        return None, None, 0.0, pil_img

# -------------------- Helper functions (unchanged) --------------------
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
st.markdown("**Automatic detection with colour awareness + LLM verification**")

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

        # Auto‑detection button (stores results in session_state.temp_geo_detection)
        if SKIMAGE_AVAILABLE:
            if st.button("🔍 Auto-detect Scale Bar & Core (skimage)", key="auto_geo"):
                with st.spinner("Analyzing image..."):
                    # Classify image type (just for info)
                    img_type = classify_image_type(geo_img)
                    if img_type == "composition":
                        st.warning("✅ Detected EDS image! Using colour‑aware core detection.")

                    scale_px, line_coords, scale_conf, annotated_scale = detect_scale_bar_skimage(geo_img)
                    if scale_px:
                        # Auto‑extract scale value from image
                        scale_nm = auto_extract_scale_value(geo_img, line_coords)
                        scale_nm_per_px = scale_nm / scale_px
                        # Detect core
                        diam_nm, diam_px, diam_conf, debug, annotated_core = detect_core_diameter_skimage(
                            geo_img, scale_nm_per_px
                        )
                        # Optional LLM verification
                        if diam_nm and st.session_state.get('llm_model'):
                            verify_prompt = f"Is a core diameter of {diam_nm:.1f} nm realistic for a Cu@Ag nanoparticle with a {scale_nm} nm scale bar? Answer only YES or NO, and if NO suggest a more realistic value based on the scale."
                            # Generate verification (simple)
                            tokenizer = st.session_state.llm_tokenizer
                            model = st.session_state.llm_model
                            inputs = tokenizer.encode(verify_prompt, return_tensors='pt', truncation=True, max_length=128)
                            if torch.cuda.is_available():
                                inputs = inputs.to('cuda')
                            with torch.no_grad():
                                outputs = model.generate(inputs, max_new_tokens=30, temperature=0.0)
                            verdict = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            if "NO" in verdict.upper():
                                st.warning(f"LLM suggests the detected diameter may be unrealistic: {verdict}")
                                # Optionally fall back to mask extraction if this was a geometry misclassified?
                                # For now just warn.

                        # Store everything temporarily
                        st.session_state.temp_geo_detection = {
                            'scale_nm_per_px': scale_nm_per_px,
                            'diam_nm': diam_nm,
                            'annotated_scale': annotated_scale,
                            'annotated_core': annotated_core,
                            'scale_px': scale_px,
                            'diam_conf': diam_conf,
                        }
                        st.success("Detection complete! Review the annotated images below.")
                    else:
                        st.error("Could not detect scale bar. Please use manual entry.")

            # Show temporary annotations if they exist
            if st.session_state.get('temp_geo_detection'):
                det = st.session_state.temp_geo_detection
                st.image(det['annotated_scale'], caption="Detected scale bar", use_container_width=True)
                st.image(det['annotated_core'], caption="Detected core", use_container_width=True)
                if det['diam_nm']:
                    st.metric("Detected diameter", f"{det['diam_nm']:.2f} nm")
                else:
                    st.warning("Core detection failed.")

                # Single accept button
                if st.button("✅ Accept All Geometry Detections", key="accept_geo"):
                    st.session_state.scale_nm_per_px = det['scale_nm_per_px']
                    st.session_state.core_diameter = det['diam_nm']
                    st.session_state.geo_annotated = det['annotated_core']  # final annotated image
                    # Clear temporary
                    del st.session_state.temp_geo_detection
                    st.success("Geometry accepted!")
                    st.rerun()

        # Manual entry section (always visible)
        st.markdown("---")
        st.markdown("**Manual Entry**")
        scale_nm_manual = st.number_input("Scale bar value (nm)", value=20.0, step=5.0, key="scale_manual_final")
        scale_px_manual = st.number_input("Scale bar length (pixels)", value=100, step=10, key="scalepx_manual_final")
        diam_manual = st.number_input("Core diameter (nm)", min_value=0.0, value=20.0, step=0.1, key="diam_manual_final")

        if st.button("Set Geometry Manually", key="set_geo_manual"):
            if scale_px_manual > 0:
                st.session_state.scale_nm_per_px = scale_nm_manual / scale_px_manual
            st.session_state.core_diameter = diam_manual
            # For manual, we don't have an annotated image; create a simple one
            if geo_img:
                # Just add a text label
                ann = geo_img.copy().convert("RGB")
                draw = ImageDraw.Draw(ann)
                draw.text((10,10), f"Manual: core = {diam_manual} nm", fill="white")
                st.session_state.geo_annotated = ann
            st.success("Manual geometry set!")
            st.rerun()

        # Show final annotated geometry if already accepted
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
            if st.button("🔍 Auto-extract Composition (color analysis)", key="auto_comp"):
                with st.spinner("Analyzing..."):
                    c_bulk, ratio_str, conf, annotated = extract_composition_skimage(comp_img)
                    if c_bulk is not None:
                        st.session_state.temp_comp_detection = {
                            'c_bulk': c_bulk,
                            'ratio_str': ratio_str,
                            'annotated': annotated,
                        }
                        st.success("Composition detected! See annotated image below.")
                    else:
                        st.warning("Could not extract automatically. Try manual entry or OCR.")

            # Show temporary composition result
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

        # Google Vision OCR (optional)
        if GOOGLE_VISION_AVAILABLE:
            if st.button("📝 Read labels with Google Vision", key="ocr_comp"):
                with st.spinner("Running OCR..."):
                    text = google_vision_ocr(comp_img)
                    if text:
                        st.text_area("OCR result", text, height=100)
                        match = re.search(r'cu\s*:\s*ag\s*=\s*(\d+)\s*:\s*(\d+)', text, re.IGNORECASE)
                        if match:
                            cu = int(match.group(1))
                            ag = int(match.group(2))
                            if ag > 0:
                                ratio = cu / ag
                                c_bulk = 1.0 / ratio
                                c_bulk = np.clip(c_bulk, 0.1, 1.0)
                                st.success(f"Found Cu:Ag = {cu}:{ag} → c_bulk = {c_bulk}")
                                # Optionally use this as detection
                                annotated = annotate_composition(comp_img, f"{cu}:{ag}", c_bulk)
                                st.session_state.temp_comp_detection = {
                                    'c_bulk': round(c_bulk, 3),
                                    'ratio_str': f"{cu}:{ag}",
                                    'annotated': annotated,
                                }
                                st.rerun()
                    else:
                        st.warning("No text detected.")

        # Manual entry
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
            # Create simple annotated image
            if comp_img:
                ann = comp_img.copy().convert("RGB")
                draw = ImageDraw.Draw(ann)
                draw.text((10,10), f"Manual: Cu:Ag = {cu_num}:{ag_num}", fill="white")
                st.session_state.comp_annotated = ann
            st.success("Manual composition set!")
            st.rerun()

        # Show final annotated composition
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
