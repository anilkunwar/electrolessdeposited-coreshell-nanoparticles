#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT – Intelligent Experimental Input Generator
--------------------------------------------------------
- AUTOMATIC: Detects scale bars, reads labels with OCR, measures core diameter
- Uses OpenCV + pytesseract for computer vision
- Optional Vision LLM for complex interpretation
- Falls back to manual entry if automatic fails
"""

import streamlit as st
import os
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter

# -------------------- Dependency checks --------------------
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.error("OpenCV not installed. Run: pip install opencv-python")
    st.stop()

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    # Point to tesseract executable if needed (Windows)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    TESSERACT_AVAILABLE = False
    st.warning("pytesseract not installed. Run: pip install pytesseract")

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    st.error("Pillow required. Run: pip install Pillow")
    st.stop()

# -------------------- Path configuration --------------------
BASE_DIR = Path(__file__).parent.resolve()
GEOMETRY_FOLDER = BASE_DIR / "experimental_images" / "geometry"
COMPOSITION_FOLDER = BASE_DIR / "experimental_images" / "composition_ratio"
GEOMETRY_FOLDER.mkdir(parents=True, exist_ok=True)
COMPOSITION_FOLDER.mkdir(parents=True, exist_ok=True)

# -------------------- IMAGE ANALYSIS FUNCTIONS --------------------

def preprocess_image_for_ocr(pil_img):
    """Enhance image for better OCR results."""
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return gray, thresh, denoised

def detect_scale_bar(pil_img):
    """
    Detect scale bar in HRTEM image.
    Returns: (scale_bar_length_px, label_text, confidence)
    """
    if not CV2_AVAILABLE:
        return None, None, 0.0
    
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Detect horizontal lines (scale bars are usually horizontal)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )
    
    # Find horizontal lines (likely scale bars)
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = abs(x2 - x1)
            line_angle = abs(y2 - y1)
            
            # Horizontal line: small vertical difference, reasonable length
            if line_angle < 5 and 20 < line_length < 200:
                horizontal_lines.append((x1, y1, x2, y2, line_length))
    
    # Sort by length (scale bar is usually one of the longest horizontal lines)
    horizontal_lines.sort(key=lambda x: x[4], reverse=True)
    
    # Step 2: OCR near detected lines to find "nm" label
    scale_info = None
    best_confidence = 0.0
    
    for i, (x1, y1, x2, y2, length_px) in enumerate(horizontal_lines[:3]):  # Check top 3 candidates
        # Define ROI around the line (usually below or above)
        roi_y1 = max(0, y2 - 30)
        roi_y2 = min(img_cv.shape[0], y2 + 50)
        roi_x1 = max(0, x1 - 20)
        roi_x2 = min(img_cv.shape[1], x2 + 20)
        
        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # OCR on ROI
        if TESSERACT_AVAILABLE:
            try:
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.nm '
                text = pytesseract.image_to_string(roi, config=custom_config).strip()
                
                # Look for pattern like "20 nm", "50nm", "100 nm"
                match = re.search(r'(\d+(?:\.\d+)?)\s*(nm|nanometers?)', text, re.IGNORECASE)
                if match:
                    scale_value = float(match.group(1))
                    unit = match.group(2).lower()
                    confidence = 0.8  # Base confidence
                    scale_info = (length_px, f"{scale_value} {unit}", confidence)
                    best_confidence = confidence
                    break
            except:
                pass
    
    # Step 3: If no OCR success, try broader search for "nm" in entire image
    if scale_info is None and TESSERACT_AVAILABLE:
        try:
            # Search entire image for scale bar text
            full_text = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Look for "nm" and nearby numbers
            for i, text in enumerate(full_text['text']):
                if 'nm' in text.lower() and text.strip():
                    # Get bounding box
                    x, y, w, h = (full_text['left'][i], full_text['top'][i], 
                                  full_text['width'][i], full_text['height'][i])
                    
                    # Look for number nearby
                    for j, num_text in enumerate(full_text['text']):
                        if re.match(r'^\d+(?:\.\d+)?$', num_text.strip()):
                            # Check if spatially close
                            dist = np.sqrt((x - full_text['left'][j])**2 + (y - full_text['top'][j])**2)
                            if dist < 100:  # Within 100 pixels
                                scale_value = float(num_text)
                                # Estimate scale bar length from nearby horizontal line
                                if horizontal_lines:
                                    length_px = horizontal_lines[0][4]
                                    scale_info = (length_px, f"{scale_value} nm", 0.6)
                                    break
        except:
            pass
    
    return scale_info if scale_info else (None, None, 0.0)

def detect_core_diameter(pil_img, scale_nm_per_px):
    """
    Detect core diameter using circle detection.
    Returns: (diameter_nm, diameter_px, confidence, debug_info)
    """
    if not CV2_AVAILABLE:
        return None, None, 0.0, {}
    
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=50,
        minRadius=20,
        maxRadius=200
    )
    
    debug_info = {'circles_found': 0, 'method': 'hough'}
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        debug_info['circles_found'] = len(circles)
        
        # Get the most prominent circle (largest or first)
        # Sort by radius (prefer larger circles for core)
        circles = sorted(circles, key=lambda k: k[2], reverse=True)
        
        x, y, radius_px = circles[0]
        diameter_px = 2 * radius_px
        diameter_nm = diameter_px * scale_nm_per_px
        
        # Confidence based on circle detection strength
        confidence = 0.7 if len(circles) > 0 else 0.3
        
        debug_info['center'] = (x, y)
        debug_info['radius_px'] = radius_px
        
        return diameter_nm, diameter_px, confidence, debug_info
    
    # Fallback: Edge detection + contour analysis
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest circular contour
        best_contour = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > best_circularity and circularity > 0.6:
                best_circularity = circularity
                best_contour = contour
        
        if best_contour is not None:
            (x, y), radius_px = cv2.minEnclosingCircle(best_contour)
            diameter_px = 2 * radius_px
            diameter_nm = diameter_px * scale_nm_per_px
            confidence = 0.5 * best_circularity
            
            debug_info['method'] = 'contour'
            debug_info['circularity'] = best_circularity
            
            return diameter_nm, diameter_px, confidence, debug_info
    
    return None, None, 0.0, debug_info

def extract_composition_from_image(pil_img):
    """
    Extract Cu:Ag ratio or concentration from composition map.
    Returns: (c_bulk, ratio_str, confidence, method)
    """
    if not TESSERACT_AVAILABLE:
        return None, None, 0.0, "no_ocr"
    
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Method 1: OCR for text patterns like "Cu:Ag = 5:1" or "c_bulk = 0.2"
    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config)
        
        # Pattern 1: Cu:Ag = X:Y
        match1 = re.search(r'cu\s*:\s*ag\s*=\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match1:
            cu_val = float(match1.group(1))
            ag_val = float(match1.group(2))
            if ag_val > 0:
                ratio = cu_val / ag_val
                c_bulk = 1.0 / ratio if ratio > 0 else 1.0
                c_bulk = np.clip(c_bulk, 0.1, 1.0)
                return round(c_bulk, 3), f"{int(cu_val)}:{int(ag_val)}", 0.9, "ocr_ratio"
        
        # Pattern 2: c_bulk = X or c = X
        match2 = re.search(r'c[_\s]*(?:bulk)?\s*=\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match2:
            c_bulk = float(match2.group(1))
            c_bulk = np.clip(c_bulk, 0.1, 1.0)
            return round(c_bulk, 3), f"1:{round(1/c_bulk,1)}", 0.8, "ocr_cbulk"
        
        # Pattern 3: Percentage like "Cu 80%" or "Ag 20%"
        cu_match = re.search(r'cu.*?(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
        ag_match = re.search(r'ag.*?(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
        
        if cu_match and ag_match:
            cu_pct = float(cu_match.group(1))
            ag_pct = float(ag_match.group(1))
            if cu_pct + ag_pct > 0:
                ratio = cu_pct / ag_pct if ag_pct > 0 else 10
                c_bulk = 1.0 / ratio if ratio > 0 else 1.0
                c_bulk = np.clip(c_bulk, 0.1, 1.0)
                return round(c_bulk, 3), f"{round(cu_pct)}:{round(ag_pct)}", 0.7, "ocr_percentage"
    
    except Exception as e:
        st.warning(f"OCR extraction failed: {e}")
    
    # Method 2: Color-based analysis (if EDS map with Cu=red, Ag=green)
    try:
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Red mask (Cu)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_cu = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Green mask (Ag)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask_ag = cv2.inRange(hsv, lower_green, upper_green)
        
        # Count pixels
        cu_pixels = np.sum(mask_cu > 0)
        ag_pixels = np.sum(mask_ag > 0)
        
        if cu_pixels > 100 and ag_pixels > 100:  # Minimum threshold
            ratio = cu_pixels / ag_pixels
            c_bulk = 1.0 / ratio if ratio > 0 else 1.0
            c_bulk = np.clip(c_bulk, 0.1, 1.0)
            ratio_str = f"{round(ratio,1)}:1"
            return round(c_bulk, 3), ratio_str, 0.6, "color_analysis"
    
    except Exception as e:
        st.warning(f"Color analysis failed: {e}")
    
    return None, None, 0.0, "failed"

# -------------------- VISION LLM APPROACH (Optional) --------------------

def analyze_with_vision_llm(pil_img, task_description):
    """
    Use a vision-capable LLM to analyze the image.
    This is a placeholder - requires API key for GPT-4V, Claude, etc.
    """
    # Example using OpenAI GPT-4V (requires: pip install openai)
    try:
        from openai import OpenAI
        import base64
        from io import BytesIO
        
        # Convert PIL to base64
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        client = OpenAI(api_key="YOUR_API_KEY")  # Replace with actual key or env var
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_description},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    except ImportError:
        return "OpenAI not installed. Run: pip install openai"
    except Exception as e:
        return f"Vision LLM error: {e}"

# -------------------- Helper functions (from your original code) --------------------

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
st.markdown("**Automatic scale bar detection + OCR + core measurement**")

# Debug info
with st.expander("🔧 System Info"):
    st.write(f"**OpenCV:** {'✅' if CV2_AVAILABLE else '❌'}")
    st.write(f"**Tesseract OCR:** {'✅' if TESSERACT_AVAILABLE else '❌'}")
    st.write(f"**Transformers:** {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")

# Sidebar
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

# Main columns
col1, col2 = st.columns(2)

# -------------------- GEOMETRY IMAGE ANALYSIS --------------------
with col1:
    st.subheader("📁 HRTEM / Geometry Image")
    
    geo_img, geo_source, geo_path = image_selector(GEOMETRY_FOLDER, "Geometry Image", "geo")
    
    if geo_img:
        st.image(geo_img, caption=geo_source, use_container_width=True)
        st.session_state['geo_image'] = geo_img
        
        # Automatic analysis button
        if st.button("🔍 Auto-detect Scale Bar & Core", key="auto_geo"):
            with st.spinner("Analyzing image..."):
                # Step 1: Detect scale bar
                scale_px, scale_label, scale_conf = detect_scale_bar(geo_img)
                
                if scale_px and scale_label:
                    # Extract number from label
                    match = re.search(r'(\d+(?:\.\d+)?)', scale_label)
                    if match:
                        scale_nm = float(match.group(1))
                        scale_nm_per_px = scale_nm / scale_px
                        
                        st.success(f"✅ Scale bar detected: {scale_label}")
                        st.info(f"📏 Scale: {scale_nm_per_px:.4f} nm/pixel (bar = {scale_px}px)")
                        
                        # Store scale
                        st.session_state['scale_nm_per_px'] = scale_nm_per_px
                        st.session_state['scale_label'] = scale_label
                        
                        # Step 2: Detect core diameter
                        diam_nm, diam_px, diam_conf, debug = detect_core_diameter(geo_img, scale_nm_per_px)
                        
                        if diam_nm:
                            st.success(f"✅ Core diameter: {diam_nm:.2f} nm ({diam_px:.1f} px)")
                            st.info(f"🎯 Detection confidence: {diam_conf:.1%}")
                            
                            if debug.get('circles_found', 0) > 0:
                                st.caption(f"Found {debug['circles_found']} circles using {debug['method']}")
                            
                            # Auto-save
                            if st.button("Use this diameter", key="accept_diam"):
                                st.session_state['core_diameter'] = diam_nm
                                st.rerun()
                        else:
                            st.warning("⚠️ Could not detect core automatically. Try manual entry below.")
                    else:
                        st.error(f"❌ Could not parse scale value from: {scale_label}")
                else:
                    st.error("❌ Could not detect scale bar. Please enter manually below.")
                    st.info("💡 Tip: Make sure your image has a visible scale bar with 'nm' label")
        
        # Manual override
        st.markdown("---")
        st.markdown("**Manual Entry (if auto-detection fails)**")
        
        # Scale bar manual
        scale_nm_manual = st.number_input("Scale bar value (nm)", value=20.0, step=5.0, key="scale_manual")
        scale_px_manual = st.number_input("Scale bar length (pixels)", value=100, step=10, key="scalepx_manual")
        
        if st.button("Set scale manually", key="set_scale_manual"):
            if scale_px_manual > 0:
                st.session_state['scale_nm_per_px'] = scale_nm_manual / scale_px_manual
                st.success(f"Scale set: {scale_nm_manual/scale_px_manual:.4f} nm/pixel")
        
        # Core diameter manual
        if 'scale_nm_per_px' in st.session_state:
            diam_manual = st.number_input("Core diameter (nm)", min_value=0.0, value=20.0, step=0.1, key="diam_manual")
            if st.button("Set core diameter", key="set_diam_manual"):
                st.session_state['core_diameter'] = diam_manual
                st.success(f"Core diameter set to {diam_manual} nm")

# -------------------- COMPOSITION IMAGE ANALYSIS --------------------
with col2:
    st.subheader("📁 Elemental Mapping / Composition Image")
    
    comp_img, comp_source, comp_path = image_selector(COMPOSITION_FOLDER, "Composition Image", "comp")
    
    if comp_img:
        st.image(comp_img, caption=comp_source, use_container_width=True)
        st.session_state['comp_image'] = comp_img
        
        # Automatic analysis
        if st.button("🔍 Auto-extract Composition", key="auto_comp"):
            with st.spinner("Analyzing composition..."):
                c_bulk, ratio_str, conf, method = extract_composition_from_image(comp_img)
                
                if c_bulk is not None:
                    st.success(f"✅ Extracted: Cu:Ag = {ratio_str}")
                    st.info(f"c_bulk = {c_bulk} (confidence: {conf:.1%}, method: {method})")
                    
                    if st.button("Use this ratio", key="accept_comp"):
                        st.session_state['c_bulk'] = c_bulk
                        st.session_state['ratio_str'] = ratio_str
                        st.rerun()
                else:
                    st.warning("⚠️ Could not extract automatically. Enter manually below.")
        
        # Manual entry
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
    
    # Display parameters
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
    
    # Generate query
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
    
    # Default query
    default_query = f"Design a core-shell with L0={params['L0']:.1f} nm, fc={params['fc']:.3f}, c_bulk={params['c_bulk']:.2f}, rs={params['rs']:.2f}, time=1e-3 s from HRTEM (core {params['core_diameter']} nm) and EDS (Cu:Ag={params['ratio_str']})."
    st.text_area("📋 Default Query:", default_query, height=100)

else:
    st.info("👈 Set core diameter and c_bulk from the images above")
