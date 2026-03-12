#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT – Experimental Input Generator (Robust Path Resolution + Manual Entry)
-----------------------------------------------------------------------------------
- Automatically finds images in experimental_images/geometry and experimental_images/composition_ratio
- Anchors paths to script location, creates folders if missing
- Case‑insensitive extension matching, ignores hidden files
- Fallback file uploaders if folders are empty
- Manual entry of core diameter and Cu:Ag ratio (no OpenCV needed)
- Optional LLM query generation (transformers + torch required)
"""

import streamlit as st
import os
import glob
from pathlib import Path
from datetime import datetime

# -------------------- Optional dependencies (LLM only) --------------------
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TORCH_AVAILABLE = False
    # LLM generation will be disabled

# Pillow is required for image display
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    st.error("Pillow is required to display images. Please install it: pip install Pillow")
    st.stop()

# -------------------- Path configuration --------------------
BASE_DIR = Path(__file__).parent.resolve()
GEOMETRY_FOLDER = BASE_DIR / "experimental_images" / "geometry"
COMPOSITION_FOLDER = BASE_DIR / "experimental_images" / "composition_ratio"

# Ensure folders exist (create if missing)
GEOMETRY_FOLDER.mkdir(parents=True, exist_ok=True)
COMPOSITION_FOLDER.mkdir(parents=True, exist_ok=True)

# -------------------- Helper: robust image listing --------------------
def list_images_in_folder(folder_path):
    """
    Return sorted list of image file paths in folder (case‑insensitive, ignores hidden).
    """
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

# -------------------- Helper: folder browser with fallback uploader --------------------
def image_selector(folder_path, label, key_prefix):
    """
    Display dropdown of images from folder, plus an uploader fallback.
    Returns (PIL Image, source_description) or (None, None) if none selected.
    """
    st.markdown(f"**{label}**")
    
    # List images from folder
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
        return img, source
    else:
        st.info(f"No images found in `{folder_path}`. You can upload one below.")
        uploaded = st.file_uploader(
            f"Upload {label.lower()} image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            key=f"{key_prefix}_uploader"
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            source = f"Uploaded: {uploaded.name}"
            return img, source
        return None, None

# -------------------- LLM loading (cached) --------------------
@st.cache_resource(show_spinner="Loading selected LLM...")
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

# -------------------- Query generation (only if LLM available) --------------------
def generate_query(llm_tokenizer, llm_model, params, backend):
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return "LLM not available. Please use the default query below."
    prompt = f"""Based on experimental images:
- Core diameter = {params['core_diameter']} nm
- Cu:Ag ratio = {params['ratio_str']} → c_bulk = {params['c_bulk']}
- Geometry mode: {params['mode']}
- fc = {params['fc']:.3f}, L0 = {params['L0']:.1f} nm, rs = {params['rs']:.2f}

Generate a natural language query for a core‑shell nanoparticle designer that includes L0, fc, c_bulk, rs, and optionally time=1e-3 s. Start with "Design a core-shell with". Output ONLY the sentence."""
    
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

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Experimental Input Generator for CoreShellGPT", layout="wide")
st.title("🧪 CoreShellGPT – Experimental Input Generator (Robust Paths + Manual Entry)")
st.markdown("**All values are entered manually – no OpenCV required.**")

# Debug expander (show path info)
with st.expander("🔧 Debug Info"):
    st.write(f"**Working directory:** {os.getcwd()}")
    st.write(f"**Script location:** {BASE_DIR}")
    st.write(f"**Geometry folder:** {GEOMETRY_FOLDER} (exists: {GEOMETRY_FOLDER.exists()})")
    if GEOMETRY_FOLDER.exists():
        st.write(f"**Contents:** {[f.name for f in GEOMETRY_FOLDER.iterdir() if not f.name.startswith('.')]}")
    st.write(f"**Composition folder:** {COMPOSITION_FOLDER} (exists: {COMPOSITION_FOLDER.exists()})")
    if COMPOSITION_FOLDER.exists():
        st.write(f"**Contents:** {[f.name for f in COMPOSITION_FOLDER.iterdir() if not f.name.startswith('.')]}")

# Sidebar: LLM selection (only if dependencies exist)
st.sidebar.header("🧠 LLM Settings")
if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
    llm_options = ["GPT-2 (default, fastest startup)", 
                   "Qwen2-0.5B-Instruct (better JSON)", 
                   "Qwen2.5-0.5B-Instruct (newest, recommended)"]
    chosen_llm = st.sidebar.selectbox("🧠 LLM for query generation", llm_options)
    if st.sidebar.button("Load LLM"):
        with st.spinner("Loading model... (this may take a moment)"):
            tokenizer, model = load_llm(chosen_llm)
            st.session_state['llm_tokenizer'] = tokenizer
            st.session_state['llm_model'] = model
            st.sidebar.success(f"{chosen_llm} loaded")
else:
    st.sidebar.info("LLM generation disabled: install transformers & torch to enable.")
    st.session_state['llm_tokenizer'] = None
    st.session_state['llm_model'] = None

# Sidebar: geometry calculation mode
st.sidebar.header("Geometry conversion")
geo_mode = st.sidebar.radio("Mode", ["Provide fc", "Provide shell distance", "Provide L0"], index=0)
if geo_mode == "Provide fc":
    fc_input = st.sidebar.number_input("fc (core fraction)", min_value=0.05, max_value=0.45, value=0.18, step=0.01)
elif geo_mode == "Provide shell distance":
    d_shell = st.sidebar.number_input("Shell distance (nm, from core surface to boundary)", min_value=0.0, value=10.0, step=1.0)
else:  # Provide L0
    L0_input = st.sidebar.number_input("L0 (nm)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)

rs_default = st.sidebar.number_input("Default rs (shell fraction)", min_value=0.01, max_value=0.6, value=0.1, step=0.01)

# Main area: two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 HRTEM / Geometry Image")
    st.markdown("**Red core, green shell, with scale bar (reference only).**")
    
    # Image selector (folder + upload fallback)
    geo_img, geo_source = image_selector(GEOMETRY_FOLDER, "Geometry Image", "geo")
    if geo_img:
        st.image(geo_img, caption=geo_source, use_container_width=True)
        st.session_state['geo_image'] = geo_img  # store for reference, not used in auto extraction
    else:
        st.info("No geometry image selected. You can still enter core diameter manually.")

    # Manual diameter entry (no automatic extraction)
    diam_manual = st.number_input("Enter core diameter (nm)", min_value=0.0, value=20.0, step=0.1)
    if st.button("Set core diameter", key="set_geo"):
        st.session_state['core_diameter'] = diam_manual
        st.success(f"Core diameter set to {diam_manual} nm")

with col2:
    st.subheader("📁 Elemental Mapping / Composition Image")
    st.markdown("**Red = Cu, Green = Ag (or explicit label like Cu:Ag=5:1) – reference only.**")
    
    # Image selector (folder + upload fallback)
    comp_img, comp_source = image_selector(COMPOSITION_FOLDER, "Composition Image", "comp")
    if comp_img:
        st.image(comp_img, caption=comp_source, use_container_width=True)
        st.session_state['comp_image'] = comp_img  # store for reference
    else:
        st.info("No composition image selected. You can still enter Cu:Ag ratio manually.")

    # Manual ratio entry (no automatic extraction)
    st.markdown("**Enter Cu:Ag ratio manually**")
    cu_num = st.number_input("Cu count", min_value=1, value=1, step=1, key="cu")
    ag_num = st.number_input("Ag count", min_value=1, value=1, step=1, key="ag")
    if st.button("Set c_bulk", key="set_comp"):
        ratio = cu_num / ag_num
        c_bulk = 1.0 / ratio
        c_bulk = max(0.1, min(1.0, c_bulk))  # clip
        st.session_state['c_bulk'] = round(c_bulk, 3)
        st.session_state['ratio_str'] = f"{cu_num}:{ag_num}"
        st.success(f"c_bulk = {c_bulk} (Cu:Ag = {cu_num}:{ag_num})")

# ------------------ Calculations ------------------
st.header("📐 Derived Parameters")
if 'core_diameter' in st.session_state and 'c_bulk' in st.session_state:
    d_core = st.session_state['core_diameter']
    r_core = d_core / 2.0

    if geo_mode == "Provide fc":
        fc = fc_input
        L0 = d_core / (2 * fc)
    elif geo_mode == "Provide shell distance":
        L0 = d_core + 2 * d_shell
        fc = r_core / L0   # because r_core = fc * L0
    else:  # Provide L0
        L0 = L0_input
        fc = r_core / L0

    rs = rs_default

    c_bulk = st.session_state['c_bulk']
    ratio_str = st.session_state.get('ratio_str', '?')

    st.write(f"**Core diameter:** {d_core} nm → core radius = {r_core:.2f} nm")
    st.write(f"**fc (core fraction):** {fc:.4f}")
    st.write(f"**L0 (domain size):** {L0:.2f} nm")
    st.write(f"**c_bulk (Ag concentration):** {c_bulk}")
    st.write(f"**rs (shell fraction):** {rs:.2f}")

    # Build parameter dict
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

    # Generate query if LLM loaded
    if st.session_state.get('llm_tokenizer') is not None and st.session_state.get('llm_model') is not None:
        if st.button("🚀 Generate Query with LLM"):
            with st.spinner("Generating..."):
                query = generate_query(
                    st.session_state['llm_tokenizer'],
                    st.session_state['llm_model'],
                    params,
                    chosen_llm
                )
                st.session_state['generated_query'] = query
                st.text_area("📋 Copy this query into CoreShellGPT:", query, height=100)
    else:
        st.info("LLM not loaded. Install transformers & torch and load from sidebar to enable.")

    # Always provide a simple default query (fallback)
    default_query = f"Design a core-shell with L0={params['L0']:.1f} nm, fc={params['fc']:.3f}, c_bulk={params['c_bulk']:.2f}, rs={params['rs']:.2f}, time=1e-3 s from HRTEM (core {params['core_diameter']} nm) and EDS (Cu:Ag={params['ratio_str']})."
    st.text_area("📋 Default query (can be edited):", default_query, height=100)

else:
    st.info("Please set both core diameter and c_bulk first.")
