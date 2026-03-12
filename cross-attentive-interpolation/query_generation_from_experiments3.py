#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT – Experimental Input Generator (Zero Dependency Version)
----------------------------------------------------------------------
- No OpenCV, no pytesseract – all values entered manually.
- Scans experimental_images/geometry and experimental_images/composition_ratio.
- Displays selected images (requires only Pillow).
- Geometry conversion (L0, fc, rs) using your exact formulas.
- Optional LLM query generation (if transformers & torch installed).
- Falls back to default query when LLM unavailable.
"""

import streamlit as st
import os
import glob
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
    # Not a problem – LLM generation will be disabled

# Pillow is required for image display
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    st.error("Pillow is required to display images. Please install it: pip install Pillow")
    st.stop()

# -------------------- Helper: LLM loading (cached) --------------------
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

# -------------------- Helper to list images in a folder --------------------
def list_images_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        return []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    return sorted(files)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Experimental Input Generator for CoreShellGPT", layout="wide")
st.title("🧪 CoreShellGPT – Experimental Input Generator (Zero Dependency)")
st.markdown("**All values are entered manually – no OpenCV required.**")

# Define folders
GEOMETRY_FOLDER = "experimental_images/geometry"
COMPOSITION_FOLDER = "experimental_images/composition_ratio"

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
    st.markdown("**Select an image from the geometry folder (for reference only).**")

    # Folder dropdown
    geo_files = list_images_in_folder(GEOMETRY_FOLDER)
    if geo_files:
        geo_filenames = [os.path.basename(f) for f in geo_files]
        selected_geo = st.selectbox("Select image from geometry folder", geo_filenames, key="geo_dropdown")
        geo_path = os.path.join(GEOMETRY_FOLDER, selected_geo)
        geo_img = Image.open(geo_path).convert("RGB")
        st.image(geo_img, caption=f"Geometry: {selected_geo}", use_container_width=True)
        st.session_state['geo_image'] = geo_img
        st.info(f"Loaded from {GEOMETRY_FOLDER}")
    else:
        st.info(f"No images found in '{GEOMETRY_FOLDER}'. You can still enter values manually.")

    # Manual diameter entry (no automatic extraction)
    diam_manual = st.number_input("Enter core diameter (nm)", min_value=0.0, value=20.0, step=0.1)
    if st.button("Set core diameter", key="set_geo"):
        st.session_state['core_diameter'] = diam_manual
        st.success(f"Core diameter set to {diam_manual} nm")

with col2:
    st.subheader("📁 Elemental Mapping / Composition Image")
    st.markdown("**Select an image from the composition_ratio folder (for reference only).**")

    # Folder dropdown
    comp_files = list_images_in_folder(COMPOSITION_FOLDER)
    if comp_files:
        comp_filenames = [os.path.basename(f) for f in comp_files]
        selected_comp = st.selectbox("Select image from composition_ratio folder", comp_filenames, key="comp_dropdown")
        comp_path = os.path.join(COMPOSITION_FOLDER, selected_comp)
        comp_img = Image.open(comp_path).convert("RGB")
        st.image(comp_img, caption=f"Composition: {selected_comp}", use_container_width=True)
        st.session_state['comp_image'] = comp_img
        st.info(f"Loaded from {COMPOSITION_FOLDER}")
    else:
        st.info(f"No images found in '{COMPOSITION_FOLDER}'. You can still enter values manually.")

    # Manual ratio entry (no automatic extraction)
    st.markdown("**Enter Cu:Ag ratio manually**")
    cu_num = st.number_input("Cu count", min_value=1, value=1, step=1, key="cu")
    ag_num = st.number_input("Ag count", min_value=1, value=1, step=1, key="ag")
    if st.button("Set c_bulk", key="set_comp"):
        ratio = cu_num / ag_num
        c_bulk = 1.0 / ratio
        c_bulk = np.clip(c_bulk, 0.1, 1.0)
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
