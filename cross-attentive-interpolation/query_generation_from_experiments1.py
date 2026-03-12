#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT Experiment Input Generator – Full Version
--------------------------------------------------------
- Automatically displays images from:
    experimental_images/geometry/
    experimental_images/composition_ratio/
- Manual input of core diameter and Cu:Ag ratio (no OpenCV needed)
- Geometry conversion: L0 from fc or from shell distance
- Uses GPT‑2 / Qwen to generate a natural‑language query
- Ready to paste into the original CoreShellGPT Intelligent Designer tab
"""

import streamlit as st
import os
from datetime import datetime

# -------------------- Transformers (LLM) --------------------
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("transformers not installed. Install with: pip install transformers")

@st.cache_resource
def load_llm(backend: str):
    """Load the selected LLM (cached)."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    if "GPT-2" in backend:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        name = "Qwen/Qwen2-0.5B-Instruct" if "0.5B" in backend else "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    return tokenizer, model

# -------------------- Folder paths --------------------
GEOMETRY_FOLDER = "experimental_images/geometry"
COMPOSITION_FOLDER = "experimental_images/composition_ratio"

# -------------------- Helper: list images --------------------
def list_image_files(folder):
    if not os.path.isdir(folder):
        return []
    exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith(exts):
            files.append(os.path.join(folder, f))
    return sorted(files)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="CoreShellGPT Experiment Input Generator", layout="wide")
st.title("🧪 CoreShellGPT – Experiment Input Generator")
st.markdown("**Automatically extract parameters from your two folder images and generate a query for the original CoreShellGPT designer.**")

# Sidebar: LLM selection
st.sidebar.header("🧠 LLM Settings")
llm_options = ["GPT-2 (default)", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"]
selected_llm = st.sidebar.selectbox("Choose backend", llm_options)
if st.sidebar.button("Load LLM"):
    with st.spinner("Loading model... (first time may take a minute)"):
        tokenizer, model = load_llm(selected_llm)
        st.session_state['llm_tokenizer'] = tokenizer
        st.session_state['llm_model'] = model
        st.sidebar.success(f"{selected_llm} loaded")

# Main area: two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 geometry folder – HRTEM (core diameter)")
    geo_files = list_image_files(GEOMETRY_FOLDER)
    if geo_files:
        geo_names = [os.path.basename(f) for f in geo_files]
        selected_geo = st.selectbox("Select image", geo_names, key="geo_selector")
        geo_path = os.path.join(GEOMETRY_FOLDER, selected_geo)
        st.image(geo_path, caption=f"Geometry: {selected_geo}", use_container_width=True)
        st.info(f"Loaded from {GEOMETRY_FOLDER}")
    else:
        st.warning(f"No images found in '{GEOMETRY_FOLDER}'. Please create the folder and add images.")
        st.stop()

    core_diameter_nm = st.number_input("Core diameter (nm) – from red core contour", 
                                        value=20.0, min_value=5.0, step=0.1)

with col2:
    st.subheader("📁 composition_ratio folder – Elemental Mapping")
    comp_files = list_image_files(COMPOSITION_FOLDER)
    if comp_files:
        comp_names = [os.path.basename(f) for f in comp_files]
        selected_comp = st.selectbox("Select image", comp_names, key="comp_selector")
        comp_path = os.path.join(COMPOSITION_FOLDER, selected_comp)
        st.image(comp_path, caption=f"Composition: {selected_comp}", use_container_width=True)
        st.info(f"Loaded from {COMPOSITION_FOLDER}")
    else:
        st.warning(f"No images found in '{COMPOSITION_FOLDER}'. Please create the folder and add images.")
        st.stop()

    cu_ag_ratio = st.number_input("Cu:Ag molar ratio (e.g. 1 for 1:1, 5 for 5:1)", 
                                   value=1.0, min_value=0.1, step=0.1)
    c_bulk = 1.0 / cu_ag_ratio
    st.success(f"✅ c_bulk = {c_bulk:.3f} (from Cu:Ag = {cu_ag_ratio:.1f}:1)")

# -------------------- Geometry Calculator --------------------
st.markdown("---")
st.subheader("3. Geometry Calculator (L0 & fc)")

mode = st.radio("Choose your preference", 
                ["I know fc (core fraction)", "I know shell distance from core surface"])

if mode == "I know fc (core fraction)":
    fc = st.slider("fc", 0.05, 0.45, 0.18, 0.01)
    L0 = core_diameter_nm / (2 * fc)
    st.metric("Calculated L0", f"{L0:.1f} nm")
else:
    shell_distance_nm = st.number_input("Shell distance from core surface (nm)", value=5.0)
    L0 = core_diameter_nm + 2 * shell_distance_nm
    fc = core_diameter_nm / (2 * L0)
    st.metric("Calculated L0", f"{L0:.1f} nm")
    st.metric("Derived fc", f"{fc:.3f}")

rs = st.number_input("Shell fraction rs (default 0.1)", value=0.1, step=0.01)
st.caption("**Formula used:** rs = desired_shell_thickness_nm / L0_nm")

# -------------------- Query Generation --------------------
st.markdown("---")
if st.button("🚀 Generate Query for Original Designer", type="primary"):
    if not TRANSFORMERS_AVAILABLE:
        st.error("Transformers library not installed. Cannot generate query.")
    elif 'llm_tokenizer' not in st.session_state or 'llm_model' not in st.session_state:
        st.error("Please load an LLM from the sidebar first.")
    else:
        tokenizer = st.session_state['llm_tokenizer']
        model = st.session_state['llm_model']
        with st.spinner("LLM is writing the perfect input..."):
            prompt = f"""Create a natural-language sentence for CoreShellGPT using these experimental values:
Core diameter (geometry folder): {core_diameter_nm} nm
c_bulk (composition_ratio folder): {c_bulk:.3f}
L0: {L0:.1f} nm, fc: {fc:.3f}, rs: {rs}
Output ONLY the sentence ready to paste."""
            
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            outputs = model.generate(inputs, max_new_tokens=120, temperature=0.0, do_sample=False)
            query = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            st.text_area("✅ Copy this into your original CoreShellGPT Intelligent Designer tab", query, height=120)
            st.download_button("📥 Download query.txt", query, "experiment_query.txt")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("**Theory summary**: core diameter → L0 via fc, Cu:Ag ratio → c_bulk = 1/ratio, rs = thickness/L0. All values taken from your geometry + composition_ratio folders.")
