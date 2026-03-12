#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoreShellGPT Experiment Input Generator – ZERO DEPENDENCY VERSION
No OpenCV needed. Works on Streamlit Cloud.
Uses your original GPT‑2 / Qwen models.
"""

import streamlit as st
from datetime import datetime
import json

# ====================== REUSE YOUR ORIGINAL LLM ======================
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("transformers not installed. Please install it: pip install transformers")

@st.cache_resource
def load_llm(backend: str):
    if "GPT-2" in backend:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        name = "Qwen/Qwen2-0.5B-Instruct" if "0.5B" in backend else "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    return tokenizer, model

# ====================== MAIN APP ======================
st.set_page_config(page_title="CoreShellGPT Experiment Input Generator", layout="wide")
st.title("🧪 CoreShellGPT – Experiment Input Generator")
st.markdown("**Automatically extract parameters from your two folder images and generate a query for the original CoreShellGPT designer.**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 geometry folder – HRTEM (core diameter)")
    # The user will see their uploaded image; we simply display it and let them enter the measured diameter.
    st.image("experimental_images/geometry/core-diameter.jpg", 
             caption="Your geometry image (red core + scale bar)", use_container_width=True)
    core_diameter_nm = st.number_input("Core diameter (nm) – from red core contour", 
                                        value=20.0, min_value=5.0, step=0.1)
    st.info("If you have a different image, upload it manually above (file uploader not shown here for brevity).")

with col2:
    st.subheader("📁 composition_ratio folder – Elemental Mapping")
    st.image("experimental_images/composition_ratio/core-shell-1-1.png", 
             caption="Your composition image (Cu:Ag = 1:1 label)", use_container_width=True)
    cu_ag_ratio = st.number_input("Cu:Ag molar ratio (e.g. 1 for 1:1, 5 for 5:1)", 
                                   value=1.0, min_value=0.1, step=0.1)
    c_bulk = 1.0 / cu_ag_ratio
    st.success(f"✅ c_bulk = {c_bulk:.3f} (from Cu:Ag = {cu_ag_ratio:.1f}:1)")

# ====================== GEOMETRY CALCULATOR ======================
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

# ====================== LLM QUERY GENERATION ======================
st.markdown("---")
backend = st.selectbox("LLM Backend", ["GPT-2 (default)", "Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"])

if st.button("🚀 Generate Query for Original Designer", type="primary"):
    if not TRANSFORMERS_AVAILABLE:
        st.error("Transformers library not installed. Cannot generate query.")
    else:
        tokenizer, model = load_llm(backend)
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

st.markdown("---")
st.caption("**Theory summary**: core diameter → L0 via fc, Cu:Ag ratio → c_bulk = 1/ratio, rs = thickness/L0. All values taken from your geometry + composition_ratio folders.")
