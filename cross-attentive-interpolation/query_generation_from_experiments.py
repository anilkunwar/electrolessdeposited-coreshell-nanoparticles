import streamlit as st
import numpy as np
import cv2
from PIL import Image
import re
import json
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM
)
import pytesseract  # optional, for reading scale‑bar text
from collections import OrderedDict

# ---------- Helper: LLM loading (cached) ----------
@st.cache_resource(show_spinner="Loading selected LLM...")
def load_llm(backend: str):
    if "GPT-2" in backend:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        # Qwen variants
        model_name = "Qwen/Qwen2-0.5B-Instruct" if "Qwen2-0.5B" in backend else "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    return tokenizer, model

# ---------- Image processing functions ----------
def extract_core_diameter(image_pil, scale_px, scale_nm):
    """Return core diameter in nm."""
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red mask (HSV wraps around 0)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find largest red contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    diameter_px = 2 * radius
    diameter_nm = diameter_px * scale_nm / scale_px
    return round(diameter_nm, 2)

def extract_c_bulk(image_pil, use_ocr=True):
    """Return (c_bulk, ratio_text) from elemental map."""
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red (Cu) mask
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Green (Ag) mask
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Optionally use OCR to read text labels
    if use_ocr:
        try:
            text = pytesseract.image_to_string(image_pil)
            match = re.search(r'Cu\s*:\s*Ag\s*=\s*(\d+)\s*:\s*(\d+)', text, re.IGNORECASE)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                ratio = x / y
                c_bulk = 1.0 / ratio if ratio != 0 else 1.0
                c_bulk = np.clip(c_bulk, 0.1, 1.0)
                return round(c_bulk, 3), f"{x}:{y}"
        except:
            pass

    # Fallback to pixel counting
    n_cu = np.sum(mask_red > 0)
    n_ag = np.sum(mask_green > 0)
    if n_ag == 0:
        return 1.0, "Ag not found"
    ratio = n_cu / n_ag
    c_bulk = 1.0 / ratio
    c_bulk = np.clip(c_bulk, 0.1, 1.0)
    ratio_str = f"{n_cu//(n_ag if n_ag>0 else 1)}:{1}"
    return round(c_bulk, 3), ratio_str

# ---------- Query generation by LLM ----------
def generate_query(llm_tokenizer, llm_model, params, backend):
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
    # Extract sentence after the prompt
    if full_prompt in answer:
        answer = answer.replace(full_prompt, "").strip()
    return answer

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Experimental Input Generator for CoreShellGPT", layout="wide")
st.title("🧪 CoreShellGPT – Experimental Input Generator")
st.markdown("Upload your HRTEM (core–shell) and elemental mapping images to extract parameters and generate a ready‑to‑paste query for the CoreShellGPT designer.")

# LLM selection
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
use_ocr = st.sidebar.checkbox("Use OCR to read scale bar / ratio labels", value=True)

# Main area: two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 HRTEM / Geometry Image")
    st.markdown("**Red core, green shell, with scale bar.**")
    geo_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="geo")
    if geo_file:
        geo_img = Image.open(geo_file).convert("RGB")
        st.image(geo_img, caption="Geometry image", use_container_width=True)

        # Scale bar input
        scale_nm = st.number_input("Scale bar length (nm)", value=20.0, key="scale_nm")
        scale_px = st.number_input("Scale bar length (pixels)", value=100, key="scale_px")
        if st.button("Extract core diameter"):
            diam = extract_core_diameter(geo_img, scale_px, scale_nm)
            if diam:
                st.session_state['core_diameter'] = diam
                st.success(f"Core diameter = {diam} nm")
            else:
                st.error("Could not detect core. Try adjusting HSV thresholds or manual input.")
        # Manual override
        diam_manual = st.number_input("Or enter core diameter manually (nm)", min_value=0.0, value=20.0, step=0.1)
        if st.button("Use manual diameter"):
            st.session_state['core_diameter'] = diam_manual
            st.success(f"Manual diameter set to {diam_manual} nm")

with col2:
    st.subheader("📁 Elemental Mapping / Composition Image")
    st.markdown("**Red = Cu, Green = Ag (or explicit label like Cu:Ag=5:1).**")
    comp_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="comp")
    if comp_file:
        comp_img = Image.open(comp_file).convert("RGB")
        st.image(comp_img, caption="Composition image", use_container_width=True)
        if st.button("Extract c_bulk"):
            c_bulk, ratio_str = extract_c_bulk(comp_img, use_ocr=use_ocr)
            st.session_state['c_bulk'] = c_bulk
            st.session_state['ratio_str'] = ratio_str
            st.success(f"Cu:Ag = {ratio_str} → c_bulk = {c_bulk}")
        # Manual ratio
        st.markdown("**Or enter ratio manually**")
        cu_num = st.number_input("Cu count", min_value=1, value=1, step=1, key="cu")
        ag_num = st.number_input("Ag count", min_value=1, value=1, step=1, key="ag")
        if st.button("Use manual ratio"):
            ratio = cu_num / ag_num
            c_bulk = 1.0 / ratio
            c_bulk = np.clip(c_bulk, 0.1, 1.0)
            st.session_state['c_bulk'] = round(c_bulk, 3)
            st.session_state['ratio_str'] = f"{cu_num}:{ag_num}"
            st.success(f"Manual ratio → c_bulk = {c_bulk}")

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
        fc = r_core / L0   # because r_core = fc * L0? Wait check: fc = r_core / L0, yes.
    else:  # Provide L0
        L0 = L0_input
        fc = r_core / L0

    # rs from default (or could be computed if user wants specific shell thickness)
    rs = rs_default
    # Optional: compute from desired shell thickness
    # if user provides t_shell_nm, then rs = t_shell_nm / L0

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
    if 'llm_tokenizer' in st.session_state and 'llm_model' in st.session_state:
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
        st.info("Load an LLM from the sidebar to generate a natural language query.")

    # Also provide a simple default query (fallback)
    default_query = f"Design a core-shell with L0={params['L0']:.1f} nm, fc={params['fc']:.3f}, c_bulk={params['c_bulk']:.2f}, rs={params['rs']:.2f}, time=1e-3 s from HRTEM (core {params['core_diameter']} nm) and EDS (Cu:Ag={params['ratio_str']})."
    st.text_area("📋 Default query (can be edited):", default_query, height=100)

else:
    st.info("Please upload both images and extract the core diameter and c_bulk first.")
