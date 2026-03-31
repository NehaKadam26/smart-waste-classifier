import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #D8E4BC; }

    [data-testid="stFileUploader"] {
        background-color: #c8d9a8;
        border: 2px dashed #2D6A4F;
        border-radius: 20px;
        padding: 1.5rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 24px;
    }

    .stButton > button {
        background-color: #2D6A4F;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1b4332;
        color: white;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Landfill", "Recyclable"]

CATEGORY_INFO = {
    "Recyclable": {
        "bin_value": "Plastics & Metal",
        "prep": ["Rinse before disposal", "Crush to save space", "Remove non-PET labels"],
        "description": "This item can be processed and reused. Recycling it reduces landfill waste, conserves raw materials, and lowers carbon emissions.",
        "badge_bg": "#2D6A4F",
        "badge_text": "white",
        "badge_icon": "♻️",
    },
    "Landfill": {
        "bin_value": "General Waste",
        "prep": ["Seal in a bag if wet", "Do not mix with recycling", "Check local disposal rules"],
        "description": "This item cannot be recycled through standard streams. Dispose of it in your general waste bin to avoid contaminating recyclables.",
        "badge_bg": "#D4A373",
        "badge_text": "white",
        "badge_icon": "🗑️",
    },
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(image):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(model, image):
    x = preprocess(image)
    prob = model.predict(x, verbose=0)[0][0]
    label_idx = int(prob >= 0.5)
    confidence = prob if label_idx == 1 else 1 - prob
    return CLASS_NAMES[label_idx], float(confidence)

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 1.5rem 0 0.5rem 0;'>
    <h1 style='color:#1b4332; font-size:2.8rem; font-weight:800; margin-bottom:0.2rem;'>♻️ Smart Waste Classifier</h1>
    <p style='color:#40916c; font-size:1.05rem; margin-top:0;'>AI-powered waste classification using MobileNetV2 · 94% accuracy</p>
</div>
""", unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader("📂  Drop your waste item image here", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    label, confidence = predict(model, image)
    pct = int(confidence * 100)
    info = CATEGORY_INFO[label]

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.1], gap="large")

    with col1:
        st.image(image, use_container_width=True)
        st.markdown(f"""
        <div style='background:{info["badge_bg"]}; color:{info["badge_text"]}; border-radius:50px;
                    padding:0.7rem 1.4rem; text-align:center; font-size:1.1rem; font-weight:800;
                    letter-spacing:0.05em; margin-top:0.8rem;'>
            {info["badge_icon"]}  {label.upper()}  ·  {pct}% confidence
        </div>
        """, unsafe_allow_html=True)

    with col2:
        name = uploaded_file.name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()

        st.markdown(f"""
        <h2 style='color:#1b4332; font-size:2rem; font-weight:800; margin-bottom:0.3rem;'>{name}</h2>
        <p style='color:#555; font-size:0.97rem; margin-bottom:1.2rem;'>{info["description"]}</p>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div style='background:#f5f0e8; border-radius:18px; padding:1.2rem;'>
                <div style='font-size:1.5rem; margin-bottom:0.4rem;'>🗂️</div>
                <div style='font-size:0.65rem; color:#888; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;'>Target Bin</div>
                <div style='font-size:1rem; font-weight:800; color:#2D6A4F; margin-top:0.3rem;'>{info["bin_value"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            prep_html = "".join([f"<div style='font-size:0.82rem; color:#444; margin:0.25rem 0;'>✓ {p}</div>" for p in info["prep"]])
            st.markdown(f"""
            <div style='background:#f5f0e8; border-radius:18px; padding:1.2rem;'>
                <div style='font-size:1.5rem; margin-bottom:0.4rem;'>🧹</div>
                <div style='font-size:0.65rem; color:#888; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;'>Preparation</div>
                {prep_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#f5f0e8; border-radius:18px; padding:1.2rem 1.4rem;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;'>
                <div>
                    <div style='font-size:1rem; font-weight:800; color:#1b4332;'>Impact Profile</div>
                    <div style='font-size:0.78rem; color:#888;'>Model confidence for this classification</div>
                </div>
                <div style='font-size:2rem; font-weight:800; color:#2D6A4F;'>{pct}%</div>
            </div>
            <div style='background:#c8d9a8; border-radius:99px; height:10px;'>
                <div style='background:linear-gradient(90deg, #74C69D, #2D6A4F); width:{pct}%; height:10px; border-radius:99px;'></div>
            </div>
            <div style='display:flex; justify-content:space-between; font-size:0.68rem; color:#888; margin-top:0.4rem; font-weight:600; letter-spacing:0.05em;'>
                <span>LOW CONFIDENCE</span><span>HIGH CONFIDENCE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("📷  Start New Scan", use_container_width=True):
            st.rerun()