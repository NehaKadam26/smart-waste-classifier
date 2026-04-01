import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

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
    div[data-testid="stImage"] img { border-radius: 24px; }
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
    .stButton > button:hover { background-color: #1b4332; color: white; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.h5")
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "class_indices.json")
IMG_SIZE = (224, 224)

# Load class indices and reverse to index → name
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
INDEX_TO_CLASS = {v: k for k, v in class_indices.items()}

CATEGORY_INFO = {
    "battery": {
        "bin_value": "Hazardous Waste",
        "prep": ["Do not crush or puncture", "Tape terminals before disposal", "Take to a battery drop-off point"],
        "description": "Batteries contain toxic chemicals. Never put them in general waste — take them to a designated hazardous waste facility.",
        "badge_bg": "#e63946",
        "badge_text": "white",
        "badge_icon": "🔋",
    },
    "biological": {
        "bin_value": "Compost / Organic Bin",
        "prep": ["Keep separate from other waste", "Use compostable bags", "Dispose within 48 hours"],
        "description": "Biological waste can be composted to create nutrient-rich soil. Keep it separate from recyclables to avoid contamination.",
        "badge_bg": "#6a994e",
        "badge_text": "white",
        "badge_icon": "🌿",
    },
    "brown-glass": {
        "bin_value": "Glass Recycling",
        "prep": ["Rinse thoroughly", "Remove lids and caps", "Do not break intentionally"],
        "description": "Brown glass is fully recyclable. Keep it separate from other glass colours where required by your local facility.",
        "badge_bg": "#7f5539",
        "badge_text": "white",
        "badge_icon": "🍺",
    },
    "cardboard": {
        "bin_value": "Paper & Cardboard Bin",
        "prep": ["Flatten boxes to save space", "Remove tape and staples", "Keep dry — wet cardboard is not recyclable"],
        "description": "Cardboard is one of the most recycled materials. Flatten and keep it dry to ensure it gets processed correctly.",
        "badge_bg": "#d4a373",
        "badge_text": "white",
        "badge_icon": "📦",
    },
    "clothes": {
        "bin_value": "Textile Recycling",
        "prep": ["Wash before donating", "Bag separately from other waste", "Check for donation drives nearby"],
        "description": "Clothes and textiles should never go to landfill. Donate wearable items or drop off at a textile recycling point.",
        "badge_bg": "#9b5de5",
        "badge_text": "white",
        "badge_icon": "👕",
    },
    "green-glass": {
        "bin_value": "Glass Recycling",
        "prep": ["Rinse thoroughly", "Remove lids and caps", "Do not break intentionally"],
        "description": "Green glass is fully recyclable. Separate by colour where required and ensure bottles are clean before disposal.",
        "badge_bg": "#2D6A4F",
        "badge_text": "white",
        "badge_icon": "🍾",
    },
    "metal": {
        "bin_value": "Metal Recycling",
        "prep": ["Rinse food tins", "Crush cans to save space", "Remove paper labels if possible"],
        "description": "Metal is infinitely recyclable without quality loss. Ensure items are clean and free of food residue.",
        "badge_bg": "#6c757d",
        "badge_text": "white",
        "badge_icon": "🥫",
    },
    "paper": {
        "bin_value": "Paper & Cardboard Bin",
        "prep": ["Keep dry", "Remove plastic windows from envelopes", "Shred sensitive documents first"],
        "description": "Paper is highly recyclable. Keep it clean and dry — greasy or wet paper cannot be processed.",
        "badge_bg": "#457b9d",
        "badge_text": "white",
        "badge_icon": "📄",
    },
    "plastic": {
        "bin_value": "Plastic Recycling",
        "prep": ["Rinse before disposal", "Crush to save space", "Check the resin code on the bottom"],
        "description": "Many plastics are recyclable. Rinse containers and check your local guidelines for accepted plastic types.",
        "badge_bg": "#0096c7",
        "badge_text": "white",
        "badge_icon": "🧴",
    },
    "shoes": {
        "bin_value": "Textile / Shoe Recycling",
        "prep": ["Tie pairs together", "Check for shoe donation bins", "Remove loose insoles separately"],
        "description": "Shoes should not go to landfill. Many retailers and charities accept used shoes for recycling or donation.",
        "badge_bg": "#e76f51",
        "badge_text": "white",
        "badge_icon": "👟",
    },
    "trash": {
        "bin_value": "General Waste",
        "prep": ["Seal in a bag if wet", "Do not mix with recycling", "Check local disposal rules"],
        "description": "This item cannot be recycled through standard streams. Dispose of it in your general waste bin.",
        "badge_bg": "#adb5bd",
        "badge_text": "white",
        "badge_icon": "🗑️",
    },
    "white-glass": {
        "bin_value": "Glass Recycling",
        "prep": ["Rinse thoroughly", "Remove lids and caps", "Do not break intentionally"],
        "description": "White/clear glass is fully recyclable. Separate by colour where required and ensure bottles are clean.",
        "badge_bg": "#dee2e6",
        "badge_text": "#333",
        "badge_icon": "🥛",
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
    preds = model.predict(x, verbose=0)[0]  # shape: (12,)
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = INDEX_TO_CLASS[idx]
    return label, confidence

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 1.5rem 0 0.5rem 0;'>
    <h1 style='color:#1b4332; font-size:2.8rem; font-weight:800; margin-bottom:0.2rem;'>♻️ Smart Waste Classifier</h1>
    <p style='color:#40916c; font-size:1.05rem; margin-top:0;'>AI-powered waste classification using MobileNetV2 · 12 waste categories</p>
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