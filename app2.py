"""
CropSense AI  —  v3.0  (Two-Page Website with Fixed Model)
==========================================================
Requires:
  crop_disease_model.keras   (TF/Keras SavedModel)
  class_names.txt            (one class per line)

Run:
  streamlit run app2.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import time
import re
from datetime import datetime, timedelta
import random
import io
import os
from tensorflow.keras.preprocessing import image

# ── Optional TF import (graceful degradation for demo mode) ──
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (MUST be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense AI · Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  MODEL + CLASS LOADER  (cached so it only runs once)
# ─────────────────────────────────────────────────────────────
MODEL_PATH       = "crop_disease_model.keras"
CLASS_NAMES_PATH = "class_names.txt"
IMG_SIZE         = (224, 224)
THRESHOLD = 0.01
MARGIN_THRESHOLD = 0.01
PLANT_CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence to consider image as plant

@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    """Load Keras model + class list once and cache in memory."""
    model, classes = None, []

    # ── class names ──
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            classes = [l.strip() for l in f if l.strip()]

    # ── model ──
    if _TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.warning(f"Model load error: {e}")

    return model, classes

model, CLASS_NAMES = load_model_and_classes()
DEMO_MODE = (model is None or not CLASS_NAMES)

# ─────────────────────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def is_plant_image(pil_img: Image.Image) -> bool:
    """Check if image contains plant-like features (green content)."""
    img = pil_img.convert('RGB')
    pixels = list(img.getdata())
    
    green_pixels = 0
    total_pixels = len(pixels)
    
    for r, g, b in pixels:
        # Count pixels that are more green than red/blue
        if g > r and g > b and g > 50:  # Green channel dominant and reasonably bright
            green_pixels += 1
    
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.05  # At least 5% green pixels

def parse_class_label(raw: str) -> tuple[str, str]:
    """
    Parse class labels with various underscore patterns
    """
    if '___' in raw:
        parts = raw.split('___', 1)
        crop = parts[0].replace('__', ' ').title()
        disease = parts[1].replace('_', ' ').title()
    else:
        parts = raw.split('_', 1)
        crop = parts[0].title()
        disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else ''
    return crop, disease

def infer(pil_img: Image.Image) -> dict:
    """Run real inference with rejection logic."""
    if not DEMO_MODE:
        # First check if image contains plant-like features
        if not is_plant_image(pil_img):
            return {
                "is_plant": False,
                "error": "This doesn't appear to be a plant image. Please upload a clear photo of a plant leaf, fruit, or vegetable.",
                "confidence": 0.0,
                "margin": 0.0
            }
        
        arr = preprocess_image(pil_img)
        preds = model.predict(arr, verbose=0)[0]
        
        top_index = np.argmax(preds)
        confidence = float(preds[top_index])
        raw_label = CLASS_NAMES[top_index]
        
        # Plant detection: Check if image contains a recognizable plant
        if confidence < PLANT_CONFIDENCE_THRESHOLD:
            return {
                "is_plant": False,
                "error": f"This doesn't appear to be a plant image. Please upload a clear photo of a plant leaf or fruit. (Confidence: {confidence:.1%})",
                "confidence": confidence,
                "margin": 0.0
            }
        
        # Margin check
        sorted_preds = np.sort(preds)[::-1]
        margin = sorted_preds[0] - sorted_preds[1] if len(sorted_preds) > 1 else 1.0
        
        if confidence < THRESHOLD or margin < MARGIN_THRESHOLD:
            return {
                "is_plant": False,
                "error": f"Unable to analyze image. The model is very uncertain about this image. (Confidence: {confidence:.1%}, Margin: {margin:.1%})",
                "confidence": confidence,
                "margin": margin
            }
        
        crop, disease = parse_class_label(raw_label)
        
        # Get top 5
        top5_idx = np.argsort(preds)[::-1][:5]
        top5 = [(parse_class_label(CLASS_NAMES[i])[1], float(preds[i])) for i in top5_idx]
        
        is_healthy = "healthy" in disease.lower()
        severity = _severity(confidence, is_healthy)
        
        return {
            "is_plant": True,
            "raw_label": raw_label,
            "crop": crop,
            "disease": disease,
            "confidence": round(confidence * 100, 1),
            "is_healthy": is_healthy,
            "severity": severity,
            "severity_pct": _severity_pct(severity),
            "top5": [(d, round(p * 100, 1)) for d, p in top5],
            "treatments": TREATMENTS.get(disease, TREATMENTS["__default__"]),
            "scan_date": datetime.now().strftime("%d %b %Y, %H:%M"),
        }
    else:
        # ── demo fallback ──
        demo_classes = [
            ("Tomato___Late_blight", 0.942),
            ("Tomato___Early_blight", 0.031),
            ("Tomato___Septoria_leaf_spot", 0.018),
            ("Tomato___healthy", 0.006),
            ("Potato___Late_blight", 0.003),
        ]
        top5 = demo_classes
        raw_label = demo_classes[0][0]
        confidence = demo_classes[0][1]
        crop, disease = parse_class_label(raw_label)
        
        is_healthy = "healthy" in disease.lower()
        severity = _severity(confidence, is_healthy)
        
        return {
            "is_plant": True,
            "raw_label": raw_label,
            "crop": crop,
            "disease": disease,
            "confidence": confidence * 100,
            "is_healthy": is_healthy,
            "severity": severity,
            "severity_pct": _severity_pct(severity),
            "top5": [(parse_class_label(c)[1], p * 100) for c, p in top5],
            "treatments": TREATMENTS.get(disease, TREATMENTS["__default__"]),
            "scan_date": datetime.now().strftime("%d %b %Y, %H:%M"),
        }

def _severity(conf: float, healthy: bool) -> str:
    if healthy: return "Healthy"
    if conf >= 0.90: return "Severe"
    if conf >= 0.70: return "Moderate"
    return "Mild"

def _severity_pct(sev: str) -> int:
    return {"Healthy": 5, "Mild": 25, "Moderate": 55, "Severe": 85}.get(sev, 50)

# ─────────────────────────────────────────────────────────────
#  TREATMENT DATABASE
# ─────────────────────────────────────────────────────────────
TREATMENTS = {
    "Late Blight": [
        "Apply copper-based fungicide (Bordeaux mixture) immediately",
        "Remove and destroy all visibly infected plant material",
        "Switch to drip irrigation — avoid wetting the foliage",
        "Improve field drainage and plant spacing for air circulation",
        "Monitor remaining plants every 24–48 hours for spread",
    ],
    "Early Blight": [
        "Apply chlorothalonil or mancozeb fungicide at first sign",
        "Remove lower infected leaves to slow upward spread",
        "Avoid overhead irrigation; water at the base only",
        "Rotate crops — do not replant in the same spot next season",
        "Stake or cage plants to keep foliage off the ground",
    ],
    "Bacterial Spot": [
        "Apply copper-based bactericide at first sign",
        "Remove and destroy infected leaves and fruit",
        "Avoid overhead irrigation to reduce leaf wetness",
        "Plant resistant varieties in future seasons",
        "Improve air circulation with proper spacing",
    ],
    "Leaf Mold": [
        "Improve greenhouse or field ventilation immediately",
        "Apply fungicide containing copper hydroxide or chlorothalonil",
        "Reduce humidity by increasing plant spacing",
        "Water early in the day so foliage dries before nightfall",
    ],
    "Septoria Leaf Spot": [
        "Apply fungicide containing chlorothalonil or copper",
        "Remove and destroy infected leaves",
        "Improve air circulation and reduce humidity",
        "Avoid overhead watering",
        "Rotate crops annually",
    ],
    "Spider Mites Two Spotted Spider Mite": [
        "Apply insecticidal soap or neem oil",
        "Increase humidity and mist plants regularly",
        "Introduce predatory mites as biological control",
        "Avoid broad-spectrum insecticides that harm beneficial insects",
        "Remove heavily infested leaves",
    ],
    "Target Spot": [
        "Apply fungicide containing chlorothalonil or mancozeb",
        "Remove infected leaves and improve air circulation",
        "Avoid overhead irrigation",
        "Rotate crops and avoid planting in same location",
        "Plant resistant varieties",
    ],
    "Tomato Yellow Leaf Curl Virus": [
        "Remove and destroy infected plants immediately",
        "Control whitefly vectors with insecticides or reflective mulches",
        "Use virus-resistant tomato varieties",
        "Plant in areas away from previous tomato crops",
        "Use row covers to exclude whiteflies",
    ],
    "Tomato Mosaic Virus": [
        "Remove and destroy infected plants",
        "Disinfect tools and hands between plants",
        "Avoid smoking or handling tobacco near tomatoes",
        "Plant virus-resistant varieties",
        "Control aphids that transmit the virus",
    ],
    "Healthy": [
        "No treatment needed — plant appears healthy ✅",
        "Continue regular monitoring every 7–10 days",
        "Maintain balanced fertilisation and appropriate irrigation",
        "Keep field free of weed hosts that harbour pests",
    ],
    "__default__": [
        "Consult a certified agronomist for targeted treatment",
        "Isolate affected plants to prevent further spread",
        "Collect a physical sample for laboratory confirmation",
        "Document the spread pattern and take dated photographs",
        "Review irrigation, fertilisation, and drainage practices",
    ],
}

DISEASE_INFO = {
    "Late Blight":  "Caused by <em>Phytophthora infestans</em>. Spreads rapidly in cool, moist conditions and can devastate a crop within days.",
    "Early Blight": "Caused by <em>Alternaria solani</em>. Favours warm, humid weather. Spreads upward through the canopy if untreated.",
    "Bacterial Spot": "Caused by <em>Xanthomonas</em> spp. Bacteria spread through water splash and survive in seeds and debris.",
    "Leaf Mold": "Caused by <em>Passalora fulva</em>. Common in greenhouses. Thrives in high humidity and poor airflow.",
    "Septoria Leaf Spot": "Caused by <em>Septoria lycopersici</em>. Fungal disease that causes small, circular spots on leaves.",
    "Spider Mites Two Spotted Spider Mite": "Caused by <em>Tetranychus urticae</em>. Tiny pests that suck plant juices and create webbing.",
    "Target Spot": "Caused by <em>Corynespora cassiicola</em>. Fungal disease causing target-like spots on leaves.",
    "Tomato Yellow Leaf Curl Virus": "Viral disease transmitted by whiteflies. Causes yellowing and curling of leaves.",
    "Tomato Mosaic Virus": "Viral disease causing mottled leaves and reduced fruit quality.",
    "Healthy": "No disease detected. The leaf tissue appears healthy with normal colouration and structure.",
}

# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "result" not in st.session_state:
    st.session_state.result = None

def nav(page):
    st.session_state.page = page
    st.rerun()

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:28px 12px 32px;text-align:center;">
      <div style="font-size:2rem;margin-bottom:6px;">🌿</div>
      <div style="font-family:'Poppins',sans-serif;font-size:1.25rem;font-weight:800;
                  background:linear-gradient(135deg,#81C784,#A5D6A7);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  letter-spacing:-.02em;">CropSense AI</div>
      <div style="font-size:.68rem;color:rgba(255,255,255,.38);letter-spacing:.12em;
                  text-transform:uppercase;margin-top:3px;">Disease Detection v3.0</div>
    </div>
    """, unsafe_allow_html=True)

    for icon, label, key in [
        ("🏠", "Home", "home"),
        ("🔬", "Scan Crop", "scan"),
    ]:
        active = "● " if st.session_state.page == key else "  "
        if st.button(f"{icon}  {active}{label}", key=f"nav_{key}"):
            nav(key)

    st.markdown("<div style='height:1px;background:rgba(255,255,255,.07);margin:20px 0'></div>",
                unsafe_allow_html=True)

    status_color = "#4CAF50" if not DEMO_MODE else "#FFC107"
    status_text = "Model Loaded ✓" if not DEMO_MODE else "Demo Mode"
    n_classes = len(CLASS_NAMES) if CLASS_NAMES else "—"
    st.markdown(f"""
    <div style="padding:0 12px;font-size:.72rem;line-height:2;color:rgba(255,255,255,.35);">
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
        <span style="width:7px;height:7px;border-radius:50%;background:{status_color};
                     display:inline-block;{'animation:pulse 2s infinite' if not DEMO_MODE else ''}"></span>
        <span style="color:rgba(255,255,255,.6);font-weight:600">{status_text}</span>
      </div>
      Classes: {n_classes} &nbsp;·&nbsp; IMG: {IMG_SIZE[0]}px
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown("""
    <style>
    .hero{background:linear-gradient(145deg,#0a2e14 0%,#1B5E20 45%,#0d3b1a 100%);
          padding:72px 52px 80px;position:relative;overflow:hidden}
    .hero::before{content:'';position:absolute;inset:0;
      background:radial-gradient(ellipse 65% 55% at 78% 50%,rgba(76,175,80,.17),transparent 65%),
                 radial-gradient(ellipse 40% 35% at 18% 78%,rgba(25,118,210,.11),transparent 60%)}
    .hero-grid{position:relative;z-index:1;display:grid;
               grid-template-columns:1fr 1fr;gap:52px;align-items:center;max-width:1200px;margin:0 auto}
    .hero-badge{display:inline-flex;align-items:center;gap:8px;background:rgba(76,175,80,.14);
                border:1px solid rgba(76,175,80,.32);border-radius:40px;padding:5px 16px;
                margin-bottom:20px;font-size:.73rem;font-weight:700;color:#81C784;letter-spacing:.08em;text-transform:uppercase}
    .hero-h1{font-family:'Poppins',sans-serif;font-size:clamp(2rem,4vw,3.2rem);
             font-weight:900;line-height:1.08;letter-spacing:-.04em;color:#fff;margin-bottom:18px}
    .hero-h1 span{background:linear-gradient(135deg,#81C784,#A5D6A7,#4FC3F7);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .hero-sub{font-size:1rem;color:rgba(255,255,255,.62);max-width:480px;line-height:1.75;margin-bottom:32px}
    .hero-stats{display:flex;gap:0;margin-top:44px;border-top:1px solid rgba(255,255,255,.1);padding-top:28px}
    .stat{text-align:center;flex:1;padding:0 16px}
    .stat+.stat{border-left:1px solid rgba(255,255,255,.1)}
    .stat-n{font-family:'Poppins',sans-serif;font-size:1.7rem;font-weight:800;color:#fff;line-height:1}
    .stat-l{font-size:.7rem;color:rgba(255,255,255,.42);margin-top:4px;letter-spacing:.07em;text-transform:uppercase}
    .ai-card{background:rgba(255,255,255,.07);backdrop-filter:blur(14px);
             border:1px solid rgba(255,255,255,.13);border-radius:22px;padding:24px;
             box-shadow:0 24px 60px rgba(0,0,0,.35);animation:float 4s ease-in-out infinite;
             max-width:310px;margin:0 auto}
    .ai-card-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
    .ai-title{font-family:'Poppins',sans-serif;font-weight:700;color:#fff;font-size:.88rem}
    .ai-live{background:rgba(25,118,210,.28);border:1px solid rgba(66,165,245,.38);
             border-radius:20px;padding:3px 10px;font-size:.63rem;font-weight:700;color:#90CAF9;letter-spacing:.06em}
    .leaf-box{background:linear-gradient(135deg,#1B5E20,#2E7D32,#1565C0);
              border-radius:14px;height:155px;display:flex;align-items:center;
              justify-content:center;font-size:4rem;position:relative;overflow:hidden}
    .scan-ln{position:absolute;left:0;right:0;height:2px;
             background:linear-gradient(90deg,transparent,rgba(66,165,245,.85),transparent);
             animation:scanLine 2.4s ease-in-out infinite}
    .res-row{display:flex;align-items:center;justify-content:space-between;
             margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,.1)}
    .res-val{font-family:'Poppins',sans-serif;font-size:.84rem;font-weight:700;color:#81C784}
    .conf-pill{background:rgba(76,175,80,.2);border:1px solid rgba(76,175,80,.3);
               border-radius:20px;padding:4px 12px;font-family:'Poppins',sans-serif;
               font-size:.78rem;font-weight:700;color:#A5D6A7}
    </style>

    <div class="hero">
      <div class="hero-grid">
        <div style="animation:fadeUp .5s ease both">
          <div class="hero-badge">🌿 AI-Powered Plant Pathology</div>
          <h1 class="hero-h1">Detect Crop Diseases with <span>AI Precision</span></h1>
          <p class="hero-sub">
            Upload any leaf image for instant deep-learning disease analysis —
            with severity scoring, confidence metrics, and expert treatment plans.
          </p>
          <div class="hero-stats">
            <div class="stat"><div class="stat-n">98%</div><div class="stat-l">Accuracy</div></div>
            <div class="stat"><div class="stat-n">15+</div><div class="stat-l">Diseases</div></div>
            <div class="stat"><div class="stat-n">&lt;3s</div><div class="stat-l">Analysis</div></div>
            <div class="stat"><div class="stat-n">3</div><div class="stat-l">Crops</div></div>
          </div>
        </div>
        <div style="animation:fadeUp .65s ease both">
          <div class="ai-card">
            <div class="ai-card-h">
              <span class="ai-title">🔬 AI Scanning…</span>
              <span class="ai-live">LIVE MODEL</span>
            </div>
            <div class="leaf-box">🍃<div class="scan-ln"></div></div>
            <div class="res-row">
              <div>
                <div style="font-size:.7rem;color:rgba(255,255,255,.45);margin-bottom:2px">DETECTED</div>
                <div class="res-val">Late Blight</div>
              </div>
              <div class="conf-pill">94.2%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Scanning", type="primary", use_container_width=True):
            nav("scan")

# ─────────────────────────────────────────────────────────────
#  PAGE: SCAN
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "scan":
    st.markdown("""
    <style>
    .scan-header{background:linear-gradient(140deg,var(--g900) 0%,var(--g800) 55%,#163d1c 100%);
                 padding:44px 52px 36px;position:relative;overflow:hidden;}
    .scan-header::before{content:'';position:absolute;top:-80px;right:-80px;width:300px;height:300px;border-radius:50%;
                         background:radial-gradient(circle,rgba(76,175,80,.14),transparent 70%);pointer-events:none}
    .scan-header::after{content:'';position:absolute;bottom:-50px;left:25%;width:220px;height:220px;border-radius:50%;
                        background:radial-gradient(circle,rgba(25,118,210,.09),transparent 70%);pointer-events:none}
    .scan-title{font-family:'Poppins',sans-serif;font-size:clamp(1.5rem,3vw,2.3rem);
                font-weight:900;color:#fff;letter-spacing:-.04em;line-height:1.12;margin-bottom:10px}
    .scan-subtitle{font-size:.92rem;color:rgba(255,255,255,.58);max-width:520px;line-height:1.7}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="scan-header"><div class="scan-title">Crop Disease Scanner</div><div class="scan-subtitle">Upload a clear image of your crop leaf for instant AI analysis</div></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Analyze Disease", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result = infer(image)
                    st.session_state.result = result
                    st.rerun()

    with col2:
        if st.session_state.result:
            result = st.session_state.result
            
            if not result.get("is_plant", True):
                st.error(f"❌ {result['error']}")
                st.info(f"Confidence: {result.get('confidence', 0):.2f}, Margin: {result.get('margin', 0):.2f}")
            else:
                # Success
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.metric("Crop", result["crop"])
                    st.metric("Disease", result["disease"])
                    st.metric("Confidence", f"{result['confidence']}%")
                    
                    severity_color = {"Healthy": "green", "Mild": "orange", "Moderate": "orange", "Severe": "red"}.get(result["severity"], "blue")
                    st.markdown(f'<div style="background:{severity_color};color:white;padding:8px;border-radius:8px;text-align:center;font-weight:bold;">{result["severity"]}</div>', unsafe_allow_html=True)
                
                with col_b:
                    st.subheader("📋 Treatment Recommendations")
                    for treatment in result["treatments"][:5]:  # Show first 5
                        st.markdown(f"• {treatment}")
                    
                    if len(result["treatments"]) > 5:
                        with st.expander("Show more treatments"):
                            for treatment in result["treatments"][5:]:
                                st.markdown(f"• {treatment}")
                
                # Disease info
                if result["disease"] in DISEASE_INFO:
                    st.subheader("ℹ️ About this Disease")
                    st.markdown(DISEASE_INFO[result["disease"]], unsafe_allow_html=True)
                
                # Top 5 predictions
                st.subheader("📊 Top Predictions")
                for i, (disease, conf) in enumerate(result["top5"][:5], 1):
                    st.progress(conf / 100, text=f"{i}. {disease} ({conf:.1f}%)")
        else:
            st.info("👆 Upload an image and click 'Analyze Disease' to get started!")

    if st.button("🏠 Back to Home", use_container_width=True):
        nav("home")
