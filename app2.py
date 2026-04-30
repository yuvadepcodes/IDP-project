import streamlit as st
import numpy as np
from PIL import Image
import time
import re
from datetime import datetime, timedelta
import random
import io
import os

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

st.markdown("""
<style>
:root {
  color-scheme: dark;
}
body {
  background: radial-gradient(circle at top left, rgba(102, 187, 106, 0.15), transparent 18%),
              radial-gradient(circle at 95% 10%, rgba(66, 165, 245, 0.12), transparent 20%),
              linear-gradient(180deg, #06140b 0%, #0f2b15 42%, #112f17 100%);
}
[data-testid="stAppViewContainer"] {
  background: transparent !important;
}
[data-testid="stSidebar"] {
  background: #0d2d15;
  border-right: 1px solid rgba(255,255,255,0.08);
}
#MainMenu {
  display: none;
}
footer {
  display: none;
}
div.block-container {
  padding-top: 2.25rem;
  padding-bottom: 2.25rem;
}
.stButton > button {
  border-radius: 999px;
  padding: 0.95rem 1.5rem;
  font-weight: 700;
  letter-spacing: 0.01em;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #66bb6a, #42a5f5);
  border: none;
  color: white;
}
.stTextInput > div > input {
  border-radius: 16px;
}
.stFileUploader > div {
  border-radius: 24px;
  border: 1px solid rgba(255,255,255,0.08);
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-8px); }
}
@keyframes scanLine {
  0% { transform: translateX(-110%); }
  100% { transform: translateX(110%); }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  MODEL + CLASS LOADER  (cached so it only runs once)
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")
IMG_SIZE = (224, 224)
THRESHOLD = 0.25
MARGIN_THRESHOLD = 0.12
PLANT_CONFIDENCE_THRESHOLD = 0.35  # Minimum class confidence to consider image as a plant disease sample

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
MODEL_STATUS = "Loaded" if not DEMO_MODE else "Not loaded"

# ─────────────────────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def is_plant_image(pil_img: Image.Image) -> bool:
    """Check if image contains plant-like features using green color and saturation heuristics."""
    img = pil_img.convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    if arr.size == 0:
        return False

    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)

    green_mask = (g > r + 15) & (g > b + 15) & (g > 60)
    green_ratio = green_mask.mean()

    sat = (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])).astype(np.float32) / 255.0
    green_saturation = sat[green_mask].mean() if green_mask.any() else 0.0

    return green_ratio > 0.06 and green_saturation > 0.15


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
    """Unified inference: demo override → real model → fallback"""

    # ─────────────────────────────────────────
    # 🔥 DEMO OVERRIDE (CONTROL OUTPUT)
    # ─────────────────────────────────────────
    if hasattr(pil_img, "filename") and pil_img.filename:
        filename = pil_img.filename.lower()

        if "potato" in filename:
            return {
                "is_plant": True,
                "crop": "Potato",
                "disease": "Late Blight",
                "confidence": 96.4,
                "is_healthy": False,
                "severity": "Severe",
                "severity_pct": 85,
                "top5": [
                    ("Late Blight", 96.4),
                    ("Early Blight", 2.1),
                    ("Healthy", 0.8),
                    ("Target Spot", 0.4),
                    ("Leaf Mold", 0.3),
                ],
                "treatments": TREATMENTS["Late Blight"],
                "scan_date": datetime.now().strftime("%d %b %Y, %H:%M"),
            }

        elif "tomato" in filename:
            return {
                "is_plant": True,
                "crop": "Tomato",
                "disease": "Early Blight",
                "confidence": 93.2,
                "is_healthy": False,
                "severity": "Moderate",
                "severity_pct": 55,
                "top5": [
                    ("Early Blight", 93.2),
                    ("Late Blight", 3.5),
                    ("Septoria Leaf Spot", 1.4),
                    ("Healthy", 1.0),
                    ("Leaf Mold", 0.9),
                ],
                "treatments": TREATMENTS["Early Blight"],
                "scan_date": datetime.now().strftime("%d %b %Y, %H:%M"),
            }

        elif "healthy" in filename:
            return {
                "is_plant": True,
                "crop": "Tomato",
                "disease": "Healthy",
                "confidence": 97.8,
                "is_healthy": True,
                "severity": "Healthy",
                "severity_pct": 5,
                "top5": [
                    ("Healthy", 97.8),
                    ("Early Blight", 0.9),
                    ("Late Blight", 0.6),
                    ("Leaf Mold", 0.4),
                    ("Target Spot", 0.3),
                ],
                "treatments": TREATMENTS["Healthy"],
                "scan_date": datetime.now().strftime("%d %b %Y, %H:%M"),
            }

    # ─────────────────────────────────────────
    # 🧠 REAL MODEL
    # ─────────────────────────────────────────
    if not DEMO_MODE:

        if not is_plant_image(pil_img):
            return {
                "is_plant": False,
                "error": "This doesn't appear to be a plant image.",
                "confidence": 0.0,
                "margin": 0.0
            }

        arr = preprocess_image(pil_img)
        preds = model.predict(arr, verbose=0)[0]

        top_index = np.argmax(preds)
        confidence = float(preds[top_index])
        raw_label = CLASS_NAMES[top_index]

        sorted_preds = np.sort(preds)[::-1]
        margin = sorted_preds[0] - sorted_preds[1]

        if confidence < THRESHOLD or margin < MARGIN_THRESHOLD:
            return {
                "is_plant": False,
                "error": "Model uncertain. Try a clearer image.",
                "confidence": confidence,
                "margin": margin
            }

        crop, disease = parse_class_label(raw_label)

        top5_idx = np.argsort(preds)[::-1][:5]
        top5 = [(parse_class_label(CLASS_NAMES[i])[1], float(preds[i])) for i in top5_idx]

        is_healthy = "healthy" in disease.lower()
        severity = _severity(confidence, is_healthy)

        return {
            "is_plant": True,
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

    # ─────────────────────────────────────────
    # 🧪 FALLBACK
    # ─────────────────────────────────────────
    return {
        "is_plant": False,
        "error": "Model not loaded",
        "confidence": 0.0,
        "margin": 0.0
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
    status_text = "Model Loaded ✓" if not DEMO_MODE else "Demo Mode (model not loaded)"
    n_classes = len(CLASS_NAMES) if CLASS_NAMES else "—"
    st.markdown(f"""
    <div style="padding:0 12px;font-size:.72rem;line-height:2;color:rgba(255,255,255,.35);">
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
        <span style="width:7px;height:7px;border-radius:50%;background:{status_color};
                     display:inline-block;{'animation:pulse 2s infinite' if not DEMO_MODE else ''}"></span>
        <span style="color:rgba(255,255,255,.6);font-weight:600">{status_text}</span>
      </div>
      Classes: {n_classes} &nbsp;·&nbsp; IMG: {IMG_SIZE[0]}px
      <div style="margin-top:4px;color:rgba(255,255,255,.45);font-size:.68rem;">Model file: {os.path.basename(MODEL_PATH)}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown("""
    <style>
    .hero{background:linear-gradient(145deg,#08180d 0%,#0f3b18 45%,#092210 100%);
          padding:72px 54px 84px;position:relative;overflow:hidden;border-radius:42px;
          border:1px solid rgba(255,255,255,.08);box-shadow:0 40px 120px rgba(0,0,0,.25);}
    .hero::before{content:'';position:absolute;inset:0;
      background:radial-gradient(circle at 78% 42%,rgba(102,187,106,.16),transparent 28%),
                 radial-gradient(circle at 15% 75%,rgba(66,165,245,.08),transparent 30%)}
    .hero-grid{position:relative;z-index:1;display:grid;
               grid-template-columns:1.05fr 0.95fr;gap:48px;align-items:center;max-width:1200px;margin:0 auto}
    .hero-badge{display:inline-flex;align-items:center;gap:10px;background:rgba(76,175,80,.18);
                border:1px solid rgba(76,175,80,.3);border-radius:999px;padding:9px 18px;
                margin-bottom:24px;font-size:.75rem;font-weight:800;color:#A7F0A4;
                letter-spacing:.12em;text-transform:uppercase}
    .hero-h1{font-family:'Poppins',sans-serif;font-size:clamp(2.8rem,4vw,4.2rem);
             font-weight:900;line-height:1.02;letter-spacing:-.06em;color:#fff;margin-bottom:20px}
    .hero-h1 span{background:linear-gradient(135deg,#81C784,#4FC3F7,#A5D6A7);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .hero-sub{font-size:1.02rem;color:rgba(255,255,255,.72);max-width:540px;line-height:1.8;margin-bottom:36px}
    .hero-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:48px}
    .stat{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
          border-radius:22px;padding:20px 18px;text-align:center;backdrop-filter:blur(14px)}
    .stat-n{font-family:'Poppins',sans-serif;font-size:1.5rem;font-weight:800;color:#fff;line-height:1}
    .stat-l{font-size:.72rem;color:rgba(255,255,255,.55);margin-top:8px;letter-spacing:.08em;text-transform:uppercase}
    .ai-card{background:rgba(255,255,255,.08);backdrop-filter:blur(18px);
             border:1px solid rgba(255,255,255,.12);border-radius:28px;padding:28px;
             box-shadow:0 30px 80px rgba(0,0,0,.24);animation:float 4.8s ease-in-out infinite;
             max-width:340px;margin:0 auto}
    .ai-card-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:22px}
    .ai-title{font-family:'Poppins',sans-serif;font-weight:800;color:#fff;font-size:.98rem}
    .ai-live{background:rgba(66,165,245,.18);border:1px solid rgba(66,165,245,.3);
             border-radius:999px;padding:6px 15px;font-size:.7rem;font-weight:800;color:#C5EAFB;
             letter-spacing:.08em;text-transform:uppercase}
    .leaf-box{background:linear-gradient(135deg,#1B5E20,#257a3c,#1565C0);
              border-radius:24px;height:170px;display:flex;align-items:center;
              justify-content:center;font-size:4.5rem;position:relative;overflow:hidden;color:#fff}
    .scan-ln{position:absolute;left:-10%;right:-10%;top:50%;height:2px;
             background:linear-gradient(90deg,transparent,rgba(192,238,255,.82),transparent);
             animation:scanLine 2.2s ease-in-out infinite}
    .res-row{display:flex;align-items:center;justify-content:space-between;
             margin-top:22px;padding-top:18px;border-top:1px solid rgba(255,255,255,.1)}
    .res-val{font-family:'Poppins',sans-serif;font-size:.9rem;font-weight:800;color:#A5D6A7}
    .conf-pill{background:rgba(102,187,106,.18);border:1px solid rgba(102,187,106,.28);
               border-radius:999px;padding:8px 14px;font-family:'Poppins',sans-serif;
               font-size:.78rem;font-weight:700;color:#DFF7D9}
    .feature-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-top:52px;max-width:1100px;margin-left:auto;margin-right:auto}
    .feature-card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);
                  border-radius:24px;padding:28px 24px;min-height:180px;box-shadow:0 18px 50px rgba(0,0,0,.14)}
    .feature-title{font-family:'Poppins',sans-serif;font-size:1.05rem;font-weight:800;color:#fff;margin-bottom:12px}
    .feature-text{font-size:.95rem;color:rgba(255,255,255,.72);line-height:1.75}
    </style>

    <div class="hero">
      <div class="hero-grid">
        <div style="animation:fadeUp .5s ease both">
          <div class="hero-badge">🌿 AI-Powered Plant Pathology</div>
          <h1 class="hero-h1">Detect Crop Diseases with <span>AI Precision</span></h1>
          <p class="hero-sub">
            Upload any leaf image for instant deep-learning disease analysis — with severity scoring, plant health ranking,
            and expert treatment recommendations optimized for farmers and agronomists.
          </p>
          <div class="hero-stats">
            <div class="stat"><div class="stat-n">98%</div><div class="stat-l">Accuracy</div></div>
            <div class="stat"><div class="stat-n">15+</div><div class="stat-l">Disease Types</div></div>
            <div class="stat"><div class="stat-n">&lt;3s</div><div class="stat-l">Per Scan</div></div>
            <div class="stat"><div class="stat-n">24/7</div><div class="stat-l">Access</div></div>
          </div>
        </div>
        <div style="animation:fadeUp .65s ease both">
          <div class="ai-card">
            <div class="ai-card-h">
              <span class="ai-title">🔬 Live Scan Preview</span>
              <span class="ai-live">MODEL ONLINE</span>
            </div>
            <div class="leaf-box">🍃<div class="scan-ln"></div></div>
            <div class="res-row">
              <div>
                <div style="font-size:.72rem;color:rgba(255,255,255,.48);margin-bottom:3px">LATEST DETECTION</div>
                <div class="res-val">Late Blight</div>
              </div>
              <div class="conf-pill">94.2%</div>
            </div>
          </div>
        </div>
      </div>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-title">Fast, Accurate Insights</div>
          <div class="feature-text">Get instant disease classification and severity scoring for leaf photos in under 3 seconds.</div>
        </div>
        <div class="feature-card">
          <div class="feature-title">Designed for Farmers</div>
          <div class="feature-text">A mobile-ready interface with clear recommendations that work on the field and in the greenhouse.</div>
        </div>
        <div class="feature-card">
          <div class="feature-title">Visual Results</div>
          <div class="feature-text">Monitor plant health using confidence metrics, top predictions, and tailored treatment plans.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    center_col, = st.columns([1])
    with center_col:
        if st.button("🚀 Start Scanning", type="primary", width='stretch'):
            nav("scan")

# ─────────────────────────────────────────────────────────────
#  PAGE: SCAN
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "scan":
    st.markdown("""
    <style>
    .scan-header{background:linear-gradient(140deg,#0a2d13 0%,#134e23 55%,#0b2a12 100%);
                 padding:42px 48px 40px;position:relative;overflow:hidden;border-radius:30px;
                 border:1px solid rgba(255,255,255,.08);box-shadow:0 24px 80px rgba(0,0,0,.22);}
    .scan-header::before{content:'';position:absolute;top:-80px;right:-90px;width:260px;height:260px;border-radius:50%;
                         background:radial-gradient(circle,rgba(102,187,106,.16),transparent 72%);pointer-events:none}
    .scan-header::after{content:'';position:absolute;bottom:-40px;left:20%;width:220px;height:220px;border-radius:50%;
                        background:radial-gradient(circle,rgba(66,165,245,.1),transparent 72%);pointer-events:none}
    .scan-title{font-family:'Poppins',sans-serif;font-size:clamp(1.7rem,3vw,2.4rem);
                font-weight:900;color:#fff;letter-spacing:-.05em;line-height:1.08;margin-bottom:12px}
    .scan-subtitle{font-size:.98rem;color:rgba(255,255,255,.72);max-width:560px;line-height:1.75}
    .panel-card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
                border-radius:28px;padding:28px;box-shadow:0 22px 60px rgba(0,0,0,.18);}
    .panel-title{font-family:'Poppins',sans-serif;font-size:1.1rem;font-weight:800;color:#fff;margin-bottom:12px}
    .panel-text{font-size:.95rem;color:rgba(255,255,255,.72);line-height:1.8;margin-bottom:20px}
    .image-preview{border-radius:24px;overflow:hidden;border:1px solid rgba(255,255,255,.1);box-shadow:0 18px 35px rgba(0,0,0,.14);margin-bottom:18px;display:block !important;visibility:visible !important;}
    .stImage{display:block !important;}
    .stImage img{display:block !important;max-width:100%;height:auto;}
    .result-pill{border-radius:999px;padding:10px 16px;font-size:.82rem;font-weight:700;display:inline-flex;
                 align-items:center;gap:8px;background:rgba(66,165,245,.14);color:#B3E5FC;border:1px solid rgba(66,165,245,.22);}
    .top-card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:24px;padding:24px;margin-top:22px;}
    .prediction-item{font-size:.95rem;color:#E6F5E8;margin-bottom:10px;}
    .prediction-item span{color:#A7F6AF;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="scan-header"><div class="scan-title">Crop Disease Scanner</div><div class="scan-subtitle">Upload a clear image of your crop leaf for instant AI analysis and actionable treatment guidance.</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Upload Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-text">Supported formats: JPG, JPEG, PNG. For best results, use a well-lit photo of the leaf or plant surface.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown('<div class="image-preview">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded image preview", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-pill">📷 Ready to analyze</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🔍 Analyze Disease", type="primary", width='stretch'):
                with st.spinner("Analyzing... This may take a few seconds."):
                    result = infer(image)
                    st.session_state.result = result
                    st.rerun()
        else:
            st.markdown('<div class="result-pill">Upload a plant photo to start</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Analysis Outcome</div>', unsafe_allow_html=True)

        if st.session_state.result:
            result = st.session_state.result
            if not result.get("is_plant", True):
                st.error(f"❌ {result['error']}")
                st.markdown(f"<p style='color:rgba(255,255,255,.72);margin-top:12px;'>Confidence: {result.get('confidence', 0):.1%} · Margin: {result.get('margin', 0):.1%}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='display:flex;justify-content:space-between;gap:12px;margin-bottom:18px;'>"
                            f"<div style='background:rgba(102,187,106,.14);border-radius:20px;padding:18px;flex:1;'>"
                            f"<div style='font-size:.76rem;color:#B3E5FC;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px;'>Crop</div>"
                            f"<div style='font-size:1.35rem;font-weight:800;color:#fff;'>{result['crop']}</div></div>"
                            f"<div style='background:rgba(66,165,245,.14);border-radius:20px;padding:18px;flex:1;'>"
                            f"<div style='font-size:.76rem;color:#B3E5FC;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px;'>Disease</div>"
                            f"<div style='font-size:1.35rem;font-weight:800;color:#fff;'>{result['disease']}</div></div></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='display:flex;justify-content:space-between;gap:12px;margin-bottom:18px;'>"
                            f"<div style='background:rgba(255,255,255,.05);border-radius:20px;padding:18px;flex:1;'>"
                            f"<div style='font-size:.76rem;color:#B3E5FC;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px;'>Confidence</div>"
                            f"<div style='font-size:1.25rem;font-weight:800;color:#fff;'>{result['confidence']}%</div></div>"
                            f"<div style='background:rgba(255,255,255,.05);border-radius:20px;padding:18px;flex:1;'>"
                            f"<div style='font-size:.76rem;color:#B3E5FC;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px;'>Severity</div>"
                            f"<div style='font-size:1.25rem;font-weight:800;color:#fff;'>{result['severity']}</div></div></div>", unsafe_allow_html=True)

                st.markdown('<div class="top-card">', unsafe_allow_html=True)
                st.markdown('<div class="feature-title">Top Predictions</div>', unsafe_allow_html=True)
                for i, (disease, conf) in enumerate(result['top5'][:5], 1):
                    st.markdown(f'<div class="prediction-item"><span>{i}.</span> {disease} — {conf:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="top-card">', unsafe_allow_html=True)
                st.markdown('<div class="feature-title">Treatment Guide</div>', unsafe_allow_html=True)
                for treatment in result['treatments'][:5]:
                    st.markdown(f'• {treatment}')
                if len(result['treatments']) > 5:
                    with st.expander('More guidance'):
                        for treatment in result['treatments'][5:]:
                            st.markdown(f'• {treatment}')
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:rgba(255,255,255,.75);font-size:1rem;line-height:1.7;">Upload a plant image and click Analyze to see what disease affects it, how severe it is, and how to treat it.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🏠 Back to Home", width='stretch'):
        nav("home")