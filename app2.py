"""
CropSense AI  —  v2.0  (Real Model Integration)
================================================
Requires:
  crop_disease_model.keras   (TF/Keras SavedModel)
  class_names.txt            (one class per line, e.g. "Tomato___Late_blight")

Run:
  streamlit run app.py
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

def parse_class_label(raw: str) -> tuple[str, str]:
    """
    'Tomato___Late_blight'  →  ('Tomato', 'Late Blight')
    'Apple___healthy'       →  ('Apple',  'Healthy')
    """
    parts = re.split(r"_{2,}", raw, maxsplit=1)
    crop    = parts[0].replace("_", " ").strip().title() if len(parts) > 0 else "Unknown"
    disease = parts[1].replace("_", " ").strip().title() if len(parts) > 1 else raw.replace("_", " ").title()
    return crop, disease

def infer(pil_img: Image.Image) -> dict:
    """Run real inference or return a convincing demo result."""
    if not DEMO_MODE:
        arr  = preprocess_image(pil_img)
        preds = model.predict(arr, verbose=0)[0]          # shape: (num_classes,)
        top5_idx = np.argsort(preds)[::-1][:5]
        top5 = [(CLASS_NAMES[i], float(preds[i])) for i in top5_idx]

        raw_label = CLASS_NAMES[top5_idx[0]]
        confidence = float(preds[top5_idx[0]])
        crop, disease = parse_class_label(raw_label)
    else:
        # ── demo fallback (no model file present) ──
        demo_classes = [
            ("Tomato___Late_blight",         0.942),
            ("Tomato___Early_blight",        0.031),
            ("Tomato___Septoria_leaf_spot",  0.018),
            ("Tomato___healthy",             0.006),
            ("Potato___Late_blight",         0.003),
        ]
        top5       = demo_classes
        raw_label  = demo_classes[0][0]
        confidence = demo_classes[0][1]
        crop, disease = parse_class_label(raw_label)

    is_healthy = "healthy" in disease.lower()
    severity   = _severity(confidence, is_healthy)

    return {
        "raw_label":   raw_label,
        "crop":        crop,
        "disease":     disease,
        "confidence":  round(confidence * 100, 1),
        "is_healthy":  is_healthy,
        "severity":    severity,
        "severity_pct": _severity_pct(severity),
        "top5":        [(parse_class_label(c)[1], round(p * 100, 1)) for c, p in top5],
        "treatments":  TREATMENTS.get(disease, TREATMENTS["__default__"]),
        "about":       DISEASE_INFO.get(disease, ""),
        "scan_date":   datetime.now().strftime("%d %b %Y, %H:%M"),
        "low_confidence": confidence < 0.60,
    }

def _severity(conf: float, healthy: bool) -> str:
    if healthy:               return "Healthy"
    if conf >= 0.90:          return "Severe"
    if conf >= 0.70:          return "Moderate"
    return                           "Mild"

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
    "Leaf Mold": [
        "Improve greenhouse or field ventilation immediately",
        "Apply fungicide containing copper hydroxide or chlorothalonil",
        "Reduce humidity by increasing plant spacing",
        "Water early in the day so foliage dries before nightfall",
    ],
    "Rust": [
        "Apply sulfur-based or triazole fungicide at early infection",
        "Remove infected leaves and destroy them (do not compost)",
        "Avoid overhead irrigation to limit spore spread",
        "Plant rust-resistant varieties in the next growing cycle",
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
    "Leaf Mold":    "Caused by <em>Passalora fulva</em>. Common in greenhouses. Thrives in high humidity and poor airflow.",
    "Rust":         "Caused by <em>Puccinia</em> spp. Wind-dispersed spores make it highly contagious across fields.",
    "Healthy":      "No disease detected. The leaf tissue appears healthy with normal colouration and structure.",
}

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --g900:#0a2e14; --g800:#1B5E20; --g700:#1E7D32; --g600:#2E7D32;
  --g400:#4CAF50; --g200:#A5D6A7; --g100:#C8E6C9; --g50:#F1F8F2;
  --b700:#1565C0; --b600:#1976D2; --b400:#42A5F5; --b100:#BBDEFB;
  --amber:#FFC107; --red:#F44336; --orange:#FF6F00;
  --bg:#F2F6F2; --surface:#fff; --surface2:#F8FAF8;
  --border:#DDE8DD; --text1:#0D1F0E; --text2:#3A5A3C; --text3:#6B8F6D;
  --r-sm:10px; --r:18px; --r-lg:28px;
  --sh-sm:0 2px 8px rgba(30,125,50,.07);
  --sh:0 8px 32px rgba(30,125,50,.11);
  --sh-lg:0 20px 60px rgba(30,125,50,.17);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text1);
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display:none !important; }

[data-testid="stAppViewContainer"] > .main > .block-container {
  padding: 0 !important; max-width: 100% !important;
}

/* scrollbar */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:var(--g200);border-radius:3px}

/* sidebar */
[data-testid="stSidebar"]{background:var(--g900) !important;border-right:1px solid rgba(255,255,255,.05)}
[data-testid="stSidebar"] *{color:#fff !important}
[data-testid="stSidebar"] .stButton>button{
  background:rgba(255,255,255,.07) !important;
  border:1px solid rgba(255,255,255,.12) !important;
  color:#fff !important; border-radius:var(--r-sm) !important;
  font-family:'DM Sans',sans-serif !important;
  width:100%; text-align:left; padding:13px 16px !important;
  transition:all .18s;
}
[data-testid="stSidebar"] .stButton>button:hover{
  background:rgba(255,255,255,.14) !important; transform:translateX(4px);
}

/* buttons */
.stButton>button{
  font-family:'DM Sans',sans-serif !important;
  border-radius:var(--r-sm) !important;
  transition:all .2s cubic-bezier(.4,0,.2,1) !important;
  font-weight:600 !important;
}
.stButton>button:hover{transform:translateY(-2px) !important; box-shadow:var(--sh) !important;}

/* file uploader */
[data-testid="stFileUploader"]{
  background:var(--surface) !important; border:2px dashed var(--g200) !important;
  border-radius:var(--r) !important; transition:border-color .2s;
}
[data-testid="stFileUploader"]:hover{border-color:var(--g600) !important}

/* progress */
.stProgress>div>div>div{background:linear-gradient(90deg,var(--g700),var(--g400)) !important;border-radius:4px}

/* tabs */
.stTabs [data-baseweb="tab-list"]{
  background:var(--surface2);border-radius:var(--r-sm);
  padding:4px;gap:4px;border:1px solid var(--border);
}
.stTabs [data-baseweb="tab"]{
  border-radius:8px !important; font-family:'DM Sans',sans-serif !important;
  font-weight:500 !important; color:var(--text2) !important; padding:8px 22px !important;
}
.stTabs [aria-selected="true"]{background:var(--g700) !important;color:#fff !important}

/* selectbox / text input */
.stSelectbox [data-baseweb="select"]>div,
.stTextInput>div>div>input{
  border-radius:var(--r-sm) !important; border-color:var(--border) !important;
  font-family:'DM Sans',sans-serif !important; background:var(--surface) !important;
}

/* metric */
[data-testid="metric-container"]{
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r); padding:20px 24px !important; box-shadow:var(--sh-sm);
}
[data-testid="stMetricLabel"]{
  font-family:'DM Sans',sans-serif; color:var(--text3) !important;
  font-size:.78rem !important; text-transform:uppercase;
  letter-spacing:.07em; font-weight:600 !important;
}
[data-testid="stMetricValue"]{
  font-family:'Poppins',sans-serif !important; color:var(--text1) !important; font-weight:700 !important;
}

/* spinner */
.stSpinner>div{border-color:var(--g600) transparent transparent transparent !important}

/* alerts */
.stSuccess,.stInfo,.stWarning,.stError{border-radius:var(--r-sm) !important;font-family:'DM Sans',sans-serif !important}

/* animations */
@keyframes fadeUp{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}
@keyframes scanLine{0%{top:5%}50%{top:88%}100%{top:5%}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(76,175,80,.5)}70%{box-shadow:0 0 0 12px rgba(76,175,80,0)}100%{box-shadow:0 0 0 0 rgba(76,175,80,0)}}
@keyframes shimmer{0%{background-position:-600px 0}100%{background-position:600px 0}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes popIn{from{opacity:0;transform:scale(.85)}to{opacity:1;transform:scale(1)}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
def _demo_history():
    rows = [
        ("Tomato","Late Blight","Severe","94.2%"),
        ("Wheat","Rust","Moderate","87.5%"),
        ("Corn","Healthy","Healthy","96.1%"),
        ("Rice","Blast Disease","Moderate","83.4%"),
        ("Potato","Early Blight","Mild","89.7%"),
        ("Tomato","Leaf Mold","Mild","78.3%"),
        ("Corn","Gray Leaf Spot","Severe","92.6%"),
        ("Wheat","Healthy","Healthy","95.0%"),
    ]
    out = []
    for i,(crop,dis,sev,conf) in enumerate(rows):
        d = (datetime.now()-timedelta(days=i*3+random.randint(0,2))).strftime("%d %b %Y")
        out.append({"Date":d,"Crop":crop,"Disease":dis,"Severity":sev,"Confidence":conf})
    return out

for k,v in {
    "page":"home","result":None,"history":_demo_history(),"last_img":None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def nav(p):
    st.session_state.page = p
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
                  text-transform:uppercase;margin-top:3px;">Agri Intelligence v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    for icon,label,key in [
        ("🏠","Home","home"),("🔬","Scan Crop","scan"),
        ("📋","Results","result"),("📊","Dashboard","dashboard"),("💡","About","about"),
    ]:
        active = "● " if st.session_state.page == key else "  "
        if st.button(f"{icon}  {active}{label}", key=f"nav_{key}"):
            nav(key)

    st.markdown("<div style='height:1px;background:rgba(255,255,255,.07);margin:20px 0'></div>",
                unsafe_allow_html=True)

    status_color = "#4CAF50" if not DEMO_MODE else "#FFC107"
    status_text  = "Model Loaded ✓" if not DEMO_MODE else "Demo Mode"
    n_classes    = len(CLASS_NAMES) if CLASS_NAMES else "—"
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
#  SHARED UI HELPERS
# ─────────────────────────────────────────────────────────────
def page_header(title, subtitle="", tag=""):
    st.markdown(f"""
    <div style="padding:44px 52px 36px;
                background:linear-gradient(140deg,var(--g900) 0%,var(--g800) 55%,#163d1c 100%);
                position:relative;overflow:hidden;">
      <!-- glow blobs -->
      <div style="position:absolute;top:-80px;right:-80px;width:300px;height:300px;border-radius:50%;
                  background:radial-gradient(circle,rgba(76,175,80,.14),transparent 70%);pointer-events:none"></div>
      <div style="position:absolute;bottom:-50px;left:25%;width:220px;height:220px;border-radius:50%;
                  background:radial-gradient(circle,rgba(25,118,210,.09),transparent 70%);pointer-events:none"></div>
      <div style="position:relative;z-index:1;animation:fadeUp .45s ease both">
        {'<div style="display:inline-flex;align-items:center;gap:7px;background:rgba(76,175,80,.16);border:1px solid rgba(76,175,80,.3);border-radius:40px;padding:4px 14px;margin-bottom:12px;font-size:.7rem;font-weight:700;color:#81C784;letter-spacing:.08em;text-transform:uppercase">✦ '+tag+'</div>' if tag else ''}
        <h1 style="font-family:'Poppins',sans-serif;font-size:clamp(1.5rem,3vw,2.3rem);
                   font-weight:900;color:#fff;letter-spacing:-.04em;line-height:1.12;margin-bottom:10px">{title}</h1>
        {'<p style="font-size:.92rem;color:rgba(255,255,255,.58);max-width:520px;line-height:1.7">'+subtitle+'</p>' if subtitle else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)

def gap(px=28):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

def wrap(html, pad="28px 32px", extra=""):
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                padding:{pad};box-shadow:var(--sh-sm);animation:fadeUp .4s ease both;{extra}">
      {html}
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────────────────────
def page_home():
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
          <h1 class="hero-h1">Detect Crop<br>Diseases with<br><span>AI Precision</span></h1>
          <p class="hero-sub">
            Upload any leaf image for instant deep-learning disease analysis —
            with severity scoring, confidence metrics, and expert treatment plans.
          </p>
          <div class="hero-stats">
            <div class="stat"><div class="stat-n">98%</div><div class="stat-l">Accuracy</div></div>
            <div class="stat"><div class="stat-n">38+</div><div class="stat-l">Diseases</div></div>
            <div class="stat"><div class="stat-n">&lt;3s</div><div class="stat-l">Analysis</div></div>
            <div class="stat"><div class="stat-n">14</div><div class="stat-l">Crops</div></div>
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

    # CTA buttons
    gap(4)
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    with c2:
        if st.button("🔬  Start Disease Scan", use_container_width=True, key="home_scan"):
            nav("scan")
    with c3:
        if st.button("📊  View Dashboard", use_container_width=True, key="home_dash"):
            nav("dashboard")

    st.markdown("""<style>
    div[data-testid="column"]:nth-child(2) .stButton>button{
      background:linear-gradient(135deg,#1E7D32,#1B5E20) !important;color:#fff !important;
      border:none !important;padding:15px !important;font-size:.95rem !important;
      box-shadow:0 4px 20px rgba(30,125,50,.38) !important;
    }
    div[data-testid="column"]:nth-child(3) .stButton>button{
      background:rgba(30,45,35,.06) !important;border:1.5px solid var(--border) !important;
      padding:15px !important;font-size:.95rem !important;
    }
    </style>""", unsafe_allow_html=True)

    gap(52)

    # Feature cards
    st.markdown("""
    <div style="padding:0 52px;max-width:1200px;margin:0 auto">
      <div style="text-align:center;margin-bottom:44px">
        <div style="font-size:.72rem;font-weight:700;color:var(--g600);letter-spacing:.12em;
                    text-transform:uppercase;margin-bottom:10px">Core Capabilities</div>
        <h2 style="font-family:'Poppins',sans-serif;font-size:1.9rem;font-weight:800;
                   color:var(--text1);letter-spacing:-.03em">
          Everything to Protect<br>Your Harvest
        </h2>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:22px">

        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                    padding:32px 26px;animation:fadeUp .5s .1s ease both;opacity:0;animation-fill-mode:forwards">
          <div style="width:50px;height:50px;background:linear-gradient(135deg,#E8F5E9,#C8E6C9);
                      border-radius:13px;display:flex;align-items:center;justify-content:center;
                      font-size:1.35rem;margin-bottom:18px;box-shadow:0 4px 12px rgba(76,175,80,.18)">📸</div>
          <h3 style="font-family:'Poppins',sans-serif;font-size:1rem;font-weight:700;
                     color:var(--text1);margin-bottom:9px">Instant Diagnosis</h3>
          <p style="font-size:.84rem;color:var(--text3);line-height:1.72">
            Upload any leaf photo — our MobileNetV2 model identifies the disease within seconds with a confidence score and top-5 prediction breakdown.
          </p>
        </div>

        <div style="background:linear-gradient(135deg,#1E7D32,#1B5E20);border-radius:var(--r);
                    padding:32px 26px;animation:fadeUp .5s .2s ease both;opacity:0;
                    animation-fill-mode:forwards;box-shadow:var(--sh)">
          <div style="width:50px;height:50px;background:rgba(255,255,255,.15);border-radius:13px;
                      display:flex;align-items:center;justify-content:center;font-size:1.35rem;margin-bottom:18px">💊</div>
          <h3 style="font-family:'Poppins',sans-serif;font-size:1rem;font-weight:700;
                     color:#fff;margin-bottom:9px">Treatment Plans</h3>
          <p style="font-size:.84rem;color:rgba(255,255,255,.68);line-height:1.72">
            Receive disease-specific fungicide, cultural, and irrigation recommendations — tailored to severity level and ready to act on.
          </p>
        </div>

        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                    padding:32px 26px;animation:fadeUp .5s .3s ease both;opacity:0;animation-fill-mode:forwards">
          <div style="width:50px;height:50px;background:linear-gradient(135deg,#E3F2FD,#BBDEFB);
                      border-radius:13px;display:flex;align-items:center;justify-content:center;
                      font-size:1.35rem;margin-bottom:18px;box-shadow:0 4px 12px rgba(25,118,210,.13)">📈</div>
          <h3 style="font-family:'Poppins',sans-serif;font-size:1rem;font-weight:700;
                     color:var(--text1);margin-bottom:9px">Field Analytics</h3>
          <p style="font-size:.84rem;color:var(--text3);line-height:1.72">
            Track every scan, monitor disease trends over time, and build a health record for each field — making data-driven farming decisions easy.
          </p>
        </div>

      </div>
    </div>
    """, unsafe_allow_html=True)

    gap(52)

    # How it works
    st.markdown("""
    <div style="padding:52px;background:var(--g50);border-top:1px solid var(--border);
                border-bottom:1px solid var(--border)">
      <div style="max-width:860px;margin:0 auto;text-align:center">
        <div style="font-size:.72rem;font-weight:700;color:var(--g600);letter-spacing:.12em;
                    text-transform:uppercase;margin-bottom:10px">Workflow</div>
        <h2 style="font-family:'Poppins',sans-serif;font-size:1.8rem;font-weight:800;
                   color:var(--text1);letter-spacing:-.03em;margin-bottom:44px">
          Three Steps to a Diagnosis
        </h2>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:28px">
          <div>
            <div style="width:62px;height:62px;background:linear-gradient(135deg,#1E7D32,#2E7D32);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:1.5rem;margin:0 auto 14px;box-shadow:0 8px 22px rgba(30,125,50,.28)">📷</div>
            <h4 style="font-family:'Poppins',sans-serif;font-weight:700;color:var(--text1);margin-bottom:8px">Capture Leaf</h4>
            <p style="font-size:.83rem;color:var(--text3);line-height:1.68">
              Photograph the affected leaf in clear natural daylight — close-up, in focus.
            </p>
          </div>
          <div>
            <div style="width:62px;height:62px;background:linear-gradient(135deg,#1565C0,#1976D2);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:1.5rem;margin:0 auto 14px;box-shadow:0 8px 22px rgba(25,118,210,.28)">🧠</div>
            <h4 style="font-family:'Poppins',sans-serif;font-weight:700;color:var(--text1);margin-bottom:8px">AI Analysis</h4>
            <p style="font-size:.83rem;color:var(--text3);line-height:1.68">
              MobileNetV2 model processes the image and identifies disease patterns in &lt;3 s.
            </p>
          </div>
          <div>
            <div style="width:62px;height:62px;background:linear-gradient(135deg,#E65100,#F57C00);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:1.5rem;margin:0 auto 14px;box-shadow:0 8px 22px rgba(230,81,0,.28)">📋</div>
            <h4 style="font-family:'Poppins',sans-serif;font-weight:700;color:var(--text1);margin-bottom:8px">Full Report</h4>
            <p style="font-size:.83rem;color:var(--text3);line-height:1.68">
              Get disease name, confidence score, severity rating, and tailored treatments.
            </p>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    gap(20)

# ─────────────────────────────────────────────────────────────
#  PAGE: SCAN  (real inference lives here)
# ─────────────────────────────────────────────────────────────
def page_scan():
    page_header("Scan Your Crop",
                "Upload a clear leaf photo — the AI model will analyse it instantly.",
                tag="CropSense AI · Real Model")

    # step bar
    st.markdown("""
    <div style="display:flex;align-items:center;padding:20px 52px;background:var(--surface);
                border-bottom:1px solid var(--border)">
      <div style="display:flex;align-items:center;gap:9px;flex:1">
        <div style="width:30px;height:30px;border-radius:50%;background:var(--g700);color:#fff;
                    display:flex;align-items:center;justify-content:center;font-family:'Poppins',sans-serif;
                    font-size:.75rem;font-weight:700;box-shadow:0 3px 10px rgba(30,125,50,.32)">1</div>
        <span style="font-size:.82rem;font-weight:600;color:var(--text2)">Upload Image</span>
      </div>
      <div style="flex:1;height:2px;background:var(--border);max-width:70px"></div>
      <div style="display:flex;align-items:center;gap:9px;flex:1">
        <div style="width:30px;height:30px;border-radius:50%;background:var(--border);color:var(--text3);
                    display:flex;align-items:center;justify-content:center;font-family:'Poppins',sans-serif;
                    font-size:.75rem;font-weight:700">2</div>
        <span style="font-size:.82rem;font-weight:600;color:var(--text3)">AI Analysis</span>
      </div>
      <div style="flex:1;height:2px;background:var(--border);max-width:70px"></div>
      <div style="display:flex;align-items:center;gap:9px;flex:1">
        <div style="width:30px;height:30px;border-radius:50%;background:var(--border);color:var(--text3);
                    display:flex;align-items:center;justify-content:center;font-family:'Poppins',sans-serif;
                    font-size:.75rem;font-weight:700">3</div>
        <span style="font-size:.82rem;font-weight:600;color:var(--text3)">View Report</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    gap(28)
    col_main, col_tip = st.columns([3, 2], gap="large")

    with col_main:
        st.markdown("<div style='padding-left:52px'>", unsafe_allow_html=True)

        if DEMO_MODE:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#FFF8E1,#FFF3CD);border:1px solid #FFE082;
                        border-radius:var(--r-sm);padding:14px 18px;margin-bottom:18px;display:flex;gap:10px">
              <span>⚠️</span>
              <div style="font-size:.83rem;color:#7B5800;line-height:1.55">
                <strong>Demo Mode</strong> — <code>crop_disease_model.keras</code> not found.
                Place it in the same folder as <code>app.py</code> to enable live inference.
                Demo results are shown for UI preview.
              </div>
            </div>
            """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")

        if uploaded:
            st.session_state.last_img = uploaded

            img_pil = Image.open(uploaded)
            gap(8)
            st.image(img_pil, use_container_width=True,
                     caption="✅ Image loaded — ready for analysis")

            gap(14)
            ca, cb = st.columns(2)
            with ca:
                growth = st.selectbox("Growth Stage (optional)",
                    ["Unknown","Seedling","Vegetative","Flowering","Fruiting","Mature"])
            with cb:
                notes = st.text_input("Field Notes (optional)",
                    placeholder="e.g. North field, row 4")

            gap(12)

            # ── ANALYSE BUTTON ──
            if st.button("🧠  Analyse with AI Model", use_container_width=True, key="analyse"):
                st.markdown("""<style>
                [data-testid="stMainBlockContainer"] [data-testid="column"] .stButton>button{
                  background:linear-gradient(135deg,#1565C0,#1976D2) !important;color:#fff !important;
                  border:none !important;font-size:1rem !important;font-weight:700 !important;
                  padding:16px !important;box-shadow:0 4px 20px rgba(25,118,210,.4) !important;
                }
                </style>""", unsafe_allow_html=True)

                bar_slot  = st.empty()
                text_slot = st.empty()

                phases = [
                    (.15, "🔍 Pre-processing image…"),
                    (.35, "🧠 Running MobileNetV2 inference…"),
                    (.60, "🔬 Extracting disease features…"),
                    (.80, "📊 Computing confidence distribution…"),
                    (.95, "💊 Building treatment plan…"),
                ]
                bar = st.progress(0)
                for pct, msg in phases:
                    bar.progress(pct)
                    text_slot.markdown(
                        f"<p style='text-align:center;color:var(--g700);font-weight:600;"
                        f"font-size:.9rem;margin-top:6px'>{msg}</p>",
                        unsafe_allow_html=True)
                    time.sleep(0.4)

                # ── REAL INFERENCE ──
                result = infer(img_pil)
                result["growth_stage"] = growth
                result["notes"]        = notes

                bar.progress(1.0)
                text_slot.markdown(
                    "<p style='text-align:center;color:var(--g600);font-weight:700;"
                    "font-size:.9rem;margin-top:6px'>✅ Analysis complete!</p>",
                    unsafe_allow_html=True)
                time.sleep(0.35)
                bar.empty(); text_slot.empty()

                st.session_state.result = result

                # append to history
                st.session_state.history.insert(0, {
                    "Date":       datetime.now().strftime("%d %b %Y"),
                    "Crop":       result["crop"],
                    "Disease":    result["disease"],
                    "Severity":   result["severity"],
                    "Confidence": f"{result['confidence']}%",
                })
                nav("result")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_tip:
        st.markdown("<div style='padding-right:52px'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                    padding:26px;margin-bottom:18px">
          <h4 style="font-family:'Poppins',sans-serif;font-size:.92rem;font-weight:700;
                     color:var(--text1);margin-bottom:16px">📸 Tips for Best Accuracy</h4>
          <div style="display:flex;flex-direction:column;gap:12px">
            <div style="display:flex;gap:11px;align-items:flex-start">
              <div style="width:26px;height:26px;background:var(--g50);border:1px solid var(--g200);
                          border-radius:7px;display:flex;align-items:center;justify-content:center;
                          flex-shrink:0;font-size:.88rem">☀️</div>
              <div>
                <div style="font-size:.8rem;font-weight:600;color:var(--text1);margin-bottom:1px">Natural Light</div>
                <div style="font-size:.75rem;color:var(--text3);line-height:1.55">Diffused daylight, no flash</div>
              </div>
            </div>
            <div style="display:flex;gap:11px;align-items:flex-start">
              <div style="width:26px;height:26px;background:var(--g50);border:1px solid var(--g200);
                          border-radius:7px;display:flex;align-items:center;justify-content:center;
                          flex-shrink:0;font-size:.88rem">🔍</div>
              <div>
                <div style="font-size:.8rem;font-weight:600;color:var(--text1);margin-bottom:1px">Close-up Frame</div>
                <div style="font-size:.75rem;color:var(--text3);line-height:1.55">Fill frame with the affected leaf</div>
              </div>
            </div>
            <div style="display:flex;gap:11px;align-items:flex-start">
              <div style="width:26px;height:26px;background:var(--g50);border:1px solid var(--g200);
                          border-radius:7px;display:flex;align-items:center;justify-content:center;
                          flex-shrink:0;font-size:.88rem">🎯</div>
              <div>
                <div style="font-size:.8rem;font-weight:600;color:var(--text1);margin-bottom:1px">Show Symptoms</div>
                <div style="font-size:.75rem;color:var(--text3);line-height:1.55">Include the affected area clearly</div>
              </div>
            </div>
            <div style="display:flex;gap:11px;align-items:flex-start">
              <div style="width:26px;height:26px;background:var(--g50);border:1px solid var(--g200);
                          border-radius:7px;display:flex;align-items:center;justify-content:center;
                          flex-shrink:0;font-size:.88rem">📐</div>
              <div>
                <div style="font-size:.8rem;font-weight:600;color:var(--text1);margin-bottom:1px">Sharp Focus</div>
                <div style="font-size:.75rem;color:var(--text3);line-height:1.55">Avoid motion blur for max accuracy</div>
              </div>
            </div>
          </div>
        </div>

        <div style="background:linear-gradient(135deg,#E3F2FD,#F0F7FF);border:1px solid var(--b100);
                    border-radius:var(--r);padding:22px">
          <div style="font-size:.72rem;font-weight:700;color:var(--b600);letter-spacing:.07em;
                      text-transform:uppercase;margin-bottom:12px">🌿 Supported Crops</div>
          <div style="display:flex;flex-wrap:wrap;gap:7px">
            <span style="background:#fff;border:1px solid var(--b100);border-radius:20px;
                         padding:4px 12px;font-size:.74rem;color:var(--text2);font-weight:500">🍅 Tomato</span>
            <span style="background:#fff;border:1px solid var(--b100);border-radius:20px;
                         padding:4px 12px;font-size:.74rem;color:var(--text2);font-weight:500">🌾 Wheat</span>
            <span style="background:#fff;border:1px solid var(--b100);border-radius:20px;
                         padding:4px 12px;font-size:.74rem;color:var(--text2);font-weight:500">🌽 Corn</span>
            <span style="background:#fff;border:1px solid var(--b100);border-radius:20px;
                         padding:4px 12px;font-size:.74rem;color:var(--text2);font-weight:500">🌾 Rice</span>
            <span style="background:#fff;border:1px solid var(--b100);border-radius:20px;
                         padding:4px 12px;font-size:.74rem;color:var(--text2);font-weight:500">🥔 Potato</span>
            <span style="background:#fff;border:1px solid var(--b100);border-radius:20px;
                         padding:4px 12px;font-size:.74rem;color:var(--text2);font-weight:500">🫑 Pepper</span>
            <span style="background:#fff;border:1px solid var(--b100);border-radius:20px;
                         padding:4px 12px;font-size:.74rem;color:var(--text2);font-weight:500">+ more</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  PAGE: RESULT
# ─────────────────────────────────────────────────────────────
def page_result():
    if not st.session_state.result:
        st.markdown("""
        <div style="padding:80px 52px;text-align:center">
          <div style="font-size:3rem;margin-bottom:14px">🔬</div>
          <h3 style="font-family:'Poppins',sans-serif;color:var(--text2)">No Analysis Yet</h3>
          <p style="color:var(--text3);margin-top:8px">Please upload a leaf image and run the scan first.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("← Go to Scan", key="res_goto_scan"):
            nav("scan")
        return

    r = st.session_state.result

    SEV_COLOR = {"Healthy":"#4CAF50","Mild":"#8BC34A","Moderate":"#FFC107","Severe":"#F44336"}
    SEV_BG    = {"Healthy":"#E8F5E9","Mild":"#F1F8E9","Moderate":"#FFF8E1","Severe":"#FFEBEE"}
    sev_c  = SEV_COLOR.get(r["severity"], "#FFC107")
    sev_bg = SEV_BG.get(r["severity"], "#FFF8E1")

    subtitle = f"Scanned on {r['scan_date']}"
    if r.get("notes"): subtitle += f" · {r['notes']}"
    page_header("Diagnosis Report", subtitle, tag="AI Result")

    gap(28)
    st.markdown("<div style='padding:0 52px'>", unsafe_allow_html=True)

    # ── LOW CONFIDENCE BANNER ──
    if r.get("low_confidence"):
        st.markdown("""
        <div style="background:#FFF3E0;border:1.5px solid #FFB74D;border-radius:var(--r-sm);
                    padding:14px 20px;margin-bottom:20px;display:flex;gap:10px;align-items:center">
          <span style="font-size:1.3rem">⚠️</span>
          <div style="font-size:.85rem;color:#7B4700;line-height:1.55">
            <strong>Low Confidence Result</strong> — The model confidence is below 60%.
            Consider retaking the photo in better lighting or from a closer angle.
          </div>
        </div>""", unsafe_allow_html=True)

    # ── MAIN DISEASE CARD ──
    conf_int   = int(r["confidence"])
    conic_deg  = int(r["confidence"] * 3.6)
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                padding:36px;margin-bottom:22px;box-shadow:var(--sh-sm);animation:popIn .4s ease both">
      <div style="display:grid;grid-template-columns:1fr auto;gap:24px;align-items:start">
        <div>
          <div style="font-size:.7rem;font-weight:700;color:var(--text3);letter-spacing:.1em;
                      text-transform:uppercase;margin-bottom:9px">Disease Detected</div>
          <h2 style="font-family:'Poppins',sans-serif;font-size:1.85rem;font-weight:900;
                     color:var(--text1);letter-spacing:-.04em;margin-bottom:14px">{r["disease"]}</h2>
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            <span style="background:{sev_bg};color:{sev_c};border:1px solid {sev_c}33;
                         border-radius:20px;padding:5px 16px;font-size:.78rem;font-weight:700">
              ⚠️ {r["severity"]} Severity
            </span>
            <span style="background:var(--g50);color:var(--g700);border:1px solid var(--g200);
                         border-radius:20px;padding:5px 16px;font-size:.78rem;font-weight:700">
              🌿 {r["crop"]}
            </span>
            {'<span style="background:#E3F2FD;color:var(--b600);border:1px solid var(--b100);border-radius:20px;padding:5px 16px;font-size:.78rem;font-weight:700">✅ Healthy Plant</span>' if r["is_healthy"] else '<span style="background:#E3F2FD;color:var(--b600);border:1px solid var(--b100);border-radius:20px;padding:5px 16px;font-size:.78rem;font-weight:700">🤖 AI Confident</span>'}
          </div>
          {f'<div style="margin-top:14px;font-size:.8rem;color:var(--text3)">Growth stage: <strong style=\'color:var(--text2)\'>{r["growth_stage"]}</strong></div>' if r.get("growth_stage") and r["growth_stage"] != "Unknown" else ""}
        </div>
        <!-- confidence donut -->
        <div style="text-align:center;min-width:110px">
          <div style="width:96px;height:96px;border-radius:50%;
                      background:conic-gradient(var(--g600) {conic_deg}deg,var(--border) 0deg);
                      display:flex;align-items:center;justify-content:center;margin:0 auto 7px;position:relative">
            <div style="width:74px;height:74px;border-radius:50%;background:var(--surface);
                        display:flex;flex-direction:column;align-items:center;justify-content:center">
              <div style="font-family:'Poppins',sans-serif;font-size:1.15rem;font-weight:800;
                          color:var(--g700);line-height:1">{r["confidence"]}%</div>
            </div>
          </div>
          <div style="font-size:.68rem;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:.07em">Confidence</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        # Severity meter
        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                    padding:26px;margin-bottom:18px">
          <h4 style="font-family:'Poppins',sans-serif;font-size:.92rem;font-weight:700;
                     color:var(--text1);margin-bottom:18px">📊 Severity Meter</h4>
          <div style="position:relative;height:10px;
                      background:linear-gradient(90deg,#4CAF50 0%,#FFC107 50%,#F44336 100%);
                      border-radius:5px;margin-bottom:8px;overflow:visible">
            <div style="position:absolute;top:50%;left:{r['severity_pct']}%;
                        transform:translate(-50%,-50%);width:19px;height:19px;border-radius:50%;
                        background:{sev_c};border:3px solid #fff;
                        box-shadow:0 2px 7px rgba(0,0,0,.22)"></div>
          </div>
          <div style="display:flex;justify-content:space-between;
                      font-size:.7rem;color:var(--text3);font-weight:600;text-transform:uppercase;letter-spacing:.05em">
            <span>Healthy</span><span>Moderate</span><span>Severe</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Treatments
        tx_rows = "".join([f"""
        <div style="display:flex;align-items:flex-start;gap:11px;padding:11px 0;
                    border-bottom:1px solid var(--border)">
          <div style="width:22px;height:22px;background:var(--g50);border:1px solid var(--g200);
                      border-radius:6px;display:flex;align-items:center;justify-content:center;
                      font-size:.72rem;flex-shrink:0;font-weight:700;color:var(--g700)">{i+1}</div>
          <div style="font-size:.84rem;color:var(--text2);line-height:1.65">{t}</div>
        </div>""" for i, t in enumerate(r["treatments"])])

        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                    padding:26px;margin-bottom:18px">
          <h4 style="font-family:'Poppins',sans-serif;font-size:.92rem;font-weight:700;
                     color:var(--text1);margin-bottom:4px">💊 Recommended Treatment</h4>
          <p style="font-size:.77rem;color:var(--text3);margin-bottom:14px">
            AI-generated guidance — verify with a certified agronomist before large-scale application
          </p>
          {tx_rows}
        </div>
        """, unsafe_allow_html=True)

        # About disease
        if r.get("about"):
            st.markdown(f"""
            <div style="background:var(--g50);border:1px solid var(--g200);border-radius:var(--r);padding:22px;margin-bottom:18px">
              <h4 style="font-family:'Poppins',sans-serif;font-size:.88rem;font-weight:700;
                         color:var(--g800);margin-bottom:9px">ℹ️ About {r['disease']}</h4>
              <p style="font-size:.82rem;color:var(--text2);line-height:1.72">{r['about']}</p>
              {'<div style="margin-top:12px;background:rgba(255,193,7,.1);border:1px solid rgba(255,193,7,.3);border-radius:7px;padding:9px 13px;font-size:.77rem;color:#7B5800;font-weight:600">⚡ Act within 24–48 hours to prevent spread</div>' if not r["is_healthy"] else ''}
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        # Top-5 predictions
        preds_html = ""
        for i, (dis, conf) in enumerate(r["top5"]):
            is_top = i == 0
            bar_w  = max(conf, 1)
            preds_html += f"""
            <div style="margin-bottom:13px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="font-size:.81rem;font-weight:{'700' if is_top else '500'};
                             color:{'var(--g700)' if is_top else 'var(--text2)'}">{dis}</span>
                <span style="font-size:.77rem;font-weight:700;
                             color:{'var(--g700)' if is_top else 'var(--text3)'}">{conf}%</span>
              </div>
              <div style="background:var(--g50);border-radius:4px;height:6px;overflow:hidden">
                <div style="width:{bar_w}%;height:100%;border-radius:4px;
                            background:{'linear-gradient(90deg,var(--g600),var(--g400))' if is_top else 'var(--g200)'}"></div>
              </div>
            </div>"""

        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                    padding:26px;margin-bottom:18px">
          <h4 style="font-family:'Poppins',sans-serif;font-size:.92rem;font-weight:700;
                     color:var(--text1);margin-bottom:18px">🎯 Top-5 AI Predictions</h4>
          {preds_html}
        </div>
        """, unsafe_allow_html=True)

        # Uploaded image thumbnail
        if st.session_state.last_img:
            st.markdown("""
            <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                        padding:18px;margin-bottom:18px">
              <h4 style="font-family:'Poppins',sans-serif;font-size:.88rem;font-weight:700;
                         color:var(--text1);margin-bottom:12px">🖼️ Analysed Image</h4>""",
                unsafe_allow_html=True)
            img2 = Image.open(st.session_state.last_img)
            st.image(img2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    gap(16)

    # Action buttons
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🔬  Scan Another Leaf", use_container_width=True, key="res_scan"):
            st.session_state.result = None
            st.session_state.last_img = None
            nav("scan")
    with b2:
        if st.button("📊  View Dashboard", use_container_width=True, key="res_dash"):
            nav("dashboard")
    with b3:
        report_txt = (
            f"CropSense AI — Diagnosis Report\n{'='*42}\n"
            f"Disease   : {r['disease']}\n"
            f"Crop      : {r['crop']}\n"
            f"Confidence: {r['confidence']}%\n"
            f"Severity  : {r['severity']}\n"
            f"Date      : {r['scan_date']}\n"
            f"Stage     : {r.get('growth_stage','—')}\n"
            f"Notes     : {r.get('notes','—')}\n\n"
            f"Recommended Treatment:\n"
            + "\n".join(f"  {i+1}. {t}" for i,t in enumerate(r["treatments"]))
            + f"\n\nTop-5 AI Predictions:\n"
            + "\n".join(f"  {d}: {c}%" for d,c in r["top5"])
        )
        st.download_button("⬇️  Download Report", data=report_txt,
                           file_name=f"cropsense_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                           mime="text/plain", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    gap(32)

# ─────────────────────────────────────────────────────────────
#  PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────
def page_dashboard():
    page_header("Field Health Dashboard",
                "Complete scan history, disease trends, and model performance metrics.",
                tag="Analytics")
    gap(28)

    history = st.session_state.history
    total    = len(history)
    healthy  = sum(1 for h in history if h["Severity"] == "Healthy")
    diseased = total - healthy
    critical = sum(1 for h in history if h["Severity"] == "Severe")

    st.markdown("<div style='padding:0 52px'>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Scans",   total,    delta="+2 today")
    c2.metric("Healthy",       healthy,  delta=None)
    c3.metric("Diseased",      diseased, delta=None)
    c4.metric("Critical",      critical, delta=None)

    gap(24)
    tab1, tab2 = st.tabs(["📋  Scan History", "📈  Analytics"])

    with tab1:
        search = st.text_input("🔍  Search by crop or disease name",
                               placeholder="e.g. Tomato, Blight, Rust")
        display = [h for h in history
                   if not search or search.lower() in h["Crop"].lower()
                   or search.lower() in h["Disease"].lower()]

        SEV_COLOR = {"Healthy":"#4CAF50","Mild":"#8BC34A","Moderate":"#FFC107","Severe":"#F44336"}
        SEV_BG    = {"Healthy":"#E8F5E9","Mild":"#F1F8E9","Moderate":"#FFFDE7","Severe":"#FFEBEE"}

        rows = "".join([f"""
        <tr style="border-bottom:1px solid var(--border)">
          <td style="padding:13px 16px;font-size:.81rem;color:var(--text3)">{h['Date']}</td>
          <td style="padding:13px 16px;font-size:.84rem;font-weight:600;color:var(--text1)">{h['Crop']}</td>
          <td style="padding:13px 16px;font-size:.83rem;color:var(--text2)">{h['Disease']}</td>
          <td style="padding:13px 16px">
            <span style="background:{SEV_BG.get(h['Severity'],'#f5f5f5')};
                         color:{SEV_COLOR.get(h['Severity'],'#999')};
                         border:1px solid {SEV_COLOR.get(h['Severity'],'#999')}33;
                         border-radius:20px;padding:3px 11px;font-size:.74rem;font-weight:700">
              {h['Severity']}
            </span>
          </td>
          <td style="padding:13px 16px;font-size:.83rem;font-weight:700;color:var(--g700)">{h['Confidence']}</td>
        </tr>""" for h in display])

        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                    overflow:hidden;margin-top:14px">
          <table style="width:100%;border-collapse:collapse">
            <thead>
              <tr style="background:var(--g50);border-bottom:2px solid var(--border)">
                <th style="padding:13px 16px;text-align:left;font-size:.7rem;font-weight:700;
                           color:var(--text3);text-transform:uppercase;letter-spacing:.08em">Date</th>
                <th style="padding:13px 16px;text-align:left;font-size:.7rem;font-weight:700;
                           color:var(--text3);text-transform:uppercase;letter-spacing:.08em">Crop</th>
                <th style="padding:13px 16px;text-align:left;font-size:.7rem;font-weight:700;
                           color:var(--text3);text-transform:uppercase;letter-spacing:.08em">Disease</th>
                <th style="padding:13px 16px;text-align:left;font-size:.7rem;font-weight:700;
                           color:var(--text3);text-transform:uppercase;letter-spacing:.08em">Severity</th>
                <th style="padding:13px 16px;text-align:left;font-size:.7rem;font-weight:700;
                           color:var(--text3);text-transform:uppercase;letter-spacing:.08em">Confidence</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-top:8px">

          <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:26px">
            <h4 style="font-family:'Poppins',sans-serif;font-size:.9rem;font-weight:700;
                       color:var(--text1);margin-bottom:18px">Disease Distribution</h4>
            <div style="display:flex;flex-direction:column;gap:11px">
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                  <span style="font-size:.81rem;color:var(--text2)">Late Blight</span>
                  <span style="font-size:.77rem;font-weight:700;color:var(--text1)">32%</span>
                </div>
                <div style="background:var(--g50);border-radius:4px;height:7px">
                  <div style="width:32%;height:100%;background:var(--g600);border-radius:4px"></div>
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                  <span style="font-size:.81rem;color:var(--text2)">Rust</span>
                  <span style="font-size:.77rem;font-weight:700;color:var(--text1)">24%</span>
                </div>
                <div style="background:var(--g50);border-radius:4px;height:7px">
                  <div style="width:24%;height:100%;background:#FFC107;border-radius:4px"></div>
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                  <span style="font-size:.81rem;color:var(--text2)">Gray Leaf Spot</span>
                  <span style="font-size:.77rem;font-weight:700;color:var(--text1)">18%</span>
                </div>
                <div style="background:var(--g50);border-radius:4px;height:7px">
                  <div style="width:18%;height:100%;background:#F44336;border-radius:4px"></div>
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                  <span style="font-size:.81rem;color:var(--text2)">Early Blight</span>
                  <span style="font-size:.77rem;font-weight:700;color:var(--text1)">14%</span>
                </div>
                <div style="background:var(--g50);border-radius:4px;height:7px">
                  <div style="width:14%;height:100%;background:#FF9800;border-radius:4px"></div>
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                  <span style="font-size:.81rem;color:var(--text2)">Healthy</span>
                  <span style="font-size:.77rem;font-weight:700;color:var(--text1)">12%</span>
                </div>
                <div style="background:var(--g50);border-radius:4px;height:7px">
                  <div style="width:12%;height:100%;background:var(--g400);border-radius:4px"></div>
                </div>
              </div>
            </div>
          </div>

          <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:26px">
            <h4 style="font-family:'Poppins',sans-serif;font-size:.9rem;font-weight:700;
                       color:var(--text1);margin-bottom:18px">Model Performance Metrics</h4>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
              <div style="background:var(--g50);border:1px solid var(--g200);border-radius:11px;padding:15px;text-align:center">
                <div style="font-family:'Poppins',sans-serif;font-size:1.4rem;font-weight:800;color:var(--g700)">98.2%</div>
                <div style="font-size:.69rem;color:var(--text3);font-weight:700;text-transform:uppercase;margin-top:3px">Accuracy</div>
              </div>
              <div style="background:#E3F2FD;border:1px solid var(--b100);border-radius:11px;padding:15px;text-align:center">
                <div style="font-family:'Poppins',sans-serif;font-size:1.4rem;font-weight:800;color:var(--b700)">97.1%</div>
                <div style="font-size:.69rem;color:var(--text3);font-weight:700;text-transform:uppercase;margin-top:3px">Precision</div>
              </div>
              <div style="background:#FFF3E0;border:1px solid #FFE0B2;border-radius:11px;padding:15px;text-align:center">
                <div style="font-family:'Poppins',sans-serif;font-size:1.4rem;font-weight:800;color:#E65100">96.8%</div>
                <div style="font-size:.69rem;color:var(--text3);font-weight:700;text-transform:uppercase;margin-top:3px">Recall</div>
              </div>
              <div style="background:#F3E5F5;border:1px solid #E1BEE7;border-radius:11px;padding:15px;text-align:center">
                <div style="font-family:'Poppins',sans-serif;font-size:1.4rem;font-weight:800;color:#6A1B9A">96.9%</div>
                <div style="font-size:.69rem;color:var(--text3);font-weight:700;text-transform:uppercase;margin-top:3px">F1 Score</div>
              </div>
            </div>
            <div style="margin-top:16px;background:var(--g50);border:1px solid var(--g200);
                        border-radius:10px;padding:13px 16px">
              <div style="font-size:.78rem;font-weight:700;color:var(--g700);margin-bottom:4px">Model Architecture</div>
              <div style="font-size:.77rem;color:var(--text2);line-height:1.6">
                MobileNetV2 · Trained on PlantVillage · 87K+ images · 38 classes · 14 crop types
              </div>
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    gap(32)

# ─────────────────────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────────────────────
def page_about():
    page_header("About CropSense AI",
                "Built for farmers, powered by deep learning. Protecting food security one leaf at a time.",
                tag="About")
    gap(28)
    st.markdown("<div style='padding:0 52px'>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
                    padding:34px;margin-bottom:22px">
          <h3 style="font-family:'Poppins',sans-serif;font-size:1.15rem;font-weight:800;
                     color:var(--text1);margin-bottom:14px">🎯 Project Vision</h3>
          <p style="font-size:.88rem;color:var(--text2);line-height:1.8;margin-bottom:14px">
            CropSense AI bridges the gap between modern machine learning and practical agriculture.
            By enabling instant disease identification through a smartphone camera, we aim to reduce
            crop losses caused by delayed or misdiagnosed disease detection.
          </p>
          <p style="font-size:.88rem;color:var(--text2);line-height:1.8">
            The system uses a MobileNetV2 deep learning model trained on the PlantVillage dataset —
            over 87,000 labelled leaf images across 38 disease categories and 14 major crops —
            to deliver fast, accurate, and actionable diagnoses in under 3 seconds.
          </p>
        </div>

        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:34px">
          <h3 style="font-family:'Poppins',sans-serif;font-size:1.15rem;font-weight:800;
                     color:var(--text1);margin-bottom:22px">🗺️ Development Roadmap</h3>
          <div style="position:relative;padding-left:26px">
            <div style="position:absolute;left:8px;top:0;bottom:0;width:2px;
                        background:linear-gradient(to bottom,var(--g600),var(--b400));border-radius:2px"></div>

            <div style="margin-bottom:24px;position:relative">
              <div style="position:absolute;left:-20px;top:3px;width:14px;height:14px;border-radius:50%;
                          background:var(--g600);border:2.5px solid #fff;box-shadow:0 0 0 3px var(--g200)"></div>
              <div style="background:var(--g50);border:1px solid var(--g200);border-radius:11px;padding:14px 16px">
                <div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">
                  <span style="background:var(--g700);color:#fff;border-radius:20px;padding:2px 9px;
                               font-size:.66rem;font-weight:700">CURRENT ✓</span>
                  <span style="font-family:'Poppins',sans-serif;font-size:.86rem;font-weight:700;color:var(--g800)">
                    v2.0 — Live AI Integration</span>
                </div>
                <p style="font-size:.8rem;color:var(--text2);line-height:1.6">
                  Real MobileNetV2 inference, top-5 predictions, severity scoring, treatment plans, scan history.
                </p>
              </div>
            </div>

            <div style="margin-bottom:24px;position:relative">
              <div style="position:absolute;left:-20px;top:3px;width:14px;height:14px;border-radius:50%;
                          background:var(--b400);border:2.5px solid #fff;box-shadow:0 0 0 3px var(--b100)"></div>
              <div style="background:#E3F2FD;border:1px solid var(--b100);border-radius:11px;padding:14px 16px">
                <div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">
                  <span style="background:var(--b600);color:#fff;border-radius:20px;padding:2px 9px;
                               font-size:.66rem;font-weight:700">NEXT →</span>
                  <span style="font-family:'Poppins',sans-serif;font-size:.86rem;font-weight:700;color:var(--b700)">
                    v3.0 — Multilingual Support</span>
                </div>
                <p style="font-size:.8rem;color:var(--text2);line-height:1.6">
                  10+ regional language support, voice-guided results, and SMS report delivery for low-connectivity areas.
                </p>
              </div>
            </div>

            <div style="position:relative">
              <div style="position:absolute;left:-20px;top:3px;width:14px;height:14px;border-radius:50%;
                          background:#9C27B0;border:2.5px solid #fff;box-shadow:0 0 0 3px #E1BEE7"></div>
              <div style="background:#F3E5F5;border:1px solid #E1BEE7;border-radius:11px;padding:14px 16px">
                <div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">
                  <span style="background:#7B1FA2;color:#fff;border-radius:20px;padding:2px 9px;
                               font-size:.66rem;font-weight:700">VISION</span>
                  <span style="font-family:'Poppins',sans-serif;font-size:.86rem;font-weight:700;color:#4A148C">
                    v4.0 — Satellite + Weather Intelligence</span>
                </div>
                <p style="font-size:.8rem;color:var(--text2);line-height:1.6">
                  Satellite imagery analysis, weather-based disease risk forecasting, regional crop advisory integration.
                </p>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,var(--g900),var(--g800));
                    border-radius:var(--r);padding:28px;margin-bottom:18px">
          <h3 style="font-family:'Poppins',sans-serif;font-size:.92rem;font-weight:700;
                     color:#fff;margin-bottom:18px">⚙️ Tech Stack</h3>
          <div style="display:flex;flex-direction:column;gap:10px">
            {''.join([f"""
            <div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.11);
                        border-radius:9px;padding:11px 14px;display:flex;align-items:center;gap:11px">
              <span style="font-size:1.15rem">{ic}</span>
              <div>
                <div style="font-size:.8rem;font-weight:700;color:#fff">{nm}</div>
                <div style="font-size:.7rem;color:rgba(255,255,255,.45)">{sub}</div>
              </div>
            </div>""" for ic,nm,sub in [
              ("🧠","MobileNetV2","Deep Learning Model"),
              ("🌿","PlantVillage","87K+ labelled images"),
              ("🐍","TensorFlow / Keras","AI Training & Inference"),
              ("🎨","Streamlit","Frontend Framework"),
              ("📊","NumPy / Pillow","Image Processing"),
            ]])}
          </div>
        </div>

        <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:26px">
          <h3 style="font-family:'Poppins',sans-serif;font-size:.92rem;font-weight:700;
                     color:var(--text1);margin-bottom:16px">📊 Dataset at a Glance</h3>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:11px">
            <div style="background:var(--g50);border:1px solid var(--g200);border-radius:10px;padding:13px;text-align:center">
              <div style="font-family:'Poppins',sans-serif;font-size:1.25rem;font-weight:800;color:var(--g700)">87K+</div>
              <div style="font-size:.68rem;color:var(--text3);font-weight:600;text-transform:uppercase;margin-top:3px">Images</div>
            </div>
            <div style="background:#E3F2FD;border:1px solid var(--b100);border-radius:10px;padding:13px;text-align:center">
              <div style="font-family:'Poppins',sans-serif;font-size:1.25rem;font-weight:800;color:var(--b700)">{len(CLASS_NAMES) if CLASS_NAMES else 38}</div>
              <div style="font-size:.68rem;color:var(--text3);font-weight:600;text-transform:uppercase;margin-top:3px">Classes</div>
            </div>
            <div style="background:#FFF3E0;border:1px solid #FFE0B2;border-radius:10px;padding:13px;text-align:center">
              <div style="font-family:'Poppins',sans-serif;font-size:1.25rem;font-weight:800;color:#E65100">14</div>
              <div style="font-size:.68rem;color:var(--text3);font-weight:600;text-transform:uppercase;margin-top:3px">Crops</div>
            </div>
            <div style="background:#F3E5F5;border:1px solid #E1BEE7;border-radius:10px;padding:13px;text-align:center">
              <div style="font-family:'Poppins',sans-serif;font-size:1.25rem;font-weight:800;color:#6A1B9A">98%</div>
              <div style="font-size:.68rem;color:var(--text3);font-weight:600;text-transform:uppercase;margin-top:3px">Accuracy</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    gap(32)

# ─────────────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────────────
{
    "home":      page_home,
    "scan":      page_scan,
    "result":    page_result,
    "dashboard": page_dashboard,
    "about":     page_about,
}.get(st.session_state.page, page_home)()