import streamlit as st
import time
import random
from datetime import datetime, timedelta
import base64
from pathlib import Path

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense AI · Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --green-900: #0a2e14;
  --green-800: #1B5E20;
  --green-700: #1E7D32;
  --green-600: #2E7D32;
  --green-500: #388E3C;
  --green-400: #4CAF50;
  --green-200: #A5D6A7;
  --green-100: #C8E6C9;
  --green-50:  #F1F8F2;
  --blue-700:  #1565C0;
  --blue-600:  #1976D2;
  --blue-400:  #42A5F5;
  --blue-100:  #BBDEFB;
  --amber:     #FFC107;
  --red:       #F44336;
  --bg:        #F4F7F4;
  --surface:   #FFFFFF;
  --surface-2: #F8FAF8;
  --border:    #E0EBE0;
  --text-1:    #0D1F0E;
  --text-2:    #3A5A3C;
  --text-3:    #6B8F6D;
  --radius-sm: 10px;
  --radius:    18px;
  --radius-lg: 28px;
  --shadow-sm: 0 2px 8px rgba(30,125,50,.08);
  --shadow:    0 8px 32px rgba(30,125,50,.12);
  --shadow-lg: 0 20px 60px rgba(30,125,50,.18);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text-1);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stAppViewContainer"] > .main > .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--green-200); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--green-900) !important;
  border-right: 1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] .stButton > button {
  background: rgba(255,255,255,.08) !important;
  border: 1px solid rgba(255,255,255,.15) !important;
  color: #fff !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: all .2s;
  width: 100%;
  text-align: left;
  padding: 12px 16px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,255,255,.15) !important;
  transform: translateX(3px);
}

/* ── Generic button reset ── */
.stButton > button {
  font-family: 'DM Sans', sans-serif !important;
  border-radius: var(--radius-sm) !important;
  transition: all .22s cubic-bezier(.4,0,.2,1) !important;
  font-weight: 600 !important;
  cursor: pointer;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 2px dashed var(--green-200) !important;
  border-radius: var(--radius) !important;
  padding: 24px !important;
  transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--green-600) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div { background: var(--green-600) !important; border-radius: 4px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface-2);
  border-radius: var(--radius-sm);
  padding: 4px;
  gap: 4px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  color: var(--text-2) !important;
  padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
  background: var(--green-700) !important;
  color: #fff !important;
}

/* ── Select / Input ── */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput > div > div > input {
  border-radius: var(--radius-sm) !important;
  border-color: var(--border) !important;
  font-family: 'DM Sans', sans-serif !important;
  background: var(--surface) !important;
}

/* ── Metric ── */
[data-testid="metric-container"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px !important;
  box-shadow: var(--shadow-sm);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-family: 'DM Sans', sans-serif;
  color: var(--text-3) !important;
  font-size: .8rem !important;
  text-transform: uppercase;
  letter-spacing: .06em;
  font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Poppins', sans-serif !important;
  color: var(--text-1) !important;
  font-weight: 700 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden;
}

/* ── Spinner ── */
.stSpinner > div { border-color: var(--green-600) transparent transparent transparent !important; }

/* ── Toast / Success / Info ── */
.stSuccess, .stInfo, .stWarning, .stError {
  border-radius: var(--radius-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Animations ── */
@keyframes fadeUp {
  from { opacity:0; transform:translateY(24px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes pulse-ring {
  0%   { transform:scale(1);   opacity:.7; }
  70%  { transform:scale(1.15); opacity:0; }
  100% { transform:scale(1);   opacity:0; }
}
@keyframes scan-line {
  0%   { top: 5%; }
  50%  { top: 90%; }
  100% { top: 5%; }
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
@keyframes countUp {
  from { opacity: 0; transform: scale(.8); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes float {
  0%, 100% { transform: translateY(0); }
  50%       { transform: translateY(-8px); }
}

.home-hero {
  padding: 72px 48px 60px;
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 40px;
  align-items: center;
  background: linear-gradient(180deg, #F7FBF7 0%, #EAF4E7 100%);
  border: 1px solid rgba(56,142,60,.16);
  border-radius: 32px;
}
.home-hero-copy {
  z-index: 1;
}
.hero-tag {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(76,175,80,.12);
  border: 1px solid rgba(76,175,80,.2);
  border-radius: 999px;
  padding: 10px 18px;
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #1B5E20;
  margin-bottom: 18px;
}
.hero-headline {
  font-family: 'Poppins', sans-serif;
  font-size: clamp(2.6rem, 5vw, 4.2rem);
  line-height: 1.02;
  font-weight: 900;
  letter-spacing: -0.04em;
  color: #0E3620;
  margin-bottom: 22px;
  max-width: 620px;
}
.hero-copy {
  font-size: 1rem;
  color: #42523a;
  max-width: 560px;
  line-height: 1.7;
  margin-bottom: 32px;
}
.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-bottom: 32px;
}
.hero-action {
  border: none;
  border-radius: 14px;
  padding: 15px 30px;
  font-family: 'Poppins', sans-serif;
  font-weight: 700;
  font-size: .94rem;
  cursor: pointer;
  transition: transform .22s ease, box-shadow .22s ease, opacity .22s ease;
}
.hero-action.primary {
  background: linear-gradient(135deg, #81C784, #1B5E20);
  color: #fff;
  box-shadow: 0 18px 45px rgba(30,125,50,.22);
}
.hero-action.secondary {
  background: #fff;
  color: #1B5E20;
  border: 1px solid rgba(27,94,32,.18);
}
.hero-action:hover {
  transform: translateY(-2px);
}
.hero-stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0,1fr));
  gap: 16px;
}
.hero-stat {
  background: #fff;
  border: 1px solid rgba(27,94,32,.12);
  border-radius: 18px;
  padding: 18px 20px;
  text-align: center;
}
.hero-stat .value {
  font-family: 'Poppins', sans-serif;
  font-weight: 800;
  font-size: 1.7rem;
  color: #1B5E20;
}
.hero-stat .label {
  font-size: .78rem;
  color: #4e6545;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-top: 6px;
}
.home-hero-panel {
  position: relative;
  border-radius: 32px;
  overflow: hidden;
  box-shadow: 0 35px 80px rgba(30,95,40,.14);
  min-height: 520px;
  background: #fff;
  padding: 34px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.hero-panel h3 {
  font-family: 'Poppins', sans-serif;
  font-size: 1.1rem;
  font-weight: 800;
  color: #1B5E20;
  margin-bottom: 12px;
}
.hero-panel .panel-block {
  background: #F4FBF5;
  border: 1px solid rgba(27,94,32,.08);
  border-radius: 22px;
  padding: 18px;
  margin-bottom: 18px;
}
.hero-panel .panel-label {
  font-size: .78rem;
  color: #4e6545;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: .08em;
}
.hero-panel .panel-value {
  font-family: 'Poppins', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  color: #1B5E20;
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 22px;
  max-width: 1200px;
  margin: 0 auto;
}
.feature-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 30px 28px;
  box-shadow: var(--shadow-sm);
  transition: transform .25s ease, box-shadow .25s ease;
}
.feature-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 24px 70px rgba(30,125,50,.12);
}
.feature-card .feature-icon {
  width: 56px;
  height: 56px;
  border-radius: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.4rem;
  margin-bottom: 18px;
}
.feature-card h3 {
  margin-bottom: 14px;
  font-family: 'Poppins', sans-serif;
  font-size: 1.05rem;
  font-weight: 800;
  color: var(--text-1);
}
.feature-card p {
  color: var(--text-3);
  line-height: 1.75;
  margin-bottom: 18px;
}
.feature-card .feature-action {
  font-size: .82rem;
  font-weight: 700;
  color: var(--green-700);
}
.workflow-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 22px;
  max-width: 1200px;
  margin: 0 auto;
}
.workflow-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 30px 28px;
  text-align: center;
}
.workflow-card .workflow-icon {
  width: 60px;
  height: 60px;
  border-radius: 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  margin-bottom: 18px;
}
.workflow-card h4 {
  margin-bottom: 12px;
  font-family: 'Poppins', sans-serif;
  font-size: 1rem;
  font-weight: 800;
  color: var(--text-1);
}
.workflow-card p {
  color: var(--text-3);
  line-height: 1.7;
}
@media (max-width: 980px) {
  .home-hero, .feature-grid, .workflow-grid { grid-template-columns: 1fr; }
  .hero-stats { grid-template-columns: 1fr; }
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "home",
        "uploaded_image": None,
        "analysis_done": False,
        "result": None,
        "history": _generate_history(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _generate_history():
    diseases = [
        ("Tomato", "Late Blight",      "Severe",   91.2),
        ("Wheat",  "Rust",             "Moderate", 87.5),
        ("Corn",   "Healthy",          "Healthy",  96.1),
        ("Rice",   "Blast Disease",    "Moderate", 83.4),
        ("Potato", "Early Blight",     "Mild",     89.7),
        ("Tomato", "Leaf Mold",        "Mild",     78.3),
        ("Corn",   "Gray Leaf Spot",   "Severe",   92.6),
        ("Wheat",  "Healthy",          "Healthy",  95.0),
    ]
    rows = []
    for i, (crop, disease, severity, conf) in enumerate(diseases):
        date = datetime.now() - timedelta(days=i*3 + random.randint(0,2))
        rows.append({
            "Date":       date.strftime("%d %b %Y"),
            "Crop":       crop,
            "Disease":    disease,
            "Severity":   severity,
            "Confidence": f"{conf}%",
        })
    return rows

def nav(page):
    st.session_state.page = page
    st.rerun()

init_state()

# ─────────────────────────────────────────────
#  SIDEBAR NAV
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 8px 32px; text-align:center;">
      <div style="font-family:'Poppins',sans-serif; font-size:1.3rem; font-weight:800;
                  background:linear-gradient(135deg,#81C784,#A5D6A7);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  letter-spacing:-.02em; margin-bottom:4px;">
        🌿 CropSense AI
      </div>
      <div style="font-size:.72rem; color:rgba(255,255,255,.45); letter-spacing:.1em; text-transform:uppercase;">
        Agri Intelligence Platform
      </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("🏠", "Home",       "home"),
        ("🔬", "Scan Crop",  "scan"),
        ("📋", "Results",    "result"),
        ("📊", "Dashboard",  "dashboard"),
        ("💡", "About",      "about"),
    ]
    for icon, label, key in nav_items:
        active = "✦ " if st.session_state.page == key else "   "
        if st.button(f"{icon}  {active}{label}", key=f"nav_{key}"):
            nav(key)

    st.markdown("<div style='padding:14px 8px; font-size:.75rem; color:rgba(255,255,255,.45); text-align:center; line-height:1.6;'>Open the left sidebar to navigate on mobile. Collapse it to save space.</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:rgba(255,255,255,.08);margin:20px 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0 8px; font-size:.72rem; color:rgba(255,255,255,.3); text-align:center; line-height:1.7;">
      v1.0 Prototype<br>AI Model: MobileNetV2<br>Dataset: PlantVillage
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPER COMPONENTS
# ─────────────────────────────────────────────
def page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="
      padding: 40px 48px 32px;
      background: linear-gradient(135deg, var(--green-900) 0%, var(--green-800) 60%, #1a4a22 100%);
      position: relative; overflow: hidden;
    ">
      <div style="
        position:absolute; top:-60px; right:-60px;
        width:260px; height:260px; border-radius:50%;
        background: radial-gradient(circle, rgba(76,175,80,.15) 0%, transparent 70%);
        pointer-events:none;
      "></div>
      <div style="
        position:absolute; bottom:-40px; left:30%;
        width:180px; height:180px; border-radius:50%;
        background: radial-gradient(circle, rgba(25,118,210,.1) 0%, transparent 70%);
        pointer-events:none;
      "></div>
      <div style="position:relative; z-index:1; animation: fadeUp .5s ease both;">
        <div style="
          display:inline-flex; align-items:center; gap:8px;
          background:rgba(76,175,80,.18); border:1px solid rgba(76,175,80,.3);
          border-radius:40px; padding:4px 14px; margin-bottom:14px;
          font-size:.73rem; font-weight:600; color:#81C784; letter-spacing:.06em; text-transform:uppercase;
        ">
          ✦ CropSense AI
        </div>
        <h1 style="
          font-family:'Poppins',sans-serif; font-size:clamp(1.6rem,3vw,2.4rem);
          font-weight:800; color:#fff; line-height:1.15;
          letter-spacing:-.03em; margin-bottom:10px;
        ">{title}</h1>
        {f'<p style="font-size:.95rem; color:rgba(255,255,255,.6); max-width:520px; line-height:1.65;">{subtitle}</p>' if subtitle else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)

def card(content_html: str, padding: str = "28px 32px", extra_style: str = ""):
    st.markdown(f"""
    <div style="
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: {padding};
      box-shadow: var(--shadow-sm);
      animation: fadeUp .45s ease both;
      {extra_style}
    ">{content_html}</div>
    """, unsafe_allow_html=True)

def section_gap(px=32):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────
def page_home():
    section_gap(24)

    st.markdown("""
    <div class="home-hero">
      <div class="home-hero-copy">
        <div class="hero-tag">🌿 CropSense AI for Smarter Harvests</div>
        <h1 class="hero-headline">Detect crop disease faster with beautiful, accurate AI.</h1>
        <p class="hero-copy">
          Upload a single leaf photo and receive instant disease diagnosis, severity scoring,
          and treatment recommendations — designed to help farmers make faster, smarter decisions.
        </p>
        <div class="hero-actions">
          <div class="hero-action primary">🔬 Start Disease Scan</div>
          <div class="hero-action secondary">📊 View Dashboard</div>
        </div>
        <div class="hero-stats">
          <div class="hero-stat">
            <div class="value">98.2%</div>
            <div class="label">Model Accuracy</div>
          </div>
          <div class="hero-stat">
            <div class="value">38+</div>
            <div class="label">Disease Classes</div>
          </div>
          <div class="hero-stat">
            <div class="value">&lt;3s</div>
            <div class="label">Analysis Time</div>
          </div>
        </div>
      </div>
      <div class="home-hero-panel">
        <div>
          <h3>Real-time crop health intelligence</h3>
          <div class="panel-block">
            <div class="panel-label">Live scan status</div>
            <div class="panel-value">94.2% confidence</div>
          </div>
          <div class="panel-block">
            <div class="panel-label">Detected disease</div>
            <div class="panel-value">Tomato Late Blight</div>
          </div>
          <div class="panel-block">
            <div class="panel-label">Treatment recommendation</div>
            <div class="panel-value">Copper fungicide + improved airflow</div>
          </div>
        </div>
        <div style="color:#42523a; font-size:.92rem; line-height:1.8;">
          <strong>Designed for on-field use.</strong> A modern interface, smart guidance and
          fast results help make every crop scan feel authoritative and reliable.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        if st.button("🔬  Start Disease Scan", key="hero_scan", use_container_width=True):
            nav("scan")
    with col2:
        if st.button("📊  View Dashboard", key="hero_dash", use_container_width=True):
            nav("dashboard")

    section_gap(42)

    st.markdown("""
    <div style="padding: 0 48px; max-width:1200px; margin:0 auto;">
      <div style="text-align:center; margin-bottom:40px;">
        <div style="font-size:.75rem; font-weight:700; color:var(--green-600); letter-spacing:.12em;
                    text-transform:uppercase; margin-bottom:10px;">Core Capabilities</div>
        <h2 style="font-family:'Poppins',sans-serif; font-size:2.2rem; font-weight:800;
                   color:var(--text-1); letter-spacing:-.03em; margin:0;">
          Everything You Need to Protect Your Crops
        </h2>
      </div>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-icon" style="background:linear-gradient(135deg,#E8F5E9,#C8E6C9);">📸</div>
          <h3>Instant Diagnosis</h3>
          <p>Upload any leaf photo and get AI-powered disease identification in seconds.</p>
          <div class="feature-action">Learn more →</div>
        </div>
        <div class="feature-card" style="background: linear-gradient(135deg,#1E7D32,#1B5E20); color:#fff; border:none; box-shadow:0 24px 70px rgba(0,0,0,.15);">
          <div class="feature-icon" style="background:rgba(255,255,255,.15);">💊</div>
          <h3 style="color:#fff;">Treatment Plans</h3>
          <p style="color:rgba(255,255,255,.8);">Actionable fungicide, pesticide and cultural guidance tailored to the disease.</p>
          <div class="feature-action" style="color:#C8E6C9;">Learn more →</div>
        </div>
        <div class="feature-card">
          <div class="feature-icon" style="background:linear-gradient(135deg,#E3F2FD,#BBDEFB);">📈</div>
          <h3>History & Analytics</h3>
          <p>Track scan outcomes, compare disease trends, and keep a field-level health log.</p>
          <div class="feature-action" style="color:var(--blue-600);">Learn more →</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    section_gap(48)

    st.markdown("""
    <div style="padding:56px 48px; background:var(--green-50); border-top:1px solid var(--border);
                border-bottom:1px solid var(--border); max-width:100%;">
      <div style="max-width:980px; margin:0 auto;">
        <div style="text-align:center; margin-bottom:48px;">
          <div style="font-size:.75rem;font-weight:700;color:var(--green-600);
                      letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px;">Workflow</div>
          <h2 style="font-family:'Poppins',sans-serif;font-size:1.95rem;font-weight:800;
                     color:var(--text-1);letter-spacing:-.03em; margin:0;">Three Steps to a Diagnosis</h2>
        </div>
        <div class="workflow-grid">
          <div class="workflow-card">
            <div class="workflow-icon" style="background:linear-gradient(135deg,#1E7D32,#2E7D32); color:#fff;">📷</div>
            <h4>Capture Leaf</h4>
            <p>Take a clear close-up photo of the affected leaf in natural light.</p>
          </div>
          <div class="workflow-card">
            <div class="workflow-icon" style="background:linear-gradient(135deg,#1565C0,#1976D2); color:#fff;">🤖</div>
            <h4>AI Analysis</h4>
            <p>Our MobileNetV2 model processes the image and identifies disease patterns.</p>
          </div>
          <div class="workflow-card">
            <div class="workflow-icon" style="background:linear-gradient(135deg,#E65100,#F57C00); color:#fff;">📋</div>
            <h4>Get Report</h4>
            <p>Receive a detailed diagnosis with confidence score and next-step treatment guidance.</p>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    section_gap(20)

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
def page_scan():
    page_header(
        "Scan Your Crop",
        "Upload a clear leaf image for instant AI-powered disease detection."
    )
    section_gap(24)

    st.markdown("""
    <style>
    .step-bar {
      display:flex; align-items:center; gap:0; padding:24px 48px;
      background:var(--surface); border-bottom:1px solid var(--border);
    }
    .step { display:flex; align-items:center; gap:10px; flex:1; }
    .step-dot {
      width:32px; height:32px; border-radius:50%;
      display:flex; align-items:center; justify-content:center;
      font-family:'Poppins',sans-serif; font-size:.78rem; font-weight:700;
      flex-shrink:0;
    }
    .step-dot.active { background:var(--green-700); color:#fff; box-shadow:0 4px 12px rgba(30,125,50,.35); }
    .step-dot.done   { background:var(--green-400); color:#fff; }
    .step-dot.idle   { background:var(--border); color:var(--text-3); }
    .step-text { font-size:.8rem; font-weight:600; color:var(--text-2); }
    .step-line { flex:1; height:2px; background:var(--border); max-width:60px; }
    </style>
    <div class="step-bar">
      <div class="step">
        <div class="step-dot active">1</div>
        <span class="step-text">Upload Image</span>
      </div>
      <div class="step-line"></div>
      <div class="step">
        <div class="step-dot idle">2</div>
        <span class="step-text">AI Analysis</span>
      </div>
      <div class="step-line"></div>
      <div class="step">
        <div class="step-dot idle">3</div>
        <span class="step-text">View Report</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    section_gap(32)

    main_col, tip_col = st.columns([3, 2], gap="large")

    with main_col:
        st.markdown("<div style='padding:0 0 0 48px;'>", unsafe_allow_html=True)

        card("""
        <div style="text-align:center; padding:16px 0;">
          <div style="font-size:2.5rem; margin-bottom:12px;">🍃</div>
          <h3 style="font-family:'Poppins',sans-serif; font-size:1.1rem; font-weight:700;
                     color:var(--text-1); margin-bottom:6px;">Upload Leaf Image</h3>
          <p style="font-size:.85rem; color:var(--text-3);">
            Drag & drop or click to select · JPG, PNG, WEBP · Max 10MB
          </p>
        </div>
        """)

        uploaded_file = st.file_uploader(
            "Upload leaf image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            st.session_state.uploaded_image = uploaded_file
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True,
                     caption="✅ Image loaded — ready for analysis")

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                crop_type = st.selectbox(
                    "Crop Type (optional)",
                    ["Auto-detect", "Tomato", "Wheat", "Corn",
                     "Rice", "Potato", "Pepper", "Other"],
                )
            with col_b:
                growth_stage = st.selectbox(
                    "Growth Stage (optional)",
                    ["Unknown", "Seedling", "Vegetative", "Flowering",
                     "Fruiting", "Mature"],
                )

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            if st.button("🤖  Analyze Disease Now", use_container_width=True):
                st.markdown("""<style>
                div[data-testid="column"] button {
                  background:linear-gradient(135deg,#1565C0,#1976D2) !important;
                  color:#fff !important; border:none !important;
                  font-size:1rem !important; font-weight:700 !important;
                  padding:16px !important;
                  box-shadow:0 4px 20px rgba(25,118,210,.4) !important;
                }
                </style>""", unsafe_allow_html=True)

                # ── Animated loader ──
                progress_html = st.empty()
                status_text   = st.empty()

                phases = [
                    (0.15, "🔍 Pre-processing image..."),
                    (0.35, "🧠 Running MobileNetV2 inference..."),
                    (0.55, "🔬 Extracting disease features..."),
                    (0.75, "📊 Computing confidence scores..."),
                    (0.90, "💊 Generating treatment plan..."),
                    (1.00, "✅ Analysis complete!"),
                ]
                bar = st.progress(0)
                for pct, msg in phases:
                    bar.progress(pct)
                    status_text.markdown(
                        f"<p style='text-align:center;color:var(--green-700);font-weight:600;"
                        f"font-size:.9rem;'>{msg}</p>",
                        unsafe_allow_html=True
                    )
                    time.sleep(0.45)

                bar.empty()
                status_text.empty()

                # Store fake result
                st.session_state.result = {
                    "disease":     "Tomato Late Blight",
                    "crop":        crop_type if crop_type != "Auto-detect" else "Tomato",
                    "confidence":  94.2,
                    "severity":    "Moderate",
                    "severity_pct": 52,
                    "scan_date":   datetime.now().strftime("%d %b %Y, %H:%M"),
                    "stage":       growth_stage,
                    "treatments": [
                        "Apply copper-based fungicide (e.g., Bordeaux mixture)",
                        "Remove and destroy all infected plant material immediately",
                        "Reduce overhead irrigation — switch to drip if possible",
                        "Ensure adequate plant spacing for air circulation",
                        "Monitor remaining plants every 48 hours",
                    ],
                    "top_predictions": [
                        ("Tomato Late Blight",    94.2),
                        ("Early Blight",           3.1),
                        ("Septoria Leaf Spot",     1.8),
                        ("Bacterial Speck",        0.9),
                    ],
                }

                # Add to history
                st.session_state.history.insert(0, {
                    "Date":       datetime.now().strftime("%d %b %Y"),
                    "Crop":       st.session_state.result["crop"],
                    "Disease":    st.session_state.result["disease"],
                    "Severity":   st.session_state.result["severity"],
                    "Confidence": f"{st.session_state.result['confidence']}%",
                })
                st.session_state.analysis_done = True
                nav("result")

        st.markdown("</div>", unsafe_allow_html=True)

    with tip_col:
        st.markdown("<div style='padding-right:48px;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:var(--surface); border:1px solid var(--border);
                    border-radius:var(--radius); padding:28px; margin-bottom:20px;">
          <h4 style="font-family:'Poppins',sans-serif; font-size:.95rem; font-weight:700;
                     color:var(--text-1); margin-bottom:18px;">
            📸 Photo Tips for Best Results
          </h4>
          <div style="display:flex; flex-direction:column; gap:14px;">
            <div style="display:flex; gap:12px; align-items:flex-start;">
              <div style="width:28px; height:28px; background:var(--green-50); border:1px solid var(--green-200);
                          border-radius:8px; display:flex; align-items:center; justify-content:center;
                          flex-shrink:0; font-size:.9rem;">☀️</div>
              <div>
                <div style="font-size:.82rem; font-weight:600; color:var(--text-1); margin-bottom:2px;">Natural Light</div>
                <div style="font-size:.77rem; color:var(--text-3); line-height:1.5;">Photograph in diffused daylight, avoid flash</div>
              </div>
            </div>
            <div style="display:flex; gap:12px; align-items:flex-start;">
              <div style="width:28px; height:28px; background:var(--green-50); border:1px solid var(--green-200);
                          border-radius:8px; display:flex; align-items:center; justify-content:center;
                          flex-shrink:0; font-size:.9rem;">🔍</div>
              <div>
                <div style="font-size:.82rem; font-weight:600; color:var(--text-1); margin-bottom:2px;">Close-up Focus</div>
                <div style="font-size:.77rem; color:var(--text-3); line-height:1.5;">Fill the frame with the affected leaf</div>
              </div>
            </div>
            <div style="display:flex; gap:12px; align-items:flex-start;">
              <div style="width:28px; height:28px; background:var(--green-50); border:1px solid var(--green-200);
                          border-radius:8px; display:flex; align-items:center; justify-content:center;
                          flex-shrink:0; font-size:.9rem;">🎯</div>
              <div>
                <div style="font-size:.82rem; font-weight:600; color:var(--text-1); margin-bottom:2px;">Show Symptoms</div>
                <div style="font-size:.77rem; color:var(--text-3); line-height:1.5;">Include the affected areas clearly</div>
              </div>
            </div>
            <div style="display:flex; gap:12px; align-items:flex-start;">
              <div style="width:28px; height:28px; background:var(--green-50); border:1px solid var(--green-200);
                          border-radius:8px; display:flex; align-items:center; justify-content:center;
                          flex-shrink:0; font-size:.9rem;">📐</div>
              <div>
                <div style="font-size:.82rem; font-weight:600; color:var(--text-1); margin-bottom:2px;">Hold Still</div>
                <div style="font-size:.77rem; color:var(--text-3); line-height:1.5;">Avoid motion blur for best accuracy</div>
              </div>
            </div>
          </div>
        </div>

        <div style="background:linear-gradient(135deg,#E3F2FD,#F0F7FF);
                    border:1px solid var(--blue-100); border-radius:var(--radius); padding:22px;">
          <div style="font-size:.78rem; font-weight:700; color:var(--blue-600);
                      letter-spacing:.06em; text-transform:uppercase; margin-bottom:10px;">
            🤖 Supported Crops
          </div>
          <div style="display:flex; flex-wrap:wrap; gap:8px;">
            <span style="background:#fff; border:1px solid var(--blue-100); border-radius:20px;
                         padding:4px 12px; font-size:.75rem; color:var(--text-2); font-weight:500;">🍅 Tomato</span>
            <span style="background:#fff; border:1px solid var(--blue-100); border-radius:20px;
                         padding:4px 12px; font-size:.75rem; color:var(--text-2); font-weight:500;">🌾 Wheat</span>
            <span style="background:#fff; border:1px solid var(--blue-100); border-radius:20px;
                         padding:4px 12px; font-size:.75rem; color:var(--text-2); font-weight:500;">🌽 Corn</span>
            <span style="background:#fff; border:1px solid var(--blue-100); border-radius:20px;
                         padding:4px 12px; font-size:.75rem; color:var(--text-2); font-weight:500;">🌾 Rice</span>
            <span style="background:#fff; border:1px solid var(--blue-100); border-radius:20px;
                         padding:4px 12px; font-size:.75rem; color:var(--text-2); font-weight:500;">🥔 Potato</span>
            <span style="background:#fff; border:1px solid var(--blue-100); border-radius:20px;
                         padding:4px 12px; font-size:.75rem; color:var(--text-2); font-weight:500;">🫑 Pepper</span>
            <span style="background:#fff; border:1px solid var(--blue-100); border-radius:20px;
                         padding:4px 12px; font-size:.75rem; color:var(--text-2); font-weight:500;">+ 12 more</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: RESULT
# ─────────────────────────────────────────────
def page_result():
    if not st.session_state.result:
        section_gap(24)
        st.markdown("""
        <div style="padding:80px 48px; text-align:center;">
          <div style="font-size:3rem; margin-bottom:16px;">🔬</div>
          <h3 style="font-family:'Poppins',sans-serif; color:var(--text-2);">No Analysis Yet</h3>
          <p style="color:var(--text-3); margin-top:8px;">Please upload and scan a leaf image first.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("← Go to Scan Page"):
            nav("scan")
        return

    r = st.session_state.result
    page_header(
        "Diagnosis Report",
        f"Analysis completed on {r['scan_date']} · {r['crop']} · {r['stage']}"
    )
    section_gap(24)

    sev_color = {"Healthy": "#4CAF50", "Mild": "#8BC34A",
                 "Moderate": "#FFC107", "Severe": "#F44336"}.get(r["severity"], "#FFC107")
    sev_bg    = {"Healthy": "#E8F5E9", "Mild": "#F1F8E9",
                 "Moderate": "#FFF8E1", "Severe": "#FFEBEE"}.get(r["severity"], "#FFF8E1")

    st.markdown("<div style='padding:0 48px;'>", unsafe_allow_html=True)

    # ── Disease card ──
    st.markdown(f"""
    <div style="
      background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
      padding:36px; margin-bottom:24px; animation:fadeUp .4s ease both;
      box-shadow:var(--shadow-sm);
    ">
      <div style="display:grid; grid-template-columns:1fr auto; gap:24px; align-items:start;">
        <div>
          <div style="font-size:.72rem; font-weight:700; color:var(--text-3);
                      letter-spacing:.1em; text-transform:uppercase; margin-bottom:10px;">
            Disease Detected
          </div>
          <h2 style="font-family:'Poppins',sans-serif; font-size:1.9rem; font-weight:800;
                     color:var(--text-1); letter-spacing:-.03em; margin-bottom:12px;">
            {r['disease']}
          </h2>
          <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            <span style="background:{sev_bg}; color:{sev_color}; border:1px solid {sev_color}33;
                         border-radius:20px; padding:5px 16px;
                         font-size:.8rem; font-weight:700;">
              ⚠️ {r['severity']} Severity
            </span>
            <span style="background:var(--green-50); color:var(--green-700); border:1px solid var(--green-200);
                         border-radius:20px; padding:5px 16px;
                         font-size:.8rem; font-weight:700;">
              🌿 {r['crop']}
            </span>
            <span style="background:#E3F2FD; color:var(--blue-600); border:1px solid var(--blue-100);
                         border-radius:20px; padding:5px 16px;
                         font-size:.8rem; font-weight:700;">
              🤖 AI Confident
            </span>
          </div>
        </div>
        <div style="text-align:center; min-width:120px;">
          <div style="
            width:100px; height:100px; border-radius:50%;
            background: conic-gradient(var(--green-600) {int(r['confidence']*3.6)}deg, var(--border) 0deg);
            display:flex; align-items:center; justify-content:center;
            margin:0 auto 8px; position:relative;
          ">
            <div style="
              width:78px; height:78px; border-radius:50%; background:var(--surface);
              display:flex; flex-direction:column; align-items:center; justify-content:center;
            ">
              <div style="font-family:'Poppins',sans-serif; font-size:1.2rem; font-weight:800;
                          color:var(--green-700); line-height:1;">{r['confidence']}%</div>
            </div>
          </div>
          <div style="font-size:.72rem; color:var(--text-3); font-weight:600; text-transform:uppercase; letter-spacing:.06em;">Confidence</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Severity meter
        st.markdown(f"""
        <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
                    padding:28px; margin-bottom:20px;">
          <h4 style="font-family:'Poppins',sans-serif; font-size:.95rem; font-weight:700;
                     color:var(--text-1); margin-bottom:20px;">📊 Severity Meter</h4>
          <div style="position:relative; height:12px; background:linear-gradient(90deg,#4CAF50,#FFC107,#F44336);
                      border-radius:6px; margin-bottom:10px; overflow:visible;">
            <div style="
              position:absolute; top:50%; left:{r['severity_pct']}%;
              transform:translate(-50%,-50%);
              width:20px; height:20px; border-radius:50%;
              background:{sev_color}; border:3px solid #fff;
              box-shadow:0 2px 8px rgba(0,0,0,.25);
            "></div>
          </div>
          <div style="display:flex; justify-content:space-between;
                      font-size:.72rem; color:var(--text-3); font-weight:600; text-transform:uppercase;">
            <span>Healthy</span><span>Moderate</span><span>Severe</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Treatments
        treatments_html = "".join([
            f"""<div style="display:flex; align-items:flex-start; gap:12px; padding:12px 0;
                            border-bottom:1px solid var(--border);">
              <div style="width:24px; height:24px; background:var(--green-50); border:1px solid var(--green-200);
                          border-radius:6px; display:flex; align-items:center; justify-content:center;
                          font-size:.75rem; flex-shrink:0; font-weight:700; color:var(--green-700);">{i+1}</div>
              <div style="font-size:.85rem; color:var(--text-2); line-height:1.6;">{t}</div>
            </div>"""
            for i, t in enumerate(r["treatments"])
        ])
        st.markdown(f"""
        <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
                    padding:28px; margin-bottom:20px;">
          <h4 style="font-family:'Poppins',sans-serif; font-size:.95rem; font-weight:700;
                     color:var(--text-1); margin-bottom:4px;">💊 Recommended Treatment</h4>
          <p style="font-size:.8rem; color:var(--text-3); margin-bottom:16px;">
            Based on AI diagnosis — consult an agronomist before large-scale application
          </p>
          {treatments_html}
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Top predictions
        st.markdown("### 🎯 AI Predictions")
        for disease, conf in r["top_predictions"]:
            is_top = conf == r["confidence"]
            if is_top:
                st.markdown(f"**{disease}**  ·  **{conf}%**")
            else:
                st.markdown(f"{disease}  ·  **{conf}%**")
            st.progress(conf / 100.0)
            st.markdown("---")

        # Quick info
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,var(--green-50),#fff);
                    border:1px solid var(--green-200); border-radius:var(--radius); padding:22px;">
          <h4 style="font-family:'Poppins',sans-serif; font-size:.88rem; font-weight:700;
                     color:var(--green-800); margin-bottom:16px;">ℹ️ About {r['disease']}</h4>
          <p style="font-size:.82rem; color:var(--text-2); line-height:1.7; margin-bottom:14px;">
            Late blight is caused by <em>Phytophthora infestans</em>. It spreads rapidly in cool,
            moist conditions and can destroy a crop within days if untreated.
          </p>
          <div style="background:rgba(255,193,7,.1); border:1px solid rgba(255,193,7,.3);
                      border-radius:8px; padding:10px 14px;
                      font-size:.78rem; color:#7B5800; font-weight:600; line-height:1.5;">
            ⚡ Act within 24–48 hours to prevent spread
          </div>
        </div>
        """, unsafe_allow_html=True)

    section_gap(20)

    # Action buttons
    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        if st.button("🔬  Scan Another Leaf", use_container_width=True):
            st.session_state.uploaded_image = None
            st.session_state.result = None
            nav("scan")
    with btn2:
        if st.button("📊  View Dashboard", use_container_width=True):
            nav("dashboard")
    with btn3:
        st.download_button(
            "⬇️  Download Report",
            data=f"CropSense AI Diagnosis Report\n{'='*40}\n"
                 f"Disease: {r['disease']}\nCrop: {r['crop']}\n"
                 f"Confidence: {r['confidence']}%\nSeverity: {r['severity']}\n"
                 f"Date: {r['scan_date']}\n\nTreatments:\n" +
                 "\n".join(f"- {t}" for t in r["treatments"]),
            file_name=f"cropsense_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    section_gap(32)

# ─────────────────────────────────────────────
#  PAGE: DASHBOARD
# ─────────────────────────────────────────────
def page_dashboard():
    page_header(
        "Field Health Dashboard",
        "Track scan history, monitor disease trends, and manage crop health records."
    )
    section_gap(24)

    history = st.session_state.history
    total       = len(history)
    healthy_n   = sum(1 for h in history if h["Disease"] == "Healthy")
    disease_n   = total - healthy_n
    critical_n  = sum(1 for h in history if h["Severity"] == "Severe")

    st.markdown("<div style='padding:0 48px;'>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scans",   total,         delta=f"+{random.randint(1,3)} this week")
    col2.metric("Healthy Cases", healthy_n,     delta=f"+{random.randint(0,2)}")
    col3.metric("Disease Cases", disease_n,     delta=None)
    col4.metric("Critical Cases", critical_n,   delta=None)

    section_gap(28)

    # ── Chart + Table ──
    tab1, tab2 = st.tabs(["📋  Scan History", "📈  Analytics"])

    with tab1:
        search = st.text_input("🔍 Search by crop or disease…", placeholder="e.g. Tomato, Wheat, Blight")

        display_history = history
        if search:
            display_history = [
                h for h in history
                if search.lower() in h["Crop"].lower()
                or search.lower() in h["Disease"].lower()
            ]

        # Color-coded table
        rows_html = ""
        for h in display_history:
            sev = h["Severity"]
            sev_c = {"Healthy": "#4CAF50", "Mild": "#8BC34A",
                     "Moderate": "#FFC107", "Severe": "#F44336"}.get(sev, "#999")
            sev_bg2 = {"Healthy": "#E8F5E9", "Mild": "#F1F8E9",
                       "Moderate": "#FFFDE7", "Severe": "#FFEBEE"}.get(sev, "#f5f5f5")
            rows_html += f"""
            <tr style="border-bottom:1px solid var(--border);">
              <td style="padding:14px 16px;font-size:.83rem;color:var(--text-3);">{h['Date']}</td>
              <td style="padding:14px 16px;font-size:.85rem;font-weight:600;color:var(--text-1);">{h['Crop']}</td>
              <td style="padding:14px 16px;font-size:.84rem;color:var(--text-2);">{h['Disease']}</td>
              <td style="padding:14px 16px;">
                <span style="background:{sev_bg2};color:{sev_c};border:1px solid {sev_c}33;
                             border-radius:20px;padding:3px 12px;font-size:.76rem;font-weight:700;">
                  {sev}
                </span>
              </td>
              <td style="padding:14px 16px;font-size:.84rem;font-weight:700;color:var(--green-700);">{h['Confidence']}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
                    overflow:hidden; margin-top:16px;">
          <table style="width:100%; border-collapse:collapse;">
            <thead>
              <tr style="background:var(--green-50); border-bottom:2px solid var(--border);">
                <th style="padding:14px 16px;text-align:left;font-size:.72rem;font-weight:700;
                           color:var(--text-3);text-transform:uppercase;letter-spacing:.08em;">Date</th>
                <th style="padding:14px 16px;text-align:left;font-size:.72rem;font-weight:700;
                           color:var(--text-3);text-transform:uppercase;letter-spacing:.08em;">Crop</th>
                <th style="padding:14px 16px;text-align:left;font-size:.72rem;font-weight:700;
                           color:var(--text-3);text-transform:uppercase;letter-spacing:.08em;">Disease</th>
                <th style="padding:14px 16px;text-align:left;font-size:.72rem;font-weight:700;
                           color:var(--text-3);text-transform:uppercase;letter-spacing:.08em;">Severity</th>
                <th style="padding:14px 16px;text-align:left;font-size:.72rem;font-weight:700;
                           color:var(--text-3);text-transform:uppercase;letter-spacing:.08em;">Confidence</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:24px; margin-top:8px;">
          <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:28px;">
            <h4 style="font-family:'Poppins',sans-serif; font-size:.9rem; font-weight:700;
                       color:var(--text-1); margin-bottom:20px;">Disease Distribution</h4>
            <div style="display:flex; flex-direction:column; gap:12px;">
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                  <span style="font-size:.82rem;color:var(--text-2);">Late Blight</span>
                  <span style="font-size:.78rem;font-weight:700;color:var(--text-1);">32%</span>
                </div>
                <div style="background:var(--green-50);border-radius:4px;height:8px;">
                  <div style="width:32%;height:100%;background:var(--green-600);border-radius:4px;"></div>
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                  <span style="font-size:.82rem;color:var(--text-2);">Rust</span>
                  <span style="font-size:.78rem;font-weight:700;color:var(--text-1);">24%</span>
                </div>
                <div style="background:var(--green-50);border-radius:4px;height:8px;">
                  <div style="width:24%;height:100%;background:#FFC107;border-radius:4px;"></div>
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                  <span style="font-size:.82rem;color:var(--text-2);">Gray Leaf Spot</span>
                  <span style="font-size:.78rem;font-weight:700;color:var(--text-1);">18%</span>
                </div>
                <div style="background:var(--green-50);border-radius:4px;height:8px;">
                  <div style="width:18%;height:100%;background:#F44336;border-radius:4px;"></div>
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                  <span style="font-size:.82rem;color:var(--text-2);">Healthy</span>
                  <span style="font-size:.78rem;font-weight:700;color:var(--text-1);">26%</span>
                </div>
                <div style="background:var(--green-50);border-radius:4px;height:8px;">
                  <div style="width:26%;height:100%;background:var(--green-400);border-radius:4px;"></div>
                </div>
              </div>
            </div>
          </div>
          <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:28px;">
            <h4 style="font-family:'Poppins',sans-serif; font-size:.9rem; font-weight:700;
                       color:var(--text-1); margin-bottom:20px;">Model Performance</h4>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
              <div style="background:var(--green-50); border:1px solid var(--green-200);
                          border-radius:12px; padding:16px; text-align:center;">
                <div style="font-family:'Poppins',sans-serif; font-size:1.5rem; font-weight:800;
                            color:var(--green-700);">98.2%</div>
                <div style="font-size:.72rem; color:var(--text-3); margin-top:4px; font-weight:600;
                            text-transform:uppercase; letter-spacing:.05em;">Accuracy</div>
              </div>
              <div style="background:#E3F2FD; border:1px solid var(--blue-100);
                          border-radius:12px; padding:16px; text-align:center;">
                <div style="font-family:'Poppins',sans-serif; font-size:1.5rem; font-weight:800;
                            color:var(--blue-700);">97.1%</div>
                <div style="font-size:.72rem; color:var(--text-3); margin-top:4px; font-weight:600;
                            text-transform:uppercase; letter-spacing:.05em;">Precision</div>
              </div>
              <div style="background:#FFF3E0; border:1px solid #FFE0B2;
                          border-radius:12px; padding:16px; text-align:center;">
                <div style="font-family:'Poppins',sans-serif; font-size:1.5rem; font-weight:800;
                            color:#E65100;">96.8%</div>
                <div style="font-size:.72rem; color:var(--text-3); margin-top:4px; font-weight:600;
                            text-transform:uppercase; letter-spacing:.05em;">Recall</div>
              </div>
              <div style="background:#F3E5F5; border:1px solid #E1BEE7;
                          border-radius:12px; padding:16px; text-align:center;">
                <div style="font-family:'Poppins',sans-serif; font-size:1.5rem; font-weight:800;
                            color:#6A1B9A;">96.9%</div>
                <div style="font-size:.72rem; color:var(--text-3); margin-top:4px; font-weight:600;
                            text-transform:uppercase; letter-spacing:.05em;">F1 Score</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    section_gap(32)

# ─────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────
def page_about():
    page_header(
        "About CropSense AI",
        "Built for farmers, powered by deep learning. Protecting the world's food supply one leaf at a time."
    )
    section_gap(24)

    st.markdown("<div style='padding:0 48px;'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("""
        <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
                    padding:36px; margin-bottom:24px;">
          <h3 style="font-family:'Poppins',sans-serif; font-size:1.2rem; font-weight:800;
                     color:var(--text-1); margin-bottom:16px;">🎯 Project Vision</h3>
          <p style="font-size:.9rem; color:var(--text-2); line-height:1.8; margin-bottom:16px;">
            CropSense AI is an intelligent crop disease detection platform designed to bridge the gap
            between modern machine learning and practical agriculture. By enabling farmers and
            agronomists to identify diseases instantly through a smartphone, we aim to reduce
            crop losses caused by delayed detection.
          </p>
          <p style="font-size:.9rem; color:var(--text-2); line-height:1.8;">
            The system leverages a MobileNetV2 deep learning model trained on the PlantVillage
            dataset — over 87,000 images across 38 disease categories — to deliver fast,
            accurate, and actionable diagnoses in under 3 seconds.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🗺️ Development Roadmap")
        st.info(
            "**CURRENT ✓** v1.0 — Frontend Prototype\n\n"
            "Complete UI/UX with all 5 core pages, mobile-responsive design,"
            " and end-to-end user journey simulation."
        )
        st.success(
            "**NEXT →** v2.0 — AI Backend Integration\n\n"
            "MobileNetV2 model trained on PlantVillage dataset with live inference API,"
            " real image processing, and confidence calibration."
        )
        st.warning(
            "**FUTURE** v3.0 — Multilingual & Mobile App\n\n"
            "Native mobile app with offline support, 10+ regional languages,"
            " and weather-integrated disease risk forecasting."
        )
        st.markdown(
            "**VISION** v4.0 — Regional Crop Intelligence\n\n"
            "Region-specific crop support, satellite imagery integration,"
            " and government advisory system API connections."
        )

    with col_right:
        st.markdown("""
        <div style="background:linear-gradient(135deg,var(--green-900),var(--green-800));
                    border-radius:var(--radius); padding:32px; margin-bottom:20px; color:#fff;">
          <h3 style="font-family:'Poppins',sans-serif; font-size:1rem; font-weight:700;
                     color:#fff; margin-bottom:20px;">⚙️ Tech Stack</h3>
          <div style="display:flex; flex-direction:column; gap:12px;">
            <div style="background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12);
                        border-radius:10px; padding:12px 16px; display:flex; align-items:center; gap:12px;">
              <span style="font-size:1.2rem;">🧠</span>
              <div>
                <div style="font-size:.82rem; font-weight:700; color:#fff;">MobileNetV2</div>
                <div style="font-size:.72rem; color:rgba(255,255,255,.5);">Deep Learning Model</div>
              </div>
            </div>
            <div style="background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12);
                        border-radius:10px; padding:12px 16px; display:flex; align-items:center; gap:12px;">
              <span style="font-size:1.2rem;">🌿</span>
              <div>
                <div style="font-size:.82rem; font-weight:700; color:#fff;">PlantVillage Dataset</div>
                <div style="font-size:.72rem; color:rgba(255,255,255,.5);">87,000+ labeled images</div>
              </div>
            </div>
            <div style="background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12);
                        border-radius:10px; padding:12px 16px; display:flex; align-items:center; gap:12px;">
              <span style="font-size:1.2rem;">🐍</span>
              <div>
                <div style="font-size:.82rem; font-weight:700; color:#fff;">Python + TensorFlow</div>
                <div style="font-size:.72rem; color:rgba(255,255,255,.5);">Backend & AI Pipeline</div>
              </div>
            </div>
            <div style="background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12);
                        border-radius:10px; padding:12px 16px; display:flex; align-items:center; gap:12px;">
              <span style="font-size:1.2rem;">🎨</span>
              <div>
                <div style="font-size:.82rem; font-weight:700; color:#fff;">Streamlit</div>
                <div style="font-size:.72rem; color:rgba(255,255,255,.5);">Frontend Framework</div>
              </div>
            </div>
          </div>
        </div>

        <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:28px;">
          <h3 style="font-family:'Poppins',sans-serif; font-size:.95rem; font-weight:700;
                     color:var(--text-1); margin-bottom:18px;">📊 Dataset Stats</h3>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <div style="background:var(--green-50); border:1px solid var(--green-200);
                        border-radius:10px; padding:14px; text-align:center;">
              <div style="font-family:'Poppins',sans-serif; font-size:1.3rem; font-weight:800;
                          color:var(--green-700);">87K+</div>
              <div style="font-size:.7rem; color:var(--text-3); font-weight:600; text-transform:uppercase;">Images</div>
            </div>
            <div style="background:#E3F2FD; border:1px solid var(--blue-100);
                        border-radius:10px; padding:14px; text-align:center;">
              <div style="font-family:'Poppins',sans-serif; font-size:1.3rem; font-weight:800;
                          color:var(--blue-700);">38</div>
              <div style="font-size:.7rem; color:var(--text-3); font-weight:600; text-transform:uppercase;">Classes</div>
            </div>
            <div style="background:#FFF3E0; border:1px solid #FFE0B2;
                        border-radius:10px; padding:14px; text-align:center;">
              <div style="font-family:'Poppins',sans-serif; font-size:1.3rem; font-weight:800;
                          color:#E65100;">14</div>
              <div style="font-size:.7rem; color:var(--text-3); font-weight:600; text-transform:uppercase;">Crops</div>
            </div>
            <div style="background:#F3E5F5; border:1px solid #E1BEE7;
                        border-radius:10px; padding:14px; text-align:center;">
              <div style="font-family:'Poppins',sans-serif; font-size:1.3rem; font-weight:800;
                          color:#6A1B9A;">98%</div>
              <div style="font-size:.7rem; color:var(--text-3); font-weight:600; text-transform:uppercase;">Accuracy</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    section_gap(32)

# ─────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────
page = st.session_state.page
if   page == "home":      page_home()
elif page == "scan":      page_scan()
elif page == "result":    page_result()
elif page == "dashboard": page_dashboard()
elif page == "about":     page_about()
