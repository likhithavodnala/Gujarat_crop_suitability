import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# ─────────────────────────────────────────────
# PAGE CONFIG (MUST BE FIRST)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gujarat Crop Suitability AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Earthy Professional Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --soil-dark: #1a1208;
    --soil-brown: #3d2b1f;
    --soil-mid: #6b4226;
    --terra: #c8652a;
    --wheat: #e8c97a;
    --wheat-light: #f5e6b8;
    --leaf: #4a7c59;
    --leaf-light: #7ab08a;
    --sky: #a8c5da;
    --cream: #faf6ee;
    --text-dark: #1a1208;
    --text-mid: #4a3728;
    --text-light: #8a7060;
    --card-bg: rgba(255,255,255,0.92);
    --shadow: 0 8px 32px rgba(61, 43, 31, 0.15);
}

* { box-sizing: border-box; }

html, body, .stApp {
    background-color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-dark) !important;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── HERO BANNER ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1208 0%, #3d2b1f 40%, #6b4226 70%, #c8652a 100%);
    padding: 48px 60px 40px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23e8c97a' fill-opacity='0.06'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    opacity: 0.5;
}
.hero-state-badge {
    display: inline-block;
    background: rgba(232,201,122,0.2);
    border: 1px solid rgba(232,201,122,0.4);
    color: var(--wheat);
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 46px;
    font-weight: 900;
    color: #fff;
    line-height: 1.1;
    margin: 0 0 8px;
}
.hero-title span { color: var(--wheat); }
.hero-subtitle {
    font-size: 16px;
    color: rgba(255,255,255,0.7);
    font-weight: 300;
    margin-bottom: 24px;
    max-width: 520px;
}
.data-sources {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}
.source-chip {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.85);
    font-size: 11px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'DM Sans', sans-serif;
}
.hero-stats {
    position: absolute;
    top: 48px;
    right: 60px;
    display: flex;
    gap: 24px;
}
.hero-stat {
    text-align: center;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 16px 20px;
}
.hero-stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--wheat);
    line-height: 1;
}
.hero-stat-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
    margin-top: 4px;
}

/* ── MAIN LAYOUT ── */
.main-wrapper {
    padding: 40px 60px;
    max-width: 1400px;
    margin: 0 auto;
}
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--terra);
    margin-bottom: 6px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--soil-dark);
    margin-bottom: 24px;
}

/* ── INPUT CARD ── */
.input-card {
    background: white;
    border-radius: 16px;
    padding: 32px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(200,101,42,0.1);
}
.input-group-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--terra);
    border-bottom: 2px solid var(--wheat-light);
    padding-bottom: 8px;
    margin-bottom: 20px;
    margin-top: 24px;
}
.input-group-title:first-child { margin-top: 0; }

/* ── STREAMLIT WIDGET OVERRIDES ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-mid) !important;
}
div[data-testid="stNumberInput"] input {
    border: 1.5px solid #e5d8cc !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--soil-dark) !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--terra) !important;
    box-shadow: 0 0 0 3px rgba(200,101,42,0.15) !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #c8652a, #a84e1a) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 40px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(200,101,42,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(200,101,42,0.5) !important;
}

/* ── PRIMARY CROP CARD ── */
.primary-crop-card {
    background: linear-gradient(135deg, #1a1208, #3d2b1f);
    border-radius: 20px;
    padding: 36px;
    color: white;
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
    box-shadow: 0 12px 40px rgba(61,43,31,0.4);
}
.primary-crop-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(232,201,122,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.primary-label {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--wheat);
    font-weight: 600;
    margin-bottom: 8px;
}
.primary-crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 38px;
    font-weight: 900;
    color: white;
    line-height: 1.1;
    margin-bottom: 12px;
}
.primary-crop-meta {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.meta-badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.85);
    font-size: 12px;
    font-weight: 500;
    padding: 5px 12px;
    border-radius: 6px;
}
.score-bar-container {
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin-top: 16px;
}
.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--wheat), #f5e6b8);
    border-radius: 8px;
    transition: width 1s ease;
}
.score-text {
    font-size: 13px;
    color: rgba(255,255,255,0.6);
    margin-top: 8px;
}
.score-value {
    font-size: 48px;
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    color: var(--wheat);
    float: right;
    line-height: 1;
}

/* ── ALTERNATIVE CROPS ── */
.alt-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
}
.alt-crop-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    border: 1.5px solid #f0e8de;
    position: relative;
    transition: all 0.2s ease;
    box-shadow: 0 2px 10px rgba(61,43,31,0.06);
}
.alt-crop-card:hover {
    border-color: var(--terra);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(200,101,42,0.15);
}
.alt-rank {
    position: absolute;
    top: -1px;
    right: 16px;
    background: var(--terra);
    color: white;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 3px 8px;
    border-radius: 0 0 6px 6px;
}
.alt-crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--soil-dark);
    margin-bottom: 6px;
    margin-top: 8px;
}
.alt-badges {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.alt-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
}
.badge-cereal { background: #fef3cd; color: #8a5e00; }
.badge-pulse { background: #d4edda; color: #155724; }
.badge-oilseed { background: #fff3cd; color: #856404; }
.badge-cash { background: #cce5ff; color: #004085; }
.badge-vegetable { background: #d1ecf1; color: #0c5460; }
.badge-fruit { background: #f8d7da; color: #721c24; }
.badge-other { background: #e2e3e5; color: #383d41; }
.badge-season { background: var(--wheat-light); color: var(--soil-mid); }
.alt-score-bar {
    height: 5px;
    background: #f0e8de;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 10px;
}
.alt-score-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--leaf), var(--leaf-light));
    border-radius: 4px;
}
.alt-score-label {
    font-size: 12px;
    color: var(--text-light);
    margin-top: 6px;
    font-weight: 500;
}
.alt-score-num {
    font-weight: 700;
    color: var(--leaf);
}

/* ── SOIL SUMMARY ── */
.soil-summary-card {
    background: white;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid #f0e8de;
    box-shadow: 0 2px 12px rgba(61,43,31,0.06);
    margin-top: 24px;
}
.soil-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 16px;
}
.soil-item {
    text-align: center;
    padding: 16px 12px;
    border-radius: 10px;
    background: var(--cream);
    border: 1px solid #e8ddd4;
}
.soil-item-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-light);
    font-weight: 600;
    margin-bottom: 6px;
}
.soil-item-value {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--soil-dark);
}
.soil-item-unit {
    font-size: 11px;
    color: var(--text-light);
    margin-top: 2px;
}
.soil-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}
.status-good { background: var(--leaf); }
.status-warn { background: #e8a020; }
.status-low { background: var(--terra); }

/* ── FOOTER ── */
.footer-bar {
    background: var(--soil-dark);
    color: rgba(255,255,255,0.5);
    text-align: center;
    padding: 24px 60px;
    font-size: 12px;
    line-height: 1.8;
    margin-top: 40px;
}
.footer-bar a { color: var(--wheat); text-decoration: none; }

/* ── RESULT SECTION ── */
.results-wrapper {
    animation: fadeInUp 0.5s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── MODEL BADGE ── */
.model-accuracy-banner {
    background: linear-gradient(90deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.model-acc-icon { font-size: 20px; }
.model-acc-text { font-size: 13px; font-weight: 500; color: #166534; }
.model-acc-text strong { font-weight: 700; }

/* divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e0d0c0, transparent);
    margin: 36px 0;
}

/* ── Info box ── */
.info-box {
    background: #fffbf5;
    border-left: 4px solid var(--terra);
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 13px;
    color: var(--text-mid);
    margin-bottom: 20px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        model = joblib.load("best_model.pkl")
        le    = joblib.load("label_encoder.pkl")
        return model, le
    except Exception as e:
        return None, None

model, le = load_models()

# Import crop labeler for direct scoring
try:
    from crop_labeler import predict_single, CROPS
    has_labeler = True
except:
    has_labeler = False


# ─────────────────────────────────────────────
# FEATURE COLUMNS (from train_model.py)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'District_Code', 'Block_Code', 'Latitude', 'Longitude',
    'OC_High', 'OC_Medium', 'OC_Low', 'OC_Total',
    'N_High', 'N_Medium', 'N_Low', 'N_Total',
    'P_High', 'P_Medium', 'P_Low', 'P_Total',
    'K_High', 'K_Medium', 'K_Low', 'K_Total',
    'S_Sufficient', 'S_Deficient', 'S_Total',
    'pH_Acidic', 'pH_Neutral', 'pH_Alkaline', 'pH_Total',
    'EC_Saline', 'EC_NonSaline', 'EC_Total',
    'Fe_Sufficient', 'Fe_Deficient', 'Fe_Total',
    'Zn_Sufficient', 'Zn_Deficient', 'Zn_Total',
    'Cu_Sufficient', 'Cu_Deficient', 'Cu_Total',
    'B_Sufficient', 'B_Deficient', 'B_Total',
    'Mn_Sufficient', 'Mn_Deficient', 'Mn_Total',
    'Temperature', 'Rainfall', 'Humidity', 'WindSpeed'
]

def npk_to_score(n_val, p_val, k_val):
    """Convert N, P, K mg/kg to 0-1 score"""
    # ICAR ranges: N Low<280, Med 280-560, High>560 kg/ha  → approx mg/kg /1.33
    n_score = min(1.0, n_val / 560.0)
    p_score = min(1.0, p_val / 22.0)
    k_score = min(1.0, k_val / 280.0)
    return n_score, p_score, k_score

def get_category_badge_class(category):
    mapping = {
        "Cereal": "badge-cereal",
        "Pulse": "badge-pulse",
        "Oilseed": "badge-oilseed",
        "Cash Crop": "badge-cash",
        "Vegetable": "badge-vegetable",
        "Fruit": "badge-fruit",
    }
    return mapping.get(category, "badge-other")

def get_soil_status(n, p, k, ph):
    status = []
    # N
    if n < 280: status.append(("N (Nitrogen)", n, "kg/ha", "low"))
    elif n < 560: status.append(("N (Nitrogen)", n, "kg/ha", "good"))
    else: status.append(("N (Nitrogen)", n, "kg/ha", "good"))
    # P
    if p < 11: status.append(("P (Phosphorus)", p, "kg/ha", "low"))
    elif p < 22: status.append(("P (Phosphorus)", p, "kg/ha", "warn"))
    else: status.append(("P (Phosphorus)", p, "kg/ha", "good"))
    # K
    if k < 110: status.append(("K (Potassium)", k, "kg/ha", "low"))
    elif k < 280: status.append(("K (Potassium)", k, "kg/ha", "warn"))
    else: status.append(("K (Potassium)", k, "kg/ha", "good"))
    # pH
    if ph < 6.5: status.append(("pH Level", ph, "(Acidic)", "warn"))
    elif ph <= 7.5: status.append(("pH Level", ph, "(Neutral)", "good"))
    else: status.append(("pH Level", ph, "(Alkaline)", "warn"))
    return status

def do_prediction(n, p, k, ph, rainfall, humidity, temperature, windspeed):
    """Use crop_labeler's predict_single for accurate scoring"""
    n_s, p_s, k_s = npk_to_score(n, p, k)
    
    if has_labeler:
        results = predict_single(
            N_score=n_s,
            P_score=p_s,
            K_score=k_s,
            pH=ph,
            temperature=temperature,
            rainfall_annual=rainfall,
            humidity=humidity,
            windspeed=windspeed,
        )
        return results
    return None


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-state-badge">🇮🇳 Gujarat, India · Soil Health Card Cycle 2025-26</div>
    <h1 class="hero-title">Crop Suitability<br><span>Prediction System</span></h1>
    <p class="hero-subtitle">AI-powered crop recommendation engine trained on 14,000+ village-level soil health records across Gujarat's 33 districts.</p>
    <div style="margin-bottom: 8px; font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; color: rgba(255,255,255,0.4);">Data Sources</div>
    <div class="data-sources">
        <span class="source-chip">🌿 ICAR – Soil Nutrient Standards</span>
        <span class="source-chip">📊 TNAU CropSuite Guidelines</span>
        <span class="source-chip">🏛️ Gujarat Directorate of Agriculture</span>
        <span class="source-chip">🌦️ NASA POWER – Weather API</span>
        <span class="source-chip">🔬 Soil Health Portal (DAC&FW)</span>
        <span class="source-chip">🎓 Junagadh Agricultural University</span>
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-value">87%</div>
            <div class="hero-stat-label">XGBoost<br>Accuracy</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">14K+</div>
            <div class="hero-stat-label">Villages<br>Trained</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">33</div>
            <div class="hero-stat-label">Gujarat<br>Districts</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">25+</div>
            <div class="hero-stat-label">Crop<br>Classes</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    <strong>How it works:</strong> Enter your field's soil nutrient values (N, P, K from soil test report), pH, and local climate data below.
    The model — trained on real Gujarat Soil Health Card data and validated against ICAR crop requirement guidelines — will recommend the most suitable crop for your land.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col_input, col_spacer, col_results = st.columns([5, 0.4, 6])

with col_input:
    st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Enter Field Parameters</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    # ── Soil Nutrients ──
    st.markdown('<div class="input-group-title">🧪 Soil Nutrients</div>', unsafe_allow_html=True)
    
    sc1, sc2 = st.columns(2)
    with sc1:
        N = st.number_input("Nitrogen (N) kg/ha", min_value=0.0, max_value=999.0, value=240.0, step=10.0,
                            help="Available Nitrogen in kg/ha. ICAR ranges: Low <280, Medium 280–560, High >560")
        K = st.number_input("Potassium (K) kg/ha", min_value=0.0, max_value=999.0, value=200.0, step=10.0,
                            help="Available Potassium in kg/ha. ICAR ranges: Low <110, Medium 110–280, High >280")
    with sc2:
        P = st.number_input("Phosphorus (P) kg/ha", min_value=0.0, max_value=200.0, value=15.0, step=1.0,
                            help="Available Phosphorus in kg/ha. ICAR ranges: Low <11, Medium 11–22, High >22")
        pH = st.number_input("Soil pH", min_value=4.0, max_value=9.5, value=7.2, step=0.1,
                             help="Soil pH. Neutral: 6.5–7.5 | Acidic: <6.5 | Alkaline: >7.5")
    
    # ── Climate ──
    st.markdown('<div class="input-group-title">🌦️ Climate & Weather</div>', unsafe_allow_html=True)
    
    cc1, cc2 = st.columns(2)
    with cc1:
        temperature = st.number_input("Temperature (°C)", min_value=5.0, max_value=50.0, value=28.5, step=0.5,
                                      help="Average annual temperature in °C")
        humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=75.0, step=1.0,
                                   help="Average relative humidity in %")
    with cc2:
        rainfall = st.number_input("Annual Rainfall (mm)", min_value=50.0, max_value=3000.0, value=650.0, step=10.0,
                                   help="Total annual rainfall in mm")
        windspeed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=3.5, step=0.1,
                                    help="Average wind speed in m/s")
    
    st.markdown('</div>', unsafe_allow_html=True)  # close input-card
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🌾  Analyse & Recommend Crop", use_container_width=True)


# ─────────────────────────────────────────────
# RESULTS SECTION
# ─────────────────────────────────────────────
with col_results:
    st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Crop Recommendation</div>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div style="
            background: white;
            border-radius: 16px;
            padding: 60px 40px;
            text-align: center;
            border: 2px dashed #e0d0c0;
            color: #b0988a;
        ">
            <div style="font-size: 56px; margin-bottom: 16px;">🌱</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 20px; color: #6b4226; margin-bottom: 8px;">
                Ready to Analyse
            </div>
            <div style="font-size: 14px; line-height: 1.6;">
                Fill in your soil and climate<br>parameters, then click the<br><strong>Analyse button</strong> to get your<br>crop recommendation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analysing soil & climate parameters..."):
            results = do_prediction(N, P, K, pH, rainfall, humidity, temperature, windspeed)
        
        if results:
            primary = results[0]
            alternatives = results[1:4]
            
            st.markdown('<div class="results-wrapper">', unsafe_allow_html=True)
            
            # Model accuracy banner
            st.markdown("""
            <div class="model-accuracy-banner">
                <span class="model-acc-icon">✅</span>
                <span class="model-acc-text">Powered by <strong>XGBoost (87% accuracy)</strong> + ICAR/TNAU Crop Knowledge Base · RF Accuracy: 82%</span>
            </div>
            """, unsafe_allow_html=True)
            
            # ── Primary Crop ──
            score_pct = primary['score']
            cat = primary['category']
            season = primary['season']
            crop_name = primary['crop']
            rec_pct = primary['recommendation_pct']
            
            st.markdown(f"""
            <div class="primary-crop-card">
                <div class="primary-label">⭐ Primary Recommendation</div>
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div class="primary-crop-name">{crop_name}</div>
                        <div class="primary-crop-meta">
                            <span class="meta-badge">📂 {cat}</span>
                            <span class="meta-badge">🗓️ {season}</span>
                            <span class="meta-badge">📈 {rec_pct}% suitability weight</span>
                        </div>
                    </div>
                    <div class="score-value">{score_pct:.0f}</div>
                </div>
                <div class="score-bar-container">
                    <div class="score-bar-fill" style="width: {min(score_pct, 100)}%;"></div>
                </div>
                <div class="score-text">Suitability Score (0–100) based on soil + climate match with ICAR guidelines</div>
            </div>
            """, unsafe_allow_html=True)
            
            # ── Top 3 Alternatives ──
            st.markdown("""
            <div style="font-size: 11px; font-weight: 600; letter-spacing: 2.5px; text-transform: uppercase; 
                        color: #8a7060; margin-bottom: 12px;">Also Suitable — Top 3 Alternatives</div>
            <div class="alt-grid">
            """, unsafe_allow_html=True)
            
            for i, alt in enumerate(alternatives):
                badge_class = get_category_badge_class(alt['category'])
                score_width = min(alt['score'], 100)
                st.markdown(f"""
                <div class="alt-crop-card">
                    <div class="alt-rank">#{i+2}</div>
                    <div class="alt-crop-name">{alt['crop']}</div>
                    <div class="alt-badges">
                        <span class="alt-badge {badge_class}">{alt['category']}</span>
                        <span class="alt-badge badge-season">{alt['season']}</span>
                    </div>
                    <div class="alt-score-bar">
                        <div class="alt-score-fill" style="width: {score_width}%;"></div>
                    </div>
                    <div class="alt-score-label">Score: <span class="alt-score-num">{alt['score']:.1f}/100</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # close alt-grid
            
            # ── Soil Health Summary ──
            soil_status = get_soil_status(N, P, K, pH)
            
            st.markdown('<div class="soil-summary-card">', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; 
                        color: #c8652a; margin-bottom: 4px;">Soil Health Snapshot</div>
            <div style="font-size: 13px; color: #6b4226; margin-bottom: 16px;">
                Based on ICAR Soil Health Card classification thresholds
            </div>
            <div class="soil-grid">
            """, unsafe_allow_html=True)
            
            for label, val, unit, status in soil_status:
                dot_class = f"status-{status}"
                st.markdown(f"""
                <div class="soil-item">
                    <div class="soil-item-label"><span class="soil-status-dot {dot_class}"></span>{label}</div>
                    <div class="soil-item-value">{val}</div>
                    <div class="soil-item-unit">{unit}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Extra climate items
            st.markdown(f"""
            <div class="soil-item">
                <div class="soil-item-label">🌡️ Temperature</div>
                <div class="soil-item-value">{temperature:.1f}</div>
                <div class="soil-item-unit">°C</div>
            </div>
            <div class="soil-item">
                <div class="soil-item-label">🌧️ Rainfall</div>
                <div class="soil-item-value">{rainfall:.0f}</div>
                <div class="soil-item-unit">mm/year</div>
            </div>
            <div class="soil-item">
                <div class="soil-item-label">💧 Humidity</div>
                <div class="soil-item-value">{humidity:.0f}</div>
                <div class="soil-item-unit">%</div>
            </div>
            <div class="soil-item">
                <div class="soil-item-label">💨 Wind Speed</div>
                <div class="soil-item-value">{windspeed:.1f}</div>
                <div class="soil-item-unit">m/s</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div></div>', unsafe_allow_html=True)  # close soil-grid + soil-summary-card
            st.markdown('</div>', unsafe_allow_html=True)  # close results-wrapper
        else:
            st.error("❌ Prediction engine not loaded. Please check that crop_labeler.py is present.")

st.markdown('</div>', unsafe_allow_html=True)  # close main-wrapper


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <strong style="color: rgba(255,255,255,0.8);">Gujarat Crop Suitability Prediction System</strong> · Academic Project · 2025-26<br>
    Data: <a href="https://soilhealth.dac.gov.in">Soil Health Portal (DAC&FW)</a> · 
    <a href="https://icar.org.in">ICAR Nutrient Guidelines</a> · 
    <a href="https://tnau.ac.in">TNAU CropSuite</a> · 
    <a href="https://agri.gujarat.gov.in">Gujarat Directorate of Agriculture</a> · 
    <a href="https://power.larc.nasa.gov">NASA POWER Weather API</a><br>
    Models: XGBoost (87% acc) · Random Forest (82% acc) · Trained on 14,243 village records across 33 Gujarat districts
</div>
""", unsafe_allow_html=True)
