"""
Gujarat Crop Suitability Labeling Engine
Based on ICAR, TNAU, and Gujarat Directorate of Agriculture guidelines
Assigns crop labels to village-level soil + climate data
"""

import pandas as pd
import numpy as np
import json

# ─────────────────────────────────────────────
# STEP 1: Derive interpretable features from raw
# soil health card aggregated counts
# ─────────────────────────────────────────────

def derive_soil_features(df):
    """
    Dataset columns store counts of soil samples in High/Medium/Low categories.
    We compute a dominant category and a weighted numeric score for N, P, K, pH, EC.
    ICAR SHC nutrient ranges:
      N: Low <280, Medium 280-560, High >560 kg/ha
      P: Low <11, Medium 11-22, High >22 kg/ha
      K: Low <110, Medium 110-280, High >280 kg/ha
      pH: Acidic <6.5, Neutral 6.5-7.5, Alkaline >7.5
      EC: NonSaline <4, Saline >=4 dS/m
    """
    d = df.copy()
    eps = 1e-6

    # N score: weighted score 1=Low, 2=Medium, 3=High → normalize to 0–1
    d['N_score'] = (d['N_Low'] * 1 + d['N_Medium'] * 2 + d['N_High'] * 3) / (d['N_Total'] * 3 + eps)
    d['P_score'] = (d['P_Low'] * 1 + d['P_Medium'] * 2 + d['P_High'] * 3) / (d['P_Total'] * 3 + eps)
    d['K_score'] = (d['K_Low'] * 1 + d['K_Medium'] * 2 + d['K_High'] * 3) / (d['K_Total'] * 3 + eps)

    # pH: Acidic=1, Neutral=2, Alkaline=3 → numeric pH proxy (6.0, 7.0, 8.0)
    d['pH_score'] = (d['pH_Acidic'] * 6.0 + d['pH_Neutral'] * 7.0 + d['pH_Alkaline'] * 8.0) / (d['pH_Total'] + eps)

    # EC: fraction saline
    d['EC_saline_frac'] = d['EC_Saline'] / (d['EC_Total'] + eps)

    # Organic Carbon: High=3,Medium=2,Low=1
    d['OC_score'] = (d['OC_Low'] * 1 + d['OC_Medium'] * 2 + d['OC_High'] * 3) / (d['OC_Total'] * 3 + eps)

    # Sulfur: fraction sufficient
    d['S_frac'] = d['S_Sufficient'] / (d['S_Total'] + eps)

    # Zn, Fe, B, Mn, Cu: fraction sufficient
    d['Zn_frac'] = d['Zn_Sufficient'] / (d['Zn_Total'] + eps)
    d['Fe_frac'] = d['Fe_Sufficient'] / (d['Fe_Total'] + eps)
    d['B_frac']  = d['B_Sufficient']  / (d['B_Total']  + eps)
    d['Mn_frac'] = d['Mn_Sufficient'] / (d['Mn_Total'] + eps)
    d['Cu_frac'] = d['Cu_Sufficient'] / (d['Cu_Total'] + eps)

    return d


# ─────────────────────────────────────────────
# STEP 2: Crop requirement knowledge base
# Sources: ICAR, TNAU CropSuite, Gujarat Agri Dept,
#          Junagadh Agricultural University, Gujarat Agricultural University
# ─────────────────────────────────────────────

CROPS = {
    # ── CEREALS ──────────────────────────────────────────────────────────
    "Cotton": {
        "category": "Cash Crop",
        "season": "Kharif",
        "pH": (6.5, 8.0),
        "N_score": (0.3, 1.0),    # Medium-High N
        "P_score": (0.3, 1.0),
        "K_score": (0.4, 1.0),
        "temp": (21, 37),
        "rainfall": (500, 1200),  # mm annual
        "humidity": (40, 90),
        "EC_saline_frac": (0, 0.4),
        "OC_score": (0.3, 1.0),
        "weight": 1.0,
    },
    "Groundnut": {
        "category": "Oilseed",
        "season": "Kharif/Rabi",
        "pH": (5.5, 7.5),
        "N_score": (0.2, 0.7),   # Low-Medium N (legume fixes own N)
        "P_score": (0.3, 1.0),
        "K_score": (0.3, 1.0),
        "temp": (20, 35),
        "rainfall": (400, 1000),
        "humidity": (50, 85),
        "EC_saline_frac": (0, 0.2),
        "OC_score": (0.2, 1.0),
        "weight": 1.0,
    },
    "Wheat": {
        "category": "Cereal",
        "season": "Rabi",
        "pH": (6.0, 7.5),
        "N_score": (0.4, 1.0),
        "P_score": (0.3, 1.0),
        "K_score": (0.3, 1.0),
        "temp": (10, 24),
        "rainfall": (200, 600),
        "humidity": (40, 75),
        "EC_saline_frac": (0, 0.3),
        "OC_score": (0.3, 1.0),
        "weight": 1.0,
    },
    "Rice (Paddy)": {
        "category": "Cereal",
        "season": "Kharif",
        "pH": (5.0, 7.5),
        "N_score": (0.4, 1.0),
        "P_score": (0.3, 1.0),
        "K_score": (0.3, 1.0),
        "temp": (20, 37),
        "rainfall": (900, 2200),
        "humidity": (70, 100),
        "EC_saline_frac": (0, 0.2),
        "OC_score": (0.3, 1.0),
        "weight": 1.0,
    },
    "Bajra (Pearl Millet)": {
        "category": "Cereal",
        "season": "Kharif",
        "pH": (6.0, 8.0),
        "N_score": (0.2, 0.8),
        "P_score": (0.2, 0.8),
        "K_score": (0.2, 0.8),
        "temp": (22, 38),
        "rainfall": (250, 800),
        "humidity": (30, 80),
        "EC_saline_frac": (0, 0.5),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    "Maize": {
        "category": "Cereal",
        "season": "Kharif",
        "pH": (5.5, 7.5),
        "N_score": (0.4, 1.0),
        "P_score": (0.3, 1.0),
        "K_score": (0.3, 1.0),
        "temp": (18, 35),
        "rainfall": (500, 900),
        "humidity": (50, 85),
        "EC_saline_frac": (0, 0.25),
        "OC_score": (0.3, 1.0),
        "weight": 1.0,
    },
    "Jowar (Sorghum)": {
        "category": "Cereal",
        "season": "Kharif/Rabi",
        "pH": (6.0, 8.0),
        "N_score": (0.3, 0.9),
        "P_score": (0.2, 0.8),
        "K_score": (0.2, 0.8),
        "temp": (20, 38),
        "rainfall": (400, 1000),
        "humidity": (35, 80),
        "EC_saline_frac": (0, 0.4),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    # ── OILSEEDS ─────────────────────────────────────────────────────────
    "Castor": {
        "category": "Oilseed",
        "season": "Kharif",
        "pH": (5.5, 8.5),
        "N_score": (0.2, 0.8),
        "P_score": (0.2, 0.8),
        "K_score": (0.2, 0.8),
        "temp": (20, 38),
        "rainfall": (300, 900),
        "humidity": (30, 80),
        "EC_saline_frac": (0, 0.5),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    "Sesame (Til)": {
        "category": "Oilseed",
        "season": "Kharif",
        "pH": (5.5, 8.0),
        "N_score": (0.2, 0.8),
        "P_score": (0.2, 0.8),
        "K_score": (0.2, 0.8),
        "temp": (24, 38),
        "rainfall": (300, 800),
        "humidity": (40, 75),
        "EC_saline_frac": (0, 0.3),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    "Mustard (Rapeseed)": {
        "category": "Oilseed",
        "season": "Rabi",
        "pH": (6.0, 7.5),
        "N_score": (0.3, 1.0),
        "P_score": (0.3, 1.0),
        "K_score": (0.2, 0.8),
        "temp": (10, 25),
        "rainfall": (200, 600),
        "humidity": (40, 75),
        "EC_saline_frac": (0, 0.3),
        "OC_score": (0.3, 0.9),
        "weight": 1.0,
    },
    # ── PULSES ───────────────────────────────────────────────────────────
    "Chickpea (Gram)": {
        "category": "Pulse",
        "season": "Rabi",
        "pH": (6.0, 8.0),
        "N_score": (0.1, 0.6),
        "P_score": (0.3, 1.0),
        "K_score": (0.2, 0.8),
        "temp": (10, 29),
        "rainfall": (200, 500),
        "humidity": (30, 70),
        "EC_saline_frac": (0, 0.25),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    "Pigeon Pea (Tur/Arhar)": {
        "category": "Pulse",
        "season": "Kharif",
        "pH": (5.5, 7.5),
        "N_score": (0.1, 0.6),
        "P_score": (0.3, 0.9),
        "K_score": (0.2, 0.8),
        "temp": (18, 38),
        "rainfall": (400, 1000),
        "humidity": (50, 85),
        "EC_saline_frac": (0, 0.25),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    "Green Gram (Moong)": {
        "category": "Pulse",
        "season": "Kharif/Summer",
        "pH": (6.0, 7.5),
        "N_score": (0.1, 0.6),
        "P_score": (0.3, 0.9),
        "K_score": (0.2, 0.8),
        "temp": (22, 38),
        "rainfall": (400, 700),
        "humidity": (50, 85),
        "EC_saline_frac": (0, 0.25),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    "Cluster Bean (Guar)": {
        "category": "Pulse",
        "season": "Kharif",
        "pH": (6.5, 8.5),
        "N_score": (0.1, 0.6),
        "P_score": (0.2, 0.7),
        "K_score": (0.2, 0.7),
        "temp": (25, 40),
        "rainfall": (200, 600),
        "humidity": (30, 75),
        "EC_saline_frac": (0, 0.5),
        "OC_score": (0.1, 0.7),
        "weight": 1.0,
    },
    # ── CASH/COMMERCIAL ──────────────────────────────────────────────────
    "Sugarcane": {
        "category": "Cash Crop",
        "season": "Annual",
        "pH": (6.0, 7.5),
        "N_score": (0.5, 1.0),
        "P_score": (0.3, 1.0),
        "K_score": (0.5, 1.0),
        "temp": (20, 38),
        "rainfall": (1000, 2000),
        "humidity": (70, 100),
        "EC_saline_frac": (0, 0.2),
        "OC_score": (0.4, 1.0),
        "weight": 1.0,
    },
    "Tobacco": {
        "category": "Cash Crop",
        "season": "Rabi",
        "pH": (5.5, 7.0),
        "N_score": (0.3, 0.8),
        "P_score": (0.2, 0.8),
        "K_score": (0.4, 1.0),
        "temp": (15, 32),
        "rainfall": (500, 1000),
        "humidity": (50, 80),
        "EC_saline_frac": (0, 0.2),
        "OC_score": (0.3, 0.8),
        "weight": 1.0,
    },
    # ── SPICES ───────────────────────────────────────────────────────────
    "Cumin (Jeera)": {
        "category": "Spice",
        "season": "Rabi",
        "pH": (6.8, 8.3),
        "N_score": (0.2, 0.8),
        "P_score": (0.2, 0.8),
        "K_score": (0.2, 0.8),
        "temp": (8, 25),
        "rainfall": (200, 500),
        "humidity": (25, 60),
        "EC_saline_frac": (0, 0.3),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    "Fennel (Saunf)": {
        "category": "Spice",
        "season": "Rabi",
        "pH": (6.5, 8.0),
        "N_score": (0.3, 0.9),
        "P_score": (0.2, 0.8),
        "K_score": (0.2, 0.8),
        "temp": (10, 25),
        "rainfall": (300, 600),
        "humidity": (30, 65),
        "EC_saline_frac": (0, 0.3),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    "Coriander": {
        "category": "Spice",
        "season": "Rabi",
        "pH": (6.5, 8.0),
        "N_score": (0.3, 0.9),
        "P_score": (0.2, 0.8),
        "K_score": (0.2, 0.8),
        "temp": (10, 28),
        "rainfall": (250, 600),
        "humidity": (35, 65),
        "EC_saline_frac": (0, 0.3),
        "OC_score": (0.2, 0.8),
        "weight": 1.0,
    },
    "Isabgul (Psyllium)": {
        "category": "Medicinal/Spice",
        "season": "Rabi",
        "pH": (7.0, 8.5),
        "N_score": (0.1, 0.6),
        "P_score": (0.2, 0.7),
        "K_score": (0.1, 0.6),
        "temp": (8, 22),
        "rainfall": (150, 400),
        "humidity": (25, 60),
        "EC_saline_frac": (0, 0.3),
        "OC_score": (0.1, 0.6),
        "weight": 1.0,
    },
    # ── HORTICULTURE (Fruits) ─────────────────────────────────────────────
    "Banana": {
        "category": "Fruit",
        "season": "Annual",
        "pH": (5.5, 7.0),
        "N_score": (0.5, 1.0),
        "P_score": (0.3, 1.0),
        "K_score": (0.6, 1.0),
        "temp": (20, 38),
        "rainfall": (1200, 2200),
        "humidity": (75, 100),
        "EC_saline_frac": (0, 0.1),
        "OC_score": (0.4, 1.0),
        "weight": 1.0,
    },
    "Mango": {
        "category": "Fruit",
        "season": "Perennial",
        "pH": (5.5, 7.5),
        "N_score": (0.3, 0.9),
        "P_score": (0.3, 0.9),
        "K_score": (0.3, 0.9),
        "temp": (22, 38),
        "rainfall": (600, 1500),
        "humidity": (50, 90),
        "EC_saline_frac": (0, 0.2),
        "OC_score": (0.3, 0.9),
        "weight": 1.0,
    },
    "Papaya": {
        "category": "Fruit",
        "season": "Annual",
        "pH": (5.5, 7.0),
        "N_score": (0.5, 1.0),
        "P_score": (0.3, 0.9),
        "K_score": (0.4, 1.0),
        "temp": (22, 38),
        "rainfall": (800, 1800),
        "humidity": (65, 95),
        "EC_saline_frac": (0, 0.15),
        "OC_score": (0.4, 1.0),
        "weight": 1.0,
    },
    "Pomegranate": {
        "category": "Fruit",
        "season": "Perennial",
        "pH": (5.5, 7.5),
        "N_score": (0.3, 0.9),
        "P_score": (0.2, 0.8),
        "K_score": (0.3, 0.9),
        "temp": (20, 38),
        "rainfall": (400, 1000),
        "humidity": (40, 80),
        "EC_saline_frac": (0, 0.3),
        "OC_score": (0.3, 0.9),
        "weight": 1.0,
    },
    "Chikoo (Sapota)": {
        "category": "Fruit",
        "season": "Perennial",
        "pH": (6.0, 8.0),
        "N_score": (0.3, 0.9),
        "P_score": (0.3, 0.9),
        "K_score": (0.3, 0.9),
        "temp": (22, 38),
        "rainfall": (700, 1800),
        "humidity": (65, 95),
        "EC_saline_frac": (0, 0.2),
        "OC_score": (0.3, 0.9),
        "weight": 1.0,
    },
    # ── VEGETABLES ───────────────────────────────────────────────────────
    "Onion": {
        "category": "Vegetable",
        "season": "Rabi",
        "pH": (6.0, 7.5),
        "N_score": (0.4, 1.0),
        "P_score": (0.4, 1.0),
        "K_score": (0.4, 1.0),
        "temp": (12, 30),
        "rainfall": (400, 800),
        "humidity": (50, 80),
        "EC_saline_frac": (0, 0.2),
        "OC_score": (0.4, 1.0),
        "weight": 1.0,
    },
    "Potato": {
        "category": "Vegetable",
        "season": "Rabi",
        "pH": (5.0, 6.5),
        "N_score": (0.5, 1.0),
        "P_score": (0.4, 1.0),
        "K_score": (0.5, 1.0),
        "temp": (10, 24),
        "rainfall": (400, 700),
        "humidity": (60, 85),
        "EC_saline_frac": (0, 0.2),
        "OC_score": (0.4, 1.0),
        "weight": 1.0,
    },
    "Tomato": {
        "category": "Vegetable",
        "season": "Kharif/Rabi",
        "pH": (5.5, 7.0),
        "N_score": (0.5, 1.0),
        "P_score": (0.4, 1.0),
        "K_score": (0.4, 1.0),
        "temp": (15, 35),
        "rainfall": (500, 1000),
        "humidity": (50, 85),
        "EC_saline_frac": (0, 0.2),
        "OC_score": (0.4, 1.0),
        "weight": 1.0,
    },
}


# ─────────────────────────────────────────────
# STEP 3: Scoring function
# ─────────────────────────────────────────────

def score_crop(row, crop_name, req):
    """
    Score a crop for a village row.
    Each parameter contributes a partial score 0-1.
    Returns overall suitability 0-100.
    """
    scores = []

    def range_score(val, low, high):
        """Gaussian-like score: 1.0 at center, drops toward 0 at edges"""
        if val is None or np.isnan(val):
            return 0.5  # uncertain → neutral
        if low <= val <= high:
            center = (low + high) / 2
            half_range = (high - low) / 2
            return 1.0 - 0.3 * abs(val - center) / (half_range + 1e-6)
        elif val < low:
            # penalty proportional to how far below
            gap = low - val
            ref = high - low + 1e-6
            return max(0, 1 - 2.0 * gap / ref)
        else:
            gap = val - high
            ref = high - low + 1e-6
            return max(0, 1 - 2.0 * gap / ref)

    # pH
    ph_score = range_score(row['pH_score'], req['pH'][0], req['pH'][1])
    scores.append(('pH', ph_score, 2.0))

    # N
    n = range_score(row['N_score'], req['N_score'][0], req['N_score'][1])
    scores.append(('N', n, 1.5))

    # P
    p = range_score(row['P_score'], req['P_score'][0], req['P_score'][1])
    scores.append(('P', p, 1.5))

    # K
    k = range_score(row['K_score'], req['K_score'][0], req['K_score'][1])
    scores.append(('K', k, 1.5))

    # Temperature
    t = range_score(row['Temperature'], req['temp'][0], req['temp'][1])
    scores.append(('Temperature', t, 2.5))

    # Rainfall
    r = range_score(row['Rainfall'] * 12, req['rainfall'][0], req['rainfall'][1])
    # Note: dataset rainfall appears to be monthly avg; multiply x12 for annual
    scores.append(('Rainfall', r, 2.5))

    # Humidity
    h = range_score(row['Humidity'], req['humidity'][0], req['humidity'][1])
    scores.append(('Humidity', h, 1.0))

    # EC (salinity) — lower is generally better for non-halophytes
    ec = range_score(row['EC_saline_frac'], req['EC_saline_frac'][0], req['EC_saline_frac'][1])
    scores.append(('EC_salinity', ec, 1.5))

    # Organic Carbon
    oc = range_score(row['OC_score'], req['OC_score'][0], req['OC_score'][1])
    scores.append(('Organic Carbon', oc, 1.0))

    # Weighted average
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(s * w for _, s, w in scores)
    final = (weighted_sum / total_weight) * 100

    return round(final, 2)


# ─────────────────────────────────────────────
# STEP 4: Label each village
# ─────────────────────────────────────────────

def label_crops(df, top_n=5, threshold=50.0):
    """
    For each village row, score all crops and assign labels.
    Returns df with:
      - top_crops: JSON list of {crop, score, category, season}
      - primary_crop: best single crop label
    """
    df = derive_soil_features(df)

    results = []
    for _, row in df.iterrows():
        scores = {}
        for crop, req in CROPS.items():
            s = score_crop(row, crop, req)
            scores[crop] = s

        # Sort descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Filter above threshold
        suitable = [(c, s) for c, s in ranked if s >= threshold]
        if not suitable:
            # take top-3 even if below threshold
            suitable = ranked[:3]

        top = suitable[:top_n]

        # Normalize scores to sum to 100 (percentage recommendation)
        total = sum(s for _, s in top) or 1
        top_normalized = [
            {
                "crop": c,
                "score": round(s, 1),
                "recommendation_pct": round(s / total * 100, 1),
                "category": CROPS[c]["category"],
                "season": CROPS[c]["season"],
            }
            for c, s in top
        ]

        results.append({
            "primary_crop": top[0][0] if top else "Unknown",
            "top_crops_json": json.dumps(top_normalized),
            "crop_label": "|".join([c for c, _ in top[:3]]),
        })

    result_df = pd.DataFrame(results)
    labeled = pd.concat([df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
    return labeled


# ─────────────────────────────────────────────
# STEP 5: Single-row predictor (for web UI)
# ─────────────────────────────────────────────

def predict_single(N_score, P_score, K_score, pH, temperature, rainfall_annual,
                   humidity, windspeed=None, ec_saline_frac=0.0, oc_score=0.5):
    """
    Given direct soil+climate values, return ranked crop recommendations.
    N_score, P_score, K_score: 0–1 (where 1=High, 0.5=Medium, 0.3=Low)
    pH: actual pH value
    temperature: °C
    rainfall_annual: mm/year
    humidity: %
    """
    row = {
        'N_score': N_score,
        'P_score': P_score,
        'K_score': K_score,
        'pH_score': pH,
        'Temperature': temperature,
        'Rainfall': rainfall_annual / 12,  # back to monthly for consistency
        'Humidity': humidity,
        'EC_saline_frac': ec_saline_frac,
        'OC_score': oc_score,
    }
    scores = {}
    for crop, req in CROPS.items():
        scores[crop] = score_crop(pd.Series(row), crop, req)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    total = sum(s for _, s in ranked[:7]) or 1

    return [
        {
            "crop": c,
            "score": round(s, 1),
            "recommendation_pct": round(s / total * 100, 1),
            "category": CROPS[c]["category"],
            "season": CROPS[c]["season"],
        }
        for c, s in ranked[:7]
    ]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os

    input_path = sys.argv[1] if len(sys.argv) > 1 else "FINAL_DATASET.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "LABELED_DATASET.csv"

    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    print("Labeling crops...")
    labeled = label_crops(df)

    # Drop derived intermediate columns to keep output clean
    drop_cols = [c for c in labeled.columns if c.endswith('_score') or c.endswith('_frac')]
    out = labeled.drop(columns=drop_cols, errors='ignore')

    out.to_csv(output_path, index=False)
    print(f"Labeled dataset saved → {output_path}")
    print(f"  Primary crop distribution:\n{out['primary_crop'].value_counts().head(15)}")
