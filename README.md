# 🌾 Gujarat Crop Suitability Prediction System

An AI-powered crop recommendation system trained on **14,243 village-level Soil Health Card records** across all 33 districts of Gujarat.

## 🔗 Live App
> Deploy to Streamlit Cloud to get your public URL

## 📊 Model Performance
| Model | Accuracy |
|-------|----------|
| XGBoost ✅ (Best) | **87%** |
| Random Forest | 82% |

## 🗂️ Data Sources
- **Soil Data**: [Soil Health Portal, DAC&FW](https://soilhealth.dac.gov.in) — Cycle 2025-26
- **Crop Guidelines**: ICAR Soil Health Card Nutrient Standards
- **Crop Requirements**: TNAU CropSuite, Gujarat Directorate of Agriculture
- **Weather Data**: [NASA POWER API](https://power.larc.nasa.gov)
- **University References**: Junagadh Agricultural University, Gujarat Agricultural University

## 🚀 Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 File Structure
```
├── app.py               # Main Streamlit app
├── crop_labeler.py      # Crop scoring engine (ICAR/TNAU rules)
├── best_model.pkl       # Trained XGBoost model
├── label_encoder.pkl    # Label encoder for crop classes
├── requirements.txt     # Python dependencies
└── README.md
```

## 🌱 How It Works
1. User enters soil N, P, K, pH and climate values
2. Crop labeler scores each crop using ICAR-based rules
3. XGBoost model provides primary classification
4. Top 3 alternative crops shown with suitability scores

## 📍 Coverage
- **33 Districts** of Gujarat
- **250+ Blocks** 
- **14,243 Village records**
- **25+ Crop classes**
