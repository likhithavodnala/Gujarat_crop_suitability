# Gujarat Crop Suitability Prediction System

## AI-Based Crop Recommendation Using Soil and Climate Data

The **Gujarat Crop Suitability Prediction System** is a machine learning-based agricultural decision-support application that recommends suitable crops using soil nutrient and climate parameters.

The system integrates **14,243 village-level Soil Health Card records** covering all **33 districts of Gujarat** with climate information and agricultural crop suitability knowledge. XGBoost and Random Forest models were evaluated, and the trained XGBoost model is used in the deployed application.

The application provides a **primary crop recommendation, alternative suitable crops, suitability scores, and a soil health summary** through an interactive Streamlit dashboard.

## Live Application

**Streamlit App:**  
https://gujaratcropsuitability-2vvepw4abayhvtqgfaj5bm.streamlit.app

## Objectives

The main objectives of the project are:

- Integrate soil and climate data for crop suitability analysis.
- Train and compare XGBoost and Random Forest classification models.
- Combine machine learning with agricultural crop suitability knowledge.
- Provide primary and alternative crop recommendations.
- Develop an interactive crop recommendation dashboard.
- Deploy the machine learning system as a Streamlit web application.

## Dataset

The project uses **14,243 village-level soil records** obtained from the Gujarat Soil Health Card Portal and enriched with climate information.

### Dataset Coverage

- **33 districts** of Gujarat
- **250+ blocks**
- **14,243 village-level records**
- **25+ crop classes**

### Input Features

The prediction system uses eight major soil and climate parameters.

#### Soil Parameters

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Soil pH

#### Climate Parameters

- Temperature
- Annual Rainfall
- Humidity
- Wind Speed

## Data Sources

| Data Source | Contribution |
|---|---|
| Soil Health Card Portal (DAC&FW) | Village-level N, P, K and pH records |
| ICAR | Nutrient standards and crop suitability validation |
| TNAU CropSuite | Crop suitability information |
| NASA POWER API | Temperature, rainfall, humidity and wind speed |
| Gujarat Directorate of Agriculture | Agricultural and cropping pattern references |
| Junagadh Agricultural University | Agricultural reference information |
| Gujarat Agricultural University | Agricultural reference information |

## Methodology

```text
Soil Data + Climate Data
          |
          v
    Data Integration
          |
          v
   Data Preprocessing
          |
          v
 Class Balancing (SMOTE)
          |
          v
   Feature Engineering
          |
          v
     Model Training
       /        \
      v          v
  XGBoost    Random Forest
      \          /
       \        /
        v      v
      Model Evaluation
             |
             v
   Crop Suitability Analysis
             |
             v
    Streamlit Deployment
             |
             v
     Crop Recommendation
```

## Machine Learning Models

Two machine learning classification models were evaluated:

- **XGBoost**
- **Random Forest**

XGBoost was selected as the final model used in the deployed application based on the reported model performance.

## Model Performance

| Model | Accuracy |
|---|---:|
| XGBoost | 87% |
| Random Forest | 82% |

The XGBoost model achieved a reported accuracy of **87%** and was serialized for use in the Streamlit application.

## Crop Suitability Analysis

The system evaluates soil and climate conditions to generate crop suitability recommendations.

The prediction process considers:

- Nitrogen availability
- Phosphorus availability
- Potassium availability
- Soil pH
- Temperature
- Annual rainfall
- Humidity
- Wind speed

The application generates:

- Primary crop recommendation
- Alternative crop recommendations
- Suitability scores
- Soil health summary

## Application Workflow

```text
User Input
    |
    v
Soil & Climate Parameters
    |
    v
Input Validation
    |
    v
Crop Suitability Prediction
    |
    v
XGBoost Model + Suitability Logic
    |
    v
Primary Crop Recommendation
    |
    +----> Alternative Crops
    |
    +----> Suitability Scores
    |
    +----> Soil Health Summary
```

## Application Architecture

```text
                    User
                     |
                     v
            Streamlit Dashboard
                     |
                     v
                   app.py
                     |
          +----------+----------+
          |                     |
          v                     v
   crop_labeler.py          ML Model
                            best_model.pkl
          |                     |
          |                     |
          +----------+----------+
                     |
                     v
              Crop Prediction
                     |
                     v
        Recommendation & Scores
```

## Project Structure

```text
Gujarat_crop_suitability/
│
├── app.py
├── crop_labeler.py
├── best_model.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### Important Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `crop_labeler.py` | Crop suitability and recommendation logic |
| `best_model.pkl` | Serialized trained XGBoost model |
| `label_encoder.pkl` | Encoded crop label mapping |
| `requirements.txt` | Required Python dependencies |
| `README.md` | Project documentation |
| `.gitignore` | Files excluded from version control |
| `LICENSE` | Project license |

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Joblib**
- **Streamlit**

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/likhithavodnala/Gujarat_crop_suitability.git
cd Gujarat_crop_suitability
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in a web browser.

## Application Features

The Streamlit dashboard allows users to enter soil and climate parameters and obtain crop suitability recommendations.

### User Inputs

The application accepts:

- Nitrogen
- Potassium
- Phosphorus
- Soil pH
- Temperature
- Humidity
- Annual Rainfall
- Wind Speed

### Prediction Outputs

The application provides:

- Primary crop recommendation
- Alternative suitable crops
- Suitability scores
- Soil health summary
- Model performance information

## Real-World Application

The system can support agricultural decision-making by providing crop recommendations based on available soil and climate conditions.

Potential users include:

- Farmers
- Agricultural extension personnel
- Agronomists
- Agricultural researchers
- Government agricultural departments
- Precision agriculture practitioners

The system is intended as a **decision-support tool** and should be used together with local agronomic knowledge and field-level information.

## My Contribution

### Data Processing, Machine Learning and Application Development

My contribution to the project focused on integrating agricultural datasets, preparing data for analysis, implementing machine learning-based crop suitability prediction, and supporting the development of the interactive application.

Key contributions include:

- Worked with village-level Soil Health Card data from Gujarat.
- Performed data preprocessing and preparation of soil and climate variables.
- Integrated soil information with climate parameters used for crop suitability analysis.
- Worked with agricultural crop suitability information and reference standards.
- Supported feature engineering and preparation of model input variables.
- Worked with class-balancing techniques including SMOTE as part of the machine learning workflow.
- Worked with XGBoost and Random Forest classification models for crop suitability prediction.
- Evaluated model performance and compared classification results.
- Worked with the trained XGBoost model used in the final application.
- Implemented crop recommendation and alternative crop suitability logic.
- Developed and integrated the Streamlit-based interactive application.
- Worked on displaying crop recommendations, suitability scores and soil health information.
- Contributed to testing and validating the application workflow.
- Worked on deploying the application for public access through Streamlit.

### Skills Demonstrated

This project demonstrates experience in:

- Agricultural Data Analytics
- Data Preprocessing
- Data Integration
- Feature Engineering
- Machine Learning
- Classification
- XGBoost
- Random Forest
- Model Evaluation
- Python
- Streamlit
- Decision Support Systems

## Limitations

- Soil information is based on available Soil Health Card records.
- Climate variables are represented using spatially averaged climate information.
- Some regions have lower data representation compared with other parts of Gujarat.
- Crop recommendations depend on the quality and range of the training data.
- The current application uses an English-language interface.
- The model does not directly incorporate real-time field sensor observations.

## Future Scope

The system can be further enhanced through:

### IoT-Based Soil Monitoring

Integration of IoT soil sensors can provide real-time field-level soil information such as nutrient levels, moisture and other soil properties.

### Satellite-Based Crop Monitoring

Satellite-derived vegetation indices such as **NDVI** can be incorporated to monitor crop condition and vegetation health.

### Multilingual Application

The application can be extended with regional language support to improve accessibility for farmers.

### Soil Image Analysis

CNN-based computer vision models could be explored for soil image analysis and additional soil characterization.

### Yield and Profitability Prediction

Crop suitability recommendations can be extended by integrating expected crop yield, market prices and profitability analysis.

### Additional Environmental Parameters

Additional factors such as soil moisture, elevation, irrigation availability and other environmental variables can be incorporated to improve the decision-support system.

## Project Outcome

The project demonstrates the application of machine learning to agricultural crop suitability analysis by integrating soil, climate and agricultural knowledge.

The evaluated models achieved:

- **XGBoost: 87% accuracy**
- **Random Forest: 82% accuracy**

The trained XGBoost model was integrated into a Streamlit dashboard to provide interactive crop recommendations based on user-provided soil and climate parameters.

## References

- Soil Health Card Portal, Department of Agriculture & Farmers Welfare, Government of India.
- Indian Council of Agricultural Research (ICAR).
- TNAU CropSuite.
- NASA POWER.
- Gujarat Directorate of Agriculture.
- Gujarat Agricultural University.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.*
- Breiman, L. (2001). *Random Forests.*
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.*

## Author

### Likhitha Vodnala

**MSc Agriculture Analytics**  
Dhirubhai Ambani University

### Areas of Interest

- Geospatial Analysis
- GIS and Remote Sensing
- Data Analytics
- Data Science
- Artificial Intelligence and Machine Learning
- Geospatial Data Science

### GitHub

https://github.com/likhithavodnala

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
