# RainPredictionApp 🌦️

## Description

This project predicts whether it will rain tomorrow in Australia based on today's weather conditions. The dataset used is **WeatherAUS**, which contains 270,941 rows and 69 columns of weather-related features. The project includes data preprocessing, handling categorical features with **LabelEncoder**, scaling numerical features, and applying a **Random Forest** machine learning model. The application is interactive and built with **Streamlit**, allowing users to enter weather data and get predictions instantly.

**Live App:** [🌧️ Open RainPredictionApp](https://weatherpredictionapp-ag6n6cdvnmdgzbsdm85kzq.streamlit.app/)

---

## Dataset

* **WeatherAUS.csv** – Weather data for multiple locations in Australia

**Target:** `RainTomorrow`  
Values: `Yes` or `No`  

**Key Features Used in App:**
- MinTemp, MaxTemp, Rainfall, WindGustDir, WindGustSpeed  
- WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm  
- Pressure3pm, Temp9am, Temp3pm, RainToday, RISK_MM  

---

## Data Processing

* Handled categorical features `RainToday` and `WindGustDir` using **LabelEncoder**
* Target variable `RainTomorrow` encoded using **LabelEncoder**
* Scaled numerical features using **StandardScaler**
* Outliers handled if needed during preprocessing
* Prepared data for model training and prediction

---

## Models Used

* **Random Forest Classifier** (best performing model for this project)

**Evaluation Metrics:**
* Accuracy, Precision, Recall, F1-score
* Confusion Matrix
* ROC Curve

---

## Streamlit App (`app.py`)

* Users enter today's weather conditions using input fields for temperature, rainfall, wind, humidity, and pressure.
* Categorical inputs (`RainToday` and `WindGustDir`) are automatically encoded.
* Numerical inputs are scaled before feeding into the model.
* After clicking **Predict**, the app displays whether it is likely to rain tomorrow.

**App Files Included:**
- `app.py` : Streamlit application code
- `RandomForest_model.pkl` : trained model
- `scaler.pkl` : feature scaler
- `RainToday_label_encoder.pkl` : encoder for RainToday
- `WindGustDir_label_encoder.pkl` : encoder for WindGustDir
- `RainTomorrow_label_encoder.pkl` : encoder for target

---

## How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/<your-username>/RainPredictionApp.git
cd RainPredictionApp
