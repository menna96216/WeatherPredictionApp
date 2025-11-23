import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# Load all saved objects
# -------------------------------
model = pickle.load(open("RandomForest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
rain_today_encoder = pickle.load(open("RainToday_label_encoder.pkl", "rb"))
wind_dir_encoder = pickle.load(open("WindGustDir_label_encoder.pkl", "rb"))
rain_tomorrow_encoder = pickle.load(open("RainTomorrow_label_encoder.pkl", "rb"))

# -------------------------------
# Streamlit app UI
# -------------------------------
st.set_page_config(page_title="Rain Prediction App", page_icon="☔", layout="centered")

st.title("🌦️ Weather Prediction: Will it Rain Tomorrow?")
st.write("Enter today's weather conditions below to predict if it will rain tomorrow.")

# -------------------------------
# Input fields
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    MinTemp = st.number_input("Min Temperature (°C)", value=10.0)
    MaxTemp = st.number_input("Max Temperature (°C)", value=25.0)
    Rainfall = st.number_input("Rainfall (mm)", value=0.0)
    WindGustDir = st.selectbox("Wind Gust Direction", wind_dir_encoder.classes_)
    WindGustSpeed = st.number_input("Wind Gust Speed (km/h)", value=35.0)
    WindSpeed9am = st.number_input("Wind Speed 9am (km/h)", value=10.0)
    WindSpeed3pm = st.number_input("Wind Speed 3pm (km/h)", value=15.0)

with col2:
    Humidity9am = st.number_input("Humidity 9am (%)", value=70.0)
    Humidity3pm = st.number_input("Humidity 3pm (%)", value=50.0)
    Pressure3pm = st.number_input("Pressure 3pm (hPa)", value=1015.0)
    Temp9am = st.number_input("Temp 9am (°C)", value=18.0)
    Temp3pm = st.number_input("Temp 3pm (°C)", value=24.0)
    RainToday = st.selectbox("Rain Today?", rain_today_encoder.classes_)
    RISK_MM = st.number_input("RISK_MM (mm)", value=5.0)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("🔍 Predict"):
    # Encode categorical inputs
    RainToday_encoded = rain_today_encoder.transform([RainToday])[0]
    WindGustDir_encoded = wind_dir_encoder.transform([WindGustDir])[0]

    # Prepare data for prediction
    input_data = np.array([[MinTemp, MaxTemp, Rainfall, WindGustDir_encoded, WindGustSpeed,
                            WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm,
                            Pressure3pm, Temp9am, Temp3pm, RainToday_encoded, RISK_MM]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Decode the prediction
    result = rain_tomorrow_encoder.inverse_transform([prediction])[0]

    # -------------------------------
    # Display result
    # -------------------------------
    if result == "Yes":
        st.success("🌧️ It is likely to rain tomorrow.")
    else:
        st.info("☀️ It is unlikely to rain tomorrow.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Random Forest Model")
