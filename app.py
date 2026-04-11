import streamlit as st
import numpy as np
import joblib

# Load models
model = joblib.load("models/crop_recommendation_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

st.title("🌾 Crop Recommendation System")

st.write("Enter soil and climate details:")

# Input fields
N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

# Prediction
if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    # Convert to label
    crop = label_encoder.inverse_transform(prediction)
    
    st.success(f"Recommended Crop: {crop[0]}")