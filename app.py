import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoder
model = joblib.load('disaster_model.pkl')
le = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Disaster Analyzer", page_icon="🌍")
st.title("🌍 Disaster Impact Predictor")
st.write("Enter details below to check if the event is a **Major Disaster**.")

# User Inputs (Exactly as per your CSV columns)
col1, col2 = st.columns(2)

with col1:
    d_type = st.selectbox("Disaster Type", ["Wildfire", "Hurricane", "Volcanic Eruption", "Drought", "Earthquake", "Flood"])
    location = st.text_input("Location (Country)", "India")
    lat = st.number_input("Latitude", value=20.0)
    lon = st.number_input("Longitude", value=78.0)
    severity = st.slider("Severity Level (1-10)", 1, 10, 5)

with col2:
    population = st.number_input("Affected Population", value=10000)
    loss = st.number_input("Economic Loss (USD)", value=500000.0)
    resp_time = st.number_input("Response Time (Hours)", value=24.0)
    aid = st.selectbox("Aid Provided", ["Yes", "No"])
    damage = st.slider("Infrastructure Damage Index (0-1)", 0.0, 1.0, 0.4)

if st.button("Analyze Impact"):
    # Preprocessing inputs (Model expects numbers)
    # Note: In real app, you'd use the LabelEncoder for text. For now, simple mapping:
    aid_val = 1 if aid == "Yes" else 0
    
    # Feature array (adjust order based on your training)
    features = np.array([[0, 0, lat, lon, severity, population, loss, resp_time, aid_val, damage, 3, 2025]]) # Dummy values for encoded text
    
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("🚨 Result: This is a MAJOR DISASTER")
    else:
        st.success("✅ Result: This is a MINOR DISASTER")
