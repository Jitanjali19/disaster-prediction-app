import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Disaster Risk Predictor", page_icon="🚨", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("disaster_model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, feature_cols

model, feature_cols = load_artifacts()

st.title("🚨 Disaster Risk Prediction App")
st.write("Enter disaster details and get prediction with probability.")

disaster_type = st.selectbox("Disaster Type", ["Flood", "Cyclone", "Earthquake", "Heatwave", "Landslide"])
state = st.text_input("State", "Maharashtra")
district = st.text_input("District", "Pune")
location = st.text_input("Location", "Riverside")
latitude = st.number_input("Latitude", value=18.52)
longitude = st.number_input("Longitude", value=73.85)
severity_level = st.slider("Severity Level", 1, 10, 8)
affected_population = st.number_input("Affected Population", min_value=0, value=50000)
casualties = st.number_input("Casualties", min_value=0, value=120)
estimated_economic_loss_usd = st.number_input("Estimated Economic Loss (USD)", min_value=0.0, value=2500000.0)
infrastructure_damage_index = st.slider("Infrastructure Damage Index", 0.0, 10.0, 7.5)
relief_centers_opened = st.number_input("Relief Centers Opened", min_value=0, value=15)
evacuation_order = st.selectbox("Evacuation Order", [0, 1], index=1)
aid_provided = st.selectbox("Aid Provided", [0, 1], index=1)
response_time_hours = st.number_input("Response Time (Hours)", min_value=0.0, value=3.5)
medical_teams_deployed = st.number_input("Medical Teams Deployed", min_value=0, value=20)
date_input = st.date_input("Date")

if st.button("Predict"):
    date_input = pd.to_datetime(date_input)

    sample = {
        "disaster_type": disaster_type,
        "state": state,
        "district": district,
        "location": location,
        "latitude": latitude,
        "longitude": longitude,
        "severity_level": severity_level,
        "affected_population": affected_population,
        "casualties": casualties,
        "estimated_economic_loss_usd": estimated_economic_loss_usd,
        "infrastructure_damage_index": infrastructure_damage_index,
        "relief_centers_opened": relief_centers_opened,
        "evacuation_order": evacuation_order,
        "aid_provided": aid_provided,
        "response_time_hours": response_time_hours,
        "medical_teams_deployed": medical_teams_deployed,
        "year": date_input.year,
        "month": date_input.month,
        "day": date_input.day,
        "dayofweek": date_input.dayofweek,
        "weekofyear": int(date_input.isocalendar().week),
        "quarter": date_input.quarter,
        "is_weekend": 1 if date_input.dayofweek in [5, 6] else 0
    }

    input_df = pd.DataFrame([sample])
    input_df = input_df.reindex(columns=feature_cols)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    st.write(f"**Prediction:** {pred}")
    st.write(f"**Probability of Major Disaster:** {prob * 100:.2f}%")

    if pred == 1:
        st.error("High Risk / Major Disaster")
    else:
        st.success("Low Risk / Not a Major Disaster")
