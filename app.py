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
st.write("Fill all the details below to predict whether the disaster is major or not.")

with st.form("prediction_form"):
    st.subheader("Enter Disaster Details")

    disaster_type = st.text_input("Disaster Type")
    state = st.text_input("State")
    district = st.text_input("District")
    location = st.text_input("Location")

    latitude = st.number_input("Latitude", format="%.6f")
    longitude = st.number_input("Longitude", format="%.6f")

    severity_level = st.number_input("Severity Level (1 to 10)", min_value=1, max_value=10, step=1)
    affected_population = st.number_input("Affected Population", min_value=0, step=1)
    casualties = st.number_input("Casualties", min_value=0, step=1)
    estimated_economic_loss_usd = st.number_input("Estimated Economic Loss (USD)", min_value=0.0, step=1000.0, format="%.2f")
    infrastructure_damage_index = st.number_input("Infrastructure Damage Index (0 to 10)", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")
    relief_centers_opened = st.number_input("Relief Centers Opened", min_value=0, step=1)

    evacuation_order = st.selectbox("Evacuation Order", ["Select", 0, 1])
    aid_provided = st.selectbox("Aid Provided", ["Select", 0, 1])

    response_time_hours = st.number_input("Response Time (Hours)", min_value=0.0, step=0.1, format="%.1f")
    medical_teams_deployed = st.number_input("Medical Teams Deployed", min_value=0, step=1)

    date_input = st.date_input("Date")

    submitted = st.form_submit_button("Predict")

if submitted:
    # Validation
    if (
        disaster_type.strip() == ""
        or state.strip() == ""
        or district.strip() == ""
        or location.strip() == ""
        or evacuation_order == "Select"
        or aid_provided == "Select"
    ):
        st.warning("Please fill all fields properly before prediction.")
    else:
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

        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {pred}")
        st.write(f"**Probability of Major Disaster:** {prob * 100:.2f}%")

        if pred == 1:
            st.error("High Risk / Major Disaster")
        else:
            st.success("Low Risk / Not a Major Disaster")
