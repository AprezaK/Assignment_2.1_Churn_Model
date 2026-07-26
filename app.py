import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and encoder once at startup (cached so they don't reload on every interaction)
@st.cache_resource
def load_artifacts():
    with open("churn_rf_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)
    with open("churn_encoder_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

st.subheader("2022 Activity")
total_num_sessions              = st.number_input("Total # of Sessions", min_value=0, max_value=500, value=10)
gross_total_session_length      = st.number_input("Gross Total Session Length (minutes)", min_value=0, max_value=20000, value=500)
active_days                     = st.number_input("Active Days", min_value=0, max_value=5, value=3)
active_quarters                 = st.number_input("Active Quarters", min_value=0, max_value=4, value=2)
avg_sessions_per_active_quarter = st.number_input("Avg Sessions per Active Quarter", min_value=0.0, max_value=200.0, value=5.0, step=0.5)

st.subheader("Demographics")
age                 = st.number_input("Age", min_value=18, max_value=100, value=35)
income_level        = st.radio("Income Level",  ["Low", "Medium", "High", "Very High"])
education           = st.radio("Education",     ["Graduate", "High School", "Other", "Post-Graduate"])
device_type         = st.radio("Device Type",   ["Desktop-only", "Mobile-only", "Multi-device"])
tech_comfort_score  = st.number_input("Tech Comfort Score", min_value=1, max_value=10, value=5)

if st.button("Predict"):

    # Build categorical DataFrame — column names/order must match encoder exactly
    raw = pd.DataFrame([{
        'EDUCATION':    education,
        'INCOME_LEVEL': income_level,
        'DEVICE_TYPE':  device_type,
    }])

    # Apply the saved encoder (transform only — never fit_transform)
    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Numeric features (activity + demographic), then encoded dummies
    numeric_df = pd.DataFrame([{
        'TOTAL_NUM_SESSIONS':              total_num_sessions,
        'GROSS_TOTAL_SESSION_LENGTH':      gross_total_session_length,
        'ACTIVE_DAYS':                     active_days,
        'ACTIVE_QUARTERS':                 active_quarters,
        'AVG_SESSIONS_PER_ACTIVE_QUARTER': avg_sessions_per_active_quarter,
        'AGE':                             age,
        'TECH_COMFORT_SCORE':              tech_comfort_score,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Reindex to guarantee the exact column order the model was trained on,
    # regardless of the order columns were assembled in above.
    input_df = input_df[model.feature_names_in_]

    # Column 1 = P(renewed), column 0 = P(churned)
    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    st.metric("Renewal Probability", f"{probability:.2f}")
    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")
