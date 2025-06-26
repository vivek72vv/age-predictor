import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Streamlit page config
st.set_page_config(page_title="Age Group Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# App title
st.title("🧠 Nutrition Survey Age Group Predictor")
st.markdown("Predict whether a person is **Adult** or **Senior** based on health features.")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter User Health Data")

    lbxglu = st.number_input("Glucose (LBXGLU)", min_value=0.0, value=100.0)
    lbxin = st.number_input("Insulin (LBXIN)", min_value=0.0, value=30.0)
    bmxbmi = st.number_input("BMI (BMXBMI)", min_value=10.0, max_value=60.0, value=25.0)

    # Submit
    submitted = st.form_submit_button("Predict Age Group")

# Make prediction
if submitted:
    input_df = pd.DataFrame({
        "LBXGLU": [lbxglu],
        "LBXIN": [lbxin],
        "BMXBMI": [bmxbmi],
        "glucose_insulin_ratio": [lbxglu / (lbxin + 1)],
        "bmi_log": [np.log1p(bmxbmi)],
        "glucose_high": [int(lbxglu > 125)],
        "bmi_high": [int(bmxbmi > 30)]
    })

    # Predict
    pred = model.predict(input_df)[0]
    label = "Senior" if pred == 1 else "Adult"
    st.success(f"🧾 Predicted Age Group: **{label}**")
