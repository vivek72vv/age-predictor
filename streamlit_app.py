import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained pipeline model
model = joblib.load("model.pkl")

# Page config
st.set_page_config(page_title="Age Group Prediction", layout="centered")
st.title("🧠 Age Group Predictor from Health & Nutrition Survey")

st.write("Please enter the details below:")

# Input fields (all required by model)
lbxglu = st.number_input("Glucose (LBXGLU)", value=100.0)
lbxin = st.number_input("Insulin (LBXIN)", value=30.0)
bmxbmi = st.number_input("BMI (BMXBMI)", value=25.0)
ridageyr = st.number_input("Age (RIDAGEYR)", value=45)
riagendr = st.selectbox("Gender (RIAGENDR)", ["Male", "Female"])
dmdeduc2 = st.selectbox("Education Level (DMDEDUC2)", [1, 2, 3, 4, 5, 7, 9])
dmdmartl = st.selectbox("Marital Status (DMDMARTL)", [1, 2, 3, 4, 5, 6, 77, 99])

if st.button("Predict Age Group"):
    # Map gender to numeric if your model used numeric (adjust if needed)
    riagendr_num = 1 if riagendr == "Male" else 2

    # Prepare input data in the same format as training
    input_data = pd.DataFrame({
        "lbxglu": [lbxglu],
        "lbxin": [lbxin],
        "bmxbmi": [bmxbmi],
        "ridageyr": [ridageyr],
        "riagendr": [riagendr_num],
        "dmdeduc2": [dmdeduc2],
        "dmdmartl": [dmdmartl]
    })

    try:
        pred = model.predict(input_data)[0]
        label = "Adult" if pred == 0 else "Senior"
        st.success(f"🧾 Predicted Age Group: **{label}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
