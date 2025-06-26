import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Age Group Predictor", layout="centered")

# Load trained model
model = joblib.load("model.pkl")

st.title("🧠 Age Group Prediction Web App")
st.markdown("Enter health and demographic details below to predict if the person is an **Adult** or **Senior**.")

# Input fields
weight = st.number_input("Weight (kg) (BMXWT)", min_value=1.0, value=70.0)
diq010_label = st.selectbox("Told you have Diabetes? (DIQ010)", ["Yes", "No", "Borderline", "Refused", "Don't know"])
lbxglu = st.number_input("Glucose Load Test (LBXGLT)", min_value=1.0, value=90.0)
paq605_label = st.selectbox("Do Moderate Activity (PAQ605)", ["Yes", "No", "Refused", "Don't know"])
riagendr_label = st.selectbox("Gender (RIAGENDR)", ["Male", "Female"])
ridageyr = st.number_input("Age (RIDAGEYR)", min_value=1, value=45)
dmdeduc2_label = st.selectbox("Education Level (DMDEDUC2)", [
    "Less than 9th grade", 
    "9-11th grade (incl. 12th, no diploma)", 
    "High school graduate/GED", 
    "Some college or AA degree", 
    "College graduate or above"
])
dmdmartl_label = st.selectbox("Marital Status (DMDMARTL)", [
    "Married", 
    "Widowed", 
    "Divorced", 
    "Separated", 
    "Never Married", 
    "Living with Partner"
])
bmxbmi = st.number_input("BMI (BMXBMI)", min_value=10.0, value=25.0)
lbxin = st.number_input("Insulin Level (LBXIN)", min_value=0.0, value=80.0)

# Mappings
diq010_map = {"Yes": 1, "No": 2, "Borderline": 3, "Refused": 7, "Don't know": 9}
paq605_map = {"Yes": 1, "No": 2, "Refused": 7, "Don't know": 9}
riagendr_map = {"Male": 1, "Female": 2}
dmdeduc2_map = {
    "Less than 9th grade": 1,
    "9-11th grade (incl. 12th, no diploma)": 2,
    "High school graduate/GED": 3,
    "Some college or AA degree": 4,
    "College graduate or above": 5
}
dmdmartl_map = {
    "Married": 1,
    "Widowed": 2,
    "Divorced": 3,
    "Separated": 4,
    "Never Married": 5,
    "Living with Partner": 6
}

# Build DataFrame
input_df = pd.DataFrame([{
    "bmxwt": weight,
    "diq010": diq010_map[diq010_label],
    "lbxglu": lbxglu,
    "paq605": paq605_map[paq605_label],
    "riagendr": riagendr_map[riagendr_label],
    "ridageyr": ridageyr,
    "dmdeduc2": dmdeduc2_map[dmdeduc2_label],
    "dmdmartl": dmdmartl_map[dmdmartl_label],
    "bmxbmi": bmxbmi,
    "lbxin": lbxin
}])

# Feature engineering
input_df["glucose_insulin_ratio"] = input_df["lbxglu"] / (input_df["lbxin"] + 1)
input_df["bmi_log"] = np.log1p(input_df["bmxbmi"])
input_df["glucose_high"] = (input_df["lbxglu"] > 125).astype(int)
input_df["bmi_high"] = (input_df["bmxbmi"] > 30).astype(int)
input_df["diq010_missing"] = 0

# Predict
if st.button("Predict Age Group"):
    try:
        result = model.predict(input_df)[0]
        st.success(f"🎯 Predicted Age Group: **{'Senior' if result == 1 else 'Adult'}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

