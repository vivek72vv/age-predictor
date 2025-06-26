import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Age Group Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🧠 Nutrition Survey Age Group Predictor")
st.markdown("Predict whether a person is **Adult** or **Senior** based on health features.")

with st.form("form"):
    st.subheader("🧾 Enter Patient Data")

    lbxglu = st.number_input("Glucose (LBXGLU)", min_value=0.0, value=100.0)
    lbxin = st.number_input("Insulin (LBXIN)", min_value=0.0, value=30.0)
    bmxbmi = st.number_input("BMI (BMXBMI)", min_value=10.0, max_value=60.0, value=25.0)

    diq010 = st.selectbox("Told you have Diabetes? (DIQ010)", options=[1, 2, 3, 7, 9])
    lbxglt = st.number_input("Glucose Load Test (LXBGLT)", min_value=0.0, value=80.0)
    paq605 = st.selectbox("Do Moderate Activity (PAQ605)", options=[1, 2, 7, 9])

    riagendr = st.selectbox("Gender (RIAGENDR)", options=[1, 2])
    ridageyr = st.number_input("Age (RIDAGEYR)", min_value=18, max_value=90, value=45)
    dmdeduc2 = st.selectbox("Education Level (DMDEDUC2)", options=[1, 2, 3, 4, 5, 7, 9])
    dmdmartl = st.selectbox("Marital Status (DMDMARTL)", options=[1, 2, 3, 4, 5, 6, 77, 99])

    submitted = st.form_submit_button("Predict Age Group")

if submitted:
    input_df = pd.DataFrame({
        "lbxglu": [lbxglu],
        "lbxin": [lbxin],
        "bmxbmi": [bmxbmi],
        "diq010": [diq010],
        "lbxglt": [lbxglt],
        "paq605": [paq605],
        "riagendr": [riagendr],
        "ridageyr": [ridageyr],
        "dmdeduc2": [dmdeduc2],
        "dmdmartl": [dmdmartl]
    })

    # 🔧 Add engineered features
    input_df["diq010_missing"] = input_df["diq010"].isna().astype(int)
    input_df["glucose_insulin_ratio"] = input_df["lbxglu"] / (input_df["lbxin"] + 1)
    input_df["bmi_log"] = np.log1p(input_df["bmxbmi"])
    input_df["glucose_log"] = np.log1p(input_df["lbxglu"])
    input_df["insulin_log"] = np.log1p(input_df["lbxin"])
    input_df["glucose_high"] = (input_df["lbxglu"] > 125).astype(int)
    input_df["bmi_high"] = (input_df["bmxbmi"] > 30).astype(int)

    try:
        pred = model.predict(input_df)[0]
        result = "Senior" if pred == 1 else "Adult"
        st.success(f"🎯 Predicted Age Group: **{result}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
