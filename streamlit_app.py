import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Age Group Predictor", layout="centered")

# Load trained model
model = joblib.load("model.pkl")

st.title("🧠 Age Group Predictor App")
st.markdown("Enter your health and lifestyle details:")

# --------------- Dropdown Inputs with Mapping ---------------
gender = st.selectbox("Gender", ["Male", "Female"])
gender_code = 1 if gender == "Male" else 2

education = st.selectbox("Education Level", [
    "Less than 9th grade", 
    "9–11th grade (incl. 12th, no diploma)", 
    "High school graduate/GED", 
    "Some college or AA degree", 
    "College graduate or above"
])
education_map = {
    "Less than 9th grade": 1,
    "9–11th grade (incl. 12th, no diploma)": 2,
    "High school graduate/GED": 3,
    "Some college or AA degree": 4,
    "College graduate or above": 5
}
education_code = education_map[education]

marital_status = st.selectbox("Marital Status", [
    "Married", "Widowed", "Divorced", "Separated", "Never Married", "Living with Partner"
])
marital_map = {
    "Married": 1,
    "Widowed": 2,
    "Divorced": 3,
    "Separated": 4,
    "Never Married": 5,
    "Living with Partner": 6
}
marital_code = marital_map[marital_status]

diabetes_status = st.selectbox("Have you been told you have diabetes?", [
    "Yes", "No", "Borderline", "Refused", "Don't know"
])
diabetes_map = {
    "Yes": 1,
    "No": 2,
    "Borderline": 3,
    "Refused": 7,
    "Don't know": 9
}
diabetes_code = diabetes_map[diabetes_status]

moderate_activity = st.selectbox("Do you do moderate physical activity?", [
    "Yes", "No", "Refused", "Don't know"
])
activity_map = {
    "Yes": 1,
    "No": 2,
    "Refused": 7,
    "Don't know": 9
}
activity_code = activity_map[moderate_activity]

# ------------------ Numeric Inputs ------------------
weight = st.number_input("Weight (kg)", min_value=1.0, value=70.0)
glucose = st.number_input("Glucose Level (LBXGLU)", min_value=1.0, value=90.0)
insulin = st.number_input("Insulin Level (LBXIN)", min_value=0.0, value=80.0)
bmi = st.number_input("BMI", min_value=10.0, value=25.0)
age = st.number_input("Age (years)", min_value=1, value=45)

# ------------------ DataFrame for Model ------------------
input_df = pd.DataFrame([{
    "bmxwt": weight,
    "diq010": diabetes_code,
    "lbxglu": glucose,
    "lbxin": insulin,
    "paq605": str(activity_code),  # if needed as string for OneHotEncoder
    "riagendr": gender_code,
    "ridageyr": age,
    "dmdeduc2": education_code,
    "dmdmartl": marital_code,
    "bmxbmi": bmi
}])

# ------------------ Feature Engineering ------------------
input_df["glucose_insulin_ratio"] = input_df["lbxglu"] / (input_df["lbxin"] + 1)
input_df["bmi_log"] = np.log1p(input_df["bmxbmi"])
input_df["glucose_high"] = (input_df["lbxglu"] > 125).astype(int)
input_df["bmi_high"] = (input_df["bmxbmi"] > 30).astype(int)
input_df["diq010_missing"] = 0  # assume value always provided

# ------------------ Prediction ------------------
if st.button("🎯 Predict Age Group"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Age Group: **{'Senior' if prediction == 1 else 'Adult'}**")
    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
