# 🧠 Age Group Prediction - Nutrition Health Survey (AIPlanet Hackathon)

This project predicts whether a person belongs to the **Adult (0)** or **Senior (1)** age group using health and nutrition features.

🏆 Built for **Summer Analytics 2025 Hackathon** by IIT Guwahati on AIPlanet.

## 🚀 Project Highlights
- Used LightGBM, XGBoost, and ensemble models to optimize predictions.
- Achieved high leaderboard scores using robust feature engineering and data preprocessing.
- Deployed a clean Streamlit Web App for real-time predictions.
- Fully compatible with Streamlit Cloud and local deployment.

## 🧪 Features Used
- GLU (Glucose Level)
- INS (Insulin Level)
- BMI (Body Mass Index)
- PAQ605 (Physical Activity Frequency)
- PAD680 (Daily Activity Level)
- DRQSPREP (Food Preparation Frequency)

## 📦 File Structure
├── models/
│ └── model.pkl
├── streamlit_app.py
├── requirements.txt
├── train_model.ipynb
├── eda.ipynb
└── README.md

markdown
Copy
Edit

## 📈 Run Locally

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run Streamlit:
    ```bash
    streamlit run streamlit_app.py
    ```

3. Or deploy via [Streamlit Cloud](https://streamlit.io/cloud)

## 📊 Example
![Streamlit Screenshot](https://via.placeholder.com/600x300.png?text=Demo+Screenshot)
