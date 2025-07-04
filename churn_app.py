# churn_app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("🔍 Churn Prediction Web App")
st.markdown("Enter customer features below to predict the likelihood of churn.")

# Input fields for top 5 features (as per feature importance)
X16 = st.number_input("X16 (e.g. user activity score)", value=0.0)
X7 = st.number_input("X7", value=0.0)
X4 = st.number_input("X4", value=0.0)
X19 = st.number_input("X19", value=0.0)
X123 = st.number_input("X123", value=0.0)

# Combine into dataframe
input_data = pd.DataFrame([[X16, X7, X4, X19, X123]], columns=["X16", "X7", "X4", "X19", "X123"])

# Prediction
if st.button("Predict Churn Probability"):
    try:
        prediction = model.predict_proba(input_data)[0][1]
        st.success(f"Predicted Churn Probability: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
