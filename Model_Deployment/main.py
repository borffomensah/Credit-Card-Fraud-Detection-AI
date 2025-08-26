import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load Model & Scaler
# -----------------------------
BASE_DIR = os.path.dirname(__file__)  # Folder where main.py is located

try:
    model = joblib.load(os.path.join(BASE_DIR, "rf.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "sc.joblib"))
except FileNotFoundError:
    st.error("⚠️ Model or scaler file not found. Please make sure 'rf.joblib' and 'sc.joblib' are inside Model_Deployment/")
    st.stop()

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Credit Card Fraud Detection")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📂 Transaction Details", "📊 Features (V1–V28)", "🔎 Prediction"])

# --- TAB 1: Transaction Details ---
with tab1:
    st.subheader("Transaction Information")
    
    uploaded_img = st.file_uploader("Upload a receipt/transaction image (optional)", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        st.image(uploaded_img, caption="Uploaded Transaction Image", use_container_width=True)

    cols = st.columns(2)
    time_val = cols[0].number_input('⏱ Time (seconds since first transaction)', value=0.0)
    amount_val = cols[1].number_input('💵 Transaction Amount', value=0.0)

# --- TAB 2: Feature Inputs ---
with tab2:
    st.subheader("Enter the anonymized features (V1–V28)")
    v_features = {}
    for i in range(1, 29, 3):  # group inputs in rows of 3
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j <= 28:
                v_features[f'V{i+j}'] = col.number_input(f'V{i+j}', value=0.0)

# --- Prepare Data ---
input_features = {'Time': time_val, 'Amount': amount_val}
input_features.update(v_features)

feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_df = pd.DataFrame([input_features])[feature_columns]

input_scaled = scaler.transform(input_df)

# --- TAB 3: Prediction ---
with tab3:
    st.subheader("Prediction Result")

    # Sample Test Button
    if st.button("🧪 Use Sample Test Data"):
        sample_data = {'Time': 5000, 'Amount': 120.55}
        for i in range(1, 29):
            sample_data[f'V{i}'] = np.random.uniform(-2, 2)  # random test values

        sample_df = pd.DataFrame([sample_data])[feature_columns]
        sample_scaled = scaler.transform(sample_df)

        prediction = model.predict(sample_scaled)
        prediction_proba = model.predict_proba(sample_scaled)

        st.info("✅ Using sample data for testing")
        if prediction[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
            st.write(f"Confidence (Probability of Fraud): {prediction_proba[0][1]:.4f}")
        else:
            st.success("✅ Non-Fraudulent Transaction")
            st.write(f"Confidence (Probability of Non-Fraud): {prediction_proba[0][0]:.4f}")

    # Real User Input Prediction
    if st.button("🚀 Run Prediction"):
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        if prediction[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
            st.write(f"Confidence (Probability of Fraud): {prediction_proba[0][1]:.4f}")
        else:
            st.success("✅ Non-Fraudulent Transaction")
            st.write(f"Confidence (Probability of Non-Fraud): {prediction_proba[0][0]:.4f}")

st.write("---")
st.caption("⚡ Note: This is a demo for educational purposes. Real-world fraud detection involves more complex models and data.")
