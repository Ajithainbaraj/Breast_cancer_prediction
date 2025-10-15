import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime

API_URL = "http://127.0.0.1:5000/predict"

st.title("🧠 Breast Cancer Prediction (DNN Model API Integrated)")

# Patient name input
patient_name = st.text_input("👩‍⚕️ Enter Patient Name")

# Feature input
st.subheader("🔢 Enter Feature Values")
important_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]

feature_values = {}
for feature in important_features:
    feature_values[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.01, format="%.3f")

if st.button("🔍 Predict Diagnosis"):
    # Prepare JSON data
    data = {"patient_name": patient_name}
    data.update(feature_values)

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            diagnosis = result["prediction"]
            st.success(f"✅ Prediction for {patient_name}: **{diagnosis}**")

            # Save to CSV
            record = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Patient Name": patient_name}
            record.update(feature_values)
            record["Prediction"] = diagnosis

            df = pd.DataFrame([record])
            file_exists = os.path.exists("predictions_dnn.csv")
            df.to_csv("predictions_dnn.csv", mode="a", index=False, header=not file_exists)
            st.info("📁 Prediction saved to `predictions_dnn.csv`")

        else:
            st.error(f"❌ API Error: {response.text}")

    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
