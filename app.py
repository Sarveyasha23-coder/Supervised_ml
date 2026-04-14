import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Life Expectancy Predictor", layout="centered")

st.title("🌍 Life Expectancy Prediction App")
st.write("Predict life expectancy using Machine Learning")

# -------------------- DEBUG INFO --------------------
st.write("📁 Files in directory:", os.listdir())

# -------------------- LOAD FILES SAFELY --------------------
model, scaler, columns = None, None, None

try:
    model = joblib.load("model.pkl")
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")

try:
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Scaler loaded successfully")
except Exception as e:
    st.error(f"❌ Scaler loading failed: {e}")

try:
    columns = joblib.load("columns.pkl")
    st.success("✅ Columns loaded successfully")
except Exception as e:
    st.error(f"❌ Columns loading failed: {e}")

# Stop execution if anything missing
if model is None or scaler is None or columns is None:
    st.warning("⚠️ Please make sure all .pkl files are uploaded correctly.")
    st.stop()

# -------------------- USER INPUT --------------------
st.subheader("🧾 Enter Input Details")

adult_mortality = st.number_input("Adult Mortality", min_value=0.0, max_value=1000.0, value=200.0)
income_composition = st.number_input("Income Composition of Resources", min_value=0.0, max_value=1.0, value=0.5)
schooling = st.number_input("Schooling (Years)", min_value=0.0, max_value=25.0, value=10.0)
alcohol = st.number_input("Alcohol Consumption", min_value=0.0, max_value=20.0, value=5.0)

status = st.selectbox("Country Status", ["Developing", "Developed"])

# -------------------- PREPARE INPUT --------------------
input_data = pd.DataFrame({
    "Adult Mortality": [adult_mortality],
    "Income composition of resources": [income_composition],
    "Schooling": [schooling],
    "Alcohol": [alcohol],
    "Status_Developing": [1 if status == "Developing" else 0]
})

# Add missing columns
for col in columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns
input_data = input_data[columns]

# -------------------- SCALING --------------------
try:
    input_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"❌ Scaling error: {e}")
    st.stop()

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict Life Expectancy"):
    try:
        prediction = model.predict(input_scaled)
        st.success(f"🌟 Predicted Life Expectancy: {prediction[0]:.2f} years")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
