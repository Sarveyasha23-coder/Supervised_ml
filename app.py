import streamlit as st
import numpy as np
import os
import joblib

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="Life Expectancy Predictor",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Life Expectancy Predictor")
st.markdown("### 🚀 Predict life expectancy using Machine Learning")

# ================================
# Show Files (Debug Section)
# ================================
st.subheader("📂 Files in directory")
st.write(os.listdir())

# ================================
# Load Files Safely
# ================================
model = None
scaler = None
columns = None

# Load Model
try:
    model = joblib.load("model.pkl")
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")

# Load Scaler
try:
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Scaler loaded successfully")
except Exception as e:
    st.error(f"❌ Scaler loading failed: {e}")

# Load Columns
try:
    columns = joblib.load("columns.pkl")
    st.success("✅ Columns loaded successfully")
except Exception as e:
    st.error(f"❌ Columns loading failed: {e}")

# Stop app if any file missing
if model is None or scaler is None or columns is None:
    st.warning("⚠️ Please make sure all .pkl files are uploaded correctly.")
    st.stop()

# ================================
# Input UI
# ================================
st.subheader("🧠 Enter Feature Values")

input_data = []

for col in columns:
    value = st.number_input(f"{col}", value=0.0)
    input_data.append(value)

# ================================
# Prediction Button
# ================================
if st.button("🔮 Predict Life Expectancy"):
    try:
        input_array = np.array(input_data).reshape(1, -1)

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Predict
        prediction = model.predict(scaled_input)

        st.success(f"🎯 Predicted Life Expectancy: {round(prediction[0], 2)} years")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
