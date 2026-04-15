import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Smart Prediction System",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Smart Prediction System")
st.markdown("### 🚀 Intelligent Predictions using Machine Learning")

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_all():
    model = XGBClassifier()
    model.load_model("model.json")   # your model

    scaler = joblib.load("scaler.pkl")   # scaling
    columns = joblib.load("columns.pkl") # feature names

    return model, scaler, columns

try:
    model, scaler, columns = load_all()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Loading error: {e}")
    st.stop()

# ---------------- USER INPUT ----------------
st.subheader("📊 Enter Important Feature Values")

# Example: show only first few important features (edit if needed)
important_features = columns[:5]

user_input = {}

for col in important_features:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# ---------------- CREATE FULL INPUT ----------------
# fill all 81 features with 0
full_input = {col: 0 for col in columns}

# update with user values
full_input.update(user_input)

# convert to dataframe
input_df = pd.DataFrame([full_input])

# ---------------- SCALE ----------------
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Scaling error: {e}")
    st.stop()

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Now"):
    try:
        prediction = model.predict(input_scaled)
        st.success(f"🎯 Prediction: {prediction[0]}")

        # probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)
            confidence = np.max(prob) * 100

            st.subheader("📈 Confidence")
            st.progress(int(confidence))
            st.write(f"{confidence:.2f}%")

        # show input
        st.subheader("📋 Input Snapshot")
        st.dataframe(input_df[important_features])

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("✨ Built with Streamlit | AI Project")
