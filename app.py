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

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #00C9A7);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🤖 AI Smart Prediction System")
st.markdown("### 🚀 Intelligent Predictions using Machine Learning")

# ---------------- LOAD MODEL + PIPELINE ----------------
@st.cache_resource
def load_pipeline():
    try:
        model = XGBClassifier()
        model.load_model("model.json")  # ✅ SAFE FORMAT

        scaler = joblib.load("scaler_fixed.pkl")   # ✅ fixed version
        columns = joblib.load("columns_fixed.pkl") # ✅ fixed version

        return model, scaler, columns

    except Exception as e:
        return str(e)

pipeline = load_pipeline()

# ---------------- ERROR HANDLING ----------------
if isinstance(pipeline, str):
    st.error(f"❌ Loading error: {pipeline}")
    st.stop()

model, scaler, columns = pipeline
st.success("✅ Model loaded successfully!")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
st.sidebar.write("Adjust inputs and click Predict")

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Enter Key Feature Values")

# show limited important features (first 8 for usability)
important_features = list(columns[:8])

user_input = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(important_features):
    if i % 2 == 0:
        with col1:
            user_input[feature] = st.number_input(feature, value=0.0)
    else:
        with col2:
            user_input[feature] = st.number_input(feature, value=0.0)

# ---------------- BUILD FULL INPUT ----------------
full_input = {col: 0 for col in columns}
full_input.update(user_input)

input_df = pd.DataFrame([full_input])

# ---------------- SCALE INPUT ----------------
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"❌ Scaling error: {e}")
    st.stop()

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Now"):
    try:
        prediction = model.predict(input_scaled)

        st.success(f"🎯 Prediction Result: {prediction[0]}")

        # ---------------- CONFIDENCE ----------------
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)
            confidence = float(np.max(prob) * 100)

            st.subheader("📈 Prediction Confidence")
            st.progress(int(confidence))
            st.write(f"Confidence Score: **{confidence:.2f}%**")

        # ---------------- INPUT SUMMARY ----------------
        st.subheader("📋 Input Summary")
        st.dataframe(input_df[important_features])

        # ---------------- AI INTERPRETATION ----------------
        st.subheader("🧠 AI Insight")
        if prediction[0] == 1:
            st.warning("⚠️ High likelihood of positive outcome.")
        else:
            st.success("✅ Low risk / safe outcome.")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("✨ Built with Streamlit | AI Project 🚀")
