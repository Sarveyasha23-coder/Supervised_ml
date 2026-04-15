import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import numpy.random._pickle  # Fix for pickle compatibility

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Smart Prediction System",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #00C9A7);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stNumberInput input {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🤖 AI Smart Prediction System")
st.markdown("### 🚀 Predict outcomes using Machine Learning")
st.write("Fill in the details below and get instant intelligent predictions.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.getcwd(), "model.pkl")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        return str(e)

model = load_model()

# ---------------- ERROR HANDLING ----------------
if isinstance(model, str):
    st.error(f"❌ Model loading failed: {model}")
    st.stop()

st.success("✅ Model loaded successfully!")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ App Controls")
st.sidebar.info("Adjust input values and click Predict")

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Enter Feature Values")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", min_value=0.0, value=10.0)
    feature2 = st.number_input("Feature 2", min_value=0.0, value=20.0)

with col2:
    feature3 = st.number_input("Feature 3", min_value=0.0, value=30.0)
    feature4 = st.number_input("Feature 4", min_value=0.0, value=40.0)

# 👉 EDIT FEATURES COUNT BASED ON YOUR MODEL
input_data = np.array([[feature1, feature2, feature3, feature4]])

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Now"):

    try:
        prediction = model.predict(input_data)

        st.success(f"🎯 Prediction Result: {prediction[0]}")

        # ---------------- PROBABILITY ----------------
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)
            confidence = np.max(prob) * 100

            st.subheader("📈 Prediction Confidence")
            st.progress(int(confidence))
            st.write(f"Confidence Score: **{confidence:.2f}%**")

        # ---------------- INPUT SUMMARY ----------------
        st.subheader("📋 Input Summary")
        df = pd.DataFrame(input_data, columns=[
            "Feature 1", "Feature 2", "Feature 3", "Feature 4"
        ])
        st.dataframe(df)

        # ---------------- SIMPLE INTERPRETATION ----------------
        st.subheader("🧠 AI Insight")
        if prediction[0] == 1:
            st.warning("⚠️ Model indicates HIGH probability of positive outcome.")
        else:
            st.success("✅ Model indicates LOW probability of risk.")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("✨ Built with ❤️ using Streamlit | AI Project by You")
