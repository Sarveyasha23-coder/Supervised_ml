import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Fix for numpy pickle issue
import numpy.random._pickle

st.title("ML Model App")

# Load model safely
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    st.success("Model loaded successfully ✅")

    # Example input (change based on your model)
    input_data = st.text_input("Enter input values (comma separated)")

    if st.button("Predict"):
        try:
            data = np.array([float(i) for i in input_data.split(",")]).reshape(1, -1)
            prediction = model.predict(data)
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.error("Model not loaded ❌")
