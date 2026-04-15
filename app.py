import streamlit as st
import joblib
import os
import numpy.random._pickle

st.title("ML Model App")

try:
    model_path = os.path.join(os.getcwd(), "model.pkl")
    model = joblib.load(model_path)

    # 🔥 FIX FOR XGBOOST ERROR
    if hasattr(model, "use_label_encoder"):
        delattr(model, "use_label_encoder")

    st.success("Model loaded successfully ✅")

except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None


if model is not None:
    input_data = st.text_input("Enter input values (comma separated)")

    if st.button("Predict"):
        try:
            import numpy as np
            data = np.array([float(i) for i in input_data.split(",")]).reshape(1, -1)
            prediction = model.predict(data)
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
