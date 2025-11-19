import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("Breast Cancer Prediction App (Using Saved Model)")

# Load the saved model
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model()

# If your PKL contains (model, scaler)
try:
    model, scaler = model_data
except:
    model = model_data
    scaler = None

# -----------------------------
# User Input Form
# -----------------------------
st.header("Enter Feature Values")

# 30 features based on your dataset
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0, format="%.5f")
    inputs.append(value)

if st.button("Predict"):
    input_data = np.array(inputs).reshape(1, -1)

    # Apply scaling if scaler exists
    if scaler:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    if prediction == 1 or prediction == "M":
        st.error("⚠️ Prediction: **Malignant (Cancer Detected)**")
    else:
        st.success("✅ Prediction: **Benign (No Cancer)**")
