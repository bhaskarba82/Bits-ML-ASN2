
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ML Disease Prediction System",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
    color:#1f4e79;
}
.sub-text {
    font-size:18px;
    color:#444444;
}
.pred-box {
    padding:20px;
    border-radius:10px;
    background-color:#f0f2f6;
    border:1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="big-title">🧠 Machine Learning Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Professional ML deployment using trained Random Forest model</p>', unsafe_allow_html=True)

st.write("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except:
    st.error("Model file not found. Please ensure model.pkl is present.")
    st.stop()

# ---------------- INPUT SECTION ----------------
st.sidebar.header("📋 Enter Patient Information")

age = st.sidebar.slider("Age", 1, 100, 25)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.5)
glucose = st.sidebar.slider("Glucose Level", 50, 200, 100)
blood_pressure = st.sidebar.slider("Blood Pressure", 50, 180, 80)
insulin = st.sidebar.slider("Insulin", 0, 500, 80)

input_data = np.array([[age, bmi, glucose, blood_pressure, insulin]])

# ---------------- PREDICTION ----------------
st.write("### 🔍 Prediction Result")

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].max()

    st.markdown('<div class="pred-box">', unsafe_allow_html=True)

    if prediction == 1:
        st.error(f"⚠️ High Risk Detected (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk Detected (Confidence: {probability:.2f})")

    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.caption("Developed for Academic Submission | Machine Learning Classification Project")
