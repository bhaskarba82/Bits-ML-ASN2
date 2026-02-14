
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Adult Income Predictor", layout="wide")

st.title("💼 Adult Income Classification App")
st.markdown("Predict whether income exceeds **$50K** using trained ML models.")

model_option = st.selectbox(
    "Choose Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
)

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

age = st.slider("Age", 18, 70, 30)
education_num = st.slider("Education Number", 1, 16, 10)
hours = st.slider("Hours per week", 1, 80, 40)

input_data = pd.DataFrame({
    "age": [age],
    "education-num": [education_num],
    "hours-per-week": [hours]
})

if st.button("Predict"):
    try:
        model = joblib.load(model_files[model_option])
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Income > 50K")
        else:
            st.info("Income <= 50K")
    except:
        st.error("Model file not found. Please upload trained model files.")
