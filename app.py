import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------

model = joblib.load("alzheimers_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Feature List
# -----------------------------

features = [
'Age',
'EducationLevel',
'BMI',
'PhysicalActivity',
'SleepQuality',
'FamilyHistoryAlzheimers',
'Hypertension',
'MMSE',
'FunctionalAssessment',
'MemoryComplaints'
]

# -----------------------------
# App Title
# -----------------------------

st.title("Alzheimer Disease Prediction System")

st.write("Machine Learning model to predict risk of Alzheimer's Disease.")

# -----------------------------
# Demographics
# -----------------------------

st.header("Demographic Information")

Age = st.number_input("Age",60,90)
EducationLevel = st.selectbox("Education Level",[0,1,2,3])

# -----------------------------
# Lifestyle
# -----------------------------

st.header("Lifestyle Factors")

BMI = st.number_input("BMI",15.0,40.0)
PhysicalActivity = st.number_input("Physical Activity (hours/week)")
SleepQuality = st.number_input("Sleep Quality (1-10)")

# -----------------------------
# Medical History
# -----------------------------

st.header("Medical History")

FamilyHistoryAlzheimers = st.selectbox("Family History of Alzheimer's",[0,1])
Hypertension = st.selectbox("Hypertension",[0,1])

# -----------------------------
# Cognitive Assessment
# -----------------------------

st.header("Cognitive Assessment")

MMSE = st.number_input("MMSE Score",0,30)
FunctionalAssessment = st.number_input("Functional Assessment Score")
MemoryComplaints = st.selectbox("Memory Complaints",[0,1])

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Alzheimer Risk"):

    input_data = np.array([[Age,
                            EducationLevel,
                            BMI,
                            PhysicalActivity,
                            SleepQuality,
                            FamilyHistoryAlzheimers,
                            Hypertension,
                            MMSE,
                            FunctionalAssessment,
                            MemoryComplaints]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("High Risk of Alzheimer's Disease")
    else:
        st.success("Low Risk of Alzheimer's Disease")

    st.write("Prediction Probability:", probability)

# -----------------------------
# Feature Importance
# -----------------------------

st.subheader("Feature Importance")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(by="Importance",ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# Model Details
# -----------------------------

st.subheader("Model Details")

st.write("Algorithm: Random Forest Classifier")

st.write("Number of Features Used:",len(features))

st.write("Number of Trees:",model.n_estimators)

st.write("Model Type: Classification")

st.write("Target Variable: Diagnosis (0 = No Alzheimer, 1 = Alzheimer)")
