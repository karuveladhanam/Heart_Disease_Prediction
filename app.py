import streamlit as st
import pickle
import numpy as np

# Load Model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Scaler
with open("heart_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details below.")

# -------------------------
# Input Fields
# -------------------------

age = st.number_input("Age", 1, 120, 50)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)
sex = 1 if sex == "Male" else 0

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3],
    help="0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic"
)

trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)

chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

fbs = st.selectbox(
    "Fasting Blood Sugar >120 mg/dl",
    ["No", "Yes"]
)
fbs = 1 if fbs == "Yes" else 0

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input("Maximum Heart Rate", 50, 250, 150)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)
exang = 1 if exang == "Yes" else 0

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels (0-4)",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict"):

    features = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    probability = model.predict_proba(features_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    st.write(f"Probability of Heart Disease: **{probability[0][1]*100:.2f}%**")
    st.write(f"Probability of No Heart Disease: **{probability[0][0]*100:.2f}%**")
