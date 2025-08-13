import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_data
def load_and_train():
    data = pd.read_csv("train.csv")

    # Fill missing values
    data['bmi'].fillna(data['bmi'].mean(), inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        data[col] = le.fit_transform(data[col])

    X = data.drop(['id', 'stroke'], axis=1)
    y = data['stroke']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, le, X.columns

model, le, feature_names = load_and_train()

st.title("🩺 Stroke Prediction App")
st.write("Enter patient details to predict stroke risk.")

import random

gif_urls = [
    "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXRmYnd4dHlvcTNoYW0yZGhzdzhxOGxra3V6eXRtYnE3NzNmNWR6MyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/1Bd7DmRvbhV5UPkoDw/giphy.gif",
    "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXRmYnd4dHlvcTNoYW0yZGhzdzhxOGxra3V6eXRtYnE3NzNmNWR6MyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/l0Iy69RBwtdmvwkIo/giphy.gif"
]

random_gif = random.choice(gif_urls)
st.image(random_gif, use_container_width=True)

# User inputs
def user_input_features():
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=85.0)
    bmi = st.number_input('BMI', min_value=0.0, value=25.0)
    smoking_status = st.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

    # Encoding
    mapping = {
        'Male': 1, 'Female': 0, 'Other': 2,
        'No': 0, 'Yes': 1,
        'Urban': 1, 'Rural': 0,
        'Private': 2, 'Self-employed': 3, 'Govt_job': 0, 'children': 1, 'Never_worked': 4,
        'never smoked': 1, 'formerly smoked': 0, 'smokes': 2, 'Unknown': 3
    }

    data = {
        'gender': mapping[gender],
        'age': age,
        'hypertension': mapping[hypertension],
        'heart_disease': mapping[heart_disease],
        'ever_married': mapping[ever_married],
        'work_type': mapping[work_type],
        'Residence_type': mapping[Residence_type],
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': mapping[smoking_status]
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠ High Stroke Risk — Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low Stroke Risk — Probability: {probability:.2f}")
