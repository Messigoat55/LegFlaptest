import streamlit as st
import pandas as pd
from joblib import load

# Streamlit app title
st.title('Lower Extremity Flap Complication Predictor')

# Input fields for patient characteristics
diabetes = st.selectbox('Diabetes', [0, 1])
cardiovascular_disease = st.selectbox('Cardiovascular Disease', [0, 1])
smoking = st.selectbox('Smoking', [0, 1])
immunosuppression = st.selectbox('Immunosuppression', [0, 1])
albumin_level = st.number_input('Albumin Level', value=4.0, step=0.1)
prealbumin_level = st.number_input('Prealbumin Level', value=20.0, step=0.1)
age = st.number_input('Age', value=50, step=1)
sex = st.selectbox('Sex', ['Male', 'Female'])

# Load models only when the button is clicked
if st.button('Predict Complications'):
    try:
        models = load('all_models.joblib')  # Load models here to save startup time
        # Convert inputs into a dataframe
        input_data = pd.DataFrame({
            'Diabetes': [diabetes],
            'Cardiovascular_Disease': [cardiovascular_disease],
            'Smoking': [smoking],
            'Immunosuppression': [immunosuppression],
            'Albumin_Level': [albumin_level],
            'Prealbumin_Level': [prealbumin_level],
            'Age': [age],
            'Sex_Male': [1 if sex == 'Male' else 0]
        })
        # Predict and display results for each outcome
        for outcome, model in models.items():
            prediction = model.predict(input_data)[0]
            st.write(f'Risk of {outcome}: {"Yes" if prediction == 1 else "No"}')
    except Exception as e:
        st.error(f"Error loading models or predicting: {str(e)}")
