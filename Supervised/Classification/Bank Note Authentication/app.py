import streamlit as st
import pandas as pd
import numpy as np
import joblib

model=joblib.load('best_model.joblib')
st.title("Bank Note Authentication Prediction")
st.write("""
This app predicts whether a bank note is authentic or not based on its features.
""")

variance = st.number_input('Variance', value=0.0)
skewness = st.number_input('Skewness', value=0.0)
curtosis = st.number_input('Curtosis', value=0.0)
entropy = st.number_input('Entropy', value=0.0)

input_features=np.array([[variance, skewness, curtosis, entropy]])

if st.button('Predict'):
    prediction=model.predict(input_features)[0]
    predicted_class=int(np.round(prediction))
    if predicted_class==1:
        st.success('The bank note is Authentic.')
    else:
        st.error('The bank note is Not Authentic.')