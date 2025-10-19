import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/student_score_predict_model.joblib")

# App title
st.title("📚 Student Exam Score Predictor")

# Sidebar inputs
st.sidebar.header("Enter Student Details")

hours_studied = st.sidebar.number_input("Hours Studied", min_value=0.0, max_value=20.0, value=5.0)
sleep_hours = st.sidebar.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=7.0)
attendance_percent = st.sidebar.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
previous_scores = st.sidebar.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=65.0)

# Feature engineering
study_attendance = hours_studied * attendance_percent
hours_studied_squared = hours_studied ** 2
sleep_efficiency = sleep_hours / hours_studied if hours_studied != 0 else 0

# Prediction
if st.sidebar.button("Predict Exam Score"):
    input_df = pd.DataFrame([{
        "hours_studied": hours_studied,
        "sleep_hours": sleep_hours,
        "attendance_percent": attendance_percent,
        "previous_scores": previous_scores,
        "study_attendance": study_attendance,
        "hours_studied_squared": hours_studied_squared,
        "sleep_efficiency": sleep_efficiency
    }])

    predicted_score = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Exam Score: **{predicted_score:.2f}**")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Scikit-learn")
