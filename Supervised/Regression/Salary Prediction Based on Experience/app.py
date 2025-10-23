import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# 1️⃣ Load trained model
# ---------------------------
model = joblib.load('salary_prediction_model.joblib')

# ---------------------------
# 2️⃣ Safe exponential function
# ---------------------------
def safe_expm1(x):
 x = np.nan_to_num(x, nan=0.0, posinf=20, neginf=-5)
 x = np.clip(x, -5, 20)
 return np.expm1(x)

# ---------------------------
# 3️⃣ Streamlit UI
# ---------------------------
st.title("💰 Salary Prediction App")

st.sidebar.header("Enter Employee Details")

# Example mapping used during training
education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3, "Other": 4}
job_map = {
 "Software Engineer": 0,
 "Data Scientist": 1,
 "Manager": 2,
 "Analyst": 3,
 "Other": 4
}

education = st.sidebar.selectbox("Education Level", list(education_map.keys()))
job_title = st.sidebar.selectbox("Job Title", list(job_map.keys()))
years_exp = st.sidebar.number_input("Years of Experience", min_value=0, max_value=50, value=5)

# ---------------------------
# 4️⃣ Encode inputs (convert to numeric)
# ---------------------------
education_encoded = education_map[education]
job_encoded = job_map[job_title]
years_exp_log = np.log1p(years_exp)

input_df = pd.DataFrame({
 "Education Level": [education_encoded],
 "Job Title": [job_encoded],
 "Years of Experience": [years_exp_log]
})

# ---------------------------
# 5️⃣ Predict Salary
# ---------------------------
if st.sidebar.button("Predict Salary"):
 pred_log = model.predict(input_df)
 salary_pred = safe_expm1(pred_log)[0]

 # Confidence range ±10%
 lower = salary_pred * 0.9
 upper = salary_pred * 1.1

 st.success(f"💵 Predicted Salary: ₹{salary_pred:,.2f}")
 st.info(f"💡 Estimated Range: ₹{lower:,.2f} - ₹{upper:,.2f}")

 # Bar chart visualization
 st.bar_chart(pd.DataFrame({
 "Predicted Salary": [salary_pred],
 "Lower Bound": [lower],
 "Upper Bound": [upper]
 }))