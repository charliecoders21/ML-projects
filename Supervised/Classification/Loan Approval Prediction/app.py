import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load model and metadata
# ---------------------------
model = joblib.load("models/loan_approval_predict.joblib")
feature_names = joblib.load("models/model_features.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval status.")

# Input fields
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", label_encoders['education'].classes_)
self_employed = st.selectbox("Self Employed", label_encoders['self_employed'].classes_)
income_annum = st.number_input("Annual Income (₹)", min_value=0.0, step=1000.0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0.0, step=1000.0)
loan_term = st.number_input("Loan Term (months)", min_value=1, step=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)

# Asset values
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0.0, step=1000.0)
commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0.0, step=1000.0)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0.0, step=1000.0)
bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0.0, step=1000.0)

# Calculate total asset value
total_asset_value = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Loan Approval"):
    # Encode categorical values
    education_encoded = label_encoders['education'].transform([education])[0]
    self_employed_encoded = label_encoders['self_employed'].transform([self_employed])[0]

    # Create input DataFrame
    input_data = pd.DataFrame([[
        no_of_dependents,
        education_encoded,
        self_employed_encoded,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        total_asset_value
    ]], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_label = label_encoders['loan_status'].inverse_transform([prediction])[0]
    # Normalize the prediction label
    normalized_label = prediction_label.strip().lower()

    if normalized_label == "rejected":
        st.error(f"❌ Loan Status Prediction: **{prediction_label}**")
    else:
        st.success(f"✅ Loan Status Prediction: **{prediction_label}**")

    