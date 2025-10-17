import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1️⃣ Page Setup
# -----------------------------
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Enter property details below to estimate its market price using a trained LightGBM model.")

# -----------------------------
# 2️⃣ Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/house_price_pipeline.pkl")

pipeline = load_model()

# -----------------------------
# 3️⃣ Input Form
# -----------------------------
with st.form("prediction_form"):
    area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2500)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 6, 2)
    stories = st.slider("Stories", 1, 4, 2)
    parking = st.slider("Parking Spaces", 0, 4, 1)

    mainroad = st.selectbox("Main Road Access", ["yes", "no"])
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"])

    submitted = st.form_submit_button("Predict Price")

# -----------------------------
# 4️⃣ Feature Engineering
# -----------------------------
if submitted:
    input_df = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus,
        'area_per_bedroom': area / (bedrooms + 1),
        'area_stories': area * stories,
        'bed_bath': bedrooms * bathrooms,
        'amenities_score': sum([mainroad=="yes", guestroom=="yes", basement=="yes",
                                hotwaterheating=="yes", airconditioning=="yes", prefarea=="yes"]),
        'parking_pref': parking * (prefarea=="yes"),
        'lux_score': int(airconditioning=="yes") + int(hotwaterheating=="yes")
    }])

    # -----------------------------
    # 5️⃣ Predict & Display
    # -----------------------------
    log_pred = pipeline.predict(input_df)[0]
    price_pred = np.expm1(log_pred)

    st.success(f"💰 Estimated House Price: ₹{price_pred:,.2f}")