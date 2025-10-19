import streamlit as st
import pandas as pd
import joblib

# Title
st.title("Titanic Survival Prediction")

# Load pre-trained model
model_path = "models/titanic_survival_model.joblib"
model = joblib.load(model_path)

st.sidebar.header("Passenger Details")
# Input fields for prediction
pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.slider("Number of Siblings/Spouses (SibSp)", 0, 8, 0)
parch = st.sidebar.slider("Number of Parents/Children (Parch)", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 512.0, 32.0)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

# Encode categorical inputs
sex_encoded = 0 if sex == "male" else 1
embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]

# Prepare input DataFrame
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_encoded],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_encoded]
})

# Prediction button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.write("✅ Survived" if prediction == 1 else "❌ Not Survived")
    st.write(f"Survival Probability: {probability:.2f}")

    # Display probability gauge
    st.progress(int(probability * 100))

# Footer
st.markdown("---")
st.markdown("Model loaded from **models/titanic_survival_model.joblib**")