import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Load and preprocess dataset
# ----------------------------
@st.cache_data
def load_data():
    iris_df = pd.read_csv("data/IRIS.csv")
    iris_df.drop_duplicates(inplace=True)
    iris_df["sepal_area"] = iris_df["sepal_length"] * iris_df["sepal_width"]
    iris_df["petal_area"] = iris_df["petal_length"] * iris_df["petal_width"]

    le = LabelEncoder()
    iris_df["species"] = le.fit_transform(iris_df["species"])
    return iris_df, le

iris_df, le = load_data()

# Split dataset
X = iris_df.drop("species", axis=1)
y = iris_df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model performance
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌸 Iris Flower Classification App")
st.write("Predict Iris flower species using sepal & petal dimensions.")

# Sidebar inputs
st.sidebar.header("Input Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Derived features
sepal_area = sepal_length * sepal_width
petal_area = petal_length * petal_width

# Prepare input DataFrame
input_data = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width],
    "sepal_area": [sepal_area],
    "petal_area": [petal_area]
})

# Predict button
if st.button("🔍 Predict Species"):
    prediction = model.predict(input_data)
    pred_species = le.inverse_transform(prediction)[0]

    st.success(f"🌼 Predicted Species: **{pred_species}**")
    st.info(f"✅ Model Accuracy: {acc:.2f}")

    # Show classification report (optional)
    st.write("### Classification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# Optional: show dataset preview
if st.checkbox("Show Dataset Sample"):
    st.dataframe(iris_df.head())
