import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("digit_recognizer_model.h5")

st.title("Digit Recognizer")
st.write("Upload a 28x28 grayscale image of a digit (0-9) to predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors if needed
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model

    st.image(image, caption='Uploaded Image', width=150)
    st.write("Predicting...")

    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    st.write(f"Predicted Digit: {predicted_digit}")
