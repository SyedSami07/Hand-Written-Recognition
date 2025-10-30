import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

st.title("✍️ Handwritten Digit Recognition")

model = tf.keras.models.load_model('hand_written_digit_recognition.keras')

uploaded_file = st.file_uploader("Upload an image of a digit (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
  
    image = Image.open(uploaded_file).convert('L')
 
    image = image.resize((28, 28))


    image_array = np.array(image)
  
    image_array = 255 - image_array
 
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)


    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    st.image(image, caption='Uploaded Image', width=150)
    st.success(f"Predicted Digit: **{predicted_digit}**")
