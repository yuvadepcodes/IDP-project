import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("crop_disease_model.keras")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = (224, 224)

def preprocess(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

st.title("🌿 Crop Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    processed = preprocess(img)
    preds = model.predict(processed)

    idx = np.argmax(preds)
    confidence = np.max(preds)

    label = class_names[idx]

    if confidence < 0.6:
        st.warning(f"⚠️ Uncertain prediction: {label} ({confidence:.2f})")
    else:
        st.success(f"✅ Prediction: {label} ({confidence:.2f})")