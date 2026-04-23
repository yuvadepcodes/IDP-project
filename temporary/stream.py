import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("crop_disease_model.keras")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = (224, 224)

# -----------------------------
# UI HEADER
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>🌿 Crop Disease Detection System</h1>
    <p style='text-align: center; color: gray;'>
    Upload a leaf image and get AI-powered disease prediction
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# UPLOAD SECTION
# -----------------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

def preprocess(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed = preprocess(image)
    preds = model.predict(processed, verbose=0)[0]

    # -----------------------------
    # TOP PREDICTIONS
    # -----------------------------
    top_indices = preds.argsort()[-3:][::-1]

    st.subheader("🔍 Prediction Results")

    for i, idx in enumerate(top_indices):
        label = class_names[idx]
        confidence = float(preds[idx])

        st.write(f"**{i+1}. {label}**")
        st.progress(confidence)
        st.write(f"{confidence*100:.2f}% confidence")
        st.write("---")

    # -----------------------------
    # FINAL RESULT
    # -----------------------------
    best_idx = top_indices[0]
    best_label = class_names[best_idx]
    best_conf = float(preds[best_idx])

    if best_conf < 0.6:
        st.warning(f"⚠️ Low confidence prediction: {best_label}")
    else:
        st.success(f"✅ Detected Disease: {best_label}")

else:
    st.info("Upload an image to start prediction")