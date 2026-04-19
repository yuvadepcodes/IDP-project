import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("crop_disease_model.keras")

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------
# Make sure you saved this during training
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
IMG_SIZE = (224, 224)

def preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(img_path):
    img = preprocess(img_path)

    preds = model.predict(img, verbose=0)
    predicted_index = np.argmax(preds)

    confidence = float(np.max(preds))
    label = class_names[predicted_index]

    return label, confidence

# -----------------------------
# SINGLE IMAGE TEST
# -----------------------------
img_path = "images.jpeg"   # change this

if os.path.exists(img_path):
    label, confidence = predict_image(img_path)

    if confidence < 0.60:
        print(f"⚠️ Uncertain prediction: {label} ({confidence:.2f})")
    else:
        print(f"✅ Prediction: {label} ({confidence:.2f})")
else:
    print("Image not found:", img_path)


# -----------------------------
# BATCH TEST (MULTIPLE IMAGES)
# -----------------------------
def predict_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, file)

            label, confidence = predict_image(path)

            status = "⚠️ Uncertain" if confidence < 0.60 else "✅"
            print(f"{file} → {status} {label} ({confidence:.2f})")


# Example usage:
# predict_folder("test_images")