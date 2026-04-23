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
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (224, 224)
THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.20

# -----------------------------
# PREPROCESS IMAGE
# -----------------------------
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

    preds = model.predict(img, verbose=0)[0]

    top_index = np.argmax(preds)
    confidence = float(preds[top_index])
    label = class_names[top_index]

    # second best score for uncertainty check
    sorted_preds = np.sort(preds)[::-1]
    margin = sorted_preds[0] - sorted_preds[1]

    # rejection logic (IMPORTANT)
    if confidence < THRESHOLD or margin < MARGIN_THRESHOLD:
        return "❌ Not a valid plant image / Uncertain", confidence

    return label, confidence

# -----------------------------
# SINGLE IMAGE TEST
# -----------------------------
img_path = "images.jpeg"  # change this

if os.path.exists(img_path):
    label, confidence = predict_image(img_path)
    print(f"{label} ({confidence:.2f})")
else:
    print("Image not found:", img_path)

# -----------------------------
# BATCH TEST (OPTIONAL)
# -----------------------------
def predict_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, file)

            label, confidence = predict_image(path)

            status = "⚠️" if "Uncertain" in label else "✅"
            print(f"{file} → {status} {label} ({confidence:.2f})")

# Example:
# predict_folder("test_images")