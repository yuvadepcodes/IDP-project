# CropSense AI - Crop Disease Detection Website

A beautiful two-page Streamlit website for detecting crop diseases using AI.

## Features

- **Home Page**: Attractive landing page with information about the service
- **Scan Page**: Upload crop images for instant disease analysis
- **AI Analysis**: Uses TensorFlow/Keras model to identify diseases
- **Plant Detection**: Automatically rejects non-plant images (checks for green content)
- **Detailed Results**: Shows crop type, disease, confidence, severity, and treatment recommendations

## Setup

1. Ensure you have Python 3.7+ installed
2. Install dependencies:
   ```bash
   pip install streamlit tensorflow pillow numpy
   ```

3. Make sure you have the model file `crop_disease_model.keras` and `class_names.txt` in the project root

## Running the Application

```bash
streamlit run app2.py
```

The website will be available at `http://localhost:8501`

## Files Structure

- `app2.py`: Main Streamlit application with two pages
- `crop_disease_model.keras`: Trained Keras model
- `class_names.txt`: List of class names

## Model Information

The model supports detection of various crop diseases including:
- Tomato diseases (Late Blight, Early Blight, Leaf Mold, etc.)
- Potato diseases (Early Blight, Late Blight)
- Pepper diseases (Bacterial Spot)
- And more...

## Troubleshooting

**Non-plant images being accepted:**
- The app now checks for green content in images
- If you see "This doesn't appear to be a plant image" error, upload a clearer plant photo
- The system requires at least 5% of pixels to be green-dominant

**Model always predicts the same disease:**
- This may indicate imbalanced training data
- Model overfitting to certain classes
- Consider retraining with more diverse data

**Low confidence predictions:**
- Model thresholds are set to accept predictions with ≥1% confidence
- Very low confidence may indicate the image doesn't match training data well

The app includes rejection logic - if confidence is low or the prediction is uncertain, it will tell you it's not a valid plant image.

Consider retraining the model with a balanced dataset for better accuracy. 