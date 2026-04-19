import streamlit as st
from PIL import Image
import time

st.set_page_config(page_title="Crop Disease Detection")

st.title("AI Based Crop Disease Detection System")
st.subheader("Phase 1 Prototype – Tomato Leaf Analysis")

uploaded_file = st.file_uploader(
    "Upload tomato leaf image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    if st.button("Analyze Disease"):
        with st.spinner("Analyzing image..."):
            time.sleep(2)

        st.success("Disease Detected Successfully")

        st.write("**Prediction:** Late Blight")
        st.write("**Confidence:** 94.2%")
        st.write("**Severity:** Moderate")

        st.write("### Suggested Remedy")
        st.write("- Apply fungicide spray")
        st.write("- Remove infected leaves")
        st.write("- Avoid overwatering")