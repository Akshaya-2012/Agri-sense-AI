
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="AgriSense AI",
    page_icon="🌱",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_model.keras",
        compile=False
    )

st.title("🌱 AgriSense AI")
st.subheader("Smart Farming. Healthier Crops.")
st.write(
    "AI-powered tomato leaf condition detection."
)

model_path = Path("best_model.keras")

if not model_path.exists():
    st.error("Model file not found. Please upload best_model.keras to the repository.")
    st.stop()

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Tomato Leaf", use_container_width=True)

    if st.button("🔍 Analyse Leaf"):
        resized_image = image.resize((224, 224))
        image_array = np.array(resized_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)[0]

        class_names = [
            "Healthy",
            "Early Blight",
            "Late Blight"
        ]

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[predicted_index]) * 100

        st.success("Analysis completed!")
        st.subheader("Prediction: " + predicted_class)
        st.metric("Model Confidence", f"{confidence:.2f}%")

        st.info(
            "This is an AI-generated prediction, not a confirmed diagnosis. "
            "Please consult an agricultural expert when needed."
        )

st.divider()
st.caption("AgriSense AI | An AI-powered agricultural project")
