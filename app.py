import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown

# Set page title and icon
st.set_page_config(page_title="BioBloom Leaf Diseases Detection", page_icon="🌿")

# Custom background color using CSS
st.markdown(
    """
    <style>
    body {
        background-color: #dce8d1;
    }
    .stApp {
        background-color: #dce8d1;
    }
    h1 {
        color: #2d4739;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1>BioBloom Leaf Diseases Detection</h1>", unsafe_allow_html=True)

# Download model from Google Drive
model_path = "tomato_leaf_model_checkpoint.h5"
file_id = "1cbhwi39QC3S9Nbb119kNGZp23xh7ODyM"
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)
st.success("Model loaded successfully and ready to predict.")
st.write("Model output shape:", model.output_shape)

# Friendly class labels
class_labels = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy"
]

# Upload and predict
st.write("Upload a tomato leaf image to identify any disease.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}% confident)")
