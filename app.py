import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Rice Grain Classifier",
    page_icon="🌾",
    layout="centered"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
        .main {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        h1, h3, p {
            color: #ffffff;
        }
        .title {
            font-size: 2.5rem;
            color: #00c26e;
            text-align: center;
        }
        .description {
            font-size: 1.1rem;
            text-align: center;
            color: #ccc;
            margin-bottom: 20px;
        }
        .footer {
            font-size: 0.9rem;
            text-align: center;
            color: #888;
            padding: 20px 0;
        }
        .stFileUploader label {
            font-weight: bold;
            font-size: 1.1rem;
            color: #ffffff;
        }
        .uploaded-img {
            border-radius: 10px;
            margin-top: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
    </style>
""", unsafe_allow_html=True)

# Load and show logo
current_dir = os.path.dirname(__file__)
# Load the logo image
logo = Image.open(os.path.join(current_dir, "Untitled.png"))
buffered = BytesIO()
logo.save(buffered, format="PNG")
logo_base64 = base64.b64encode(buffered.getvalue()).decode()
# Center logo using the same style as title

st.markdown(f"""
    <div style='text-align: center; margin-bottom: 20px;'>
        <img src='data:image/png;base64,{logo_base64}' width='350'/>
    </div>
""", unsafe_allow_html=True)
# Title and subtitle
st.markdown('<h1 class="title">Rice Grain Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Upload an image of a rice grain and we will classify it into one of five types.</p>', unsafe_allow_html=True)

# Load the model
model = tf.keras.models.load_model('models/rice_classifier.h5')
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Upload file section
uploaded_file = st.file_uploader("Upload a rice grain image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="JPEG", clamp=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    # Show result
    st.markdown(f"""
        <div style="background-color:#0d6efd; padding: 1rem; border-radius: 8px; margin-top: 20px;">
            <h3 style="color:white; text-align:center;">Predicted Rice Type: <b>{class_names[class_index]}</b></h3>
        </div>
    """, unsafe_allow_html=True)
