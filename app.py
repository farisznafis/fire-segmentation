import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
try:
    model = load_model('models/segmentation_model.hdf5')
except Exception as e:
    st.error(f"Error loading model: {e}")

def perform_segmentation(image, model, threshold=0.5):
    image_resized = cv2.resize(image, (256, 256))
    image_input = img_to_array(image_resized)
    image_input = np.expand_dims(image_input, axis=0)
    image_input = np.expand_dims(image_input, axis=3)
    image = image_input / 255.0
    prediction = (model.predict(image)[0, :, :, 0] > threshold).astype(np.uint8)
    return prediction

# Streamlit app
st.title("Fire Segmentation App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Perform segmentation
        prediction = perform_segmentation(image, model)

        # Display the original image and the segmentation result
        st.subheader("Original Image")
        st.image(image, channels="GRAY", use_container_width=True)
        
        st.subheader("Segmentation Result")
        st.image(prediction * 255, channels="GRAY", use_container_width=True)
    else:
        st.error("Error reading the uploaded image.")