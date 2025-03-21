import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Flower Classification", page_icon="🌸", layout="centered")
st.title("🌼 Flower Classification App 🌻")
st.markdown("""
Upload an image of a flower, and the AI model will classify it into one of the following categories:
- **Daisy**
- **Dandelion**
- **Rose**
- **Sunflower**
- **Tulip**
""")

flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
model = load_model('Flower_Recog.h5')

def classify_images(images_path):
    input_image = tf.keras.utils.load_img(images_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = {
        "label": flower_names[np.argmax(result)],
        "confidence": np.max(result) * 100
    }
    return outcome

uploaded_file = st.file_uploader("📤 Upload a Flower Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

    image_path = os.path.join('upload', uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    
    st.image(image_path, width=300, caption="Uploaded Image")

    with st.spinner("Analyzing the image..."):
        prediction = classify_images(image_path)
        st.success(f"**🌸 Prediction:** {prediction['label']}")
        st.info(f"**Confidence Score:** {prediction['confidence']:.2f}%")
        st.caption(f"**The image is classified as a {prediction['label']} with a confidence score of {prediction['confidence']:.2f}%.**")

    st.markdown("""
    ---
    🌼 **Tips**: Try uploading different flowers to test the model's capabilities!
    🌻 **Note**: Ensure the image is clear for better accuracy.
    """)
