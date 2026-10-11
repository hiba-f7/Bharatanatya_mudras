import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Page Configuration
st.set_page_config(page_title="Mudra Recognition", page_icon="🪷")


CLASS_NAMES = ["Musti", "Pataka", "Sikhara", "Simhamukha", "Trisula"]

#Trained Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mudra_model.keras")

model = load_model()

#Navigation
page = st.sidebar.radio("Navigation", ["Home", "Predict", "About"])

#HOME
if page == "Home":
    st.title("🪷 Bharatanatyam Mudra Recognition")
    st.write(
        "Welcome! This AI app identifies Bharatanatyam hand mudras and "
        "gives instant feedback during dance practice."
    )
    st.write("### Recognized Mudras:")
    for name in CLASS_NAMES:
        st.write(f"- **{name}**")

#PREDICT
elif page == "Predict":
    st.title("Predict Hand Mudra")
    uploaded_file = st.file_uploader("Upload an image of a mudra", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=260)

        # Preprocess to match 128x128 input
        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized)
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_batch, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        conf = float(preds[top_idx]) * 100

        # Show Results
        st.success(f"**Predicted Mudra:** {CLASS_NAMES[top_idx]}")
        st.info(f"**Confidence Score:** {conf:.2f}%")

        st.write("---")
        st.write("#### Class Probabilities:")
        for name, prob in zip(CLASS_NAMES, preds):
            st.write(f"{name}: {prob * 100:.2f}%")
            st.progress(float(prob))

#ABOUT
elif page == "About":
    st.title("About Dataset & Model")
    st.write("- **Domain:** Cultural AI — Classical Indian Dance Gesture Recognition")
    st.write("- **Classes:** Musti, Pataka, Sikhara, Simhamukha, Trisula")
    st.write("- **Architecture:** Custom Convolutional Neural Network (CNN)")
    st.write("- **Input Resolution:** 128 x 128 RGB")
