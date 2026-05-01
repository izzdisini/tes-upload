import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(page_title="Deteksi Paru-Paru", layout="centered")
st.title("Deteksi Penyakit Paru-Paru (X-ray)")
st.write("Upload gambar X-ray untuk mendapatkan prediksi.")

# =========================
# LOAD MODEL (LEBIH AMAN DEPLOY)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_parurasio801010.keras")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# =========================
# LABEL KELAS (WAJIB SESUAI TRAINING)
# =========================
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]

# =========================
# PREPROCESS FUNCTION (LEBIH AMAN)
# =========================
IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    # jaga-jaga kalau channel aneh
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # =========================
    # PREPROCESSING
    # =========================
    img_array = preprocess_image(image)

    # =========================
    # PREDIKSI
    # =========================
    with st.spinner("Menganalisis gambar..."):
        prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # =========================
    # OUTPUT
    # =========================
    st.success(f"Hasil Prediksi: **{class_names[predicted_class]}**")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # =========================
    # BAR CHART PROBABILITAS
    # =========================
    st.subheader("Probabilitas Tiap Kelas")
    prob_df = pd.DataFrame({
        "Probabilitas": prediction[0]
    }, index=class_names)
    st.bar_chart(prob_df)

# =========================
# DEBUG INFO (OPSIONAL)
# =========================
with st.expander("Debug Info"):
    st.write("Model path:", MODEL_PATH)
    st.write("Model loaded:", model is not None)
