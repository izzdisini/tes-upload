import os
# WAJIB: Letakkan ini di baris paling atas sebelum import tensorflow

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(page_title="Deteksi Paru-Paru", layout="centered")
st.title("Deteksi Penyakit Paru-Paru (X-ray)")
st.write("Upload gambar X-ray untuk mendapatkan prediksi.")

# =========================
# LOAD MODEL (PASTIKAN NAMA FILE SESUAI DI GITHUB)
# =========================
# Gunakan relative path sederhana agar tidak bingung dengan direktori /mount/src/ di cloud
MODEL_PATH = "model_parurasio801010.h5" 

@st.cache_resource
def load_model():
    # Tambahkan safe_mode=False untuk menangani isu deserialisasi Keras 3
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    return model

# Coba muat model dengan penanganan error
try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

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
