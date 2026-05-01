import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Deteksi Paru-Paru", layout="centered")
st.title("Deteksi Penyakit Paru-Paru (X-ray)")
st.write("Upload gambar X-ray untuk mendapatkan prediksi.")

# =========================
# LOAD MODEL (DARI REPO)
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_parurasio801010.keras", compile=False)
    return model

model = load_model()

# =========================
# LABEL KELAS (WAJIB SESUAI TRAINING)
# =========================
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # FIX 2: Image.open yang benar (bukan hyperlink)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # =========================
    # PREPROCESSING
    # =========================
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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
    # FIX 3: st.info yang benar (bukan hyperlink)
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # FIX 4: Bar chart dengan label kelas yang benar
    st.subheader("Probabilitas Tiap Kelas")
    prob_df = pd.DataFrame({
        "Probabilitas": prediction[0]
    }, index=class_names)
    st.bar_chart(prob_df)
