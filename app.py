import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN

# SETUP PATH & LOAD MODEL
# pakai path lokal
MODEL_PATH = 'xception_model_latest.h5'
LOG_PATH = 'xception_training_log_latest.csv'
IMG_SIZE = 128

@st.cache_resource
def load_deeplearning_model():
    """Fungsi untuk load model sekali saja (cached)"""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        return None

@st.cache_resource
def load_face_detector():
    return MTCNN()

def process_and_crop_face(image, detector):
    img_array = np.array(image)
    
    # konversi rgba ke rgb
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    results = detector.detect_faces(img_array)
    
    if not results:
        return None # kalau ga ada wajah yang terdeteksi

    largest_face =  max(results, key=lambda face: face['box'][2] * face['box'][3])

    # ambil wajah dengan confidence tertinggi (biasanya di index 0)
    x, y, w, h = largest_face['box']
    
    # add margin 20% agar rahang dan leher ikut terpotong
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_array.shape[1], x + w + margin_x)
    y2 = min(img_array.shape[0], y + h + margin_y)
    
    # cropping
    face_crop = img_array[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return None
        
    # resize ke ukuran input xception (128x128)
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    return face_resized


#  STREAMLIT INTERFACE
st.set_page_config(
    page_title="Deepfake Detection Dashboard",
    layout="wide"
)

st.title(" Deepfake Face Detection")
st.markdown("Aplikasi ini menggunakan model Deep Learning (**Xception**) untuk menganalisis apakah sebuah foto wajah adalah asli (Real) atau hasil manipulasi (Deepfake).")

# load model utama
model = load_deeplearning_model()
detector = load_face_detector()

if model is None:
    st.error(f"File model tidak ditemukan! pastikan file `{MODEL_PATH}` ada di folder yang sama dengan `app.py`.")
    st.stop()

# SIDEBAR: PENGATURAN TAMPILAN
st.sidebar.header("Pengaturan")
show_metrics = st.sidebar.checkbox("Tampilkan Kurva Training (Loss & Acc)")

# HALAMAN UTAMA: PREDIKSI GAMBAR
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Screenshot / Foto")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # convert ke rgb
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar Original (Full Frame)", use_container_width=True)

with col2:
    st.subheader("Hasil Analisis")
    if uploaded_file is not None:
        with st.spinner("Mendeteksi wajah dan menganalisis..."):
            
            # jalankan pipeline MTCNN
            cropped_face = process_and_crop_face(image, detector)
            
            if cropped_face is None:
                st.error("Wajah tidak terdeteksi! pastikan gambar menampilkan wajah yang cukup jelas tanpa terhalang objek terlalu banyak.")
            else:
                # tampilkan area wajah yang berhasil di-crop dan yang diproses
                st.image(cropped_face, caption="Area Wajah yang Dianalisis", width=150)
                
                # preprocessing untuk model Xception (Normalisasi & Tambah Dimensi Batch)
                img_array = cropped_face / 255.0 
                img_tensor = np.expand_dims(img_array, axis=0) 
                
                # prediksi
                prediction_prob = model.predict(img_tensor).flatten()[0]
                
                # tentukan hasil kelas (1 = Fake, 0 = Real)
                threshold = 0.2 # bisa dinaikkan/diturunkan tergantung kepekaan model nanti
                
                if prediction_prob > threshold:
                    result_label = "FAKE (Deepfake / Manipulasi)"
                    confidence = prediction_prob * 100
                    st.error(f"### Hasil: **{result_label}**")
                else:
                    result_label = "REAL (Manusia Asli)"
                    confidence = (1 - prediction_prob) * 100
                    st.success(f"### Hasil: **{result_label}**")
                    
                # tampilkan progress bar confidence score
                st.write("**Confidence Score:**")
                st.progress(int(confidence))
                st.write(f"Tingkat Keyakinan Model: **{confidence:.2f}%**")
    else:
        st.info("Silakan upload foto di panel sebelah kiri untuk memulai analisis otomatis.")

# VISUALISASI
if show_metrics:
    st.markdown("---")
    st.subheader("Grafik Training Model")
    
    if os.path.exists(LOG_PATH):
        df_log = pd.read_csv(LOG_PATH)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # plot Accuracy
        ax1.plot(df_log['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
        if 'val_accuracy' in df_log.columns:
            ax1.plot(df_log['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        ax1.set_title('Kurva Akurasi')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, linestyle='--')
        
        # plot Loss
        ax2.plot(df_log['loss'], label='Train Loss', color='red', linewidth=2)
        if 'val_loss' in df_log.columns:
            ax2.plot(df_log['val_loss'], label='Validation Loss', color='green', linewidth=2)
        ax2.set_title('Kurva Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, linestyle='--')
        
        st.pyplot(fig)
    else:
        st.warning(f"File log CSV `{LOG_PATH}` tidak ditemukan di folder ini.")
