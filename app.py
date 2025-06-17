import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost

# Load model dan komponen terkait
with open('model_xgboost1.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tinggi_mean.pkl', 'rb') as file:
    tinggi_mean = pickle.load(file)

with open('fitur_model.pkl', 'rb') as file:
    fitur_model = pickle.load(file)

# Judul aplikasi
st.title("Prediksi Stunting pada Balita")
st.markdown("Masukkan data berikut untuk mengetahui prediksi status gizi:")

# Form input data pengguna
umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
berat_badan = st.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0, step=0.1, value=10.0)
tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, step=0.1, value=80.0)

# Proses input
if st.button("Prediksi"):
    # Ubah jenis kelamin ke numerik
    jk_numeric = 1 if jenis_kelamin == 'Laki-laki' else 0

    # Buat fitur 'tinggi di atas rata-rata'
    tinggi_di_atas_rata = 1 if tinggi_badan > tinggi_mean else 0

    # Buat DataFrame input
    data_input = pd.DataFrame([{
        'Umur (bulan)': umur,
        'Jenis Kelamin': jk_numeric,
        'Berat Badan (kg)': berat_badan,
        'Tinggi Badan (cm)': tinggi_badan,
        'Tinggi di atas rata-rata': tinggi_di_atas_rata
    }])

    # Sesuaikan urutan fitur agar sama dengan saat training
    data_input = data_input[fitur_model]

    # Prediksi
    prediksi = model.predict(data_input)[0]
    probabilitas = model.predict_proba(data_input)[0]

    # Tampilkan hasil
    st.success(f"Prediksi Status Gizi: **{prediksi}**")
    st.write("Probabilitas:")
    st.write({f"{model.classes_[i]}": f"{round(prob * 100, 2)}%" for i, prob in enumerate(probabilitas)})
