# NOTE: This script is intended to run in a Streamlit-compatible Python environment.
# If you encounter 'ModuleNotFoundError: No module named ...', install dependencies with:
# pip install streamlit pandas xgboost matplotlib seaborn

import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


# ==== Muat model terlatih ====
try:
    with open('model_xgboost1.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("File model_xgboost1.pkl tidak ditemukan. Pastikan file sudah tersedia di direktori.")
    st.stop()

# === Konstanta preprocessing ===
TINGGI_MEAN = 86.04  # cm – hasil rata‑rata setelah hapus duplikat

# === Urutan fitur yang dibutuhkan model ===
Fitur_Model = [
    'Umur (bulan)',
    'Jenis Kelamin',
    'Berat Badan (kg)',
    'Tinggi Badan (cm)',
    'Tinggi di atas rata-rata'
]

# === Judul aplikasi ===
st.title("Prediksi Stunting pada Balita")
st.markdown("Masukkan data berikut untuk mengetahui prediksi status gizi:")

# === Input Form ===
umur = st.number_input("Umur (bulan)", 0, 60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
berat_badan = st.number_input("Berat Badan (kg)", 2.0, 30.0, step=0.1, value=10.0)
tinggi_badan = st.number_input("Tinggi Badan (cm)", 30.0, 120.0, step=0.1, value=80.0)

# === Prediksi ===
if st.button("Prediksi"):
    jk_numeric = 1 if jenis_kelamin == 'Laki-laki' else 0
    tinggi_di_atas_rata = 1 if tinggi_badan > TINGGI_MEAN else 0

    data_input = pd.DataFrame([{
        'Umur (bulan)': umur,
        'Jenis Kelamin': jk_numeric,
        'Berat Badan (kg)': berat_badan,
        'Tinggi Badan (cm)': tinggi_badan,
        'Tinggi di atas rata-rata': tinggi_di_atas_rata
    }])

    try:
        data_input = data_input[Fitur_Model]
        prediksi = model.predict(data_input)[0]
        probabilitas = model.predict_proba(data_input)[0]

        st.success(f"Prediksi Status Gizi: **{prediksi}**")

        # === Visualisasi Probabilitas ===
        fig, ax = plt.subplots()
        sns.barplot(x=model.classes_, y=probabilitas, ax=ax, palette="Set2")
        ax.set_ylabel("Probabilitas")
        ax.set_ylim(0, 1)
        ax.set_title("Probabilitas Tiap Kelas")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
